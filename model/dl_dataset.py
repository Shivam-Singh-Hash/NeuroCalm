import os
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
import joblib

# WESAD Frequencies
FS_EDA = 4
FS_TEMP = 4
FS_BVP = 64
FS_ACC = 32
FS_LABEL = 700
# Target Deep Learning Unified Frequency
TARGET_FREQ = 64

def load_wesad_subject(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    return data

def resample_signal(signal, original_fs, target_fs, total_seconds):
    """Upsample or downsample signals cleanly to unify temporal sequences."""
    num_samples_target = int(total_seconds * target_fs)
    x_old = np.linspace(0, total_seconds, len(signal))
    x_new = np.linspace(0, total_seconds, num_samples_target)
    
    # Linear interpolation
    f = interp1d(x_old, signal, axis=0, bounds_error=False, fill_value="extrapolate")
    return f(x_new)

def extract_subject_windows(data, window_size_sec=10, step_size_sec=5):
    """
    Extracts strictly aligned 10-second multi-channel matrices.
    Channels (6): BVP, EDA, TEMP, ACC_X, ACC_Y, ACC_Z
    """
    signal = data['signal']['wrist']
    chest_signal = data['signal']['chest']
    labels = data['label']
    
    eda = signal['EDA'].flatten()
    temp = signal['TEMP'].flatten()
    bvp = signal['BVP'].flatten()
    acc = signal['ACC'] # shape (N, 3)
    resp = chest_signal['Resp'].flatten() # Respiration from chest at 700Hz

    # Base on the slowest time or label length
    total_seconds_label = len(labels) / FS_LABEL
    total_seconds_eda = len(eda) / FS_EDA
    total_seconds = min(total_seconds_label, total_seconds_eda)

    # Resample everything to 64Hz (TARGET_FREQ)
    eda_r = resample_signal(eda, FS_EDA, TARGET_FREQ, total_seconds)
    temp_r = resample_signal(temp, FS_TEMP, TARGET_FREQ, total_seconds)
    bvp_r = resample_signal(bvp, FS_BVP, TARGET_FREQ, total_seconds)
    acc_r = resample_signal(acc, FS_ACC, TARGET_FREQ, total_seconds)
    resp_r = resample_signal(resp, FS_LABEL, TARGET_FREQ, total_seconds)

    # Create master matrix: Shape (Total Samples, 7) now including Respiration
    master_matrix = np.column_stack([bvp_r, eda_r, temp_r, acc_r[:,0], acc_r[:,1], acc_r[:,2], resp_r])
    
    windows = []
    y_labels = []
    
    total_sec_int = int(total_seconds)
    for start_sec in range(0, total_sec_int - window_size_sec, step_size_sec):
        # Master Matrix indexing (at TARGET_FREQ)
        start_idx = start_sec * TARGET_FREQ
        end_idx = (start_sec + window_size_sec) * TARGET_FREQ
        
        # Label indexing (at FS_LABEL)
        label_start = start_sec * FS_LABEL
        label_end = (start_sec + window_size_sec) * FS_LABEL
        window_labels = labels[label_start:label_end]

        valid_mask = np.isin(window_labels, [1, 2, 3])
        if np.sum(valid_mask) < len(window_labels) * 0.5:
            continue
            
        counts = np.bincount(window_labels[valid_mask].astype(int))
        majority_label = np.argmax(counts)
        # Shift index down (1,2,3 -> 0,1,2 for PyTorch CrossEntropy)
        majority_label -= 1  
        
        x_window = master_matrix[start_idx:end_idx, :]
        if len(x_window) == TARGET_FREQ * window_size_sec:
            windows.append(x_window)
            y_labels.append(majority_label)
            
    return windows, y_labels

class WesadDataset(Dataset):
    def __init__(self, X, y):
        # Convert X to tensor of shape (N, Channels, TimeSteps) suitable for Conv1d/LSTM
        # Current X shape: (N, 640, 6) -> We need (N, 6, 640)
        X = np.array(X, dtype=np.float32)
        X = np.transpose(X, (0, 2, 1))
        
        self.X = torch.tensor(X)
        self.y = torch.tensor(y, dtype=torch.long)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def get_dataloaders(data_dir, batch_size=64):
    """Parses standard Train/Val/Test architecture and yields DataLoader wrappers."""
    subject_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    all_X = []
    all_y = []
    
    for subj in subject_dirs:
        pkl_path = os.path.join(data_dir, subj, f'{subj}.pkl')
        if os.path.exists(pkl_path):
            print(f"DL Processor: Parsing {subj}...")
            data = load_wesad_subject(pkl_path)
            windows, labels = extract_subject_windows(data)
            all_X.extend(windows)
            all_y.extend(labels)
            
    if not all_X:
        raise ValueError("No WESAD data found. Please run data generator.")
        
    # Standard 70/15/15 split simulation
    from sklearn.model_selection import train_test_split
    X_train, X_temp, y_train, y_temp = train_test_split(all_X, all_y, test_size=0.3, random_state=42, stratify=all_y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    # Normalization strictly bounded to Train data
    scaler = StandardScaler()
    
    # Flatten strictly for scaling feature columns independently
    # X_train shape: (N, 640, 6)
    N_tr, T, C = np.array(X_train).shape
    
    X_tr_flat = np.array(X_train).reshape(-1, C)
    scaler.fit(X_tr_flat)
    
    X_train = scaler.transform(X_tr_flat).reshape(N_tr, T, C)
    
    N_v = np.array(X_val).shape[0]
    X_val = scaler.transform(np.array(X_val).reshape(-1, C)).reshape(N_v, T, C)
    
    N_t = np.array(X_test).shape[0]
    X_test = scaler.transform(np.array(X_test).reshape(-1, C)).reshape(N_t, T, C)
    
    # Save scaler for inferences
    out_dir = os.path.join(os.path.dirname(__file__), 'saved_models')
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(out_dir, 'dl_scaler.pkl'))
    print("StandardScaler successfully mathematically optimized and saved!")

    train_loader = DataLoader(WesadDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(WesadDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(WesadDataset(X_test, y_test), batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, y_train
