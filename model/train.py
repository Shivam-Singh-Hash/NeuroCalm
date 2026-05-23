import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

from feature_extraction import process_window, get_feature_names

# Frequencies for wrist data in WESAD
FS_EDA = 4
FS_TEMP = 4
FS_BVP = 64
FS_LABEL = 700 

def load_wesad_subject(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    return data

def build_dataset_from_subject(data, window_size_sec=10, step_size_sec=5):
    """
    Slides a window across the data and extracts features for each valid window.
    """
    signal = data['signal']['wrist']
    labels = data['label']
    
    eda = signal['EDA'].flatten()
    temp = signal['TEMP'].flatten()
    bvp = signal['BVP'].flatten()
    
    features_list = []
    y_list = []
    
    # We iterate using the slowest signal frequency (label is 700hz, let's map time)
    total_seconds = len(eda) // FS_EDA
    
    for start_sec in range(0, total_seconds - window_size_sec, step_size_sec):
        # Indices
        eda_start, eda_end = start_sec * FS_EDA, (start_sec + window_size_sec) * FS_EDA
        temp_start, temp_end = start_sec * FS_TEMP, (start_sec + window_size_sec) * FS_TEMP
        bvp_start, bvp_end = start_sec * FS_BVP, (start_sec + window_size_sec) * FS_BVP
        
        # Determine strict label for the window (majority voting in that timeframe)
        # Label is at 700Hz
        label_start, label_end = start_sec * FS_LABEL, (start_sec + window_size_sec) * FS_LABEL
        window_labels = labels[label_start:label_end]
        
        # Valid labels: 1=Baseline, 2=Stress, 3=Amusement, 4=Meditation
        # Filter for 1, 2, 3
        valid_mask = np.isin(window_labels, [1, 2, 3])
        if np.sum(valid_mask) < len(window_labels) * 0.5:
            # If majority of window is unwated label (e.g. 0-transient), skip
            continue
            
        counts = np.bincount(window_labels[valid_mask].astype(int))
        majority_label = np.argmax(counts)
        
        # Extract features
        eda_w = eda[eda_start:eda_end]
        temp_w = temp[temp_start:temp_end]
        bvp_w = bvp[bvp_start:bvp_end]
        
        feats = process_window(eda_w, temp_w, bvp_w)
        features_list.append(feats)
        y_list.append(majority_label)
        
    return pd.DataFrame(features_list, columns=get_feature_names()), np.array(y_list)

def train_model():
    print("Loading datasets...")
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    wesad_dir = os.path.join(data_dir, 'WESAD')
    
    all_features = []
    all_labels = []
    
    # Process all available mock subjects (or real ones if they exist)
    subject_dirs = [d for d in os.listdir(wesad_dir) if os.path.isdir(os.path.join(wesad_dir, d))]
    for subj in subject_dirs:
        pkl_path = os.path.join(wesad_dir, subj, f'{subj}.pkl')
        if os.path.exists(pkl_path):
            print(f"Processing {subj}...")
            data = load_wesad_subject(pkl_path)
            X, y = build_dataset_from_subject(data)
            all_features.append(X)
            all_labels.append(y)
    
    if not all_features:
        print("No WESAD data found. Please run generate_data.py first!")
        return
        
    X_full = pd.concat(all_features, ignore_index=True)
    y_full = np.concatenate(all_labels)
    
    print(f"Total structured windows extracted: {len(X_full)}")
    
    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.2, random_state=42, stratify=y_full)
    
    # Scaling
    print("Scaling and standardizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Training
    print("Training Random Forest Classifier...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf.fit(X_train_scaled, y_train)
    
    y_pred = rf.predict(X_test_scaled)
    print("Evaluation Results:")
    print(classification_report(y_test, y_pred))
    
    # Saving models
    out_dir = os.path.join(os.path.dirname(__file__), 'saved_models')
    os.makedirs(out_dir, exist_ok=True)
    
    joblib.dump(scaler, os.path.join(out_dir, 'scaler.pkl'))
    joblib.dump(rf, os.path.join(out_dir, 'rf_model.pkl'))
    print("Models saved successfully in model/saved_models/")

if __name__ == '__main__':
    train_model()
