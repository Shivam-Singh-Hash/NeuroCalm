import os
import pickle
import numpy as np
import pandas as pd

def generate_mock_wesad_pkl(output_path, subject_id='S2'):
    """Generates a mock WESAD .pkl file for pipeline testing."""
    # WESAD .pkl structure:
    # {'subject': 'S2', 'signal': {'wrist': {'EDA': [...], 'TEMP': [...], 'BVP': [...]}}, 'label': [...]}
    # Data is usually 64 Hz for wrist sensors or 4 Hz (EDA/TEMP). Let's simulate a standard generic flow.
    
    # 5 minutes of data = 300 seconds
    # Simulate EDA at 4 Hz
    eda_length = 300 * 4
    eda_signal = np.sin(np.linspace(0, 10, eda_length)) + np.random.normal(0, 0.1, eda_length)
    eda_signal = eda_signal.reshape(-1, 1) # Standard column vector in WESAD
    
    # Simulate TEMP at 4 Hz
    temp_signal = 32 + np.cumsum(np.random.normal(0, 0.05, eda_length)) * 0.1
    temp_signal = temp_signal.reshape(-1, 1)

    # Simulate BVP at 64 Hz
    bvp_length = 300 * 64
    bvp_signal = np.sin(np.linspace(0, 100, bvp_length)) + np.random.normal(0, 0.5, bvp_length)
    bvp_signal = bvp_signal.reshape(-1, 1)
    
    # Simulating wrist ACC at 32Hz, not using much but good for form
    
    # Simulate labels at 700Hz original frequency (WESAD labels are sampled at 700Hz for all)
    # Let's say: 0=Transient, 1=Baseline, 2=Stress, 3=Amusement, 4=Meditation
    label_length = 300 * 700
    labels = np.ones(label_length) # Baseline
    # minute 2-3 is Stress
    labels[60*700:120*700] = 2 
    # minute 4-5 is Amusement
    labels[180*700:240*700] = 3

    data = {
        'subject': subject_id,
        'signal': {
            'wrist': {
                'EDA': eda_signal,
                'TEMP': temp_signal,
                'BVP': bvp_signal
            }
        },
        'label': labels
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(data, f, protocol=2) # WESAD uses protocol 2 usually
        
    print(f"Mock WESAD data generated at {output_path}")

def generate_mock_csvs(output_dir):
    """Generates mock Empatica E4 CSV files for frontend testing."""
    os.makedirs(output_dir, exist_ok=True)
    
    # EDA.csv
    # first row freq, rest data
    eda_data = [4.0] + list(np.abs(np.random.normal(0.5, 0.2, 500)))
    pd.DataFrame(eda_data).to_csv(os.path.join(output_dir, 'EDA.csv'), index=False, header=False)
    
    # TEMP.csv
    temp_data = [4.0] + list(np.random.normal(32.5, 0.3, 500))
    pd.DataFrame(temp_data).to_csv(os.path.join(output_dir, 'TEMP.csv'), index=False, header=False)
    
    # HR.csv
    hr_data = [1.0] + list(np.random.normal(75, 5, 125))
    pd.DataFrame(hr_data).to_csv(os.path.join(output_dir, 'HR.csv'), index=False, header=False)

    print(f"Mock E4 CSV files generated in {output_dir}")

if __name__ == '__main__':
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    # WESAD structure
    wesad_dir = os.path.join(data_dir, 'WESAD', 'S2')
    os.makedirs(wesad_dir, exist_ok=True)
    generate_mock_wesad_pkl(os.path.join(wesad_dir, 'S2.pkl'), subject_id='S2')
    
    # Mock uploads
    upload_dir = os.path.join(data_dir, 'test_uploads')
    generate_mock_csvs(upload_dir)
