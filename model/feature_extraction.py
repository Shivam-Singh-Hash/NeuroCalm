import numpy as np
import pandas as pd
from scipy.stats import linregress

def extract_eda_features(eda_segment):
    """
    Extract features from a window of EDA data.
    """
    if len(eda_segment) == 0:
        return [0]*5
    mean_eda = np.mean(eda_segment)
    std_eda = np.std(eda_segment)
    min_eda = np.min(eda_segment)
    max_eda = np.max(eda_segment)
    
    # slope
    if len(eda_segment) > 1:
        slope = linregress(np.arange(len(eda_segment)), eda_segment).slope
    else:
        slope = 0
        
    return [mean_eda, std_eda, min_eda, max_eda, slope]

def extract_temp_features(temp_segment):
    """
    Extract features from a window of TEMP data.
    """
    if len(temp_segment) == 0:
        return [0]*3
        
    mean_temp = np.mean(temp_segment)
    std_temp = np.std(temp_segment)
    
    # slope
    if len(temp_segment) > 1:
        slope = linregress(np.arange(len(temp_segment)), temp_segment).slope
    else:
        slope = 0
        
    return [mean_temp, std_temp, slope]

def extract_bvp_features(bvp_segment, sampling_rate=64):
    """
    Extract dummy HR/HRV features from BVP window.
    Real extraction requires peak detection algorithms like neurokit2.
    Here we implement a simplified mock version mapping variance to HRV.
    """
    if len(bvp_segment) == 0:
        return [0]*3
        
    # approximate HR based on BVP energy variance
    # This is a proxy for the actual HR for our mock model
    mean_hr = 60 + (np.std(bvp_segment) * 10)
    sdnn = np.std(bvp_segment) * 2
    rmssd = np.mean(np.abs(np.diff(bvp_segment)))
    
    return [mean_hr, sdnn, rmssd]

def process_window(eda_window, temp_window, bvp_window):
    """
    Combine all extracted features for a specific temporal window.
    """
    eda_f = extract_eda_features(eda_window)
    temp_f = extract_temp_features(temp_window)
    bvp_f = extract_bvp_features(bvp_window)
    
    # Return as single flat list
    return eda_f + temp_f + bvp_f

def get_feature_names():
    return [
        'eda_mean', 'eda_std', 'eda_min', 'eda_max', 'eda_slope',
        'temp_mean', 'temp_std', 'temp_slope',
        'hr_mean', 'hrv_sdnn', 'hrv_rmssd'
    ]
