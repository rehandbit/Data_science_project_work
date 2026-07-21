import pandas as pd
import numpy as np

def preprocess_new_data(new_data, preprocessing_info):
    """Preprocess new house data same as training data"""
    data = new_data.copy()
    
    # Handle missing values
    for col in data.columns:
        if col in preprocessing_info['feature_names']:
            if data[col].isnull().any():
                data[col].fillna(data[col].median(), inplace=True)
    
    # Ensure all features are present
    for feature in preprocessing_info['feature_names']:
        if feature not in data.columns:
            data[feature] = 0
    
    # Reorder columns
    data = data[preprocessing_info['feature_names']]
    
    return data