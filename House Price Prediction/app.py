import numpy as np
from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load ONLY Gradient Boost model
model_gb = joblib.load('models/house_price_pred_model_gb.pkl')
preprocessing_info = joblib.load('models/preprocessing_info.pkl')
encoder = joblib.load('models/label_encoder.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        house_data = request.form.to_dict()
        print(f'Received data: {house_data}')
        
        # Convert to DataFrame
        house_df = pd.DataFrame([house_data])
        print(f"Data frame: {house_df}")
        
        # Preprocess
        processed_data = preprocess_simple(house_df, preprocessing_info, encoder)
        print(f"Processed data shape: {processed_data.shape}")
        print(f"Processed data types: {processed_data.dtypes}")
        print(f"Processed data sample: {processed_data.iloc[0]}")
        
        # Predict
        prediction = model_gb.predict(processed_data)[0]
        print(f"Raw prediction: {prediction}")
        
        return jsonify({
            'predicted_price': f"${prediction:,.2f}",
            'price_range': f"${prediction*0.9:,.2f} - ${prediction*1.1:,.2f}",
            'model_used': 'Gradient Boosting',
            'accuracy': f"{preprocessing_info['r2_score']*100:.1f}%"
        })
    
    except Exception as e:
        print(f'Error: {e}')
        import traceback
        print(f'Traceback: {traceback.format_exc()}')
        return jsonify({'error': str(e)})

def preprocess_simple(new_data, preprocessing_info, encoder):
    """Preprocessing using manual encoding to match training data"""
    data = new_data.copy()
    
    
    # Convert categorical strings to integers
    for col, mapping in categorical_mapping.items():
        if col in data.columns:
            data[col] = data[col].map(mapping).fillna(0).astype(int)
    
    # Convert numerical fields
    if 'GrLivArea' in data.columns:
        data['GrLivArea'] = data['GrLivArea'].astype(float)
    if 'OverallQual' in data.columns:
        data['OverallQual'] = data['OverallQual'].astype(int)
    
    # Set defaults for missing features
    gr_liv_area = float(data.get('GrLivArea', [0])[0]) if 'GrLivArea' in data else 0
    
    smart_defaults = {
        'GarageCars': 1, 'FullBath': 2, 'BedroomAbvGr': 3, 
        'TotalBsmtSF': int(gr_liv_area * 0.3), 'YearBuilt': 1990, 'OverallCond': 5,
    }
    
    for feature in preprocessing_info['feature_names']:
        if feature not in data.columns:
            data[feature] = smart_defaults.get(feature, 0)
    
    data = data[preprocessing_info['feature_names']]
    return data
if __name__ == '__main__':
    app.run(debug=True)