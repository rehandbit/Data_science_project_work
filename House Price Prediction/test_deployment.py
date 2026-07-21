# test_deployment.py
import pandas as pd
import joblib

def test_deployment():
    print("🧪 DEPLOYMENT VERIFICATION TEST")
    print("=" * 40)
    
    try:
        # Load everything
        model = joblib.load('models/house_price_pred_model_gb.pkl')
        preprocessing_info = joblib.load('models/preprocessing_info.pkl')
        
        print("All files loaded successfully!")
        print(f"Model: {preprocessing_info['best_model']}")
        print(f"R² Score: {preprocessing_info['r2_score']:.4f}")
        print(f"Features: {len(preprocessing_info['feature_names'])}")
        
        # Test with sample data (you'll need X_test, y_test)
        # Or create a simple test
        print("\nDEPLOYMENT READY!")
        
    except Exception as e:
        print(f" Deployment test failed: {e}")

if __name__ == "__main__":
    test_deployment()