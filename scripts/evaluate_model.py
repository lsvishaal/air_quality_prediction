# scripts/evaluate_model.py

import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

def evaluate_model():
    # Load the model
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'air_quality_prediction_model.pkl')
    model = joblib.load(model_path)
    
    # Load the testing data
    X_test_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'X_test.csv')
    y_test_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'y_test.csv')
    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path).values.ravel()
    
    # Check and add missing columns
    required_columns = {'Year', 'Month', 'Day'}
    missing_columns = required_columns - set(X_test.columns)
    if missing_columns:
        for col in missing_columns:
            X_test[col] = 0  # or use appropriate default values or extract from a date column
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")

if __name__ == "__main__":
    evaluate_model()