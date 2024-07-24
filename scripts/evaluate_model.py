# scripts/evaluate_model.py

import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import joblib

def evaluate_model():
    # Load the model
    model = joblib.load('models/random_forest_model.pkl')
    
    # Load the testing data
    X_test = pd.read_csv('data/X_test.csv')
    y_test = pd.read_csv('data/y_test.csv').values.ravel()
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")

if __name__ == "__main__":
    evaluate_model()
