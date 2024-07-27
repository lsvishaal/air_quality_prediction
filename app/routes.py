# app/routes.py

from flask import Blueprint, request, jsonify
import joblib
import pandas as pd
import os

routes = Blueprint('routes', __name__)

# Load the model and the preprocessing pipeline
model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'air_quality_prediction_model.pkl')
model = joblib.load(model_path)

preprocessor_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'preprocessor.pkl')
preprocessor = joblib.load(preprocessor_path)

@routes.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the JSON data from the request
        data = request.get_json(force=True)
        
        # Convert JSON data to DataFrame
        df = pd.DataFrame([data])
        
        # Check and add missing columns if needed
        required_columns = {'Year', 'Month', 'Day', 'Status', 'Country'}
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            for col in missing_columns:
                df[col] = ''  # or use appropriate default values

        # Apply the preprocessing pipeline
        df_processed = preprocessor.transform(df)
        
        # Make prediction
        prediction = model.predict(df_processed)
        
        # Return prediction as JSON
        return jsonify({'prediction': prediction[0]})
    
    except Exception as e:
        return jsonify({'error': str(e)})