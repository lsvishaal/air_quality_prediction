# app/app.py

from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the model
model = joblib.load('models/random_forest_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the JSON data from the request
        data = request.get_json(force=True)
        
        # Convert JSON data to DataFrame
        df = pd.DataFrame([data])
        
        # Make prediction
        prediction = model.predict(df)
        
        # Return prediction as JSON
        return jsonify({'prediction': prediction[0]})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
