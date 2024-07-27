from flask import Flask, request, jsonify
import joblib
import os
import pandas as pd

def create_app():
    app = Flask(__name__)

    # Load the model and encoders/transformers
    model_path = 'C:/Users/Atomic/Documents/Code/air_quality_prediction/models/air_quality_prediction_model.pkl'
    encoder_path = 'C:/Users/Atomic/Documents/Code/air_quality_prediction/models/encoder.pkl'
    model = None
    encoder = None

    try:
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            print(f"Model loaded successfully from {model_path}")
        else:
            print(f"Model file not found at {model_path}")

        if os.path.exists(encoder_path):
            encoder = joblib.load(encoder_path)
            print(f"Encoder loaded successfully from {encoder_path}")
        else:
            print(f"Encoder file not found at {encoder_path}")
    except Exception as e:
        print(f"Error loading model or encoder: {e}")

    @app.route('/')
    def index():
        return "Air Quality Prediction App"

    @app.route('/predict', methods=['POST'])
    def predict():
        if not model or not encoder:
            return jsonify({"error": "Model or encoder not loaded"}), 500

        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid input"}), 400

        try:
            # Extract features from the input data
            year = data['Year']
            month = data['Month']
            day = data['Day']
            status = data['Status']
            country = data['Country']

            # Preprocess the input data
            input_data = pd.DataFrame([{
                'Year': year,
                'Month': month,
                'Day': day,
                'Status': status,
                'Country': country
            }])

            # Apply the same transformations as during training
            input_data_transformed = encoder.transform(input_data)

            # Make a prediction
            prediction = model.predict(input_data_transformed)

            # Return the prediction result
            return jsonify({"prediction": prediction[0]})
        except KeyError as e:
            return jsonify({"error": f"Missing key: {e}"}), 400
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=5000, debug=True)