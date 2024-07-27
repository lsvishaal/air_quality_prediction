from sklearn.preprocessing import OneHotEncoder
import joblib

# Example training code
encoder = OneHotEncoder(handle_unknown='ignore')
# Fit the encoder on your training data
encoder.fit(training_data)

# Save the encoder
encoder_path = 'C:/Users/Atomic/Documents/Code/air_quality_prediction/models/encoder.pkl'
joblib.dump(encoder, encoder_path)