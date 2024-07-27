# create_preprocessor.py

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import os

# Define the preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['Status', 'Country']),
        # Add other transformers for numerical columns if needed
    ],
    remainder='passthrough'
)

# Define the path to save the preprocessor
preprocessor_path = os.path.join(os.path.dirname(__file__), 'models', 'preprocessor.pkl')

# Ensure the models directory exists
os.makedirs(os.path.dirname(preprocessor_path), exist_ok=True)

# Save the preprocessor
joblib.dump(preprocessor, preprocessor_path)