import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# Step 1: Preprocess the Data
# Load the data
X_train = pd.read_csv(r'C:\Users\Atomic\Documents\Code\air_quality_prediction\data\X_train.csv')
X_test = pd.read_csv(r'C:\Users\Atomic\Documents\Code\air_quality_prediction\data\X_test.csv')
y_train = pd.read_csv(r'C:\Users\Atomic\Documents\Code\air_quality_prediction\data\y_train.csv')['AQI Value']
y_test = pd.read_csv(r'C:\Users\Atomic\Documents\Code\air_quality_prediction\data\y_test.csv')['AQI Value']

# Preprocessing steps
def preprocess_data(X):
    # Convert 'Date' to datetime and extract year, month, and day
    X['Date'] = pd.to_datetime(X['Date'])
    X['Year'] = X['Date'].dt.year
    X['Month'] = X['Date'].dt.month
    X['Day'] = X['Date'].dt.day
    X.drop('Date', axis=1, inplace=True)
    
    return X

X_train = preprocess_data(X_train)
X_test = preprocess_data(X_test)

# Define categorical columns for one-hot encoding
categorical_features = ['Country', 'Status']
one_hot_encoder = OneHotEncoder()

# Create a preprocessing and modeling pipeline
preprocessor = ColumnTransformer(transformers=[
    ('onehot', one_hot_encoder, categorical_features)],
    remainder='passthrough')

model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('regressor', RandomForestRegressor())])

# Step 2: Train the Model
model.fit(X_train, y_train)

# Predict and evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Step 3: Save the Model
model_path = r'C:\Users\Atomic\Documents\Code\air_quality_prediction\models\air_quality_prediction_model.pkl'
joblib.dump(model, model_path)