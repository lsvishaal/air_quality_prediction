import pandas as pd

# Load the air quality data
air_quality_data = pd.read_csv('data/air_quality_data.csv')
# Load the predictions
predictions = pd.read_csv('predictions.csv')
# Load the test data
X_test = pd.read_csv('data/X_test.csv')

# Extract unique countries from air_quality_data
original_countries = set(air_quality_data['Country'].unique())

# Extract the dates from X_test that correspond to predictions
predicted_dates = X_test['Date']

# Merge predictions with the test data to get the corresponding countries
X_test['Predicted AQI'] = predictions['Predicted AQI']
predicted_countries = set(X_test['Country'].unique())

# Identify countries that are in the predictions but not in the original data
new_countries_in_predictions = predicted_countries - original_countries

print("Countries in predictions but not in the original data:")
print(new_countries_in_predictions)
