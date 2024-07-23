import pandas as pd
from datetime import datetime

def preprocess_data(file_path):
    data = pd.read_csv(file_path)

    data['Date'] = pd.to_datetime(data['Date'])
    data['Month'] = data['Date'].dt.month
    data['Day'] = data['Date'].dt.day
    data['DayOfWeek'] = data['Date'].dt.dayofweek
    data = data.dropna()
    # Correcting the column name from 'AQI' to 'AQI Value'
    data['AQI_Normalized'] = (data['AQI Value'] - data['AQI Value'].mean()) / data['AQI Value'].std()

    data.to_csv('data/preprocessed_air_quality_data.csv', index=False)
    print("Data PreProcessing Completed.")

if __name__ == "__main__":
    preprocess_data('data/air_quality_data.csv')