# scripts/split_data.py

import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(file_path, test_size=0.2, random_state=42):
    data = pd.read_csv(file_path)
    
    # Assuming the target variable is the last column
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Save the splits
    X_train.to_csv('../data/X_train.csv', index=False)
    X_test.to_csv('../data/X_test.csv', index=False)
    y_train.to_csv('../data/y_train.csv', index=False)
    y_test.to_csv('../data/y_test.csv', index=False)

if __name__ == "__main__":
    split_data('../../data/air_quality_data.csv')
