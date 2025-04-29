import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np

class SimpleRFModel:
    def __init__(self, dataset_path):
        # Read the dataset
        df = pd.read_csv(dataset_path)

        # Separate features and target variable
        self.y = df.iloc[:, 0]  # Assuming the first column is the target (e.g., 'fracture')
        self.X = df.iloc[:, 1:]  # The rest are features

        # Split into training and testing data
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        # Train the Random Forest model
        self.model = RandomForestClassifier(random_state=1)
        self.model.fit(X_train, y_train)

    def predict(self, new_data):
        # Ensure input data is a DataFrame and matches feature names
        df = pd.DataFrame(new_data, columns=self.X.columns)
        return self.model.predict(df)
    