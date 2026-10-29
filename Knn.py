# Manual KNN implementation as a drop-in replacement for sklearn's KNN
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from sklearn.naive_bayes import GaussianNB        # not used in this snippet but left out intentionally
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# from sklearn.neighbors import KNN # DO NOT import sklearn KNN; we implement it below
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.impute import SimpleImputer
from collections import Counter

# ---------------------------
# Manual KNN class (drop-in)
# ---------------------------
class KNN:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self._fit_X = None
        self._fit_y = None

    def fit(self, X, y):
        """Just store the training data."""
        self._fit_X = np.array(X)
        self._fit_y = np.array(y)
        return self

    def _euclidean_distance(self, a, b):
        """Compute Euclidean distance between one test sample and one train sample."""
        dist = 0.0
        for i in range(len(a)):
            dist += (a[i] - b[i]) ** 2
        return dist ** 0.5

    def predict(self, X):
        """Predict class using simple KNN logic with loops."""
        X = np.array(X)
        predictions = []

        # Loop through each test sample
        for test_point in X:

            distances = []

            # Compute distance to EVERY train point
            for i in range(len(self._fit_X)):
                train_point = self._fit_X[i]
                dist = self._euclidean_distance(test_point, train_point)
                distances.append((dist, self._fit_y[i]))

            # Sort neighbors by distance
            distances.sort(key=lambda x: x[0])

            # Take the first k neighbors
            k_neighbors = distances[:self.n_neighbors]

            # Count votes
            class_votes = {}
            for _, label in k_neighbors:
                class_votes[label] = class_votes.get(label, 0) + 1

            # Pick the class with the most votes
            prediction = max(class_votes, key=class_votes.get)
            predictions.append(prediction)

        return np.array(predictions)

# ---------------------------
# Your original pipeline
# ---------------------------

# Loading and Previewing Dataset:
data = pd.read_csv("heart_modified.csv")

# Display first few rows:
print("Dataset preview:")
print(data.head())

# Check sum of null values before handling:
print("\nNull values before handling:")
print(data.isnull().sum())

# Separate numeric and categorical columns:
numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = data.select_dtypes(include=['object']).columns

# Imputers:
num_imputer = SimpleImputer(strategy='mean')
cat_imputer = SimpleImputer(strategy='most_frequent')

data[numeric_cols] = num_imputer.fit_transform(data[numeric_cols])
if len(categorical_cols) > 0:
    data[categorical_cols] = cat_imputer.fit_transform(data[categorical_cols])

# Check sum of null values after handling:
print("\nNull values after handling:")
print(data.isnull().sum())

# One-Hot Encode categorical features:
data = pd.get_dummies(data, drop_first=True)

# Prepare features and target:
# Keep same column name 'HeartDisease' as you requested
X = data.drop(columns=['HeartDisease']).values
y = data['HeartDisease'].values

# Train-test split:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

# Scale features:
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Applying manual KNN (drop-in)
knn = KNN(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
y_pred_knn = knn.predict(X_test_scaled)
knn_score = accuracy_score(y_test, y_pred_knn)

print(f"\nKNN Accuracy: {knn_score:.2f}")
print("\nConfusion Matrix - KNN:")
print(confusion_matrix(y_test, y_pred_knn))
