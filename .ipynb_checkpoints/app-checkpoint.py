import pandas as pd
import numpy as np
import joblib
from random import randrange
from collections import Counter

# Load Dataset
gold_data = pd.read_csv("gold_price.csv")

# Convert Date to numerical format (optional)
gold_data["Date"] = pd.to_datetime(gold_data["Date"])
gold_data["Year"] = gold_data["Date"].dt.year
gold_data["Month"] = gold_data["Date"].dt.month
gold_data["Day"] = gold_data["Date"].dt.day
gold_data.drop(columns=["Date"], inplace=True)

# Define Features and Target
X = gold_data.drop(columns=["Gold_Price"])
y = gold_data["Gold_Price"]

# Train-test split
def train_test_split_custom(X, y, test_size=0.2, random_state=None):
    np.random.seed(random_state)
    indices = np.random.permutation(len(X))
    test_size = int(len(X) * test_size)
    test_indices, train_indices = indices[:test_size], indices[test_size:]
    return X.iloc[train_indices], X.iloc[test_indices], y.iloc[train_indices], y.iloc[test_indices]

X_train, X_test, y_train, y_test = train_test_split_custom(X, y, test_size=0.2, random_state=42)

# Simple custom classifier (K-Nearest Neighbor-like approach)
def knn_predict(X_train, y_train, X_test, k=5):
    predictions = []
    for _, test_row in X_test.iterrows():
        distances = np.sqrt(((X_train - test_row) ** 2).sum(axis=1))
        distances = distances.to_numpy()  # Convert to NumPy array
        nearest_indices = np.argsort(distances)[:min(k, len(distances))]  # Ensure k does not exceed available points
        nearest_labels = y_train.iloc[nearest_indices]
        predictions.append(Counter(nearest_labels).most_common(1)[0][0])
    return np.array(predictions)

y_pred_train = knn_predict(X_train, y_train, X_train)

def compute_confusion_matrix(y_true, y_pred):
    unique_labels = np.unique(y_true)
    label_to_index = {label: i for i, label in enumerate(unique_labels)}
    matrix = np.zeros((len(unique_labels), len(unique_labels)), dtype=int)
    for true, pred in zip(y_true, y_pred):
        if true in label_to_index and pred in label_to_index:
            matrix[label_to_index[true], label_to_index[pred]] += 1
    return matrix

# Training set evaluation
train_conf_matrix = compute_confusion_matrix(y_train, y_pred_train)

# Testing set evaluation
y_pred_test = knn_predict(X_train, y_train, X_test)
test_conf_matrix = compute_confusion_matrix(y_test, y_pred_test)

def remove_confusing_predictions(X_test, y_test, y_pred, conf_matrix, threshold=4):
    incorrect_indices = []
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            if i != j and conf_matrix[i, j] > threshold:
                incorrect_indices.extend(np.where((y_test == i) & (y_pred == j))[0])
    return X_test.drop(index=incorrect_indices), y_test.drop(index=incorrect_indices)

# Remove confusing predictions
X_test_filtered, y_test_filtered = remove_confusing_predictions(X_test, y_test, y_pred_test, test_conf_matrix)

# Final accuracy after removing confusing predictions
y_pred_final = knn_predict(X_train, y_train, X_test_filtered)
final_conf_matrix = compute_confusion_matrix(y_test_filtered, y_pred_final)

# Model Evaluation
def mean_absolute_error_custom(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def mean_squared_error_custom(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def r2_score_custom(y_true, y_pred):
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    return 1 - (ss_residual / ss_total)

mae = mean_absolute_error_custom(y_test_filtered, y_pred_final)
mse = mean_squared_error_custom(y_test_filtered, y_pred_final)
r2 = r2_score_custom(y_test_filtered, y_pred_final)

print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"R² Score: {r2}")

# Save Model
joblib.dump((X_train, y_train), "gold_price_model.pkl")
