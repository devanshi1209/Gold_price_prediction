import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the trained model
with open("best_gold_price_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Load dataset
data = pd.read_csv("gold_price.csv")
data['Date'] = pd.to_datetime(data['Date'])
data.sort_values('Date', inplace=True)
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['Day'] = data['Date'].dt.day
data['DayOfWeek'] = data['Date'].dt.dayofweek
data['Rolling_Mean'] = data['Gold_Price'].rolling(window=5).mean()
data.dropna(inplace=True)

X = data[['Year', 'Month', 'Day', 'DayOfWeek', 'Rolling_Mean']]
y = data['Gold_Price']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and Evaluate on Training Data
y_train_pred = model.predict(X_train)
train_mae = mean_absolute_error(y_train, y_train_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
train_r2 = r2_score(y_train, y_train_pred)

# Evaluate on Test Data
y_test_pred = model.predict(X_test)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_r2 = r2_score(y_test, y_test_pred)

# Print performance metrics
print(f"Training MAE: {train_mae:.2f}, RMSE: {train_rmse:.2f}, R²: {train_r2:.2f}")
print(f"Testing MAE: {test_mae:.2f}, RMSE: {test_rmse:.2f}, R²: {test_r2:.2f}")

# Visualization: Training vs Testing Errors
metrics_df = pd.DataFrame({
    "Dataset": ["Training", "Testing"],
    "MAE": [train_mae, test_mae],
    "RMSE": [train_rmse, test_rmse],
    "R² Score": [train_r2, test_r2]
})

plt.figure(figsize=(8, 5))
sns.barplot(x="Dataset", y="RMSE", data=metrics_df, palette="coolwarm")
plt.title("Comparison of RMSE for Training and Testing Data")
plt.show()

# Detect Overfitting or Underfitting
if train_rmse < test_rmse * 0.75:
    print("Warning: Model is likely overfitting!")
elif test_rmse < train_rmse * 0.75:
    print("Warning: Model is likely underfitting!")
else:
    print("Model has a balanced fit.")