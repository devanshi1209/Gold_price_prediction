import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Load CSV file
data = pd.read_csv("gold_price.csv")

data['Date'] = pd.to_datetime(data['Date'])
data.sort_values('Date', inplace=True)

# Feature Engineering
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['Day'] = data['Date'].dt.day
data['DayOfWeek'] = data['Date'].dt.dayofweek

# Dynamic Rolling Mean Calculation
rolling_window = 5
data['Rolling_Mean'] = data['Gold_Price'].rolling(window=rolling_window).mean()
data.dropna(inplace=True)

X = data[['Year', 'Month', 'Day', 'DayOfWeek', 'Rolling_Mean']]
y = data['Gold_Price']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(random_state=42),
    "SVR": SVR(),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42)
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)
results = {}

for name, model in models.items():
    errors = []
    for train_index, test_index in kf.split(X_train):
        X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[test_index]
        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[test_index]
        model.fit(X_train_fold, y_train_fold)
        y_pred = model.predict(X_val_fold)
        errors.append(mean_absolute_error(y_val_fold, y_pred))
    results[name] = np.mean(errors)

# Select the best model based on lowest MAE
best_model_name = min(results, key=results.get)
best_model = models[best_model_name]
best_model.fit(X_train, y_train)

# Save the best model
with open("best_gold_price_model.pkl", "wb") as model_file:
    pickle.dump(best_model, model_file)

print(f"Best model ({best_model_name}) saved successfully.")

# Prediction Function
def predict_gold_price(year, month, day):
    day_of_week = pd.Timestamp(year, month, day).dayofweek
    last_five_prices = data[data['Year'] == year]['Gold_Price'].tail(rolling_window)
    rolling_mean = last_five_prices.mean() if not last_five_prices.empty else data['Gold_Price'].mean()
    input_data = np.array([[year, month, day, day_of_week, rolling_mean]])
    return float(best_model.predict(input_data)[0])
