import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import mplfinance as mpf
import json
from io import BytesIO
import base64

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

# Predictions
y_pred = model.predict(X)

def get_encoded_plot(fig):
    img = BytesIO()
    fig.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

def get_gold_price_graph():
    plots = {}
    
    # Actual vs Predicted Prices
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data['Date'], y, label="Actual Price", color='blue')
    ax.plot(data['Date'], y_pred, label="Predicted Price", color='red', linestyle='dashed')
    ax.set_xlabel("Date")
    ax.set_ylabel("Gold Price")
    ax.set_title("Actual vs Predicted Gold Price")
    ax.legend()
    plots['actual_vs_predicted'] = get_encoded_plot(fig)
    
    # Error Distribution
    errors = y - y_pred
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(errors, bins=30, kde=True, color='purple', ax=ax)
    ax.set_xlabel("Prediction Error")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Prediction Errors")
    plots['error_distribution'] = get_encoded_plot(fig)
    
    # Feature Importance (for Tree-Based Models)
    if hasattr(model, 'feature_importances_'):
        feature_importance = model.feature_importances_
        feature_names = X.columns
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x=feature_importance, y=feature_names, palette="viridis", ax=ax)
        ax.set_xlabel("Importance Score")
        ax.set_ylabel("Features")
        ax.set_title("Feature Importance in Gold Price Prediction")
        plots['feature_importance'] = get_encoded_plot(fig)

    # Candlestick Chart
    data['Open'] = data['Gold_Price'].shift(1)
    data['Close'] = data['Gold_Price']
    data['High'] = data['Gold_Price'].rolling(5).max()
    data['Low'] = data['Gold_Price'].rolling(5).min()
    candle_data = data[['Date', 'Open', 'High', 'Low', 'Close']].dropna()
    candle_data.set_index('Date', inplace=True)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    mpf.plot(candle_data, type='candle', style='charles', ax=ax)
    ax.set_title("Gold Price Candlestick Chart")
    plots['candlestick_chart'] = get_encoded_plot(fig)
    
    return json.dumps(plots)