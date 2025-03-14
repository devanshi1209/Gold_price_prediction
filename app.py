from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd
import json
from graph import get_gold_price_graph

app = Flask(__name__)

# Load the trained model
with open("best_gold_price_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Load dataset
data = pd.read_csv("gold_price.csv")
data['Date'] = pd.to_datetime(data['Date'])
data.sort_values('Date', inplace=True)
data['Year'] = data['Date'].dt.year
data['Gold_Price'] = data['Gold_Price']

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/graph-data')
def graph_data():
    graph_json = get_gold_price_graph()
    return jsonify(json.loads(graph_json))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        request_data = request.get_json()
        year = int(request_data['year'])
        month = int(request_data['month'])
        day = int(request_data['day'])

        # Validate date range (2015-2029)
        if year < 2015 or year > 2029:
            return jsonify({"error": "Year must be between 2015 and 2029"}), 400

        # Calculate input features
        day_of_week = pd.Timestamp(year, month, day).dayofweek
        last_five_prices = data[data['Year'] == year]['Gold_Price'].tail(5)
        rolling_mean = last_five_prices.mean() if not last_five_prices.empty else data['Gold_Price'].mean()
        
        # Model prediction in USD
        input_data = np.array([[year, month, day, day_of_week, rolling_mean]])
        prediction_usd = model.predict(input_data)[0]
        print(f"USD Prediction for {year}-{month}-{day}: {prediction_usd}")  # Debug print
        
        return jsonify({
            "price": float(prediction_usd)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/how-to-use')
def how_to_use():
    return render_template("how_to_use.html")

@app.route('/contact')
def contact():
    return render_template("contact.html")

if __name__ == '__main__':
    app.run()