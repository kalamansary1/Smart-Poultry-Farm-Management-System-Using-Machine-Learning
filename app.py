# app.py
from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle

app = Flask(__name__)

# Load model and feature names
with open("model.pkl", "rb") as file:
    model, selected_features = pickle.load(file)

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        try:
            # Get form data
            form_data = request.form

            # Convert input data into a DataFrame
            data = pd.DataFrame([[form_data[feature] for feature in selected_features]], columns=selected_features)

            # Convert numeric fields to float
            for col in ["temperature", "humidity", "air_quality", "weight", "feed_intake"]:
                data[col] = data[col].astype(float)

            # Convert timestamp to UNIX time (if necessary)
            data["timestamp"] = pd.to_datetime(data["timestamp"]).astype(int) / 10**9

            # Make prediction
            prediction = model.predict(data)[0]

            # Return prediction as JSON
            return jsonify({"prediction": round(float(prediction), 2)})

        except Exception as e:
            return jsonify({"error": str(e)})

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
