from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Base Directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load Saved Pipeline
model_path = os.path.join(BASE_DIR, "models", "rent_pipeline.pkl")
model = joblib.load(model_path)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    city = request.form["city"]
    area = float(request.form["area"])
    beds = int(request.form["beds"])
    bathrooms = int(request.form["bathrooms"])
    area_rate = float(request.form["area_rate"])

    # Create DataFrame
    data = pd.DataFrame({
        "city": [city],
        "area": [area],
        "beds": [beds],
        "bathrooms": [bathrooms],
        "area_rate": [area_rate]
    })

    # Prediction
    prediction = model.predict(data)[0]

    return render_template(
        "index.html",
        prediction=f"Predicted Rent: ₹ {prediction:,.0f}"
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)