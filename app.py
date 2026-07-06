from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load Saved Pipeline
model = joblib.load("models/rent_pipeline.pkl")

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

    data = pd.DataFrame({
        "city": [city],
        "area": [area],
        "beds": [beds],
        "bathrooms": [bathrooms],
        "area_rate": [area_rate]
    })

    prediction = model.predict(data)

    return render_template(
        "index.html",
        prediction=f"Predicted Rent: ₹ {prediction[0]:,.0f}"
    )

if __name__ == "__main__":
    app.run(debug=True)
