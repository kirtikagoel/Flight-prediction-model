from flask import Flask, request, render_template
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load trained model (correct path)
model = pickle.load(open("model/flight_model.pkl", "rb"))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        airline = int(request.form["airline"])
        source = int(request.form["source"])
        destination = int(request.form["destination"])
        stops = int(request.form["stops"])
        hours = int(request.form["hours"])
        minutes = int(request.form["minutes"])

        duration = hours * 60 + minutes

        features = np.array([[airline, source, destination, stops, duration]])
        prediction = model.predict(features)[0]
        output = round(prediction, 2)

        return render_template(
            "index.html",
            prediction_text=f"Estimated Flight Price: ₹ {output}"
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction_text=f"Error: {str(e)}"
        )


if __name__ == "__main__":
    # For deployment (Render/Heroku etc.)
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
