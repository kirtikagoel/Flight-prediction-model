from flask import Flask, request, render_template
import pickle
import numpy as np

# Create Flask app
app = Flask(__name__)

# Load trained model
model = pickle.load(open("model/flight_model.pkl", "rb"))



# Home Page
@app.route("/")
def home():
    return render_template("index.html")


# Prediction Route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get values from form
        airline = int(request.form["airline"])
        source = int(request.form["source"])
        destination = int(request.form["destination"])
        stops = int(request.form["stops"])
        hours = int(request.form["hours"])
        minutes = int(request.form["minutes"])

        # Convert duration to total minutes
        duration = hours * 60 + minutes

        # Arrange features in same order used while training
        final_features = np.array([[airline, source, destination, stops, duration]])

        # Predict price
        prediction = model.predict(final_features)[0]
        output = round(prediction, 2)

        return render_template("index.html",
                               prediction_text=f"Estimated Flight Price: ₹ {output}")

    except Exception as e:
        return render_template("index.html",
                               prediction_text=f"Error: {str(e)}")


# Run the app
if __name__ == "__main__":
    app.run(debug=True)
