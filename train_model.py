import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pickle
import os

# Create simple dataset manually
data = {
    "Airline": [1, 2, 3, 1, 2, 3],
    "Source": [1, 2, 3, 2, 3, 1],
    "Destination": [3, 1, 2, 3, 1, 2],
    "Stops": [0, 1, 2, 1, 0, 2],
    "Duration": [90, 120, 300, 180, 95, 250],
    "Price": [4000, 6000, 10000, 7500, 4200, 9500]
}

df = pd.DataFrame(data)

X = df[["Airline", "Source", "Destination", "Stops", "Duration"]]
y = df["Price"]

model = RandomForestRegressor()
model.fit(X, y)

os.makedirs("model", exist_ok=True)
pickle.dump(model, open("model/flight_model.pkl", "wb"))

print("Model trained and saved successfully!")
