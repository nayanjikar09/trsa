from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model
with open("nadi_model.pkl", "rb") as f:
    model, label_encoder = pickle.load(f)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        pulse_rate = float(request.form["pulse_rate"])
        rr_interval = float(request.form["rr_interval"])
        amplitude = float(request.form["pulse_amplitude"])
        variability = float(request.form["pulse_variability"])

        features = np.array([[pulse_rate, rr_interval, amplitude, variability]])
        pred = model.predict(features)
        prediction = label_encoder.inverse_transform(pred)[0]

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
