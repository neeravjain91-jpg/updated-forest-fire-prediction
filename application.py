from pathlib import Path

from flask import Flask, render_template, request

from model_training import load_or_train_model

app = Flask(__name__)

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

model, metrics = load_or_train_model()


@app.route("/")
def index():
    return render_template("index.html", metrics=metrics)


@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():
    result = None

    if request.method == "POST":
        values = {
            "Temperature": float(request.form["Temperature"]),
            "RH": float(request.form["RH"]),
            "Ws": float(request.form["Ws"]),
            "Rain": float(request.form["Rain"]),
            "FFMC": float(request.form["FFMC"]),
            "DMC": float(request.form["DMC"]),
            "DC": float(request.form["DC"]),
            "ISI": float(request.form["ISI"]),
            "BUI": float(request.form["BUI"]),
            "Region": int(request.form["Region"]),
        }

        probability = float(model.predict_proba([list(values.values())])[0][1])
        prediction = int(probability >= 0.5)

        result = {
            "prediction": prediction,
            "probability": probability * 100,
            "label": "Fire Risk Detected" if prediction else "Low Fire Risk",
        }

    return render_template("home.html", result=result, metrics=metrics)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
