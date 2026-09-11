from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)

# Load trained model
with open("models/model.pkl", "rb") as file:
    model = pickle.load(file)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict/", methods=["POST"])
def predict():

    years = float(request.form["years"])

    prediction = model.predict(np.array([[years]]))

    output = round(prediction[0], 2)

    return render_template(
        "index.html",
        prediction_text="Predicted Salary = " + str(output)
    )


if __name__ == "__main__":
    app.run(debug=True)
