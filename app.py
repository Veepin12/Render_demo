from flask import Flask, render_template, request
import pickle
import numpy as np

# ✅ FIRST create app
app = Flask(__name__)

# ✅ THEN load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# ✅ Home route
@app.route("/")
def home():
    return render_template("index.html")

# ✅ Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_features = [float(x) for x in request.form.values()]
        final_input = np.array(input_features).reshape(1, -1)

        prediction = model.predict(final_input)

        return render_template(
            "index.html",
            prediction_text=f"Prediction Result: {prediction[0]}"
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction_text=f"Error: {str(e)}"
        )

# ✅ LAST run app
if __name__ == "__main__":
    app.run(debug=True)