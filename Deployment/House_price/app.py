from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load trained model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html', prediction_text=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form inputs and convert to numpy array
        features = [float(x) for x in request.form.values()]
        final_features = np.array(features).reshape(1, -1)

        # Make prediction
        prediction = model.predict(final_features)[0]

        return render_template(
            'index.html',
            prediction_text=f"🏡 Estimated House Price (MEDV): ${prediction:,.2f}",
            success=True
        )

    except Exception as e:
        return render_template(
            'index.html',
            prediction_text=f"⚠️ Error: {str(e)}",
            success=False
        )

if __name__ == "__main__":
    app.run(debug=True)