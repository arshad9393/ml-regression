from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

# Route for homepage
@app.route('/')
def home():
    return render_template('index.html')

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Retrieve all form inputs (13 independent variables)
        features = [float(x) for x in request.form.values()]
        final_features = np.array(features).reshape(1, -1)
        prediction = model.predict(final_features)[0]

        return render_template('index.html', prediction_text=f'🏠 Predicted House Price (MEDV): ${prediction:,.2f}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {e}')

if __name__ == "__main__":
    app.run(debug=True)
