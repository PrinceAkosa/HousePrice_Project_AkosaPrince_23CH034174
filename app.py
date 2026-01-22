from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the trained model and preprocessing objects
model_package = joblib.load('model/titanic_survival_model.pkl')
model = model_package['model']
scaler = model_package['scaler']
label_encoders = model_package['label_encoders']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        pclass = int(request.form['pclass'])
        sex = request.form['sex']
        age = float(request.form['age'])
        fare = float(request.form['fare'])
        embarked = request.form['embarked']
        
        # Encode categorical variables
        sex_encoded = 1 if sex == 'male' else 0
        embarked_encoded = {'C': 0, 'Q': 1, 'S': 2}[embarked]
        
        # Create DataFrame with passenger data
        passenger_data = pd.DataFrame({
            'Pclass': [pclass],
            'Sex': [sex_encoded],
            'Age': [age],
            'Fare': [fare],
            'Embarked': [embarked_encoded]
        })
        
        # Scale the features
        passenger_scaled = scaler.transform(passenger_data)
        
        # Make prediction
        prediction = model.predict(passenger_scaled)[0]
        probability = model.predict_proba(passenger_scaled)[0]
        
        # Prepare result
        result = {
            'prediction': 'Survived' if prediction == 1 else 'Did Not Survive',
            'survived': bool(prediction == 1),
            'probability': float(probability[prediction] * 100)
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)