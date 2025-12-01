from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import pickle
from catboost import CatBoostRegressor
import os

app = Flask(__name__)

# Load the trained model
with open("catboost_marks_model.pkl", "rb") as file:
    model = pickle.load(file)

# Feature columns (matching your training data)
FEATURE_COLUMNS = [
    'Q1 (5)', 'Q2 (5)', 'Q3 (5)', 'Q4 (5)', 'Q5 (5)', 'Q6 (5)', 'Q7 (5)', 
    'Q8 (5)', 'Q9 (5)', 'Q10 (5)', 'Q11 (5)', 'Q12 (5)', 'Out of 30'
]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Prepare input data
        input_data = {
            'Q1 (5)': [float(data.get('q1', 0))],
            'Q2 (5)': [float(data.get('q2', 0))],
            'Q3 (5)': [float(data.get('q3', 0))],
            'Q4 (5)': [float(data.get('q4', 0))],
            'Q5 (5)': [float(data.get('q5', 0))],
            'Q6 (5)': [float(data.get('q6', 0))],
            'Q7 (5)': [float(data.get('q7', 0))],
            'Q8 (5)': [float(data.get('q8', 0))],
            'Q9 (5)': [float(data.get('q9', 0))],
            'Q10 (5)': [float(data.get('q10', 0))],
            'Q11 (5)': [float(data.get('q11', 0))],
            'Q12 (5)': [float(data.get('q12', 0))],
            'Out of 30': [float(data.get('out_of_30', 0))]
        }
        
        df = pd.DataFrame(input_data)
        prediction = model.predict(df)[0]
        
        return jsonify({
            'success': True,
            'prediction': round(float(prediction), 2),
            'message': f'Predicted Top 9 Score: {round(float(prediction), 2)}/45'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/model-info')
def model_info():
    return jsonify({
        'r2_score': 98.27,
        'rmse': 1.0,
        'features': FEATURE_COLUMNS
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)