"""
Energy Consumption Forecasting Application
LSTM-based forecasting for household power consumption optimization
"""

from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Load pre-trained model and scaler
try:
    model = load_model('lstm_power_model.h5')
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('seq_length.pkl', 'rb') as f:
        seq_length = pickle.load(f)
    print("✓ Model and scaler loaded successfully")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    model = None
    scaler = None
    seq_length = 60


def prepare_data_for_prediction(data_points):
    """Prepare data for LSTM prediction"""
    if len(data_points) < seq_length:
        return None
    
    # Normalize the data
    data_array = np.array(data_points).reshape(-1, 1)
    scaled_data = scaler.transform(data_array)
    
    # Create sequence
    X = scaled_data[-seq_length:].reshape(1, seq_length, 1)
    return X


def predict_power_consumption(recent_data):
    """Predict next power consumption value"""
    if model is None:
        return None
    
    X = prepare_data_for_prediction(recent_data)
    if X is None:
        return None
    
    # Make prediction
    scaled_prediction = model.predict(X, verbose=0)
    
    # Inverse transform to get actual value
    prediction = scaler.inverse_transform(scaled_prediction)
    return float(prediction[0][0])


def generate_forecast(recent_data, steps=24):
    """Generate multi-step forecast"""
    if model is None or len(recent_data) < seq_length:
        return None
    
    forecast = []
    current_data = list(recent_data)
    
    for _ in range(steps):
        pred = predict_power_consumption(current_data)
        if pred is None:
            break
        forecast.append(pred)
        current_data.append(pred)
        current_data = current_data[-seq_length:]
    
    return forecast


@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for single prediction"""
    try:
        data = request.json
        recent_data = data.get('recent_data', [])
        
        if not recent_data or len(recent_data) < seq_length:
            return jsonify({
                'error': f'Need at least {seq_length} data points',
                'status': 'error'
            }), 400
        
        prediction = predict_power_consumption(recent_data)
        
        return jsonify({
            'prediction': prediction,
            'unit': 'kW',
            'timestamp': datetime.now().isoformat(),
            'status': 'success'
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500


@app.route('/api/forecast', methods=['POST'])
def api_forecast():
    """API endpoint for multi-step forecast"""
    try:
        data = request.json
        recent_data = data.get('recent_data', [])
        steps = data.get('steps', 24)
        
        if not recent_data or len(recent_data) < seq_length:
            return jsonify({
                'error': f'Need at least {seq_length} data points',
                'status': 'error'
            }), 400
        
        forecast = generate_forecast(recent_data, steps)
        
        if forecast is None:
            return jsonify({
                'error': 'Forecast generation failed',
                'status': 'error'
            }), 500
        
        # Generate timestamps
        now = datetime.now()
        timestamps = [(now + timedelta(hours=i)).isoformat() for i in range(1, len(forecast) + 1)]
        
        return jsonify({
            'forecast': forecast,
            'timestamps': timestamps,
            'unit': 'kW',
            'steps': len(forecast),
            'status': 'success'
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500


@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """API endpoint for energy consumption analysis"""
    try:
        data = request.json
        consumption_data = data.get('consumption_data', [])
        
        if not consumption_data:
            return jsonify({
                'error': 'No consumption data provided',
                'status': 'error'
            }), 400
        
        consumption_array = np.array(consumption_data)
        
        # Calculate metrics
        avg_consumption = float(consumption_array.mean())
        peak_consumption = float(consumption_array.max())
        min_consumption = float(consumption_array.min())
        std_consumption = float(consumption_array.std())
        
        # Identify high consumption periods
        high_threshold = avg_consumption * 1.5
        high_periods = int((consumption_array > high_threshold).sum())
        high_percentage = (high_periods / len(consumption_array)) * 100
        
        # Optimization recommendations
        recommendations = []
        if peak_consumption / avg_consumption > 2.5:
            recommendations.append("High peak-to-average ratio detected. Consider load shifting strategies.")
        if high_percentage > 30:
            recommendations.append("Frequent high consumption periods. Implement demand response programs.")
        if std_consumption > avg_consumption * 0.5:
            recommendations.append("High consumption variability. Optimize scheduling and automation.")
        
        return jsonify({
            'metrics': {
                'average': avg_consumption,
                'peak': peak_consumption,
                'minimum': min_consumption,
                'std_dev': std_consumption,
                'peak_to_avg_ratio': peak_consumption / avg_consumption
            },
            'high_consumption_periods': {
                'count': high_periods,
                'percentage': high_percentage,
                'threshold': high_threshold
            },
            'recommendations': recommendations,
            'status': 'success'
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    model_loaded = model is not None
    return jsonify({
        'status': 'healthy' if model_loaded else 'degraded',
        'model_loaded': model_loaded,
        'timestamp': datetime.now().isoformat()
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
