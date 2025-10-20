"""
Utility functions for the Energy Consumption Forecasting Application
"""

import numpy as np
import pickle
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def load_model_files(model_path: str, scaler_path: str, seq_length_path: str) -> Tuple:
    """
    Load model, scaler, and sequence length from files
    
    Args:
        model_path: Path to the trained model
        scaler_path: Path to the scaler pickle file
        seq_length_path: Path to the sequence length pickle file
    
    Returns:
        Tuple of (model, scaler, seq_length) or (None, None, None) if loading fails
    """
    try:
        from tensorflow.keras.models import load_model
        
        model = load_model(model_path)
        logger.info(f"Model loaded from {model_path}")
        
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        logger.info(f"Scaler loaded from {scaler_path}")
        
        with open(seq_length_path, 'rb') as f:
            seq_length = pickle.load(f)
        logger.info(f"Sequence length loaded from {seq_length_path}")
        
        return model, scaler, seq_length
    
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return None, None, None
    except Exception as e:
        logger.error(f"Error loading model files: {e}")
        return None, None, None


def validate_data(data: List[float], min_length: int = 1, max_length: int = 10000) -> bool:
    """
    Validate input data
    
    Args:
        data: List of numerical values
        min_length: Minimum required data points
        max_length: Maximum allowed data points
    
    Returns:
        True if data is valid, False otherwise
    """
    if not isinstance(data, list):
        return False
    
    if len(data) < min_length or len(data) > max_length:
        return False
    
    try:
        # Check if all values are numeric
        for value in data:
            float(value)
        return True
    except (ValueError, TypeError):
        return False


def normalize_data(data: np.ndarray, scaler) -> np.ndarray:
    """
    Normalize data using provided scaler
    
    Args:
        data: Input data array
        scaler: Fitted scaler object
    
    Returns:
        Normalized data array
    """
    try:
        return scaler.transform(data.reshape(-1, 1))
    except Exception as e:
        logger.error(f"Error normalizing data: {e}")
        return None


def denormalize_data(data: np.ndarray, scaler) -> np.ndarray:
    """
    Denormalize data using provided scaler
    
    Args:
        data: Normalized data array
        scaler: Fitted scaler object
    
    Returns:
        Denormalized data array
    """
    try:
        return scaler.inverse_transform(data.reshape(-1, 1))
    except Exception as e:
        logger.error(f"Error denormalizing data: {e}")
        return None


def create_sequences(data: np.ndarray, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for LSTM input
    
    Args:
        data: Input data array
        seq_length: Length of each sequence
    
    Returns:
        Tuple of (X, y) sequences
    """
    X, y = [], []
    
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    
    return np.array(X), np.array(y)


def calculate_statistics(data: List[float]) -> dict:
    """
    Calculate statistics for consumption data
    
    Args:
        data: List of consumption values
    
    Returns:
        Dictionary with statistical metrics
    """
    data_array = np.array(data)
    
    return {
        'mean': float(np.mean(data_array)),
        'median': float(np.median(data_array)),
        'std': float(np.std(data_array)),
        'min': float(np.min(data_array)),
        'max': float(np.max(data_array)),
        'q25': float(np.percentile(data_array, 25)),
        'q75': float(np.percentile(data_array, 75)),
        'iqr': float(np.percentile(data_array, 75) - np.percentile(data_array, 25))
    }


def detect_anomalies(data: List[float], threshold: float = 2.0) -> List[int]:
    """
    Detect anomalies using z-score method
    
    Args:
        data: List of consumption values
        threshold: Z-score threshold for anomaly detection
    
    Returns:
        List of indices where anomalies are detected
    """
    data_array = np.array(data)
    mean = np.mean(data_array)
    std = np.std(data_array)
    
    if std == 0:
        return []
    
    z_scores = np.abs((data_array - mean) / std)
    anomaly_indices = np.where(z_scores > threshold)[0].tolist()
    
    return anomaly_indices


def format_forecast_response(forecast: List[float], timestamps: List[str]) -> dict:
    """
    Format forecast data for API response
    
    Args:
        forecast: List of predicted values
        timestamps: List of corresponding timestamps
    
    Returns:
        Formatted dictionary
    """
    return {
        'forecast': [float(v) for v in forecast],
        'timestamps': timestamps,
        'statistics': {
            'mean': float(np.mean(forecast)),
            'max': float(np.max(forecast)),
            'min': float(np.min(forecast)),
            'std': float(np.std(forecast))
        }
    }


def get_optimization_recommendations(metrics: dict) -> List[str]:
    """
    Generate optimization recommendations based on metrics
    
    Args:
        metrics: Dictionary of consumption metrics
    
    Returns:
        List of recommendations
    """
    recommendations = []
    
    peak_to_avg = metrics.get('peak_to_avg_ratio', 0)
    if peak_to_avg > 2.5:
        recommendations.append("High peak-to-average ratio detected. Consider load shifting strategies.")
    
    high_percentage = metrics.get('high_consumption_percentage', 0)
    if high_percentage > 30:
        recommendations.append("Frequent high consumption periods. Implement demand response programs.")
    
    std_dev = metrics.get('std_dev', 0)
    mean = metrics.get('mean', 1)
    if std_dev > mean * 0.5:
        recommendations.append("High consumption variability. Optimize scheduling and automation.")
    
    if peak_to_avg > 3.0:
        recommendations.append("Very high peak consumption. Consider energy storage solutions.")
    
    if high_percentage < 10:
        recommendations.append("Consistent consumption pattern. Maintain current efficiency practices.")
    
    return recommendations
