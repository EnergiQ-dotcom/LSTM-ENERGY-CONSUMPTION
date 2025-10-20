"""
Configuration file for the Energy Consumption Forecasting Application
"""

import os
from datetime import timedelta

# Flask Configuration
class Config:
    """Base configuration"""
    DEBUG = False
    TESTING = False
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
    
    # Model Configuration
    MODEL_PATH = 'lstm_power_model.h5'
    SCALER_PATH = 'scaler.pkl'
    SEQ_LENGTH_PATH = 'seq_length.pkl'
    
    # Default sequence length
    DEFAULT_SEQ_LENGTH = 60
    
    # API Configuration
    MAX_FORECAST_STEPS = 168  # 7 days
    MIN_FORECAST_STEPS = 1
    DEFAULT_FORECAST_STEPS = 24
    
    # Data Configuration
    MIN_DATA_POINTS = 60
    MAX_DATA_POINTS = 10000
    
    # Server Configuration
    HOST = '0.0.0.0'
    PORT = 5000
    THREADED = True
    
    # Logging
    LOG_LEVEL = 'INFO'
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'


class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = False


class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False
    SECRET_KEY = os.environ.get('SECRET_KEY')
    if not SECRET_KEY:
        raise ValueError("SECRET_KEY environment variable must be set in production")


class TestingConfig(Config):
    """Testing configuration"""
    DEBUG = True
    TESTING = True
    MODEL_PATH = 'test_model.h5'


# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}


def get_config(env=None):
    """Get configuration based on environment"""
    if env is None:
        env = os.environ.get('FLASK_ENV', 'development')
    return config.get(env, config['default'])
