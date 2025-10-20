"""
Main entry point for the Energy Consumption Forecasting Application
Handles initialization and startup
"""

import os
import sys
from app import app
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_requirements():
    """Check if all required files exist"""
    required_files = ['lstm_power_model.h5', 'scaler.pkl', 'seq_length.pkl']
    missing_files = []
    
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        logger.warning(f"Missing model files: {missing_files}")
        logger.warning("Please run the Jupyter notebook to train and save the model first.")
        return False
    
    logger.info("✓ All required model files found")
    return True


def main():
    """Main application entry point"""
    logger.info("=" * 60)
    logger.info("Energy Consumption Forecasting Application")
    logger.info("LSTM-based Smart Energy Optimization System")
    logger.info("=" * 60)
    
    # Check requirements
    if not check_requirements():
        logger.error("Cannot start application - missing model files")
        sys.exit(1)
    
    logger.info("Starting Flask application...")
    logger.info("Server running at http://localhost:5000")
    logger.info("Press CTRL+C to stop the server")
    
    try:
        app.run(
            debug=False,
            host='0.0.0.0',
            port=5000,
            use_reloader=False
        )
    except KeyboardInterrupt:
        logger.info("Application stopped by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Application error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
