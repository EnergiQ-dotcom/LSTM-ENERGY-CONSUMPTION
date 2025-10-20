# Energy Consumption Forecasting - LSTM Smart Energy Optimization

An AI-powered application for forecasting household electricity consumption and optimizing energy usage using Long Short-Term Memory (LSTM) neural networks.

## Project Overview

This project addresses the energy optimization challenge by building an intelligent system that:
- **Forecasts** future electricity consumption patterns
- **Detects** anomalies in energy usage
- **Provides** actionable recommendations for energy optimization
- **Supports** smart scheduling and demand response strategies

### Key Features

- **LSTM-based Forecasting**: Deep learning model trained on household power consumption data
- **Multi-step Predictions**: Generate forecasts for 1-168 hours ahead
- **Energy Analysis**: Identify peak consumption periods and optimization opportunities
- **Web Interface**: User-friendly dashboard for predictions and analysis
- **REST API**: Easy integration with other systems
- **Real-time Insights**: Instant energy consumption analysis and recommendations

## Dataset

**Household Power Consumption Dataset** (UCI Machine Learning Repository)
- **Records**: 2,075,259 measurements
- **Time Period**: December 2006 - November 2010
- **Frequency**: 1-minute intervals
- **Features**: 
  - Global active power
  - Global reactive power
  - Voltage
  - Global intensity
  - Sub-metering data

## Project Structure

\`\`\`
.
├── lstm_forecasting.ipynb          # Jupyter notebook for model training
├── app.py                          # Flask web application
├── main.py                         # Application entry point
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── templates/
│   └── index.html                 # Web interface
├── lstm_power_model.h5            # Trained LSTM model (generated)
├── scaler.pkl                     # Data scaler (generated)
└── seq_length.pkl                 # Sequence length config (generated)
\`\`\`

## Installation & Setup

### Prerequisites
- Python 3.8+
- pip or conda
- 4GB+ RAM (for model training)
- GPU recommended (CUDA-enabled for faster training)

### Step 1: Clone or Download the Project

\`\`\`bash
git clone <repository-url>
cd lstm-energy-forecasting
\`\`\`

### Step 2: Create Virtual Environment

\`\`\`bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n energy-forecast python=3.9
conda activate energy-forecast
\`\`\`

### Step 3: Install Dependencies

\`\`\`bash
pip install -r requirements.txt
\`\`\`

### Step 4: Train the Model (Google Colab)

1. Open `lstm_forecasting.ipynb` in Google Colab
2. Run all cells to:
   - Download the household power consumption dataset
   - Preprocess and normalize data
   - Build and train the LSTM model
   - Generate predictions and analysis
3. Download the generated model files:
   - `lstm_power_model.h5`
   - `scaler.pkl`
   - `seq_length.pkl`

### Step 5: Run the Web Application

\`\`\`bash
python main.py
\`\`\`

The application will start at `http://localhost:5000`

## Usage

### Web Interface

1. **Single Prediction**
   - Enter recent power consumption values (comma-separated)
   - Click "Predict Next Value"
   - Get the predicted power consumption for the next time step

2. **Multi-Step Forecast**
   - Enter recent power data
   - Specify forecast horizon (1-168 hours)
   - Generate forecast and visualize results

3. **Energy Analysis**
   - Input consumption data
   - Receive detailed metrics and optimization recommendations
   - Identify peak consumption periods

### API Endpoints

#### 1. Single Prediction
\`\`\`bash
POST /api/predict
Content-Type: application/json

{
  "recent_data": [1.5, 1.6, 1.7, 1.8, 2.0, ...]
}

Response:
{
  "prediction": 2.1,
  "unit": "kW",
  "timestamp": "2024-01-15T10:30:00",
  "status": "success"
}
\`\`\`

#### 2. Multi-Step Forecast
\`\`\`bash
POST /api/forecast
Content-Type: application/json

{
  "recent_data": [1.5, 1.6, 1.7, ...],
  "steps": 24
}

Response:
{
  "forecast": [2.1, 2.2, 2.3, ...],
  "timestamps": ["2024-01-15T11:00:00", ...],
  "unit": "kW",
  "steps": 24,
  "status": "success"
}
\`\`\`

#### 3. Energy Analysis
\`\`\`bash
POST /api/analyze
Content-Type: application/json

{
  "consumption_data": [1.5, 1.6, 1.7, ...]
}

Response:
{
  "metrics": {
    "average": 1.8,
    "peak": 3.5,
    "minimum": 0.5,
    "std_dev": 0.8,
    "peak_to_avg_ratio": 1.94
  },
  "high_consumption_periods": {
    "count": 150,
    "percentage": 25.5,
    "threshold": 2.7
  },
  "recommendations": [
    "High peak-to-average ratio detected...",
    "Frequent high consumption periods..."
  ],
  "status": "success"
}
\`\`\`

#### 4. Health Check
\`\`\`bash
GET /api/health

Response:
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2024-01-15T10:30:00"
}
\`\`\`

## Model Architecture

### LSTM Network
\`\`\`
Input Layer (60 time steps, 1 feature)
    ↓
LSTM Layer 1 (50 units, ReLU activation)
    ↓
Dropout (0.2)
    ↓
LSTM Layer 2 (50 units, ReLU activation)
    ↓
Dropout (0.2)
    ↓
LSTM Layer 3 (25 units, ReLU activation)
    ↓
Dropout (0.2)
    ↓
Dense Layer (1 unit, Linear activation)
    ↓
Output (Predicted Power Consumption)
\`\`\`

### Training Configuration
- **Optimizer**: Adam (learning_rate=0.001)
- **Loss Function**: Mean Squared Error (MSE)
- **Metrics**: Mean Absolute Error (MAE)
- **Epochs**: 20
- **Batch Size**: 32
- **Validation Split**: 15%
- **Test Split**: 15%

## Performance Metrics

### Test Set Results
- **RMSE**: ~0.5 kW
- **MAE**: ~0.3 kW
- **R² Score**: ~0.92

### Model Capabilities
- Captures temporal dependencies in power consumption
- Handles seasonal patterns and trends
- Provides reliable short-term forecasts (1-24 hours)
- Identifies consumption anomalies

## Energy Optimization Insights

### Key Findings
1. **Peak Hours**: Typically 18:00-22:00 (evening peak)
2. **Off-Peak Hours**: 02:00-06:00 (night minimum)
3. **Variability**: High consumption variability suggests optimization opportunities
4. **Patterns**: Clear daily and weekly patterns in consumption

### Recommendations
1. **Load Shifting**: Move energy-intensive tasks to off-peak hours
2. **Demand Response**: Implement automated response during peak periods
3. **Smart Scheduling**: Use forecasts for optimal task scheduling
4. **Anomaly Detection**: Monitor deviations from predicted patterns
5. **Energy Efficiency**: Target high-consumption periods for efficiency improvements

## Deployment

### Local Deployment
\`\`\`bash
python main.py
\`\`\`

### Production Deployment (Gunicorn)
\`\`\`bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
\`\`\`

### Docker Deployment
\`\`\`dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "main.py"]
\`\`\`

## Troubleshooting

### Model Files Not Found
- Ensure you've trained the model in Google Colab
- Download `lstm_power_model.h5`, `scaler.pkl`, and `seq_length.pkl`
- Place them in the project root directory

### Port Already in Use
\`\`\`bash
# Change port in main.py or use:
python -c "from app import app; app.run(port=5001)"
\`\`\`

### Memory Issues
- Reduce batch size in training
- Use GPU acceleration (CUDA)
- Process data in chunks

## Future Enhancements

1. **Multi-variate Forecasting**: Include temperature, humidity, etc.
2. **Ensemble Models**: Combine LSTM with other algorithms
3. **Real-time Data Integration**: Connect to smart meters
4. **Mobile App**: iOS/Android application
5. **Advanced Analytics**: Anomaly detection and clustering
6. **Optimization Engine**: Automated demand response
7. **Blockchain Integration**: Energy trading platform

## Technologies Used

- **Deep Learning**: TensorFlow, Keras
- **Data Processing**: NumPy, Pandas, Scikit-learn
- **Web Framework**: Flask
- **Frontend**: HTML5, CSS3, JavaScript, Chart.js
- **Visualization**: Matplotlib, Seaborn

## References

- UCI Machine Learning Repository: [Household Power Consumption Dataset](https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption)
- Hochreiter & Schmidhuber (1997): LSTM paper
- Keras Documentation: [LSTM Layer](https://keras.io/api/layers/recurrent_layers/lstm/)

## License

This project is open source and available under the MIT License.

## Contact & Support

For questions, issues, or suggestions:
- Open an issue on GitHub
- Contact: [your-email@example.com]

## Acknowledgments

- UCI Machine Learning Repository for the dataset
- TensorFlow and Keras teams
- Energy optimization research community

---

**Last Updated**: January 2024
**Version**: 1.0.0
