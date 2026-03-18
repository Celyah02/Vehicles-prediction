# Vehicle Analytics System

A Django web application integrating machine learning models for comprehensive vehicle sales analysis and prediction.

## Overview

This system provides intelligent vehicle analytics through three core ML models:
- **Price Prediction**: Forecast vehicle selling prices using regression analysis
- **Income Classification**: Categorize customer income levels
- **Customer Segmentation**: Group clients using K-Means clustering

## Technology Stack

- **Backend**: Django 4.x
- **Machine Learning**: scikit-learn, pandas, numpy
- **Data Visualization**: matplotlib, seaborn, plotly
- **Frontend**: Bootstrap 5, HTML templates
- **Model Persistence**: joblib

## Project Structure

```
vehicles-prediction/
├── manage.py
├── config/                    # Django configuration
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
├── predictor/                 # Main application
│   ├── views.py
│   ├── urls.py
│   ├── models.py
│   └── templates/predictor/
│       ├── index.html
│       ├── regression_analysis.html
│       ├── classification_analysis.html
│       └── clustering_analysis.html
├── model_generators/          # ML training scripts
│   ├── regression/
│   ├── classification/
│   └── clustering/
├── dummy-data/
│   └── vehicles_ml_dataset.csv
├── requirements.txt
└── *.pkl                     # Trained models
```

## Installation

### Prerequisites
- Python 3.9+
- pip package manager
- Virtual environment (recommended)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Goal651/vehicles-prediction.git
   cd vehicles-prediction
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   # Windows: venv\Scripts\activate
   # Mac/Linux: source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare dataset**
   - Place `vehicles_ml_dataset.csv` in `dummy-data/`
   - Required columns: `year`, `kilometers_driven`, `seating_capacity`, `estimated_income`, `selling_price`, `income_level`, `client_name`

5. **Train ML models**
   ```bash
   python model_generators/regression/train_regression.py
   python model_generators/classification/train_classifier.py
   python model_generators/clustering/train_cluster.py
   ```

6. **Run the application**
   ```bash
   python manage.py runserver
   ```

Access the application at `http://127.0.0.1:8000`

## Machine Learning Models

### Price Prediction
- **Algorithm**: Random Forest Regressor
- **Features**: Vehicle year, kilometers driven, seating capacity, estimated income
- **Target**: Selling price
- **Evaluation**: R² Score

### Income Classification
- **Algorithm**: Random Forest Classifier
- **Features**: Vehicle year, kilometers driven, seating capacity, estimated income
- **Target**: Income level category
- **Evaluation**: Accuracy Score

### Customer Segmentation
- **Algorithm**: K-Means Clustering
- **Features**: Estimated income, selling price
- **Clusters**: 3 segments (Economy, Standard, Premium)
- **Evaluation**: Silhouette Score

## Usage

### Application Interface
1. **Data Exploration**: View dataset statistics and exploratory analysis
2. **Regression Analysis**: Input vehicle specifications for price predictions
3. **Classification Analysis**: Predict income categories based on vehicle data
4. **Clustering Analysis**: Customer segmentation analysis

### Making Predictions
Navigate to any analysis page and provide:
- Model Year
- Kilometers Driven
- Number of Seats
- Owner Income

## Development Tasks

### Enhancement Opportunities
- **Geographic Visualization**: Integrate interactive Rwanda district mapping
- **Model Optimization**: Improve clustering silhouette score above 0.9
- **Advanced Metrics**: Calculate coefficient of variation for clustering evaluation

## Troubleshooting

### Common Issues

**File not found errors**
- Run commands from project root directory
- Verify file paths match actual structure

**Dataset not loading**
- Confirm CSV exists in `dummy-data/`
- Check column names match training scripts

**Model files missing**
- Execute training scripts to generate `.pkl` files
- Ensure models are in project root

**Django import errors**
- Activate virtual environment
- Verify package installation with `pip list`

## Dependencies

```
django>=4.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
joblib>=1.1.0
plotly>=5.3.0
numpy>=1.21.0
```

## License

Educational project for Django/ML learning curriculum.

## Contributing

1. Fork repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Open Pull Request

We would like to work with anyone interested
