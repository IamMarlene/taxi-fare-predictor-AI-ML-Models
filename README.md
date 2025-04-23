# Taxi Fare Predictor using AI/ML Models

A Databricks-based ML pipeline using PySpark for taxi fare prediction. Implements distributed data processing, feature engineering with StringIndexer and VectorAssembler, and compares RandomForest, GBT and LinearRegression models. Features hyperparameter tuning, residual analysis, and SQL prediction UDFs for production deployment.

## Project Overview

This project leverages Databricks and Apache Spark (PySpark) to build a machine learning pipeline that predicts taxi fares based on trip data. The model considers features such as trip distance, passenger count, time of day, and other relevant factors to provide accurate fare estimates.

## Key Features

- **Distributed Data Processing**: Utilizes Spark's parallel processing capabilities for large-scale data handling
- **Robust Data Cleaning**: Handles missing values, outliers, and data type issues
- **Feature Engineering**: Creates derived features including:
  - Time-based features (hour of day, weekday/weekend)
  - Trip speed calculations
  - Fare efficiency metrics
- **Multiple ML Models**:
  - Random Forest Regression
  - Gradient Boosted Trees
  - Linear Regression
- **Model Evaluation**: Comprehensive performance analysis with RMSE metrics
- **Hyperparameter Tuning**: CrossValidator implementation for optimal model selection
- **Model Interpretability**: Feature importance analysis and residual plots
- **Production-Ready Components**: SQL UDFs for seamless integration with existing data pipelines

## Technical Implementation

```
TaxiTripAnalysis/
├── notebook/
│   ├── data_preparation      # Data cleaning and preprocessing
│   ├── exploratory_analysis  # Visualizations and initial insights
│   ├── model_training        # ML pipeline implementation
│   └── model_deployment      # SQL UDFs and deployment code
└── README.md
```

### Environment

- Databricks Runtime: 10.4 ML (or your specific version)
- Python: 3.8+
- PySpark: 3.2+

### Core Libraries

- PySpark ML for distributed machine learning
- PySpark SQL for data manipulation
- Matplotlib and Seaborn for visualizations

## Getting Started

1. Import the notebooks into your Databricks workspace
2. Mount or connect to your taxi trip dataset
3. Run the notebooks in sequence to:
   - Prepare and clean your data
   - Analyze patterns and trends
   - Train and evaluate prediction models
   - Deploy prediction functions

## Model Performance

| Model | RMSE | Features Used |
|-------|------|---------------|
| Random Forest | 9.43 | All features |
| GBT | 9.21 | All features |
| Linear Regression | 10.87 | All features |
| Simple Linear | 12.34 | Distance, time only |

## Business Insights

The project reveals several valuable insights:
- Peak profitability hours for taxi operations
- Factors that most significantly influence fare amounts
- Optimal pricing strategies based on trip characteristics
- Passenger behavior patterns affecting tipping

## Future Improvements

- Integration with geospatial data for location-based fare prediction
- Real-time prediction API implementation
- Time series forecasting for demand-based pricing
- A/B testing framework for model comparison

## Contact

For questions or collaboration, please open an issue or reach out to [your contact info].

