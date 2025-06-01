# Air Quality Prediction Project

This repository contains the source code and automation setup for predicting air quality using machine learning models. The dataset is sourced from the Kaggle Air Quality dataset and includes pollutant levels, meteorological data, and engineered features.

## 📁 Project Structure

```
.
├── AirQualityData.csv            # Dataset file
├── models.py                     # Main Python script for training, evaluation, and plotting
├── requirements.txt              # Python dependencies
├── outputs/                      # Automatically created output folder (models, plots, metrics)
└── .github/
    └── workflows/
        └── main.yaml             # GitHub Actions workflow for automation
```

## 🧠 Models Implemented

We trained and evaluated four different models for predicting the Air Quality Index:

- **Linear Regression**
- **Random Forest Regressor**
- **K-Nearest Neighbors (KNN)**
- **Feedforward Neural Network (Keras/TensorFlow)**

These models were selected to compare a mix of traditional regressors and deep learning on structured data.

## 📊 Evaluation Results

| Model           | R² Score      | RMSE         |
|----------------|---------------|--------------|
| LinearRegression | -0.0125       | 144.99       |
| RandomForest     | -0.0210       | 145.60       |
| KNN              | -0.1892       | 157.14       |
| NeuralNetwork    | -0.0406       | 146.99       |

Although none of the models achieved positive R² scores (indicating poor fit), Linear Regression performed slightly better with the lowest RMSE. Further data cleaning, feature selection, or alternative modeling approaches may improve results.

## ⚙️ Model Automation with GitHub Actions

A CI/CD workflow is integrated using GitHub Actions. On every push:
- The repository installs required dependencies
- Runs the `models.py` script
- Generates predictions, model files, plots, and evaluation metrics
- Saves results to the `outputs/` folder

## 📈 Outputs

All output files are saved in the `outputs/` directory:
- `model_metrics.csv`: Model performance summary
- `correlation_heatmap.png`: Feature correlation matrix
- `daily_pollutants.png`: Daily trend visualization
- `.pkl` and `.h5` files: Saved models and scaler

## 🧪 How to Run Locally

```bash
git clone https://github.com/yourusername/air-quality-prediction.git
cd air-quality-prediction
pip install -r requirements.txt
python models.py
```

## ✅ Requirements

- Python 3.8+
- Packages listed in `requirements.txt` (includes TensorFlow, pandas, seaborn, scikit-learn, etc.)

## 📌 Next Steps

- Add cross-validation
- Explore other ML models like XGBoost or SVR
- Build a dashboard for real-time predictions
- Use external datasets for enrichment

## 📄 License

This project is part of an academic assignment (MSDS 422 - Deep Learning) and is not intended for commercial use.
