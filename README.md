# Rossmann Store Sales Prediction

## Project Overview

This project focuses on building machine learning models to predict daily store sales for the next six weeks. It covers data preprocessing, model training, feature importance analysis, confidence interval estimation for predictions, and model serialization. The models are serialized daily to track predictions and future comparisons.

---

## Table of Contents

1. [Preprocessing](#preprocessing)
2. [Model Training](#model-training)
3. [Feature Importance](#feature-importance)
4. [Confidence Interval for Predictions](#confidence-interval-for-predictions)
5. [Model Serialization](#model-serialization)
6. [How to Run the Project](#how-to-run-the-project)
7. [Dependencies](#dependencies)
8. [Project Structure](#project-structure)

---

## Preprocessing

### Description:
Data preprocessing involves handling missing values, transforming categorical data, and scaling numerical features. These steps are crucial for improving model performance.

### Key Steps:

- **Handling Missing Data**: Imputed using the `SimpleImputer` from `scikit-learn`.
- **Scaling**: Numerical features are scaled using `StandardScaler`.
- **Encoding**: Categorical features are encoded using `OneHotEncoder`.
## Model Training
Description:
The project uses a Random Forest Regressor and LSTM models for time series forecasting. The Random Forest model was selected for its robustness in handling feature importance and model interpretability, while LSTM was chosen for handling sequential data effectively.
Evaluation:
The model's performance is measured using Mean Absolute Error (MAE) and Mean Squared Error (MSE).
## Feature Importance
Description:
Feature importance helps identify which features contribute the most to model predictions. This information can be used for model optimization and interpretability.
## Confidence Interval for Predictions
Description:
Estimating a confidence interval for the predictions provides insights into the range in which future predictions are likely to fall. A 95% confidence interval is calculated for each prediction.
## Model Serialization
Description:
To track model predictions over time, models are serialized with a timestamp. This allows us to manage and track multiple versions of the model for each day's prediction.

## How to Run the Project
Clone the repository:
git clone https://github.com/your-repo-url.git
cd your-repo-directory
Install dependencies:
pip install -r requirements.txt
##  Dependencies
The project requires the following libraries:

Python 3.8+
pandas
numpy
scikit-learn
joblib
tensorflow (for LSTM model)