import pandas as pd
import joblib
from datetime import datetime
import logging
import pickle
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer



def datetime_features_added(df, date_column):
    try:
        df['DayOfWeek'] = df[date_column].dt.weekday
        df['IsWeekend'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
        df['IsBeginningOfMonth'] = df[date_column].dt.day <= 10
        df['IsMidMonth'] = (df[date_column].dt.day > 10) & (df[date_column].dt.day <= 20)
        df['IsEndOfMonth'] = df[date_column].dt.day > 20
        
        logging.info("Datetime features extracted.")
        return df
    except Exception as e:
        logging.error(f"Error during datetime feature extraction: {e}")

def scale_features(df):
    try:
        scaler = StandardScaler()
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        
        logging.info("Data scaled successfully.")
        return df
    except Exception as e:
        logging.error(f"Error during scaling: {e}")

# Function to build a Random Forest model pipeline with encoding
def build_random_forest_pipeline_with_encoding(X, y):
    # Convert non-numeric columns to strings to ensure OneHotEncoder can process them
    non_numeric_columns = X.select_dtypes(include=['object', 'category']).columns
    X[non_numeric_columns] = X[non_numeric_columns].astype(str)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define categorical and numeric columns
    categorical_cols = ['StateHoliday', 'StoreType', 'Assortment', 'PromoInterval']  # Adjust based on your dataset
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Preprocessing for categorical and numerical data
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    numeric_transformer = SimpleImputer(strategy='mean')  # Example numeric processing
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )
    
    # Define pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('scaler', StandardScaler()),  # Apply scaling after encoding
        ('model', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    # Fit the pipeline on the training data
    pipeline.fit(X_train, y_train)
    
    # Log training completion
    logging.info("Random Forest model with encoding trained successfully.")
    
    return pipeline, X_test, y_test

def evaluate_model(y_true, y_pred):
    try:
        mae = mean_absolute_error(y_true, y_pred)
        logging.info(f"Model evaluation - MAE: {mae}")
        return mae
    except Exception as e:
        logging.error(f"Error during model evaluation: {e}")        
def get_feature_names_after_preprocessing(preprocessor, X):
    # Check if preprocessor exists and retrieve transformed feature names
    try:
        categorical_features = preprocessor.transformers_[0][1].get_feature_names_out()
        # Retrieve numeric feature names (unchanged after scaling or imputation)
        numeric_features = X.select_dtypes(exclude=['object']).columns
        # Combine both categorical and numeric feature names
        all_features = np.concatenate([categorical_features, numeric_features])
        return all_features
    except AttributeError as e:
        logging.error(f"Error extracting feature names after preprocessing: {e}")
        raise

def feature_importance(model, X):
    # Ensure that the model has a preprocessor
    try:
        preprocessor = model.named_steps['preprocessor']
    except KeyError:
        logging.error("Preprocessor not found in the model pipeline.")
        raise ValueError("Ensure the model has a preprocessing step.")
    
    # Get the feature names after the preprocessor step
    feature_names = get_feature_names_after_preprocessing(preprocessor, X)
    
    # Get feature importances from the trained model
    try:
        importances = model.named_steps['model'].feature_importances_
    except AttributeError:
        logging.error("Model does not have feature importances.")
        raise ValueError("The model must have a feature_importances_ attribute (e.g., tree-based models).")
    
    # Check if feature names and importances are of the same length
    if len(feature_names) != len(importances):
        logging.error(f"Length mismatch: feature_names({len(feature_names)}) and importances({len(importances)})")
        raise ValueError("Feature names and importances must have the same length.")
    
    # Create DataFrame for feature importance
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    logging.info("Feature importance calculated successfully.")
    return feature_importance_df
def calculate_confidence_intervals(model, X, alpha=0.05):
    preds = model.predict(X)
    std_error = np.std(preds)
    z_score = 1.96  # For a 95% confidence interval
    margin_of_error = z_score * (std_error / np.sqrt(len(preds)))
    
    lower_bound = preds - margin_of_error
    upper_bound = preds + margin_of_error
    
    logging.info("Confidence intervals calculated successfully.")
    return lower_bound, upper_bound

def serialize_model(model, timestamp):
    filename = f'model_{timestamp}.pkl'
    joblib.dump(model, filename)
    logging.info(f"Model serialized and saved as {filename}")


