import logging
import pickle
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline


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

def build_random_forest_pipeline_with_encoding():
    categorical_columns = ['StateHoliday', 'StoreType', 'Assortment', 'PromoInterval'] 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the column transformer to apply one-hot encoding to categorical columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)
        ],
        remainder='passthrough'  # Leave other columns untouched
    )
    # Define the pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    return pipeline

def evaluate_model(y_true, y_pred):
    try:
        mae = mean_absolute_error(y_true, y_pred)
        logging.info(f"Model evaluation - MAE: {mae}")
        return mae
    except Exception as e:
        logging.error(f"Error during model evaluation: {e}")
def plot_feature_importance(model, feature_names):
    try:
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(10, 6))
        plt.title("Feature importances")
        plt.bar(range(len(importances)), importances[indices], align="center")
        plt.xticks(range(len(importances)), feature_names[indices], rotation=90)
        plt.tight_layout()
        plt.show()
        
        logging.info("Feature importance plot created.")
    except Exception as e:
        logging.error(f"Error during feature importance plotting: {e}")

def save_model(model, filename=None):
    try:
        if not filename:
            timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            filename = f"model-{timestamp}.pkl"
            
        with open(filename, 'wb') as file:
            pickle.dump(model, file)
        
        logging.info(f"Model saved as {filename}.")
    except Exception as e:
        logging.error(f"Error during model saving: {e}")
def prepare_time_series_data(df, target_col, window_size):
    X, y = [], []
    for i in range(len(df) - window_size):
        X.append(df[i:(i + window_size)])
        y.append(df[target_col][i + window_size])
        
    logging.info("Time series data prepared.")
    return np.array(X), np.array(y)


