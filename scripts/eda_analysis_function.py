import pandas as pd
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import logging


def setup_logger():
    logging.basicConfig(filename='analysis.log', 
                        format='%(asctime)s:%(levelname)s:%(message)s', 
                        level=logging.INFO)


def load_data(file_path):
    """
    Function to load dataset from a given CSV file path.
    """
    try:
        data = pd.read_csv(file_path)
        logging.info('Data loaded successfully.')
        return data
    except Exception as e:
        logging.error(f'Error loading data: {e}')
        return None

def clean_data(df):
    """
    Function to clean the dataset.
    - Handle missing values
    - Detect and handle outliers (only in numeric columns)
    """
    try:
        # Handling missing values by forward fill (or use other strategies as needed)
        df.fillna(method='ffill', inplace=True)
        logging.info('Missing values handled with forward fill.')

        # Select only numeric columns for outlier detection
        numeric_cols = df.select_dtypes(include=['number'])
        
        # Detect outliers using the IQR method
        Q1 = numeric_cols.quantile(0.25)
        Q3 = numeric_cols.quantile(0.75)
        IQR = Q3 - Q1

        # Filtering out outliers
        df_cleaned = df[~((numeric_cols < (Q1 - 1.5 * IQR)) | (numeric_cols > (Q3 + 1.5 * IQR))).any(axis=1)]
        logging.info('Outliers removed using IQR method from numeric columns.')

        return df_cleaned
    except Exception as e:
        logging.error(f'Error during data cleaning: {e}')
        return None
def analyze_promo_distribution(df):
    """
    Analyze promotion distribution between training and test sets.
    """
    try:
        # Check if 'Promo2' exists in the DataFrame
        if 'Promo2' not in df.columns:
            logging.error('Promo2 column not found in DataFrame.')
            return
        
        plt.figure(figsize=(10, 6))
        sns.countplot(x='Promo2', data=df)
        plt.title('Promo2 Distribution')
        plt.show()
        logging.info('Promo2 distribution analysis done.')
    except Exception as e:
        logging.error(f'Error during promo distribution analysis: {e}')
def analyze_competition_distance(df):
    """
    Analyze how competition distance affects stores.
    """
    try:
        plt.figure(figsize=(10, 6))
        sns.histplot(df['CompetitionDistance'], kde=True)
        plt.title('Competition Distance Distribution')
        plt.show()
        logging.info('Competition distance analysis done.')
    except Exception as e:
        logging.error(f'Error during competition distance analysis: {e}')

def analyze_store_type_distribution(df):
    """
    Analyze distribution of Store Types.
    """
    try:
        plt.figure(figsize=(10, 6))
        sns.countplot(x='StoreType', data=df)
        plt.title('Store Type Distribution')
        plt.show()
        logging.info('Store Type distribution analysis done.')
    except Exception as e:
        logging.error(f'Error during StoreType distribution analysis: {e}')

def analyze_assortment_distribution(df):
    
    """
    Analyze distribution of Assortments.
    """
    try:
        plt.figure(figsize=(10, 6))
        sns.countplot(x='Assortment', data=df)
        plt.title('Assortment Type Distribution')
        plt.show()
        logging.info('Assortment distribution analysis done.')
    except Exception as e:
        logging.error(f'Error during Assortment distribution analysis: {e}')