import pandas as pd
import logging
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_merge_data(train_path, store_path):
    """
    Load the train and store datasets, and merge them on 'Store' column.
    """
    try:
        # Load the datasets
        train_df = pd.read_csv(train_path)
        store_df = pd.read_csv(store_path)
        
        # Merge datasets on 'Store' column
        merged_df = pd.merge(train_df, store_df, how='left', on='Store')
        logging.info('Data merged successfully.')
        return merged_df
    except Exception as e:
        logging.error(f'Error during data loading and merging: {e}')
        return None

def clean_merged_data(df):
    """
    Clean the merged dataset.
    - Handle missing values
    - Detect and handle outliers
    """
    try:
        # Fill missing values in numeric columns
        df.fillna(method='ffill', inplace=True)
        # Detect outliers using IQR method in relevant columns
        numeric_cols = df.select_dtypes(include=['float64', 'int64'])
        Q1 = numeric_cols.quantile(0.25)
        Q3 = numeric_cols.quantile(0.75)
        IQR = Q3 - Q1
        df_cleaned = df[~((numeric_cols < (Q1 - 1.5 * IQR)) | (numeric_cols > (Q3 + 1.5 * IQR))).any(axis=1)]
        logging.info('Data cleaned successfully.')
        return df_cleaned
    except Exception as e:
        logging.error(f'Error during data cleaning: {e}')
        return None
def analyze_promo_sales(df):
    """
    Analyze the effect of Promo on sales.
    """
    try:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Promo', y='Sales', data=df)
        plt.title('Sales with and without Promo')
        plt.show()
        logging.info('Promo vs Sales analysis done.')
    except Exception as e:
        logging.error(f'Error during Promo vs Sales analysis: {e}')
def analyze_store_sales(df):
    """
    Analyze sales across different stores and store types.
    """
    try:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='StoreType', y='Sales', data=df)
        plt.title('Sales by Store Type')
        plt.show()
        logging.info('StoreType vs Sales analysis done.')
    except Exception as e:
        logging.error(f'Error during StoreType vs Sales analysis: {e}')
def analyze_competition_effect(df):
    """
    Analyze the effect of Competition Distance on sales.
    """
    try:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='CompetitionDistance', y='Sales', data=df)
        plt.title('Sales vs Competition Distance')
        plt.show()
        logging.info('Competition Distance vs Sales analysis done.')
    except Exception as e:
        logging.error(f'Error during Competition Distance vs Sales analysis: {e}')
def analyze_sales_over_time(df):
    """
    Analyze sales trends over time.
    """
    try:
        # Convert the 'Date' column to datetime if it exists in the data
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)

            plt.figure(figsize=(12, 6))
            df['Sales'].resample('M').sum().plot()
            plt.title('Sales Over Time (Monthly)')
            plt.xlabel('Date')
            plt.ylabel('Total Sales')
            plt.show()

            logging.info('Sales over time analysis done.')
        else:
            logging.error('Date column not found for time analysis.')
    except Exception as e:
        logging.error(f'Error during sales over time analysis: {e}')

def analyze_sales_by_day_of_week(df):
    """
    Analyze sales distribution by day of the week.
    """
    try:
        # Assuming 'DayOfWeek' column exists in the dataset
        if 'DayOfWeek' in df.columns:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x='DayOfWeek', y='Sales', data=df)
            plt.title('Sales Distribution by Day of the Week')
            plt.xlabel('Day of the Week')
            plt.ylabel('Sales')
            plt.show()

            logging.info('Sales by day of the week analysis done.')
        else:
            logging.error('DayOfWeek column not found in the dataset.')
    except Exception as e:
        logging.error(f'Error during sales by day of the week analysis: {e}')
def analyze_sales_vs_promo_by_day(df):
    """
    Analyze sales vs promo by day of the week.
    """
    try:
        if 'Promo' in df.columns and 'DayOfWeek' in df.columns:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x='DayOfWeek', y='Sales', hue='Promo', data=df)
            plt.title('Sales with/without Promo by Day of the Week')
            plt.xlabel('Day of the Week')
            plt.ylabel('Sales')
            plt.show()

            logging.info('Sales vs Promo by day of the week analysis done.')
        else:
            logging.error('Promo or DayOfWeek column not found.')
    except Exception as e:
        logging.error(f'Error during sales vs promo by day analysis: {e}')

def analyze_sales_by_assortment(df):
    """
    Plot a bar chart to visualize the distribution of assortments.
    """
    try:
        assortment_counts = df['Assortment'].value_counts()
        plt.figure(figsize=(8, 6))
        assortment_counts.plot(kind='bar', color='lightgreen')
        plt.title('Assortment Distribution')
        plt.xlabel('Assortment Type')
        plt.ylabel('Count')
        plt.show()
        logging.info('Bar chart for Assortment distribution done.')
    except Exception as e:
        logging.error(f'Error during Assortment bar chart: {e}')

def analyze_sales_vs_competition_open(df):
    """
    Analyze how competition opening date affects sales.
    """
    try:
        if 'CompetitionOpenSinceYear' in df.columns and 'CompetitionOpenSinceMonth' in df.columns:
            df['CompetitionOpen'] = pd.to_datetime(df['CompetitionOpenSinceYear'].astype('int').astype(str) + '-' + df['CompetitionOpenSinceMonth'].astype('int').astype(str), errors='coerce')

            plt.figure(figsize=(10, 6))
            sns.scatterplot(x='CompetitionOpen', y='Sales', data=df)
            plt.title('Sales vs Competitor Open Date')
            plt.xlabel('Competitor Open Date')
            plt.ylabel('Sales')
            plt.xticks(rotation=45)
            plt.show()

            logging.info('Sales vs competitor open date analysis done.')
        else:
            logging.error('CompetitionOpenSinceYear or CompetitionOpenSinceMonth columns not found.')
    except Exception as e:
        logging.error(f'Error during sales vs competition open analysis: {e}')

def setup_logger():
    logging.basicConfig(filename='analysis.log', 
                        format='%(asctime)s:%(levelname)s:%(message)s', 
                        level=logging.INFO)