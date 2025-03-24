# METADATA model_inference.py] - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    # Description: Performs inference using a saved model and test, validation, and supervalidation datasets.

    # Developed By: 
        # Name: Mohini T and Vansh R
        # Role: Developer
        # Code ownership rights: Mohini T and Vansh R

# CODE - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    # Dependencies:
        # Python 3.11.5
        # Libraries:
            # Pandas 2.2.2
            # SQLAlchemy 2.0.31

# Importing the necessary libraries
import joblib # For loading the model
import pandas as pd # For data manipulation
from sqlalchemy import create_engine # For connecting to MySQL database
from sqlalchemy.exc import SQLAlchemyError # For handling database errors
from sklearn.preprocessing import LabelEncoder # For encoding and decoding labels

# Importing the necessary functions from .py helper file to evaluate the model
from evaluate import evaluate_model

def load_data(db_url, table_name):
    try:
        # Create database engine
        engine = create_engine(db_url)
        # Load data from the specified table
        data = pd.read_sql(f"SELECT * FROM {table_name}", con=engine)
        return data
    except SQLAlchemyError as e:
        # Print error message if data loading fails
        print(f"Error loading data from {table_name}: {e}")
        return None

def model_inference(model_path, mysql_db_url, mysql_db_name):
    # Load the model
    model = joblib.load(model_path)
    
    # Set the db_url
    db_url = f"{mysql_db_url}/{mysql_db_name}"

    # Load label encoder to decode labels
    label_encoder = LabelEncoder()
    
    # Fetching a sample of training data to fit the label encoder
    engine = create_engine(db_url)
    train_data = pd.read_sql('SELECT * FROM train_data', con=engine)
    label_encoder.fit(train_data['weather'])

    # Load the datasets
    datasets = {
        'Testing': 'test_data',
        'Validation': 'val_data',
        'Supervalidation': 'superval_data'
    }

    metrics = {}
    for dataset_name, table_name in datasets.items():
        # Print dataset name
        print(f"Evaluating on {dataset_name} Data:")
        # Load data from the specified table
        data = load_data(db_url, table_name)
        if data is not None:
            # Drop 'day' column if it exists
            if 'day' in data.columns:
                data = data.drop(columns=['day'])
            # Fit label encoder with 'weather' column
            if 'weather' in data.columns:
                label_encoder.fit(data['weather'])
            # Evaluate the model on the data
            accuracy, report, cross_val_scores = evaluate_model(model, data, label_encoder)
            # Store the metrics for the dataset
            metrics[dataset_name] = {
                'accuracy': accuracy,
                'report': report,
                'cross_val_scores': cross_val_scores
            }
    return metrics