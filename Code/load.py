# METADATA [load.py] - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    # Description: Loads the training, test, validation, and supervalidation datasets from the MySQL database.

    # Developed By: 
        # Name: Mohini T and Vansh R
        # Role: Developer
        # Code ownership rights: Mohini T and Vansh R

# CODE - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    # Dependencies:
        # Python 3.11.5
        # Libraries:
            # Pandas 2.2.2
            # mysql-connector-python 8.4.0
            # SQLAlchemy 2.0.31

# Importing the necessary libraries
import pandas as pd # For data manipulation
from sqlalchemy import create_engine # For connecting to MySQL database

def load_train_data(mysql_db_url, mysql_db_name):
    # Connect to the MySQL database
    db_url = f"{mysql_db_url}/{mysql_db_name}"
    engine = create_engine(db_url)
    
    # Load the training data from the 'train_data' table
    train_data = pd.read_sql('SELECT * FROM train_data', con=engine)
    
    # Separate the features (X_train) and the target variable (y_train)
    X_train = train_data.drop(columns=['weather', 'day'])
    y_train = train_data['weather']
    
    return X_train, y_train

def load_test_val_superval_data(mysql_db_url, mysql_db_name):
    # Connect to the MySQL database
    db_url = f"{mysql_db_url}/{mysql_db_name}"
    engine = create_engine(db_url)
    
    # Load the test, validation, and supervalidation data from their respective tables
    test_data = pd.read_sql('SELECT * FROM test_data', con=engine)
    val_data = pd.read_sql('SELECT * FROM val_data', con=engine)
    superval_data = pd.read_sql('SELECT * FROM superval_data', con=engine)
    
    # Drop the 'day' column from each dataset
    test_data = test_data.drop(columns=['day'])
    val_data = val_data.drop(columns=['day'])
    superval_data = superval_data.drop(columns=['day'])
    
    return test_data, val_data, superval_data