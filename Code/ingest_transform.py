# METADATA [ingest_transform.py] - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    # Description: This code ingests raw data from a CSV file, preprocesses it, splits it into
    # datasets, and stores it in a MySQL database.

    # Developed By: 
        # Name: Mohini T and Vansh R
        # Role: Developer
        # Code ownership rights: Mohini T and Vansh R

# CODE - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    # Dependencies:
        # Python 3.11.5
        # Libraries:
            # Pandas 2.2.2
            # Scikit-learn 1.5.0
            # mysql-connector-python 8.4.0
            # SQLAlchemy 2.0.31

# Importing the necessary libraries
import pandas as pd # For data manipulation
from sklearn.model_selection import train_test_split # For splitting the data
from sqlalchemy import create_engine, text # For connecting to MySQL database
from sklearn.preprocessing import LabelEncoder, StandardScaler # For preprocessing the data

def create_database_if_not_exists(engine):
    with engine.connect() as conn:
        conn.execute(text("CREATE DATABASE IF NOT EXISTS preprod_db"))
        conn.execute(text("USE preprod_db"))

def ingest_and_transform(data_path, mysql_db_url, mysql_db_name):
    # Load the raw data from CSV
    df = pd.read_csv(data_path)
    
    # Preprocess data: Extract 'day' feature as datetime type
    df['day'] = pd.to_datetime(df['day'], format='%d/%m/%Y')
    
    # Encode the 'weather' column
    label_encoder = LabelEncoder()
    df['weather'] = label_encoder.fit_transform(df['weather'])
    
    # Print the mapping of original labels to encoded labels
    print("Encoded weather classes:", label_encoder.classes_)
    
    # Scale the numerical columns
    scaler = StandardScaler()
    df[['temperature', 'humidity', 'wind_speed']] = scaler.fit_transform(df[['temperature', 'humidity', 'wind_speed']])
    
    # Split the data
    train_data, temp_data = train_test_split(df, train_size=600, random_state=42)
    test_data, temp_data = train_test_split(temp_data, train_size=150, random_state=42)
    val_data, superval_data = train_test_split(temp_data, train_size=150, random_state=42)

    # Connect to MySQL database using SQLAlchemy
    engine = create_engine(mysql_db_url)
    create_database_if_not_exists(engine)
    
    engine = create_engine(f"{mysql_db_url}/{mysql_db_name}")

    # Store the split data to the database
    train_data.to_sql('train_data', con=engine, if_exists='replace', index=False)
    test_data.to_sql('test_data', con=engine, if_exists='replace', index=False)
    val_data.to_sql('val_data', con=engine, if_exists='replace', index=False)
    superval_data.to_sql('superval_data', con=engine, if_exists='replace', index=False)

if __name__ == "__main__":
    ingest_and_transform("Data/Master/Mock_Data.csv")