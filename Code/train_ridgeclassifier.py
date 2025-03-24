# METADATA [train_naivebayes.py] - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    # Description: Trains a Ridge Classifier model on the training data and evaluates it using the evaluate_model function.

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

# Importing the necessary libraries
from sklearn.linear_model import RidgeClassifier # For training the Ridge Classifier model
import joblib # For saving the model
import pandas as pd # For data manipulation
from sklearn.preprocessing import LabelEncoder # For encoding and decoding labels

# Importing the necessary functions from .py helper file to load the training data and evaluate the model
from load import load_train_data
from evaluate import evaluate_model

def train_model(model_path, mysql_db_url, mysql_db_name):
    # Load the preprocessed training data from the database
    X_train, y_train = load_train_data(mysql_db_url, mysql_db_name)

    # Train the Ridge Classifier model
    model = RidgeClassifier()
    model.fit(X_train, y_train)

    # Save the model
    joblib.dump(model, model_path)

    # Load label encoder to decode labels
    label_encoder = LabelEncoder()
    label_encoder.fit(y_train)
    
    # Evaluate the model on the training dataset
    print("Evaluating Ridge Classifier on Training Data:")
    train_data = pd.concat([X_train, y_train], axis=1)
    return evaluate_model(model, train_data, label_encoder)