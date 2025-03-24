# METADATA [train_naivebayes.py] - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    # Description: Trains a Gaussian Naive Bayes model on the training data and evaluates it using the evaluate_model function.

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
from sklearn.naive_bayes import GaussianNB # For training the Naive Bayes model
import joblib # For saving the model
import pandas as pd # For data manipulation
from sklearn.preprocessing import LabelEncoder # For encoding and decoding labels

# Importing the necessary functions from .py helper file to load the training data and evaluate the model
from load import load_train_data # For loading the training data
from evaluate import evaluate_model # For evaluating the model

def train_model(model_path, mysql_db_url, mysql_db_name):
    # Load the preprocessed training data from the database
    X_train, y_train = load_train_data(mysql_db_url, mysql_db_name)

    # Train the Gaussian Naive Bayes model
    model = GaussianNB()
    model.fit(X_train, y_train)
    
    # Save the model
    joblib.dump(model, model_path)

    # Load label encoder to decode labels
    label_encoder = LabelEncoder()
    label_encoder.fit(y_train)
    
    # Evaluate the model on the training dataset
    print("Evaluating Gaussian Naive Bayes on Training Data:")
    train_data = pd.concat([X_train, y_train], axis=1)
    return evaluate_model(model, train_data, label_encoder)

if __name__ == "__main__":
    train_model('naive_bayes.pkl')