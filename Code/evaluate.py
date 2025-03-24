# METADATA [evaluate.py] - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    # Description: Evaluates a given model on the test, validation, and supervalidation
    # datasets from the database and prints the classification report.

    # Developed By: 
        # Name: Mohini T and Vansh R
        # Role: Developer
        # Code ownership rights: Mohini T and Vansh R

# CODE - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    # Dependencies:
        # Python 3.11.5
        # Libraries:
            # Scikit-learn 1.5.0

# Importing the necessary libraries
from sklearn.metrics import accuracy_score, classification_report # For evaluating the model
from sklearn.model_selection import cross_val_score # For cross-validation

def evaluate_model(model, data, label_encoder=None):
    # Exclude the 'day' column if it exists
    if 'day' in data.columns:
        data = data.drop(columns=['day'])
        
    # Prepare the input features
    X = data.drop(columns=['weather'])  
    # Prepare the true labels
    y_true = data['weather']   

    # Decode the labels if a label encoder is provided
    if label_encoder is not None:
        y_true = label_encoder.inverse_transform(y_true)
        
    # Predict the labels using the trained model
    y_pred = model.predict(X)   
    
    # Encode the predictions if a label encoder is provided
    if label_encoder is not None:
        y_pred = label_encoder.inverse_transform(y_pred)
        
    # Calculate performance metrics
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, zero_division=0)
    cross_val_scores = cross_val_score(model, X, y_true, cv=5)   
    return accuracy, report, cross_val_scores