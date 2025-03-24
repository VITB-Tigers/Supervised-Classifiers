# DIY-SupervisedClassifiers

This is the Linear Model Algorithms Branch.

# Linear Model Supervised Classifiers

|                Naive Bayes            |              Ridge Classifier         |                
|-------------------------------------|-------------------------------------|
Naive Bayes is a probabilistic classification algorithm based on Bayes' Theorem, which assumes independence between features given the class. Despite this strong independence assumption, it performs well in practice for text classification, spam filtering, and sentiment analysis. By calculating the posterior probability for each class, Naive Bayes provides a straightforward and interpretable approach that is computationally efficient, handles large datasets well, and is robust to noisy features.          |Ridge Classifier is a linear classification method that extends logistic regression with L2 regularization to penalize large coefficients and prevent overfitting. By adding a penalty term proportional to the square of the magnitude of coefficients, Ridge Classifier stabilizes the model in the presence of multicollinearity and high-dimensional data, improving generalization on unseen data.

## Problem Definition

1. Assume you are a developer who is developing a weather app. You have all the weather data required and now want to implement a feature in the app that can predict the future weather for your customers. 
2. You take the necessary data you have from the your app or find a dataset online for (temperature, humidity, wind-speed) and the target value - weather.

## Data Definition

This is mock generated data just for learning purposes. It contains the Date, Temperature, Humidity, Wind Speed and Weather data. Since it is a mock data, it will not give the most accurate results, but it is perfect to understand the underlying concept of training a Naive Bayes Model.

> **Note:** The dataset consists of 1000 samples, leading to potential overfitting with a high training accuracy. This would not occur in real-life scenarios with larger and more varied datasets, providing a more realistic accuracy.

## Directory Structure

- **Code/**: Contains all the scripts for data ingestion, transformation, loading, evaluation, model training, inference, manual prediction, and API.
- **Data/**: Contains the raw data and processed database.

## Data Splitting

- **Training Samples**: 600
- **Testing Samples**: 150
- **Validation Samples**: 150
- **Supervalidation Samples**: 100

> **Note:** Since the data was in <ins>continuous form (numeric data)</ins> instead of categorical, we have used Gaussian Naive Bayes. If the data were categorical, then we would have used Categorical Naive Bayes.

## Program Flow

1. **Data Ingestion and Transformation:** Extract data from 'Data/Master', transform it, and store it in a MySQL database. [`ingest_transform.py`]
2. **Data Loading:** Load transformed data from the MySQL database. [`load.py`]
3. **Evaluation:** This code has the function to evaluate the performance of trained models on a given dataset. [`evalaute.py`]
4. **Model Training:** Train a Naive Bayes model [`train_naivebayes.py`] and a Ridge Classifier model [`train_ridgeclassifier.py`] using the training data.
5. **Model Evaluation:** Evaluate the models on testing, validation, and supervalidation datasets, and generate classification reports. [`model_inference.py`]
6. **Manual Prediction:** Takes user inputs for features; scales thrm; uses a saved model to predict the weather; decodes and displays the weather prediction. [`manual_prediction.py`]
7. **Web Application:** Streamlit app to provide a user-friendly GUI for predictions. [`app.py`]

## Steps to Run

1. Install the necessary packages: `pip install -r requirements.txt`
2. Run the Streamlit web application: `app.py`