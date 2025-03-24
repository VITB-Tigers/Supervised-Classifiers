# METADATA [app.py] - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    # Description: This code loads and preprocesses weather data from a CSV file, connects to a
    # MySQL database, creates a weather table, and ingests the data into the database.

    # Developed By: 
        # Name: Mohini T and Vansh R
        # Role: Developer
        # Code ownership rights: Mohini T and Vansh R

# CODE - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    # Dependencies:
        # Python 3.11.5
        # Stremlit 1.36.0

# Importing the necessary libraries
import streamlit as st # Used to create the web app

# # Importing the .py helper files
from ingest_transform import ingest_and_transform # Used to ingest the data from the Master into the database, preproces and split the data
from train_naivebayes import train_model as train_model_naive_bayes # Used to train the model
from train_ridgeclassifier import train_model as train_model_ridge # Used to train the model
from model_inference import model_inference # Used to get the test, validation and supervalidation metrics
from manual_prediction import manual_prediction # Used to predict the weather based on the user's input


# # Setting the page configuration
st.set_page_config(page_title="Weather Prediction", page_icon=":cash:", layout="centered")
st.markdown("<h1 style='text-align: center; color: white;'>Weather Prediction</h1>", unsafe_allow_html=True)
st.divider()

# Declaring session states(streamlit variables) for saving the path throughout page reloads
# This is how we declare session state variables in streamlit.

if "master_data_path" not in st.session_state:
    st.session_state.master_data_path = "Data/Master/Mock_Data.csv"

if "naive_bayes_model_path" not in st.session_state:
    st.session_state.naive_bayes_model_path = "naive_bayes.pkl"
    
if "ridge_model_path" not in st.session_state:
    st.session_state.ridge_model_path = "ridge.pkl"
    
if "mysql_db_url" not in st.session_state:
    st.session_state.mysql_db_url = "mysql+mysqlconnector://root:password@localhost"
    
if "mysql_db_name" not in st.session_state:
    st.session_state.mysql_db_name = 'preprod_db'
    
tab1, tab2, tab3, tab4 =  st.tabs(["Model Config", "Model Training", "Model Evaluation", "Model Prediction"])

# Tab for Model Config
with tab1:
    st.subheader("Model Config")
    st.write("This is where you can set your paths for the model.")
    
    with st.form(key="model_config"):
        master_data_path = st.text_input("Master Data Path", st.session_state.master_data_path)
        st.session_state.master_data_path = master_data_path
        
        naive_bayes_model_path = st.text_input("Naive Bayes Model Path", st.session_state.naive_bayes_model_path)
        st.session_state.naive_bayes_model_path = naive_bayes_model_path
        
        ridge_model_path = st.text_input("Ridge Model Path", st.session_state.ridge_model_path)
        st.session_state.ridge_model_path = ridge_model_path
        
        mysql_db_url = st.text_input("MySQL DB URL", st.session_state.mysql_db_url)
        st.session_state.mysql_db_url = mysql_db_url
        
        mysql_db_name = st.text_input("MySQL DB Name", st.session_state.mysql_db_name)
        st.session_state.mysql_db_name = mysql_db_name

        if st.form_submit_button("Save Config", use_container_width=True):
            st.success("Config Saved!")
            
# Tab for Model Training
with tab2:
    st.subheader("Model Training")
    st.write("This is where you can train your model.")
    st.divider()
    
    st.markdown("<h3 style='text-align: center; color: white;'>Naive Bayes</h3>", unsafe_allow_html=True)
    
    if st.button("Train Naive Bayes Model", use_container_width=True):
        with st.status("Training model..."):
            st.write("Ingesting data..")
            ingest_and_transform(st.session_state.master_data_path, st.session_state.mysql_db_url, st.session_state.mysql_db_name)
            st.write("Data ingested, preprocesed and split and stored in the database. ✅")

            
            st.write("Training Naive Bayes Model..")
        st.success("Model Trained Successfully ! ✅")
        
        st.markdown("<h4 style='text-align: center; color: white;'>Model Training Metrics</h4>", unsafe_allow_html=True)
        nb_accuracy, nb_report, nb_cross_val_scores = train_model_naive_bayes(st.session_state.naive_bayes_model_path, st.session_state.mysql_db_url, st.session_state.mysql_db_name)
        
        st.text(f"Accuracy: {nb_accuracy}")
        st.text(f"Classification Report: {nb_report}")
        st.text(f"Cross Validation Scores: {nb_cross_val_scores}")
        
    st.divider()
    
    st.markdown("<h3 style='text-align: center; color: white;'>Ridge Classifier</h3>", unsafe_allow_html=True)
    
    if st.button("Train Ridge Classifier Model", use_container_width=True):
        with st.status("Training model..."):
            st.write("Ingesting data..")
            ingest_and_transform(st.session_state.master_data_path, st.session_state.mysql_db_url, st.session_state.mysql_db_name)
            st.write("Data ingested, preprocesed and split and stored in the database. ✅")

            
            st.write("Training Ridge Classifier Model..")
        st.success("Model Trained Successfully ! ✅")
        
        st.markdown("<h4 style='text-align: center; color: white;'>Model Training Metrics</h4>", unsafe_allow_html=True)
        ridge_accuracy, ridge_report, ridge_cross_val_scores = train_model_ridge(st.session_state.ridge_model_path, st.session_state.mysql_db_url, st.session_state.mysql_db_name)
        
        st.text(f"Accuracy: {ridge_accuracy}")
        st.text(f"Classification Report: {ridge_report}")
        st.text(f"Cross Validation Scores: {ridge_cross_val_scores}")
             
# Tab for Model Evaluation
with tab3:
    st.subheader("Model Evaluation")
    st.write("This is where you see the current metrics of your trained models.")
    st.divider()
    
    st.markdown("<h3 style='text-align: center; color: white;'>Naive Bayes</h3>", unsafe_allow_html=True)
    nb_metrics = model_inference(st.session_state.naive_bayes_model_path, st.session_state.mysql_db_url, st.session_state.mysql_db_name)
    
    st.markdown("<h4 style='text-align: center; color: white;'>Testing Data</h4>", unsafe_allow_html=True)
    st.text(f"Accuracy: {nb_metrics['Testing']['accuracy']}")
    st.text(f"Classification Report: {nb_metrics['Testing']['report']}")
    st.text(f"Cross Validation Scores: {nb_metrics['Testing']['cross_val_scores']}")
    
    st.markdown("<h4 style='text-align: center; color: white;'>Validation Data</h4>", unsafe_allow_html=True)
    st.text(f"Accuracy: {nb_metrics['Validation']['accuracy']}")
    st.text(f"Classification Report: {nb_metrics['Validation']['report']}")
    st.text(f"Cross Validation Scores: {nb_metrics['Validation']['cross_val_scores']}")
    
    st.markdown("<h4 style='text-align: center; color: white;'>Supervalidation Data</h4>", unsafe_allow_html=True)
    st.text(f"Accuracy: {nb_metrics['Supervalidation']['accuracy']}")
    st.text(f"Classification Report: {nb_metrics['Supervalidation']['report']}")
    st.text(f"Cross Validation Scores: {nb_metrics['Supervalidation']['cross_val_scores']}")
    
    st.divider()
    
    st.markdown("<h3 style='text-align: center; color: white;'>Ridge Classifier</h3>", unsafe_allow_html=True)
    ridge_metrics = model_inference(st.session_state.ridge_model_path, st.session_state.mysql_db_url, st.session_state.mysql_db_name)
    
    st.markdown("<h4 style='text-align: center; color: white;'>Testing Data</h4>", unsafe_allow_html=True)
    st.text(f"Accuracy: {ridge_metrics['Testing']['accuracy']}")
    st.text(f"Classification Report: {ridge_metrics['Testing']['report']}")
    st.text(f"Cross Validation Scores: {ridge_metrics['Testing']['cross_val_scores']}")
    
    st.markdown("<h4 style='text-align: center; color: white;'>Validation Data</h4>", unsafe_allow_html=True)
    st.text(f"Accuracy: {ridge_metrics['Validation']['accuracy']}")
    st.text(f"Classification Report: {ridge_metrics['Validation']['report']}")
    st.text(f"Cross Validation Scores: {ridge_metrics['Validation']['cross_val_scores']}")
    
    st.markdown("<h4 style='text-align: center; color: white;'>Supervalidation Data</h4>", unsafe_allow_html=True)
    st.text(f"Accuracy: {ridge_metrics['Supervalidation']['accuracy']}")
    st.text(f"Classification Report: {ridge_metrics['Supervalidation']['report']}")
    st.text(f"Cross Validation Scores: {ridge_metrics['Supervalidation']['cross_val_scores']}")
    
    st.divider()
    
# Tab for Model Prediction
with tab4:
    st.subheader("Model Prediction")
    st.write("This is where you can predict the weather based on the user's input.")
    st.divider()

    with st.form(key="weather_prediction"):
        # Divided the screen into 3 columns for better UI
        col1,col2,col3 = st.columns(3)
        with col1:
            # Taking the temperature input from the user
            temp = st.number_input("Temperature", min_value=0.0, max_value=50.0, step=0.5)

        with col2:
            # Taking the humidity input from the user
            humidity = st.number_input("Humidity", min_value=10.0, max_value=90.0, step=0.5)

        with col3:
            # Taking the wind speed input from the user
            wind_speed = st.number_input("Wind Speed", min_value=0.0, max_value=30.0, step=0.5)

        selected_model = st.radio("Select Model", ["Naive Bayes", "Ridge Classifier"], horizontal=True)
        
        if selected_model == "Naive Bayes":
            selected_model_path = st.session_state.naive_bayes_model_path
        else:
            selected_model_path = st.session_state.ridge_model_path
        
        # Predict button that predicts the weather based on the user's input
        if st.form_submit_button("Predict", use_container_width=True):
            prediction = manual_prediction(temp, humidity, wind_speed, selected_model_path, mysql_db_url, mysql_db_name)
            st.write(f"Prediction: {prediction}")