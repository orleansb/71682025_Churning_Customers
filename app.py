import tensorflow as tf
from tensorflow import keras
import streamlit as st
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

import joblib
import pickle
import numpy as np

with open('scaler.pkl', 'rb') as f:
    loaded_scaler = pickle.load(f)

encoder = LabelEncoder()
encoder.classes = np.load('classes.npy', allow_pickle = True)

model_path = 'model.h5'
model = load_model(model_path)



st.title("Customer Churn Prediction")


TotalCharges= st.number_input('Total Charges', min_value=0.0, step=0.01, key='potential')
MonthlyCharges = st.number_input('Monthly Charges', min_value=0.0, step=0.01, key='MonthlyCharges')
tenure = st.number_input('Tenure', min_value=0.0, step=0.01, key='tenure')

contract_options = ['Month-to-month', 'One year', 'Two year']
Contract = st.radio('How long is the contract', contract_options, key='Contract')

paymentMethod_options = ['Electronic check', 'Mailed check', 'Bank transfer(automatic)','Credit card(automatic)']
PaymentMethod = st.radio('What type of payment method do you use', paymentMethod_options, key='PaymentMethod')

techSupport_options = ['Yes', 'No', 'No internet service']
TechSupport = st.radio('Do you have tech support', techSupport_options, key='TechSupport')

onlineSecurity_options = ['Yes', 'No', 'No internet service']
OnlineSecurity = st.radio('Do you have online security', onlineSecurity_options, key='OnlineSecurity')

internetService_options = ['DSL', 'Fibre Optic', 'No']
InternetService = st.radio('Do you have internet service', internetService_options, key='InternetService')

gender_options = ['Female', 'Male']
Gender = st.radio('What is your gender', gender_options, key='Gender')

onlineBackup_options = ['Yes', 'No', 'No internet service']
OnlineBackup = st.radio('Do you have online backup', onlineBackup_options, key='OnlineBackup')



if st.button("Predict"):
    input_numerical = np.array([[TotalCharges, MonthlyCharges, tenure]])
    input_categorical = np.array([[Contract, PaymentMethod, TechSupport, OnlineSecurity, InternetService, Gender, OnlineBackup]])
    encoder.fit(input_categorical.flatten())
    
    new_categorical = encoder.transform(input_categorical.flatten())
    new_categorical_reshaped = new_categorical.reshape(-1, 1)
    new_numerical_reshaped = input_numerical.reshape(-1, 1)

    new_data = np.concatenate((new_numerical_reshaped, new_categorical_reshaped), axis=0)
    new_data = new_data.T
    new_data_scaled = loaded_scaler.transform(new_data)


    prediction_probabilities = model.predict(new_data_scaled)
    prediction = np.argmax(prediction_probabilities, axis=-1)[0]
    confidence_score = prediction_probabilities[0,prediction] * 100

    if prediction == 1 :
        st.write("The customer is likely to churn")
    elif prediction == 0 :
        st.write("The customer is not likely to churn")
    st.write(f"Confidence Score: {confidence_score:.2f}%")
    
    user_feedback = st.radio("Do you agree with the prediction?", ["Yes", "No"])
    if user_feedback == "Yes":
        st.write("Thank you for your feedback!")
    else:
        st.write("Sorry for the error.")