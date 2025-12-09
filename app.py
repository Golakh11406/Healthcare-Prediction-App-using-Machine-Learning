# app.py
import streamlit as st
import pickle
import pandas as pd
from predict import predict_disease

# Load model and symptom columns
model, label_encoder = pickle.load(open('models/disease_prediction_model.pkl', 'rb'))
with open('models/symptom_columns.pkl', 'rb') as f:
    all_symptoms = pickle.load(f)

# Streamlit app title
st.title('Disease Prediction App')

# Streamlit app description
st.write('Select symptoms to predict the possible disease.')

# Multi-select input for symptoms
selected_symptoms = st.multiselect(
    'Select symptoms:',
    options=all_symptoms,
    default=[]
)

# Predict disease when user selects symptoms
if selected_symptoms:
    disease = predict_disease(selected_symptoms)
    st.write(f"The predicted disease is: **{disease}**")
    # Load description and precaution datasets
    desc_df = pd.read_csv("data/symptom_Description.csv")
    prec_df = pd.read_csv("data/symptom_Precaution.csv")

    # Strip whitespace from 'Disease' column
    desc_df['Disease'] = desc_df['Disease'].str.strip()
    prec_df['Disease'] = prec_df['Disease'].str.strip()

    # Fetch and display disease description
    description_row = desc_df[desc_df['Disease'].str.lower() == disease.lower()]
    description = description_row['Description'].values[0] if not description_row.empty else "No description available."
    st.subheader("Disease Description")
    st.write(description)

    # Fetch and display precautions
    precautions_row = prec_df[prec_df['Disease'].str.lower() == disease.lower()]
    precautions = precautions_row.iloc[0, 1:].dropna().tolist() if not precautions_row.empty else ["No precautions available."]
    st.subheader("Precautions")
    for i, precaution in enumerate(precautions, start=1):
        st.write(f"{i}. {precaution}")

else:
    st.write("Please select symptoms to get a prediction.")