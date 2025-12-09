# predict.py
import pickle
import pandas as pd

# Load the model and label encoder
model, label_encoder = pickle.load(open('models/disease_prediction_model.pkl', 'rb'))

# Load the list of symptoms used during training
with open('models/symptom_columns.pkl', 'rb') as f:
    all_symptoms = pickle.load(f)

def predict_disease(symptoms):
    # Create the input vector
    input_vector = pd.DataFrame(columns=all_symptoms, index=[0])
    input_vector.loc[0] = 0

    # Mark selected symptoms as 1
    for symptom in symptoms:
        if symptom in all_symptoms:
            input_vector.at[0, symptom] = 1

    # Ensure the columns match the trained model
    input_vector = input_vector[all_symptoms]

    # Make prediction
    prediction = model.predict(input_vector)
    disease = label_encoder.inverse_transform(prediction)[0]
    return disease