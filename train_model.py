# train_model.py
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def train_model():
    # Load the dataset
    data = pd.read_csv('data/dataset.csv')

    # Preprocess data
    data.fillna('None', inplace=True)
    for col in data.columns:
        data[col] = data[col].astype(str).str.strip()

    all_symptoms = sorted(set(symptom.strip() for symptoms in data.iloc[:, 1:].values for symptom in symptoms if symptom != 'None'))

    # Create binary symptom presence matrix
    X = pd.DataFrame(0, index=range(len(data)), columns=all_symptoms)
    for i, row in data.iterrows():
        for symptom in row[1:]:
            if symptom != 'None':
                X.at[i, symptom] = 1

    # Encode target variable (Disease)
    y = LabelEncoder().fit_transform(data['Disease'])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save the model and symptoms
    with open('models/disease_prediction_model.pkl', 'wb') as f:
        pickle.dump((model, LabelEncoder().fit(data['Disease'])), f)

    with open('models/symptom_columns.pkl', 'wb') as f:
        pickle.dump(all_symptoms, f)

    print("Model training completed and saved.")
    
if __name__ == "__main__":
    train_model()