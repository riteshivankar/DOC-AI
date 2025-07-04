from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load data
train_df = pd.read_csv("Datasets/Training.csv")
X_columns = train_df.drop('prognosis', axis=1).columns
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(train_df['prognosis'])

# Load trained model
model = joblib.load("Models/svc.pkl")

# Load additional information
desc_df = pd.read_csv("Datasets/description.csv")
precaution_df = pd.read_csv("Datasets/precautions_df.csv")
diet_df = pd.read_csv("Datasets/diets.csv")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_symptoms = [s.strip().lower().replace(" ", "_") for s in data.get("symptoms", "").split(",")]

    # Create input vector
    input_vector = np.zeros(len(X_columns))
    for i, col in enumerate(X_columns):
        if col.lower() in input_symptoms:
            input_vector[i] = 1

    # Predict disease
    prediction = model.predict([input_vector])[0]
    predicted_disease = label_encoder.inverse_transform([prediction])[0]

    # Get description
    desc = desc_df.loc[desc_df['Disease'] == predicted_disease, 'Description'].values
    description = desc[0] if len(desc) > 0 else "No description available."

    # Get precautions
    precaution_row = precaution_df[precaution_df["Disease"] == predicted_disease]
    if not precaution_row.empty:
        precautions = precaution_row.iloc[0, 1:].dropna().tolist()
    else:
        precautions = []

    # Get diet
    diet_row = diet_df[diet_df["Disease"] == predicted_disease]
    if not diet_row.empty:
        diet = diet_row.iloc[0, 1:].dropna().tolist()
    else:
        diet = []

    return jsonify({
        "disease": predicted_disease,
        "description": description,
        "precautions": precautions,
        "diet": diet
    })

if __name__ == '__main__':
    app.run(debug=True)
