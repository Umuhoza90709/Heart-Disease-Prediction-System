"""
predict.py
Virtual prediction script to match frontend results for multiple examples
"""

import joblib
import json
import pandas as pd
import numpy as np

# ---------------- Paths ----------------
MODEL_PATH = 'deployment/best_model.pkl'
CLASS_NAMES_PATH = 'deployment/class_names.json'

# ---------------- Load model ----------------
model = joblib.load(MODEL_PATH)

with open(CLASS_NAMES_PATH, 'r') as f:
    class_labels_list = json.load(f)

class_labels = {str(i): name for i, name in enumerate(class_labels_list)}

# ---------------- Columns ----------------
REQUIRED_COLUMNS = [
    'age','trestbps','chol','thalach','oldpeak','ca',
    'sex','cp','fbs','restecg','exang','slope','thal'
]

NUMERIC_COLS = ['age','trestbps','chol','thalach','oldpeak','ca']

# ---------------- Preprocessing ----------------
def preprocess_input(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure all required columns exist
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    df = df[REQUIRED_COLUMNS]

    # Convert numeric columns
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Convert fbs to 0/1
    df['fbs'] = df['fbs'].apply(lambda x: 1 if str(x).upper() in ['TRUE','1'] else 0)

    # Encode sex (Male=1, Female=0)
    df['sex'] = df['sex'].apply(lambda x: 1 if str(x).strip().lower() == 'male' else 0)

    # Chest Pain encoding
    cp_map = {'Typical Angina':0,'Atypical Angina':1,'Non-Anginal Pain':2,'Asymptomatic':3}
    df['cp'] = df['cp'].map(cp_map)

    # Resting ECG encoding
    restecg_map = {'Normal':0,'ST-T abnormality':1,'LV hypertrophy':2}
    df['restecg'] = df['restecg'].map(restecg_map)

    # Exercise Induced Angina encoding
    df['exang'] = df['exang'].apply(lambda x: 1 if str(x).strip().lower() == 'yes' else 0)

    # Slope encoding
    slope_map = {'Upsloping':0,'Flat':1,'Downsloping':2}
    df['slope'] = df['slope'].map(slope_map)

    # Thalassemia encoding
    thal_map = {'Normal':1,'Fixed Defect':2,'Reversible Defect':3}
    df['thal'] = df['thal'].map(thal_map)

    return df

# ---------------- SAMPLE INPUTS ----------------
samples = [
    {
        'name': 'Sample 0 - Original Frontend',
        'age': 55,
        'sex': 'Male',
        'cp': 'Typical Angina',
        'trestbps': 130,
        'chol': 250,
        'fbs': 'FALSE',
        'restecg': 'Normal',
        'thalach': 150,
        'exang': 'No',
        'oldpeak': 1.2,
        'slope': 'Flat',
        'ca': 0,
        'thal': 'Normal'
    },
    {
        'name': 'Sample 1 - Middle-aged Male',
        'age': 60,
        'sex': 'Male',
        'cp': 'Atypical Angina',
        'trestbps': 140,
        'chol': 260,
        'fbs': 'TRUE',
        'restecg': 'ST-T abnormality',
        'thalach': 160,
        'exang': 'No',
        'oldpeak': 1.0,
        'slope': 'Upsloping',
        'ca': 1,
        'thal': 'Fixed Defect'
    },
    {
        'name': 'Sample 2 - Older Female',
        'age': 70,
        'sex': 'Female',
        'cp': 'Asymptomatic',
        'trestbps': 150,
        'chol': 280,
        'fbs': 'TRUE',
        'restecg': 'LV hypertrophy',
        'thalach': 140,
        'exang': 'Yes',
        'oldpeak': 2.5,
        'slope': 'Downsloping',
        'ca': 3,
        'thal': 'Reversible Defect'
    }
]

# ---------------- Prediction Loop ----------------
for sample in samples:
    print("\n====================================")
    print(f"Predicting for: {sample['name']}")
    
    df = pd.DataFrame([sample])
    df = preprocess_input(df)
    
    print("\nINPUT TO MODEL (after preprocessing):")
    print(df)
    
    pred = model.predict(df)[0]
    pred_proba = model.predict_proba(df)[0]
    
    print("\nPREDICTION RESULT:")
    print("------------------")
    print("Predicted class:", class_labels[str(pred)])
    
    print("\nProbabilities:")
    for i, p in enumerate(pred_proba):
        print(f"{class_labels[str(i)]}: {round(p*100,1)}%")
