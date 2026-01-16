from flask import Flask, request, jsonify, render_template
import joblib
import json
import pandas as pd
import numpy as np
from datetime import datetime

app = Flask(__name__)

# ---------------- Load model ----------------
MODEL_PATH = 'deployment/best_model.pkl'
CLASS_NAMES_PATH = 'deployment/class_names.json'

model = joblib.load(MODEL_PATH)

with open(CLASS_NAMES_PATH) as f:
    class_labels_list = json.load(f)

class_labels = {i: name for i, name in enumerate(class_labels_list)}

# ---------------- Preprocessing (SINGLE SOURCE) ----------------
REQUIRED_COLUMNS = [
    'age','trestbps','chol','thalach','oldpeak','ca',
    'sex','cp','fbs','restecg','exang','slope','thal'
]

NUMERIC_COLS = ['age','trestbps','chol','thalach','oldpeak','ca']

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    df = df[REQUIRED_COLUMNS]

    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df['fbs'] = df['fbs'].apply(lambda x: 1 if str(x).upper() == 'TRUE' else 0)
    df['exang'] = df['exang'].apply(lambda x: 1 if str(x).lower() == 'yes' else 0)
    df['sex'] = df['sex'].apply(lambda x: 1 if x == 'Male' else 0)

    df['cp'] = df['cp'].map({
        'Typical Angina': 0,
        'Atypical Angina': 1,
        'Non-Anginal Pain': 2,
        'Asymptomatic': 3
    })

    df['restecg'] = df['restecg'].map({
        'Normal': 0,
        'ST-T abnormality': 1,
        'LV hypertrophy': 2
    })

    df['slope'] = df['slope'].map({
        'Upsloping': 0,
        'Flat': 1,
        'Downsloping': 2
    })

    df['thal'] = df['thal'].map({
        'Normal': 1,
        'Fixed Defect': 2,
        'Reversible Defect': 3
    })

    return df

# ---------------- Routes ----------------
@app.route('/')
def home():
    return render_template('index_25RP18166.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json()

    df = pd.DataFrame([data])
    df = preprocess(df)

    pred = int(model.predict(df)[0])
    probs = model.predict_proba(df)[0]

    confidence = round(probs[pred] * 100, 1)

    risk_levels = {
        0: 'No Risk',
        1: 'Very Mild Risk',
        2: 'Mild Risk',
        3: 'Severe Risk',
        4: 'Immediate Danger'
    }

    probabilities = []
    for i, p in enumerate(probs):
        probabilities.append({
            'class_name': class_labels[i],
            'probability': round(p * 100, 1)
        })

    probabilities.sort(key=lambda x: x['probability'], reverse=True)

    return jsonify({
        'success': True,
        'prediction': {
            'class_name': class_labels[pred],
            'risk_level': risk_levels[pred],
            'confidence': confidence
        },
        'probabilities': probabilities,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })

if __name__ == '__main__':
    app.run(debug=True)
