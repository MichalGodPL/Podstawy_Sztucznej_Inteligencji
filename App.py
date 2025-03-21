from flask import Flask, render_template, request, jsonify
import torch
import numpy as np
from Analiza import ImprovedHeartDiseaseModel, scaler  # Import modelu i skalera z Analiza.py

app = Flask(__name__)

# Ładowanie modelu
input_dim = 20  # Liczba cech zgodna z modelem
model = ImprovedHeartDiseaseModel(input_dim=input_dim)
model.load_state_dict(torch.load('heart_disease_model.pth'))
model.eval()

# Główna strona z HTML
@app.route('/')
def index():
    return render_template('index.html')

# Endpoint do predykcji
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Pobierz dane z formularza HTML
        form_data = request.json
        features = [
            float(form_data['age']),
            1 if form_data['sex'] == 'Mężczyzna' else 0,
            float(form_data['cholesterol']),
            float(form_data['blood_pressure']),
            float(form_data['heart_rate']),
            1 if form_data['diabetes'] else 0,
            1 if form_data['smoking'] else 0,
            float(form_data['exercise_hours_per_week']),
            {'Zdrowa': 0, 'Umiarkowana': 1, 'Słaba': 2}[form_data['diet']],
            1 if form_data['obesity'] else 0,
            1 if form_data['alcohol_consumption'] else 0,
            1 if form_data['medication_use'] else 0,
            float(form_data['stress_level']),
            float(form_data['sedentary_hours_per_day']),
            float(form_data['income']),
            float(form_data['triglycerides']),
            1 if form_data['physical_activity_days_per_week'] else 0,
            float(form_data['sleep_hours_per_day']),
            {'Afryka': 0, 'Ameryka Północna': 1, 'Ameryka Południowa': 2, 'Azja': 3, 'Australia': 4, 'Europa': 5}[form_data['continent']],
            0 if form_data['hemisphere'] == 'Północna' else 1
        ]

        # Standaryzacja danych
        scaled_data = scaler.transform(np.array([features]))

        # Konwersja na tensor
        input_tensor = torch.tensor(scaled_data, dtype=torch.float32)

        # Predykcja
        with torch.no_grad():
            prediction = model(input_tensor)
            risk_prob = torch.sigmoid(prediction).item()
            risk = 'Wysokie' if risk_prob > 0.5 else 'Niskie'

        # Zwróć wynik jako JSON
        return jsonify({
            'date': form_data['date'],
            'time': form_data['time'],
            'age': form_data['age'],
            'cholesterol': form_data['cholesterol'],
            'blood_pressure': form_data['blood_pressure'],
            'heart_rate': form_data['heart_rate'],
            'risk': risk,
            'probability': risk_prob
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)