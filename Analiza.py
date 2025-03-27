import torch
import torch.nn as nn
import pickle
import numpy as np
from flask import Flask, render_template, request, jsonify
import os
import sys

# Definicja modelu z elastyczną architekturą
class ImprovedHeartDiseaseModel(nn.Module):
    def __init__(self, input_dim, hidden_sizes, dropout):
        super(ImprovedHeartDiseaseModel, self).__init__()
        layers = []
        prev_size = input_dim
        for size in hidden_sizes:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.BatchNorm1d(size))
            layers.append(nn.SiLU())
            layers.append(nn.Dropout(dropout))
            prev_size = size
        self.layers = nn.Sequential(*layers)
        self.output = nn.Linear(prev_size, 1)
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.layers(x)
        return self.output(x)

# Inicjalizacja Flask
app = Flask(__name__)

# Wczytywanie najlepszych hiperparametrów
try:
    if not os.path.exists('best_params.pkl'):
        raise FileNotFoundError("Plik 'best_params.pkl' nie istnieje. Uruchom 'PyTorch.py' najpierw.")
    with open('best_params.pkl', 'rb') as f:
        best_params = pickle.load(f)
    print("Wczytano najlepsze hiperparametry z 'best_params.pkl'.")
except FileNotFoundError as e:
    print(f"Błąd: {e}")
    sys.exit(1)

# Ładowanie modelu i skalera
try:
    if not os.path.exists('heart_disease_model_final.pth') or not os.path.exists('scaler.pkl'):
        raise FileNotFoundError("Plik 'heart_disease_model_final.pth' lub 'scaler.pkl' nie istnieje. Uruchom 'PyTorch.py' najpierw.")

    input_dim = 17  # Liczba cech po usunięciu 'Physical Activity Days Per Week'
    model = ImprovedHeartDiseaseModel(input_dim=input_dim, hidden_sizes=best_params['hidden_sizes'], dropout=best_params['dropout'])
    model.load_state_dict(torch.load('heart_disease_model_final.pth'))
    model.eval()
    print("Wczytano model z 'heart_disease_model_final.pth'.")

    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    print("Wczytano scaler z 'scaler.pkl'.")
except FileNotFoundError as e:
    print(f"Błąd: {e}")
    sys.exit(1)
except Exception as e:
    print(f"Błąd podczas wczytywania modelu lub skalera: {e}")
    sys.exit(1)

# Funkcja predykcji
def predict_risk(data, model, scaler, threshold=0.5):
    data_scaled = scaler.transform(data)
    data_tensor = torch.tensor(data_scaled, dtype=torch.float32)
    with torch.no_grad():
        logits = model(data_tensor)
        probabilities = torch.sigmoid(logits)
        predictions = (probabilities > threshold).float()
    return probabilities.numpy(), predictions.numpy()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
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
            float(form_data['triglycerides']),
            float(form_data['sleep_hours_per_day']),
            {'Afryka': 0, 'Ameryka Północna': 1, 'Ameryka Południowa': 2, 'Azja': 3, 'Australia': 4, 'Europa': 5}[form_data['continent']]
        ]

        scaled_data = scaler.transform(np.array([features]))
        input_tensor = torch.tensor(scaled_data, dtype=torch.float32)
        with torch.no_grad():
            prediction = model(input_tensor)
            risk_prob = torch.sigmoid(prediction).item()
            risk = 'Wysokie' if risk_prob > 0.5 else 'Niskie'

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

if __name__ == "__main__":
    new_data = np.random.rand(1, 17)
    probabilities, predictions = predict_risk(new_data, model, scaler)
    print(f"Prawdopodobieństwo zawału: {probabilities[0]:.4f}")
    print(f"Predykcja (0=brak ryzyka, 1=ryzyko): {int(predictions[0])}")
    app.run(debug=True)