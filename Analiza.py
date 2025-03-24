import torch
import torch.nn as nn
import pickle
import numpy as np
from flask import Flask, render_template, request, jsonify

# Definicja modelu
class ImprovedHeartDiseaseModel(nn.Module):
    def __init__(self, input_dim=18):  # Ustawiamy na 18, zgodnie z modelem
        super(ImprovedHeartDiseaseModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)  # 18 -> 256
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)       # 256 -> 128
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)        # 128 -> 64
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, 32)         # 64 -> 32
        self.bn4 = nn.BatchNorm1d(32)
        self.output = nn.Linear(32, 1)       # 32 -> 1
        self.dropout = 0.3                   # Dropout
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = torch.nn.SiLU()(self.bn1(self.fc1(x)))
        x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = torch.nn.SiLU()(self.bn2(self.fc2(x)))
        x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = torch.nn.SiLU()(self.bn3(self.fc3(x)))
        x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = torch.nn.SiLU()(self.bn4(self.fc4(x)))
        x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
        return self.output(x)

# Inicjalizacja aplikacji Flask
app = Flask(__name__)

# Ładowanie modelu i skalera
model = ImprovedHeartDiseaseModel(input_dim=18)
model.load_state_dict(torch.load('heart_disease_model.pth'))
model.eval()

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Funkcja predykcji
def predict_risk(data, model, scaler, threshold=0.5):
    """
    Przewiduje ryzyko zawału dla nowych danych.
    data: numpy array o kształcie [n_samples, 18]
    """
    data_scaled = scaler.transform(data)
    data_tensor = torch.tensor(data_scaled, dtype=torch.float32)
    
    with torch.no_grad():
        logits = model(data_tensor)
        probabilities = torch.sigmoid(logits)
        predictions = (probabilities > threshold).float()
    
    return probabilities.numpy(), predictions.numpy()

# Główna strona z HTML
@app.route('/')
def index():
    return render_template('index.html')

# Endpoint do predykcji
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Pobierz dane z formularza HTML (18 cech)
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
            1 if form_data['physical_activity_days_per_week'] else 0,
            float(form_data['sleep_hours_per_day']),
            {'Afryka': 0, 'Ameryka Północna': 1, 'Ameryka Południowa': 2, 'Azja': 3, 'Australia': 4, 'Europa': 5}[form_data['continent']]
        ]

        # Standaryzacja danych
        scaled_data = scaler.transform(np.array([features]))

        # Konwersja na tensor i predykcja
        input_tensor = torch.tensor(scaled_data, dtype=torch.float32)
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

# Uruchomienie aplikacji (opcjonalne, jeśli używasz Flask)
if __name__ == "__main__":
    # Test predykcji
    new_data = np.random.rand(1, 18)  # Losowe dane testowe
    probabilities, predictions = predict_risk(new_data, model, scaler)
    print(f"Prawdopodobieństwo zawału: {probabilities[0]:.4f}")
    print(f"Predykcja (0=brak ryzyka, 1=ryzyko): {int(predictions[0])}")
    
    # Uruchomienie Flask
    app.run(debug=True)