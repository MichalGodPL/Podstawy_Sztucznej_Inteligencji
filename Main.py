import webview
import threading
import torch
import numpy as np
from Analiza import ImprovedHeartDiseaseModel, scaler, model  # Import modelu i skalera

# Klasa API do komunikacji z JavaScript
class Api:
    def predict(self, data):
        try:
            # Przetwarzanie danych z formularza HTML
            features = [
                float(data['age']),
                1 if data['sex'] == 'Mężczyzna' else 0,
                float(data['cholesterol']),
                float(data['blood_pressure']),
                float(data['heart_rate']),
                1 if data['diabetes'] else 0,
                1 if data['smoking'] else 0,
                float(data['exercise_hours_per_week']),
                {'Zdrowa': 0, 'Umiarkowana': 1, 'Słaba': 2}[data['diet']],
                1 if data['obesity'] else 0,
                1 if data['alcohol_consumption'] else 0,
                1 if data['medication_use'] else 0,
                float(data['stress_level']),
                float(data['sedentary_hours_per_day']),
                float(data['income']),
                float(data['triglycerides']),
                1 if data['physical_activity_days_per_week'] else 0,
                float(data['sleep_hours_per_day']),
                {'Afryka': 0, 'Ameryka Północna': 1, 'Ameryka Południowa': 2, 'Azja': 3, 'Australia': 4, 'Europa': 5}[data['continent']],
                0 if data['hemisphere'] == 'Północna' else 1
            ]

            # Standaryzacja danych
            scaled_data = scaler.transform(np.array([features]))

            # Konwersja na tensor
            input_tensor = torch.tensor(scaled_data, dtype=torch.float32)

            # Predykcja za pomocą modelu
            with torch.no_grad():
                prediction = model(input_tensor)
                risk_prob = torch.sigmoid(prediction).item()
                risk = 'Wysokie' if risk_prob > 0.5 else 'Niskie'

            # Zwróć wyniki do HTML
            return {
                'date': data['date'],
                'time': data['time'],
                'age': data['age'],
                'cholesterol': data['cholesterol'],
                'blood_pressure': data['blood_pressure'],
                'heart_rate': data['heart_rate'],
                'risk': risk
            }
        except Exception as e:
            return {'error': str(e)}

# Funkcja startowa (opcjonalnie możesz dodać inne skrypty)
def start_scripts():
    print("Aplikacja uruchomiona.")
    # Jeśli masz inne skrypty do uruchomienia, dodaj je tutaj, np.:
    # threading.Thread(target=run_some_script, daemon=True).start()

# Tworzenie okna WebView
api = Api()
window = webview.create_window("Status SI", "index.html", width=1600, height=900, js_api=api)
webview.start(start_scripts)