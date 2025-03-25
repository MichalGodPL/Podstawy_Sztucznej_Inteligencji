import webview
import threading
import torch
import numpy as np
import pickle
import importlib
from Analiza import ImprovedHeartDiseaseModel, scaler, model  # Import modelu i skalera

# Klasa API do komunikacji z JavaScript
class Api:
    def predict(self, data):
        try:
            # Przetwarzanie danych z formularza HTML (17 cech po usunięciu 'Physical Activity Days Per Week')
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
                float(data['triglycerides']),
                float(data['sleep_hours_per_day']),
                {'Afryka': 0, 'Ameryka Północna': 1, 'Ameryka Południowa': 2, 'Azja': 3, 'Australia': 4, 'Europa': 5}[data['continent']]
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

            # Wczytaj metryki modelu
            try:
                with open('metrics.pkl', 'rb') as f:
                    metrics = pickle.load(f)
            except FileNotFoundError:
                metrics = {
                    'accuracy': 'N/A',
                    'precision': 'N/A',
                    'recall': 'N/A',
                    'f1_score': 'N/A'
                }

            # Zwróć wyniki i metryki do HTML
            return {
                'date': data['date'],
                'time': data['time'],
                'age': data['age'],
                'cholesterol': data['cholesterol'],
                'blood_pressure': data['blood_pressure'],
                'heart_rate': data['heart_rate'],
                'risk': risk,
                'metrics': metrics
            }
        except Exception as e:
            return {'error': str(e)}

    def run_script(self, script_name):
        """Uruchamia skrypt w osobnym wątku i czeka na jego zakończenie."""
        try:
            # Dynamiczne załadowanie i uruchomienie modułu w nowym wątku
            def run():
                module = importlib.import_module(script_name)
                print(f"Uruchomiono {script_name} w wątku.")

            thread = threading.Thread(target=run, daemon=True)
            thread.start()
            thread.join()  # Czekaj na zakończenie wątku
            return {'status': 'success'}
        except Exception as e:
            return {'error': str(e)}

# Funkcja do uruchamiania skryptu w wątku (dla start_scripts)
def run_script(script_name):
    try:
        module = importlib.import_module(script_name)
        print(f"Uruchomiono {script_name} w wątku.")
    except Exception as e:
        print(f"Błąd podczas uruchamiania {script_name}: {e}")

# Funkcja startowa
def start_scripts():
    print("Aplikacja uruchomiona.")
    # Uruchamianie EdycjaDanych.py w osobnym wątku
    threading.Thread(target=run_script, args=('EdycjaDanych',), daemon=True).start()
    # Uruchamianie PyTorch.py w osobnym wątku (opcjonalnie przy starcie)
    threading.Thread(target=run_script, args=('PyTorch',), daemon=True).start()

# Tworzenie okna WebView
api = Api()
window = webview.create_window("Status SI", "index.html", width=1600, height=900, js_api=api)
webview.start(start_scripts)