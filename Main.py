import webview

import threading

import torch

import numpy as np

import pickle

import importlib

import os

import sys

import signal


# Początkowe wartości dla modelu i skalera

model = None

scaler = None

api = None

window = None


# Funkcja do wczytywania zapisanego modelu i skalera

def load_model_and_scaler():

    global model, scaler

    try:

        # Sprawdzenie, czy wszystkie wymagane pliki istnieją

        required_files = ['scaler.pkl', 'best_params.pkl', 'heart_disease_model_final.pth']

        for file in required_files:

            if not os.path.exists(file):

                raise FileNotFoundError(f"Plik '{file}' nie istnieje. Uruchom 'PyTorch.py', aby wygenerować brakujące pliki.")


        # Wczytaj scaler

        with open('scaler.pkl', 'rb') as f:

            scaler = pickle.load(f)

        print("Wczytano scaler z 'scaler.pkl'.")


        # Wczytaj najlepsze hiperparametry

        with open('best_params.pkl', 'rb') as f:

            best_params = pickle.load(f)

        print("Wczytano najlepsze hiperparametry z 'best_params.pkl'.")


        # Definicja modelu

        class ImprovedHeartDiseaseModel(torch.nn.Module):

            def __init__(self, input_dim, hidden_sizes, dropout):

                super(ImprovedHeartDiseaseModel, self).__init__()

                layers = []

                prev_size = input_dim

                for size in hidden_sizes:

                    layers.append(torch.nn.Linear(prev_size, size))

                    layers.append(torch.nn.BatchNorm1d(size))

                    layers.append(torch.nn.SiLU())

                    layers.append(torch.nn.Dropout(dropout))

                    prev_size = size

                self.layers = torch.nn.Sequential(*layers)

                self.output = torch.nn.Linear(prev_size, 1)

                self.initialize_weights()


            def initialize_weights(self):

                for m in self.modules():

                    if isinstance(m, torch.nn.Linear):

                        torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

                        torch.nn.init.zeros_(m.bias)


            def forward(self, x):

                x = self.layers(x)

                return self.output(x)


        # Inicjalizacja modelu

        input_dim = 17  # Liczba cech po usunięciu 'Physical Activity Days Per Week'

        model = ImprovedHeartDiseaseModel(input_dim=input_dim, hidden_sizes=best_params['hidden_sizes'], dropout=best_params['dropout'])

        model.load_state_dict(torch.load('heart_disease_model_final.pth'))

        model.eval()

        print("Wczytano model z 'heart_disease_model_final.pth'.")

    except FileNotFoundError as e:

        print(f"Błąd: {e}")

        sys.exit(1)

    except Exception as e:

        print(f"Błąd podczas wczytywania modelu lub skalera: {e}")

        sys.exit(1)


# Klasa API do komunikacji z JavaScript

class Api:

    def predict(self, data):

        global model, scaler

        try:

            if scaler is None or model is None:

                raise ValueError("Model lub scaler nie zostały poprawnie wczytane. Uruchom 'PyTorch.py' najpierw.")


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


# Funkcja do uruchamiania skryptu w wątku

def run_script(script_name):

    try:

        module = importlib.import_module(script_name)

        print(f"Uruchomiono {script_name} w wątku.")

    except Exception as e:

        print(f"Błąd podczas uruchamiania {script_name}: {e}")

        raise


# Funkcja startowa

def start_scripts():

    global api, window

    print("Aplikacja uruchomiona.")
    

    # Pytanie w terminalu

    while True:

        response = input("Czy Chcesz Wytrenować Model od Nowa? [T/N]: ").strip().lower()

        if response in ['t', 'n']:

            break

        print("Proszę wpisać 'T' lub 'N'.")


    if response == 't':

        print("Uruchamianie EdycjaDanych.py...")

        # Uruchamianie EdycjaDanych.py w osobnym wątku

        edycja_thread = threading.Thread(target=run_script, args=('EdycjaDanych',), daemon=True)

        edycja_thread.start()

        edycja_thread.join()  # Czekaj na zakończenie EdycjaDanych.py


        print("Uruchamianie PyTorch.py...")

        # Uruchamianie PyTorch.py w osobnym wątku

        pytorch_thread = threading.Thread(target=run_script, args=('PyTorch',), daemon=True)

        pytorch_thread.start()

        pytorch_thread.join()  # Czekaj na zakończenie PyTorch.py


        # Wczytaj model i scaler po treningu

        print("Wczytywanie modelu i skalera po treningu...")

        load_model_and_scaler()

    else:

        # Wczytaj istniejący model i scaler z plików

        print("Wczytywanie istniejącego modelu i skalera...")

        load_model_and_scaler()


    # Tworzenie i uruchamianie okna WebView dopiero po zakończeniu wszystkich operacji

    print("Uruchamianie okna aplikacji...")

    api = Api()

    window = webview.create_window("Status SI", "index.html", width=1600, height=900, js_api=api)

    webview.start()


# Obsługa sygnału SIGINT (Ctrl+C)

def signal_handler(sig, frame):

    print("Zamykanie aplikacji...")

    if window is not None:

        window.destroy()

    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# Uruchomienie funkcji startowej

start_scripts()