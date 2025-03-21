import torch
import torch.nn as nn
import numpy as np
import pickle

# Definicja modelu (taka sama jak w treningu)
class ImprovedHeartDiseaseModel(nn.Module):
    def __init__(self, input_dim):
        super(ImprovedHeartDiseaseModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, 32)
        self.bn4 = nn.BatchNorm1d(32)
        self.output = nn.Linear(32, 1)
        self.dropout = 0
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

# Ładowanie modelu i scaler'a
input_dim = 20  # Zastąp odpowiednią liczbą cech z Twojego zbioru danych
model = ImprovedHeartDiseaseModel(input_dim=input_dim)
model.load_state_dict(torch.load('heart_disease_model.pth'))
model.eval()

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Funkcja do zbierania danych od użytkownika
def get_user_input():
    print("Wprowadź dane pacjenta do predykcji ryzyka ataku serca:")
    
    features = []
    features.append(float(input("Wiek: ")))
    features.append(1 if input("Płeć (Mężczyzna/Kobieta): ").lower() == "mężczyzna" else 0)
    features.append(float(input("Poziom cholesterolu (mg/dL): ")))
    features.append(float(input("Ciśnienie krwi (mmHg, tylko systolic): ")))  # Zakładam, że to systolic BP
    features.append(float(input("Tętno (BPM): ")))
    features.append(1 if input("Cukrzyca (Tak/Nie): ").lower() == "tak" else 0)
    features.append(1 if input("Palenie (Tak/Nie): ").lower() == "tak" else 0)
    features.append(float(input("Godziny ćwiczeń tygodniowo: ")))
    features.append({"Zdrowa": 0, "Umiarkowana": 1, "Słaba": 2}[input("Jakość diety (Zdrowa/Umiarkowana/Słaba): ")])
    features.append(1 if input("Otyłość (Tak/Nie): ").lower() == "tak" else 0)
    features.append(1 if input("Spożycie alkoholu (Tak/Nie): ").lower() == "tak" else 0)
    features.append(1 if input("Używanie leków (Tak/Nie): ").lower() == "tak" else 0)
    features.append(float(input("Poziom stresu (0-9): ")))
    features.append(float(input("Godziny siedzące dziennie: ")))
    features.append(float(input("Dochód roczny (PLN): ")))
    features.append(float(input("Poziom trójglicerydów (mg/dL): ")))
    features.append(1 if input("Aktywność fizyczna (dni w tygodniu, Tak/Nie): ").lower() == "tak" else 0)
    features.append(float(input("Godziny snu dziennie: ")))
    features.append({"Afryka": 0, "Ameryka Północna": 1, "Ameryka Południowa": 2, "Azja": 3, "Australia": 4, "Europa": 5}[input("Kontynent (Afryka/Ameryka Północna/Ameryka Południowa/Azja/Australia/Europa): ")])
    features.append(0 if input("Półkula (Północna/Południowa): ").lower() == "północna" else 1)

    return np.array([features])

# Główna funkcja predykcji
def predict_heart_attack():
    try:
        # Pobierz dane od użytkownika
        user_data = get_user_input()

        # Standaryzacja danych
        scaled_data = scaler.transform(user_data)

        # Konwersja na tensor
        input_tensor = torch.tensor(scaled_data, dtype=torch.float32)

        # Predykcja
        with torch.no_grad():
            prediction = model(input_tensor)
            risk_prob = torch.sigmoid(prediction).item()
            risk = 'Wysokie' if risk_prob > 0.5 else 'Niskie'

        # Wyświetl wyniki
        print("\nWyniki predykcji:")
        print(f"Prawdopodobieństwo ryzyka ataku serca: {risk_prob:.4f}")
        print(f"Ryzyko ataku serca: {risk}")

    except Exception as e:
        print(f"Wystąpił błąd: {e}")
        print("Upewnij się, że wprowadziłeś poprawne dane.")

# Uruchom program
if __name__ == "__main__":
    print("Predykcja ryzyka ataku serca przy użyciu wytrenowanego modelu.")
    while True:
        predict_heart_attack()
        again = input("\nCzy chcesz dokonać kolejnej predykcji? (Tak/Nie): ").lower()
        if again != "tak":
            break
    print("Dziękuję za skorzystanie z programu!")