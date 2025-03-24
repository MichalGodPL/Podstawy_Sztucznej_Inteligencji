import torch
import torch.nn as nn
import pickle
import numpy as np

# Definicja modelu zgodna z zapisanym modelem
class ImprovedHeartDiseaseModel(nn.Module):
    def __init__(self, input_dim=18):
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
        self.dropout = 0.3                   # Możesz dostosować dropout
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')  # Zgodne z poprzednią poprawką
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

# Ładowanie modelu i skalera
input_dim = 18
model = ImprovedHeartDiseaseModel(input_dim=input_dim)
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

# Przykład użycia (opcjonalny)
if __name__ == "__main__":
    new_data = np.random.rand(1, 18)  # Losowe dane testowe
    probabilities, predictions = predict_risk(new_data, model, scaler)
    print(f"Prawdopodobieństwo zawału: {probabilities[0]:.4f}")
    print(f"Predykcja (0=brak ryzyka, 1=ryzyko): {int(predictions[0])}")