import torch
import torch.nn as nn
import pickle

# Definicja modelu
class ImprovedHeartDiseaseModel(nn.Module):
    def __init__(self, input_dim=18):  # Zmiana na 18 cech
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

# Ładowanie modelu i skalera
input_dim = 18  # Ustawiamy na 18, aby pasowało do zapisanego modelu
model = ImprovedHeartDiseaseModel(input_dim=input_dim)
model.load_state_dict(torch.load('heart_disease_model.pth'))
model.eval()

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)