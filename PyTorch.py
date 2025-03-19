import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

# Wczytywanie danych
data = pd.read_csv('DanePrzeczyszczone.csv')

# Przygotowanie danych
X = data.drop('Heart Attack Risk', axis=1).values
y = data['Heart Attack Risk'].values

# **Oversampling klasy 1**
df = pd.DataFrame(X)
df['target'] = y
df_majority = df[df['target'] == 0]
df_minority = df[df['target'] == 1]

df_minority_upsampled = resample(df_minority, replace=True, n_samples=len(df_majority), random_state=42)
df_balanced = pd.concat([df_majority, df_minority_upsampled])

X = df_balanced.drop('target', axis=1).values
y = df_balanced['target'].values

# **Standaryzacja**
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# **Podział na zbiory treningowe i testowe**
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# **Konwersja na tensory**
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# **Tworzenie DataLoaderów**
batch_size = 128
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=len(X_test_tensor), shuffle=False)

# **Nowa architektura sieci**
class FinalHeartDiseaseModel(nn.Module):
    def __init__(self, input_dim):
        super(FinalHeartDiseaseModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)

        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)

        self.fc4 = nn.Linear(64, 32)
        self.bn4 = nn.BatchNorm1d(32)

        self.output = nn.Linear(32, 1)

        self.dropout = 0.3

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.fc1(x)))
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = F.leaky_relu(self.bn2(self.fc2(x)))
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = F.leaky_relu(self.bn3(self.fc3(x)))
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = F.leaky_relu(self.bn4(self.fc4(x)))
        x = F.dropout(x, p=self.dropout, training=self.training)

        return self.output(x)  # **Sigmoid w BCEWithLogitsLoss**

# **Tworzenie modelu**
model = FinalHeartDiseaseModel(input_dim=X_train.shape[1])

# **Używamy SGD zamiast Adam**
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# **Waga strat dla klasy 1**
pos_weight = torch.tensor([len(y_train) / sum(y_train)])

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# **Trening modelu**
epochs = 100
train_accuracy_list = []
test_accuracy_list = []

for epoch in range(epochs):
    model.train()
    correct = 0
    total = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        
        predicted = (y_pred > 0).float()
        correct += (predicted == y_batch).sum().item()
        total += y_batch.size(0)

    train_accuracy = correct / total
    train_accuracy_list.append(train_accuracy * 100)

    # **Testowanie modelu**
    model.eval()
    with torch.no_grad():
        X_test, y_test = next(iter(test_loader))
        y_test_pred = model(X_test)
        test_predicted = (y_test_pred > 0).float()
        test_accuracy = (test_predicted == y_test).float().mean().item() * 100
        test_accuracy_list.append(test_accuracy)

    if (epoch + 1) % 10 == 0:  # Drukujemy co 10 epok
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Train Acc: {train_accuracy*100:.2f}%, Test Acc: {test_accuracy:.2f}%")

# **Wykres dokładności**
import matplotlib.pyplot as plt
plt.figure(figsize=(12,6))
plt.plot(train_accuracy_list, 'o--', label='Train Accuracy')
plt.plot(test_accuracy_list, 'o--', label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# **Ocena modelu**
model.eval()
with torch.no_grad():
    y_test_pred = model(X_test_tensor)
    y_test_pred = (y_test_pred > 0).float()

accuracy = accuracy_score(y_test_tensor, y_test_pred)
conf_matrix = confusion_matrix(y_test_tensor, y_test_pred)
recall = recall_score(y_test_tensor, y_test_pred)
precision = precision_score(y_test_tensor, y_test_pred)
f1 = f1_score(y_test_tensor, y_test_pred)

print(f'Accuracy: {accuracy:.4f}')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Recall: {recall:.4f}, Precision: {precision:.4f}, F1-score: {f1:.4f}')
