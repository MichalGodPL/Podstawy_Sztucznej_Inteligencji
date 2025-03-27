import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score, f1_score
from torch.utils.data import DataLoader, TensorDataset
from itertools import product

# Wczytywanie i przygotowanie danych
data = pd.read_csv('DanePrzeczyszczone.csv')
X = data.drop('Heart Attack Risk', axis=1).values
y = data['Heart Attack Risk'].values

df = pd.DataFrame(X)
df['target'] = y
df_majority = df[df['target'] == 0]
df_minority = df[df['target'] == 1]
df_minority_upsampled = resample(df_minority, replace=True, n_samples=len(df_majority), random_state=42)
df_balanced = pd.concat([df_majority, df_minority_upsampled])

X = df_balanced.drop('target', axis=1).values
y = df_balanced['target'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_temp, X_test, y_temp, y_test = train_test_split(X_scaled, y, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1111, random_state=42)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# Definicja modelu z elastyczną architekturą
class HeartDiseaseModel(nn.Module):
    def __init__(self, input_dim, hidden_sizes, dropout):
        super(HeartDiseaseModel, self).__init__()
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

# Funkcja treningu
def train_model(model, X_train_tensor, y_train_tensor, batch_size, epochs, optimizer, criterion, scheduler):
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    best_train_acc = 0
    for epoch in range(epochs):
        model.train()
        correct, total, epoch_loss = 0, 0, 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            predicted = (y_pred > 0).float()
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)
        train_accuracy = correct / total
        scheduler.step(epoch_loss)
        if train_accuracy > best_train_acc:
            best_train_acc = train_accuracy
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Train Acc: {train_accuracy*100:.2f}%")
    print(f"Najlepsza dokładność treningowa: {best_train_acc*100:.2f}%")
    return best_train_acc

# Funkcja walidacji
def validate_model(model, X_val_tensor, y_val_tensor):
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=len(X_val_tensor), shuffle=False)
    
    model.eval()
    with torch.no_grad():
        X_val, y_val = next(iter(val_loader))
        y_val_pred = model(X_val)
        val_predicted = (y_val_pred > 0).float()
        val_accuracy = (val_predicted == y_val).float().mean().item() * 100
    print(f"Dokładność walidacyjna: {val_accuracy:.2f}%")
    return val_accuracy

# Funkcja testowania
def test_model(model, X_test_tensor, y_test_tensor, batch_size):
    print("\n=== Etap 3: Testowanie modelu z najlepszymi hiperparametrami ===")
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    model.eval()
    with torch.no_grad():
        X_test, y_test = next(iter(test_loader))
        y_test_pred = model(X_test)
        y_test_pred = (y_test_pred > 0).float()
        
        accuracy = accuracy_score(y_test, y_test_pred)
        conf_matrix = confusion_matrix(y_test, y_test_pred)
        recall = recall_score(y_test, y_test_pred)
        precision = precision_score(y_test, y_test_pred)
        f1 = f1_score(y_test, y_test_pred)
        
        print(f"\nWyniki na zbiorze testowym:")
        print(f'Accuracy: {accuracy:.4f}')
        print(f'Confusion Matrix:\n{conf_matrix}')
        print(f'Recall: {recall:.4f}, Precision: {precision:.4f}, F1-score: {f1:.4f}')
        
        return accuracy, recall, precision, f1

# Automatyczne dostrajanie hiperparametrów
print("\n=== Rozpoczęcie automatycznego dostrajania hiperparametrów ===")
param_grid = {
    'lr': [0.01, 0.001, 0.0001],
    'dropout': [0, 0.2, 0.4],
    'batch_size': [64, 128, 256],
    'hidden_sizes': [
        [256, 128, 64, 32],  # Oryginalna architektura
        [128, 64, 32],       # Mniejsza sieć
        [512, 256, 128]      # Większa sieć
    ]
}

best_val_acc = 0
best_params = {}
max_attempts = len(param_grid['lr']) * len(param_grid['dropout']) * len(param_grid['batch_size']) * len(param_grid['hidden_sizes'])
attempt = 0

for lr, dropout, batch_size, hidden_sizes in product(param_grid['lr'], param_grid['dropout'], param_grid['batch_size'], param_grid['hidden_sizes']):
    attempt += 1
    print(f"\nPróba {attempt}/{max_attempts} - lr={lr}, dropout={dropout}, batch_size={batch_size}, hidden_sizes={hidden_sizes}")
    
    # Inicjalizacja modelu
    model = HeartDiseaseModel(input_dim=X_train.shape[1], hidden_sizes=hidden_sizes, dropout=dropout)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=False)
    pos_weight = torch.tensor([len(y_train) / sum(y_train)])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Etap 1: Trening
    print("--- Trening ---")
    best_train_acc = train_model(model, X_train_tensor, y_train_tensor, batch_size, epochs=50, optimizer=optimizer, criterion=criterion, scheduler=scheduler)
    torch.save(model.state_dict(), 'heart_disease_model_after_training.pth')
    print("Model po treningu zapisany jako 'heart_disease_model_after_training.pth'.")
    
    # Etap 2: Walidacja
    print("--- Walidacja ---")
    val_acc = validate_model(model, X_val_tensor, y_val_tensor)
    torch.save(model.state_dict(), 'heart_disease_model_after_validation.pth')
    print("Model po walidacji zapisany jako 'heart_disease_model_after_validation.pth'.")
    
    # Ocena różnicy i aktualizacja najlepszych parametrów
    diff = best_train_acc * 100 - val_acc
    print(f"Różnica trening-walidacja: {diff:.2f}%, Val Acc: {val_acc:.2f}%")
    
    if val_acc > best_val_acc and diff < 10:  # Próg 10% dla różnicy
        best_val_acc = val_acc
        best_params = {'lr': lr, 'dropout': dropout, 'batch_size': batch_size, 'hidden_sizes': hidden_sizes}
        torch.save(model.state_dict(), 'heart_disease_model_best.pth')
        print(f"Znaleziono lepsze parametry: lr={lr}, dropout={dropout}, batch_size={batch_size}, hidden_sizes={hidden_sizes}, val_acc={val_acc:.2f}%")
    
    # Wczesne zatrzymanie
    if diff < 5 and val_acc > 85:  # Można dostosować próg
        print("Wyniki są zadowalające, przerywam dostrajanie.")
        break

print(f"\nNajlepsze hiperparametry: {best_params}, val_acc={best_val_acc:.2f}%")

# Wczytanie najlepszego modelu
model = HeartDiseaseModel(input_dim=X_train.shape[1], hidden_sizes=best_params['hidden_sizes'], dropout=best_params['dropout'])
model.load_state_dict(torch.load('heart_disease_model_best.pth'))

# Etap 3: Testowanie
accuracy, recall, precision, f1 = test_model(model, X_test_tensor, y_test_tensor, best_params['batch_size'])

# Zapis metryk i modelu
metrics = {
    'accuracy': str(round(accuracy, 4)),
    'precision': str(round(precision, 4)),
    'recall': str(round(recall, 4)),
    'f1_score': str(round(f1, 4))
}
with open('metrics.pkl', 'wb') as f:
    pickle.dump(metrics, f)

torch.save(model.state_dict(), 'heart_disease_model_final.pth')
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("\nOstateczny model, scaler i metryki zapisane jako 'heart_disease_model_final.pth', 'scaler.pkl' i 'metrics.pkl'.")