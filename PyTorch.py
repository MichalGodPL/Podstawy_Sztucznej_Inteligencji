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

batch_size = 256
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=len(X_val_tensor), shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=len(X_test_tensor), shuffle=False)

# Definicja modelu
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

# Inicjalizacja modelu i hiperparametrów
model = ImprovedHeartDiseaseModel(input_dim=X_train.shape[1])
optimizer = optim.Adam(model.parameters(), lr=0.01)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
pos_weight = torch.tensor([len(y_train) / sum(y_train)])
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# Etap 1: Trening modelu
def train_model(model, train_loader, epochs, optimizer, criterion, scheduler):
    best_train_acc = 0
    best_epoch = 0
    train_accuracy_list = []

    print("\n=== Etap 1: Trening modelu ===")
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
        train_accuracy_list.append(train_accuracy * 100)

        scheduler.step(epoch_loss)

        if train_accuracy > best_train_acc:
            best_train_acc = train_accuracy
            best_epoch = epoch + 1

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Train Acc: {train_accuracy*100:.2f}%")

    print(f"\nNajlepsza dokładność na zbiorze treningowym: {best_train_acc*100:.2f}% w epoce {best_epoch}")
    print("Sugestie hiperparametrów po treningu: lr=0.01, batch_size=256, dropout=0 (obecne wartości).")
    return best_train_acc

# Trening
epochs = 100
best_train_acc = train_model(model, train_loader, epochs, optimizer, criterion, scheduler)
torch.save(model.state_dict(), 'heart_disease_model_after_training.pth')
print("Model po treningu zapisany jako 'heart_disease_model_after_training.pth'.")

# Etap 2: Walidacja modelu
def validate_model(model, val_loader, best_train_acc):
    print("\n=== Etap 2: Walidacja modelu ===")
    model.eval()
    with torch.no_grad():
        X_val, y_val = next(iter(val_loader))
        y_val_pred = model(X_val)
        val_predicted = (y_val_pred > 0).float()
        val_accuracy = (val_predicted == y_val).float().mean().item() * 100

    print(f"Dokładność na zbiorze walidacyjnym: {val_accuracy:.2f}%")
    print(f"\nPodsumowanie po treningu i walidacji:")
    print(f"Najlepsza dokładność treningowa: {best_train_acc*100:.2f}%")
    print(f"Dokładność walidacyjna: {val_accuracy:.2f}%")
    if val_accuracy < best_train_acc * 100 - 10:  # Próg 10% różnicy
        print("Sugestie: Overfitting wykryty. Zwiększ dropout (np. 0.2) lub zmniejsz lr (np. 0.001).")
    else:
        print("Sugestie: Hiperparametry wyglądają dobrze (lr=0.01, dropout=0).")
    return val_accuracy

# Walidacja
best_val_acc = validate_model(model, val_loader, best_train_acc)
torch.save(model.state_dict(), 'heart_disease_model_after_validation.pth')
print("Model po walidacji zapisany jako 'heart_disease_model_after_validation.pth'.")

# Etap 3: Testowanie modelu
def test_model(model, test_loader):
    print("\n=== Etap 3: Testowanie modelu ===")
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

# Testowanie
accuracy, recall, precision, f1 = test_model(model, test_loader)

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