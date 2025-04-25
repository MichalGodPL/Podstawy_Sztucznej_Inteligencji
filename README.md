<<<<<<< HEAD
<div align="center">

# Heart Attack Prediction with PyTorch AI Model

</div>

<div align="center">

=======
# Heart Attack Prediction with PyTorch AI Model

>>>>>>> c232322dcf6c45a3e448e3cf5ba6dc1c2287bf48
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)

</div>

---

## 🚀 Overview

This repository hosts a **PyTorch-based AI model** for predicting heart attack risk using medical data. It includes comprehensive data preprocessing, model training, evaluation, and a sleek web interface built with **Flask** and **PyWebView**. Available in **Polish** and **English**, this project is perfect for researchers, developers, and healthcare professionals exploring AI-driven solutions.

---

## Overview

This repository features a **PyTorch-based AI model** designed to predict the risk of a heart attack using medical data. The project includes robust data preprocessing, model training, evaluation, and a user-friendly web interface built with **Flask** and **PyWebView**. Available in both **Polish** and **English**, this project is ideal for researchers, developers, and medical professionals exploring AI-driven healthcare solutions.

---

## 🇵🇱 Wersja Polska

<<<<<<< HEAD
### 🛠 Cechy Projektu

- **Przetwarzanie Danych**: Obsługuje brakujące wartości, skalowanie cech i normalizację danych z użyciem **Pandas** i **Scikit-learn**.

- **Model AI**: Sieć neuronowa w **PyTorch** do binarnej klasyfikacji ryzyka ataku serca.

- **Ocena Modelu**: Wizualizuje wyniki za pomocą metryk (dokładność, precyzja, czułość) oraz wykresów (macierz pomyłek, krzywa ROC).

- **Interfejs Użytkownika**: Intuicyjny interfejs webowy oparty na **Flask** i **PyWebView** do wprowadzania danych i przeglądania predykcji.

### 📋 Wymagania

- Python 3.8 lub nowszy

- Zainstalowany **Git** do klonowania repozytorium

- Pliki: `model.pth`, `scaler.pkl`, `metrics.pkl` (jeśli dostępne)

### 🔧 Instalacja
=======
### Cechy Projektu

- **Przetwarzanie Danych**: Obsługa brakujących wartości, skalowanie cech i normalizacja danych z użyciem **Pandas** i **Scikit-learn**.

- **Model AI**: Sieć neuronowa w **PyTorch** do binarnej klasyfikacji ryzyka ataku serca.

- **Ocena Modelu**: Wizualizacja wyników za pomocą metryk (dokładność, precyzja, czułość) oraz wykresów (macierz pomyłek, krzywa ROC).

- **Interfejs Użytkownika**: Intuicyjny interfejs webowy oparty na **Flask** i **PyWebView** do wprowadzania danych i przeglądania predykcji.

### Wymagania

- Python 3.8 lub nowszy

- Zainstalowany **Git** do klonowania repozytorium

- Pliki: `model.pth`, `scaler.pkl`, `metrics.pkl` (jeśli dostępne)

### Instalacja
>>>>>>> c232322dcf6c45a3e448e3cf5ba6dc1c2287bf48

1. Sklonuj repozytorium:
   ```bash
   git clone https://github.com/MichalGodPL/Podstawy_Sztucznej_Inteligencji.git
   ```

2. Przejdź do katalogu projektu:
   ```bash
   cd Podstawy_Sztucznej_Inteligencji
   ```

3. Zainstaluj zależności:
   ```bash
   pip install torch numpy pandas scikit-learn matplotlib seaborn flask pywebview
   ```

<<<<<<< HEAD
### ▶️ Uruchamianie
=======
### Uruchamianie
>>>>>>> c232322dcf6c45a3e448e3cf5ba6dc1c2287bf48

1. Upewnij się, że pliki `model.pth`, `scaler.pkl` i `metrics.pkl` znajdują się w katalogu projektu.

2. Uruchom skrypt główny:
   ```bash
   python Main.py
   ```

3. Otwórz interfejs webowy, aby wprowadzić dane pacjenta i zobaczyć predykcje.

<<<<<<< HEAD
### ℹ️ Uwagi
=======
### Uwagi
>>>>>>> c232322dcf6c45a3e448e3cf5ba6dc1c2287bf48

- Jeśli brak wymaganych plików, konieczne może być wytrenowanie modelu. Sprawdź repozytorium w poszukiwaniu skryptów treningowych.

---

## 🇬🇧 English Version

<<<<<<< HEAD
### 🛠 Project Features
=======
### Project Features
>>>>>>> c232322dcf6c45a3e448e3cf5ba6dc1c2287bf48

- **Data Preprocessing**: Handles missing values, feature scaling, and normalization using **Pandas** and **Scikit-learn**.

- **AI Model**: A **PyTorch** neural network for binary classification of heart attack risk.

- **Model Evaluation**: Visualizes performance with metrics (accuracy, precision, recall) and plots (confusion matrix, ROC curve).

- **Web Interface**: A user-friendly GUI built with **Flask** and **PyWebView** for inputting data and viewing predictions.
<<<<<<< HEAD

### 📋 Prerequisites

=======

### Prerequisites

>>>>>>> c232322dcf6c45a3e448e3cf5ba6dc1c2287bf48
- Python 3.8 or higher

- **Git** installed for cloning the repository

- Required files: `model.pth`, `scaler.pkl`, and `metrics.pkl` (if available)

<<<<<<< HEAD
### 🔧 Installation
=======
### Installation
>>>>>>> c232322dcf6c45a3e448e3cf5ba6dc1c2287bf48

1. Clone the repository:
   ```bash
   git clone https://github.com/MichalGodPL/Podstawy_Sztucznej_Inteligencji.git
   ```

2. Navigate to the project directory:
   ```bash
   cd Podstawy_Sztucznej_Inteligencji
   ```

3. Install dependencies:
   ```bash
   pip install torch numpy pandas scikit-learn matplotlib seaborn flask pywebview
   ```

<<<<<<< HEAD
### ▶️ Usage
=======
### Usage
>>>>>>> c232322dcf6c45a3e448e3cf5ba6dc1c2287bf48

1. Ensure `model.pth`, `scaler.pkl`, and `metrics.pkl` are in the project directory.

2. Run the main script:
   ```bash
   python Main.py
   ```

3. Access the web interface to input patient data and view predictions.

<<<<<<< HEAD
### ℹ️ Notes
=======
### Notes
>>>>>>> c232322dcf6c45a3e448e3cf5ba6dc1c2287bf48

- If required files are missing, you may need to train the model. Check the repository for training scripts.

---

<<<<<<< HEAD
## 📸 Screenshots

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; padding: 20px;">
  <img src="1.png" style="border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); width: 100%;">
  <img src="2.png" style="border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); width: 100%;">
  <img src="3.png" style="border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); width: 100%;">
  <img src="4.png" style="border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); width: 100%;">
  <img src="5.png" style="border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); width: 100%;">
  <img src="6.png" style="border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); width: 100%;">
</div>

---
=======
## Screenshots

| ![Screenshot 1](1.png) | ![Screenshot 2](2.png) |
|------------------------|------------------------|
| ![Screenshot 3](3.png) | ![Screenshot 4](4.png) |
| ![Screenshot 5](5.png) |                        |

---
>>>>>>> c232322dcf6c45a3e448e3cf5ba6dc1c2287bf48
