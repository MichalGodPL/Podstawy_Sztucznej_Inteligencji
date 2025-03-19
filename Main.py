import subprocess

import time


# Powiadomienie o rozpoczęciu działania SI

print("Maszyna w Ogniu...")


# Rozpoczęcie licznika czasu

start_time = time.time()


# Uruchomienie skryptu EdycjaDanych.py

print("Uruchamianie EdycjaDanych.py...")

subprocess.run(["python", "EdycjaDanych.py"], check=True)


# Uruchomienie skryptu PyTorch.py

print("Uruchamianie PyTorch.py...")

subprocess.run(["python", "PyTorch.py"], check=True)


# Zakończenie licznika czasu

end_time = time.time()

elapsed_time = end_time - start_time


# Wyświetlenie czasu działania

print(f"Maszyna na OIOMie! Całkowity Czas: {elapsed_time:.2f} Sekundy.")