import subprocess

import webview

import threading

import time


# Function to run scripts

def run_scripts():

    print("Maszyna w Ogniu...")

    start_time = time.time()


    try:

        print("Uruchamianie EdycjaDanych.py...")

        subprocess.run(["python", "EdycjaDanych.py"], check=True)


        print("Uruchamianie PyTorch.py...")

        subprocess.run(["python", "PyTorch.py"], check=True)


    except subprocess.CalledProcessError as e:

        print(f"Błąd w skrypcie: {e}")


    end_time = time.time()

    elapsed_time = end_time - start_time

    print(f"Maszyna na OIOMie! Całkowity Czas: {elapsed_time:.2f} Sekundy.")


# Uruchamianie skryptów w osobnym wątku, żeby nie blokować WebView

def start_scripts():

    threading.Thread(target=run_scripts, daemon=True).start()


# Tworzenie okna WebView w rozdzielczości 1600x900

webview.create_window("Status SI", "index.html", width=1600, height=900)

webview.start(start_scripts)