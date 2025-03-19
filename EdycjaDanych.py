import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns


# 1. Wczytanie i Wstępna Analiza Danych

df = pd.read_csv("AtakSerca.csv")

# Tworzenie Podsumowania Danych

data_summary = f"""

<b>Liczba rekordów:</b> {len(df)}<br>

<b>Kolumny:</b> {', '.join(df.columns)}<br>

<b>Typy danych:</b><br>{df.dtypes.to_frame().to_html()}<br>

<b>Brakujące wartości:</b><br>{df.isnull().sum().to_frame().to_html()}

"""


# 2. Czyszczenie Danych

drop_cols = ["Patient ID", "Country"]

df = df.drop(columns=[col for col in drop_cols if col in df.columns])


num_cols = df.select_dtypes(include=[np.number]).columns

cat_cols = df.select_dtypes(exclude=[np.number]).columns


for col in num_cols:

    df[col] = df[col].fillna(df[col].median())


for col in cat_cols:

    df[col] = df[col].fillna(df[col].mode()[0])
    
    df[col] = pd.Categorical(df[col]).codes

df = df.drop_duplicates()


# Usuwanie kolumn o niskiej korelacji z "Heart Attack Risk"

threshold = 0.0057

if "Heart Attack Risk" in df.columns:

    corr_values = df.corr()["Heart Attack Risk"].abs()

    low_corr_cols = corr_values[corr_values < threshold].index

    print(f"Usunięto kolumny: {list(low_corr_cols)}")

    df = df.drop(columns=low_corr_cols)


# 3. Eksploracyjna Analiza danych (EDA)

# Generowanie Wykresów i Raportu EDA w Przyszłości

# Zapis Przygotowanego zbioru

df.to_csv("DanePrzeczyszczone.csv", index=False)