#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 09:34:53 2026

@author: carla
"""

import pandas as pd
from sklearn.model_selection import train_test_split

# Archivos
excel_6280 = "mol_peq_con_descriptores_5uM.xlsx"
excel_6281 = "mol_peq_sin_duplicados.xlsx"

# Leer los Excel
df_6280 = pd.read_excel(excel_6280)
df_6281 = pd.read_excel(excel_6281)

# Encontrar los SMILES que sobran en el segundo archivo
smiles_extra = set(df_6281["SMILES_canonico"]) - set(df_6280["SMILES_canonico"])

print("SMILES que sobran:")
print(smiles_extra)

if len(smiles_extra) != 1:
    print(f"Atención: se encontraron {len(smiles_extra)} SMILES distintos.")
else:
    print(f"El SMILES extra es: {next(iter(smiles_extra))}")
    
    
# Eliminar las filas cuyo SMILES está de más
df_6281_filtrado = df_6281[
    ~df_6281["SMILES_canonico"].isin(smiles_extra)
]

# Guardar el resultado
df_6281_filtrado.to_excel("archivo_6281_filtrado.xlsx", index=False)

print(f"Filas originales: {len(df_6281)}")
print(f"Filas finales: {len(df_6281_filtrado)}")


# =========================
# CARGA DE DATOS
# =========================


X = df_6281_filtrado
y = df_6281_filtrado["IC50/EC50(microM)"]


# =========================
# SPLIT: TRAIN / TEST
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#RECUPERAR DATASETS TRAIN Y TEST

train_set = X_train.copy()
#train_set["IC50/EC50(microM)"] = y_train

test_set = X_test.copy()
#test_set["IC50/EC50(microM)"] = y_test

train_set.to_excel("dataset_train.xlsx", index=False)
test_set.to_excel("dataset_test.xlsx", index=False)