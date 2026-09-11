#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 12:21:25 2026

@author: carla
"""

# =========================
# INTERPRETACIÓN DEL MODELO (SHAP)
# =========================

import shap
import numpy as np

print(f'Calculando SHAP para el modelo: {best_model_name}...')

# =========================
# PREPARAR DATOS
# =========================

# Si usas pipeline con scaler:
if hasattr(best_model, "named_steps") and "scaler" in best_model.named_steps:
    X_shap = best_model.named_steps["scaler"].transform(X_train)
else:
    X_shap = X_train.copy()

FEATURES = X_train.columns

# =========================
# SHAP SEGÚN MODELO
# =========================

if best_model_name in ["RF", "XGB", "GB", "ExtraTrees"]:
    
    # Extraer modelo real del pipeline
    model = best_model.named_steps["model"] if "model" in best_model.named_steps else best_model
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_shap)
    
    # Para clasificación binaria
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

else:
    # Para modelos tipo SVM, KNN, LR, etc.
    
    # Reducir tamaño por coste computacional
    X_sample = X_shap[:100]
    
    explainer = shap.KernelExplainer(
        best_model.predict_proba,
        X_sample
    )
    
    shap_values = explainer.shap_values(X_sample)
    
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    
    X_shap = X_sample

print("SHAP calculado")

# =========================
# PLOT
# =========================

plt.figure(figsize=(10, 8))

shap.summary_plot(
    shap_values,
    X_shap,
    feature_names=FEATURES,
    max_display=20,
    show=False
)

plt.title(f"SHAP Summary - {best_model_name}")
plt.tight_layout()
plt.savefig("shap_summary.png", dpi=150)
plt.show()

print("Figura guardada: shap_summary.png")

# =========================
# IMPORTANCIA GLOBAL
# =========================

importance_df = pd.DataFrame({
    "descriptor": FEATURES,
    "mean_abs_shap": np.abs(shap_values).mean(axis=0)
}).sort_values(by="mean_abs_shap", ascending=False)

importance_df.to_excel("shap_importance.xlsx", index=False, engine="openpyxl")

print("\nTop 10 descriptores más importantes:")
print(importance_df.head(10))

#df_prueba = df.to_excel("df_prueba.xlsx", engine="openpyxl")


# =========================
# Applicability Domain DURANTE ENTRENAMIENTO PARA CONSTRUIR DOMINIO DE APLICABILIDAD
# =========================
#kNN-AD
# Una molécula está dentro del dominio si está “cerca” de moléculas del training set.

from sklearn.neighbors import NearestNeighbors
import numpy as np

# =========================
# 1. PREPARAR TRAINING SET
# =========================

if "scaler" in best_model.named_steps:
    X_train_ad = best_model.named_steps["scaler"].transform(X_train)
else:
    X_train_ad = X_train.values


# =========================
# 2. FIT DEL MODELO kNN
# =========================

nn = NearestNeighbors(n_neighbors=5)
nn.fit(X_train_ad)

# =========================
# 3. DISTANCIAS EN TRAINING (para definir umbral)
# =========================

distances_train, _ = nn.kneighbors(X_train_ad)

# media de la distancia a vecinos
mean_dist_train = distances_train.mean(axis=1)

# umbral (regla clásica)
threshold = mean_dist_train.mean() + 2 * mean_dist_train.std()

print(f"Threshold AD: {threshold:.4f}")


# =========================
# 4. SCREENING CON NUEVAS MOLÉCULAS
# =========================

import pandas as pd
from rdkit import Chem
from mordred import Calculator, descriptors

# Cargar datos
df_screening = pd.read_excel("protease_inhibitor.xlsx")
df_screening["SMILES"] = df_screening["SMILES"].str.replace('"', '', regex=False)

# Convertir SMILES → moléculas RDKit
df_screening["RDKit"] = df_screening["SMILES"].apply(
    lambda x: Chem.MolFromSmiles(x) if isinstance(x, str) else None
)

# Eliminar SMILES inválidos
df_screening = df_screening[df_screening["RDKit"].notna()].copy()

# Inicializar calculadora
calc = Calculator(descriptors)

# Calcular descriptores 
desc = calc.map(df_screening["RDKit"])
desc = pd.DataFrame([d.asdict() for d in desc])
desc = desc.apply(pd.to_numeric, errors="coerce")
desc = desc.replace([np.inf, -np.inf], np.nan)

# Unir resultados
df_screening_mordred = pd.concat(
    [df_screening.reset_index(drop=True), desc.reset_index(drop=True)],
    axis=1
)

FEATURES = X_train.columns.tolist()
print(f"Número de features: {len(FEATURES)}")

# asegurar mismas columnas
X_screen = df_screening_mordred.reindex(columns=FEATURES)
X_screen = X_screen.fillna(0) #por si hay algun valor con na

y_prob = best_model.predict_proba(X_screen)[:, 1]


if "scaler" in best_model.named_steps:
    X_screen_ad = best_model.named_steps["scaler"].transform(X_screen)
else:
    X_screen_ad = X_screen.values


distances_screen, _ = nn.kneighbors(X_screen_ad)

mean_dist_screen = distances_screen.mean(axis=1)

# =========================
# 5. DEFINIR DOMINIO
# =========================

AD_screen = mean_dist_screen < threshold

print("Dentro del dominio:", AD_screen.sum())
print("Fuera del dominio:", (~AD_screen).sum())


final_hits = (y_prob > 0.8) & (AD_screen)



# =========================
# AÑADIR RESULTADOS
# =========================

df_screening["prob_activo"] = y_prob
df_screening["AD"] = AD_screen
df_screening["hit"] = final_hits

# =========================
# GUARDAR TODO EL SCREENING
# =========================

df_screening.to_excel("screening_resultados_completo.xlsx", index=False)

# =========================
# GUARDAR SOLO LOS HITS
# =========================

df_hits = df_screening[df_screening["hit"]].copy()

df_hits = df_hits.sort_values(by="prob_activo", ascending=False)

df_hits.to_excel("hits_finales.xlsx", index=False)

print("Número de hits:", len(df_hits))