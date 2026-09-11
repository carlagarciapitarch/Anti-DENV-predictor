#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 12:21:25 2026

@author: carla
"""

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold

from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, matthews_corrcoef, roc_curve, auc
)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier
)

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from xgboost import XGBClassifier


# =========================
# CARGA DE DATOS
# =========================

df = pd.read_excel("datos_finales_molpeq_para_ML_0,6_pearson.xlsx")

X = df.drop(columns=["Actividad(0/1)"])
y = df["Actividad(0/1)"]
#Recordar que 1 es activo y 0 inactivo

# =========================
# SPLIT: TRAIN / TEST
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# =========================
# MÉTRICAS
# =========================

def evaluate_model(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "specificity": tn / (tn + fp),
        "sensitivity": tp / (tp + fn),
        "mcc": matthews_corrcoef(y_true, y_pred)
    }

# =========================
# CV CONFIG
# =========================

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scoring = {
    "roc_auc": "roc_auc",
    "f1": "f1",
    "mcc": "matthews_corrcoef"
}

# =========================
# MODELOS
# =========================

models = {

    # "RF": (
    #     Pipeline([
    #         ("scaler", StandardScaler()),
    #         ("model", RandomForestClassifier(class_weight="balanced"))
    #     ]),
    #     {
    #         "model__n_estimators": [100, 300],
    #         "model__max_depth": [None, 10],
    #     }
    # ),

    "SVM": (
        Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(probability=True, class_weight="balanced"))
        ]),
        {
            "model__kernel": ["linear", "rbf", "poly", "sigmoid"],
            "model__C": [0.1, 1, 10],
            "model__gamma": ["scale", "auto"]
        }
    )
    
    #),


    # "SVM": (
    #     Pipeline([
    #         ("scaler", StandardScaler()),
    #         ("model", SVC(probability=True, class_weight="balanced"))
    #     ]),
    #     {
    #         "model__C": [0.1, 1, 10],
    #     }
    # ),

    # "KNN": (
    #     Pipeline([
    #         ("scaler", StandardScaler()),
    #         ("model", KNeighborsClassifier())
    #     ]),
    #     {
    #         "model__n_neighbors": [3, 5, 7],
    #     }
    # ),

    # "NB": (
    #     Pipeline([
    #         ("scaler", StandardScaler()),
    #         ("model", GaussianNB())
    #     ]),
    #     {}
    # ),

    # "LR": (
    #     Pipeline([
    #         ("scaler", StandardScaler()),
    #         ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))
    #     ]),
    #     {
    #         "model__C": [0.1, 1, 10],
    #     }
    # ),

    # "AdaBoost": (
    #     Pipeline([
    #         ("scaler", StandardScaler()),
    #         ("model", AdaBoostClassifier())
    #     ]),
    #     {
    #         "model__n_estimators": [50, 100, 200],
    #         "model__learning_rate": [0.01, 0.1, 1],
    #     }
    # ),

    # "GB": (
    #     Pipeline([
    #         ("scaler", StandardScaler()),
    #         ("model", GradientBoostingClassifier())
    #     ]),
    #     {
    #         "model__n_estimators": [100, 300],
    #         "model__learning_rate": [0.01, 0.1],
    #         "model__max_depth": [3, 5],
    #     }
    # ),

    # "ExtraTrees": (
    #     Pipeline([
    #         ("scaler", StandardScaler()),
    #         ("model", ExtraTreesClassifier(class_weight="balanced"))
    #     ]),
    #     {
    #         "model__n_estimators": [100, 300],
    #         "model__max_depth": [None, 10],
    #     }
    # ),

    # "MLP": (
    #     Pipeline([
    #         ("scaler", StandardScaler()),
    #         ("model", MLPClassifier(max_iter=1000))
    #     ]),
    #     {
    #         "model__hidden_layer_sizes": [(50,), (100,)],
    #     }
    # ),

    # "XGB": (
    #     Pipeline([
    #         ("scaler", StandardScaler()),
    #         ("model", XGBClassifier(eval_metric='logloss'))
    #     ]),
    #     {
    #         "model__n_estimators": [100, 300],
    #         "model__max_depth": [3, 6],
    #         "model__learning_rate": [0.01, 0.1],
    #     }
    # )
}

# =========================
# GRID SEARCH
# =========================

results_test = {}
best_models = {}
best_params = {}

with pd.ExcelWriter("gridsearch_resultados.xlsx") as writer:

    for name, (pipe, params) in models.items():
        print(f"Entrenando {name}...")
        
        grid = GridSearchCV(
            pipe,
            params,
            cv=cv,
            scoring=scoring,
            refit="roc_auc",
            n_jobs=-1
        )

        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_
        best_models[name] = best_model
        best_params[name] = grid.best_params_

        # Guardar CV completo
        df_grid = pd.DataFrame(grid.cv_results_)
        df_grid.to_excel(writer, sheet_name=name[:31], index=False)

        # Evaluación en TEST
        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1]

        results_test[name] = evaluate_model(y_test, y_pred, y_prob)

# =========================
# RESULTADOS
# =========================

df_results = pd.DataFrame(results_test).T
df_results = df_results.sort_values(by="roc_auc", ascending=False)

df_results.to_excel("resultados_test.xlsx")
pd.DataFrame(best_params).T.to_excel("mejores_parametros.xlsx")

print("\nResultados en TEST:")
print(df_results)

# Mejor modelo
best_model_name = df_results.index[0]
print("\nMejor modelo:", best_model_name)

# =========================
# ROC GLOBAL
# =========================

plt.figure(figsize=(8,6))

for name, model in best_models.items():
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    
    plt.plot(fpr, tpr, label=f"{name} (AUC={results_test[name]['roc_auc']:.3f})")

plt.plot([0,1], [0,1], linestyle="--", color="gray")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves - Test Set")
plt.legend()
plt.grid()
plt.savefig("ROC_global.png", dpi=150)
plt.show()

print("Figura guardada: ROC_global.png")

# =========================
# ROC INDIVIDUALES
# =========================

for name, model in best_models.items():
    plt.figure(figsize=(7,5))
    
    # TRAIN
    y_prob_train = model.predict_proba(X_train)[:, 1]
    fpr_train, tpr_train, _ = roc_curve(y_train, y_prob_train)
    auc_train = auc(fpr_train, tpr_train)
    
    # TEST
    y_prob_test = model.predict_proba(X_test)[:, 1]
    fpr_test, tpr_test, _ = roc_curve(y_test, y_prob_test)
    auc_test = auc(fpr_test, tpr_test)
    
    plt.plot(fpr_train, tpr_train, label=f"Train (AUC={auc_train:.3f})")
    plt.plot(fpr_test, tpr_test, label=f"Test (AUC={auc_test:.3f})")
    
    plt.plot([0,1], [0,1], linestyle="--", color="gray")
    
    plt.title(f"ROC - {name}")
    plt.legend()
    plt.grid()
    
    filename = f"ROC_{name}.png"
    plt.savefig(filename, dpi=150, bbox_inches="tight")
   
    plt.show()
    plt.close()
   
print(f"Guardado: {filename}")



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
df_screening = pd.read_excel("DATASET_SCREENING_FINAL.xlsx")

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




