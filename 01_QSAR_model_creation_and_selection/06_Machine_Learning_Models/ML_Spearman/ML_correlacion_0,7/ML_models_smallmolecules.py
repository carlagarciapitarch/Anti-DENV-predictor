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

df = pd.read_excel("datos_finales_molpeq_para_ML_0,7_spearman.xlsx")

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

from sklearn.metrics import confusion_matrix, make_scorer

def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scoring = {
    "accuracy": "accuracy",
    "roc_auc": "roc_auc",
    "f1": "f1",
    "mcc": "matthews_corrcoef",
    "sensitivity": "recall",
    "specificity": make_scorer(specificity_score)
}

# =========================
# MODELOS
# =========================

models = {

    "RF": (
        Pipeline([
            ("scaler", StandardScaler()),
            ("model", RandomForestClassifier(class_weight="balanced"))
        ]),
        {
            "model__n_estimators": [100, 300],
            "model__max_depth": [None, 10],
        }
    ),

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
    ),


    # "SVM": (
    #     Pipeline([
    #         ("scaler", StandardScaler()),
    #         ("model", SVC(probability=True, class_weight="balanced"))
    #     ]),
    #     {
    #         "model__C": [0.1, 1, 10],
    #     }
    # ),

    "KNN": (
        Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier())
        ]),
        {
            "model__n_neighbors": [3, 5, 7],
        }
    ),

    "NB": (
        Pipeline([
            ("scaler", StandardScaler()),
            ("model", GaussianNB())
        ]),
        {}
    ),

    "LR": (
        Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))
        ]),
        {
            "model__C": [0.1, 1, 10],
        }
    ),

    "AdaBoost": (
        Pipeline([
            ("scaler", StandardScaler()),
            ("model", AdaBoostClassifier())
        ]),
        {
            "model__n_estimators": [50, 100, 200],
            "model__learning_rate": [0.01, 0.1, 1],
        }
    ),

    "GB": (
        Pipeline([
            ("scaler", StandardScaler()),
            ("model", GradientBoostingClassifier())
        ]),
        {
            "model__n_estimators": [100, 300],
            "model__learning_rate": [0.01, 0.1],
            "model__max_depth": [3, 5],
        }
    ),

    "ExtraTrees": (
        Pipeline([
            ("scaler", StandardScaler()),
            ("model", ExtraTreesClassifier(class_weight="balanced"))
        ]),
        {
            "model__n_estimators": [100, 300],
            "model__max_depth": [None, 10],
        }
    ),

    "MLP": (
        Pipeline([
            ("scaler", StandardScaler()),
            ("model", MLPClassifier(max_iter=1000))
        ]),
        {
            "model__hidden_layer_sizes": [(50,), (100,)],
        }
    ),

    "XGB": (
        Pipeline([
            ("scaler", StandardScaler()),
            ("model", XGBClassifier(eval_metric='logloss'))
        ]),
        {
            "model__n_estimators": [100, 300],
            "model__max_depth": [3, 6],
            "model__learning_rate": [0.01, 0.1],
        }
    )
}

# =========================
# GRID SEARCH
# =========================

results_train = {}
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


        # TRAIN
        y_pred_train = best_model.predict(X_train)
        y_prob_train = best_model.predict_proba(X_train)[:, 1]
        
        results_train[name] = evaluate_model(
            y_train,
            y_pred_train,
            y_prob_train
        )
        
        # TEST
        y_pred_test = best_model.predict(X_test)
        y_prob_test = best_model.predict_proba(X_test)[:, 1]
        
        results_test[name] = evaluate_model(
            y_test,
            y_pred_test,
            y_prob_test
        )

# =========================
# RESULTADOS
# =========================

df_train = pd.DataFrame(results_train).T
df_test = pd.DataFrame(results_test).T

df_train = df_train.sort_values(by="roc_auc", ascending=False)
df_test = df_test.sort_values(by="roc_auc", ascending=False)

df_train.to_excel("resultados_train.xlsx")
df_test.to_excel("resultados_test.xlsx")

pd.DataFrame(best_params).T.to_excel("mejores_parametros.xlsx")



# Mejor modelo
best_model_name = df_test.index[0]
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

