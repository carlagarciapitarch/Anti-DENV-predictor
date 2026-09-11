#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SHAP + Applicability Domain + Screening para un modelo SVM guardado con joblib.

Estructura esperada del .pkl:

joblib.dump(
    {
        "model": best_models["SVM"],
        "features": X.columns.tolist(),
        "X_train": X_train
        # opcional, recomendado si lo tienes:
        # "y_train": y_train,
        # "scaler": scaler
    },
    "modelo_svm_pearson_0,6.pkl"
)

Notas:
- Este script NO asume que exista un pipeline.
- Si el .pkl contiene "scaler", se usará automáticamente.
- Si el .pkl no contiene "y_train", los boxplots se agrupan por clase predicha.
"""

# =========================
# CONFIGURACIÓN
# =========================

import os
import re
import warnings

import joblib
import shap
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # útil si ejecutas el script en terminal/servidor
import matplotlib.pyplot as plt

from sklearn.neighbors import NearestNeighbors
from rdkit import Chem
from mordred import Calculator, descriptors


MODEL_PATH = "modelo_svm_pearson_0,6.pkl"
SCREENING_PATH = "DATASET_SCREENING_FINAL.xlsx"
SMILES_COL = "SMILES"

OUTPUT_DIR = "resultados_svm_screening"

PROB_THRESHOLD = 0.80
N_NEIGHBORS_AD = 5

# Kernel SHAP puede ser lento en SVM. Sube estos valores si quieres más precisión.
SHAP_BACKGROUND_SIZE = 100
SHAP_EXPLAIN_SIZE = 100
SHAP_NSAMPLES = 200

TOP_N_SHAP = 20
TOP_N_BOXPLOTS = 10

RANDOM_STATE = 42
SHOW_PLOTS = False

warnings.filterwarnings("ignore", category=UserWarning)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================
# FUNCIONES AUXILIARES
# =========================

def safe_filename(text, max_len=90):
    """Convierte nombres de descriptores en nombres de archivo seguros."""
    text = str(text)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text[:max_len].strip("_") or "variable"


def save_fig(path, dpi=150):
    """Guarda la figura actual y la cierra."""
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close()


def load_model_bundle(model_path):
    """Carga el diccionario guardado con joblib y valida las claves mínimas."""
    bundle = joblib.load(model_path)

    if not isinstance(bundle, dict):
        raise TypeError(
            "El archivo .pkl debe contener un diccionario con al menos "
            "las claves: 'model', 'features' y 'X_train'."
        )

    required_keys = {"model", "features", "X_train"}
    missing = required_keys - set(bundle.keys())
    if missing:
        raise KeyError(f"Faltan claves en el .pkl: {sorted(missing)}")

    model = bundle["model"]
    features = list(bundle["features"])
    X_train = bundle["X_train"]

    if not hasattr(model, "predict_proba"):
        raise AttributeError(
            "El modelo cargado no tiene predict_proba(). "
            "Para SVC, entrena el modelo con probability=True."
        )

    X_train = ensure_feature_dataframe(X_train, features)

    return bundle, model, features, X_train


def ensure_feature_dataframe(data, features):
    """
    Convierte cualquier entrada a DataFrame con las columnas exactas del entrenamiento.
    Rellena missing values con 0, igual que en el screening.
    """
    if isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        df = pd.DataFrame(data, columns=features)

    df = df.reindex(columns=features)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)

    return df


def transform_for_model(X_df, scaler=None):
    """
    Devuelve la matriz que verá el modelo.
    Si el .pkl incluye scaler, se aplica.
    Si no, se usan las variables tal cual.
    """
    X_df = ensure_feature_dataframe(X_df, FEATURES)

    if scaler is not None:
        return scaler.transform(X_df)

    return X_df


def transform_for_ad(X_df, scaler=None):
    """
    Devuelve un array numérico para kNN-AD.
    Usa el mismo espacio que el modelo, aplicando scaler si existe.
    """
    X_model = transform_for_model(X_df, scaler=scaler)
    return np.asarray(X_model)


def model_predict_proba(data):
    """Wrapper para SHAP y predicción con columnas correctas."""
    X_df = ensure_feature_dataframe(data, FEATURES)
    X_model = transform_for_model(X_df, scaler=SCALER)
    return MODEL.predict_proba(X_model)


def model_predict(data):
    """Wrapper para predecir clase con columnas correctas."""
    X_df = ensure_feature_dataframe(data, FEATURES)
    X_model = transform_for_model(X_df, scaler=SCALER)
    return MODEL.predict(X_model)


def find_class_position(classes, candidates, default_pos):
    """
    Busca la posición de una clase aunque venga como int, str, bool, etc.
    """
    classes_list = list(classes)

    for candidate in candidates:
        for i, cls in enumerate(classes_list):
            if cls == candidate or str(cls).strip().lower() == str(candidate).strip().lower():
                return i

    return default_pos


def same_label(a, b):
    """Compara etiquetas aunque tengan tipos distintos."""
    return str(a).strip().lower() == str(b).strip().lower()


def select_shap_class(shap_values_raw, class_pos):
    """
    Normaliza los formatos de salida de SHAP:
    - lista: [clase0, clase1]
    - array 3D: (n_samples, n_features, n_classes)
    - array 3D: (n_classes, n_samples, n_features)
    - array 2D: (n_samples, n_features)
    """
    if isinstance(shap_values_raw, list):
        return np.asarray(shap_values_raw[class_pos])

    arr = np.asarray(shap_values_raw)

    if arr.ndim == 2:
        return arr

    if arr.ndim == 3:
        # Formato habitual nuevo: n_samples x n_features x n_outputs
        if arr.shape[2] > class_pos:
            return arr[:, :, class_pos]

        # Formato alternativo: n_outputs x n_samples x n_features
        if arr.shape[0] > class_pos:
            return arr[class_pos, :, :]

    raise ValueError(f"Formato de shap_values no reconocido: shape={arr.shape}")


def select_expected_value(expected_value_raw, class_pos):
    """Obtiene el expected_value correcto para una clase."""
    if isinstance(expected_value_raw, (list, tuple, np.ndarray)):
        return np.asarray(expected_value_raw)[class_pos]
    return expected_value_raw


def sample_dataframe(df, n, random_state=42):
    """Muestra aleatoria reproducible, conservando DataFrame."""
    n = min(n, len(df))
    if n <= 0:
        raise ValueError("No hay filas disponibles para muestrear.")
    return df.sample(n=n, random_state=random_state).reset_index(drop=True)


def save_force_plot(expected_value, shap_matrix, X_explain_df, row_idx, html_path, png_path, title):
    """
    Guarda un force plot en HTML y, si SHAP lo permite, también en PNG.
    El HTML suele ser más fiable para force plots.
    """
    row_values = X_explain_df.iloc[row_idx, :]

    force_plot = shap.force_plot(
        expected_value,
        shap_matrix[row_idx, :],
        row_values,
        feature_names=FEATURES,
        matplotlib=False
    )
    shap.save_html(html_path, force_plot)

    try:
        shap.force_plot(
            expected_value,
            shap_matrix[row_idx, :],
            row_values,
            feature_names=FEATURES,
            matplotlib=True,
            show=False
        )
        plt.title(title)
        save_fig(png_path, dpi=150)
    except Exception as exc:
        print(f"No se pudo guardar el force plot PNG ({png_path}). HTML guardado correctamente.")
        print(f"Motivo: {exc}")


def plot_boxplots_by_class(X_df, class_labels, top_features, output_path):
    """
    Crea una figura con boxplots + nube de puntos (jitter)
    de las variables más importantes, separadas por clase.
    """
    class_labels = pd.Series(class_labels).reset_index(drop=True)
    X_df = X_df.reset_index(drop=True)

    display_class = class_labels.apply(
        lambda x: "Activo" if same_label(x, ACTIVE_LABEL)
        else ("No activo" if same_label(x, INACTIVE_LABEL) else str(x))
    )

    n_features = len(top_features)
    n_cols = 2
    n_rows = int(np.ceil(n_features / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(14, max(4, 4 * n_rows))
    )

    if n_features == 1:
        axes = np.asarray([axes])
    axes = axes.ravel()

    for ax, feature in zip(axes, top_features):

        data_inactive = X_df.loc[display_class == "No activo", feature].dropna().values
        data_active = X_df.loc[display_class == "Activo", feature].dropna().values

        if len(data_inactive) == 0:
            data_inactive = np.array([np.nan])
        if len(data_active) == 0:
            data_active = np.array([np.nan])

        # =========================
        # BOXPLOT
        # =========================
        ax.boxplot(
            [data_inactive, data_active],
            tick_labels=["No activo", "Activo"],
            showfliers=False
        )

        # =========================
        # NUVE DE PUNTOS (JITTER)
        # =========================
        jitter_inactive = np.random.normal(1, 0.04, size=len(data_inactive))
        jitter_active = np.random.normal(2, 0.04, size=len(data_active))

        ax.scatter(jitter_inactive, data_inactive, alpha=0.5, s=15)
        ax.scatter(jitter_active, data_active, alpha=0.5, s=15)

        ax.set_title(str(feature), fontsize=10)
        ax.set_ylabel("Valor descriptor")
        ax.grid(axis="y", alpha=0.25)

    for ax in axes[n_features:]:
        ax.axis("off")

    fig.suptitle(
        "Boxplots + nube de puntos de variables por clase",
        fontsize=14,
        y=1.0
    )

    save_fig(output_path, dpi=150)

    boxplot_export = X_df[top_features].copy()
    boxplot_export["clase_para_boxplot"] = display_class
    return boxplot_export

# def plot_boxplots_by_class(X_df, class_labels, top_features, output_path):
#     """
#     Crea una figura con boxplots de las variables más importantes,
#     separadas por No activo y Activo.
#     """
#     class_labels = pd.Series(class_labels).reset_index(drop=True)
#     X_df = X_df.reset_index(drop=True)

#     display_class = class_labels.apply(
#         lambda x: "Activo" if same_label(x, ACTIVE_LABEL)
#         else ("No activo" if same_label(x, INACTIVE_LABEL) else str(x))
#     )

#     n_features = len(top_features)
#     n_cols = 2
#     n_rows = int(np.ceil(n_features / n_cols))

#     fig, axes = plt.subplots(
#         n_rows,
#         n_cols,
#         figsize=(14, max(4, 4 * n_rows))
#     )

#     if n_features == 1:
#         axes = np.asarray([axes])
#     axes = axes.ravel()

#     for ax, feature in zip(axes, top_features):
#         data_inactive = X_df.loc[display_class == "No activo", feature].dropna().values
#         data_active = X_df.loc[display_class == "Activo", feature].dropna().values

#         # Evita errores si una de las clases no aparece.
#         if len(data_inactive) == 0:
#             data_inactive = np.array([np.nan])
#         if len(data_active) == 0:
#             data_active = np.array([np.nan])

#         ax.boxplot(
#             [data_inactive, data_active],
#             tick_labels=["No activo", "Activo"],
#             showfliers=False
#         )
#         ax.set_title(str(feature), fontsize=10)
#         ax.set_ylabel("Valor descriptor")
#         ax.grid(axis="y", alpha=0.25)

#     # Ocultar ejes sobrantes
#     for ax in axes[n_features:]:
#         ax.axis("off")

#     fig.suptitle(
#         "Boxplots de variables más influyentes según SHAP",
#         fontsize=14,
#         y=1.0
#     )

#     save_fig(output_path, dpi=150)

#     boxplot_export = X_df[top_features].copy()
#     boxplot_export["clase_para_boxplot"] = display_class
#     return boxplot_export


# =========================
# 1. CARGAR MODELO GUARDADO
# =========================

print(f"Cargando modelo desde: {MODEL_PATH}")

BUNDLE, MODEL, FEATURES, X_TRAIN = load_model_bundle(MODEL_PATH)

# Opcional: si algún día guardas scaler e y_train en el .pkl, el script los usará.
SCALER = BUNDLE.get("scaler", None)
Y_TRAIN = BUNDLE.get("y_train", None)

MODEL_NAME = BUNDLE.get("model_name", "SVM")

classes = list(getattr(MODEL, "classes_", [0, 1]))
ACTIVE_POS = find_class_position(classes, candidates=[1, "1", "activo", "active", True], default_pos=min(1, len(classes) - 1))
INACTIVE_POS = find_class_position(classes, candidates=[0, "0", "inactivo", "inactive", "no activo", False], default_pos=0)

ACTIVE_LABEL = classes[ACTIVE_POS]
INACTIVE_LABEL = classes[INACTIVE_POS]

print(f"Modelo cargado: {MODEL_NAME}")
print(f"Número de features: {len(FEATURES)}")
print(f"Clases del modelo: {classes}")
print(f"Clase considerada como NO ACTIVO: {INACTIVE_LABEL}")
print(f"Clase considerada como ACTIVO: {ACTIVE_LABEL}")

if SCALER is None:
    print("No se ha encontrado scaler en el .pkl. Se usarán las features tal cual.")
else:
    print("Scaler encontrado en el .pkl. Se aplicará antes de predecir y para AD.")


# =========================
# 2. INTERPRETACIÓN DEL MODELO CON SHAP
# =========================

print(f"\nCalculando SHAP para el modelo: {MODEL_NAME}...")

X_BACKGROUND = sample_dataframe(X_TRAIN, SHAP_BACKGROUND_SIZE, random_state=RANDOM_STATE)
X_EXPLAIN = sample_dataframe(X_TRAIN, SHAP_EXPLAIN_SIZE, random_state=RANDOM_STATE + 1)

# KernelExplainer es el más apropiado para SVM no lineal sin pipeline.
EXPLAINER = shap.KernelExplainer(
    model_predict_proba,
    X_BACKGROUND
)

SHAP_VALUES_RAW = EXPLAINER.shap_values(
    X_EXPLAIN,
    nsamples=SHAP_NSAMPLES
)

SHAP_VALUES_ACTIVE = select_shap_class(SHAP_VALUES_RAW, ACTIVE_POS)
SHAP_VALUES_INACTIVE = select_shap_class(SHAP_VALUES_RAW, INACTIVE_POS)

EXPECTED_ACTIVE = select_expected_value(EXPLAINER.expected_value, ACTIVE_POS)
EXPECTED_INACTIVE = select_expected_value(EXPLAINER.expected_value, INACTIVE_POS)

print("SHAP calculado.")


# =========================
# 3. SHAP SUMMARY PLOT PARA CLASE ACTIVO
# =========================

plt.figure(figsize=(10, 8))

shap.summary_plot(
    SHAP_VALUES_ACTIVE,
    X_EXPLAIN,
    feature_names=FEATURES,
    max_display=TOP_N_SHAP,
    show=False
)

plt.title(f"SHAP Summary - {MODEL_NAME} - clase activo")
summary_path = os.path.join(OUTPUT_DIR, "shap_summary_activo.png")
save_fig(summary_path, dpi=150)

print(f"Figura guardada: {summary_path}")


# =========================
# 4. IMPORTANCIA GLOBAL SHAP
# =========================

importance_df = pd.DataFrame({
    "descriptor": FEATURES,
    "mean_abs_shap_activo": np.abs(SHAP_VALUES_ACTIVE).mean(axis=0)
}).sort_values(by="mean_abs_shap_activo", ascending=False)

importance_path = os.path.join(OUTPUT_DIR, "shap_importance_activo.xlsx")
importance_df.to_excel(importance_path, index=False, engine="openpyxl")

print("\nTop 10 descriptores más importantes para clase activo:")
print(importance_df.head(10))
print(f"Excel guardado: {importance_path}")


# =========================
# 5. SHAP FORCE PLOTS: RESULTADO 0 Y RESULTADO 1 ACTIVO
# =========================

proba_explain = model_predict_proba(X_EXPLAIN)
pred_explain = model_predict(X_EXPLAIN)
prob_active_explain = proba_explain[:, ACTIVE_POS]

inactive_candidates = [
    i for i, pred in enumerate(pred_explain)
    if same_label(pred, INACTIVE_LABEL)
]
active_candidates = [
    i for i, pred in enumerate(pred_explain)
    if same_label(pred, ACTIVE_LABEL)
]

if inactive_candidates:
    # Escogemos un ejemplo claramente no activo: menor probabilidad de activo.
    idx_inactive = min(inactive_candidates, key=lambda i: prob_active_explain[i])
else:
    idx_inactive = int(np.argmin(prob_active_explain))
    print("Aviso: no hay ejemplos predichos como no activo en X_EXPLAIN. Se usa el de menor probabilidad de activo.")

if active_candidates:
    # Escogemos un ejemplo claramente activo: mayor probabilidad de activo.
    idx_active = max(active_candidates, key=lambda i: prob_active_explain[i])
else:
    idx_active = int(np.argmax(prob_active_explain))
    print("Aviso: no hay ejemplos predichos como activo en X_EXPLAIN. Se usa el de mayor probabilidad de activo.")

force_0_html = os.path.join(OUTPUT_DIR, "shap_force_resultado_0_no_activo.html")
force_0_png = os.path.join(OUTPUT_DIR, "shap_force_resultado_0_no_activo.png")

save_force_plot(
    EXPECTED_INACTIVE,
    SHAP_VALUES_INACTIVE,
    X_EXPLAIN,
    idx_inactive,
    force_0_html,
    force_0_png,
    title="SHAP Force Plot - resultado 0 / no activo"
)

force_1_html = os.path.join(OUTPUT_DIR, "shap_force_resultado_1_activo.html")
force_1_png = os.path.join(OUTPUT_DIR, "shap_force_resultado_1_activo.png")

save_force_plot(
    EXPECTED_ACTIVE,
    SHAP_VALUES_ACTIVE,
    X_EXPLAIN,
    idx_active,
    force_1_html,
    force_1_png,
    title="SHAP Force Plot - resultado 1 / activo"
)

force_examples = pd.DataFrame([
    {
        "force_plot": "resultado_0_no_activo",
        "fila_en_X_EXPLAIN": idx_inactive,
        "clase_predicha": pred_explain[idx_inactive],
        "prob_activo": prob_active_explain[idx_inactive],
        "html": force_0_html,
        "png": force_0_png
    },
    {
        "force_plot": "resultado_1_activo",
        "fila_en_X_EXPLAIN": idx_active,
        "clase_predicha": pred_explain[idx_active],
        "prob_activo": prob_active_explain[idx_active],
        "html": force_1_html,
        "png": force_1_png
    }
])

force_examples_path = os.path.join(OUTPUT_DIR, "shap_force_ejemplos.xlsx")
force_examples.to_excel(force_examples_path, index=False, engine="openpyxl")

print(f"Force plot resultado 0 guardado: {force_0_html}")
print(f"Force plot resultado 1 activo guardado: {force_1_html}")
print(f"Resumen de ejemplos force plot guardado: {force_examples_path}")


# =========================
# 6. BOXPLOTS DE VARIABLES MÁS INFLUYENTES POR CLASE
# =========================

top_box_features = importance_df["descriptor"].head(TOP_N_BOXPLOTS).tolist()

if Y_TRAIN is not None:
    boxplot_classes = pd.Series(Y_TRAIN).reset_index(drop=True)
    class_source = "clase real guardada en y_train"
else:
    boxplot_classes = pd.Series(model_predict(X_TRAIN)).reset_index(drop=True)
    class_source = "clase predicha por el modelo, porque el .pkl no contiene y_train"

boxplot_path = os.path.join(OUTPUT_DIR, "boxplots_variables_mas_influyentes_por_clase.png")
boxplot_export = plot_boxplots_by_class(
    X_TRAIN,
    boxplot_classes,
    top_box_features,
    boxplot_path
)

boxplot_data_path = os.path.join(OUTPUT_DIR, "boxplots_variables_mas_influyentes_datos.xlsx")
boxplot_export.to_excel(boxplot_data_path, index=False, engine="openpyxl")

print(f"\nBoxplots guardados: {boxplot_path}")
print(f"Datos usados para boxplots guardados: {boxplot_data_path}")
print(f"Clases usadas para boxplots: {class_source}")


# =========================
# 7. APPLICABILITY DOMAIN DURANTE ENTRENAMIENTO
# =========================
# kNN-AD: una molécula está dentro del dominio si está cerca de moléculas del training set.
    
print("\nConstruyendo dominio de aplicabilidad kNN-AD...")

X_TRAIN_AD = transform_for_ad(X_TRAIN, scaler=SCALER)

n_neighbors = min(N_NEIGHBORS_AD, len(X_TRAIN_AD))
if n_neighbors < 2:
    raise ValueError("X_train necesita al menos 2 filas para construir el dominio de aplicabilidad.")

nn = NearestNeighbors(n_neighbors=n_neighbors)
nn.fit(X_TRAIN_AD)

distances_train, _ = nn.kneighbors(X_TRAIN_AD)
mean_dist_train = distances_train.mean(axis=1)

threshold = mean_dist_train.mean() + 2 * mean_dist_train.std()

threshold_path = os.path.join(OUTPUT_DIR, "ad_threshold.txt")
with open(threshold_path, "w", encoding="utf-8") as f:
    f.write(f"Threshold AD: {threshold:.8f}\n")
    f.write(f"N vecinos: {n_neighbors}\n")

print(f"Threshold AD: {threshold:.4f}")
print(f"Threshold guardado: {threshold_path}")


# =========================
# 8. SCREENING CON NUEVAS MOLÉCULAS
# =========================

print(f"\nCargando moléculas para screening desde: {SCREENING_PATH}")

df_screening = pd.read_excel(SCREENING_PATH)

if SMILES_COL not in df_screening.columns:
    raise KeyError(f"No existe la columna '{SMILES_COL}' en {SCREENING_PATH}")

#df_screening[SMILES_COL] = df_screening[SMILES_COL].apply(
    #lambda x: x.replace('"', "") if isinstance(x, str) else x
#)

df_screening["RDKit"] = df_screening[SMILES_COL].apply(
    lambda x: Chem.MolFromSmiles(x) if isinstance(x, str) else None
)

n_invalid = df_screening["RDKit"].isna().sum()
if n_invalid > 0:
    print(f"SMILES inválidos eliminados: {n_invalid}")

df_screening = df_screening[df_screening["RDKit"].notna()].reset_index(drop=True)

print("Calculando descriptores Mordred...")
calc = Calculator(descriptors)

desc_results = list(calc.map(df_screening["RDKit"]))
desc = pd.DataFrame([d.asdict() for d in desc_results])

desc = desc.apply(pd.to_numeric, errors="coerce")
desc = desc.replace([np.inf, -np.inf], np.nan)

df_screening_mordred = pd.concat(
    [df_screening.reset_index(drop=True), desc.reset_index(drop=True)],
    axis=1
)

# Asegurar mismas columnas/features que durante entrenamiento
X_SCREEN = ensure_feature_dataframe(df_screening_mordred, FEATURES)

n_missing_screen_features = sum([feature not in df_screening_mordred.columns for feature in FEATURES])
if n_missing_screen_features > 0:
    print(
        f"Aviso: {n_missing_screen_features} features del entrenamiento no aparecen "
        "en los descriptores calculados. Se rellenan con 0."
    )

# Predicción
y_proba_screen = model_predict_proba(X_SCREEN)
y_prob_active = y_proba_screen[:, ACTIVE_POS]
y_pred_screen = model_predict(X_SCREEN)

# AD para screening
X_SCREEN_AD = transform_for_ad(X_SCREEN, scaler=SCALER)
distances_screen, _ = nn.kneighbors(X_SCREEN_AD)
mean_dist_screen = distances_screen.mean(axis=1)

AD_screen = mean_dist_screen < threshold
final_hits = (y_prob_active > PROB_THRESHOLD) & AD_screen

print("Dentro del dominio:", int(AD_screen.sum()))
print("Fuera del dominio:", int((~AD_screen).sum()))


# =========================
# 9. GUARDAR RESULTADOS
# =========================

df_resultados = df_screening.drop(columns=["RDKit"], errors="ignore").copy()

df_resultados["prediccion"] = y_pred_screen
df_resultados["prob_activo"] = y_prob_active
df_resultados["AD"] = AD_screen
df_resultados["mean_distance_AD"] = mean_dist_screen
df_resultados["hit"] = final_hits

screening_full_path = os.path.join(OUTPUT_DIR, "screening_resultados_completo.xlsx")
df_resultados.to_excel(screening_full_path, index=False, engine="openpyxl")

df_hits = df_resultados[df_resultados["hit"]].copy()
df_hits = df_hits.sort_values(by="prob_activo", ascending=False)

hits_path = os.path.join(OUTPUT_DIR, "hits_finales.xlsx")
df_hits.to_excel(hits_path, index=False, engine="openpyxl")

print(f"\nResultados completos guardados: {screening_full_path}")
print(f"Hits finales guardados: {hits_path}")
print("Número de hits:", len(df_hits))

print("\nProceso terminado correctamente.")
