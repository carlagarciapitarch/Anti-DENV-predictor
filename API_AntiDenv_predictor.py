#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""API FastAPI para screening molecular con el SVM validado.

Entrada
-------
POST /screening, multipart/form-data, campo ``file``.
El CSV debe contener exactamente dos columnas: ``id`` y ``smiles``.

Salida
------
CSV con exactamente cuatro columnas:
``id``, ``smiles``, ``probabilidad_activo`` y ``dominio_aplicabilidad``.

Las filas con SMILES inválidos se eliminan. El dominio de aplicabilidad se
calcula mediante kNN en el espacio estandarizado por el StandardScaler del
Pipeline. Al calibrar el umbral con X_train se excluye la propia fila y se usan
cinco vecinos reales. El umbral mantiene la regla original: media + 2·DE.
"""

from __future__ import annotations

import io
import logging
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status
#from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from mordred import Calculator, descriptors
from rdkit import Chem, RDLogger
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


# =============================================================================
# CONFIGURACIÓN
# =============================================================================

MODEL_PATH = Path("modelo_svm_pearson_0,6_api.pkl")
N_NEIGHBORS_AD = 5
AD_STD_MULTIPLIER = 2.0
MAX_ROWS = 5000
MAX_FILE_SIZE_MB = 20
OUTPUT_FILENAME = "resultados_screening_svm.csv"

# Configuración de los mensajes del servidor
logging.basicConfig(
    level="INFO",
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
LOGGER = logging.getLogger("svm-screening-api")
RDLogger.DisableLog("rdApp.error")


# =============================================================================
# EXCEPCIONES Y ESTADO
# =============================================================================


class CSVValidationError(ValueError):
    """El CSV no cumple el contrato de entrada."""


class ModelBundleError(RuntimeError):
    """El modelo o sus metadatos no son válidos."""


@dataclass(frozen=True)
class ModelAssets:
    """Objetos preparados una sola vez al iniciar el proceso."""

    model: Any
    features: list[str]
    x_train: pd.DataFrame
    scaler: Any
    classes: list[Any]
    active_class_position: int
    nn_model: NearestNeighbors
    ad_threshold: float


# =============================================================================
# MODELO Y DOMINIO DE APLICABILIDAD
# =============================================================================

def ensure_feature_dataframe(data: Any, features: list[str]) -> pd.DataFrame:
    """Devuelve un DataFrame numérico con las columnas exactas del modelo."""
    if isinstance(data, pd.DataFrame):
        frame = data.copy()
    else:
        frame = pd.DataFrame(data, columns=features)
    
    frame = frame.reindex(columns=features)
    frame = frame.apply(pd.to_numeric, errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan)

    # Es el mismo tratamiento utilizado en el entrenamiento/screening original.
    return frame.fillna(0.0)


def resolve_scaler(model: Any, bundle: dict[str, Any]) -> Any:
    """Obtiene el StandardScaler ajustado utilizado realmente por el SVM."""
    named_steps = getattr(model, "named_steps", None)

    if named_steps:
        scaler = named_steps.get("scaler")
        if scaler is not None and hasattr(scaler, "transform"):
            return scaler

        for step in named_steps.values():
            if isinstance(step, StandardScaler):
                return step

    # Solo para bundles cuyo modelo no sea un Pipeline.
    external_scaler = bundle.get("scaler")
    if external_scaler is not None and hasattr(external_scaler, "transform"):
        return external_scaler

    raise ModelBundleError(
        "No se encontró el StandardScaler ajustado. El AD no debe calcularse "
        "con descriptores Mordred sin estandarizar."
    )


def find_active_class_position(model: Any) -> tuple[list[Any], int]:
    """Localiza la columna de predict_proba correspondiente a la clase activa."""
    classes = list(getattr(model, "classes_", []))
    if not classes:
        raise ModelBundleError("El modelo no contiene classes_.")

    candidates = (1, "1", True, "activo", "active")
    for candidate in candidates:
        for position, label in enumerate(classes):
            if label == candidate:
                return classes, position
            if str(label).strip().lower() == str(candidate).strip().lower():
                return classes, position

    raise ModelBundleError(
        f"No se pudo identificar la clase activa. Clases encontradas: {classes}."
    )


def training_distances_without_self(
    nn_model: NearestNeighbors,
    x_train_scaled: np.ndarray,
    n_neighbors: int,
) -> np.ndarray:
    """Calcula k distancias por fila, eliminando su propio índice."""
    distances, indices = nn_model.kneighbors(
        x_train_scaled,
        n_neighbors=n_neighbors + 1,
    )

    cleaned = np.empty((len(x_train_scaled), n_neighbors), dtype=float)

    for row_index, (row_distances, row_indices) in enumerate(
        zip(distances, indices)
    ):
        # Se elimina por índice y no simplemente la primera distancia. Esto
        # sigue funcionando cuando existen moléculas duplicadas a distancia 0.
        selected = row_distances[row_indices != row_index][:n_neighbors]
        if len(selected) != n_neighbors:
            raise ModelBundleError(
                "No se pudieron obtener suficientes vecinos excluyendo la "
                "propia fila del entrenamiento."
            )
        cleaned[row_index] = selected

    return cleaned


def build_applicability_domain(
    x_train: pd.DataFrame,
    scaler: Any,
    n_neighbors: int,
) -> tuple[NearestNeighbors, float]:
    """Ajusta kNN y calcula el umbral del AD en el espacio estandarizado."""
    if n_neighbors < 1:
        raise ModelBundleError("N_NEIGHBORS_AD debe ser al menos 1.")
    if len(x_train) <= n_neighbors:
        raise ModelBundleError(
            f"X_train debe contener más de {n_neighbors} filas."
        )

    x_train_scaled = np.asarray(scaler.transform(x_train), dtype=float)
    if not np.isfinite(x_train_scaled).all():
        raise ModelBundleError("X_train escalado contiene valores no finitos.")

    nn_model = NearestNeighbors(metric="euclidean")
    nn_model.fit(x_train_scaled)

    distances = training_distances_without_self(
        nn_model=nn_model,
        x_train_scaled=x_train_scaled,
        n_neighbors=n_neighbors,
    )
    mean_distances = distances.mean(axis=1)

    threshold = float(
        mean_distances.mean()
        + AD_STD_MULTIPLIER * mean_distances.std(ddof=0)
    )

    if not np.isfinite(threshold) or threshold <= 0:
        raise ModelBundleError("El umbral calculado para el AD no es válido.")

    LOGGER.info(
        "AD preparado | k=%s | training=%s | threshold=%.8f",
        n_neighbors,
        len(x_train),
        threshold,
    )
    return nn_model, threshold


def predict_active_probability(
    assets: ModelAssets,
    x_screen: pd.DataFrame,
) -> np.ndarray:
    """Predice P(activo) sin aplicar dos veces el scaler."""
    if not hasattr(assets.model, "predict_proba"):
        raise ModelBundleError(
            "El modelo no implementa predict_proba(). El SVC debe estar "
            "entrenado con probability=True."
        )

    if getattr(assets.model, "named_steps", None):
        # El Pipeline recibe variables sin escalar; él mismo aplica el scaler.
        probabilities = assets.model.predict_proba(x_screen)
    else:
        # Solo para un SVC guardado fuera de un Pipeline.
        probabilities = assets.model.predict_proba(
            assets.scaler.transform(x_screen)
        )

    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 2:
        raise ModelBundleError("predict_proba() no devolvió una matriz 2D.")

    return probabilities[:, assets.active_class_position]


def load_model_assets(model_path: Path) -> ModelAssets:
    """Carga el bundle y prepara el AD una vez al iniciar la API."""
    if not model_path.is_file():
        raise ModelBundleError(
            f"No se encontró el modelo en '{model_path}'."
        )

    try:
        bundle = joblib.load(model_path)
    except Exception as exc:
        raise ModelBundleError(f"No se pudo cargar el modelo: {exc}") from exc

    if not isinstance(bundle, dict):
        raise ModelBundleError("El archivo debe contener un diccionario.")

    required = {"model", "features", "X_train"}
    missing = required.difference(bundle)
    if missing:
        raise ModelBundleError(
            f"Faltan claves obligatorias: {sorted(missing)}."
        )

    model = bundle["model"]
    features = [str(feature) for feature in bundle["features"]]

    if not features:
        raise ModelBundleError("La lista de features está vacía.")
    if len(features) != len(set(features)):
        raise ModelBundleError("La lista de features contiene duplicados.")

    x_train = ensure_feature_dataframe(bundle["X_train"], features)
    scaler = resolve_scaler(model, bundle)
    classes, active_position = find_active_class_position(model)
    nn_model, threshold = build_applicability_domain(
        x_train=x_train,
        scaler=scaler,
        n_neighbors=N_NEIGHBORS_AD,
    )

    LOGGER.info(
        "Modelo cargado | archivo=%s | features=%s | training=%s | clases=%s",
        model_path,
        len(features),
        len(x_train),
        classes,
    )

    return ModelAssets(
        model=model,
        features=features,
        x_train=x_train,
        scaler=scaler,
        classes=classes,
        active_class_position=active_position,
        nn_model=nn_model,
        ad_threshold=threshold,
    )


# =============================================================================
# CSV Y DESCRIPTORES
# =============================================================================


def read_csv_bytes(raw_bytes: bytes, filename: str | None) -> pd.DataFrame:
    """Lee y valida un CSV con exactamente las columnas id y smiles."""
    safe_filename = filename or "archivo.csv"

    if not safe_filename.lower().endswith(".csv"):
        raise CSVValidationError("El archivo debe tener extensión .csv.")
    if not raw_bytes:
        raise CSVValidationError("El archivo está vacío.")
    if len(raw_bytes) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise CSVValidationError(
            f"El archivo supera el límite de {MAX_FILE_SIZE_MB} MB."
        )

    dataframe: pd.DataFrame | None = None
    last_error: Exception | None = None

    # Acepta coma o punto y coma y las codificaciones más habituales.
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            dataframe = pd.read_csv(
                io.BytesIO(raw_bytes),
                sep=None,
                engine="python",
                dtype=str,
                keep_default_na=False,
                encoding=encoding,
            )
            break
        except Exception as exc:
            last_error = exc

    if dataframe is None:
        raise CSVValidationError(
            f"No se pudo leer el CSV. Detalle: {last_error}"
        )
    if dataframe.empty:
        raise CSVValidationError("El CSV no contiene filas de datos.")
    if len(dataframe) > MAX_ROWS:
        raise CSVValidationError(
            f"El CSV contiene {len(dataframe)} filas; el máximo es {MAX_ROWS}."
        )

    columns = [str(column).strip().lower() for column in dataframe.columns]
    if len(columns) != len(set(columns)):
        raise CSVValidationError("El CSV contiene columnas duplicadas.")

    dataframe.columns = columns
    if len(columns) != 2 or set(columns) != {"id", "smiles"}:
        raise CSVValidationError(
            "El CSV debe contener exactamente dos columnas: 'id' y 'smiles'."
        )

    dataframe = dataframe[["id", "smiles"]].copy()
    dataframe["id"] = dataframe["id"].astype(str).str.strip()
    dataframe["smiles"] = dataframe["smiles"].astype(str).str.strip()

    empty_ids = dataframe["id"].eq("")
    if empty_ids.any():
        rows = (np.flatnonzero(empty_ids.to_numpy()) + 2).tolist()[:10]
        raise CSVValidationError(
            f"La columna 'id' contiene valores vacíos en las filas: {rows}."
        )

    return dataframe


def split_valid_and_invalid_smiles(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convierte SMILES con RDKit y separa las filas inválidas."""
    molecules: list[Any | None] = []

    for smiles in dataframe["smiles"]:
        if not smiles:
            molecules.append(None)
            continue
        try:
            molecules.append(Chem.MolFromSmiles(smiles))
        except Exception:
            molecules.append(None)

    working = dataframe.copy()
    working["_mol"] = molecules

    valid = working[working["_mol"].notna()].reset_index(drop=True)
    invalid = working[working["_mol"].isna()][["id", "smiles"]].reset_index(
        drop=True
    )
    return valid, invalid


def calculate_mordred_features(
    valid_rows: pd.DataFrame,
    required_features: list[str],
) -> pd.DataFrame:
    """Calcula Mordred y selecciona los 80 descriptores del modelo."""
    # Todas las variables del modelo validado son descriptores 2D.
    calculator = Calculator(descriptors, ignore_3D=True)

    try:
        results = list(calculator.map(valid_rows["_mol"].tolist()))
    except Exception as exc:
        raise ModelBundleError(
            f"Falló el cálculo de descriptores Mordred: {exc}"
        ) from exc

    mordred_frame = pd.DataFrame([result.asdict() for result in results])

    missing_features = [
        feature
        for feature in required_features
        if feature not in mordred_frame.columns
    ]
    if missing_features:
        preview = ", ".join(missing_features[:10])
        suffix = "..." if len(missing_features) > 10 else ""
        raise ModelBundleError(
            "Mordred no generó todas las variables requeridas. "
            f"Faltan {len(missing_features)}: {preview}{suffix}. "
            "Comprueba las versiones de RDKit y Mordred."
        )

    return ensure_feature_dataframe(mordred_frame, required_features)


def run_screening(
    input_frame: pd.DataFrame,
    assets: ModelAssets,
) -> tuple[pd.DataFrame, int]:
    """Ejecuta validación, descriptores, predicción y AD."""
    valid_rows, invalid_rows = split_valid_and_invalid_smiles(input_frame)

    if valid_rows.empty:
        raise CSVValidationError("Ningún SMILES del archivo es válido.")

    x_screen = calculate_mordred_features(valid_rows, assets.features)
    active_probability = predict_active_probability(assets, x_screen)

    # Para el AD sí se aplica explícitamente el scaler del Pipeline.
    x_screen_scaled = np.asarray(assets.scaler.transform(x_screen), dtype=float)
    if not np.isfinite(x_screen_scaled).all():
        raise ModelBundleError(
            "Los descriptores escalados contienen valores no finitos."
        )

    distances, _ = assets.nn_model.kneighbors(
        x_screen_scaled,
        n_neighbors=N_NEIGHBORS_AD,
    )
    mean_distances = distances.mean(axis=1)
    inside_ad = mean_distances <= assets.ad_threshold

    output = valid_rows[["id", "smiles"]].copy()
    output["probabilidad_activo"] = np.round(active_probability, 6)
    output["dominio_aplicabilidad"] = np.where(
        inside_ad,
        "Dentro",
        "Fuera",
    )

    return (
        output[
            [
                "id",
                "smiles",
                "probabilidad_activo",
                "dominio_aplicabilidad",
            ]
        ],
        len(invalid_rows),
    )


# =============================================================================
# FASTAPI
# =============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model_assets = load_model_assets(MODEL_PATH)
    yield


app = FastAPI(
    title="API de screening molecular SVM",
    version="1.1.0",
    description=(
        "Recibe un CSV con id y smiles y devuelve probabilidad de actividad y "
        "dominio de aplicabilidad para los SMILES válidos."
    ),
    lifespan=lifespan,
)



@app.get("/", tags=["Estado"])
def root() -> dict[str, str]:
    return {
        "servicio": "screening molecular SVM",
        "endpoint": "POST /screening",
        "documentacion": "/docs",
    }


@app.get("/health", tags=["Estado"])
def health(request: Request) -> dict[str, Any]:
    assets: ModelAssets = request.app.state.model_assets
    return {
        "status": "ok",
        "modelo_cargado": True,
        "clases": [str(value) for value in assets.classes],
        "clase_activa": str(assets.classes[assets.active_class_position]),
        "numero_features": len(assets.features),
        "filas_entrenamiento": len(assets.x_train),
        "vecinos_ad": N_NEIGHBORS_AD,
        "umbral_ad": round(assets.ad_threshold, 8),
    }


@app.post(
    "/screening",
    tags=["Screening"],
    summary="Procesa un CSV de moléculas",
    response_class=Response,
    responses={
        200: {
            "content": {"text/csv": {}},
            "description": "CSV de resultados para los SMILES válidos.",
        },
        422: {"description": "CSV de entrada no válido."},
        500: {"description": "Error del modelo o de los descriptores."},
    },
)
def screening(
    request: Request,
    file: Annotated[
        UploadFile,
        File(description="CSV con exactamente las columnas id y smiles"),
    ],
) -> Response:
    """Recibe el CSV como multipart/form-data y devuelve otro CSV."""
    try:
        max_bytes = MAX_FILE_SIZE_MB * 1024 * 1024
        raw_bytes = file.file.read(max_bytes + 1)

        if len(raw_bytes) > max_bytes:
            raise CSVValidationError(
                f"El archivo supera el límite de {MAX_FILE_SIZE_MB} MB."
            )

        input_frame = read_csv_bytes(raw_bytes, file.filename)
        assets: ModelAssets = request.app.state.model_assets
        output, invalid_count = run_screening(input_frame, assets)

        csv_bytes = output.to_csv(index=False).encode("utf-8-sig")
        headers = {
            "Content-Disposition": f'attachment; filename="{OUTPUT_FILENAME}"',
            "X-Input-Rows": str(len(input_frame)),
            "X-Valid-Smiles": str(len(output)),
            "X-Invalid-Smiles-Removed": str(invalid_count),
        }

        LOGGER.info(
            "Screening completado | entrada=%s | válidos=%s | eliminados=%s "
            "| dentro_AD=%s",
            len(input_frame),
            len(output),
            invalid_count,
            int(output["dominio_aplicabilidad"].eq("Dentro").sum()),
        )

        return Response(
            content=csv_bytes,
            media_type="text/csv; charset=utf-8",
            headers=headers,
            status_code=status.HTTP_200_OK,
        )

    except CSVValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except ModelBundleError as exc:
        LOGGER.exception("Error del modelo durante el screening.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc
    except HTTPException:
        raise
    except Exception as exc:
        LOGGER.exception("Error inesperado durante el screening.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error inesperado durante el screening molecular.",
        ) from exc
    finally:
        file.file.close()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "massencillo_api_svm_screening_validada:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=False,
    )
