#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enriquecimiento de bits Morgan con RDKit >= 2025.x

Correcciones principales:
- GetFingerprint() devuelve un ExplicitBitVect, no (fp, bitInfo).
- La información de bits para dibujar Morgan environments se obtiene con AdditionalOutput.
- DrawMorganBit(..., useSVG=False) devuelve bytes PNG, no un objeto PIL con .save().
- Los denominadores usan solo moléculas válidas, no todas las filas del Excel.
"""

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Draw import DrawMorganBit
from rdkit.Chem.rdFingerprintGenerator import AdditionalOutput, GetMorganGenerator


# -------------------------
# CONFIGURACIÓN
# -------------------------
ACTIVOS_XLSX = "train_activos.xlsx"
INACTIVOS_XLSX = "train_inactivos.xlsx"
SMILES_COL = "SMILES"

RADIUS = 2
FP_SIZE = 2048
ENRICHMENT_THRESHOLD = 2.0
EPS = 1e-10

QUERY_SMILES = "C#C[C@]1(CO)O[C@@H](n2cnc3c(N)nc(F)nc32)C[C@@H]1O"
OUT_DIR = Path("enriched_bits")


# -------------------------
# FUNCIONES AUXILIARES
# -------------------------
def mol_from_smiles(smi):
    """Convierte un SMILES a Mol, devolviendo None si está vacío o no es válido."""
    if pd.isna(smi):
        return None

    smi = str(smi).strip()
    if not smi:
        return None

    return Chem.MolFromSmiles(smi)


def count_on_bits(smiles_series, fpgen, label):
    """Cuenta presencia/ausencia de bits Morgan por molécula válida."""
    counts = Counter()
    n_valid = 0
    n_invalid = 0

    for row_idx, smi in smiles_series.items():
        mol = mol_from_smiles(smi)
        if mol is None:
            n_invalid += 1
            print(f"Aviso: SMILES inválido/ vacío en {label}, fila índice {row_idx}: {smi!r}")
            continue

        fp = fpgen.GetFingerprint(mol)   # ExplicitBitVect
        counts.update(fp.GetOnBits())    # bits activos de esa molécula
        n_valid += 1

    if n_valid == 0:
        raise ValueError(f"No hay moléculas válidas en {label}.")

    return counts, n_valid, n_invalid


def get_fingerprint_and_bitinfo(mol, fpgen):
    """Devuelve fingerprint y bitInfoMap compatible con DrawMorganBit."""
    additional_output = AdditionalOutput()
    additional_output.AllocateBitInfoMap()

    fp = fpgen.GetFingerprint(mol, additionalOutput=additional_output)
    bit_info = additional_output.GetBitInfoMap()

    return fp, bit_info


def write_png(path, png_data):
    """RDKit devuelve bytes PNG cuando DrawMorganBit se usa con useSVG=False."""
    with open(path, "wb") as f:
        f.write(png_data)


# -------------------------
# SCRIPT PRINCIPAL
# -------------------------
def main():
    df_act = pd.read_excel(ACTIVOS_XLSX)
    df_inact = pd.read_excel(INACTIVOS_XLSX)

    for name, df in [(ACTIVOS_XLSX, df_act), (INACTIVOS_XLSX, df_inact)]:
        if SMILES_COL not in df.columns:
            raise ValueError(f"El archivo {name} no tiene una columna llamada {SMILES_COL!r}.")

    fpgen = GetMorganGenerator(radius=RADIUS, fpSize=FP_SIZE)

    # Activos e inactivos
    active_counts, n_active, invalid_active = count_on_bits(df_act[SMILES_COL], fpgen, "activos")
    inactive_counts, n_inactive, invalid_inactive = count_on_bits(df_inact[SMILES_COL], fpgen, "inactivos")

    print(f"Moléculas activas válidas: {n_active} | inválidas/vacías: {invalid_active}")
    print(f"Moléculas inactivas válidas: {n_inactive} | inválidas/vacías: {invalid_inactive}")

    # Tabla de probabilidades / enriquecimiento
    all_bits = sorted(set(active_counts) | set(inactive_counts))
    rows = []

    for bit in all_bits:
        p_active = active_counts[bit] / n_active
        p_inactive = inactive_counts[bit] / n_inactive

        enrichment = (p_active + EPS) / (p_inactive + EPS)
        log_odds = np.log(enrichment)

        rows.append({
            "bit": int(bit),
            "active_count": int(active_counts[bit]),
            "inactive_count": int(inactive_counts[bit]),
            "P_active": p_active,
            "P_inactive": p_inactive,
            "enrichment": enrichment,
            "log_odds": log_odds,
        })

    df_bits = pd.DataFrame(rows).sort_values("enrichment", ascending=False)
    df_bits.to_csv("enrichment_bits.csv", index=False)

    top_bits = df_bits.loc[df_bits["enrichment"] > ENRICHMENT_THRESHOLD, "bit"].astype(int).tolist()
    print("Top enriched bits:", top_bits[:20])
    print("Tabla guardada en: enrichment_bits.csv")

    # Visualizar bits enriquecidos en la molécula query
    query_mol = mol_from_smiles(QUERY_SMILES)
    if query_mol is None:
        raise ValueError("QUERY_SMILES no es válido.")

    query_fp, query_bit_info = get_fingerprint_and_bitinfo(query_mol, fpgen)
    query_bits = set(query_fp.GetOnBits())

    # Solo dibujamos bits que están en query_bits y tienen información de entorno.
    relevant_bits = [bit for bit in top_bits if bit in query_bits and bit in query_bit_info]
    print("Bits relevantes en tu molécula:", relevant_bits)

    OUT_DIR.mkdir(exist_ok=True)
    for bit in relevant_bits[:10]:
        png = DrawMorganBit(
            query_mol,
            bit,
            query_bit_info,
            useSVG=False,
            molSize=(300, 300),
        )
        out_file = OUT_DIR / f"enriched_bit_{bit}.png"
        write_png(out_file, png)
        print(f"Imagen guardada: {out_file}")


if __name__ == "__main__":
    main()

