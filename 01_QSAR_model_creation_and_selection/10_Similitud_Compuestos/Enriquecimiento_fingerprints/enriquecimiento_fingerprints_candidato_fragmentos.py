#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enriquecimiento de bits Morgan y relación de los bits relevantes de una
molécula candidata con el fragmento/subestructura correspondiente.

Compatible con RDKit >= 2025.x.

Importante:
- Los bits Morgan NO tienen un "nombre" químico intrínseco.
- Lo más útil es relacionarlos con el fragmento de la molécula candidata
  que activa cada bit (por ejemplo, como SMILES de fragmento).
- Este script guarda una tabla con los bits relevantes presentes en la
  molécula candidata y el/los fragmentos asociados.
"""

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Draw import DrawMorganBits
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
GRID_COLS = 3
SUB_IMG_SIZE = (350, 300)
MAX_BITS_TO_DRAW = 12  # pon None para dibujar todos


# -------------------------
# FUNCIONES AUXILIARES
# -------------------------
def mol_from_smiles(smi):
    if pd.isna(smi):
        return None
    smi = str(smi).strip()
    if not smi:
        return None
    return Chem.MolFromSmiles(smi)


def count_on_bits(smiles_series, fpgen, label):
    counts = Counter()
    n_valid = 0
    n_invalid = 0

    for row_idx, smi in smiles_series.items():
        mol = mol_from_smiles(smi)
        if mol is None:
            n_invalid += 1
            print(f"Aviso: SMILES inválido/vacío en {label}, fila índice {row_idx}: {smi!r}")
            continue

        fp = fpgen.GetFingerprint(mol)
        counts.update(fp.GetOnBits())
        n_valid += 1

    if n_valid == 0:
        raise ValueError(f"No hay moléculas válidas en {label}.")

    return counts, n_valid, n_invalid


def get_fingerprint_and_bitinfo(mol, fpgen):
    additional_output = AdditionalOutput()
    additional_output.AllocateBitInfoMap()
    fp = fpgen.GetFingerprint(mol, additionalOutput=additional_output)
    bit_info = additional_output.GetBitInfoMap()
    return fp, bit_info


def get_environment_fragment(mol, center_atom_idx, radius):
    """Devuelve una descripción del entorno Morgan como fragmento SMILES."""
    bond_indices = list(Chem.FindAtomEnvironmentOfRadiusN(mol, radius, center_atom_idx))

    atom_indices = {center_atom_idx}
    for bond_idx in bond_indices:
        bond = mol.GetBondWithIdx(bond_idx)
        atom_indices.add(bond.GetBeginAtomIdx())
        atom_indices.add(bond.GetEndAtomIdx())

    atom_indices = sorted(atom_indices)

    fragment_smiles = Chem.MolFragmentToSmiles(
        mol,
        atomsToUse=atom_indices,
        bondsToUse=bond_indices,
        rootedAtAtom=center_atom_idx,
        canonical=True,
        isomericSmiles=True,
    )

    center_symbol = mol.GetAtomWithIdx(center_atom_idx).GetSymbol()

    return {
        "center_atom_idx": int(center_atom_idx),
        "center_atom_symbol": center_symbol,
        "radius": int(radius),
        "atom_indices": ",".join(map(str, atom_indices)),
        "bond_indices": ",".join(map(str, bond_indices)),
        "fragment_name": fragment_smiles,  # el "nombre" más útil aquí
        "fragment_smiles": fragment_smiles,
    }


def describe_relevant_bits(query_mol, query_bit_info, df_bits, top_bits):
    """
    Construye una tabla relacionando cada bit relevante del candidato con
    el/los fragmentos que lo activan.
    """
    enrichment_map = df_bits.set_index("bit").to_dict("index")
    rows = []

    for bit in top_bits:
        if bit not in query_bit_info:
            continue

        occurrences = query_bit_info[bit]
        fragments = []
        radii = []
        centers = []

        for center_atom_idx, radius in occurrences:
            frag = get_environment_fragment(query_mol, center_atom_idx, radius)
            fragments.append(frag["fragment_smiles"])
            radii.append(str(frag["radius"]))
            centers.append(f"{frag['center_atom_symbol']}{frag['center_atom_idx']}")

        unique_fragments = sorted(set(fragments))
        stats = enrichment_map[int(bit)]

        rows.append({
            "candidate_smiles": Chem.MolToSmiles(query_mol, isomericSmiles=True),
            "bit": int(bit),
            "fragment_name": " | ".join(unique_fragments),
            "fragment_smiles": " | ".join(unique_fragments),
            "n_occurrences_in_candidate": len(occurrences),
            "centers": " | ".join(centers),
            "radii": " | ".join(radii),
            "active_count": int(stats["active_count"]),
            "inactive_count": int(stats["inactive_count"]),
            "P_active": stats["P_active"],
            "P_inactive": stats["P_inactive"],
            "enrichment": stats["enrichment"],
            "log_odds": stats["log_odds"],
        })

    if not rows:
        return pd.DataFrame(columns=[
            "candidate_smiles", "bit", "fragment_name", "fragment_smiles",
            "n_occurrences_in_candidate", "centers", "radii",
            "active_count", "inactive_count", "P_active", "P_inactive",
            "enrichment", "log_odds"
        ])

    return pd.DataFrame(rows).sort_values("enrichment", ascending=False)


def write_svg(path, svg_text):
    if isinstance(svg_text, bytes):
        svg_text = svg_text.decode("utf-8")
    with open(path, "w", encoding="utf-8") as f:
        f.write(svg_text)


def build_legends(df_candidate_bits):
    legends = []
    for _, row in df_candidate_bits.iterrows():
        frag = str(row["fragment_name"])
        if len(frag) > 28:
            frag = frag[:25] + "..."
        legends.append(
            f"bit {int(row['bit'])}\nfrag: {frag}\nenr={row['enrichment']:.2f}"
        )
    return legends


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

    active_counts, n_active, invalid_active = count_on_bits(df_act[SMILES_COL], fpgen, "activos")
    inactive_counts, n_inactive, invalid_inactive = count_on_bits(df_inact[SMILES_COL], fpgen, "inactivos")

    print(f"Moléculas activas válidas: {n_active} | inválidas/vacías: {invalid_active}")
    print(f"Moléculas inactivas válidas: {n_inactive} | inválidas/vacías: {invalid_inactive}")

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
    print("Tabla global guardada en: enrichment_bits.csv")

    top_bits = df_bits.loc[df_bits["enrichment"] > ENRICHMENT_THRESHOLD, "bit"].astype(int).tolist()
    print("Top enriched bits:", top_bits[:20])

    query_mol = mol_from_smiles(QUERY_SMILES)
    if query_mol is None:
        raise ValueError("QUERY_SMILES no es válido.")

    query_fp, query_bit_info = get_fingerprint_and_bitinfo(query_mol, fpgen)
    query_bits = set(query_fp.GetOnBits())

    relevant_bits = [bit for bit in top_bits if bit in query_bits and bit in query_bit_info]
    print("Bits relevantes presentes en la molécula candidata:", relevant_bits)

    # Tabla de bits relevantes del candidato con sus fragmentos/nombres
    df_candidate_bits = describe_relevant_bits(query_mol, query_bit_info, df_bits, relevant_bits)

    OUT_DIR.mkdir(exist_ok=True)
    out_csv = OUT_DIR / "candidate_relevant_bits.csv"
    df_candidate_bits.to_csv(out_csv, index=False)
    print(f"Tabla de bits relevantes del candidato guardada en: {out_csv}")

    if len(df_candidate_bits) == 0:
        print("No hay bits enriquecidos presentes en la molécula candidata.")
        return

    # Figura única opcional con leyenda bit + fragmento + enrichment
    bits_to_draw_df = df_candidate_bits if MAX_BITS_TO_DRAW is None else df_candidate_bits.head(MAX_BITS_TO_DRAW)
    tpls = [(query_mol, int(bit), query_bit_info) for bit in bits_to_draw_df["bit"].tolist()]
    legends = build_legends(bits_to_draw_df)

    svg = DrawMorganBits(
        tpls,
        molsPerRow=GRID_COLS,
        subImgSize=SUB_IMG_SIZE,
        legends=legends,
        useSVG=True,
    )

    out_svg = OUT_DIR / "candidate_relevant_bits_grid.svg"
    write_svg(out_svg, svg)
    print(f"Figura única guardada en: {out_svg}")

    print("\nResumen de bits relevantes del candidato:")
    print(df_candidate_bits[["bit", "fragment_name", "enrichment"]].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
