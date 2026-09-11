#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 12:46:02 2026

@author: carla
"""

import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit import DataStructs
from rdkit.Chem import Draw

df = pd.read_excel("dataset_train.xlsx")

df["Mol"] = df["SMILES"].apply(Chem.MolFromSmiles)
df = df[df["Mol"].notnull()].copy()

fpgen = rdFingerprintGenerator.GetMorganGenerator(
    radius = 2,
    fpSize = 2048)

df["FP"] = df["Mol"].apply(fpgen.GetFingerprint)

#CANDIDATO 1
islatravir_smiles = "C#C[C@]1(CO)O[C@@H](n2cnc3c(N)nc(F)nc32)C[C@@H]1O"
islatravir = Chem.MolFromSmiles(islatravir_smiles)
islatravir_fp = fpgen.GetFingerprint(islatravir)

#SIMILITUD TANIMOTO

df["Similarity"] = df["FP"].apply(
    lambda fp: DataStructs.TanimotoSimilarity(islatravir_fp, fp))

resultados = df.sort_values(
    "Similarity",
    ascending = False)

print(resultados[["SMILES", "Similarity"]].head(10))

resultados.to_excel("similaridad_islatravir.xlsx")

mols = [islatravir] + list(resultados.head(9)["Mol"])

img = Draw.MolsToGridImage(
    mols,
    molsPerRow=5,
    subImgSize=(250,250))

img.save("islatravir.png")


#CANDIDATO 2: ZABICIPRIL
zabicipril_smiles = "CCOC(=O)[C@H](CCc1ccccc1)N[C@@H](C)C(=O)N1C2CCC(CC2)[C@H]1C(=O)O"
zabicipril = Chem.MolFromSmiles(zabicipril_smiles)
zabicipril_fp = fpgen.GetFingerprint(zabicipril)

#SIMILITUD TANIMOTO

df["Similarity"] = df["FP"].apply(
    lambda fp: DataStructs.TanimotoSimilarity(zabicipril_fp, fp))

resultados = df.sort_values(
    "Similarity",
    ascending = False)

print(resultados[["SMILES", "Similarity"]].head(10))

resultados.to_excel("similaridad_zabicipril.xlsx")

mols = [zabicipril] + list(resultados.head(9)["Mol"])

img = Draw.MolsToGridImage(
    mols,
    molsPerRow=5,
    subImgSize=(250,250))

img.save("zabicipril.png")


#CANDIDATO 3: SABIZABULIN
sabizabulin_smiles = "COc1cc(C(=O)c2cnc(-c3c[nH]c4ccccc34)[nH]2)cc(OC)c1OC"
sabizabulin = Chem.MolFromSmiles(sabizabulin_smiles)
sabizabulin_fp = fpgen.GetFingerprint(sabizabulin)

#SIMILITUD TANIMOTO

df["Similarity"] = df["FP"].apply(
    lambda fp: DataStructs.TanimotoSimilarity(sabizabulin_fp, fp))

resultados = df.sort_values(
    "Similarity",
    ascending = False)

print(resultados[["SMILES", "Similarity"]].head(10))

resultados.to_excel("similaridad_sabizabulin.xlsx")

mols = [sabizabulin] + list(resultados.head(9)["Mol"])

img = Draw.MolsToGridImage(
    mols,
    molsPerRow=5,
    subImgSize=(250,250))

img.save("sabizabulin.png")


#CANDIDATO 4: TRIMETHOPRIM
trimethoprim_smiles = "COc1cc(Cc2cnc(N)nc2N)cc(OC)c1OC"
trimethoprim = Chem.MolFromSmiles(trimethoprim_smiles)
trimethoprim_fp = fpgen.GetFingerprint(trimethoprim)

#SIMILITUD TANIMOTO

df["Similarity"] = df["FP"].apply(
    lambda fp: DataStructs.TanimotoSimilarity(trimethoprim_fp, fp))

resultados = df.sort_values(
    "Similarity",
    ascending = False)

print(resultados[["SMILES", "Similarity"]].head(10))

resultados.to_excel("similaridad_trimethoprim.xlsx")

mols = [trimethoprim] + list(resultados.head(9)["Mol"])

img = Draw.MolsToGridImage(
    mols,
    molsPerRow=5,
    subImgSize=(250,250))

img.save("trimethoprim.png")


#CANDIDATO 5: RAMIPRIL
ramipril_smiles = "CCOC(=O)[C@H](CCc1ccccc1)N[C@@H](C)C(=O)N1[C@H](C(=O)O)C[C@@H]2CCC[C@@H]21"
ramipril = Chem.MolFromSmiles(zabicipril_smiles)
ramipril_fp = fpgen.GetFingerprint(ramipril)

#SIMILITUD TANIMOTO

df["Similarity"] = df["FP"].apply(
    lambda fp: DataStructs.TanimotoSimilarity(ramipril_fp, fp))

resultados = df.sort_values(
    "Similarity",
    ascending = False)

print(resultados[["SMILES", "Similarity"]].head(10))

resultados.to_excel("similaridad_ramipril.xlsx")

mols = [ramipril] + list(resultados.head(9)["Mol"])

img = Draw.MolsToGridImage(
    mols,
    molsPerRow=5,
    subImgSize=(250,250))

img.save("ramipril.png")


#CANDIDATO 6: COMBRETASTATIN A-1
combretastatin_smiles = "COc1ccc(/C=C\c2cc(OC)c(OC)c(OC)c2)c(O)c1O"
combretastatin = Chem.MolFromSmiles(combretastatin_smiles)
combretastatin_fp = fpgen.GetFingerprint(combretastatin)

#SIMILITUD TANIMOTO

df["Similarity"] = df["FP"].apply(
    lambda fp: DataStructs.TanimotoSimilarity(combretastatin_fp, fp))

resultados = df.sort_values(
    "Similarity",
    ascending = False)

print(resultados[["SMILES", "Similarity"]].head(10))

resultados.to_excel("similaridad_combretastatin.xlsx")

mols = [combretastatin] + list(resultados.head(9)["Mol"])

img = Draw.MolsToGridImage(
    mols,
    molsPerRow=5,
    subImgSize=(250,250))

img.save("combretastatin.png")


#CANDIDATO 7: EMVODODSTAT
emvododstat_smiles = "COc1ccc([C@H]2c3[nH]c4ccc(Cl)cc4c3CCN2C(=O)Oc2ccc(Cl)cc2)cc1"
emvododstat = Chem.MolFromSmiles(emvododstat_smiles)
emvododstat_fp = fpgen.GetFingerprint(emvododstat)

#SIMILITUD TANIMOTO

df["Similarity"] = df["FP"].apply(
    lambda fp: DataStructs.TanimotoSimilarity(emvododstat_fp, fp))

resultados = df.sort_values(
    "Similarity",
    ascending = False)

print(resultados[["SMILES", "Similarity"]].head(10))

resultados.to_excel("similaridad_emvododstat.xlsx")

mols = [emvododstat] + list(resultados.head(9)["Mol"])

img = Draw.MolsToGridImage(
    mols,
    molsPerRow=5,
    subImgSize=(250,250))

img.save("emvododstat.png")


df_train = pd.read_excel("dataset_train.xlsx")

# Separar activos e inactivos
df_activos = df_train[df_train["IC50/EC50(microM)"] <= 5]
df_inactivos = df_train[df_train["IC50/EC50(microM)"] > 5]

# (Opcional) comprobar tamaños
print("Activos:", len(df_activos))
print("Inactivos:", len(df_inactivos))

# Guardar en Excel
df_activos.to_excel("train_activos.xlsx", index=False)
df_inactivos.to_excel("train_inactivos.xlsx", index=False)