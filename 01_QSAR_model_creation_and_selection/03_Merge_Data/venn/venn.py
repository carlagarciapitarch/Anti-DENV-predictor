# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal.
"""

import pandas as pd
import matplotlib.pyplot as plt
from venn import venn 
import upsetplot
from rdkit import Chem

def smiles_canonico(smiles):
    if pd.isna(smiles):
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


df_A = pd.read_excel("BindingDB_Data_Ready_to_merge.xlsx")
df_B = pd.read_excel("ChEMBL_Data_Ready_to_merge.xlsx")
df_C = pd.read_excel("DenvInD_Data_Ready_to_merge.xlsx")
df_D = pd.read_excel("DrugRepV_Data_Ready_to_merge.xlsx")
df_E = pd.read_excel("PubChem_Data_Ready_to_merge.xlsx")

# Crear columna de SMILES canónico
df_A['SMILES_canonico'] = df_A['SMILES'].apply(smiles_canonico)
df_B['SMILES_canonico'] = df_B['SMILES'].apply(smiles_canonico)
df_C['SMILES_canonico'] = df_C['SMILES'].apply(smiles_canonico)
df_D['SMILES_canonico'] = df_D['SMILES'].apply(smiles_canonico)
df_E['SMILES_canonico'] = df_E['SMILES'].apply(smiles_canonico)



sets = {
    "BindingDB": set(df_A["SMILES_canonico"].dropna()),
    "ChEMBL": set(df_B["SMILES_canonico"].dropna()),
    "DenvInD": set(df_C["SMILES_canonico"].dropna()),
    "DrugRepV": set(df_D["SMILES_canonico"].dropna()),
    "PubChem": set(df_E["SMILES_canonico"].dropna())
}

plt.figure(figsize=(8,8))
venn(sets)
plt.show()