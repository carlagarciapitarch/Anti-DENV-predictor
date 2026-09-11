#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 17:49:29 2026

@author: carla
"""

import pandas as pd
from rdkit import Chem

def smiles_canonico(smiles):
    if pd.isna(smiles):
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


df_mol_peq = pd.read_excel("df_mol_pequenas.xlsx")

# Crear columna de SMILES canónico
df_mol_peq['SMILES_canonico'] = df_mol_peq['SMILES'].apply(smiles_canonico)

#mirar si hay duplicados en mol pequeñas
duplicados_mol_peq = df_mol_peq[df_mol_peq.duplicated(subset=['SMILES_canonico', 'IC50/EC50(microM)'], keep=False)]
print(duplicados_mol_peq)


#eliminadar entradas con igual smile canonico e igual valor de inhibicion porque serían duplicados
df_mol_peq_sin_dup = df_mol_peq.drop_duplicates(subset=['SMILES_canonico', 'IC50/EC50(microM)']) # mol pequeñas

df_mol_peq_sin_dup.to_excel("mol_peq_sin_duplicados.xlsx", index=False)

