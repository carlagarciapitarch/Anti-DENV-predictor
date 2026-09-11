#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 12:21:25 2026

@author: carla
"""

import pandas as pd
from rdkit import Chem
from mordred import Calculator, descriptors

# Cargar datos
df_mol_peq = pd.read_excel("mol_peq_con_0_1_5uM.xlsx")

df_mol_peq = df_mol_peq[["SMILES_canonico", "Actividad(0/1)"]]


#MOL PEQUEÑAS
# Convertir SMILES → moléculas RDKit
df_mol_peq["RDKit"] = df_mol_peq["SMILES_canonico"].apply(
    lambda x: Chem.MolFromSmiles(x) if isinstance(x, str) else None
)

# Eliminar SMILES inválidos
df_mol_peq = df_mol_peq[df_mol_peq["RDKit"].notna()].copy()

# Inicializar calculadora
calc = Calculator(descriptors)

# Calcular descriptores 
desc = calc.map(df_mol_peq["RDKit"])
desc = pd.DataFrame([d.asdict() for d in desc])

# Unir resultados
df_mol_peq_mordred = pd.concat(
    [df_mol_peq.reset_index(drop=True), desc.reset_index(drop=True)],
    axis=1
)

# Guardar
df_mol_peq_mordred.to_excel("mol_peq_con_descriptores_5uM.xlsx", index=False)
