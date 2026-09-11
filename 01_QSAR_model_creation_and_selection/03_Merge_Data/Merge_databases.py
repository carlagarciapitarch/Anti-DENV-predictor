#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 17:49:29 2026

@author: carla
"""

import pandas as pd
# # PEPTIDOS
# df = pd.read_excel("ADPDB_Data_Ready_to_merge.xlsx")
# df1 = pd.read_excel("AVPdb_Data_Ready_to_merge.xlsx")
# df2 = pd.read_excel("DRAVP_Data_Ready_to_merge.xlsx")
# #print(df)

# peptidos_db = pd.concat([df, df1, df2], axis=0)
# print(peptidos_db)

# # Filtrar filas donde columnas estén vacias
# df_peptidos_limpio = peptidos_db.dropna(subset=["SMILES", "IC50/EC50(microM)"])

# df_peptidos_limpio.to_excel("df_peptidos.xlsx", index=False)


#MOLECULAS PEQUEÑAS

df3 = pd.read_excel("BindingDB_Data_Ready_to_merge.xlsx")
df4 = pd.read_excel("ChEMBL_Data_Ready_to_merge.xlsx")
df5 = pd.read_excel("DenvInD_Data_Ready_to_merge.xlsx")
df6 = pd.read_excel("DrugRepV_Data_Ready_to_merge.xlsx")
df7 = pd.read_excel("PubChem_Data_Ready_to_merge.xlsx")

mol_pequenas_db = pd.concat([df3, df4, df5, df6, df7], axis=0)

df_mol_pequenas_limpio = mol_pequenas_db.dropna(subset=["SMILES", "IC50/EC50(microM)"])

df_mol_pequenas_limpio.to_excel("df_mol_pequenas.xlsx", index=False)

