#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 17:49:29 2026

@author: carla
"""

import pandas as pd
import openpyxl
df = pd.read_excel("ChEMBL_Data_processing.xlsx")
print(df)

# Filtrar filas donde la columna 'Target name' contenga 'dengue'.
# Es para comprobar que todas las entradas sean correctas porque corresponde con dengue
filtro = df[df["Target Name"].str.contains("dengue", case=False, na=False)]
print(filtro)

# Filtrar filas donde columna "Standard Type" tenga IC50 o EC50

filtro2 = filtro[filtro["Standard Type"].str.contains("IC50|EC50", case=False, na=False) &
(df["Standard Relation"] == "'='")]

# Guardar el resultado en un nuevo archivo
filtro2.to_excel("filtrado_dengue_ChEMBL.xlsx", index=False)

# El documento filtrado_dengue_ChEMBL lo he cambiado a ChEMBL_Data_processing para 
# que esté todo igual


