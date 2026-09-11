#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 17:49:29 2026

@author: carla
"""

import pandas as pd
import openpyxl
df = pd.read_excel("BindingDB_Data_processing.xlsx")
print(df)

# Filtrar filas donde columna esté vacia
df_limpio = df.dropna(subset=["IC50/EC50(nM)"])

# Guardar el resultado en un nuevo archivo
df_limpio.to_excel("BindingDB_filtrado_nulos.xlsx", index=False)

# El documento BindingDB_filtrado_nulos lo he cambiado a BindingDB_Data_processing para 
# que esté todo igual
