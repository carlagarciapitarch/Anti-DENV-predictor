#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 17:49:29 2026

@author: carla
"""

import pandas as pd
import openpyxl
df = pd.read_excel("DenvInD_Data_processing.xlsx")
print(df)

# Filtrar filas donde columna esté vacia
df_limpio = df.dropna(subset=["IC50(microM)"])
print(df_limpio)

# Guardar el resultado en un nuevo archivo
df_limpio.to_excel("DenvInD_filtrado_nulos.xlsx", index=False)

# El documento DenvInD_filtrado_nulos lo he cambiado a DenvInD_Data_processing para 
# que esté todo igual
