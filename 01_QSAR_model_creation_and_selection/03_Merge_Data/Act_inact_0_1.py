#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 17:49:29 2026

@author: carla
"""

import pandas as pd

#cambiamos valores de IC50 por 0 y 1. 0 será molécula inactiva y 1 molécula activa
#0 es si es mayor a 5 microM y 1 si es igual o menor a 5 microM. 
def actividad_activo_inactivo(valor):
  if valor <= 5:
    return 1
  else:
    return 0

df_mol_peq = pd.read_excel("mol_peq_sin_duplicados.xlsx")

df_mol_peq['Actividad(0/1)'] = df_mol_peq['IC50/EC50(microM)'].apply(actividad_activo_inactivo)

print(df_mol_peq['Actividad(0/1)'].value_counts())

df_mol_peq.to_excel("mol_peq_con_0_1_5uM.xlsx")