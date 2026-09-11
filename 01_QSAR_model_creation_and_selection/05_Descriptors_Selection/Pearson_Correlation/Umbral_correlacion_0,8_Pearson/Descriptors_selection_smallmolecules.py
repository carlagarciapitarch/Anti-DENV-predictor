#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 12:21:25 2026

@author: carla
"""

import pandas as pd
import numpy as np
# Cargar datos
df_mol_peq = pd.read_excel("mol_peq_con_descriptores_5uM.xlsx")


columnas_con_texto = []

for col in df_mol_peq.columns:
    # intentar convertir a numérico
    col_num = pd.to_numeric(df_mol_peq[col], errors='coerce')
    
    # si hay al menos un valor no convertible original → NaN en conversión
    if col_num.isna().sum() > 0 and df_mol_peq[col].isna().sum() != col_num.isna().sum():
        columnas_con_texto.append(col)

print(columnas_con_texto)


df_mol_peq_limpio = df_mol_peq.drop(columns=columnas_con_texto)


#solo predictores tiene todo menos la columna actividad a predecir
solo_predictores_mol_peq = df_mol_peq_limpio.drop(columns=["Actividad(0/1)"])
#utilizaremos solo_predictores para el procesamiento

#eliminar columnas con varianza menor a 0,05
varianzas_mol_peq = solo_predictores_mol_peq.var() # Calcular varianza de cada columna
solo_predictores_mol_peq_var = solo_predictores_mol_peq.loc[:, varianzas_mol_peq >= 0.05] # Filtrar columnas con varianza >= 0.05


#CORRELACION SPEARMAN Y ELIMINAR VARIABLES CON CORRELACION MAYOR A 0,8
corr = solo_predictores_mol_peq_var.corr(method='pearson')
corr_abs = corr.abs() #para detectar correlaciones tanto positivas como negativas. Es ponerlo en absoluto

# Umbral de correlación
threshold = 0.8

# Matriz superior (evita duplicados)
upper = corr_abs.where(np.triu(np.ones(corr_abs.shape), k=1).astype(bool))

# Columnas a eliminar
to_drop = [col for col in upper.columns if any(upper[col] > threshold)]

# Dataset reducido
solo_predictores_mol_peq_final = solo_predictores_mol_peq_var.drop(columns=to_drop)


predictores_y_respuesta_mol_peq_final = pd.concat(
    [solo_predictores_mol_peq_final, df_mol_peq_limpio["Actividad(0/1)"]], axis=1)
#predictores_y_respuesta_mol_peq_final es el que usaremos para los modelos de ML

#Guardar
predictores_y_respuesta_mol_peq_final.to_excel("datos_finales_molpeq_para_ML_0,8_pearson.xlsx", index=False)

