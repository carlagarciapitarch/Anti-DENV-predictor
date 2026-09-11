#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 17:49:29 2026

@author: carla
"""

import os
import pandas as pd

# Carpeta donde están los CSV
carpeta = "/home/carla/Documentos/MasterBioinformatica/MasterThesis-UNIR2026/01_Raw_Data/PubChem_BioAssays_Raw_Data"

# Lista para acumular resultados
resultados = []

for archivo in os.listdir(carpeta):
    if archivo.endswith(".csv"):
        ruta_completa = os.path.join(carpeta, archivo)
        
        try:
            df = pd.read_csv(ruta_completa)

            # Filtrar condiciones
            filtrado = df[
                (df["Standard Type"].isin(["IC50", "EC50"])) &
                (df["Standard Relation"] == "=")
            ]

            # Seleccionar columnas
            seleccion = filtrado[["PUBCHEM_EXT_DATASOURCE_SMILES", "PubChem Standard Value"]]

            resultados.append(seleccion)

        except Exception as e:
            print(f"Error en {archivo}: {e}")

# Unir todos los resultados en un solo DataFrame
df_final = pd.concat(resultados, ignore_index=True)

# Guardar resultado final
df_final.to_excel("pubchem_resultado_filtrado.xlsx", index=False)

print("Proceso completado.")
