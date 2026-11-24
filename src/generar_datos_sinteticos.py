"""Genera un CSV de datos FICTICIOS con las columnas de vw_dataset_desercion.

Sirve solo para probar que el flujo de entrenamiento funciona antes de tener
los datos reales. NO usar estos resultados en el informe final.

Uso: python -m src.generar_datos_sinteticos --n 2000
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from . import config


def generar(n: int, semilla: int = 7) -> pd.DataFrame:
    r = np.random.default_rng(semilla)
    estrato = r.choice([1, 2, 3, 4, 5], n, p=[0.3, 0.35, 0.2, 0.1, 0.05])
    trabaja = r.random(n) < 0.35
    horas = np.where(trabaja, r.integers(10, 48, n), 0)
    inscritos = r.integers(10, 21, n)
    asistencia = np.clip(r.normal(85, 10, n), 30, 100)
    prom_acum = np.clip(r.normal(3.6, 0.5, n), 1.0, 5.0)
    prom_per = np.clip(prom_acum + r.normal(0, 0.35, n), 0.5, 5.0)
    reprob = np.clip(np.round((3.6 - prom_per) * 1.8 + r.normal(0, 0.8, n)), 0, 6).astype(int)
    cred_rep = np.minimum(reprob * 3, inscritos)
    beca = r.random(n) < 0.15
    distancia = np.round(r.gamma(2, 6, n), 1)
    zona = np.where(r.random(n) < 0.25, "rural", "urbana")

    # Probabilidad "real" de desertar (relación inventada para la prueba)
    z = (-1.2 - 1.6 * (prom_per - 3.3) - 0.05 * (asistencia - 85) + 0.35 * reprob
         + 0.03 * horas - 0.25 * (estrato - 2) - 0.8 * beca + 0.02 * distancia
         + r.normal(0, 0.6, n))
    desercion = (1 / (1 + np.exp(-z)) > r.random(n)).astype(int)

    df = pd.DataFrame({
        "estudiante_id": np.arange(1, n + 1),
        "codigo": [f"E{i:05d}" for i in range(1, n + 1)],
        "programa_id": r.integers(1, 6, n),
        "periodo_id": 1,
        "edad": r.integers(16, 35, n),
        "genero": r.choice(["masculino", "femenino"], n),
        "puntaje_saber11": np.clip(r.normal(260, 40, n), 120, 450).astype(int),
        "tipo_colegio": r.choice(["publico", "privado"], n, p=[0.7, 0.3]),
        "semestre_cursado": r.integers(1, 11, n),
        "creditos_inscritos": inscritos,
        "creditos_aprobados": inscritos - cred_rep,
        "creditos_reprobados": cred_rep,
        "tasa_aprobacion": np.round((inscritos - cred_rep) / inscritos, 4),
        "promedio_periodo": np.round(prom_per, 2),
        "promedio_acumulado": np.round(prom_acum, 2),
        "asistencia_pct": np.round(asistencia, 2),
        "materias_reprobadas": reprob,
        "materias_canceladas": r.poisson(0.3, n),
        "materias_repetidas": r.poisson(0.4, n),
        "estrato": estrato,
        "ingreso_familiar": np.round(estrato * 900_000 * r.uniform(0.6, 1.4, n), -3),
        "trabaja": trabaja,
        "horas_trabajo_semana": horas,
        "personas_a_cargo": r.poisson(0.4, n),
        "tipo_vivienda": r.choice(["propia", "arriendo", "familiar"], n),
        "zona_residencia": zona,
        "distancia_km": distancia,
        "tiene_beca": beca,
        "apoyo_financiero": r.random(n) < 0.2,
        "nivel_educativo_padres": r.choice(["primaria", "secundaria", "tecnico", "universitario"], n),
        "desercion": desercion,
    })
    df["estado"] = np.where(desercion == 1, "desertor", "activo")
    # Algunos faltantes para probar la imputación
    for col in ["puntaje_saber11", "ingreso_familiar", "distancia_km", "nivel_educativo_padres"]:
        df.loc[r.random(n) < 0.05, col] = np.nan
    return df


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--salida", default=str(config.DIR_DATOS / "sinteticos.csv"))
    a = p.parse_args()
    config.DIR_DATOS.mkdir(parents=True, exist_ok=True)
    datos = generar(a.n)
    datos.to_csv(a.salida, index=False)
    print(f"{len(datos)} filas ficticias guardadas en {a.salida} (desertores: {datos.desercion.sum()})")
