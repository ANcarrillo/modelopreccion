"""Fase 2.1 a 2.3 - Recolección, exploración y limpieza de los datos."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from . import config


def cargar_desde_bd(url: str | None = None) -> pd.DataFrame:
    """Lee la vista vw_dataset_desercion de PostgreSQL."""
    from sqlalchemy import create_engine

    url = url or os.getenv("DATABASE_URL")
    if not url:
        raise ValueError("Defina DATABASE_URL en el archivo .env o use --csv")
    motor = create_engine(url)
    with motor.connect() as con:
        return pd.read_sql("SELECT * FROM vw_dataset_desercion", con)


def cargar_desde_csv(ruta: str) -> pd.DataFrame:
    return pd.read_csv(ruta)


def resumen_exploratorio(df: pd.DataFrame) -> dict:
    """Fase 2.2 - Tamaño, balance de clases y porcentaje de valores faltantes."""
    variables = config.VARIABLES_NUMERICAS + config.VARIABLES_CATEGORICAS
    faltantes = (df[variables].isna().mean() * 100).round(2).sort_values(ascending=False)
    resumen = {
        "filas": int(len(df)),
        "columnas": int(df.shape[1]),
        "faltantes_pct": faltantes[faltantes > 0].to_dict(),
    }
    if config.OBJETIVO in df:
        resumen["balance_clases"] = df[config.OBJETIVO].value_counts().to_dict()
    return resumen


def limpiar(df: pd.DataFrame) -> pd.DataFrame:
    """Fase 2.3 - Tratamiento de datos inconsistentes y tipos de variables.

    La misma función la replica el backend (app/ml/motor.py -> preparar_variables).
    """
    df = df.copy()
    if "estado" in df:
        df = df[~df["estado"].isin(config.ESTADOS_EXCLUIDOS)]
    if "estudiante_id" in df:
        df = df.drop_duplicates(subset="estudiante_id", keep="last")

    df = preparar_variables(df)

    if config.OBJETIVO in df:
        df = df.dropna(subset=[config.OBJETIVO])
        df[config.OBJETIVO] = df[config.OBJETIVO].astype(int)
    return df


def preparar_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte tipos y anula valores fuera de rango (no elimina filas)."""
    df = df.copy()
    for col in config.VARIABLES_BOOLEANAS:
        if col in df:
            df[col] = df[col].map({True: 1, False: 0, "true": 1, "false": 0, 1: 1, 0: 0})
    for col in config.VARIABLES_NUMERICAS:
        if col not in df:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
    for col, (minimo, maximo) in config.RANGOS_VALIDOS.items():
        if col in df:
            df.loc[(df[col] < minimo) | (df[col] > maximo), col] = np.nan
    for col in config.VARIABLES_CATEGORICAS:
        if col not in df:
            df[col] = np.nan
        df[col] = df[col].astype("object").where(df[col].notna(), np.nan)
        df[col] = df[col].map(lambda v: v.strip().lower() if isinstance(v, str) else v)
    return df


def reporte_atipicos(df: pd.DataFrame) -> pd.DataFrame:
    """Fase 2.4 - Identificación de valores atípicos con el método IQR."""
    filas = []
    for col in config.VARIABLES_NUMERICAS:
        if col in config.VARIABLES_BOOLEANAS:
            continue
        serie = df[col].dropna()
        if serie.empty:
            continue
        q1, q3 = serie.quantile([0.25, 0.75])
        iqr = q3 - q1
        lim_inf, lim_sup = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n = int(((serie < lim_inf) | (serie > lim_sup)).sum())
        filas.append({"variable": col, "limite_inferior": lim_inf, "limite_superior": lim_sup,
                      "atipicos": n, "atipicos_pct": round(100 * n / len(serie), 2)})
    return pd.DataFrame(filas)
