"""Fase 2.4, 2.5, 3.4 y 4.4 - Preprocesamiento, balanceo y algoritmos candidatos."""
from __future__ import annotations

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as PipelineImb
from scipy.stats import randint, uniform
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config


def crear_preprocesador() -> ColumnTransformer:
    numericas = Pipeline([
        ("imputar", SimpleImputer(strategy="median")),   # valores faltantes
        ("normalizar", StandardScaler()),                # normalización (2.4)
    ])
    categoricas = Pipeline([
        ("imputar", SimpleImputer(strategy="constant", fill_value="desconocido")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),  # (2.3)
    ])
    return ColumnTransformer([
        ("num", numericas, config.VARIABLES_NUMERICAS),
        ("cat", categoricas, config.VARIABLES_CATEGORICAS),
    ])


def algoritmos_candidatos() -> dict:
    """Algoritmo -> (estimador, espacio de hiperparámetros)."""
    s = config.SEMILLA
    candidatos = {
        "random_forest": (
            RandomForestClassifier(random_state=s, n_jobs=-1),
            {
                "clf__n_estimators": randint(100, 500),
                "clf__max_depth": [None, 5, 10, 15, 20],
                "clf__min_samples_split": randint(2, 11),
                "clf__min_samples_leaf": randint(1, 6),
                "clf__max_features": ["sqrt", "log2", None],
            },
        ),
        "gradient_boosting": (
            GradientBoostingClassifier(random_state=s),
            {
                "clf__n_estimators": randint(100, 400),
                "clf__learning_rate": uniform(0.01, 0.2),
                "clf__max_depth": randint(2, 6),
                "clf__subsample": uniform(0.6, 0.4),
            },
        ),
    }
    try:
        from xgboost import XGBClassifier
        candidatos["xgboost"] = (
            XGBClassifier(random_state=s, eval_metric="logloss", n_jobs=-1),
            {
                "clf__n_estimators": randint(100, 500),
                "clf__learning_rate": uniform(0.01, 0.2),
                "clf__max_depth": randint(3, 9),
                "clf__subsample": uniform(0.6, 0.4),
                "clf__colsample_bytree": uniform(0.6, 0.4),
            },
        )
    except ImportError:
        print("[aviso] xgboost no está instalado; se omite.")
    try:
        from lightgbm import LGBMClassifier
        candidatos["lightgbm"] = (
            LGBMClassifier(random_state=s, verbose=-1, n_jobs=-1),
            {
                "clf__n_estimators": randint(100, 500),
                "clf__learning_rate": uniform(0.01, 0.2),
                "clf__num_leaves": randint(15, 64),
                "clf__min_child_samples": randint(5, 40),
                "clf__subsample": uniform(0.6, 0.4),
                "clf__colsample_bytree": uniform(0.6, 0.4),
            },
        )
    except ImportError:
        print("[aviso] lightgbm no está instalado; se omite.")
    return candidatos


def crear_pipeline(estimador, minoritaria: int) -> PipelineImb:
    """Preprocesamiento + SMOTE (solo se aplica al entrenar, 2.5) + clasificador."""
    vecinos = max(1, min(5, minoritaria - 1))
    return PipelineImb([
        ("pre", crear_preprocesador()),
        ("smote", SMOTE(random_state=config.SEMILLA, k_neighbors=vecinos)),
        ("clf", estimador),
    ])
