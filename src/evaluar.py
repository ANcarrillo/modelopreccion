"""Fase 5.2 y 5.3 - Métricas, matriz de confusión, curva ROC e interpretabilidad (SHAP)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def calcular_metricas(y_real, y_pred, y_prob) -> dict:
    return {
        "accuracy": round(float(accuracy_score(y_real, y_pred)), 4),
        "precision": round(float(precision_score(y_real, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_real, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_real, y_pred, zero_division=0)), 4),
        "auc_roc": round(float(roc_auc_score(y_real, y_prob)), 4),
        "matriz_confusion": confusion_matrix(y_real, y_pred).tolist(),
    }


def generar_graficos(modelo, X_prueba, y_prueba, carpeta: Path, nombre: str) -> None:
    carpeta.mkdir(parents=True, exist_ok=True)
    y_pred = modelo.predict(X_prueba)

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_prueba, y_pred, display_labels=["No deserta", "Deserta"], cmap="Blues", ax=ax)
    ax.set_title(f"Matriz de confusión - {nombre}")
    fig.tight_layout()
    fig.savefig(carpeta / "matriz_confusion.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_estimator(modelo, X_prueba, y_prueba, ax=ax, name=nombre)
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_title("Curva ROC")
    fig.tight_layout()
    fig.savefig(carpeta / "curva_roc.png", dpi=120)
    plt.close(fig)

    (carpeta / "reporte_clasificacion.txt").write_text(
        classification_report(y_prueba, y_pred, target_names=["No deserta", "Deserta"]),
        encoding="utf-8")


def importancia_variables(modelo, top: int = 15) -> list[dict]:
    """Importancia global según el clasificador (árboles)."""
    pre, clf = modelo.named_steps["pre"], modelo.named_steps["clf"]
    if not hasattr(clf, "feature_importances_"):
        return []
    nombres = pre.get_feature_names_out()
    imp = np.asarray(clf.feature_importances_, dtype=float)
    if imp.sum() > 0:
        imp = imp / imp.sum()
    orden = np.argsort(imp)[::-1][:top]
    return [{"variable": str(nombres[i]).split("__", 1)[-1], "importancia": round(float(imp[i]), 4)}
            for i in orden]


def analisis_shap(modelo, X_prueba: pd.DataFrame, carpeta: Path, muestras: int = 500) -> bool:
    """Gráfico resumen SHAP (si la librería está disponible)."""
    try:
        import shap
    except ImportError:
        print("[aviso] shap no está instalado; se omite el análisis SHAP.")
        return False
    try:
        pre, clf = modelo.named_steps["pre"], modelo.named_steps["clf"]
        X = X_prueba.sample(min(muestras, len(X_prueba)), random_state=0)
        Xt = pre.transform(X)
        nombres = [n.split("__", 1)[-1] for n in pre.get_feature_names_out()]
        explicador = shap.TreeExplainer(clf)
        valores = explicador.shap_values(Xt)
        if isinstance(valores, list):          # algunas versiones devuelven [clase0, clase1]
            valores = valores[1]
        elif getattr(valores, "ndim", 2) == 3:
            valores = valores[:, :, 1]
        shap.summary_plot(valores, Xt, feature_names=nombres, show=False, max_display=15)
        plt.tight_layout()
        plt.savefig(carpeta / "shap_resumen.png", dpi=120)
        plt.close("all")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[aviso] No se pudo generar SHAP: {exc}")
        return False
