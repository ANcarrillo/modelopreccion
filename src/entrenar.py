"""Entrenamiento completo del modelo de deserción.

Recorre las fases 2.1 → 2.6, 4.4, 4.5 y 5.2 → 5.3 de la metodología:
  1. Carga datos (BD o CSV)               2. Limpieza y reporte de faltantes/atípicos
  3. División entrenamiento / prueba       4. Búsqueda de hiperparámetros con CV estratificada
  5. Comparación de algoritmos             6. Evaluación final + gráficos + SHAP
  7. Guarda el modelo (.pkl) y sus metadatos (.json) para el backend

Uso:
  python -m src.entrenar                      # lee la vista vw_dataset_desercion (DATABASE_URL)
  python -m src.entrenar --csv data/datos.csv  # lee un CSV con las mismas columnas
  python -m src.entrenar --csv data/sinteticos.csv --rapido   # prueba rápida
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime

import joblib
import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from . import config, datos, evaluar, modelos


def main() -> None:
    load_dotenv(config.RAIZ / ".env")
    p = argparse.ArgumentParser(description="Entrena y compara modelos de deserción")
    p.add_argument("--csv", help="Ruta a un CSV en lugar de la base de datos")
    p.add_argument("--metrica", default="f1", choices=["f1", "recall", "roc_auc", "precision", "accuracy"],
                   help="Métrica para elegir hiperparámetros y el mejor modelo")
    p.add_argument("--algoritmos", nargs="*", help="Limitar algoritmos (random_forest gradient_boosting xgboost lightgbm)")
    p.add_argument("--rapido", action="store_true", help="5 pliegues y 3 iteraciones (solo para pruebas)")
    args = p.parse_args()

    pliegues = 5 if args.rapido else config.PLIEGUES_CV
    iteraciones = 3 if args.rapido else config.ITERACIONES_BUSQUEDA

    # 2.1 Recolección
    df = datos.cargar_desde_csv(args.csv) if args.csv else datos.cargar_desde_bd()
    print(f"Registros cargados: {len(df)}")

    # 2.2 / 2.3 Exploración y limpieza
    df = datos.limpiar(df)
    resumen = datos.resumen_exploratorio(df)
    print("Resumen exploratorio:", json.dumps(resumen, ensure_ascii=False, default=str))
    clases = df[config.OBJETIVO].value_counts()
    if len(clases) < 2 or clases.min() < 5:
        raise SystemExit("Se necesitan al menos 5 estudiantes de cada clase (desertores y no desertores).")

    version = datetime.now().strftime("%Y%m%d_%H%M%S")
    carpeta_rep = config.DIR_REPORTES / version
    carpeta_rep.mkdir(parents=True, exist_ok=True)
    datos.reporte_atipicos(df).to_csv(carpeta_rep / "atipicos_iqr.csv", index=False)   # 2.4

    # 2.6 Matriz de variables y división estratificada
    variables = config.VARIABLES_NUMERICAS + config.VARIABLES_CATEGORICAS
    X, y = df[variables], df[config.OBJETIVO]
    X_ent, X_prueba, y_ent, y_prueba = train_test_split(
        X, y, test_size=config.TAMANO_PRUEBA, stratify=y, random_state=config.SEMILLA)

    # 4.4 / 4.5 Entrenamiento con búsqueda de hiperparámetros y CV estratificada
    cv = StratifiedKFold(n_splits=pliegues, shuffle=True, random_state=config.SEMILLA)
    minoritaria_pliegue = int(y_ent.value_counts().min() * (pliegues - 1) / pliegues)
    candidatos = modelos.algoritmos_candidatos()
    if args.algoritmos:
        candidatos = {k: v for k, v in candidatos.items() if k in args.algoritmos}

    resultados = {}
    for nombre, (estimador, espacio) in candidatos.items():
        print(f"\n>> Entrenando {nombre} ({pliegues} pliegues, {iteraciones} combinaciones)...")
        busqueda = RandomizedSearchCV(
            modelos.crear_pipeline(estimador, minoritaria_pliegue), espacio,
            n_iter=iteraciones, scoring=args.metrica, cv=cv, n_jobs=-1,
            random_state=config.SEMILLA, refit=True)
        busqueda.fit(X_ent, y_ent)
        resultados[nombre] = busqueda
        print(f"   {args.metrica} CV = {busqueda.best_score_:.4f}")

    # Comparación y selección del mejor
    tabla = []
    for nombre, b in resultados.items():
        prob = b.predict_proba(X_prueba)[:, 1]
        met = evaluar.calcular_metricas(y_prueba, (prob >= 0.5).astype(int), prob)
        tabla.append({"algoritmo": nombre, f"cv_{args.metrica}": round(b.best_score_, 4),
                      **{k: v for k, v in met.items() if k != "matriz_confusion"}})
    comparacion = pd.DataFrame(tabla).sort_values(f"cv_{args.metrica}", ascending=False)
    comparacion.to_csv(carpeta_rep / "comparacion_algoritmos.csv", index=False)
    print("\nComparación de algoritmos (prueba):\n", comparacion.to_string(index=False))

    mejor = comparacion.iloc[0]["algoritmo"]
    entrenado = resultados[mejor].best_estimator_
    # Se guarda sin SMOTE: el balanceo solo se usa al entrenar
    modelo_final = Pipeline([("pre", entrenado.named_steps["pre"]), ("clf", entrenado.named_steps["clf"])])

    # 5.2 / 5.3 Evaluación final
    prob = modelo_final.predict_proba(X_prueba)[:, 1]
    metricas = evaluar.calcular_metricas(y_prueba, (prob >= 0.5).astype(int), prob)
    evaluar.generar_graficos(modelo_final, X_prueba, y_prueba, carpeta_rep, mejor)
    evaluar.analisis_shap(modelo_final, X_prueba, carpeta_rep)

    # Valores de referencia para explicar predicciones en el backend
    referencia = {c: float(X_ent[c].median()) for c in config.VARIABLES_NUMERICAS if X_ent[c].notna().any()}
    referencia.update({c: X_ent[c].mode().iloc[0] for c in config.VARIABLES_CATEGORICAS if X_ent[c].notna().any()})

    config.DIR_MODELOS.mkdir(parents=True, exist_ok=True)
    archivo_pkl = config.DIR_MODELOS / f"modelo_desercion_{version}.pkl"
    joblib.dump(modelo_final, archivo_pkl)

    metadatos = {
        "nombre": "modelo_desercion",
        "version": version,
        "algoritmo": mejor,
        "archivo": archivo_pkl.name,
        "fecha_entrenamiento": datetime.now().isoformat(timespec="seconds"),
        "metrica_seleccion": args.metrica,
        "metricas": metricas,
        "hiperparametros": {k.replace("clf__", ""): (v if isinstance(v, (int, float, str, type(None))) else str(v))
                            for k, v in resultados[mejor].best_params_.items()},
        "variables": {"numericas": config.VARIABLES_NUMERICAS, "categoricas": config.VARIABLES_CATEGORICAS,
                      "booleanas": config.VARIABLES_BOOLEANAS},
        "valores_referencia": referencia,
        "umbrales_riesgo": config.UMBRALES_RIESGO,
        "importancia_global": evaluar.importancia_variables(modelo_final),
        "comparacion": comparacion.to_dict(orient="records"),
        "registros": {"entrenamiento": int(len(X_ent)), "prueba": int(len(X_prueba)),
                      "balance": {str(k): int(v) for k, v in y.value_counts().items()}},
        "reportes": str(carpeta_rep.relative_to(config.RAIZ)),
    }
    (config.DIR_MODELOS / f"modelo_desercion_{version}.json").write_text(
        json.dumps(metadatos, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print(f"\nMejor algoritmo: {mejor}")
    print("Métricas en prueba:", {k: v for k, v in metricas.items() if k != "matriz_confusion"})
    print(f"Modelo guardado en: {archivo_pkl}")
    print(f"Reportes en: {carpeta_rep}")


if __name__ == "__main__":
    main()
