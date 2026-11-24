# Modelo de Machine Learning — Riesgo de deserción

Proyecto independiente que entrena, compara y evalúa los modelos. El backend solo **carga** el modelo ya entrenado (`.pkl`).

| Fase de la metodología | Dónde está |
|---|---|
| 2.1 Recolección y exploración | `notebooks/01_exploracion_datos.ipynb`, `src/datos.py` |
| 2.2 Variables relevantes y faltantes | `src/config.py` (lista de variables), `datos.resumen_exploratorio` |
| 2.3 Datos inconsistentes y categóricas | `datos.limpiar` (rangos válidos), One-Hot en `src/modelos.py` |
| 2.4 Normalización y atípicos | `StandardScaler` en `src/modelos.py`, `datos.reporte_atipicos` (IQR) |
| 2.5 Balanceo de clases | SMOTE dentro del pipeline (solo en entrenamiento) |
| 2.6 Matriz de variables y división | `src/entrenar.py` (80 % entrenamiento / 20 % prueba, estratificado) |
| 3.4 Diseño del modelo | `src/config.py`, `src/modelos.py` |
| 4.4 Random Forest, Gradient Boosting, XGBoost, LightGBM | `src/modelos.py` |
| 4.5 Hiperparámetros + CV estratificada de 10 particiones | `RandomizedSearchCV` + `StratifiedKFold(10)` en `src/entrenar.py` |
| 5.2 Accuracy, Precision, Recall, F1, AUC-ROC | `src/evaluar.py` |
| 5.3 SHAP y matriz de confusión | `src/evaluar.py` → `reports/<versión>/` |

## Instalación (una sola vez)

Requiere Python 3.11 o superior.

```bash
py -m venv .venv
```

```bash
.venv\Scripts\activate
```

```bash
pip install -r requirements.txt
```

Copie `.env.example` como `.env` y ponga la contraseña de PostgreSQL.

## Entrenar

Con los datos ya cargados en la base `desercion_db` (lee la vista `vw_dataset_desercion`):

```bash
python -m src.entrenar
```

Opciones útiles:

- `--metrica recall` → elige el modelo que menos estudiantes en riesgo deja sin detectar (por defecto `f1`).
- `--algoritmos random_forest lightgbm` → entrena solo esos.
- `--csv data/mis_datos.csv` → entrena desde un CSV con las mismas columnas de la vista.
- `--rapido` → 5 pliegues y 3 combinaciones (solo para probar).

Resultado:

- `models/modelo_desercion_<versión>.pkl` → modelo entrenado (preprocesamiento + clasificador).
- `models/modelo_desercion_<versión>.json` → métricas, variables, hiperparámetros, comparación de algoritmos.
- `reports/<versión>/` → `comparacion_algoritmos.csv`, `matriz_confusion.png`, `curva_roc.png`, `shap_resumen.png`, `atipicos_iqr.csv`, `reporte_clasificacion.txt`.

Después, en la aplicación web: **Modelo ML → Registrar y activar** (o `POST /modelo/registrar`).

## Probar sin datos reales

Genera datos **ficticios** solo para comprobar que todo funciona (no usar en el informe):

```bash
python -m src.generar_datos_sinteticos --n 2000
```

```bash
python -m src.entrenar --csv data/sinteticos.csv --rapido
```

Borre luego el modelo de prueba de `models/` para no confundirlo con el real.

## Variable objetivo

`desercion = 1` cuando `estudiantes.estado = 'desertor'`. Se excluyen los estudiantes con estado `inactivo`. Defina con la institución qué cuenta como desertor (ver entrevista, Parte 2, bloque 2.5).

## Variables sensibles

`genero`, `victima_conflicto`, `pertenece_etnia` y `discapacidad` **no** se usan por defecto. Para agregar alguna, edite `VARIABLES_CATEGORICAS` / `VARIABLES_NUMERICAS` en `src/config.py` (y en ese caso añádala también a la vista SQL y al backend).

## Entrenar con los datos del servidor y publicar el modelo

Para entrenar en este PC con los datos de la base de datos **publicada** (Azure) y que el modelo aparezca en la web publicada:

```powershell
cd ml
.\entrenar_servidor.ps1 -Publicar
```

1. Abre un túnel SSH (`localhost:5434` → PostgreSQL del servidor). La base de datos no queda expuesta a internet.
2. Entrena con `python -m src.entrenar` (acepta las mismas opciones, por ejemplo `.\entrenar_servidor.ps1 --metrica recall`).
3. Con `-Publicar` copia el `.pkl` y el `.json` al servidor (`~/app/ml/models`).
4. En la web: **Modelo ML → Registrar y activar**. No hay que reiniciar nada.

Para publicar un modelo ya entrenado: `.\publicar_modelo.ps1` (el más reciente) o `.\publicar_modelo.ps1 modelo_desercion_AAAAMMDD_HHMMSS`.

La IP del servidor y la ruta de la clave SSH se configuran en `ml/servidor.ps1`. Las versiones de scikit-learn / XGBoost / LightGBM del `.venv` deben ser las mismas de `backend/requirements.lock.txt`; si no, el modelo no carga en el servidor.
