"""Configuración central del modelo (Fase 3.4 - Diseño del modelo de ML).

Las variables deben coincidir con las columnas de la vista `vw_dataset_desercion`
de la base de datos. El backend lee esta misma lista desde el archivo de metadatos
que se guarda junto al modelo, así que entrenamiento y predicción usan las mismas variables.
"""
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
DIR_MODELOS = RAIZ / "models"
DIR_REPORTES = RAIZ / "reports"
DIR_DATOS = RAIZ / "data"

# Columna objetivo: 1 = desertó, 0 = no desertó
OBJETIVO = "desercion"

# Variables numéricas (los booleanos se convierten a 0/1)
VARIABLES_NUMERICAS = [
    "edad",
    "puntaje_saber11",
    "semestre_cursado",
    "creditos_inscritos",
    "creditos_aprobados",
    "creditos_reprobados",
    "tasa_aprobacion",
    "promedio_periodo",
    "promedio_acumulado",
    "asistencia_pct",
    "materias_reprobadas",
    "materias_canceladas",
    "materias_repetidas",
    "estrato",
    "ingreso_familiar",
    "trabaja",
    "horas_trabajo_semana",
    "personas_a_cargo",
    "distancia_km",
    "tiene_beca",
    "apoyo_financiero",
]

# Variables categóricas (se codifican con One-Hot)
# Nota ética: 'genero' está disponible en la vista pero NO se usa por defecto.
# Agréguela aquí solo si los interesados lo aprueban (entrevista Fase 3, pregunta 16).
VARIABLES_CATEGORICAS = [
    "tipo_colegio",
    "tipo_vivienda",
    "zona_residencia",
    "nivel_educativo_padres",
]

VARIABLES_BOOLEANAS = ["trabaja", "tiene_beca", "apoyo_financiero"]

# Estados que se excluyen del entrenamiento (resultado desconocido o no aplica)
ESTADOS_EXCLUIDOS = ["inactivo"]

# Rangos válidos: valores fuera de rango se tratan como inconsistentes (-> NaN)
RANGOS_VALIDOS = {
    "edad": (14, 80),
    "puntaje_saber11": (0, 500),
    "semestre_cursado": (1, 14),
    "promedio_periodo": (0, 5),
    "promedio_acumulado": (0, 5),
    "asistencia_pct": (0, 100),
    "tasa_aprobacion": (0, 1),
    "estrato": (0, 6),
    "horas_trabajo_semana": (0, 80),
    "distancia_km": (0, 1000),
}

# Umbrales para clasificar la probabilidad en nivel de riesgo
UMBRALES_RIESGO = {"medio": 0.40, "alto": 0.70}

# Entrenamiento
SEMILLA = 42
TAMANO_PRUEBA = 0.20
PLIEGUES_CV = 10          # Validación cruzada estratificada de 10 particiones (Fase 4.5)
ITERACIONES_BUSQUEDA = 15  # Combinaciones de hiperparámetros probadas por algoritmo
