# Proyecto 1 — Clasificación con Machine Learning
### Predicción de nivel de ingresos a partir del censo de EE. UU. (Adult / Census Income Dataset)

**Curso:** Machine Learning
**Autor:** Samuel

## Descripción

Este proyecto aborda un problema de **clasificación binaria**: predecir si el ingreso anual de una persona
supera los 50.000 USD (`>50K`) o no (`<=50K`), a partir de variables demográficas y laborales extraídas del
censo de EE. UU. de 1994.

El desarrollo incluye:

- Análisis exploratorio de datos (EDA): distribuciones, outliers, correlaciones y balance de clases.
- Preprocesamiento: limpieza de valores faltantes, escalado de variables numéricas y codificación one-hot
  de variables categóricas.
- Entrenamiento y ajuste de hiperparámetros (`GridSearchCV`) con validación cruzada estratificada de 10 folds
  para dos modelos: **Regresión Logística** y **SVM**.
- Evaluación y comparación de modelos sobre un conjunto de prueba independiente, usando Accuracy, Precision,
  Recall y F1-score.

## Fuente de los datos

Becker, B. & Kohavi, R. (1996). *Adult* [Dataset]. UCI Machine Learning Repository.
https://doi.org/10.24432/C5XW20

El archivo `adult.csv` contiene 48.842 registros con 14 variables predictoras (6 numéricas, 8 categóricas)
más la variable de salida `income`.

## Estructura del notebook

1. Descripción de la base de datos
2. Carga de datos
3. Inspección inicial y limpieza básica
4. Análisis exploratorio de datos (EDA)
   - 4.1 Distribuciones de las variables (histogramas)
   - 4.2 Diagramas de caja / Matriz de correlación
   - 4.3 Distribución de la variable de salida
5. Preprocesamiento de los datos
6. Entrenamiento con validación cruzada de 10 folds
   - 6.1 Regresión Logística (búsqueda en malla + evaluación en test)
   - 6.2 SVM (búsqueda en malla + evaluación en test)
7. Comparación de modelos y selección del mejor
8. Conclusiones

## Requisitos

```
python >= 3.9
numpy
pandas
matplotlib
seaborn
scikit-learn
```

Instalación rápida:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## Cómo ejecutar

1. Colocar `adult.csv` en la misma carpeta que el notebook.
2. Abrir `Proyecto1_Clasificacion_Adult.ipynb` en Jupyter y ejecutar todas las celdas en orden.

> El paso de `GridSearchCV` para la Regresión Logística se entrena sobre el conjunto de entrenamiento completo;
> el de SVM se entrena sobre una submuestra de 3.000 registros para mantener el tiempo de cómputo razonable.
> El tiempo total de ejecución del notebook es de aproximadamente 2 minutos.

## Resultados

| Modelo | F1 (CV, 10 folds) | Accuracy (test) | Precision (test) | Recall (test) | F1 (test) |
|---|---|---|---|---|---|
| Regresión Logística | 0.6681 | 0.8432 | 0.7280 | 0.5866 | **0.6497** |
| SVM | 0.6574 | 0.8454 | 0.7443 | 0.5732 | 0.6477 |

El modelo seleccionado según F1-score en test es la **Regresión Logística**, con un desempeño muy similar
al del SVM pero con un costo computacional bastante menor.

## Conclusiones principales

- Ambos modelos logran un desempeño razonable prediciendo si el ingreso supera los 50.000 USD.
- El desbalance de clases (≈75%/25%) hace que Precision, Recall y F1 sean más informativas que la Accuracy
  simple.
- La Regresión Logística resulta más eficiente computacionalmente con un desempeño competitivo frente al SVM.
- **Limitaciones:** el SVM se entrenó sobre una submuestra de 3.000 registros; el dataset refleja patrones
  socioeconómicos de EE. UU. en 1994 y no debe extrapolarse a contextos actuales sin cautela.
- **Trabajo futuro:** probar Random Forest / Gradient Boosting, técnicas de balanceo de clases (SMOTE,
  `class_weight`), selección de variables, y ejecutar el ajuste de hiperparámetros del SVM sobre el conjunto
  completo de entrenamiento si se cuenta con más recursos computacionales.
