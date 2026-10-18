# Protocolo de implementación y evaluación experimental

**Proyecto:** Decodificación continua de la flexión de cinco dedos mediante señales electrocorticográficas (ECoG) y redes encoder–decoder 1D  
**Fuente de datos:** BCI Competition IV — Dataset 4  
**Sujetos:** S1, S2 y S3  
**Versión documental:** 1.0 — especificación inicial de implementación  
**Fecha:** 18 de octubre de 2026  
**Entorno previsto:** Python, PyTorch, MNE-Python, scikit-learn, Optuna y Google Colab Pro  
**Estado:** diseño de implementación; pendiente de verificación técnica integral y de definición posterior del número de épocas.

> **Principio rector.** Implementar una sola cadena reproducible de análisis, limpieza, procesamiento, entrenamiento y evaluación para los tres sujetos. Antes de iniciar la optimización bayesiana se ejecutará una verificación técnica integral. Si todas las pruebas de esa verificación son satisfactorias, se congelarán el control de calidad, el preprocesamiento, la alineación temporal, las particiones, las métricas y la arquitectura parametrizable. Si alguna prueba falla, se corregirá únicamente el componente afectado, se regenerarán sus dependencias y se repetirán las pruebas pertinentes. **El número de épocas de entrenamiento no se determina en este documento**: se fijará posteriormente, con las señales ya verificadas y los recursos de Google Colab Pro.

---

## 1. Objetivo y alcance

Implementar un sistema de regresión que estime continuamente la flexión de cinco dedos a partir de las características temporoespectrales de las señales ECoG. Cada sujeto tendrá su propio modelo y sus propios hiperparámetros seleccionados, sin entrenar un modelo conjunto entre sujetos.

La investigación comparará tres configuraciones bajo un protocolo común:

| Variante | Arquitectura | Hiperparámetros de entrenamiento | Finalidad |
|---|---|---|---|
| **A. Modelo base** | U-Net 1D de referencia | Valores de referencia fijos | Establecer una comparación controlada |
| **B. Optimización de entrenamiento** | Misma U-Net de referencia | `learning_rate` y `weight_decay` optimizados | Medir el efecto de la primera etapa |
| **C. Optimización de arquitectura** | U-Net parametrizable optimizada | Valores seleccionados en B y mantenidos fijos | Medir el efecto adicional de la segunda etapa |

A, B y C deberán utilizar exactamente las mismas definiciones de datos válidos, limpieza, transformación, pérdida, métricas, semillas comparables y evaluación final. La única diferencia entre A y B será la selección de los hiperparámetros de entrenamiento; entre B y C, la selección arquitectónica.

### 1.1. Decisiones establecidas

- Banda de ECoG: **40–200 Hz**.
- Características: **40 frecuencias Morlet** entre 40 y 200 Hz.
- Frecuencia de trabajo: **100 Hz**.
- Arquitectura: **U-Net 1D** con **BatchNorm1d**, **GELU**, *dropout* y conexiones de salto.
- Decoder: **interpolación lineal seguida de Conv1d**, no convoluciones transpuestas.
- Optimizador de pesos: **AdamW**, sin *learning-rate scheduler* en la comparación principal.
- Pérdida: **0.5 MSE + 0.5 (1 − similitud coseno)**.
- Métrica principal: **Pearson macro de los cinco dedos**, calculado sobre trayectorias temporales reconstruidas.
- Optimización bayesiana: **Optuna con TPESampler** en dos etapas, por sujeto.
- Control de calidad: **detección y revisión en EDA; aplicación reproducible de decisiones de limpieza en preprocesamiento**.

### 1.2. Parámetros deliberadamente no cerrados

- `SEARCH_EPOCHS`: número fijo de épocas de cada entrenamiento de búsqueda.
- `FINAL_MAX_EPOCHS`: límite para selección de época durante entrenamiento final.
- `PURGE_SAMPLES`: separación temporal efectiva entre conjuntos, derivada de la verificación de soporte temporal de las transformaciones.
- `BATCH_SIZE`: valor común de ejecución, inicialmente candidato a 128, condicionado a la memoria real disponible para las 27 arquitecturas en Colab Pro.
- Umbrales exactos del QC y la lista de canales/interválicos a excluir: se determinarán mediante el EDA de señales crudas y se aprobarán **antes** de evaluar modelos.

Los parámetros pendientes deben registrarse explícitamente como `null` o `pending` hasta su aprobación; no utilizar valores provisionales como si fueran decisiones definitivas.

---

## 2. Organización del proyecto

Los notebooks dirigirán las etapas y presentarán resultados; los módulos Python contendrán la lógica reutilizable. Una función de procesamiento o entrenamiento debe tener una única implementación, utilizada tanto en las comprobaciones de factibilidad como en las optimizaciones y evaluaciones.

```text
TT-ECoG/
├── data/
│   ├── raw/                            # Archivos originales inmutables
│   ├── processed/                      # Datos transformados, versionados
│   └── manifests/                      # Procedencia, particiones y checksums
├── notebooks/
│   ├── 01_exploracion_ecog.ipynb        # EDA y detección de calidad
│   ├── 02_preprocesamiento_ecog.ipynb   # Limpieza y características
│   ├── 03_verificacion_pipeline.ipynb   # Pruebas de aprobación
│   ├── 04_factibilidad_entrenamiento.ipynb
│   ├── 05_optimizacion_etapa1.ipynb
│   ├── 06_optimizacion_etapa2.ipynb
│   └── 07_evaluacion_final.ipynb
├── src/tt_ecog/
│   ├── io.py
│   ├── quality_control.py
│   ├── preprocessing.py
│   ├── splits.py
│   ├── datasets.py
│   ├── model.py
│   ├── losses.py
│   ├── metrics.py
│   ├── validation.py
│   ├── training.py
│   ├── optimization.py
│   └── reporting.py
├── configs/
│   └── experiment_config.yaml          # Única configuración declarativa oficial
├── results/
│   ├── eda_qc/
│   ├── verification/
│   ├── feasibility/
│   ├── optuna/
│   └── final/
├── tests/
│   ├── test_quality_control.py
│   ├── test_preprocessing.py
│   ├── test_alignment.py
│   ├── test_splits.py
│   ├── test_no_leakage.py
│   ├── test_overlap_add.py
│   ├── test_architecture.py
│   └── test_metrics.py
├── docs/
│   ├── implementation_protocol.md
│   ├── quality_decisions.md
│   └── experimental_log.md
├── requirements.txt
└── README.md
```

**Convenciones:** rutas configurables, identificadores únicos de ejecución, semillas registradas, archivos `.mat` originales de solo lectura, artefactos derivados identificados por hash de configuración y ningún secreto embebido en el repositorio.

---

## 3. Datos, unidades y esquema temporal

### 3.1. Entradas y objetivos

- Señales ECoG con frecuencia nominal de muestreo original de **1000 Hz**.
- Cinco trayectorias continuas de flexión del guante, nominalmente a **25 Hz**.
- Tres sujetos tratados de forma independiente: S1, S2 y S3.
- Desarrollo y prueba se mantendrán separados. Las etiquetas de prueba no participarán en análisis, decisiones de calidad, selección de épocas ni selección de hiperparámetros.
- Al cargar cada archivo, verificar explícitamente las dimensiones reales, unidades, duraciones, orden de canales, tiempos iniciales y presencia de valores no finitos; no suponer que todos los archivos tienen la misma cantidad de electrodos.

### 3.2. Convenciones de índices

Los índices de canal almacenados para procesamiento serán **cero-basados** y se distinguirán de su presentación humana. Se registrarán ambas convenciones cuando se muestren gráficas o tablas.

El tiempo se representará en **segundos del registro original** además de índices de muestra. Todas las transformaciones y recortes conservarán un mapeo al tiempo de origen.

### 3.3. Esquema experimental de separación

El conjunto de desarrollo de cada sujeto se dividirá en **cinco bloques temporales cronológicos de longitud aproximada equivalente**: B1, B2, B3, B4 y B5.

- Para validar cada bloque se entrenará con las porciones restantes **que no invadan las regiones de seguridad**.
- En la **etapa 1** se puntuarán los folds con validación B2 y B4, seleccionados de la partición común de cinco bloques.
- En la **etapa 2** se puntuarán los cinco folds.
- Los conjuntos disjuntos de entrenamiento no se concatenarán en el eje temporal antes de crear ventanas; se construirán datasets separados por segmento.
- Esta estrategia es **validación cruzada temporal por bloques con purga**, no *walk-forward*: el entrenamiento puede incluir segmentos anteriores y posteriores al bloque validado. La interpretación científica debe reflejarlo.

La separación de seguridad comenzará con **256 muestras a 100 Hz como candidato de verificación**, pero su tamaño aprobado se derivará del soporte temporal del filtrado, las wavelets, la decimación y el desplazamiento de etiquetas. No se considerará suficiente únicamente por ser igual a la longitud de ventana.

---

## 4. Paso 1 — Exploración de datos y control de calidad (EDA + QC)

**Responsables:** `01_exploracion_ecog.ipynb` y `quality_control.py`.  
**Entrada:** señales ECoG crudas del conjunto de desarrollo.  
**Salida:** diagnósticos, visualizaciones y un archivo de decisiones de limpieza auditado.

El EDA **no modificará** los archivos originales. Su objetivo es describir las señales, localizar anomalías de adquisición y decidir reglas de tratamiento basadas en evidencia ECoG, no en el desempeño predictivo.

### 4.1. Inspección obligatoria para los tres sujetos

| Inspección | Medidas o visualizaciones | Finalidad |
|---|---|---|
| Integridad | Dimensiones, frecuencias, tiempos, NaN, infinitos | Detectar archivos o canales incorrectos |
| Amplitud | Mediana, MAD, máximos, cuantiles, rangos por canal | Identificar valores extremos |
| Discontinuidades | Diferencia entre muestras, saltos y recuperación | Detectar artefactos transitorios |
| Señal plana | Desviación local y duración de segmentos constantes | Detectar pérdida de señal |
| Distribución temporal | Ventanas por canal, percentiles por intervalo | Identificar cambios de régimen |
| Espectro | PSD y energía en banda; ruido de línea | Detectar ruido espectral atípico |
| Comparación entre canales | Distribuciones y actividad simultánea | Distinguir problemas localizados y globales |
| Impacto temporal | Inicio, fin, duración, fracción afectada | Decidir alcance de la intervención |

La proximidad de índices **no permite asumir proximidad física** entre electrodos sin un mapa de posiciones.

### 4.2. Detectores reutilizables

Implementar detectores independientes del modelo:

1. **Extremos robustos:** diferencia frente a la mediana, normalizada por MAD con tratamiento seguro de MAD casi nula.
2. **Saltos:** primera diferencia temporal robustamente anómala.
3. **Planitud:** ventanas con variabilidad extremadamente reducida.
4. **Ruido espectral:** anomalías marcadas y persistentes frente a la distribución de referencia de los canales.
5. **Eventos agrupados:** consolidación de detecciones próximas para obtener episodios con comienzo, fin y duración.

Los umbrales serán **candidatos técnicos configurables**, no disparadores automáticos e irrevocables de exclusión. Una detección dispara una revisión documentada. Los criterios se congelarán antes de iniciar el entrenamiento comparativo.

Se prohíbe elegir umbrales, ventanas o canales basándose en el Pearson, en errores de predicción o en estadísticas calculadas con las etiquetas de prueba.

### 4.3. Decisión de tratamiento

La intervención se seleccionará por calidad de adquisición y podrá ser:

- **Conservar:** detección falsa o variación fisiológica plausible sin evidencia de defecto.
- **Excluir un canal completo:** canal con defectos graves, repetidos o no tratables de manera fiable. Se excluirá del sujeto de forma consistente en desarrollo, validación y prueba.
- **Enmascarar intervalos:** episodios localizados; las máscaras deben expandirse al soporte temporal de las transformaciones y aplicarse **antes** de definir muestras válidas para evaluar.
- **Reparación local:** solo con una regla de reconstrucción documentada, validada técnicamente y aprobada antes de su aplicación. Nunca concatenar intervalos separados como si fueran continuos.

La política principal favorecerá intervenciones simples y auditables, sin preservar artificialmente muestras contaminadas para obtener mejores puntuaciones. Si se decide excluir un canal, la lista aprobada se aplicará **antes** de calcular la referencia espacial, filtrar o realizar Morlet.

### 4.4. Detección frente a calibración por fold

Es válido realizar una inspección descriptiva de **todo el desarrollo** para establecer un mapa fijo de calidad de adquisición por sujeto, siempre que no se utilicen etiquetas ni métricas predictivas. No obstante, como esa inspección también observa señales que después actuarán como validación, **no debe presentarse como una política de QC completamente anidada dentro de cada fold**.

Toda operación que **aprenda estadísticas** para transformar entradas o detectar anomalías adaptativamente dentro de CV se ajustará únicamente con el entrenamiento del fold correspondiente. La lista de exclusiones de adquisición se fijará una vez y no se modificará en función de los resultados de cada fold.

No se inspeccionará el conjunto de prueba para modificar los criterios de QC. Los procedimientos definidos se aplicarán posteriormente sin recalibración a partir de sus etiquetas o resultados.

### 4.5. Entregables del EDA

```text
results/eda_qc/
  qc_channels.csv
  qc_events.csv
  qc_summary.csv
  qc_decisions.json
  qc_figures/
  qc_review_log.md
```

`qc_decisions.json` contendrá por sujeto índices cero-basados, acción, justificación, intervalos afectados, detector que originó la marca, fecha de revisión y estado `approved`.

**Criterio de salida del paso 1:** los tres sujetos han sido inspeccionados con el mismo procedimiento, las decisiones están aprobadas y los datos originales permanecen intactos.

---

## 5. Paso 2 — Limpieza y preprocesamiento ECoG

**Responsables:** `02_preprocesamiento_ecog.ipynb`, `preprocessing.py`, `splits.py` y `datasets.py`.  
**Entrada:** datos crudos, decisiones QC aprobadas, parámetros de procesamiento y esquema de folds.  
**Salida:** datos y máscaras temporales versionados por sujeto, fold y partición.

### 5.1. Orden de procesamiento

| Orden | Operación | Restricción |
|---|---|---|
| 1 | Cargar y validar ECoG y flexión originales | Datos de origen inmutables |
| 2 | Crear particiones temporales y coordenadas válidas | Registrar índices originales y segundos |
| 3 | Aplicar exclusiones o máscaras del QC | Antes de filtrado, referencia común y Morlet |
| 4 | Separar segmentos independientes con guardas temporales | No filtrar conjuntos concatenados a través de una frontera |
| 5 | Aplicar referencia de mediana común instantánea (*Common Median Reference*) | Calcular con los canales efectivamente conservados |
| 6 | Filtrar la banda **40–200 Hz** y rechazo de línea **60, 120, 180 Hz** | Registrar tipo, orden, longitud, fase y padding del filtro |
| 7 | Transformada Morlet en **40 frecuencias** de 40–200 Hz | Usar 7 ciclos y `zero_mean=True` como especificación |
| 8 | Reducir de 1000 a **100 Hz**, con antialiasing | Mantener mapeo preciso de tiempos |
| 9 | Interpolar las cinco etiquetas de 25 a **100 Hz** | Usar tiempos reales de adquisición |
| 10 | Alinear ECoG y flexión con el retardo fijo | Ningún par temporal cruza una frontera |
| 11 | Ajustar escaladores del fold con entrenamiento | Prohibido hacer `fit` con validación o test |
| 12 | Transformar train/validación y crear ventanas | Solo ventanas enteras dentro de segmentos válidos |
| 13 | Persistir características, máscaras, escaladores y manifiestos | Hash de la configuración y de cada partición |

**Precisión temporal:** si una implementación cambia el orden operativo de la decimación o de Morlet, deberá demostrar equivalencia temporal y espectral; no se asumirá equivalencia por tener la misma frecuencia de salida.

### 5.2. Referencia espacial y normalización

- Calcular **mediana común instantánea** sobre canales aceptados. La mediana es una operación espacial en cada instante, sin estimar estadísticas temporales del registro completo.
- **No** realizar z-score global por canal sobre todo desarrollo antes de crear folds.
- Escalar las características ECoG mediante:

```python
RobustScaler(quantile_range=(10, 90), unit_variance=True)
```

- Escalar las etiquetas mediante:

```python
MinMaxScaler()
```

- Ajustar ambos escaladores con entrenamiento del fold; almacenar sus parámetros y aplicar exclusivamente `transform()` a validación.
- En el entrenamiento final, ajustar los escaladores usando solamente el desarrollo habilitado para ese entrenamiento; aplicarlos sin `fit()` sobre el conjunto de prueba.
- Para ajustar el escalador de entradas, reorganizar temporalmente cada tensor `(C, 40, T)` como `(T, C×40)` y escalar cada combinación canal–frecuencia con estadísticas de los tiempos **de entrenamiento**. Conservar el mismo orden de columnas al transformar validación o prueba. Para el guante, ajustar sobre `(T, 5)`.
- Los percentiles indicados son **10 y 90**, no 0.1 y 0.9.
- El escalado robusto **no equivale** a recortar valores extremos. No introducir clipping arbitrario como sustituto del QC.

### 5.3. Transformada Morlet

- Frecuencias: 40 puntos espaciados logarítmicamente entre 40 y 200 Hz, con parámetros y arreglo de frecuencias persistidos.
- Estructura de salida: **`(n_canales_efectivos, 40, n_tiempos)`**, por segmento.
- Definir y documentar la magnitud usada como entrada (por ejemplo, potencia `|W|²`), el tipo numérico y cualquier transformación adicional. **No mezclar magnitud y potencia entre sujetos o experimentos**.
- Verificar que la elección de frecuencia/filtrado no produce valores no finitos ni una contaminación no controlada en los bordes.

### 5.4. Alineación temporal

Fijar **200 ms de adelanto del objetivo** respecto a la muestra ECoG. A 100 Hz corresponden a 20 muestras:

\[
X(t) \longrightarrow Y(t+0.2\,\mathrm{s})
\]

El mapeo no debe cambiar entre entrenamiento, validación o evaluación final. Las primeras o últimas muestras sin pareja temporal válida se excluirán de forma determinista y documentada, sin desplazar la línea temporal de salida de manera ambigua.

**Restricción:** una etiqueta situada en otro fold o fuera del tramo permitido no puede utilizarse para un ejemplo cuya ECoG pertenezca al fold actual. La comprobación se realizará en coordenadas temporales originales.

**Alcance:** este desplazamiento se considera una condición experimental fija; no demuestra por sí solo predicción causal en tiempo real, porque las convoluciones y los filtros pueden utilizar contexto futuro.

### 5.5. Filtrado acausal, guardas y purga

Las convoluciones del filtrado, las wavelets y la decimación pueden utilizar muestras vecinas de ambos lados del instante analizado. Se implementará una política estricta para impedir influencia de validación sobre las características de entrenamiento y viceversa.

- Delimitar particiones **antes** de ejecutar operaciones que mezclen contexto temporal entre conjuntos.
- Procesar segmentos separados o utilizar guardas equivalentes, comprobadas con perturbaciones sintéticas.
- Medir el soporte temporal **efectivo de toda la composición**: filtros, Morlet, antialiasing, recorte y alineación; registrar las tolerancias numéricas utilizadas.
- Excluir del entrenamiento y de la evaluación los bordes donde no exista contexto temporal válido.
- Fijar `PURGE_SAMPLES` solamente después de la prueba de aislamiento del paso 3.
- Nunca usar el Pearson para escoger longitud de purga o reducir bordes.

### 5.6. Dataset de ventanas

- Longitud de ventana: **256 muestras**, equivalentes a 2.56 s a 100 Hz.
- Deslizamiento de entrenamiento: **1 muestra** dentro de cada segmento continuo aceptado.
- Para un segmento de longitud `T`, número de ventanas con paso `s`: `0` si `T < 256`; de otro modo `1 + floor((T - 256)/s)`.
- No crear ventanas que crucen cambios entre entrenamiento, purga, validación o intervalos inválidos.
- Entradas para la red: `(batch, canales, 40, 256)`; etiquetas: `(batch, 5, 256)`.

### 5.7. Archivos de salida

```text
data/processed/<pipeline_hash>/
  S1/
    B1/ B2/ B3/ B4/ B5/
  S2/
    B1/ B2/ B3/ B4/ B5/
  S3/
    B1/ B2/ B3/ B4/ B5/
  manifests/
    qc_decisions.json
    preprocessing_config.json
    split_manifest.json
    filter_support.json
    processed_manifest.json
```

Cada fold conservará escaladores entrenados, rangos temporales originales, máscaras de muestras válidas, lista de canales, forma de tensores, total de ventanas y procedencia reproducible.

**Criterio de salida del paso 2:** generación correcta y trazable de características de los tres sujetos. La salida no se considera aprobada hasta completar el paso 3.

---

## 6. Paso 3 — Verificación técnica integral (puerta de aprobación)

**Responsables:** `03_verificacion_pipeline.ipynb` y pruebas automatizadas de `tests/`.  
**Objeto:** demostrar que el pipeline implementa la metodología descrita, sin seleccionar configuraciones a partir del rendimiento del modelo.

### 6.1. Matriz de aceptación

| Prueba | Verificación requerida | Criterio de aprobación |
|---|---|---|
| **V01. Integridad** | Hash y formato de los datos; frecuencia, dimensiones y tipos | Los originales permanecen inmutables y las lecturas son coherentes |
| **V02. QC consistente** | Aplicación de exclusiones/máscaras por sujeto | Decisiones aprobadas aplicadas antes de transformaciones; ningún evento grave queda sin acción o justificación |
| **V03. Referencia espacial** | Mediana común instantánea | Coincide con una implementación de referencia sobre señales sintéticas |
| **V04. Separación temporal** | Bloques B1–B5, índices, tiempos y purga | No hay intersección indebida entre pares o contextos temporales de train y validación |
| **V05. Aislamiento de escaladores** | Perturbar validación y reconstruir fold | Escaladores y entradas de entrenamiento permanecen invariantes |
| **V06. Alineación** | Señales sintéticas con marcas temporales únicas | `ECoG(t)` corresponde a `flexión(t + 20 muestras)` en train/validación/prueba |
| **V07. Aislamiento del filtrado** | Impulso en validación | No modifica las características **válidas** de entrenamiento, dentro de tolerancia documentada |
| **V08. Aislamiento inverso** | Impulso en entrenamiento | No modifica características **válidas** de validación antes del escalado |
| **V09. Morlet/decimación** | Frecuencias, tiempos, antialiasing, bordes | Salida a 100 Hz y soporte/márgenes técnicamente consistentes |
| **V10. Ventanas** | Casos sintéticos de longitudes y segmentación | Conteo correcto; ninguna ventana cruza límites o regiones inválidas |
| **V11. Finitud y shapes** | Tensores de tres sujetos, folds y máscaras | Sin NaN/inf inesperados; tamaños esperados; escala y extremos registrados |
| **V12. Reconstrucción temporal** | Overlap-add con solapamiento y última ventana | Cada instante válido evaluado tiene cobertura y alineación exactas |
| **V13. Arquitectura** | Forward y backward de las 27 combinaciones por sujeto | Salida de cinco dedos con longitud 256; gradientes finitos; sin errores de dimensiones |
| **V14. Métricas y trazabilidad** | Pearson manual, pérdida, hashes y versiones | Coincidencia numérica; manejo explícito de indefinidos y evidencia completa |

**Pruebas complementarias:** reproducibilidad de resultados con una semilla en un entorno controlado, equivalencia entre procesamiento completo y por fragmentos dentro de tolerancias, capacidad de cargar configuraciones y escaladores guardados, y ausencia de acceso al test desde las rutinas de optimización.

### 6.2. Política del Pearson

Definir para cada dedo:

\[
r_d=\operatorname{corr}(y_d,\widehat y_d),\qquad
r_{\mathrm{macro}}=\frac{1}{5}\sum_{d=1}^{5}r_d.
\]

- Calcular sobre la **trayectoria reconstruida**, no el promedio de Pearson de batches o ventanas.
- Utilizar los mismos instantes válidos predefinidos para las cinco salidas.
- Detectar explícitamente series constantes, varianzas próximas a cero y resultados no finitos. Una evaluación indefinida debe marcarse inválida con diagnóstico; **no** eliminar dedos silenciosamente del promedio.
- Registrar el valor de cada dedo y el promedio de forma independiente.
- Ninguna prueba del paso 3 puede aprobarse por alcanzar un umbral arbitrario de Pearson predictivo.

### 6.3. Decisión tras el paso 3

**Caso A — Todas las verificaciones pasan.**

1. Guardar el reporte firmado de verificación y los hashes de todos los artefactos.
2. Congelar **QC, limpieza, referencia espacial, filtrado, Morlet, alineación, escalado, particiones, inferencia, definición de métricas y arquitectura parametrizable**.
3. No modificar esos componentes para intentar aumentar Pearson.
4. Continuar a la medición de factibilidad computacional y, posteriormente, a la fijación del número de épocas.

**Caso B — Alguna verificación falla.**

1. Identificar la prueba y el componente responsable.
2. Corregir **solo** ese componente y sus dependencias estrictamente necesarias.
3. Regenerar únicamente los datos o artefactos afectados.
4. Ejecutar de nuevo la prueba fallida y las pruebas de regresión relacionadas.
5. Registrar el motivo del cambio, versión y resultados; repetir hasta aprobar el paso 3.

Un error real descubierto después de la congelación debe tratarse como una incidencia versionada: invalidar los resultados afectados y repetirlos. No efectuar cambios silenciosos.

Para ejecutar V13 y V14 se implementará primero la arquitectura parametrizable mínima, la pérdida y las métricas, **sin efectuar entrenamientos experimentales**. Esta dependencia de implementación no adelanta las búsquedas ni la decisión de épocas.

**Archivos de aprobación:** `verification_report.json`, `verification_summary.md`, `test_results.xml`, `filter_support.json`, `approved_pipeline_manifest.json`.

---

## 7. Modelo U-Net 1D definitivo

### 7.1. Entrada y salida

- Entrada por batch: `(B, C_efectivos, 40, 256)`.
- Reducir dimensiones canal–frecuencia mediante una operación entrenable compartida a lo largo del tiempo para obtener características de forma `(B, F, 256)`.
- Encoder con convoluciones dilatadas unidimensionales, BatchNorm1d, GELU, Dropout y reducción temporal.
- Decoder simétrico con interpolación lineal hasta la longitud exacta del *skip*, Conv1d, normalización y concatenación de *skip connections*.
- Proyección final con Conv1d de kernel 1 para obtener `(B, 5, 256)`.

### 7.2. Parámetros de referencia

| Parámetro | Valor de referencia |
|---|---|
| Niveles del encoder | 5 |
| Filtros base | 64 |
| Canales por nivel | `[64, 64, 128, 256, 512, 512]` (incluye representación profunda) |
| Kernels del encoder | `[7, 7, 5, 5, 5]` |
| Dilataciones | `[1, 2, 3, 1, 2]` |
| Reducción temporal por nivel | Factor 2 |
| Activación | GELU |
| Normalización | BatchNorm1d |
| Dropout | 0.3 |
| Decoder | Interpolación lineal + Conv1d; longitud fijada por el *skip* |

Se aplicará **un mismo criterio de inicialización** a todas las variantes, sin reutilizar pesos de modelos anteriores entre folds o trials.

### 7.3. Parametrización arquitectónica

Sea `L = encoder_levels`, `f = base_filters`, `k = kernel_size`:

- `L ∈ {3,4,5}`.
- `f ∈ {32,64,96}`.
- `k ∈ {3,5,7}`.
- Canales: tomar los primeros `L+1` elementos de `[f, f, 2f, 4f, 8f, 8f]`.
- Kernels: `[7,7] + [k] * (L-2)`, con correspondientes kernels del decoder definidos de forma consistente con los niveles espejo.
- Dilataciones: primeros `L` elementos de `[1,2,3,1,2]`.
- Todos los demás componentes permanecen fijos.

Esto define **3 × 3 × 3 = 27 configuraciones** discretas. La combinación `(k=5, L=5, f=64)` reproduce la arquitectura de referencia dentro de esta parametrización.

**Restricción de consistencia:** no cambiar simultáneamente decoder, dropout, estrategia de normalización o función de pérdida como parte de la búsqueda, porque añadiría factores no contemplados en el espacio experimental.

---

## 8. Objetivo de entrenamiento, evaluación y checkpoint

### 8.1. Función de pérdida

Utilizar, por ventana de salida:

\[
\mathcal L = 0.5\,\operatorname{MSE}(\hat Y,Y)
+0.5\left(1-\operatorname{CosSim}_{\mathrm{temporal}}(\hat Y,Y)\right).
\]

La similitud coseno se calculará **a lo largo del eje temporal de cada dedo**, promediando después entre dedos y ejemplos del batch, con un `eps` explícito para estabilidad numérica. Esta pérdida sirve para actualizar pesos mediante retropropagación.

La optimización de hiperparámetros **no** utiliza esta pérdida como objetivo principal: maximiza Pearson de validación tras reconstrucción de las trayectorias.

### 8.2. Optimizador

- AdamW.
- Learning rate constante durante el entrenamiento del experimento principal; **sin scheduler**.
- Aplicar una misma política documentada de `weight_decay` a los parámetros de la red en A, B y C. Se fijará para la implementación principal la configuración simple de AdamW que aplica `weight_decay` a todos los parámetros entrenables; cualquier cambio de grupos de parámetros antes de la congelación deberá quedar reflejado en la configuración y pruebas.
- Inicializar aleatoriedad por sujeto, fold y ejecución, con semillas predefinidas y registradas.

### 8.3. Métrica de validación

Durante inferencia, formar ventanas de 256 muestras con **stride inicial 64**, incluir la última ventana necesaria para cubrir el final y reconstruir con **overlap-add** mediante promedio ponderado por cobertura.

La puntuación principal `val_pearson_full` será el promedio de Pearson de los cinco dedos sobre los instantes válidos de la trayectoria completa. Conservar métricas por dedo y pérdidas auxiliares separadamente.

### 8.4. Selección de checkpoints

**Durante la búsqueda con TPE:** cada trial/fold recibe un presupuesto de épocas fijo, igual para todos. No se aplicará Early Stopping basado en el mismo bloque utilizado para puntuar el trial. Se puntuará el modelo de la **última época del presupuesto**.

**Durante el entrenamiento final:** utilizar una validación interna de desarrollo con Early Stopping monitorizando `val_pearson_full`. Seleccionar el número de épocas mediante ese procedimiento y, de ser viable, reinicializar y entrenar sobre el desarrollo completo durante la duración seleccionada. Aplicar la misma regla a A, B y C.

La validación utilizada para elegir el checkpoint **no** constituye una estimación independiente de desempeño final.

---

## 9. Paso 4 — Factibilidad en Colab Pro y definición posterior de épocas

Este paso se realiza **después de aprobar el paso 3** y **antes de iniciar las optimizaciones TPE**.

### 9.1. Recursos

- Registrar la GPU realmente asignada por Colab Pro, memoria física, CUDA, versiones de PyTorch y duración de cada sesión.
- Verificar al menos arquitecturas representativas pequeñas, medianas y grandes, incluyendo el caso de mayor consumo (`L=5`, `f=96`).
- Medir memoria pico de PyTorch **durante varios pasos completos de forward + backward + actualización**, no únicamente en un forward.
- Utilizar batch size **128 como candidato inicial**. Si alguna arquitectura no cabe, elegir **antes de la búsqueda** un batch común viable para todas las configuraciones o documentar otra política controlada; no variar silenciosamente el batch entre trials.
- Medir el tiempo por época para estimar el presupuesto total de cómputo. Colab Pro puede asignar GPUs diferentes en sesiones distintas.

### 9.2. Decisión pendiente: número de épocas

El protocolo **no define actualmente** cuántas épocas utilizarán las búsquedas ni el máximo final.

Para cerrar esta decisión:

1. Utilizar exclusivamente el pipeline de datos ya aprobado.
2. Ejecutar un análisis acotado de curvas de aprendizaje con varias tasas de aprendizaje representativas del rango previsto.
3. Registrar por época pérdida de entrenamiento, Pearson de validación reconstruida, tiempo y memoria.
4. Comprobar que las tasas bajas dispongan de una oportunidad razonable de converger y que la duración no favorezca únicamente aprendizajes rápidos.
5. Seleccionar y documentar un **mismo `SEARCH_EPOCHS`** para todas las configuraciones de ambas etapas, con justificación de costo y convergencia.
6. Definir separadamente `FINAL_MAX_EPOCHS` y la política de Early Stopping final.

**No adoptar automáticamente 20, 30, 40 ni ninguna otra cifra** sin finalizar este análisis. El valor aprobado se añadirá a `configs/experiment_config.yaml`; esta decisión **no reabre** el diseño de preprocesamiento ya verificado.

---

## 10. Paso 5 — Optimización bayesiana secuencial

### 10.1. Requisitos de inicio

No iniciar búsquedas hasta contar con: QC aprobado, pipeline congelado, las 14 verificaciones completas, presupuesto de épocas decidido y factibilidad de memoria/tiempo comprobada.

Optuna se configurará con almacenamiento persistente por sujeto y etapa, un generador pseudoaleatorio con semilla y registros de ejecuciones interrumpidas. Evitar sesiones que dependan exclusivamente de almacenamiento efímero.

### 10.2. Etapa 1: hiperparámetros de entrenamiento

| Campo | Especificación |
|---|---|
| Variables | `learning_rate`, `weight_decay` |
| `learning_rate` | Log-uniforme entre `1e-5` y `1e-3` |
| `weight_decay` | Log-uniforme entre `1e-6` y `1e-2` |
| Arquitectura | U-Net de referencia, fija |
| Sujeto | Estudio independiente para S1, S2 y S3 |
| Ensayos | **20 trials por sujeto** |
| Folds por ensayo | **B2 y B4** |
| Optimizador bayesiano | `TPESampler` con semilla documentada y `n_startup_trials=5` |
| Entrenamiento | `SEARCH_EPOCHS` fijas, sin Early Stopping |
| Objetivo | Media del `val_pearson_full` de ambos folds |

Presupuesto: `20 × 2 × 3 = 120` entrenamientos.

**Modelo base A:** utilizar la arquitectura de referencia con `learning_rate = 8.42e-5` y `weight_decay = 1e-4` como valores de referencia fijados de antemano. **Modelo B:** seleccionar por sujeto el mejor par de la etapa 1. La evaluación final controlada de ambos se realizará con el procedimiento común del apartado 11.

### 10.3. Etapa 2: hiperparámetros de arquitectura

| Campo | Especificación |
|---|---|
| Variables | `kernel_size`, `encoder_levels`, `base_filters` |
| Valores | `{3,5,7}`, `{3,4,5}`, `{32,64,96}` |
| Espacio discreto | 27 combinaciones |
| Learning rate y weight decay | Mejores valores de etapa 1, **fijos por sujeto** |
| Ensayos | **10 trials por sujeto** |
| Folds por ensayo | **B1, B2, B3, B4, B5** |
| Sampler | `TPESampler(seed=..., n_startup_trials=3)` |
| Entrenamiento | `SEARCH_EPOCHS` fijas, sin Early Stopping |
| Objetivo | Media del `val_pearson_full` de cinco folds |

Encolar como una de las diez propuestas la arquitectura `(kernel_size=5, encoder_levels=5, base_filters=64)` para disponer de una referencia arquitectónica dentro de esta etapa. Registrar las configuraciones repetidas si el sampler las propone; no contabilizar ensayos adicionales sin documentarlos.

Presupuesto: `10 × 5 × 3 = 150` entrenamientos.

### 10.4. Presupuesto total

| Fase | Entrenamientos por sujeto | Total de tres sujetos |
|---|---:|---:|
| Etapa 1 | 40 | 120 |
| Etapa 2 | 50 | 150 |
| **Total de búsqueda** | **90** | **270** |

Este presupuesto **excluye** EDA, verificación, evaluación de factibilidad, entrenamientos finales, repeticiones y reintentos por fallos de infraestructura. Una rejilla de las 27 arquitecturas con cinco folds necesitaría 135 entrenamientos por sujeto; el presupuesto TPE de diez trials no garantiza localizar el óptimo global.

### 10.5. Registro y persistencia de Optuna

- Un estudio persistente por sujeto y etapa, con `load_if_exists=True` o mecanismo equivalente.
- Guardar parámetros, objetivo, Pearson por fold y por dedo, estado del trial, época presupuestada, duración, GPU, semilla y hash del conjunto de datos.
- Respaldar la base de datos y archivos de estudio en almacenamiento persistente; no asumir que la sesión de Colab continuará activa.
- Evitar escrituras concurrentes de múltiples procesos sobre una base SQLite alojada directamente en una unidad sincronizada.

---

## 11. Paso 6 — Entrenamiento final y evaluación comparativa

### 11.1. Variantes por sujeto

- **A:** arquitectura y hiperparámetros de referencia.
- **B:** arquitectura de referencia y mejores parámetros de entrenamiento seleccionados.
- **C:** mejores parámetros de entrenamiento seleccionados y arquitectura elegida en etapa 2.

Todos se entrenan de nuevo desde cero, con el mismo pipeline, política de semillas y procedimiento de selección de duración final.

### 11.2. Procedimiento

1. Seleccionar la duración final usando exclusivamente desarrollo y un esquema interno común de validación.
2. Opcionalmente, reentrenar sobre todo el desarrollo una vez establecida la duración, sin consultar prueba.
3. Guardar checkpoint, escaladores, QC aplicado, lista de canales y configuración.
4. Aplicar al test el **mismo** procedimiento de limpieza aprobado y la misma alineación de 200 ms; no ajustar parámetros con el test.
5. Inferir por ventanas y reconstruir con overlap-add, respetando las máscaras de validez predefinidas.
6. Calcular las métricas en los instantes válidos de la línea temporal correspondiente; no elegir cobertura después de observar las predicciones.
7. Si se repiten semillas, usar un conjunto de semillas predefinido y reportar todas las ejecuciones sin seleccionar la mejor a partir de test.

### 11.3. Métricas y productos finales

| Métrica | Uso |
|---|---|
| **Pearson macro** | Comparación principal A/B/C |
| Pearson por dedo | Identificar diferencias de desempeño entre dedos |
| MSE y RMSE | Magnitud de errores; usar unidades originales cuando corresponda |
| CCC | Concordancia entre predicción y objetivo |
| Tiempo/memoria de inferencia | Caracterización técnica, sin afirmar automáticamente viabilidad en tiempo real |

Se guardarán trayectorias reales y predichas alineadas para visualización, tablas por sujeto, gráficos A/B/C, parámetros óptimos y un informe de cobertura temporal.

La interpretación debe limitarse al esquema de evaluación realmente ejecutado. No atribuir a TPE una mejora ocasionada por un cambio de preprocesamiento entre variantes, ni afirmar superioridad sobre Random Search sin un comparador de igual presupuesto.

---

## 12. Reproducibilidad, auditoría y gestión de resultados

### 12.1. Artefactos mínimos por entrenamiento

```text
results/<fase>/<sujeto>/<run_id>/
  config.json
  data_manifest.json
  split_manifest.json
  environment.json
  epoch_metrics.csv
  pearson_per_finger.csv
  runtime_metrics.csv
  run_summary.json
  checkpoint/                 # cuando corresponda
  predictions/                # para evaluación y diagnóstico
```

Cada archivo de métricas por época debe registrar, al menos: `run_id`, `subject`, `fold`, `seed`, `epoch`, `train_loss`, `val_pearson_full` cuando se calcule, `lr`, `elapsed_seconds` y, cuando proceda, tiempo y memoria máximos de la GPU.

### 12.2. Metadatos obligatorios

- Semillas Python/NumPy/PyTorch, determinismo y versiones de bibliotecas.
- Identificador de sujeto, canales originales y canales utilizados.
- Parámetros y decisiones de EDA/QC aplicados.
- Versiones y hashes de preprocesamiento, folds y escaladores.
- Arquitectura, LR, weight decay, batch size y **número de épocas efectivamente utilizado**.
- Modelo de GPU de Colab, memoria total, memoria pico reservada/asignada, tiempos y fallos.
- Estado de finalización y motivo de exclusión de resultados inválidos.

### 12.3. Reglas de modificación

| Situación | Acción |
|---|---|
| EDA/QC sin aprobar | No iniciar preprocesamiento definitivo |
| Paso 3 falla | Corregir únicamente el componente defectuoso y repetir las pruebas dependientes |
| Paso 3 pasa | Congelar los componentes metodológicos comprobados |
| Épocas sin definir | Realizar análisis posterior de convergencia; no lanzar Optuna |
| GPU de Colab insuficiente | Establecer una política uniforme de batch/precisión antes de la búsqueda |
| Error de implementación descubierto tras congelación | Versionar corrección, invalidar resultados afectados y repetirlos |
| Diferencias pequeñas por redondeo | Registrar precisión sin tratar redondeos normales como errores metodológicos |

---

## 13. Hoja de ejecución

| Orden | Actividad | Entregable | Condición para avanzar |
|---|---|---|---|
| **1** | Exploración y QC de S1/S2/S3 | `qc_decisions.json`, reportes y figuras | Decisiones de limpieza aprobadas |
| **2** | Preprocesamiento y generación de datos por fold | Datos procesados, escaladores y manifiestos | Datos generados y trazables |
| **3** | Verificación técnica integral | Informe de las 14 pruebas | **Todas aprobadas** |
| **4** | Medición de factibilidad en Colab Pro | Perfil de GPU, memoria, tiempo y batch común | Recursos viables |
| **5** | Definición del presupuesto de épocas | `SEARCH_EPOCHS` y política final registrados | Decisión documentada |
| **6** | Optimización TPE, etapa 1 | 20 trials × 2 folds × 3 sujetos | Parámetros de entrenamiento seleccionados |
| **7** | Optimización TPE, etapa 2 | 10 trials × 5 folds × 3 sujetos | Arquitecturas seleccionadas |
| **8** | Entrenamiento y evaluación final | Resultados A/B/C y figuras | Evaluación independiente completada |
| **9** | Documentación científica | Tablas, discusión y limitaciones | Resultados reproducibles y auditables |

### Regla de cierre

**La aprobación del paso 3 congela el pipeline metodológico.** El proyecto avanza a factibilidad y a la elección posterior de épocas, sin realizar más ajustes de procesamiento por desempeño. Los parámetros de duración y recursos solo se fijarán cuando exista evidencia suficiente; antes de eso permanecerán formalmente pendientes.

---

## 14. Configuración declarativa oficial

Toda la configuración operativa se almacena en un único archivo: **`configs/experiment_config.yaml`**, distribuido junto con este protocolo. Es la fuente de verdad de los valores concretos de ejecución, siempre bajo las restricciones metodológicas aquí descritas. No se mantendrán copias divergentes de la misma configuración en distintos notebooks o archivos.

Los campos `training.search_epochs`, `training.final_max_epochs`, `training.batch_size`, `validation.purge_samples_at_100hz`, los umbrales de QC y los parámetros finales de parada permanecerán en `null` hasta que se cumpla el procedimiento de verificación y aprobación correspondiente. Una ejecución no puede rellenarlos con valores por defecto silenciosos.

Antes de ejecutar Optuna, deberá existir un manifiesto de aprobación del paso 3, el presupuesto común de épocas documentado, la comprobación de memoria sobre la GPU asignada por Colab Pro y estudios persistentes listos. Para ejemplos de configuración y los criterios de pruebas, consultar `docs/validation_requirements.md` y `docs/dataset_spec.md`.

**Fin del protocolo.**
