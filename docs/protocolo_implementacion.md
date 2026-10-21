# Protocolo de implementación y evaluación experimental

**Proyecto:** Decodificación continua de la flexión de cinco dedos mediante señales electrocorticográficas (ECoG) y redes encoder–decoder 1D  
**Fuente de datos:** BCI Competition IV — Dataset 4  
**Sujetos:** S1, S2 y S3  
**Versión documental:** 1.3 — Reevaluación de los finalistas de la etapa 1
**Fecha:** 20 de octubre de 2026  
**Entorno previsto:** Python, PyTorch, MNE-Python, scikit-learn, Optuna y Google Colab Pro  
**Estado:** diseño de implementación; pendiente de verificación técnica integral y de definición posterior del número de épocas.

> **Cambio autorizado v1.3 (2026-10-20).** Por decisión de la persona usuaria, los 3 mejores trials de la etapa 1 de cada sujeto (media de B2 y B4) se entrenan también en B1, B3 y B5, con el mismo contrato. El learning rate y weight decay que heredan la etapa 2 y las variantes B y C son los del finalista con mayor media de los cinco folds. La búsqueda TPE, sus 20 trials y la etapa 2 no cambian; el presupuesto de búsqueda pasa de 270 a 297 entrenamientos (§10.2 y §10.4). Motivo: el par de la etapa 1 se arrastra a todo lo demás, y elegirlo con solo dos folds favorece al ganador por azar.

> **Cambio autorizado v1.2 (2026-10-18).** El EDA es descriptivo y opcional: no es precondición del preprocesamiento ni de la verificación. El flujo obligatorio comienza en 02 → 03. La exclusión fija y las decisiones científicas no cambian. El 02 publica un `pipeline_id` que el 03 y las etapas posteriores conservan; los hashes de código/configuración completa describen la ejecución, no crean otra identidad al cambiar de notebook.

> **Antecedente v1.1 (2026-10-18).** Por decisión de la persona usuaria se eliminó la etapa de control de calidad con umbrales, `qc_decisions.json` y aprobación. El flujo de entonces, EDA → Preprocesamiento, fue sustituido por v1.2. Se mantiene la exclusión fija (S1-54, S2-20, S2-37) definida en `configs/experiment_config.yaml` con su evidencia ECoG, aplicada antes de la referencia común. Los parámetros explícitos de los filtros FIR y del antialias permanecen fijados. Ver `docs/experimental_log.md` y `docs/channel_exclusion.md`.

> **Principio rector.** Implementar una sola cadena reproducible de análisis, limpieza, procesamiento, entrenamiento y evaluación para los tres sujetos. Antes de iniciar la optimización bayesiana se ejecutará una verificación técnica integral. Si todas las pruebas de esa verificación son satisfactorias, se congelarán la exclusión de canales, el preprocesamiento, la alineación temporal, las particiones, las métricas y la arquitectura parametrizable. Si alguna prueba falla, se corregirá únicamente el componente afectado, se regenerarán sus dependencias y se repetirán las pruebas pertinentes. **El número de épocas de entrenamiento no se determina en este documento**: se fijará posteriormente, con las señales ya verificadas y los recursos de Google Colab Pro.

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
- Canales: **exclusión fija de S1-54, S2-20 y S2-37** (índices cero-basados), aplicada en preprocesamiento antes de la referencia común; sin máscaras temporales ni reparaciones.
- Filtros: FIR Hamming de fase cero (kernel centrado, una pasada): pasabanda con transiciones de 10/20 Hz (331 taps) y notch de 2 Hz de rechazo y 2 Hz de transición (1651 taps); antialias Kaiser 35/50 Hz, 60 dB, antes de decimar la potencia Morlet.

### 1.2. Parámetros deliberadamente no cerrados

- `SEARCH_EPOCHS`: número fijo de épocas de cada entrenamiento de búsqueda.
- `FINAL_MAX_EPOCHS`: límite para selección de época durante entrenamiento final.
- `PURGE_SAMPLES`: separación temporal efectiva entre conjuntos, derivada de la verificación de soporte temporal de las transformaciones.
- `BATCH_SIZE`: valor común de ejecución, inicialmente candidato a 128, condicionado a la memoria real disponible para las 27 arquitecturas en Colab Pro.

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
│   ├── 01_exploracion_ecog.ipynb        # EDA descriptivo opcional
│   ├── 02_preprocesamiento_ecog.ipynb   # Exclusión de canales y características
│   ├── 03_verificacion_pipeline.ipynb   # Pruebas de aprobación
│   ├── 04_entrenamiento.ipynb           # Recursos y piloto en el mismo notebook
│   ├── 05_optuna_entrenamiento.ipynb
│   ├── 06_optuna_arquitectura.ipynb
│   └── 07_evaluacion_final.ipynb
├── src/tt_ecog/
│   ├── io.py
│   ├── descriptive.py                  # Estadísticos descriptivos del EDA
│   ├── eda/                            # Secciones del EDA y gráficas
│   ├── preprocessing.py                # Primitivas únicas (EDA y paso 2)
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
│   ├── eda/
│   ├── verification/
│   ├── feasibility/
│   ├── optuna/
│   └── final/
├── tests/
│   ├── test_descriptive.py
│   ├── test_eda.py
│   ├── test_preprocessing.py
│   ├── test_alignment.py
│   ├── test_splits.py
│   ├── test_no_leakage.py
│   ├── test_overlap_add.py
│   ├── test_architecture.py
│   └── test_metrics.py
├── docs/
│   ├── protocolo_implementacion.md
│   ├── channel_exclusion.md
│   └── experimental_log.md
├── requirements.txt
└── README.md
```

**Convenciones:** rutas configurables, identificadores únicos de ejecución, semillas registradas, archivos `.mat` originales de solo lectura, artefactos derivados identificados por `pipeline_id` (contrato efectivo y contenido numérico) y ningún secreto embebido en el repositorio.

---

## 3. Datos, unidades y esquema temporal

### 3.1. Entradas y objetivos

- Señales ECoG con frecuencia nominal de muestreo original de **1000 Hz**.
- Cinco trayectorias continuas de flexión del guante, nominalmente a **25 Hz**.
- Tres sujetos tratados de forma independiente: S1, S2 y S3.
- Desarrollo y prueba se mantendrán separados. Las etiquetas de prueba no participarán en análisis, exclusión de canales, selección de épocas ni selección de hiperparámetros.
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

## 4. Notebook 01 — EDA descriptivo opcional

**Responsables:** `01_exploracion_ecog.ipynb`, `tt_ecog/eda/` y `tt_ecog/descriptive.py`.
**Entrada:** ECoG y guante del conjunto de desarrollo.
**Salida:** figuras, tablas y manifiestos que justifican cada decisión de preprocesamiento.

El EDA **no modificará** los archivos originales ni persistirá datos transformados. Las transformaciones se aplican **en memoria** con `tt_ecog.preprocessing` (la misma implementación del paso 2) sobre el registro completo de desarrollo, tratado como un segmento único, solo para graficar. Nada de lo calculado en el EDA alimenta al paso 2, que recalcula desde los datos crudos por segmento de fold.

### 4.1. Secciones obligatorias para los tres sujetos

| Sección | Gráficas y tablas | Decisión que justifica |
|---|---|---|
| s01 Datos | Integridad, hashes, mapa crudo y trazas | Contexto |
| s02 Canales | Variabilidad local por canal y en el tiempo, ranking pico/típico, episodios, PSD de excluidos | Exclusión fija de S1-54, S2-20, S2-37 |
| s03 Referencia | CAR frente a CMR durante los episodios, robustez, espectro y correlación antes/después | Mediana común instantánea de canales conservados |
| s04 Banda | Espectro 1–500 Hz, cambio movimiento/reposo, ERSP, diseño del pasabanda | 40–200 Hz |
| s05 Notch | Espectros de alta resolución, estacionariedad de la línea, fracción de línea por etapa | Notch 60/120/180 Hz y su ancho |
| s06 Morlet | Familia de wavelets, escalograma, `zero_mean` y bordes | 40 frecuencias, 7 ciclos, máscaras de borde |
| s07 Decimación | Ancho de banda de la envolvente, submuestreo directo frente a antialias, coherencia | 100 Hz con antialias |
| s08 Guante | Patrón de almacenamiento a 1 kHz, recuperación, espectro, validación por muestras ocultas | 25 Hz y spline cúbico a 100 Hz |
| s09 Retardo | Correlación alta gamma–flexión según el retardo | 200 ms, ECoG(t) → flexión(t+0.2 s) |
| s10 Escalado | Colas pesadas, STD frente a escala robusta, deriva entre bloques, MinMax sin el bloque | RobustScaler(10, 90) y MinMax por fold |
| s11 Validación | Bloques B1–B5, autocorrelación, soporte medido de la cadena | CV por bloques; insumo para fijar la purga |

La proximidad de índices **no permite asumir proximidad física** entre electrodos sin un mapa de posiciones.

### 4.2. Uso de etiquetas

`train_dg` de desarrollo se usa **solo de forma descriptiva** (s04, s06–s11): las gráficas son confirmatorias y no ajustan parámetros, que permanecen fijos por este protocolo. Como observan también los bloques que actuarán como validación, si alguna gráfica contradijera un parámetro fijo se registrará y se pedirá autorización; no se cambiará el parámetro en silencio. Las etiquetas de prueba no se leen.

### 4.3. Exclusión fija de canales

La exclusión se decidió por calidad de adquisición, con evidencia exclusivamente de ECoG (transitorios recurrentes o extremos y ruido de línea dominante), sin usar etiquetas, predicciones ni Pearson. Vive solo en `channels.excluded` del YAML, con razón y figura de evidencia, y se aplica **antes** de la referencia común, el filtrado y Morlet, igual en desarrollo, validación y prueba. No se usan máscaras temporales ni reparaciones locales. El ruido de línea compartido (notable en S2) se trata con la referencia común y el notch, no con exclusiones.

Como la inspección observó todo el desarrollo, **no debe presentarse como un criterio anidado dentro de cada fold**. Toda operación que aprenda estadísticas para transformar entradas se ajustará únicamente con el entrenamiento del fold correspondiente. No se inspeccionará el conjunto de prueba para modificar la lista.

### 4.4. Entregables del EDA

```text
results/eda/<run_id>/
  fig/                      # PNG de cada sección (mostrados también en el notebook)
  tables/                   # channel_summary.csv, line_fraction.csv, glove_storage.csv,
                            # lag_xcorr.csv, block_stats.csv, filter_designs.json, support.json, ...
  index.md
  run_summary.json
  environment.json
  effective_config.json
  experiment_config.yaml
  data_manifest.json        # hashes de origen, mapas de canales, metadatos del guante
  artifact_manifest.json    # SHA-256 de cada archivo
```

**Criterio de completitud del EDA opcional (no condiciona otras etapas):** los tres sujetos se describieron con el mismo procedimiento, las once secciones terminaron en `COMPLETE`, existe la figura de evidencia de cada canal excluido y los datos originales permanecen intactos.

---

## 5. Paso 2 — Preprocesamiento ECoG

**Responsables:** `02_preprocesamiento_ecog.ipynb`, `preprocessing.py`, `splits.py` y `datasets.py`.  
**Entrada:** datos crudos, exclusión fija de canales, parámetros de procesamiento y esquema de folds.  
**Salida:** datos y máscaras temporales versionados por sujeto, fold y partición.

### 5.1. Orden de procesamiento

| Orden | Operación | Restricción |
|---|---|---|
| 1 | Cargar y validar ECoG y flexión originales | Datos de origen inmutables |
| 2 | Crear particiones temporales y coordenadas válidas | Registrar índices originales y segundos |
| 3 | Aplicar la exclusión fija `channels.excluded` | Antes de filtrado, referencia común y Morlet; registrar mapa original→efectivo |
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
- El escalado robusto **no equivale** a recortar valores extremos. No introducir clipping arbitrario.

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
data/processed/<pipeline_id>/
  S1/
    B1/ B2/ B3/ B4/ B5/
  S2/
    B1/ B2/ B3/ B4/ B5/
  S3/
    B1/ B2/ B3/ B4/ B5/
  manifests/
    channel_map.json
    preprocessing_config.json
    split_manifest.json
    filter_support.json
    processed_manifest.json
  folds/
    S1/val_B1/ ... val_B5/
    S2/val_B1/ ... val_B5/
    S3/val_B1/ ... val_B5/
    folds_manifest.json
```

El formato de persistencia es `schema_version: 2`. Se escribe primero en una carpeta provisional; solo una preparación completa se publica. Las corridas antiguas quedan intactas y no se migran automáticamente.

El `pipeline_id` es SHA-256 canónico del contrato efectivo de datos y del inventario de hashes de arrays numéricos (características, objetivos, índices, máscaras, escaladores y ventanas). Incluye archivos originales, exclusiones, señal, particiones, purga efectiva y escalado. Excluye fechas, rutas, texto, figuras, EDA, aprobación y parámetros de entrenamiento. Si se aprueba el mismo valor de purga candidato, la identidad no cambia.

Seleccionar explícitamente `paths.processed_pipeline_id` en el YAML con el valor emitido por 02. El 03 verifica y reutiliza esa identidad; nunca busca la corrida más reciente ni la recalcula desde su código o entorno. La aprobación y los reportes se guardan fuera de los datos inmutables. Las etapas posteriores registran ese mismo identificador junto con su propio `run_id`, commit, configuración completa, versiones y hashes de artefactos.

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
| **V02. Exclusión de canales** | Aplicación de `channels.excluded` por sujeto | Lista aplicada antes de la CMR y las transformaciones; reindexado correcto (C_válidos 61/46/64); un canal excluido extremo no afecta a la CMR |
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
2. Congelar **exclusión de canales, referencia espacial, filtrado, Morlet, alineación, escalado, particiones, inferencia, definición de métricas y arquitectura parametrizable**.
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

**Simplificación autorizada (2026-10-19):** integrar la comprobación de recursos al inicio del
notebook de entrenamiento, sin una etapa de software ni notebook adicionales. La implementación
de estas funciones puede prepararse antes de la aprobación del paso 3; su ejecución experimental
sigue requiriendo esa aprobación. La comprobación inicial usa el sujeto de más canales, un fold
de etapa 1 y tres arquitecturas (pequeña, referencia y grande), con dos actualizaciones de
calentamiento y tres medidas por modelo. Un OOM se registra y requiere repetir con otro candidato
de batch, sin reducirlo silenciosamente. Los tiempos iniciales son estimaciones de entrenamiento;
el piloto mide épocas completas con validación. Esta comprobación breve no certifica por sí sola
todas las arquitecturas ni determina automáticamente el batch común.

- Registrar la GPU realmente asignada por Colab Pro, memoria física, CUDA, versiones de PyTorch y duración de cada sesión.
- Verificar al menos arquitecturas representativas pequeñas, medianas y grandes, incluyendo el caso de mayor consumo (`L=5`, `f=96`).
- Medir memoria pico de PyTorch **durante varios pasos completos de forward + backward + actualización**, no únicamente en un forward.
- Utilizar batch size **128 como candidato inicial**. Si alguna arquitectura no cabe, elegir **antes de la búsqueda** un batch común viable para todas las configuraciones o documentar otra política controlada; no variar silenciosamente el batch entre trials.
- Medir el tiempo por época para estimar el presupuesto total de cómputo. Colab Pro puede asignar GPUs diferentes en sesiones distintas.

### 9.2. Decisión pendiente: número de épocas

El protocolo **no define actualmente** cuántas épocas utilizarán las búsquedas ni el máximo final.

**Propuesta de la persona usuaria (2026-10-19): 50 épocas**, registrada en
`training.search_epochs_candidate`. Se puede ejecutar un piloto acotado con ese presupuesto
después de la aprobación del pipeline. Sus curvas permiten revisar la propuesta; no constituyen
una aprobación automática ni evidencia de convergencia para todos los sujetos y tasas de aprendizaje.
`training.search_epochs` y `training.final_max_epochs` permanecen pendientes y separados.

Para cerrar esta decisión:

1. Utilizar exclusivamente el pipeline de datos ya aprobado.
2. Ejecutar un análisis acotado de curvas de aprendizaje con varias tasas de aprendizaje representativas del rango previsto.
3. Registrar por época pérdida de entrenamiento, Pearson de validación reconstruida, tiempo y memoria.
4. Comprobar que las tasas bajas dispongan de una oportunidad razonable de converger y que la duración no favorezca únicamente aprendizajes rápidos.
5. Seleccionar y documentar un **mismo `SEARCH_EPOCHS`** para todas las configuraciones de ambas etapas, con justificación de costo y convergencia.
6. Definir separadamente `FINAL_MAX_EPOCHS` y la política de Early Stopping final.

**Diagnóstico autorizado (2026-10-20, `convergence_study` del YAML).** Por sujeto (S1, S2, S3), en el fold B2 y hasta un horizonte de 60 épocas, se entrenan siete configuraciones con el batch y las funciones de la búsqueda: la referencia con dos semillas; los learning rates 3e-5, 2.5e-4 y 7.5e-4 con la arquitectura de referencia; y las arquitecturas k3-L3-f32 y k7-L5-f96 con el LR de referencia (weight decay de referencia en todas). No son trials, no cuentan en el presupuesto de búsqueda, no guían ni acotan la búsqueda y se ejecutan en la misma GPU que ella. Regla, fijada antes de ver resultados: por sujeto, el horizonte es la media de las épocas 56–60 y δ es el mayor entre la diferencia de las dos semillas de la referencia en el horizonte y la variación entre épocas de cada semilla en ese tramo. Por grupo (learning rates; arquitecturas) y candidato E ∈ {30, 40, 50}, gana la configuración con mejor Pearson en la época E y su pérdida es el mejor horizonte del grupo menos el horizonte del ganador. Se propone el menor E con pérdida ≤ δ en todos los sujetos y grupos; si ninguno cumple, el mayor candidato con la limitación documentada. Las inversiones de orden se informan sin decidir. Las curvas sirven además como evidencia para `FINAL_MAX_EPOCHS`, que se decide aparte.

**No adoptar automáticamente 20, 30, 40 ni ninguna otra cifra** sin finalizar este análisis. El valor aprobado se añadirá a `configs/experiment_config.yaml`; esta decisión **no reabre** el diseño de preprocesamiento ya verificado.

---

## 10. Paso 5 — Optimización bayesiana secuencial

### 10.1. Requisitos de inicio

**Preparación autorizada (2026-10-19):** implementar ahora los notebooks 05/06 y sus funciones
reutilizables. Esta autorización permite probarlos con datos sintéticos; no inicia búsquedas
experimentales ni resuelve los valores pendientes del YAML. Se conservan los requisitos de ejecución siguientes.

No iniciar búsquedas hasta contar con: pipeline congelado, las 14 verificaciones completas, presupuesto de épocas decidido y factibilidad de memoria/tiempo comprobada.

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
| Reevaluación (v1.3) | Los **3 mejores** trials por ese objetivo (empate: el trial anterior) se entrenan también en **B1, B3 y B5**, con la arquitectura de referencia, su LR/WD, la semilla del fold y las mismas épocas fijas |
| Selección | Finalista con mayor media del `val_pearson_full` de los **cinco** folds (empate: el mejor de la etapa 1) |

Presupuesto: `20 × 2 × 3 = 120` entrenamientos de búsqueda más `3 × 3 × 3 = 27` de reevaluación. La reevaluación no crea trials ni modifica la guía de TPE; sus resultados se guardan en el mismo estudio persistente.

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
| Reevaluación de finalistas de la etapa 1 (v1.3) | 9 | 27 |
| Etapa 2 | 50 | 150 |
| **Total de búsqueda** | **99** | **297** |

Este presupuesto **excluye** EDA, verificación, evaluación de factibilidad, entrenamientos finales, repeticiones y reintentos por fallos de infraestructura. Una rejilla de las 27 arquitecturas con cinco folds necesitaría 135 entrenamientos por sujeto; el presupuesto TPE de diez trials no garantiza localizar el óptimo global.

### 10.5. Registro y persistencia de Optuna

- Un estudio persistente por sujeto y etapa, con `load_if_exists=True` o mecanismo equivalente.
- Guardar parámetros, objetivo, Pearson por fold y por dedo, estado del trial, época presupuestada, duración, GPU, semilla y hash del conjunto de datos.
- Respaldar la base de datos y archivos de estudio en almacenamiento persistente; no asumir que la sesión de Colab continuará activa.
- Evitar escrituras concurrentes de múltiples procesos sobre una base SQLite alojada directamente en una unidad sincronizada.

La implementación usa SQLite en un directorio temporal local y publica atómicamente `study.zip`
en `results/optimization/<pipeline_id>/<sujeto>/<etapa>/`. El archivo reúne la base y el estado
del sampler, con versión de Optuna registrada. Usar un único runtime escritor por estudio;
no abrir el mismo estudio desde dos sesiones de Colab simultáneas.
El archivo se actualiza antes de entrenar una propuesta y al completar cada fold y trial.
Cada fold guarda además su última época completa en Drive, junto con AdamW, RNG e historial.
Al reconectar se recupera el trial RUNNING y se conservan los folds terminados. Los estados FAIL
se informan y detienen el estudio; no se agregan propuestas ni reintentos automáticamente.
La etapa 2 exige los 20 trials completos de la etapa 1 y la reevaluación completa de sus finalistas para ese mismo sujeto y contrato de ejecución. Un fallo durante la reevaluación queda registrado y la detiene para diagnóstico, sin reemplazar al finalista.

---

## 11. Paso 6 — Entrenamiento final y evaluación comparativa

### 11.1. Variantes por sujeto

- **A:** arquitectura y hiperparámetros de referencia.
- **B:** arquitectura de referencia y mejores parámetros de entrenamiento seleccionados (finalista de la etapa 1 con mayor media de los cinco folds).
- **C:** mejores parámetros de entrenamiento seleccionados y arquitectura elegida en etapa 2.

Todos se entrenan de nuevo desde cero, con el mismo pipeline, política de semillas y procedimiento de selección de duración final.

### 11.2. Procedimiento

1. Seleccionar la duración final usando exclusivamente desarrollo. **Decisión autorizada el 2026-10-19:** usar el fold purgado B5 como validación interna común para A/B/C y los tres sujetos (`training.final_validation_block: B5`). La duración corresponde al mayor `val_pearson_full` observado, con el primer empate; la paciencia cuenta épocas sin una mejora mayor que `min_delta`. Máximo, paciencia y `min_delta` siguen pendientes de evidencia de desarrollo.
2. Reentrenar desde cero sobre todo el desarrollo durante la duración seleccionada, como fija `final_retrain_full_development_after_epoch_selection: true`. Reajustar escaladores con las columnas válidas de desarrollo, compartidos por A/B/C; conservar los límites originales B1–B5 y sus guardas. No consultar prueba.
3. Guardar checkpoint, escaladores, lista de canales (mapa original→efectivo) y configuración.
4. Aplicar al test la **misma** exclusión de canales, el mismo preprocesamiento y la misma alineación de 200 ms; no ajustar parámetros con el test.
5. Inferir por ventanas y reconstruir con overlap-add, respetando las máscaras de validez predefinidas.
6. Calcular las métricas en los instantes válidos de la línea temporal correspondiente; no elegir cobertura después de observar las predicciones.
7. Si se repiten semillas, usar un conjunto de semillas predefinido y reportar todas las ejecuciones sin seleccionar la mejor a partir de test.

La preparación de `07_evaluacion_final.ipynb` y sus pruebas sintéticas fue autorizada el
2026-10-19 antes de completar experimentos reales. La ejecución conserva los bloqueos del
protocolo: V01–V14 aprobado, estudios 05/06 completos y parámetros finales decididos. Los
nueve modelos, escaladores, parámetros y duraciones se congelan antes del primer acceso a
test. La selección de B5 sirve para elegir duración, no para reportar desempeño independiente
tras la búsqueda. Las interrupciones recuperan la última época completa; tras abrir test solo
se permite completar inferencia o reportes con los modelos ya congelados.

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

Cada archivo de métricas por época debe registrar, al menos: `pipeline_id`, `run_id`, `subject`, `fold`, `seed`, `epoch`, `train_loss`, `val_pearson_full` cuando se calcule, `lr`, `elapsed_seconds` y, cuando proceda, tiempo y memoria máximos de la GPU.

### 12.2. Metadatos obligatorios

- Semillas Python/NumPy/PyTorch, determinismo y versiones de bibliotecas.
- Identificador de sujeto, canales originales y canales utilizados.
- Exclusión de canales aplicada y `pipeline_id` seleccionado. El EDA, si se ejecutó, tiene su propio `run_id` descriptivo y no forma parte de la identidad.
- Versiones y hashes de preprocesamiento, folds y escaladores.
- Arquitectura, LR, weight decay, batch size y **número de épocas efectivamente utilizado**.
- Modelo de GPU de Colab, memoria total, memoria pico reservada/asignada, tiempos y fallos.
- Estado de finalización y motivo de exclusión de resultados inválidos.

### 12.3. Reglas de modificación

| Situación | Acción |
|---|---|
| EDA ausente o incompleto | Registrar su propio estado; no bloquea 02 ni 03 |
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
| **Opcional** | EDA descriptivo de S1/S2/S3 | Figuras, tablas y manifiestos en `results/eda/<run_id>/` | No condiciona otras etapas |
| **2** | Preprocesamiento y generación de datos por fold | Datos procesados, escaladores y manifiestos | Datos generados y trazables |
| **3** | Verificación técnica integral | Informe de las 14 pruebas | **Todas aprobadas** |
| **4** | Medición de factibilidad en Colab Pro | Perfil de GPU, memoria, tiempo y batch común | Recursos viables |
| **5** | Definición del presupuesto de épocas | `SEARCH_EPOCHS` y política final registrados | Decisión documentada |
| **6** | Optimización TPE, etapa 1 | 20 trials × 2 folds × 3 sujetos, más 3 finalistas × 3 folds × 3 sujetos | Parámetros de entrenamiento seleccionados con cinco folds |
| **7** | Optimización TPE, etapa 2 | 10 trials × 5 folds × 3 sujetos | Arquitecturas seleccionadas |
| **8** | Entrenamiento y evaluación final | Resultados A/B/C y figuras | Evaluación independiente completada |
| **9** | Documentación científica | Tablas, discusión y limitaciones | Resultados reproducibles y auditables |

### Regla de cierre

**La aprobación del paso 3 congela el pipeline metodológico.** El proyecto avanza a factibilidad y a la elección posterior de épocas, sin realizar más ajustes de procesamiento por desempeño. Los parámetros de duración y recursos solo se fijarán cuando exista evidencia suficiente; antes de eso permanecerán formalmente pendientes.

---

## 14. Configuración declarativa oficial

Toda la configuración operativa se almacena en un único archivo: **`configs/experiment_config.yaml`**, distribuido junto con este protocolo. Es la fuente de verdad de los valores concretos de ejecución, siempre bajo las restricciones metodológicas aquí descritas. No se mantendrán copias divergentes de la misma configuración en distintos notebooks o archivos.

Los campos `training.search_epochs`, `training.final_max_epochs`, `training.batch_size`, `validation.purge_samples_at_100hz` y los parámetros finales de parada permanecerán en `null` hasta que se cumpla el procedimiento de verificación y aprobación correspondiente. Una ejecución no puede rellenarlos con valores por defecto silenciosos.

Antes de ejecutar Optuna, deberá existir un manifiesto de aprobación del paso 3, el presupuesto común de épocas documentado, la comprobación de memoria sobre la GPU asignada por Colab Pro y estudios persistentes listos. Para ejemplos de configuración y los criterios de pruebas, consultar `docs/validation_requirements.md` y `docs/dataset_spec.md`.

**Fin del protocolo.**
