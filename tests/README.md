# Pruebas

Instalación: `python -m pip install -e ".[test,dev]"`. Ejecución: `python -m pytest -q`. Tarda alrededor de 2 minutos: la integración corre el EDA completo sobre tres sujetos sintéticos de 30 s y el paso 2 completo sobre tres de 60 s.

Todo usa datos sintéticos en carpetas temporales.

| Archivo | Qué comprueba |
|---|---|
| `test_preprocessing.py` | Máscara exacta de columnas válidas frente al mismo segmento dentro de un registro más largo (incluido un largo no múltiplo de 10); nodos absolutos del guante por segmento; reconstrucción de escaladores; además, exclusión y reindexado, CMR frente a `np.median`, respuestas FIR y soporte, aislamiento entre segmentos, Morlet (frecuencia, localización, DC), antialias frente a alias, recuperación del guante (retención/lineal/desconocido), spline cúbico, dirección del retardo, bloques, escaladores; además, que `preprocessing` no importa `eda`. |
| `test_config.py` | `reference_run` (null o run_id); la purga solo sale como candidato etiquetado mientras sea `null`; validación de los bloques `validation` y `scaling`; además, valores pendientes en `null`; fijación de la lista de exclusión; validación de exclusiones inválidas, invariantes de señal y bloque `eda`; rechazo de un bloque `quality_control`. |
| `test_io.py` | Lectores MATLAB clásico/HDF5, aislamiento del test, contratos de forma y finitud, hashes, lector del guante, excepciones `BLOCKED`. |
| `test_descriptive.py` | Estadísticos descriptivos; el resumen de canales no depende del guante. |
| `test_eda.py` | Corrida completa `COMPLETE` con trazabilidad; los canales excluidos encabezan el ranking; guante y retardo sintéticos detectados; carpetas nunca reutilizadas; `BLOCKED`/`FAIL` correctos; regresión: un `FileNotFoundError` dentro de una sección es `FAIL`; notebook ligero y versión commiteada sin salidas. |
| `test_splits.py` | Folds de las etapas; purga sin off-by-one antes y después de validación, con P=0, P mayor que un bloque y B1/B5 como validación; rechazo de purga `None`, booleana, negativa o no entera. |
| `test_datasets.py` | Conteo de ventanas (T<L, T=L, T=L+1, strides); ninguna ventana cruza tramos; la ventana final de inferencia cubre cada columna válida; un tramo corto o un stride excesivo fallan. |
| `test_prep.py` | Paso 2 completo con una evidencia EDA sintética: `COMPLETE` y manifiestos verificables; contrato de arrays (61/46/64 × 40, `y = x + 20`, máscaras exactas); alineación ECoG(t) → flexión(t+20); bloques independientes bit a bit; escaladores inmunes a perturbar validación; rango nulo o dedo constante dan FAIL; `load_fold` rechaza candidatos y devuelve ventanas escaladas; reutilización verificada; carpetas corruptas nunca se reutilizan; rechazos de la compuerta; el código del paso 2 no nombra variables de test; notebook 02 ligero. |

En Windows, usa la carpeta temporal por defecto o un `--basetemp` corto fuera de OneDrive. Con rutas de más de 260 caracteres la escritura de figuras falla; ahora eso se reporta como `FAIL` con diagnóstico.

Estas pruebas no equivalen a la aprobación V01–V14 del paso 3.
