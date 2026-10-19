# Pruebas

Instalación: `python -m pip install -e ".[test,dev]"`. Ejecución: `python -m pytest -q`. Tarda alrededor de 1 minuto: la integración corre el EDA completo sobre tres sujetos sintéticos de 30 s.

Todo usa datos sintéticos en carpetas temporales.

| Archivo | Qué comprueba |
|---|---|
| `test_preprocessing.py` | Exclusión y reindexado, CMR frente a `np.median`, respuestas FIR y soporte, aislamiento entre segmentos, Morlet (frecuencia, localización, DC), antialias frente a alias, recuperación del guante (retención/lineal/desconocido), spline cúbico, dirección del retardo, bloques, escaladores; además, que `preprocessing` no importa `eda`. |
| `test_config.py` | Valores pendientes en `null`; fijación de la lista de exclusión; validación de exclusiones inválidas, invariantes de señal y bloque `eda`; rechazo de un bloque `quality_control`. |
| `test_io.py` | Lectores MATLAB clásico/HDF5, aislamiento del test, contratos de forma y finitud, hashes, lector del guante, excepciones `BLOCKED`. |
| `test_descriptive.py` | Estadísticos descriptivos; el resumen de canales no depende del guante. |
| `test_eda.py` | Corrida completa `COMPLETE` con trazabilidad; los canales excluidos encabezan el ranking; guante y retardo sintéticos detectados; carpetas nunca reutilizadas; `BLOCKED`/`FAIL` correctos; regresión: un `FileNotFoundError` dentro de una sección es `FAIL`; notebook ligero y versión commiteada sin salidas. |

En Windows, usa la carpeta temporal por defecto o un `--basetemp` corto fuera de OneDrive. Con rutas de más de 260 caracteres la escritura de figuras falla; ahora eso se reporta como `FAIL` con diagnóstico.

Estas pruebas no equivalen a la aprobación V01–V14 del paso 3.
