# Pruebas

Instalación: `python -m pip install -e ".[test,dev,model]"`. Ejecución: `python -m pytest -q`.

Tarda alrededor de 6 minutos en CPU. La integración corre:
- el EDA completo sobre tres sujetos sintéticos de 30 s;
- el paso 2 sobre tres de 60 s;
- las 14 verificaciones sobre ese resultado.

**Marcador `verification`.** `@pytest.mark.verification("Vxx")` liga cada prueba con su check del paso 3. Con `--verification-json ruta.json`, `conftest.py` guarda el resultado de cada prueba y sus ids, y el notebook 03 lo combina con la parte de datos reales. Una prueba omitida no cuenta como PASS.

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
| `test_architecture.py` (V13) | Arquitectura de referencia y regla de las 27; estructura (Conv1d k=3, dilataciones, MaxPool, decoder espejo, cabeza 2f→5); forward + pérdida + backward finitos en las 27; tres números de canales; longitudes distintas de 256; misma semilla, misma inicialización. |
| `test_metrics.py` (V14) | Pearson frente a SciPy y CCC/MSE/RMSE frente a sus fórmulas; CCC penaliza escala y desplazamiento; dedo constante, casi constante o con NaN invalida la evaluación; la pérdida coincide con su fórmula y es finita con ventanas nulas. |
| `test_overlap_add.py` (V12) | Reconstrucción identidad con huecos y ventana final; promedio de solapes; cobertura incompleta y ventanas fuera de rango fallan; Pearson 1 en un fold procesado y predicción constante inválida. |
| `test_verification.py` | Las 14 pruebas sobre el paso 2 sintético: todo PASS salvo V04/V09 `BLOCKED` con purga candidata; el reporte nunca aprueba así; con purga aprobada V04/V09 pasan sin reprocesar bloques; regla de combinación real + sintético; mapeo de pytest por id; datos ausentes = `BLOCKED`; notebook 03 ligero. |
| `test_prep.py` | Paso 2 completo con una evidencia EDA sintética: `COMPLETE` y manifiestos verificables; contrato de arrays (61/46/64 × 40, `y = x + 20`, máscaras exactas); alineación ECoG(t) → flexión(t+20); bloques independientes bit a bit; escaladores inmunes a perturbar validación; rango nulo o dedo constante dan FAIL; `load_fold` rechaza candidatos y devuelve ventanas escaladas; reutilización verificada; carpetas corruptas nunca se reutilizan; rechazos de la compuerta; el código del paso 2 no nombra variables de test; notebook 02 ligero. |

En Windows, usa la carpeta temporal por defecto o un `--basetemp` corto fuera de OneDrive. Con rutas de más de 260 caracteres la escritura de figuras falla; ahora eso se reporta como `FAIL` con diagnóstico.

Estas pruebas no equivalen a la aprobación V01–V14 del paso 3.
