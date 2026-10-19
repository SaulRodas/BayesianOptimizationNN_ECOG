# Bitácora del proyecto y registro de decisiones

Todas las decisiones técnicas deben documentarse antes de lanzar los entrenamientos afectados.

## Plantilla de entrada

```text
Fecha y hora UTC:
Autor:
Etapa:
Hash del commit:
Archivo(s) cambiado(s):
Motivo tecnico:
Prueba de aceptacion relacionada:
Evidencia (ruta/hash):
Resultado: PASS | FAIL | BLOCKED | PENDING
Impacto sobre datos y resultados anteriores:
Decision posterior requerida:
```

## Puertas de aprobación

| Hito | Evidencia obligatoria | Estado inicial |
|---|---|---|
| EDA de justificación completo | `results/eda/<run_id>/` en `COMPLETE` + figuras de exclusión | PENDING |
| Preprocesamiento generado | manifests, escaladores y shapes | PENDING |
| Paso 3 completo | V01-V14 PASS real, hashes, reportes | PENDING |
| Recursos Colab Pro | GPU, memoria, batch y tiempo | PENDING |
| Duración de búsqueda | search_epochs justificado | PENDING |
| Duración final | final_max_epochs / Early Stopping | PENDING |
| Optuna etapa 1 | estudios persistentes S1-S3 | PENDING |
| Optuna etapa 2 | estudios persistentes S1-S3 | PENDING |
| Evaluación final | modelos A/B/C y test reservado | PENDING |

Las incidencias descubiertas tras una congelacion invalidaran solamente los artefactos afectados, según su dependencia registrada. No modificar resultados silenciosamente.

## 2026-10-18 — Primera implementación EDA/QC

- Etapa: cargadores y EDA/QC de desarrollo; sin limpieza ni entrenamiento.
- Autor: Codex, según el plan solicitado por la persona usuaria.
- Control de versiones: Git inicializado en `main`, `origin` configurado al remoto indicado. Sin commit ni publicación remota en esta entrega; procedencia local registrada mediante hashes de fuentes.
- Implementación: paquete instalable `tt_ecog`, lector MATLAB clásico y v7.3 selectivo, detectores sin etiquetas, reportes y manifiestos por ejecución, validador de decisiones, notebook Colab y comprobación de dependencias EDA.
- Configuración: rutas relativas a `MyDrive/TT_ECoG`, entradas en `raw`; resoluciones descriptivas explícitas en `quality_control.analysis`. Umbrales de detección pendientes, incluidos los nuevos parámetros de persistencia espectral. No se alteran épocas, batch, purga ni parámetros de entrenamiento.
- Pruebas: 51 casos sintéticos PASS, sin omisiones; integración con tres sujetos artificiales de 2 segundos y canales nominales, figuras, hashes, rechazo de datos inválidos, aislamiento respecto a etiquetas y estados pendientes. No es evidencia de EDA real ni de V01–V14 completos.
- Entorno: Python 3.12.12, CPU, entorno local aislado. Versiones completas en `results/implementation_checks/environment.json`.
- Comandos: instalación editable de dependencias EDA; `python -m pytest -q --tb=short --basetemp=results/implementation_checks/tmp_run_04 --junitxml=results/implementation_checks/test_results.xml`; comprobación del entorno; compilación Python; construcción de wheel sin dependencias de entrenamiento.
- Incidencia de entorno resuelta: la primera invocación de pytest no pudo utilizar la carpeta temporal global por permisos; se repitió usando un directorio desechable dentro de `results`. No se omitieron las pruebas afectadas.
- Evidencia: `results/implementation_checks/test_results.xml`, `implementation_report.json`, `environment.json`, wheel y manifiesto de artefactos. `SHA256SUMS.txt` identifica los archivos de entrega. Las salidas de pruebas y fixtures se excluyen de Git.
- Estado de datos reales: BLOCKED en este entorno; Drive no está montado ni se dispone localmente de los tres `.mat` reales. No se abrió ninguna etiqueta real de prueba.
- Estado QC: PENDING; no hay decisiones aprobadas. Estado del paso 3: PENDING; las comprobaciones con datos reales permanecen bloqueadas y el pipeline no está congelado.
- Pendiente: publicar el código revisado, ejecutar el notebook mediante la extensión de Colab con Drive, revisar evidencia de S1/S2/S3, fundamentar candidatos y aprobar QC antes de implementar preprocesamiento.

## 2026-10-18 — v0.3.0: EDA de justificación y exclusión fija de canales (protocolo v1.1)

- Etapa: EDA descriptivo y primitivas de preprocesamiento compartidas; sin datos procesados persistidos, sin entrenamiento.
- Autor: Claude (Claude Code), a petición y con aprobación del plan por la persona usuaria.
- Control de versiones: base `cf96d24`; cambios aún sin commit en esta entrada.
- Decisión metodológica (autorizada por la persona usuaria): se elimina la etapa de QC con umbrales, `qc_decisions.json`, esquema y aprobación. El flujo pasa a ser EDA → Preprocesamiento. La exclusión fija es S1-54, S2-20 y S2-37 (cero-basados), antes de la CMR, sin máscaras ni reparaciones. Evidencia y razones en `docs/channel_exclusion.md`. El "canal 30" de S2 mencionado era el 20; el 30 es normal.
- Advertencia de procedencia: las cifras de la exclusión vienen de `results/eda_qc_review/20261018T230228Z`, que no verificó los hashes de los `.mat` (`raw_source_hashes_verified: false`). La primera corrida real del nuevo EDA regenera la evidencia con hashes.
- Decisión técnica nueva (aprobada con el plan):
  - Filtros FIR Hamming de fase cero (kernel centrado, una pasada): pasabanda 40–200 Hz con transiciones de 10/20 Hz (331 taps); notch 60/120/180 Hz con 2 Hz de rechazo y 2 Hz de transición (1651 taps). El kernel combinado tiene 1981 taps (±990 ms).
  - Antialias Kaiser (35/50 Hz, 60 dB, 243 taps) antes de decimar la potencia Morlet a 100 Hz.
  - Guante: patrón de almacenamiento verificado (método y fase exactos) y tiempo en la primera fila de cada bloque.
  - Con los valores por defecto de MNE, el notch habría tenido ~6601 taps (±3.3 s).
- Implementación:
  - `tt_ecog.preprocessing`: única implementación, por segmento.
  - `tt_ecog.descriptive`, `tt_ecog.splits` (bloques).
  - `tt_ecog.eda`: once secciones, figuras y tablas en `MyDrive/TT_ECoG/results/eda/<run_id>/`, notebook ligero que muestra los mismos PNG.
  - Se eliminaron `quality_control.py`, `decisions.py`, el esquema, la plantilla y sus pruebas.
  - Corregido: un `FileNotFoundError` al escribir resultados ya no se reporta como `BLOCKED` (ahora `FAIL`).
- Ejecución: VS Code local con kernel Colab; código por git push/pull; salidas del notebook excluidas de Git con `nbstripout` (`.gitattributes`).
- Pruebas: 102 PASS (`python -m pytest -q`, ~1 min). Incluyen una corrida completa del EDA sobre tres sujetos sintéticos de 30 s y la regresión del error anterior. No equivalen a V01–V14.
- Corrida de referencia de escala real con datos **sintéticos** (3 sujetos × 400 s, 62/48/64 canales, Windows, Python 3.12.12, MNE 1.13.2, CPU local):
  - `COMPLETE` en 574 s; RAM pico 5.47 GB.
  - s04 (ERSP de diagnóstico, 50 frecuencias a 1 kHz) ocupa 481 s; el resto de secciones suma menos de 100 s.
  - Es una medida local; el tiempo en el CPU de Colab puede ser mayor.
- Hallazgo para el paso 2: tras la decimación con antialias, ~1.5 % de los valores de potencia Morlet (ruido blanco sintético) son ligeramente negativos por el rizado del FIR. Se reportan y no se recortan (protocolo §5.2). Cualquier transformación logarítmica futura debe tenerlo en cuenta; el ERSP usa el cociente de potencias promediadas.
- Estado de datos reales: el EDA con los `.mat` reales no se ejecutó en esta entrega (los datos están en Drive). No se abrió ninguna etiqueta de prueba.
- Pendiente: commit + push; ejecutar el notebook en Colab con los datos reales; revisar las figuras; registrar aquí el `run_id` y el hash de `artifact_manifest.json`; después, preprocesamiento por fold (notebook 02).

## 2026-10-18 — v0.4.0: Paso 2, preprocesamiento por bloque, folds y escaladores

- **Etapa:** paso 2 (protocolo §5), solo implementación. Los datos reales todavía no se procesaron: la corrida la ejecuta la persona usuaria en Colab con el notebook 02.
- **Autor y control de versiones:**
  - Claude (Claude Code), a petición y con aprobación del plan por la persona usuaria.
  - Base `286dbc2`; cambios aún sin commit en esta entrada.
- **EDA de referencia:**
  - Corrida `20261019T004831Z_3a68db`, `COMPLETE`, commit `286dbc2`, tomada de las salidas locales del notebook 01.
  - Queda fijada en `channels.reference_run`.
  - La compuerta del paso 2 la verifica contra Drive antes de procesar: estado, `test_accessed: false`, SHA-256 de cada `.mat`, `channels.excluded`, configuración `signal` y SHA-256 de todos sus artefactos, incluidas las figuras de evidencia.
  - El hash de su `artifact_manifest.json` queda en `p01_eda_reference.json` de la corrida del paso 2.
- **Decisiones autorizadas por la persona usuaria:**
  1. **Unidad independiente = bloque.**
     - Cada bloque B1–B5 se procesa una sola vez y solo con su ECoG cruda y sus muestras efectivas del guante.
     - Consecuencias:
       - ninguna característica depende de otro bloque;
       - se pierden 125 columnas a 100 Hz por borde de cada bloque (contexto FIR→Morlet→antialias de H = 990 + 139 + 121 = 1250 muestras fuente) y 20 columnas finales por el retardo;
       - ninguna ventana cruza fronteras entre bloques de train;
       - cambiar la purga no obliga a reprocesar.
  2. **Purga pendiente.**
     - Los artefactos de fold exigen una purga entera explícita (`None` es error).
     - Mientras `purge_samples_at_100hz` sea `null`, se construyen con `purge_candidate_for_tests` (256) y quedan etiquetados `candidate_pending_V04_V09` en todos los manifiestos.
     - `datasets.load_fold` los rechaza salvo con `allow_candidate=True`, reservado al paso 3.
- **Semántica de la purga P**, a nivel de columnas, solo para train y sumada a las máscaras de borde:
  - un bloque de train anterior a validación exige `y_index ≤ inicio_val − 1 − P`;
  - uno posterior exige `x_index ≥ fin_val + P`;
  - validación no se purga.
  - Con P = 256 y estos bordes, el hueco efectivo es de 381 muestras antes de validación (P + 125) y 361 después (P + 105). En la prueba sintética coincidió así en los 15 folds.
- **Decisión técnica nueva:** los escaladores fallan de forma explícita si una característica canal-frecuencia tiene rango 10–90 nulo en train o un dedo es constante. scikit-learn reemplazaría esa escala por 1 sin avisar.
- **Implementación:**
  - `tt_ecog.prep` (compuerta, bloques, folds, pasos, runner y CLI `python -m tt_ecog.prep`).
  - `tt_ecog.splits` (folds y elegibilidad), `tt_ecog.datasets` (ventanas y `load_fold`) y `tt_ecog.store` (identidad por hash, manifiesto como marca de commit, reutilización solo verificada y sin sobrescribir).
  - Notebook `02_preprocesamiento_ecog.ipynb`.
  - En `preprocessing`: `segment_features`, `segment_glove`, `feature_valid_mask`, reconstrucción de escaladores, y los traslados de `measured_support` (desde s11) y de la inspección y recuperación del guante (desde `eda/bundle.py`). El EDA llama ahora a esas funciones; para registros de 400 s su máscara de validez es idéntica a la anterior.
  - Configuración: `paths.preprocessing_dir`; validación de los bloques `validation` y `scaling`; `purge_for_build()` y `approved_purge()`.
- **Layout en Drive:**
  - `data/processed/<pipeline_hash>/`, con `S*/B*/` (`x.npy` float32 `(C, 40, N)` de potencia sin escalar, `y.npy` float64, `x_index`, `y_index = x_index + 20`, `valid`, `block_manifest.json`), `manifests/` y `folds/purge_<P>_<hash>/`.
  - `pipeline_hash` cubre: señal, exclusiones, bloques, corrida EDA, SHA-256 crudos, fuentes que procesan y versiones de numpy/scipy/mne.
- **Pruebas:**
  - 169 PASS y 1 SKIP (versión commiteada del notebook 02, aún inexistente), con `python -m pytest -q`, en ~2 min, en el entorno local de la entrada anterior.
  - Integración del paso 2 con tres sujetos sintéticos de 60 s: `COMPLETE` en ~43 s.
  - No equivalen a V01–V14.
- **Hallazgo sintético:** ~1.5 % de valores de potencia negativos en columnas válidas, consistente con lo informado en v0.3.0; se reportan por bloque y no se recortan.
- **Impacto sobre resultados anteriores:** ninguno. La corrida EDA de referencia sigue siendo válida (su configuración de señal y exclusiones no cambió).
- **Pendiente:**
  - commit + push;
  - ejecutar el notebook 02 en Colab;
  - registrar aquí `pipeline_hash`, `run_id` y SHA-256 de `processed_manifest.json` y `folds_manifest.json`;
  - después, el paso 3 (V01–V14), que decide la purga.

## 2026-10-18 — Corrida real del paso 2

- **Ejecución:** la persona usuaria corrió el notebook 02 en Colab, en el commit `7f5b45f`, sin cambios locales.
  - Corrida `20261019T022241Z_90bd22`, estado `COMPLETE`.
  - EDA de referencia `20261019T004831Z_3a68db` validada por la compuerta.
- **Artefactos en Drive:**
  - datos procesados: `data/processed/89ba1f9c12977498`;
  - folds: `data/processed/89ba1f9c12977498/folds/purge_256_4e3d382b582e7c1c`, purga candidata 256 (`candidate_pending_V04_V09`).
  - Los SHA-256 de `processed_manifest.json` y `folds_manifest.json` están en esas carpetas; el paso 3 los registra en su reporte.
- **Resultados:**
  - Cada bloque tiene X de forma `(61|46|64, 40, 7980)` con 7750 columnas válidas: 230 sin contexto ECoG completo y 3 sin flexión interpolada al final del bloque.
  - Por fold: ~30 700–30 870 columnas de train, ~29 700–29 850 ventanas de train, 7750 columnas de validación y 119 ventanas de inferencia.
  - Huecos efectivos: 381 muestras antes de validación y 361 después.
  - Potencia negativa por rizado del antialias: ~1.65 % de los valores válidos (no se recorta).

## 2026-10-18 — v0.5.0: Paso 3, U-Net mínima, pérdida, métricas y verificación V01–V14

- **Etapa:** paso 3 (protocolo §6), solo implementación. Sin entrenamientos experimentales ni Optuna. La corrida real la ejecuta la persona usuaria en Colab con el notebook 03.
- **Autor y control de versiones:**
  - Claude (Claude Code), a petición de la persona usuaria.
  - Base `7f5b45f`; cambios aún sin commit en esta entrada.
- **Decisiones de arquitectura de la persona usuaria** (detalles abiertos en §7, fijados antes de V13 y registrados en `model.*` del YAML):
  - Reducción canal×frecuencia: Conv1d(C·40 → f, kernel 3, compartida en el tiempo) → BN → GELU → Dropout 0.3.
  - Encoder: Conv1d dilatada (stride 1, padding same, sin sesgo) → BN → GELU → Dropout → **MaxPool1d(2)**; el skip se toma antes de reducir.
  - Decoder: **interpolación lineal al largo exacto del skip → Conv1d → BN → GELU → Dropout → concatenación del skip**; cabeza Conv1d(2f → 5, kernel 1).
- **Elegido por Claude y pendiente de visto bueno** (también en el YAML):
  - el decoder espeja los kernels del encoder sin dilatación (`decoder_dilation: 1`);
  - el Dropout está en todos los bloques convolucionales;
  - inicialización por defecto de PyTorch para todas las variantes.
- **Pérdida y métricas:**
  - Pérdida: el coseno temporal usa `sqrt(max(‖p‖²‖t‖², eps²))`, finito y con gradiente finito para ventanas nulas. Esto importa porque el MinMax deja dedos en reposo exactamente en 0.
  - Métricas en numpy. Pearson es indefinido si la serie no es finita o si su STD ≤ `metrics.degenerate_std_relative` (1e-9) × máx|x|; en ese caso la evaluación queda inválida con diagnóstico y el macro es `None`.
  - Errores en unidades originales tras la inversa del MinMax del fold.
  - Overlap-add con pesos uniformes y error si falta cobertura.
- **Verificación:**
  - Cada V01–V14 tiene una parte con datos reales (`tt_ecog.verification.checks`) y pruebas sintéticas marcadas `@pytest.mark.verification("Vxx")`.
  - Es PASS solo si pasan ambas (una omisión no cuenta).
  - V07/V08 recalculan los bloques desde los `.mat` con impulsos de 1e6×máx|ECoG| y retenciones del guante desplazadas, y exigen igualdad bit a bit (`verification.isolation_tolerance: 0`) con una línea base de la misma sesión. También exigen que el bloque perturbado sí cambie.
  - V11 compara lo persistido con lo recalculado (`reproducibility_rtol: 1e-6`).
  - Con todo en PASS se emite `approved_pipeline_manifest.json` como candidato; no se congela nada automáticamente.
- **Purga en dos rondas:**
  1. Con la purga `null`, V04/V09 quedan `BLOCKED` con la evidencia. Con procesamiento por bloque, la purga mínima que aísla contextos y etiquetas es 0; con el candidato 256, el hueco efectivo mínimo es de 361 muestras.
  2. La persona usuaria fija `purge_samples_at_100hz`; se rehacen los folds (notebook 02, sin reprocesar bloques) y se verifica de nuevo.
  - Nunca se elige la purga por Pearson.
- **Configuración:**
  - `paths.verification_dir`, que reemplaza a `verification.report_dir`;
  - claves fijas de `model`; `metrics.degenerate_std_relative`; tolerancias de `verification`;
  - validación de los bloques `model`, `training.loss`, `metrics` y `verification`.
  - Sin cambios en `signal`, exclusiones, bloques ni escalado: los datos del paso 2 siguen siendo válidos (mismo `pipeline_hash`).
- **Entorno local:** PyTorch 2.14.1+cpu instalado con `uv` en `.venv`. El extra `model` de `pyproject.toml` declara torch. En Colab no se reinstala PyTorch.
- **Pruebas:** 235 PASS y 1 SKIP (versión commiteada del notebook 03, aún inexistente), con `python -m pytest -q`, en 4 min 38 s en CPU local. Las 14 verificaciones sobre el paso 2 sintético dan PASS salvo V04/V09 `BLOCKED` (purga candidata); con purga aprobada, V04/V09 pasan sin reprocesar bloques. No equivalen a la corrida real de V01–V14.
- **Pendiente:**
  - commit + push;
  - notebook 03 en Colab (primera ronda);
  - decidir la purga;
  - notebook 02 y notebook 03 (segunda ronda);
  - aprobación explícita de la congelación.


## 2026-10-19 — v0.6.0: reorganización autorizada v1.2

- **Autorización:** plan explícito de la persona usuaria, registrado en protocolo v1.2 (2026-10-18). Implementación por Codex; cambios locales conservados, sin entrenamientos, Optuna ni evaluación final.
- **Flujo:** 02 → 03; 01 es descriptivo y opcional. Se elimina `channels.reference_run` y la compuerta que comprobaba la corrida y las figuras EDA. Las exclusiones y sus razones históricas permanecen.
- **Identidad:** 02 calcula `pipeline_id` SHA-256 del contrato efectivo y los arrays numéricos, incluidos índices, máscaras, escaladores y ventanas. 03 lo lee de la selección explícita `paths.processed_pipeline_id`. Las fechas, textos, gráficos, código, aprobación y parámetros de entrenamiento se registran como procedencia o reportes, sin formar parte de esa identidad.
- **Purga:** aprobar el mismo valor candidato conserva el pipeline. Basta repetir 03 con el YAML actualizado; regenerar con 02 solo si cambia el valor efectivo u otra transformación/dato. Esto sustituye el procedimiento de dos rondas descrito en la entrada v0.5.0.
- **Persistencia:** esquema 2, `data/processed/<pipeline_id>/S*/B*/` y `folds/S*/val_B*/`. Publicación desde una carpeta provisional; integridad obligatoria antes de reutilizar. Reportes y aprobación fuera de los datos inmutables. Las corridas históricas no se migran ni se sobrescriben.
- **Código:** operaciones y bucles visibles en los notebooks; Morlet y antialias por canal en 02. Lectura conjunta de las variables permitidas con hash antes y después. Persistencia en `io`, reportes en `reporting`. Se retiran `steps`, la compuerta EDA, contextos duplicados, documentos de comandos repetidos, dependencias repetidas e inventario manual `SHA256SUMS.txt`.
- **Estado científico:** se conservan exclusiones, filtros, retardo, particiones, escaladores, arquitectura y valores pendientes. Los datos reales no están disponibles en este entorno: V01–V14 real no se declara aprobado. Las pruebas de integración ejecutan las celdas portables de 01/02/03 con datos sintéticos; se omiten únicamente el montaje/actualización propios de Colab y la invocación recursiva de pytest desde 03.
- **Validación local:** 250 pruebas PASS, cero fallos y cero omisiones, en 386.62 s (CPU). Comando: `python -m pytest -q --basetemp results/t10 --tb=short --junitxml=results/reorganization_tests.xml`, con TMP/TEMP dentro de `results/tmp`. El EDA sintético produjo sus 11 secciones COMPLETE y 41 figuras; 02 publicó sin EDA; 03 conservó el mismo identificador y V04/V09 quedaron BLOCKED con purga pendiente. La prueba separada de aprobación del mismo valor conservó la identidad y habilitó esas comprobaciones. Evidencia local ignorada por Git: `results/reorganization_tests.xml` y entregables de `results/t10/`.
- **Comprobaciones adicionales:** importación/compilación de módulos, entorno de dependencias comunes, enlaces locales de documentación y `git diff --check` satisfactorios. Instalación editable local actualizada a 0.6.0, sin reinstalar PyTorch/CUDA.
- **Orden de ejecución en Colab:** commit/push → 02 → copiar el identificador al YAML → commit/push → 03. El 01 puede ejecutarse independientemente. La purga, épocas y batch común siguen pendientes; no se modifica la aprobación científica con estas pruebas.

## 2026-10-19 — Entrenamiento reutilizable y comprobación breve de recursos

- **Autorización:** la persona usuaria pidió continuar la implementación, propuso 50 épocas y solicitó una factibilidad simple, con comprobaciones esenciales. Se prepara el código antes de disponer de V01–V14 real; la ejecución experimental continúa requiriendo el pipeline aprobado. No se ejecutan entrenamientos reales, búsquedas ni evaluación final.
- **Presupuesto:** `training.search_epochs_candidate: 50` configura el primer piloto. `search_epochs`, `final_max_epochs` y el batch común permanecen pendientes de evidencia; la propuesta no se convierte automáticamente en una decisión final.
- **Notebook 04:** operaciones y bucle de épocas visibles. Usa un fold de desarrollo del sujeto con más canales. Integra tres modelos representativos con dos actualizaciones de calentamiento y tres medidas, incluyendo forward, backward y AdamW. Registra GPU, memoria pico y tiempo. El batch inicial sigue siendo 128; un OOM se informa y detiene el piloto, sin reducir el batch automáticamente. No se añade otro notebook de factibilidad.
- **Entrenamiento:** funciones compartidas en `training.py`, ventanas en `datasets`, checkpoints en `io` y reportes en `reporting`. AdamW sin scheduler; validación reconstruida con overlap-add y métricas en unidades originales. No se selecciona el mejor pico ni se introduce early stopping. Se conserva el `pipeline_id` del 02 y un `run_id` propio.
- **Recuperación:** el último checkpoint completo conserva pesos, AdamW y estados aleatorios de PyTorch y del cargador. La escritura provisional no sobrescribe el checkpoint anterior si se interrumpe. La recarga exige el mismo pipeline, sujeto, fold y batch.
- **Alcance:** Optuna sigue pendiente; reutilizará estas funciones. Las mediciones de pocos batches solo estiman el costo de entrenamiento; el piloto mide épocas completas con validación. Un piloto aislado no demuestra convergencia de todos los sujetos y tasas de aprendizaje.
- **Pruebas iniciales:** 62 PASS en CPU (113.81 s), incluyendo ejecución de las celdas portables del 04 durante dos épocas sintéticas, equivalencia de Pearson reconstruido, recuperación de AdamW/aleatoriedad y fallos de memoria, valores no finitos y checkpoint interrumpido. No constituyen factibilidad en GPU ni aprobación científica. Evidencia local: `results/training_tests.xml`.
- **Regresión y correcciones:** la suite completa ejecutó 264 casos: 260 PASS, un fallo del fixture de recuperación (intentaba modificar un dataclass inmutable) y tres errores del fixture EDA por cálculos aún ausentes en sus celdas. Se corrigió el fixture con una copia y se completaron en 01 los cálculos explícitos requeridos por las figuras, usando las mismas funciones numéricas que el adaptador CLI. La repetición de EDA y entrenamiento dio 22 PASS y reveló las dependencias restantes del EDA; completadas estas, las 11 pruebas de EDA pasaron en 115.37 s, con 11 secciones COMPLETE y 41 figuras. El resultado más reciente de cada uno de los **264 casos es PASS**, sin omisiones ni fallos pendientes; se repitieron las pruebas afectadas, no toda la suite una segunda vez. Evidencia: `results/training_regression.xml`, `results/training_corrections.xml` y `results/eda_corrections.xml`. Compilación de módulos y `git diff --check` satisfactorios. Único aviso de pytest: el backend gráfico de CPU no muestra ventanas interactivas; la figura sí se guardó.

## 2026-10-19 — Implementación autorizada de Optuna 05/06

- **Autorización y alcance:** la persona usuaria aprobó preparar las dos etapas y sus notebooks, con ejecución real condicionada al pipeline aprobado, purga, batch común y épocas decididos. Se conservaron todos los valores pendientes y los cambios locales anteriores. No se ejecutaron búsquedas experimentales ni se accedió a etiquetas de test.
- **Método:** etapa 1, 20 trials × B2/B4 por sujeto, learning rate y weight decay log-uniformes, arquitectura fija. Etapa 2, 10 trials × cinco folds, tres parámetros arquitectónicos y mejores LR/WD de etapa 1 para el mismo sujeto. La referencia se encola dentro de las diez propuestas; las repeticiones cuentan y quedan registradas. TPE secuencial con semillas, sin pruning ni early stopping; objetivo: media de Pearson reconstruido de la última época de todos los folds.
- **Organización:** `optimization.py` administra propuestas, presupuestos y transferencia entre etapas. Los notebooks muestran bucles de sujetos, trials, folds y épocas, usando las funciones existentes de entrenamiento. Persistencia en `io`; reportes en `reporting`. Se añadió el extra opcional `search` (Optuna 4.x, probado con 4.9.0), sin reinstalar PyTorch/CUDA. El 03 no importa ni ejecuta las pruebas de Optuna: mantiene todas las comprobaciones V01–V14.
- **Persistencia:** SQLite local y publicación atómica de base + sampler en `results/optimization/<pipeline_id>/<sujeto>/<etapa>/study.zip`; un único runtime escritor por estudio. Reabrir conserva las propuestas de TPE, recupera el trial RUNNING y omite folds ya terminados. Cada fold conserva un `run_id`, el `pipeline_id` original y el último checkpoint completo con AdamW, RNG e historial. Los CSV se reconstruyen desde ese checkpoint al recuperar una época. Un fallo numérico, OOM o de datos detiene el estudio con diagnóstico, sin sustituir trials ni ampliar presupuestos automáticamente.
- **Pruebas:** 88 PASS de regresión (106.93 s), dos comprobaciones de integración PASS (63.88 s) y dos de publicación/errores PASS (5.91 s): 90 casos distintos aprobados. Las pruebas de presupuesto contabilizan los 270 fits con objetivos sintéticos, no con 270 entrenamientos experimentales. Las celdas reales de 05/06 ejecutaron U-Net en CPU sobre folds sintéticos, recuperaron una interrupción entre épocas, descartaron una fila CSV posterior al checkpoint y transfirieron los parámetros de etapa 1. También se comprobó corrupción, publicación interrumpida y recuperación del resultado tras `tell`. El primer recorrido detectó conexiones SQLite sin cerrar en Windows; se corrigieron y las pruebas dependientes pasaron. Evidencias locales: `results/optimization_regression.xml`, `results/optimization_integration.xml`, `results/optimization_publication.xml`.
- **Comprobaciones adicionales:** 264 pruebas de las etapas previas se pueden recolectar con la importación de Optuna bloqueada; esto comprueba independencia de dependencias, no una nueva ejecución completa de esas 264 pruebas. Notebooks válidos, celdas compilables y sin salidas guardadas; `git diff --check` satisfactorio. El aviso gráfico de Agg en el piloto sintético no impide guardar la figura.
- **Pendiente real:** commit/push, 02 seleccionado, 03 aprobado, 04 en GPU y decisión de batch/épocas; luego ejecutar 05 → 06. La evaluación final A/B/C no se implementó en esta continuación y las pruebas sintéticas no aprueban V01–V14 real.
