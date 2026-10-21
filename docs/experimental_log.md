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

## 2026-10-19 — Notebook 07: selección de duración y evaluación final A/B/C

- **Autorización:** la persona usuaria pidió implementar el 07 y eligió explícitamente **B5**
  como validación interna común para A/B/C y los tres sujetos. Se actualizan protocolo,
  contrato, YAML e instrucciones. Máximo final, paciencia y `min_delta` permanecen pendientes;
  las 50 épocas propuestas siguen correspondiendo al piloto.
- **Implementación:** `07_evaluacion_final.ipynb` muestra selección de épocas con el fold B5
  purgado, refit desde cero con todo desarrollo, congelación y evaluación. A conserva
  parámetros de referencia; B utiliza LR/WD de 05; C añade arquitectura de 06. Se exigen
  estudios completos del mismo sujeto/pipeline. Se usan las funciones existentes de AdamW,
  pérdida, métricas, preprocesamiento y ventanas; no hay concatenación temporal entre bloques.
- **Recuperación y trazabilidad:** `paths.final_run_id` selecciona explícitamente la corrida.
  Checkpoints conservan AdamW, RNG e historial. Un solo par de escaladores por sujeto se
  ajusta con desarrollo válido y sirve a A/B/C. `frozen.json` se escribe atómicamente después
  de completar todos los modelos, vincula decisiones y hashes al mismo `pipeline_id` y
  congela componentes numéricos, sin dependencia operativa del EDA o de sus figuras.
- **Test:** el lector requiere aprobación V01–V14 y modelos congelados antes de resolver las
  etiquetas. Lee exclusivamente `test_data`/`test_dg`, comprueba formato clásico/HDF5 y hashes
  antes/después, y exige el mismo archivo de competencia que originó el pipeline. Las
  máscaras, tiempos, escaladores y cobertura son comunes a A/B/C. Resultados por sujeto se
  publican solo completos; una reanudación no reentrena modelos congelados ni reabre test
  ya terminado. Tablas, figuras, predicciones, verdad alineada y cobertura quedan en
  `results/evaluation/<run_id>/`, separados de los arrays de desarrollo.
- **Pruebas:** 140 PASS en regresión (79.02 s), 38 PASS en integridad/lectura (16.57 s) y
  dos PASS de integración (76.17 s): **142 casos distintos** entre estas corridas. Los casos
  repetidos no se suman como pruebas nuevas. Se ejecutaron las celdas del 07 con U-Net y
  archivos MATLAB sintéticos, con interrupciones en selección y refit. Se comprobó igualdad
  exacta de las entradas visibles con las primitivas compartidas, tiempos, máscaras y
  etiquetas; además, recuperación, corrupción, publicación interrumpida, prohibición de
  lectura de test desde desarrollo y métricas en unidades originales. El aviso de Agg en
  la prueba del piloto 04 solo indica que el backend de pruebas no abre una ventana gráfica.
  Evidencias: `results/final_regression.xml`, `results/final_integrity.xml` y
  `results/final_notebook_integration.xml`.
- **Alcance real:** no se ejecutaron entrenamientos experimentales, búsquedas reales ni se
  abrieron etiquetas de test reales. No se declara V01–V14 aprobado. Faltan ejecutar las
  etapas reales en Colab, documentar aprobación y decidir los parámetros aún `null` antes de
  habilitar el 07. Se preservan cambios locales y corridas históricas; no se hizo commit/push.

## 2026-10-19 — Decisión: purga de 256 muestras a 100 Hz

- **Autorización:** decisión explícita de la persona usuaria tras revisar V04 y V09 de la primera ronda real del notebook 03, con el pipeline `b26db4c45dc163882b7ca8ca06432b822bd231e4ec35780a0285afd34c920fff` seleccionado en `e8bfe07`. El `run_id` de esa primera ronda no se registró; las cifras citadas abajo son las que reportó.
- **Valor:** `validation.purge_samples_at_100hz: 256`, igual a `purge_candidate_for_tests`. El contrato efectivo no cambia: se conserva el mismo `pipeline_id` y los folds publicados por el 02; basta repetir el 03.
- **Evidencia V09 (BLOCKED solo por la purga pendiente, sin problemas):** FIR pasabanda+notch de 1981 taps y antialias Kaiser dentro de diseño (líneas ≤ −40 dB; antialias ≤ −59 dB desde 50 Hz); Morlet 40 frecuencias, 7 ciclos, `zero_mean`. Medio soporte analítico de la cadena: FIR 990 + Morlet 139 + antialias 121 = 1250 muestras a 1000 Hz (1.25 s, ±125 a 100 Hz). Soporte medido con impulso: ±70 a 100 Hz (tolerancia 1e-6). Las máscaras exigen ±125 columnas de contexto dentro del bloque en los 15 bloques; rejilla de 100 Hz uniforme y alineada.
- **Evidencia V04 (BLOCKED solo por la purga pendiente, sin incidencias en los 15 folds):**
  - `minimal_isolating_purge = 0` en S1–S3 y B1–B5: el procesamiento por bloques independientes con máscaras de borde ya aísla contextos y etiquetas; la purga no es necesaria para evitar fuga.
  - Con 256, hueco efectivo de 381 muestras antes de validación y 361 después (3.81 s / 3.61 s). Sin purga sería ≈230 a cada lado (máscaras ±125 y retardo de 20).
  - Costo: 131 columnas del bloque posterior y 151 del anterior; columnas de train 30 869 (val_B1), 30 718 (B2–B4), 30 849 (val_B5) frente a 31 000 sin purga, ≈0.9 %. Idéntico en los tres sujetos.
- **Justificación:** la purga no se deriva del aislamiento (mínimo 0) ni de la longitud de ventana. Se elige como margen adicional frente a la dependencia temporal entre muestras vecinas de train y validación, con un costo menor al 1 % de las columnas de entrenamiento y sin regenerar el pipeline. No se usó Pearson ni ningún resultado de entrenamiento.
- **Seguimiento:** segunda ronda del 03 y aprobación del paso 3 registradas en la entrada del 2026-10-20.
- **Tests ajustados:** `tests/test_config.py` dejaba fijos en `null` la purga y `paths.processed_pipeline_id` del YAML real; ahora bloquea la purga aprobada (256, `approved`) y conserva la validación de formato del identificador. `tests/conftest.py::project` parte explícitamente de la purga pendiente para que los escenarios sintéticos de candidato no dependan de la decisión registrada. `tests/test_config.py`: 49 PASS en local; la suite completa se ejecuta en la segunda ronda del 03.

## 2026-10-20 — Aprobación del paso 3 (V01–V14)

- **Autorización:** la persona usuaria registró `verification.status: approved` y `verification.approved_manifest_file` tras la segunda ronda del notebook 03.
- **Corrida:** `20261020T021853Z_19507c`, con la purga aprobada (256) y el pipeline `b26db4c45dc163882b7ca8ca06432b822bd231e4ec35780a0285afd34c920fff`. Commit esperado: `adab2c6` (purga aprobada); el commit exacto queda en el reporte de la corrida.
- **Evidencia en Drive:** `TT_ECoG/results/verification/20261020T021853Z_19507c/`: `approved_pipeline_manifest.json`, `verification_report.json`, `verification_summary.md`, `test_results.xml` y `artifact_manifest.json`. El 03 solo emite el manifiesto de aprobación con V01–V14 PASS y pytest PASS; las etapas 04–07 vuelven a comprobar ese reporte y los hashes antes de cargar datos.
- **Impacto:** los datos y folds conservan el mismo `pipeline_id`. Un cambio posterior en `preprocessing.py`, `splits.py`, `datasets.py`, `model.py`, `losses.py`, `metrics.py`, `validation.py`, `prep/blocks.py`, `prep/folds.py` o en las secciones `model`, `training.loss` y `metrics` del YAML invalida la aprobación y exige repetir el 03.
- **Siguiente:** notebook 04 en GPU (recursos y piloto de 50 épocas candidatas); después decidir `training.batch_size` y `training.search_epochs`.

## 2026-10-20 — Carga de datos en paralelo para el entrenamiento

- **Autorización:** la persona usuaria aprobó optimizar la carga de datos antes de decidir `training.search_epochs`.
- **Motivo (medido en el 04, S3/val_B2, batch 128, float32):** tiempo por actualización completa (copia a GPU, forward, backward y AdamW) de 0.138 / 0.361 / 0.579 s para las arquitecturas pequeña / referencia / grande, ~233 pasos por época: 32 / 84 / 135 s por época sin carga de datos. El piloto de referencia midió ~134 s por época con validación, así que ~50 s (≈37 %) correspondían a armar batches en el proceso principal (`num_workers=0`; ~335 MB por batch). Memoria pico ≤ 1.24 GB reservados.
- **Cambio:** `make_loaders` recibe `workers` desde `training.loader_workers: 2`. Solo el loader de entrenamiento usa procesos auxiliares y, con CUDA, `pin_memory`; la copia en `train_batch` es `non_blocking`. Validación sin workers (un batch). `persistent_workers` permanece desactivado: con PyTorch 2.14, el iterador toma una semilla del generador al crearse y la reutilización de workers dejaría de hacerlo desde la época 2, cambiando el orden. Las ventanas son una función determinista del índice y el orden lo decide el generador del proceso principal, por lo que batches, orden, estado del generador guardado en checkpoints y resultados no cambian.
- **Alcance:** solo rendimiento. `training.py`, `config.py`, YAML, notebooks 04–07 y tests; `datasets.py` y los demás componentes verificados por el 03 no cambian, así que la aprobación de `20261020T021853Z_19507c` sigue vigente. El nuevo campo queda registrado en el `config.json` de cada corrida.
- **Pruebas locales (CPU, Windows):** nuevo test de equivalencia 0 frente a 2 workers durante tres épocas (mismos tensores, orden y estado del generador) y validación del campo. `tests/test_training.py` + `tests/test_config.py`: 66 PASS. Ejecución sintética de las celdas de 05/06 y 07: PASS. Se corrigió `test_training_blocked_before_any_approval`, que dependía de que el YAML real no tuviera aprobación. Dos pruebas de `tests/test_optimization.py` con trials sintéticos (sin `DataLoader`) fallan en este equipo con `PermissionError` al reemplazar `study.zip` dentro de la carpeta sincronizada por OneDrive; no dependen de este cambio y quedan por confirmar fuera de OneDrive o en Colab.
- **Pendiente:** medir en la GPU de Colab el tiempo real por época con workers (próximo piloto) antes de recalcular el costo de 30/40/50 épocas.

## 2026-10-20 — Protocolo v1.3: reevaluación de los finalistas de la etapa 1

- **Autorización:** la persona usuaria eligió, frente a usar 3 folds en todos los trials de la etapa 1 o combinar ambas opciones, reevaluar a los 3 mejores trials en los folds restantes.
- **Motivo:** el LR/WD de la etapa 1 se arrastra a la etapa 2 y a las variantes B y C. Elegirlo con dos folds favorece al ganador por azar, porque se elige y se mide en los mismos folds. Reevaluar a 3 finalistas en 3 folds nuevos reduce la varianza del puntaje final (de 2 a 5 folds) y corrige ese sesgo con +27 entrenamientos, frente a +60 de usar 3 folds en los 20 trials.
- **Regla:** con los 20 trials completos, los 3 mejores por la media de B2/B4 (empate: el trial anterior) se entrenan en B1, B3 y B5 con el mismo contrato (arquitectura de referencia, su LR/WD, semilla del fold, épocas fijas, última época). Se elige el finalista con mayor media de los cinco folds (empate: el mejor de la etapa 1). No se crean trials ni cambia la guía de TPE. Presupuesto de búsqueda: 120 + 27 + 150 = **297** entrenamientos.
- **Implementación:** `optimization.stage1.reevaluation` y `budgets` en el YAML, validados al cargar (bloques que completan los cinco folds, finalistas ≤ trials, presupuesto coherente). `optimization.py`: candidatos, estado persistido en el mismo `study.zip` de la etapa 1, corrida por finalista y fold con ubicación publicada antes de entrenar, reanudación y registro de fallos; `stage1_selection` sustituye a la selección por B2/B4 en la etapa 2 (contrato `parent`) y en el 07 (`selected_variants`). `study_summary.json` incluye el mejor trial por B2/B4, la reevaluación y los estados `REEVALUATION_PENDING`/`REEVALUATION_FAILED`. Notebook 05: sección visible con el bucle de finalistas, folds y épocas, y tabla de finalistas. `validation.stage1_validation_blocks` (parte del `pipeline_id`) no cambia: el pipeline y la aprobación del 03 siguen vigentes.
- **Documentación:** protocolo v1.3 (nota de cambio, §10.2, §10.4, §10.5, §11.1 y plan), `AGENTS.md`, `README.md` y `docs/CODEX_START.md`.
- **Pruebas locales (CPU, Windows):** 96 PASS en `tests/test_config.py`, `tests/test_optimization.py`, `tests/test_final_evaluation.py` y `tests/test_training.py`, incluidas la ejecución sintética de las celdas del 05 (trials, reevaluación) y del 06 (hereda el finalista) y la herencia de B/C en el 07. Seis pruebas de Optuna que llenan 20 trials seguidos fallan en este equipo con `PermissionError` al reemplazar `study.zip` dentro de la carpeta sincronizada por OneDrive (el mismo error ya visto en pruebas sin relación con este cambio); con rutas largas desactivadas no se pudieron ejecutar fuera de OneDrive. Su lógica se comprobó con un estudio de Optuna en memoria y las mismas funciones: finalistas, selección por cinco folds distinta del ganador de B2/B4, empate, espera de la reevaluación, fallo registrado, estudio incompleto y presupuesto 297. **Pendiente:** ejecutar `tests/test_optimization.py` completo en Colab (Linux) antes de lanzar el 05 real.

## 2026-10-20 — Diagnóstico de convergencia para `search_epochs` (implementación)

- **Autorización:** la persona usuaria aprobó ampliar el diagnóstico (decidió invertir en cómputo adicional): tres sujetos, siete corridas por sujeto, horizonte de 60 épocas y la regla de pérdida ≤ δ. Los learning rates del diagnóstico no entran en la etapa 1 ni la guían.
- **Diseño (`convergence_study` del YAML, validado al cargar):** S1–S3, fold B2, 60 épocas, candidatos 30/40/50, horizonte 56–60. Por sujeto: referencia con la semilla del fold y con semilla +1000; LR 3e-5, 2.5e-4 y 7.5e-4 con la arquitectura de referencia; k3-L3-f32 y k7-L5-f96 con el LR de referencia; weight decay de referencia. 21 corridas, fuera del presupuesto de búsqueda.
- **Regla (protocolo §9.2, fijada antes de ver resultados):** δ por sujeto = máximo entre la diferencia de semillas de la referencia en el horizonte y la variación entre épocas de cada semilla en ese tramo. Por grupo y candidato E: ganador = mejor Pearson en la época E; pérdida = mejor horizonte del grupo − horizonte del ganador. Se propone el menor E con pérdida ≤ δ en todos los sujetos y grupos; si ninguno, el mayor candidato con la limitación documentada.
- **Implementación:** `src/tt_ecog/convergence.py` (plan, registro por pipeline y batch, reanudación, carga de resultados que exige corridas completas y una sola GPU incluidas las reanudaciones, regla de decisión). Notebook 04: sección `convergence` con plan, bucle visible por corrida y época, decisión, tabla y curvas; guarda `registry.json`, `convergence_decision.json` y `convergence_curves.png` en `results/training/convergence/<pipeline_id>/batch_<b>/`. No modifica el YAML. Protocolo §9.2 y README actualizados.
- **Pruebas locales (CPU, Windows):** `tests/test_convergence.py` y `tests/test_config.py`: 71 PASS (plan de 21 corridas, configuraciones inválidas, regla con curvas sintéticas —mismo ganador, ganador rápido rechazado, ningún candidato, tolerancia de ruido, curvas incompletas o no finitas—, corridas incompletas o con GPU mezclada y ejecución real de las celdas del 04 con siete U-Net, una desconexión a mitad, reanudación y reapertura sin reentrenar). `tests/test_training.py`: 17 PASS (la celda `setup` del 04 cambió); tardó 18 min frente a ~2 min en la corrida anterior, sin causa identificada.
- **Pendiente:** elegir la GPU (medir L4/A100 con la celda de recursos y `nproc`), commit/push y ejecutar el diagnóstico en esa GPU; después decidir y registrar `training.batch_size` y `training.search_epochs`.

## 2026-10-21 — Interpolación lineal del decoder vectorizada (`model.py`)

- **Autorización:** la persona usuaria aprobó el cambio y la repetición del 03.
- **Evidencia (A100-SXM4-40GB, S3/val_B2, batch 128, datos ya en GPU):** paso completo de 0.030 / 0.232 / 0.390 s para las arquitecturas pequeña / referencia / grande; `cudnn.benchmark` sin efecto. El perfil de un paso de la referencia atribuyó **~220 ms de ~228 ms de GPU (96.5 %)** a `upsample_linear1d` (ida y vuelta, 5 niveles); las convoluciones cuDNN con TF32 sumaron ~3 ms. El kernel CUDA de PyTorch para esa interpolación paraleliza solo sobre las posiciones de salida (16–256) y recorre batch × canales en serie, por lo que la A100 apenas superaba a la T4. Copia de un batch (~335 MB): 0.071 s desde memoria normal frente a 0.027 s fijada; la tabla de recursos usa la primera y sobrestima el paso real.
- **Cambio:** `F.interpolate(..., mode="linear", align_corners=False)` se sustituye por `linear_upsample`, que aplica la misma fórmula de índice fuente y pesos de PyTorch como selección de índices y combinación ponderada (índices y pesos en caché por longitud, tipo y dispositivo). Sigue siendo interpolación lineal + Conv1d; el protocolo no cambia.
- **Pruebas locales (CPU):** valores y gradientes iguales a `F.interpolate` (tolerancias de `torch.testing.assert_close`) en 18 combinaciones de longitudes y tipos, con ampliación, reducción, misma longitud y longitud 1; el modelo completo da salida y gradientes iguales con ambas implementaciones. `tests/test_architecture.py` (V13): 57 PASS. Suite completa salvo `tests/test_optimization.py`: 325 PASS (un fallo inicial se debió a que el notebook 04 en disco había perdido sus celdas del diagnóstico al guardarse una versión anterior desde el editor; restaurado desde `55536ee` con sus salidas, el test pasó).
- **Impacto:** `model.py` está entre los componentes verificados, así que la aprobación `20261020T021853Z_19507c` deja de ser válida para 04–07: repetir el 03 (no el 02; el `pipeline_id` no depende del modelo) y registrar la nueva aprobación. El piloto en T4 usó la implementación anterior y queda como evidencia histórica; el diagnóstico de convergencia y Optuna aún no se habían ejecutado.
- **Pendiente:** repetir el 03; medir de nuevo el paso en las GPU candidatas antes de elegir GPU y `loader_workers`.
