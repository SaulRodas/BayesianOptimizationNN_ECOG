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
