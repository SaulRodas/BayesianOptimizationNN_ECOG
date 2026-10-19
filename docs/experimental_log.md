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
