# Encargo para agentes: implementar TT ECoG por etapas

Implementa el proyecto descrito por los documentos de este repositorio. Antes de generar código, lee `AGENTS.md`, `docs/protocolo_implementacion.md` (v1.1), `docs/dataset_spec.md`, `docs/validation_requirements.md` y `configs/experiment_config.yaml`.

**No reutilices implementaciones previas ni infieras parámetros que están `null`.** Debes preservar el número de épocas sin decidir y no comenzar Optuna hasta que exista una aprobación verificable del paso 3.

## Primer encargo: EDA de justificación — HECHO (v0.3.0)

- Paquete instalable `src/tt_ecog/` y notebook `01_exploracion_ecog.ipynb` para tres sujetos, ejecutado desde VS Code con kernel Colab.
- `tt_ecog.preprocessing`: primitivas únicas y por segmento (exclusión de canales, CMR, FIR pasabanda+notch, Morlet, antialias + decimación, guante 25 Hz e interpolación cúbica, retardo, fábricas de escaladores).
- `tt_ecog.eda`: once secciones que justifican cada decisión de preprocesamiento; figuras, tablas y manifiestos en `MyDrive/TT_ECoG/results/eda/<run_id>/`.
- La etapa de QC con umbrales y `qc_decisions.json` fue **eliminada** (decisión de la persona usuaria, 2026-10-18). La exclusión de canales es una lista fija en el YAML (`channels.excluded`: S1-54, S2-20, S2-37) con evidencia ECoG. La regla "no codificar decisiones por canal/sujeto en el código" se cumple manteniéndolas solo en el YAML.
- Pendiente de la persona usuaria: ejecutar el notebook con los datos reales y revisar las figuras.

## Segundo encargo: preprocesamiento — IMPLEMENTADO (v0.4.0), pendiente la corrida real

Hecho por bloque: B1–B5 se procesan de forma independiente, la purga se aplica a nivel de columnas y los folds usan el candidato etiquetado mientras la purga sea `null`. Ver la entrada v0.4.0 de `docs/experimental_log.md`. Encargo original:

Implementar `02_preprocesamiento_ecog.ipynb` y completar `splits.py` y `datasets.py`, **reutilizando** las primitivas de `tt_ecog.preprocessing` (no reimplementarlas). Aplicar la exclusión de canales desde datos crudos antes de la CMR, procesar por segmentos independientes de cada fold con guardas, alinear ECoG(t) con flexión(t+0.2 s), interpolar el guante por segmento, generar etiquetas válidas y escaladores por fold sin fugas, y persistir arrays, máscaras, escaladores y manifiestos (`channel_map.json`, `filter_support.json`, `split_manifest.json`).

## Tercer encargo: verificación — IMPLEMENTADO (v0.5.0), pendiente la corrida real y la aprobación de la purga

Hecho: U-Net mínima con los detalles elegidos por la persona usuaria, pérdida, métricas, overlap-add, `tt_ecog.verification` y notebook 03. Ver la entrada v0.5.0 de `docs/experimental_log.md`. Encargo original:

Primero implementar el modelo U-Net parametrizable mínimo y funciones de loss/métricas necesarios para V13-V14, **sin entrenar ni ejecutar Optuna**. Después implementar `03_verificacion_pipeline.ipynb` y pruebas V01-V14. Si una prueba falla, corregir el componente y volver a comprobarlo. Reportar `verification_report.json`, `verification_summary.md` y manifiesto con hashes. Marcar como BLOQUEADO cualquier prueba dependiente de datos ausentes: nunca declarar PASS por omision.

## Siguientes encargos (no ejecutar sin aprobación)

Completar entrenamiento del modelo, perfilar Colab Pro; seleccionar después el número fijo de épocas; configurar Optuna TPE persistente para la etapa 1 y la etapa 2; entrenar variantes A/B/C y evaluar finalmente test con criterios congelados.

**Formato de cada entrega:** cambios de archivos, comandos ejecutados, pruebas y resultados, limitaciones, pendientes y solicitud de aprobación para la etapa siguiente.
