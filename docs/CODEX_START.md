# Encargo para Codex: implementar TT ECoG desde cero

Implementa el proyecto descrito por los documentos presentes en este repositorio. Antes de generar código, lee `AGENTS.md`, `docs/protocolo_implementacion.md`, `docs/dataset_spec.md`, `docs/validation_requirements.md` y `configs/experiment_config.yaml`.

**No reutilices implementaciones previas ni infieras parámetros que están `null`.** Debes preservar el número de épocas sin decidir y no comenzar Optuna hasta que exista una aprobación verificable del paso 3.

## Primer encargo: solo EDA + QC

- Crear estructura Python `src/tt_ecog/` instalable y notebook `01_exploracion_ecog.ipynb` para tres sujetos.
- Implementar validación de formato `.mat`, hashes y gráficas de datos crudos, amplitud, MAD, discontinuidades, planitud y PSD.
- El análisis debe generar `qc_channels.csv`, `qc_events.csv`, `qc_summary.csv`, `qc_decisions.json` propuesto (no aprobado por defecto) y gráficas.
- No incluir decisiones especificas para un canal/sujeto en el código. No abrir etiquetas test.
- Implementar tests con señales sintéticas y reportar que falta para una aprobación humana.
- No modificar datos de origen, no entrenar y no ejecutar TPE.

## Segundo encargo: preprocesamiento

Solo tras aprobar `qc_decisions.json`, implementar `02_preprocesamiento_ecog.ipynb` y módulos `preprocessing.py`, `splits.py`, `datasets.py`. Aplicar QC desde datos crudos antes de transformar, respetar segmentos, guardas y límites de fold, alinear ECoG(t) con flexión(t+0.2s), generar etiquetas validas y escaladores por fold sin fugas.

## Tercer encargo: verificación

Primero implementar el modelo U-Net parametrizable mínimo y funciones de loss/métricas necesarios para V13-V14, **sin entrenar ni ejecutar Optuna**. Después implementar `03_verificacion_pipeline.ipynb` y pruebas V01-V14. Si una prueba falla, corregir el componente y volver a comprobarlo. Reportar `verification_report.json`, `verification_summary.md` y manifiesto con hashes. Marcar como BLOQUEADO cualquier prueba dependiente de datos ausentes: nunca declarar PASS por omision.

## Siguientes encargos (no ejecutar sin aprobación)

Completar entrenamiento del modelo, perfilar Colab Pro; seleccionar después el número fijo de épocas; configurar Optuna TPE persistente para la etapa 1 y la etapa 2; entrenar variantes A/B/C y evaluar finalmente test con criterios congelados.

**Formato de cada entrega de Codex:** cambios de archivos, comandos ejecutados, pruebas y resultados, limitaciones, pendientes y solicitud de aprobación para la etapa siguiente.
