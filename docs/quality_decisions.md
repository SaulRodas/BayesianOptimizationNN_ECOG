# Registro de decisiones de control de calidad

**Estado inicial: SIN APROBAR.** Está plantilla se completara después de ejecutar el EDA sobre las señales de desarrollo. Ningun sujeto debe recibir exclusiones precargadas sin evidencia de la señal.

## Proceso de revision

1. Generar estadísticas por canal: mediana, MAD, percentiles, extremos, discontinuidades, planitud, PSD y duración de eventos.
2. Identificar episodios candidatos sin consultar flexión, predicciones, train loss o Pearson.
3. Revisar visualmente los candidatos y describir evidencia instrumental; no atribuir causa física sin evidencia suficiente.
4. Para cada evento aprobar una acción: conservar, descartar canal completo, máscara temporal ampliada por soporte o reparacion local validada.
5. Registrar razones, índices cero-basados, tiempos físicos, fraccion afectada, hashes y firma/aprobación.
6. Aplicar todas las decisiones en `02_preprocesamiento_ecog.ipynb` desde la señal cruda.

## Registro por sujeto

| Sujeto | Canales excluidos (0-based) | Intervalos enmascarados (s) | Evidencia | Revisado por | Estado |
|---|---|---|---|---|---|
| S1 | Pendiente | Pendiente | Pendiente | Pendiente | PENDING |
| S2 | Pendiente | Pendiente | Pendiente | Pendiente | PENDING |
| S3 | Pendiente | Pendiente | Pendiente | Pendiente | PENDING |

Los resultados del EDA deben generar `results/eda_qc/qc_decisions.json` con schema versionado. Un registro `PENDING` bloquea el preprocesamiento definitivo.

## Controles de sesgo

- Umbrales definidos por propiedades de adquisición y sujetos tratados con el mismo procedimiento.
- Si se revisa descriptivamente todo el desarrollo para aprobar excluidos por sujeto, declararlo en el método: no equivale a QC estrictamente ajustado por fold.
- Reglas adaptativas que aprendan estadísticas temporales en CV se calibraran solo con fold-train.
- No excluir muestras por alto error de predicción ni por bajo Pearson.
- No consultar etiquetas de test para corregir QC.

## Contrato JSON

Utilizar `configs/qc_decisions.template.json` como punto de partida y validar contra `schemas/qc_decisions.schema.json`. `approved` requiere revisión y evidencia; no cambiar estados de forma automática.
