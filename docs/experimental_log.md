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
| QC revisado y aprobado | qc_decisions.json + reportes | PENDING |
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
