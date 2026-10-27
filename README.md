# TT ECoG.

Decodificación continua de cinco dedos con ECoG y U-Net 1D. Metodología: [protocolo v1.4](docs/protocolo_implementacion.md). Configuración única: `configs/experiment_config.yaml`.

## Ejecutar en VS Code con kernel Colab.

1. Guardar `sub1_comp.mat`, `sub2_comp.mat`, `sub3_comp.mat` en `MyDrive/TT_ECoG/raw`. No colocar etiquetas de test ahí.
2. Hacer commit/push local. La primera celda actualiza el clon de Colab con `git pull --ff-only` y muestra el commit.
3. Ejecutar **02_preprocesamiento_ecog.ipynb**, operación por operación. El **01 es opcional** y solo produce visualizaciones descriptivas.
4. Copiar el `pipeline_id` emitido por 02 a `paths.processed_pipeline_id` del YAML; hacer commit/push.
5. Ejecutar **03_verificacion_pipeline.ipynb**. Comprueba exactamente los datos seleccionados y conserva su identidad.
6. Con la purga decidida y V01–V14 PASS, registrar la aprobación en `verification.status: approved` y
   `verification.approved_manifest_file` (ruta relativa a `TT_ECoG` del manifiesto emitido por 03).
   Hacer commit/push y ejecutar **04_entrenamiento.ipynb**: comprobación breve de GPU/memoria y piloto con las
   **50 épocas propuestas**. No hay un notebook adicional de factibilidad.
7. Tras decidir `training.batch_size` y `training.search_epochs`, hacer commit/push y ejecutar
   **05_optuna_entrenamiento.ipynb** y después **06_optuna_arquitectura.ipynb**.
8. Decidir `training.final_max_epochs`, `final_early_stopping_patience` y
   `final_early_stopping_min_delta` con desarrollo. Ejecutar **07_evaluacion_final.ipynb**:
   B5 para elegir duración y el σ de D → refit de A/B/C y de los miembros de D con desarrollo
   completo → congelar → test.
   Indicar `paths.test_labels_dir` solo para esta etapa, relativo a `TT_ECoG`.

El esquema 2 genera carpetas nuevas. Las corridas anteriores quedan intactas: no se convierten ni se sobrescriben.

## Identificadores y salidas

| Campo | Significado |
|---|---|
| `pipeline_id` | SHA-256 del contrato efectivo y los arrays preparados; compartido por 02, 03 y resultados posteriores. |
| `run_id` | Una ejecución y sus reportes. |
| Hash de archivo | Comprobación de integridad de un artefacto. |

El pipeline incluye datos de origen, exclusión, señal, particiones, purga, escalado y ventanas. No incluye fechas, rutas, figuras, notebooks, EDA, estado de aprobación ni parámetros posteriores de entrenamiento. La procedencia (commit, versiones, configuración completa) se registra por separado.

Datos: `TT_ECoG/data/processed/<pipeline_id>/S*/B*/`; folds: `folds/S*/val_B*/`; metadatos: `manifests/`. Reportes: `TT_ECoG/results/<etapa>/<run_id>/`. El EDA nunca guarda arrays procesados. Una preparación incompleta permanece provisional; una corrupta no se reutiliza.

## Responsabilidades del código

| Componente | Propósito |
|---|---|
| `io` | Lectura conjunta de desarrollo, guardado, manifiestos e identidad. |
| `preprocessing` | Única implementación de exclusión, CMR, FIR, Morlet, antialias, guante y alineación. |
| `splits`, `datasets` | Bloques, purga, ventanas, escaladores por fold y carga de datos preparados. |
| `prep` | Adaptador CLI y metadatos de bloques/folds; el notebook muestra las operaciones. |
| `descriptive`, `eda`, `plotting` | Estadística descriptiva y figuras; el EDA es independiente. |
| `reporting` | Registro de ejecuciones, tablas, figuras y diagnósticos. |
| `model`, `losses`, `metrics`, `validation` | U-Net, pérdida, métricas y reconstrucción overlap-add. |
| `verification`, `tests` | V01–V14 y regresiones sintéticas. |
| `training` | Actualización AdamW, épocas, validación reconstruida y medición breve de recursos; reutilizable por Optuna. |
| `optimization` | Propuestas TPE, presupuesto total, recuperación del trial y selección entre etapas. |
| `final_evaluation` | Política final de duración, parámetros A/B/C, variante D (semillas, promedio y suavizado con σ elegido en B5) e inferencia sin ajuste sobre test. |

`pyproject.toml` permite instalar el paquete; `requirements.txt` contiene las dependencias comunes. `.gitignore` protege datos y resultados; `.gitattributes` elimina salidas de notebooks al commitear. Los JSON conservan información que no está en los arrays: canales, tiempos, configuración, cobertura y procedencia.

## Instalación y comandos

Local: `python -m pip install -e ".[test,dev,model,search]"`; después `nbstripout --install --attributes .gitattributes`. El extra `model` instala PyTorch localmente y `search` instala Optuna. En Colab se comprueba el entorno y se conserva el PyTorch/CUDA disponible.

```bash
python -m tt_ecog.prep --config configs/experiment_config.yaml --drive-root /content/drive/MyDrive
python -m tt_ecog.verification --config configs/experiment_config.yaml --drive-root /content/drive/MyDrive
# Opcional:
python -m tt_ecog.eda --config configs/experiment_config.yaml --drive-root /content/drive/MyDrive
python -m pytest -q
```

## Límites y estado

La purga, épocas y batch común siguen pendientes. 02 usa la purga candidata solo para verificación. Si se aprueba el mismo valor, los datos conservan su identidad; si cambia, se genera otro pipeline. `load_fold` rechaza candidatos fuera de verificación.

La propuesta de **50 épocas** está en `training.search_epochs_candidate`; `search_epochs` y
`final_max_epochs` siguen en `null`. El 04 mide las arquitecturas pequeña, de referencia y grande
con el sujeto de más canales: dos pasos de calentamiento y tres medidos, incluyendo backward y AdamW.
Si falta memoria, cambiar el candidato de batch y repetir; no hay reducción automática.
El piloto guarda curvas por época y `checkpoint/last.pt` en `results/training/<sujeto>/<run_id>/`,
con el mismo `pipeline_id`. La prueba de recursos no demuestra convergencia ni certifica las 27 arquitecturas.
El tiempo inicial por época es una estimación; el piloto registra train y validación medidos.
Después, la sección `convergence` del 04 ejecuta el diagnóstico de `convergence_study`: 7 corridas
por sujeto hasta 60 épocas, retomables, y la regla del protocolo §9.2 propone el menor de 30/40/50
cuyo ganador pierde a lo más el ruido entre semillas. Resultados en
`results/training/convergence/<pipeline_id>/batch_<b>/` (`registry.json`, `convergence_decision.json`,
curvas); el YAML no se modifica.
Optuna está implementado en 05/06 y reutiliza estas funciones. El 04 no lo inicia. Se conserva la
última época, sin early stopping ni pruning; el objetivo es la media del Pearson reconstruido de
todos los folds. Etapa 1: 20 trials × B2/B4 por sujeto; después, los 3 mejores se reevalúan en
B1/B3/B5 y la etapa 2, B y C heredan el LR/WD con mayor media de cinco folds (v1.3).
Etapa 2: 10 trials × cinco folds por sujeto,
incluyendo la arquitectura de referencia dentro de las diez propuestas. Las repeticiones cuentan.
En total, 297 entrenamientos de búsqueda.

Los estudios se guardan en `results/optimization/<pipeline_id>/<sujeto>/stage1|stage2/`:
`study.zip` permite recuperar base y sampler juntos; `trials.csv`, `trials.json` y
`study_summary.json` permiten inspeccionar resultados. Los entrenamientos por fold conservan
los reportes y checkpoints en `results/training/`, enlazados desde cada trial. SQLite trabaja
localmente y publica una copia completa en Drive; usar **un solo runtime por estudio**.
Volver a ejecutar recupera el trial y la última época completa, sin ampliar el presupuesto.
Un FAIL detiene el estudio para diagnóstico; no se reemplaza automáticamente. La etapa 2 requiere
todos los trials de etapa 1 y la reevaluación de sus finalistas completos para el mismo sujeto,
pipeline, batch y épocas; la reevaluación vive en el mismo `study.zip` de la etapa 1.
Conservar la versión de Optuna al recuperar el sampler y cargar solo archivos propios.
La persistencia del sampler sigue la [guía oficial de Optuna](https://optuna.readthedocs.io/en/v4.5.0/tutorial/20_recipes/001_rdb.html).

El 07 guarda `results/evaluation/<run_id>/`: plan, procedencia, selección interna (con la
trayectoria B5 de la época elegida), checkpoints por sujeto/variante y por miembro de D
(`ensemble_smoothed_D/member_k/`), el σ de D con su tabla de B5 y escaladores comunes por sujeto.
Muestra el `run_id`; copiarlo a `paths.final_run_id` para reanudar la última época completa. No
selecciona la carpeta más reciente. `frozen.json` vincula los modelos de A/B/C y los miembros de D,
el σ de D (recalculado al congelar) y las decisiones al mismo `pipeline_id` antes de abrir test.
Después guarda trayectorias, máscara/cobertura, índices alineados, métricas por dedo y sujeto,
figuras A/B/C/D, la tabla de cada miembro de D y su promedio sin suavizar, y manifiestos de
integridad. Un resultado terminado se reutiliza sin reabrir etiquetas. Una interrupción durante
test solo permite repetir inferencia con los mismos modelos congelados; no elegir otra
configuración por su resultado.

V01–V14 requiere datos reales y pruebas sintéticas satisfactorias. La reorganización y sus tests sintéticos no aprueban el pipeline experimental. Las corridas históricas constan en [la bitácora](docs/experimental_log.md). No se ejecutan automáticamente entrenamientos, Optuna ni evaluación real. El acceso a etiquetas de test se limita a la celda final del 07 después de congelar decisiones.
