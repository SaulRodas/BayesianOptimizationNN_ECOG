# Instrucciones de trabajo para Codex — TT ECoG

## Propósito y jerarquía documental

Implementar desde cero un proyecto reproducible de decodificación continua de flexión de cinco dedos con ECoG mediante U-Net 1D y optimización bayesiana secuencial.

Orden de autoridad en conflictos: (1) `docs/protocolo_implementacion.md`; (2) `docs/validation_requirements.md` para criterios verificables; (3) `docs/dataset_spec.md` para contratos de datos; (4) `configs/experiment_config.yaml` para valores de ejecución; (5) estas instrucciones. Si dos documentos se contradicen, **no inventar una conciliacion**: registrar la discrepancia y solicitar autorización. Ningun documento autoriza a usar datos de prueba para tomar decisiones experimentales.

El número de épocas de búsqueda, la duración máxima final, el batch size común y la purga efectiva permanecen PENDIENTES. Los campos `null` del YAML son bloqueos deliberados, no licencias para sustituirlos por valores por defecto.

**Continuación autorizada (2026-10-19):** implementar el entrenamiento reutilizable y una
comprobación breve de recursos integrada en `04_entrenamiento.ipynb`, sin un notebook separado
de factibilidad. La persona usuaria propone 50 épocas: `training.search_epochs_candidate: 50`
para el piloto; no equivale a aprobar `search_epochs` ni `final_max_epochs`. Conservar únicamente
las comprobaciones necesarias para datos aprobados, aislamiento, finitud, cobertura y memoria.
Preparar este código no autoriza a ejecutar entrenamientos reales ni Optuna antes de V01–V14.

**Preparación de Optuna autorizada (2026-10-19):** implementar 05/06 y probar sus operaciones
con datos sintéticos antes de ejecutar experimentos. Mantener los presupuestos 20×2 y 10×5
(más la reevaluación de finalistas autorizada en v1.3),
la referencia dentro de las diez propuestas y el mismo `pipeline_id`. La ejecución real sigue
condicionada a V01–V14 aprobado, purga, batch común y épocas de búsqueda decididos. Recuperar
estudios sin ampliar presupuestos ni reemplazar trials fallidos automáticamente.

**Preparación de 07 autorizada (2026-10-19):** implementar selección de épocas, reentrenamiento
y evaluación A/B/C con pruebas sintéticas. La persona usuaria eligió **B5** como validación
interna común para los tres sujetos y variantes; después se reentrena con todo desarrollo,
manteniendo los bloques independientes y ajustando escaladores solo con desarrollo.
Máximo final, paciencia y min_delta siguen pendientes. Congelar los nueve modelos antes de
abrir test; no reentrenar ni retocar decisiones después. Preparar código no autoriza la ejecución
experimental, que requiere aprobación V01–V14, 05/06 completos y decisiones finales documentadas.

**Cambio metodológico autorizado (2026-10-20, protocolo v1.3):** los 3 mejores trials de la etapa 1
de cada sujeto (media de B2 y B4) se entrenan también en B1, B3 y B5 con el mismo contrato. La
etapa 2, B y C heredan el LR/WD del finalista con mayor media de los cinco folds. No se crean
trials ni cambia la guía de TPE; la búsqueda pasa de 270 a 297 entrenamientos.

**Cambio metodológico autorizado (2026-10-18, protocolo v1.1):** se eliminó la etapa de QC con umbrales, `qc_decisions.json` y aprobación. El flujo obligatorio es **02 Preprocesamiento → 03 Verificación**; el EDA es opcional (actualización autorizada v1.2). La exclusión de canales es una lista fija en el YAML (ver reglas). Ver `docs/experimental_log.md` y `docs/channel_exclusion.md`.

## Plan de implementación obligatorio

1. Crear cargadores y contratos del dataset, con comprobaciones de formato, hashes y tests sintéticos.
2. Mantener `01_exploracion_ecog.ipynb` como herramienta opcional: EDA descriptivo de los tres sujetos que justifica con gráficas cada decisión de preprocesamiento (exclusión de canales, CMR, banda, notch, Morlet, decimación, guante, retardo, escalado y bloques). Las transformaciones se aplican solo **en memoria** con `tt_ecog.preprocessing` para graficar: **no persistir datos transformados** en esta etapa.
3. Implementar `02_preprocesamiento_ecog.ipynb`: aplicar la exclusión fija de canales (`channels.excluded`) sobre datos crudos, referencia de mediana común instantanea, filtrado, Morlet, decimacion, alineación, particiones y escaladores por fold, reutilizando las primitivas de `tt_ecog.preprocessing` (ya implementadas y probadas). No concatenar segmentos separados temporalmente.
4. Implementar U-Net 1D parametrizable, funciones de perdida y métricas minimas para poder ejecutar V13 y V14, sin realizar entrenamientos experimentales.
5. Implementar `03_verificacion_pipeline.ipynb` y `tests/` según V01-V14 de `docs/validation_requirements.md`. Guardar evidencias y manifiestos PASS/FAIL. **No avanzar** con pruebas pendientes, omitidas o fallidas.
6. Si V01-V14 pasan, congelar hash de pipeline y documentar la aprobación. Si fallan, corregir SOLO el componente responsable y repetir las verificaciones dependientes.
7. Completar el entrenamiento reutilizable en `src/tt_ecog/` con notebooks como orquestadores, y medir factibilidad en la GPU efectivamente asignada por Colab Pro.
8. Ejecutar análisis de convergencia para **decidir después** `search_epochs`, `final_max_epochs` y la política final de parada; no fijar cifras sin evidencia.
9. Solo después de aprobar pasos anteriores, implementar y ejecutar etapa 1 TPE, etapa 2 TPE y evaluación final A/B/C.

## Reglas metodologicas no negociables

- Tres sujetos independientes: S1, S2, S3; cinco salidas por muestra; espacio arquitectonico 27 combinaciones.
- ECoG 40-200 Hz; notch 60/120/180 Hz; Morlet 40 frecuencias, 7 ciclos, `zero_mean=True`; representacion a 100 Hz; retardo objetivo de 200 ms en la dirección ECoG(t) -> flexión(t+0.2 s).
- No hay QC con umbrales ni aprobación. La exclusión de canales vive **solo** en `configs/experiment_config.yaml` (`channels.excluded`: S1-54, S2-20, S2-37, cero-basados), con razón y figura de evidencia ECoG; se aplica antes de la referencia, los filtros y las wavelets, igual en desarrollo, validación y test. **No reintroducir** umbrales, aprobaciones, máscaras temporales ni reparaciones locales sin autorización. No codificar excepciones por sujeto en el código ni elegir canales a partir de Pearson, predicciones o etiquetas.
- La exclusión se decidió con una inspección descriptiva de **todo el desarrollo**; no presentarla como un criterio anidado por fold. Las estadísticas adaptativas de transformaciones se ajustan exclusivamente con entrenamiento de cada fold.
- `train_dg` de desarrollo puede usarse en el EDA **solo de forma descriptiva** (gráficas confirmatorias de banda y retardo); no ajusta parámetros.
- No acceder a `test_dg` o archivos `*_testlabels.mat` en EDA, preparacion de folds, piloto, búsqueda o selección de épocas. El test es de uso exclusivo de `07_evaluacion_final.ipynb`, tras congelacion de decisiones.
- Dividir antes de cualquier operacion que aprenda estadísticas o use contexto temporal a traves de particiones. Procesar por segmentos cronológicos independientes con guardas; validar alcance de FIR, Morlet y alineación con tests de perturbación.
- Aplicar RobustScaler `(10, 90)` y MinMaxScaler ajustados con fold-train; validación y prueba solo `transform`.
- U-Net 1D con BatchNorm1d, GELU, dropout 0.3, interpolación lineal + Conv1d, skips concatenativos; `forward` conserva longitud 256 y entrega cinco dedos.
- Entrenar con AdamW sin scheduler; loss 0.5 MSE + 0.5(1-coseno temporal). Optuna maximiza `val_pearson_full` reconstruido con overlap-add.
- Durante Optuna, todos los entrenamientos usan la duración fija que se apruebe más adelante y puntuan la **ultima época**, sin early stopping sobre el mismo fold puntuado. Guardar curvas por época para diagnóstico sin escoger el mejor pico a posteriori.
- Selección final de época con validación interna de desarrollo; test solo una vez decidido todo. CCC, MSE y RMSE son métricas secundarias, NO objetivos multiobjetivo en la implementación principal.
- Etapa 1: 20 trials x 2 folds (B2, B4) por sujeto, 2 hiperparametros; después, los 3 mejores se reevalúan en B1, B3 y B5 y se elige el de mayor media de cinco folds (v1.3). Etapa 2: 10 trials x 5 folds por sujeto, 3 hiperparametros; incluye arquitectura de referencia dentro de las 10 propuestas. 297 entrenamientos de búsqueda en total (120 + 27 + 150), sin incluir pilotos ni entrenamientos finales.

## Reglas de código y seguridad

- Python modular y tipado cuando sea práctico. Una sola implementación de transformaciones, perdida y métricas; notebooks con operaciones explícitas en celdas, sin reimplementar las primitivas. Las gráficas antes/después del EDA llaman a `tt_ecog.preprocessing`; las alternativas de contraste (CAR, submuestreo directo, interpolación lineal) viven solo en `tt_ecog/eda/comparators.py` y nunca entran al pipeline.
- Configuración única por YAML y parámetros/paths relativos; prohibido repetir bloques de configuración escritos a mano en notebooks.
- No incluir tokens, secretos, rutas privadas ni claves W&B en código/documentación. No copiar sin inspección scripts heredados al repositorio.
- No guardar datos ECoG crudos, etiquetas de prueba, checkpoints ni estudios Optuna grandes en Git. Usar Google Drive o almacenamiento persistente externo y guardar hashes en manifiestos.
- No reinstalar PyTorch o CUDA a ciegas en Colab; comprobar compatibilidad GPU. Registrar versiones, GPU, memoria, seed y commit.
- No afirmar que un smoke test equivale a un entrenamiento completo, ni que CV purgado equivale a predicción causal en tiempo real.
- Los fallos no se silencian. Fallo de Pearson indefinido, NaN, cobertura incompleta, ventanas invalidas o mezcla entre conjuntos deben producir diagnóstico y FAIL.
- Mostrar plan y resultado por etapa a la persona usuaria; solicitar autorización antes de entrenamientos costosos o decisiones que cambien la metodología.

## Ejecución: VS Code local + kernel Colab

- El repositorio vive en la máquina local; el kernel del notebook es Colab (extensión oficial de VS Code), que **no ve archivos locales**.
- El código llega a Colab por Git: commit + push local; la primera celda del notebook hace `git pull --ff-only` y muestra el commit, que debe coincidir con el HEAD local.
- Datos crudos en `MyDrive/TT_ECoG/raw`. Resultados en `MyDrive/TT_ECoG/results/eda/<run_id>/` (figuras PNG en `fig/`, tablas, manifiestos); el notebook muestra esas mismas figuras y quedan en el `.ipynb` local al guardarlo.
- Las salidas del notebook no se commitean: filtro `nbstripout` declarado en `.gitattributes` (cada clon local ejecuta `nbstripout --install --attributes .gitattributes`).

## Definition of Done por etapa

- EDA opcional (sin bloquear otras etapas): tres sujetos descritos; las 11 secciones en `COMPLETE` con figuras, tablas y `artifact_manifest.json` en Drive; figura de evidencia de cada canal excluido presente.
- Preprocesamiento: arrays y metadatos trazables; exclusión de canales aplicada antes de la CMR con mapa original→efectivo; labels de test sin tocar.
- Paso 3: V01-V14 PASS real (no mocks que simulen exito), reporte, hashes y evidencia persistidos. De no haber archivos `.mat`, informar BLOQUEADO, no PASS.
- Factibilidad: GPU real, memoria peak en entrenamiento, batch común y costos documentados.
- Optuna: estudios persistentes recuperables con folds, semillas y parámetros registrados.
- Evaluación: tres variantes A/B/C, por sujeto y dedo; métrica Pearson principal, CCC y errores secundarios, misma cobertura y criterios.

Para iniciar: leer `docs/CODEX_START.md`, `docs/protocolo_implementacion.md` y `docs/validation_requirements.md`, confirmar la estructura del dataset sin inspeccionar las etiquetas de test y desarrollar por fases. No lanzar Optuna de forma automática.

## Reorganización autorizada v1.2 (2026-10-18)

- No requerir `channels.reference_run`, corrida EDA COMPLETE ni figuras para 02/03. Las exclusiones fijas y sus razones permanecen en el YAML.
- 02 publica un `pipeline_id` SHA-256 de contrato efectivo y arrays numéricos. 03 y etapas posteriores lo leen desde la selección `paths.processed_pipeline_id` y lo conservan. No usar hashes de notebook, fechas, aprobación o parámetros de entrenamiento como identidad de datos.
- Cada ejecución tiene un `run_id`; conservar procedencia completa y hashes individuales automáticamente. Aprobar el mismo valor de purga no cambia el pipeline.
- Usar esquema de persistencia 2; preservar corridas históricas, sin conversión ni sobrescritura. No publicar carpetas provisionales incompletas.
- Mostrar las operaciones reales en las celdas, con entradas y resultados inspeccionables. Persistencia en `io`, reportes en `reporting`; ninguna dependencia operativa del EDA.
