# Instrucciones de trabajo para Codex — TT ECoG

## Propósito y jerarquía documental

Implementar desde cero un proyecto reproducible de decodificación continua de flexión de cinco dedos con ECoG mediante U-Net 1D y optimización bayesiana secuencial.

Orden de autoridad en conflictos: (1) `docs/protocolo_implementacion.md`; (2) `docs/validation_requirements.md` para criterios verificables; (3) `docs/dataset_spec.md` para contratos de datos; (4) `configs/experiment_config.yaml` para valores de ejecución; (5) estas instrucciones. Si dos documentos se contradicen, **no inventar una conciliacion**: registrar la discrepancia y solicitar autorización. Ningun documento autoriza a usar datos de prueba para tomar decisiones experimentales.

El número de épocas de búsqueda, la duración máxima final, el batch size común, la purga efectiva y los umbrales definitivos de QC permanecen PENDIENTES. Los campos `null` del YAML son bloqueos deliberados, no licencias para sustituirlos por valores por defecto.

## Plan de implementación obligatorio

1. Crear cargadores y contratos del dataset, con comprobaciones de formato, hashes y tests sintéticos.
2. Implementar `01_exploracion_ecog.ipynb`: EDA y deteccion de QC en los tres sujetos; emitir `qc_decisions.json` y reportes. No transformar el dataset en está etapa.
3. Implementar `02_preprocesamiento_ecog.ipynb`: aplicar decisiones QC aprobadas sobre datos crudos, referencia de mediana común instantanea, filtrado, Morlet, decimacion, alineación, particiones y escaladores por fold. No concatenar segmentos separados temporalmente.
4. Implementar U-Net 1D parametrizable, funciones de perdida y métricas minimas para poder ejecutar V13 y V14, sin realizar entrenamientos experimentales.
5. Implementar `03_verificacion_pipeline.ipynb` y `tests/` según V01-V14 de `docs/validation_requirements.md`. Guardar evidencias y manifiestos PASS/FAIL. **No avanzar** con pruebas pendientes, omitidas o fallidas.
6. Si V01-V14 pasan, congelar hash de pipeline y documentar la aprobación. Si fallan, corregir SOLO el componente responsable y repetir las verificaciones dependientes.
7. Completar el entrenamiento reutilizable en `src/tt_ecog/` con notebooks como orquestadores, y medir factibilidad en la GPU efectivamente asignada por Colab Pro.
8. Ejecutar análisis de convergencia para **decidir después** `search_epochs`, `final_max_epochs` y la política final de parada; no fijar cifras sin evidencia.
9. Solo después de aprobar pasos anteriores, implementar y ejecutar etapa 1 TPE, etapa 2 TPE y evaluación final A/B/C.

## Reglas metodologicas no negociables

- Tres sujetos independientes: S1, S2, S3; cinco salidas por muestra; espacio arquitectonico 27 combinaciones.
- ECoG 40-200 Hz; notch 60/120/180 Hz; Morlet 40 frecuencias, 7 ciclos, `zero_mean=True`; representacion a 100 Hz; retardo objetivo de 200 ms en la dirección ECoG(t) -> flexión(t+0.2 s).
- QC se detecta y documenta durante EDA; limpieza se aplica antes de referencia, filtros y wavelets, mediante decisiones aprobadas. No codificar excepciones por sujeto ni canales descubiertos a partir de Pearson.
- QC descriptivo global sobre datos de **desarrollo** puede fijar decisiones de adquisición; no presentarlo como un QC totalmente anidado. Las estadísticas adaptativas de transformaciones se ajustan exclusivamente con entrenamiento de cada fold.
- No acceder a `test_dg` o archivos `*_testlabels.mat` en EDA, QC, preparacion de folds, piloto, búsqueda o selección de épocas. El test es de uso exclusivo de `07_evaluacion_final.ipynb`, tras congelacion de decisiones.
- Dividir antes de cualquier operacion que aprenda estadísticas o use contexto temporal a traves de particiones. Procesar por segmentos cronológicos independientes con guardas; validar alcance de FIR, Morlet y alineación con tests de perturbación.
- Aplicar RobustScaler `(10, 90)` y MinMaxScaler ajustados con fold-train; validación y prueba solo `transform`.
- U-Net 1D con BatchNorm1d, GELU, dropout 0.3, interpolación lineal + Conv1d, skips concatenativos; `forward` conserva longitud 256 y entrega cinco dedos.
- Entrenar con AdamW sin scheduler; loss 0.5 MSE + 0.5(1-coseno temporal). Optuna maximiza `val_pearson_full` reconstruido con overlap-add.
- Durante Optuna, todos los entrenamientos usan la duración fija que se apruebe más adelante y puntuan la **ultima época**, sin early stopping sobre el mismo fold puntuado. Guardar curvas por época para diagnóstico sin escoger el mejor pico a posteriori.
- Selección final de época con validación interna de desarrollo; test solo una vez decidido todo. CCC, MSE y RMSE son métricas secundarias, NO objetivos multiobjetivo en la implementación principal.
- Etapa 1: 20 trials x 2 folds (B2, B4) por sujeto, 2 hiperparametros. Etapa 2: 10 trials x 5 folds por sujeto, 3 hiperparametros; incluye arquitectura de referencia dentro de las 10 propuestas. 270 entrenamientos de búsqueda en total, sin incluir pilotos ni entrenamientos finales.

## Reglas de código y seguridad

- Python modular y tipado cuando sea práctico. Una sola implementación de transformaciones, perdida y métricas; notebooks ligeros.
- Configuración única por YAML y parámetros/paths relativos; prohibido repetir bloques de configuración escritos a mano en notebooks.
- No incluir tokens, secretos, rutas privadas ni claves W&B en código/documentación. No copiar sin inspección scripts heredados al repositorio.
- No guardar datos ECoG crudos, etiquetas de prueba, checkpoints ni estudios Optuna grandes en Git. Usar Google Drive o almacenamiento persistente externo y guardar hashes en manifiestos.
- No reinstalar PyTorch o CUDA a ciegas en Colab; comprobar compatibilidad GPU. Registrar versiones, GPU, memoria, seed y commit.
- No afirmar que un smoke test equivale a un entrenamiento completo, ni que CV purgado equivale a predicción causal en tiempo real.
- Los fallos no se silencian. Fallo de Pearson indefinido, NaN, cobertura incompleta, ventanas invalidas o mezcla entre conjuntos deben producir diagnóstico y FAIL.
- Mostrar plan y resultado por etapa a la persona usuaria; solicitar autorización antes de entrenamientos costosos o decisiones que cambien la metodología.

## Definition of Done por etapa

- EDA: tres sujetos descritos; `qc_decisions.json` valido y aprobado.
- Preprocesamiento: arrays y metadatos trazables; decisiones QC aplicadas; labels de test sin tocar.
- Paso 3: V01-V14 PASS real (no mocks que simulen exito), reporte, hashes y evidencia persistidos. De no haber archivos `.mat`, informar BLOQUEADO, no PASS.
- Factibilidad: GPU real, memoria peak en entrenamiento, batch común y costos documentados.
- Optuna: estudios persistentes recuperables con folds, semillas y parámetros registrados.
- Evaluación: tres variantes A/B/C, por sujeto y dedo; métrica Pearson principal, CCC y errores secundarios, misma cobertura y criterios.

Para iniciar: leer `docs/CODEX_START.md`, `docs/protocolo_implementacion.md` y `docs/validation_requirements.md`, confirmar la estructura del dataset sin inspeccionar las etiquetas de test y desarrollar por fases. No lanzar Optuna de forma automática.
