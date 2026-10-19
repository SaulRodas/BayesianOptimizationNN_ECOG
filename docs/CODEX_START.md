# Inicio de trabajo — TT ECoG

Leer `AGENTS.md`, el protocolo v1.2, los requisitos de verificación, el contrato del dataset y el YAML. El README principal describe comandos y responsabilidades.

## Estado de implementación

EDA opcional, preprocesamiento, arquitectura mínima y verificación implementados. Reorganización v0.6.0: celdas explícitas, persistencia esquema 2 e identidad compartida. Las corridas reales históricas están documentadas en la bitácora; no certifican el formato nuevo ni sustituyen V01–V14 real.

## Flujo pendiente con datos reales

1. Ejecutar 02 en Colab desde datos crudos; no necesita 01.
2. Copiar `pipeline_id` a `paths.processed_pipeline_id`, hacer commit/push y ejecutar 03.
3. Mantener V04/V09 BLOCKED mientras la purga siga pendiente. Aprobar después con su evidencia; si el valor coincide con el candidato, repetir 03 sobre los mismos datos. Si cambia, generar y seleccionar otro pipeline con 02.
4. Congelar solo después de V01–V14 PASS real y aprobación documentada.
5. Ejecutar `04_entrenamiento.ipynb`: recursos y entrenamiento en el mismo notebook, con
   `training.search_epochs_candidate: 50` como propuesta de piloto. Elegir el batch común
   con la memoria medida. Revisar las curvas antes de fijar las épocas de búsqueda.

6. Registrar `training.batch_size` y `training.search_epochs` tras la decisión basada en 04.
7. Ejecutar 05 para learning rate/weight decay; después 06 para arquitectura. Los estudios son
   independientes por sujeto, con recuperación de trials y checkpoints. Usar un único runtime
   escritor por estudio; los fallos detienen la búsqueda sin añadir propuestas.

8. Decidir máximo final, paciencia y min_delta usando desarrollo. Ejecutar 07: seleccionar
   duración con B5 (autorizado), reentrenar A/B/C con desarrollo completo, congelar y evaluar test.
   Registrar `paths.final_run_id` para recuperar la misma corrida, y resolver `test_labels_dir`
   únicamente en la celda final de acceso a test.

Los notebooks 02–07 están implementados. Su ejecución experimental sigue pendiente.
La propuesta de 50 épocas no sustituye los valores aprobados.
La aprobación se selecciona con los campos existentes `verification.status: approved` y
`verification.approved_manifest_file`, relativo a `TT_ECoG`. El 04 comprueba el reporte PASS,
su correspondencia con el pipeline y los componentes congelados. El candidato emitido por 03
no activa entrenamiento mientras el YAML siga en `pending`.

No fijar épocas, batch ni otros `null` sin evidencia. No ejecutar entrenamientos, Optuna ni evaluación final automáticamente. El EDA puede ejecutarse aparte para revisar visualizaciones, sin habilitar ni bloquear este flujo.
