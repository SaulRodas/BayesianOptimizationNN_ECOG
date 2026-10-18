# Primera etapa implementada

El paquete `tt_ecog` contiene una sola implementación de EDA/QC:

- `config.load_config(path)`: valida configuración EDA sin completar campos pendientes. Las rutas se resuelven respecto a MyDrive/`persistent_storage_dir`.
- `io.read_ecog(config, drive_root, subject)`: carga solo `train_data`; devuelve un `DevelopmentRecord` con array readonly `(C,T)` y metadatos/hash.
- `io.validate_development_labels(config, drive_root, record)`: comprueba `train_dg` separadamente; devuelve metadatos, nunca etiquetas a QC.
- `quality_control.analyze_ecog(ecog, subject, fs, qc_config, signal_config)`: estadísticas, PSD y candidatos. No modifica datos ni recibe etiquetas.
- `eda.run_eda(config_path, drive_root)`: procesa S1/S2/S3 y devuelve una carpeta nueva de evidencias.
- `decisions.validate_decisions(...)`: valida esquema, límites, procedencia y evidencia; no aprueba decisiones.

No se implementan todavía limpieza, preprocesamiento, folds, U-Net, entrenamiento ni Optuna. Los arrays HDF5 solo admiten la disposición MATLAB v7.3 con dimensiones invertidas y atributo `MATLAB_class`. Otras disposiciones requieren un adaptador explícito.
