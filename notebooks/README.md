# Notebooks

Previstos: `01_exploracion_ecog.ipynb`, `02_preprocesamiento_ecog.ipynb`, `03_verificacion_pipeline.ipynb`, `04_factibilidad_entrenamiento.ipynb`, `05_optimizacion_etapa1.ipynb`, `06_optimizacion_etapa2.ipynb`, `07_evaluacion_final.ipynb`.

Implementados:
- `01_exploracion_ecog.ipynb`: el EDA que justifica el preprocesamiento.
  - Las celdas solo llaman a `tt_ecog.eda`.
  - Cada figura se guarda en `MyDrive/TT_ECoG/results/eda/<run_id>/fig/` y se muestra en el notebook.
- `02_preprocesamiento_ecog.ipynb`: el paso 2. Las celdas solo llaman a `tt_ecog.prep`.
  - **Compuerta del EDA:** valida `channels.reference_run`.
  - **Bloques:** procesa cada uno por separado en `MyDrive/TT_ECoG/data/processed/<pipeline_hash>/`.
  - **Folds:** construye escaladores y ventanas en `folds/purge_<P>_<hash>/`.
  - **Evidencia:** queda en `MyDrive/TT_ECoG/results/preprocessing/<run_id>/`.
- `03_verificacion_pipeline.ipynb`: el paso 3. Las celdas solo llaman a `tt_ecog.verification`.
  - **Preparación:** localiza los datos procesados de este código y configuración.
  - **Pruebas sintéticas:** ejecuta la batería de `tests/`.
  - **V01–V14:** corre cada prueba con los datos reales.
  - **Reporte:** escribe `MyDrive/TT_ECoG/results/verification/<run_id>/`. No congela nada.

Todos se abren en VS Code con el kernel **Colab**:
- La primera celda hace `git pull --ff-only` del repositorio en el runtime, porque Colab no ve los archivos locales: haz push antes de ejecutar.
- Las salidas quedan en tu `.ipynb` local al guardarlo. El filtro `nbstripout` (`.gitattributes`) evita commitearlas.

Los notebooks 04–07 se implementarán secuencialmente. No hay notebooks vacíos que aparenten funcionalidad.
