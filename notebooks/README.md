# Notebooks

Previstos: `01_exploracion_ecog.ipynb`, `02_preprocesamiento_ecog.ipynb`, `03_verificacion_pipeline.ipynb`, `04_factibilidad_entrenamiento.ipynb`, `05_optimizacion_etapa1.ipynb`, `06_optimizacion_etapa2.ipynb`, `07_evaluacion_final.ipynb`.

Implementado: `01_exploracion_ecog.ipynb`, el EDA que justifica el preprocesamiento.
- Se abre en VS Code con el kernel **Colab**. La primera celda hace `git pull --ff-only` del repositorio en el runtime, porque Colab no ve los archivos locales: haz push antes de ejecutar.
- Las celdas solo llaman a `tt_ecog.eda`. Cada figura se guarda en `MyDrive/TT_ECoG/results/eda/<run_id>/fig/` y se muestra en el notebook.
- Las salidas quedan en tu `.ipynb` local al guardarlo. El filtro `nbstripout` (`.gitattributes`) evita commitearlas.

Los notebooks 02–07 se implementarán secuencialmente. No hay notebooks vacíos que aparenten funcionalidad.
