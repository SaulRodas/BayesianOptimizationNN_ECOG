# Pruebas de la primera entrega

Instalar el paquete y las herramientas: `python -m pip install -e ".[test]"`. Ejecutar `python -m pytest -q`.

La batería usa exclusivamente datos sintéticos generados en directorios temporales. Comprueba lectores MATLAB clásico/HDF5, aislamiento de prueba, contratos de dimensiones, finitud, hashes, configuración, detectores, invariancia respecto a etiquetas y generación completa de reportes para tres sujetos artificiales. Incluye validación estructural y ausencia de salidas guardadas en el notebook.

La prueba de integración genera figuras y evidencia reales sobre señales artificiales; no simula un resultado aprobado. Datos ausentes producen `BLOCKED`, integridad incorrecta produce `FAIL`, y el QC generado sigue `pending`.

En Windows con restricciones sobre la carpeta temporal, usar una carpeta nueva dentro de `results`, por ejemplo `python -m pytest -q --basetemp=results/test_tmp_01 --junitxml=results/implementation_checks/test_results.xml`. Pytest elimina el contenido de `--basetemp` si ya existe: reservarla exclusivamente para fixtures desechables; nunca usar una carpeta de datos reales o reportes que deban conservarse.

Estas pruebas no constituyen aprobación V01–V14. La verificación integral del paso 3 permanece pendiente, incluyendo datos reales de S1/S2/S3, preprocesamiento, aislamiento temporal y arquitectura.
