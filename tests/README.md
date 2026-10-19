# Pruebas

Instalar `python -m pip install -e ".[test,dev,model,search]"` y ejecutar `python -m pytest -q`.

Las pruebas usan desarrollo sintético, incluyendo ejecución de las celdas portables de los notebooks 01–03. Verifican exclusión, filtros, aislamiento, alineación, escaladores, ventanas, identidad compartida, persistencia, métricas y 27 arquitecturas. El EDA no se crea como requisito del fixture de preparación.

El 04 se ejecuta también con un fold sintético diminuto en CPU: sus actualizaciones, métricas,
curvas y checkpoint son reales, pero no equivalen a entrenar con el dataset ni a medir una GPU.
Las pruebas de aprobación usan evidencia fabricada únicamente para comprobar rechazo/aceptación
del lector; nunca se presenta como V01–V14 real.

Optuna se prueba con estudios reales sobre objetivos sintéticos: presupuesto de 270 fits,
transferencia de parámetros por sujeto, referencia dentro del presupuesto, sampler recuperado,
última época, fallos y corrupción. Las celdas de 05/06 también entrenan una U-Net con folds
sintéticos diminutos en CPU y recuperan una interrupción entre épocas; se excluyen únicamente
las celdas de admisión de datos reales/GPU y montaje de Colab. Esto no autoriza búsquedas reales.
El 07 prueba selección de época, refit independiente por bloques, recuperación, congelación,
lectura restringida de test sintético, inferencia y tablas/figuras con las mismas celdas del notebook.
No abre etiquetas reales ni presenta la admisión sintética como aprobación científica.
El comando general `pytest` incluye estas pruebas. El 03 excluye `test_optimization.py` y
`test_final_evaluation.py` por
corresponder a una etapa posterior sin marcas V01–V14: no requiere instalar Optuna para verificar
el pipeline. Ninguna prueba científica V01–V14 se omite por esta separación.

`@pytest.mark.verification("Vxx")` relaciona una prueba con la verificación correspondiente. `--verification-json ruta.json` guarda los resultados reales de pytest; omisiones o fallos no cuentan como PASS. Los tests de combinación de estados son tests del reporte, no evidencia de aprobación experimental.

En Windows puede usarse `--basetemp results/pruebas` cuando la carpeta temporal del sistema no sea accesible. No usar una carpeta que contenga datos: pytest administra esa ruta. La corrida de V01–V14 sobre los tres sujetos reales sigue siendo necesaria.
