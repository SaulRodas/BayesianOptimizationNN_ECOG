# TT ECoG — EDA/QC reproducible

Repositorio de arranque para implementar desde cero un sistema de decodificación continua de la flexión de cinco dedos a partir de ECoG (BCI Competition IV, Dataset 4), con U-Net 1D y optimización TPE de Optuna en dos etapas.

**Primera entrega implementada: cargadores, EDA/QC, notebook Colab y pruebas sintéticas.** El EDA real y la aprobación humana de QC siguen pendientes. No se han implementado preprocesamiento, modelo, entrenamiento ni Optuna.

## Archivos principales

| Ruta | Uso |
|---|---|
| `AGENTS.md` | Reglas obligatorias de Codex y secuencia de trabajo. |
| `docs/CODEX_START.md` | Encargo de implementación por etapas, listo para pegar en Codex. |
| `docs/protocolo_implementacion.md` | Fuente de verdad metodológica, redactada como implementación desde cero. |
| `docs/dataset_spec.md` | Contrato de archivos `.mat`, claves, dimensiones y política de test. |
| `docs/validation_requirements.md` | Pruebas V01-V14 y puerta de aprobación del pipeline. |
| `configs/experiment_config.yaml` | Configuración central con decisiones cerradas y pendientes `null`. |
| `docs/quality_decisions.md` | Plantilla de aprobación de calidad por sujeto. |
| `configs/qc_decisions.template.json`, `schemas/qc_decisions.schema.json` | Contrato JSON de decisiones de QC sin exclusiones precargadas. |
| `docs/experimental_log.md` | Bitácora experimental y control de cambios. |
| `docs/referencias.md` | Fuentes públicas y documentación técnica. |
| `requirements.txt` | Dependencias sugeridas, sujetas a validación del entorno Colab. |

El paquete `src/tt_ecog/` contiene la implementación reutilizable y el notebook `notebooks/01_exploracion_ecog.ipynb` la orquesta. Aún no hay código del modelo.

## Preparacion

1. Mantener los originales `sub1_comp.mat`, `sub2_comp.mat` y `sub3_comp.mat` en `MyDrive/TT_ECoG/raw`. Las etiquetas de prueba permanecen separadas; su ruta sigue sin resolver.
2. Publicar el código revisado en la rama `main` del remoto configurado. No publicar datos ni resultados. El notebook necesita que esa rama exista antes de clonarla.
3. Abrir `notebooks/01_exploracion_ecog.ipynb` en VS Code y seleccionar el kernel Colab. Ejecutar sus celdas de código remoto, comprobación de dependencias y montaje de Drive.
4. Ejecutar EDA y revisar `TT_ECoG/results/eda_qc/<run_id>/`. El notebook usa `requirements-eda.txt`; no instala PyTorch/CUDA ni solicita GPU.
5. Revisar los diagnósticos de los tres sujetos. Registrar candidatos de umbral en el YAML y repetir. `null` produce detectores pendientes, no valores por defecto.
6. Revisar y aprobar decisiones concretas antes de iniciar preprocesamiento. Las verificaciones V01–V14 y todas las etapas posteriores siguen pendientes.

Para trabajar localmente con datos sintéticos:

```bash
python -m pip install -e ".[test]"
python -m pytest -q
```

Para ejecutar EDA con Drive montado:

```bash
python scripts/check_eda_environment.py --install-missing
python -m pip install --no-deps -e .
python -m tt_ecog.eda --config configs/experiment_config.yaml --drive-root /content/drive/MyDrive
```

El argumento `--drive-root` solo identifica el montaje. Las rutas y parámetros operativos están en el YAML. En un entorno local puede apuntar a un directorio equivalente que contenga `TT_ECoG/raw`. `requirements.txt` conserva las dependencias de etapas futuras; no es la instalación recomendada para EDA.

## Resultados y estados

Cada ejecución genera CSV de canales, eventos, resumen y evolución temporal; 5 figuras por sujeto; propuesta `qc_decisions.json`; configuración efectiva; metadatos de archivos; versiones; y manifiestos SHA-256. Una carpeta nueva evita sobrescribir evidencia previa. Los CSV de eventos pueden estar vacíos cuando los detectores están pendientes: eso no demuestra ausencia de artefactos.

- `PENDING_THRESHOLDS`: descripción completada, faltan candidatos explícitos para uno o más detectores.
- `PENDING_REVIEW`: detecciones realizadas, falta revisión y aprobación humana.
- `BLOCKED`: faltan archivos o almacenamiento. CLI devuelve código 2.
- `FAIL`: contrato de datos, cálculo o configuración inválido. CLI devuelve código 1.

Una ejecución descriptiva completa devuelve código 0, pero **nunca aprueba QC**. Ningún resultado de esta etapa declara V01–V14 PASS. Las etiquetas de desarrollo solo se validan en un lector separado; no alimentan los detectores.

El EDA trata unidades y tasas como nominales hasta comprobar la fuente real. La recuperación del guante efectivo y su interpolación pertenecen a la etapa siguiente. Los umbrales `spectral_z` y `spectral_min_duration_s` añadidos también permanecen pendientes. Las ventanas descriptivas, Welch, agrupación de eventos y tolerancia numérica se registran explícitamente en `quality_control.analysis`; no autorizan limpieza.

## Decisiones congelables y pendientes

La validación técnica congela QC, preprocesamiento, alineación, particiones, métricas y arquitectura parametrizable. Los valores `search_epochs`, `final_max_epochs`, `batch_size` y `purge_samples` se mantienen pendientes hasta la etapa correspondiente. La duración no debe fijarse anticipadamente. La búsqueda principal sigue siendo **monoobjetivo: maximizar Pearson**; CCC, MSE y RMSE se informan como métricas adicionales.

## Almacenamiento y privacidad experimental

No versionar archivos `.mat`, etiquetas de prueba, checkpoints, estudios completos, identificadores personales ni secretos. Guardar salidas persistentes fuera de la sesión efimera de Colab, con hash y versión registrada. Ver `.gitignore` y `AGENTS.md`.

## Alcance de resultados

La validación cruzada es por bloques temporales con purga; no implica que la red sea causal. El test se utiliza al final y no se consulta para ajustar el procesamiento ni los hiperparametros. El esquema oficial de la competencia y el promedio de Pearson de cinco dedos deben distinguirse cuando se presenten comparativas historicas.
