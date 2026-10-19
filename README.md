# TT ECoG — EDA de justificación y preprocesamiento reproducible

Decodificación continua de la flexión de cinco dedos a partir de ECoG (BCI Competition IV, Dataset 4) con U-Net 1D y optimización TPE de Optuna en dos etapas.

**Estado (v0.5.0):**
- Implementados:
  - lectores y primitivas de preprocesamiento compartidas (`tt_ecog.preprocessing`);
  - EDA de justificación en once secciones;
  - preprocesamiento del paso 2 por bloque, con folds y escaladores (`tt_ecog.prep`, notebook 02);
  - U-Net 1D parametrizable, pérdida, métricas y overlap-add;
  - verificación V01–V14 del paso 3 (`tt_ecog.verification`, notebook 03);
  - pruebas sintéticas.
- Corridas reales:
  - EDA `20261019T004831Z_3a68db`, `COMPLETE`, fijada en `channels.reference_run`;
  - paso 2 `20261019T022241Z_90bd22`, `COMPLETE`, con purga candidata.
- Pendientes:
  - ejecutar el notebook 03 en Colab;
  - aprobar la purga, rehacer los folds y verificar de nuevo;
  - aprobar la congelación;
  - factibilidad, épocas, entrenamiento y Optuna.

Flujo: **EDA → Preprocesamiento**. No hay etapa de QC con umbrales ni aprobación (eliminada en el protocolo v1.1). La exclusión de canales es fija: **S1-54, S2-20, S2-37** (ver `docs/channel_exclusion.md`).

## Archivos principales

| Ruta | Uso |
|---|---|
| `AGENTS.md` | Reglas obligatorias para agentes y secuencia de trabajo. |
| `docs/protocolo_implementacion.md` | Fuente de verdad metodológica (v1.1). |
| `docs/dataset_spec.md` | Contrato de archivos `.mat`, claves, dimensiones y política de test. |
| `docs/validation_requirements.md` | Pruebas V01-V14 y puerta de aprobación del pipeline. |
| `docs/channel_exclusion.md` | Evidencia y razones de la exclusión fija de canales. |
| `configs/experiment_config.yaml` | Configuración central: exclusión de canales, señal, EDA y decisiones pendientes en `null`. |
| `docs/experimental_log.md` | Bitácora experimental y control de cambios. |
| `src/tt_ecog/preprocessing.py` | Única implementación de las transformaciones (EDA y paso 2). |
| `src/tt_ecog/eda/` | Secciones del EDA, gráficas y orquestación. |
| `src/tt_ecog/prep/`, `splits.py`, `datasets.py`, `store.py` | Paso 2: bloques procesados, purga, folds, escaladores, ventanas y manifiestos. |
| `notebooks/01_exploracion_ecog.ipynb` | Notebook ligero que llama a `tt_ecog.eda`. |
| `notebooks/02_preprocesamiento_ecog.ipynb` | Notebook ligero que llama a `tt_ecog.prep`. |
| `src/tt_ecog/model.py`, `losses.py`, `metrics.py`, `validation.py` | U-Net 1D (27 arquitecturas), pérdida 0.5 MSE + 0.5(1 − coseno), Pearson/CCC/errores y overlap-add. |
| `src/tt_ecog/verification/` | Paso 3: pruebas V01–V14 con datos reales y combinación con las pruebas sintéticas. |
| `notebooks/03_verificacion_pipeline.ipynb` | Notebook ligero que llama a `tt_ecog.verification`. |

## Flujo de trabajo: VS Code local + kernel Colab

1. **Una vez por clon local:** `python -m pip install -e ".[test,dev,model]"` y `nbstripout --install --attributes .gitattributes`. El extra `model` instala PyTorch; localmente basta la versión CPU. Así las salidas del notebook se quedan en tu archivo local pero no se commitean.
2. Mantener `sub1_comp.mat`, `sub2_comp.mat` y `sub3_comp.mat` en `MyDrive/TT_ECoG/raw`. Las etiquetas de prueba siguen fuera de ese árbol y su ruta no se resuelve.
3. Hacer commit y push del código. Colab no ve los archivos locales: el código llega por Git.
4. Abrir `notebooks/01_exploracion_ecog.ipynb` en VS Code con el kernel **Colab** y ejecutar las celdas:
   - La primera hace `git pull --ff-only` y muestra el commit, que debe coincidir con tu `git rev-parse HEAD` local.
   - Monta Drive y crea `MyDrive/TT_ECoG/results/eda/<run_id>/`.
5. Cada sección guarda sus figuras (`fig/*.png`) y tablas (`tables/`) en Drive y las muestra en el notebook. Al guardar el notebook local, las gráficas quedan en él.
6. La última celda escribe `run_summary.json`, `index.md` y `artifact_manifest.json`, y falla si alguna sección no terminó en `COMPLETE`.

El notebook `02_preprocesamiento_ecog.ipynb` sigue el mismo flujo:

1. Valida la corrida EDA de `channels.reference_run`: estado `COMPLETE`, hashes de los `.mat`, exclusiones, señal y SHA-256 de sus artefactos.
2. Procesa cada bloque B1–B5 por separado en `MyDrive/TT_ECoG/data/processed/<pipeline_hash>/`.
3. Construye los folds en `folds/purge_<P>_<hash>/`.
4. Deja su evidencia en `MyDrive/TT_ECoG/results/preprocessing/<run_id>/`.

Si los datos procesados ya existen y su manifiesto verifica, se reutilizan; nunca se sobrescriben.

El notebook `03_verificacion_pipeline.ipynb` hace lo siguiente:

1. Localiza los datos procesados que corresponden a este código y configuración. Si no existen, se detiene y pide ejecutar antes el notebook 02.
2. Ejecuta la batería sintética de `tests/`.
3. Ejecuta las 14 pruebas con los datos reales.
4. Escribe `MyDrive/TT_ECoG/results/verification/<run_id>/` con `verification_report.json`, `verification_summary.md`, `test_results.xml` y `filter_support.json`.

Cada `Vxx` es `PASS` solo si pasan su parte real y todas sus pruebas sintéticas. Con todo en `PASS` emite `approved_pipeline_manifest.json` como **candidato**; la congelación requiere aprobación explícita.

Ejecución equivalente por línea de comandos (por ejemplo, con Drive montado o una carpeta local que contenga `TT_ECoG/raw`):

```bash
python -m tt_ecog.eda --config configs/experiment_config.yaml --drive-root /content/drive/MyDrive
python -m tt_ecog.prep --config configs/experiment_config.yaml --drive-root /content/drive/MyDrive
python -m tt_ecog.verification --config configs/experiment_config.yaml --drive-root /content/drive/MyDrive
```

Pruebas sintéticas locales:

```bash
python -m pytest -q
```

## Estados de una corrida del EDA

- `COMPLETE`: las once secciones terminaron y existe la figura de evidencia de cada canal excluido. La CLI devuelve 0.
- `BLOCKED`: falta Drive o algún `.mat` de desarrollo. La CLI devuelve 2.
- `FAIL`: contrato de datos inválido o una sección falló; el traceback queda en `errors/`. La CLI devuelve 1. Un error al escribir resultados es `FAIL`, nunca `BLOCKED`.

El paso 2 usa los mismos estados. `BLOCKED` también cubre una compuerta del EDA no superada: `reference_run` es `null` o la corrida no valida, y el mensaje lista las corridas válidas. Un directorio procesado corrupto es `FAIL`.

## Decisiones fijas y pendientes

- **Fijas en el YAML:**
  - Banda 40–200 Hz y notch 60/120/180 Hz, con FIR Hamming de fase cero.
  - CMR instantánea de los canales conservados.
  - Morlet de 40 frecuencias, 7 ciclos y `zero_mean`.
  - Antialias Kaiser y decimación a 100 Hz.
  - Guante: 25 Hz e interpolación con spline cúbico.
  - Retardo de 200 ms en la dirección ECoG(t) → flexión(t+0.2 s).
  - RobustScaler(10, 90) y MinMax ajustados por fold.
- **Pendientes en `null`:** `search_epochs`, `final_max_epochs`, `batch_size` y `purge_samples`.
  - La sección s11 del EDA mide el soporte temporal de la cadena como insumo para fijar la purga en el paso 3.
  - Mientras la purga sea `null`, los folds del paso 2 usan el candidato de verificación (256) y quedan etiquetados `candidate_pending_V04_V09`. `datasets.load_fold` los rechaza salvo con `allow_candidate=True`, que queda reservado al paso 3.
  - La purga se fija en dos rondas:
    1. La primera verificación deja V04/V09 en `BLOCKED` y reporta la evidencia, incluida la purga mínima que aísla.
    2. Con el valor aprobado en el YAML se rehacen los folds (notebook 02) y se verifica de nuevo.
- **U-Net (decisiones de la persona usuaria, 2026-10-18):**
  - Reducción canal×frecuencia con Conv1d de kernel 3.
  - Encoder: Conv1d dilatada → BN → GELU → Dropout → MaxPool1d(2), con el skip antes de reducir.
  - Decoder: interpolación lineal al largo del skip → Conv1d (kernel espejo, sin dilatación) → BN → GELU → Dropout → concatenación del skip.
  - Salida: Conv1d de kernel 1 a cinco dedos.
- **Procesamiento por bloque:** cada bloque se procesa solo con sus datos.
  - Se pierden 125 columnas por borde por contexto incompleto y 20 al final por el retardo.
  - Ninguna ventana cruza fronteras entre bloques.
  - La purga solo retira columnas de train junto al bloque de validación, así que cambiarla no obliga a reprocesar.
- **Objetivo de la búsqueda:** monoobjetivo, maximizar Pearson. CCC, MSE y RMSE se informan como métricas adicionales.

## Almacenamiento y privacidad experimental

No versionar archivos `.mat`, etiquetas de prueba, checkpoints, estudios completos, salidas de notebooks, identificadores personales ni secretos. Guardar las salidas persistentes en Drive, con hash y versión registrados. Ver `.gitignore`, `.gitattributes` y `AGENTS.md`.

## Alcance de resultados

- La validación cruzada es por bloques temporales con purga; no implica que la red sea causal.
- Las gráficas del EDA que usan `train_dg` de desarrollo son confirmatorias y no ajustan parámetros.
- El test se usa solo al final y no se consulta para ajustar el procesamiento ni los hiperparámetros.
