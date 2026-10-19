# Paquete `tt_ecog`

- `config.load_config(path)`: valida la configuración sin rellenar campos pendientes; expone `excluded_channels(subject)` y `channel_exclusions(subject)`. Las rutas se resuelven respecto a MyDrive/`persistent_storage_dir`.
- `io.read_ecog(...)` / `io.read_development_glove(...)`: leen solo `train_data` y `train_dg` (solo lectura, hash antes y después). Nunca leen variables ni archivos de test. `MissingDevelopmentData` y `StorageUnavailable` señalan `BLOCKED`.
- `preprocessing`: **única implementación** de las transformaciones, por segmento y sin mutar entradas. Incluye:
  - exclusión de canales con mapa original→efectivo;
  - CMR instantánea;
  - FIR pasabanda + notch de fase cero;
  - Morlet de MNE;
  - antialias Kaiser y decimación;
  - inspección y recuperación del guante a 25 Hz e interpolación con spline cúbico;
  - retardo;
  - fábricas de escaladores;
  - soporte temporal.
  La usan el EDA (en memoria) y el notebook 02.
- `descriptive`: estadísticos descriptivos (z robusto, STD por ventanas, Welch, fracción de línea).
- `splits.chronological_blocks`: bloques B1–B5. Folds y purga llegan en el paso 2.
- `eda`: orquestación (`start_run`, `load_subjects`, `finish_run`, `run_eda`, CLI `python -m tt_ecog.eda`), secciones `eda/sections/s01…s11` y `show()` para el notebook. `eda/comparators.py` contiene alternativas solo de contraste (CAR, submuestreo directo, interpolación lineal) que nunca entran al pipeline.
- `reporting`: entorno (versiones, commit, remoto sin credenciales), hashes y JSON.

Pendiente: `datasets.py`, folds y escaladores ajustados por fold, U-Net, entrenamiento y Optuna.
