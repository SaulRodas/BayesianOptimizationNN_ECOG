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
  - fábricas de escaladores y su reconstrucción desde parámetros persistidos;
  - soporte temporal analítico y medido, y máscara exacta de columnas válidas (`feature_valid_mask`);
  - cadenas por segmento: `segment_features` (ECoG completa) y `segment_glove` (solo los nodos del guante dentro del segmento).
  La usan el EDA (en memoria) y el paso 2 (por bloque).
- `descriptive`: estadísticos descriptivos (z robusto, STD por ventanas, Welch, fracción de línea).
- `splits`:
  - `chronological_blocks`: bloques B1–B5.
  - `make_folds`: un bloque de validación y los demás como segmentos de train.
  - `train_eligibility`: aplica la purga. Antes de validación exige `y ≤ inicio_val − 1 − P`; después, `x ≥ fin_val + P`. Una purga `None` es un error.
- `datasets`: solo numpy.
  - Tramos contiguos y conteo de ventanas.
  - Ventanas de entrenamiento con stride 1 y de inferencia con stride 64 más una ventana anclada al final, con cobertura comprobada.
  - `load_fold`: verifica hashes, aplica los escaladores del fold y rechaza folds con purga candidata salvo `allow_candidate=True`.
- `store`: identidad por hash canónico y layout de los datos procesados. El manifiesto se escribe al final como marca de commit; una carpeta existente solo se reutiliza si verifica y nunca se sobrescribe.
- `prep`: orquestación del paso 2 (`start_run`, `load_subjects`, `steps.p01_gate`/`p02_blocks`/`p03_folds`, `finish_run`; CLI `python -m tt_ecog.prep`).
  - `gate` valida la corrida EDA de referencia.
  - `blocks` procesa cada bloque y escribe los manifiestos.
  - `folds` crea los escaladores y ventanas por fold y falla si un rango 10–90 es nulo o un dedo es constante.
- `eda`: orquestación (`start_run`, `load_subjects`, `finish_run`, `run_eda`, CLI `python -m tt_ecog.eda`), secciones `eda/sections/s01…s11` y `show()` para el notebook. `eda/comparators.py` contiene alternativas solo de contraste (CAR, submuestreo directo, interpolación lineal) que nunca entran al pipeline.
- `reporting`: entorno (versiones, commit, remoto sin credenciales), hashes y JSON.

Pendiente: dataset de PyTorch, overlap-add y métricas, U-Net, entrenamiento y Optuna.
