# Preparación del entorno EDA

`python scripts/check_eda_environment.py` informa las dependencias instaladas, ausentes e incompatibles usando `requirements-eda.txt`.

`--install-missing` permite instalar únicamente las ausentes. Si una versión existente es incompatible, detenerse y resolverla explícitamente. El script no instala PyTorch/CUDA ni el conjunto de dependencias de etapas futuras.

El EDA se ejecuta mediante `python -m tt_ecog.eda --config configs/experiment_config.yaml --drive-root <MyDrive>`. El notebook utiliza exactamente el mismo orquestador.
