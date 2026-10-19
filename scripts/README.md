# Preparación del entorno EDA

`python scripts/check_eda_environment.py` informa qué dependencias de `requirements-eda.txt` están instaladas, cuáles faltan y cuáles son incompatibles. `requirements-eda.txt` incluye MNE y scikit-learn.

`--install-missing` instala únicamente las que faltan. Si una versión existente es incompatible, el script se detiene para que lo resuelvas explícitamente. No instala PyTorch/CUDA ni las dependencias de etapas futuras.

El EDA se ejecuta con `python -m tt_ecog.eda --config configs/experiment_config.yaml --drive-root <MyDrive>`. El notebook usa exactamente el mismo orquestador.
