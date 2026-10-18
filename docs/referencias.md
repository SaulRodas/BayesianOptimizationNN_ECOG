# Referencias y documentación técnica

Estas fuentes son material de consulta científica y técnica. La implementación concreta se rige por `docs/protocolo_implementacion.md`.

## Dataset, benchmark y artículos relacionados

1. **BCI Competition IV — sitio oficial.** Descripción de dataset 4 (finger movements ECoG, tres sujetos, 1000 Hz, cinco dedos). https://www.bbci.de/competition/iv/
2. **Descarga oficial del Dataset 4.** https://www.bbci.de/competition/iv/download/
3. **Tangermann, M. et al. (2012).** Review of the BCI Competition IV. *Frontiers in Neuroscience*, 6:55. https://doi.org/10.3389/fnins.2012.00055
4. **Liang, N. y Bougrain, L. (2012).** Decoding Finger Flexión from Band-Specific ECoG Signals in Humans. *Frontiers in Neuroscience*, 6:91. https://doi.org/10.3389/fnins.2012.00091
5. **Lomtev, V., Kovalev, A. y Timchenko, A. (2022).** FingerFlex: Inferring Finger Trajectories from ECoG signals. arXiv:2211.01960. https://arxiv.org/abs/2211.01960
6. **FingerFlex — repositorio público de referencia.** https://github.com/Irautak/FingerFlex

## Herramientas y métodos

7. **MNE-Python, filtrado y Morlet.** https://mne.tools/stable/auto_tutorials/preprocessing/30_filtering_resampling.html y https://mne.tools/stable/generated/mne.time_frequency.tfr_array_morlet.html
8. **scikit-learn, RobustScaler.** `quantile_range` se expresa en percentiles [0,100]. https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html
9. **scikit-learn, MinMaxScaler.** https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html
10. **PyTorch, AdamW.** https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html
11. **PyTorch, interpolación 1D.** https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
12. **PyTorch Lightning, entrenamiento.** https://lightning.ai/docs/pytorch/stable/
13. **Optuna, TPESampler.** https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html
14. **Optuna, almacenamiento persistente.** https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/001_rdb.html
15. **Colab, preguntas frecuentes sobre recursos.** https://research.google.com/colaboratory/faq.html

## Criterios de citacion científica

- Registrar formato del dataset y nombre del proveedor.
- Citar el artículo de revision de BCI Competition IV y, cuando corresponda, la descripción del conjunto de datos y la investigacion FingerFlex.
- No afirmar replicacion exacta de los números de una publicación sin replicar preprocesamiento, cobertura temporal, selección de dedos y métrica de la evaluación original.
- No presentar como hecho una mejora causada por el modelo cuando podria deberse a tratamiento de QC o cambio de evaluación.
- Contrastar referencias con la versión publicada antes de integrarlas a un manuscrito; el repositorio no sustituye la fuente científica primaria.
