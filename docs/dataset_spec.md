# Especificación del dataset y contrato de datos

**Proyecto:** regresion continua de cinco trayectorias de flexión a partir de ECoG.
**Dataset de origen:** BCI Competition IV, Dataset 4, de la Universidad de Washington y Wadsworth Center.
**Regla:** confirmar los metadatos de cada archivo `.mat` al cargarlo; la información siguiente describe el esquema esperado y no sustituye esa validación.

## 1. Sujetos y adquisición

| Sujeto | Canales ECoG nominales | Duración desarrollo nominal | Duración test nominal |
|---|---:|---:|---:|
| S1 | 62 | 400 s | 200 s |
| S2 | 48 | 400 s | 200 s |
| S3 | 64 | 400 s | 200 s |

- ECoG muestreada a **1000 Hz**; los registros originales presentan una banda de adquisición aproximada de 0.15-200 Hz, distinta de nuestra banda analitica 40-200 Hz.
- Guante con **cinco** mediciones: pulgar, índice, medio, anular y menique; muestreo físico del guante a **25 Hz**.
- En archivos de competencia, la serie de flexión de desarrollo puede estar **sobremuestreada a 1000 filas por segundo** para compartir la base temporal de ECoG. **No confundir** frecuencia efectiva del guante con frecuencia de filas almacenadas.
- Los identificadores de canal son índices de adquisición, no posiciones anatomicas. No asumir vecindad física entre canales consecutivos.
- No asumir las unidades eléctricas exactas o una calibración física de la flexión si no están documentadas por la fuente. Preservar unidades originales y cualquier conversion aplicada.

## 2. Archivos de entrada

Estructura de referencia, adaptable mediante configuración de rutas:

```text
data/raw/
  sub1_comp.mat
  sub2_comp.mat
  sub3_comp.mat

# Las etiquetas de evaluacion deben mantenerse fuera del arbol
# accesible a EDA, QC, preprocesamiento, pilotos y Optuna:
<test_labels_storage>/sub1_testlabels.mat
<test_labels_storage>/sub2_testlabels.mat
<test_labels_storage>/sub3_testlabels.mat
```

La ruta `<test_labels_storage>` es simbolica y **no** debe resolverse antes de la evaluación final.

Variables esperadas en `subX_comp.mat`:

| Clave | Forma esperada (sin transponer) | Acceso |
|---|---|---|
| `train_data` | `(N_train_ecog, C)`; nominal `400000 x C` | Permitido desde EDA |
| `train_dg` | `(N_train_glove_storage, 5)`; habitualmente `400000 x 5` | Permitido para entrenamiento y validación, NO para QC basado en predicción |
| `test_data` | `(N_test_ecog, C)`; nominal `200000 x C` | Únicamente en transformación/evaluación final; no ajustar decisiones a partir de test |

Variables esperadas en el archivo de test labels:

| Clave | Forma esperada | Acceso |
|---|---|---|
| `test_dg` | `(N_test_glove_storage, 5)`; suele compartir la escala temporal almacenada de test | **Exclusivo de evaluación final** |

Si la distribucion de datos contiene claves, disposiciones u orientaciones distintas, el cargador debe emitir un error explicito o utilizar un adaptador documentado. No transponer silenciosamente ante dimensiones ambiguas.

Los archivos publicados pueden ser MATLAB clasico (`scipy.io.loadmat`); si se detecta MATLAB v7.3/HDF5, utilizar un lector compatible y registrar la diferencia. El directorio original es inmutable.

## 3. Contratos de tensores dentro de Python

| Objeto | Dimensiones | Ejemplo |
|---|---|---|
| `ecog_raw` | `(C, T1000)` | `(62, 400000)` S1 nominal |
| `flexion_raw_storage` | `(5, T_storage)` | `(5, 400000)` si almacenado a 1 kHz |
| `wavelet_power` | `(C_validos, 40, T100)` | salida del procesamiento |
| `targets_100hz` | `(5, T100)` | interpolación temporal del glove |
| `X_window` | `(C_validos, 40, 256)` | muestra de entrenamiento |
| `Y_window` | `(5, 256)` | etiqueta alineada |
| `X_batch` | `(B, C_validos, 40, 256)` | entrada de red |
| `Y_batch`, `Y_hat` | `(B, 5, 256)` | objetivo y predicción |

Los números de canales validos pueden diferir por sujeto después de QC. La reduccion espacial entrenable debe construirse a partir de `C_validos`, no de una constante global.

## 4. Tiempos y remuestreo

1. Construir un vector de tiempos a partir de la frecuencia de adquisición, la cantidad de muestras y cualquier metadato de desfase original disponible.
2. Distinguir muestras de `train_dg` guardadas a 1 kHz de mediciones efectivas del guante a 25 Hz. Verificar el patron de sobremuestreo antes de recuperar las muestras efectivas.
3. Interpolar cada dedo a la rejilla de 100 Hz, usando tiempo físico y un criterio declarado para los extremos. Evitar duplicar o perder muestras por índices fuera de rango.
4. Procesar ECoG a 100 Hz con transformación Morlet y antialias; comprobar correspondencia exacta entre timestamps de entradas y etiquetas.
5. Aplicar retardo de 0.2 s con sentido definido: para el par valido en tiempo `t`, `X(t)` se empareja con `Y(t+0.2 s)`; a 100 Hz son 20 muestras.
6. Una pareja solo es valida si entrada, contexto temporal de filtros/wavelets, etiqueta desplazada y ventana completa pertenecen a la particion temporal autorizada. Preservar índices originales y máscara de validez.

**Nota:** el test utiliza la misma transformación y retardo, pero sus etiquetas se leen solamente al abrir la etapa de evaluación final. La simple lectura de etiquetas de test durante EDA cuenta como acceso no autorizado.

## 5. Particiones y cobertura

- **Desarrollo:** `train_data` y `train_dg`, divididos cronologicamente en cinco bloques B1-B5 por sujeto.
- **Etapa 1:** validar B2 y B4; el resto del desarrollo es candidato a entrenamiento con purga y guardas.
- **Etapa 2:** validar B1, B2, B3, B4 y B5, uno por entrenamiento.
- **Prueba independiente:** `test_data` y `test_dg`, reservados hasta evaluación final.
- No existe intercambio de datos entre sujetos, ni tratamiento de folds disjuntos como una sola serie continua.
- La purga aprobada se define tras medir soporte temporal efectivo de filtros/Morlet; no asumir que 256 muestras bastan.
- Al enmascarar artefactos, reconstruir trayectorias sobre sus segmentos validos sin interpolar huecos arbitrariamente para aumentar cobertura.

## 6. Metadatos minimos y validaciones

Por archivo registrar:

```json
{
  "subject": "S1",
  "dataset": "BCI_Competition_IV_Dataset_4",
  "path": "data/raw/sub1_comp.mat",
  "sha256": "CALCULATE_ON_LOAD",
  "mat_keys": ["train_data", "train_dg", "test_data"],
  "n_ecog_channels_original": null,
  "n_ecog_channels_valid": null,
  "sampling_hz_ecog": 1000,
  "sampling_hz_glove_effective": 25,
  "n_fingers": 5,
  "n_train_samples_actual": null,
  "n_test_samples_actual": null,
  "time_origin_s": 0.0,
  "qc_decisions_hash": null
}
```

Las cadenas con `CALCULATE_ON_LOAD` y los `null` no son valores de datos; deben rellenarse programaticamente al cargar o implementar la etapa correspondiente.

Validar: claves esperadas; matrices bidimensionales de tipo numérico; finitud; forma temporal; continuidad de tiempo; etiquetas con cinco dedos; índices de canal correctamente mapeados; hashes estables; ausencia de acceso a etiquetas de test en cualquier módulo anterior a evaluación.

## 7. Persistencia y restricciones

- Mantener originales de solo lectura, fuera de Git.
- Versionar arrays procesados o reportes mediante hash de configuración, sin sobrescribir silenciosamente.
- Persistir escaladores **por fold** y sus hashes, nunca reutilizar ajustes de otro sujeto/fold.
- No guardar checkpoints o reportes del test en la misma carpeta de desarrollo.
- Conservar por sujeto el mapa entre canales de origen (índices 0-based), canales validos y su nueva posicion.

## 8. Fuentes verificables

- Sitio oficial BCI Competition IV: https://www.bbci.de/competition/iv/
- Descarga oficial: https://www.bbci.de/competition/iv/download/
- Liang y Bougrain, descripción del dataset y su adquisición: https://pmc.ncbi.nlm.nih.gov/articles/PMC3384842/

La descripción externa informa los rasgos nominales del dataset. La implementación debe validar los archivos **realmente suministrados** antes de dar sus dimensiones como hechos medidos.
