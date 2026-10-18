# Requisitos de verificación y criterios de aceptacion — Paso 3

**Objetivo:** comprobar integridad, aislamiento temporal, alineación y reproducibilidad del pipeline de ECoG antes de permitir experimentos con Optuna.

**Regla de aprobación:** V01-V14 deben reportar `PASS` con evidencia real. `SKIP`, `XFAIL` por ausencia de datos, `BLOCKED`, errores de configuración o un archivo vacio de pruebas **NO equivalen a PASS**. La aprobación requiere ejecución real sobre datos sintéticos y, donde se indica, sobre los tres sujetos del desarrollo. Ninguna prueba necesita abrir etiquetas de test.

## 1. Preconditions y ambiente

- `docs/dataset_spec.md` define forma, tiempos, índices y política de particion.
- `results/eda_qc/qc_decisions.json` debe existir con estado `approved` para los tres sujetos. Sin aprobación, detener la generación definitiva.
- `configs/experiment_config.yaml` debe validar estructura y tipos. `purge_samples` se determina durante estas pruebas; no utilizar valor provisional silencioso.
- Las señales sintéticas deben contener un marcador temporal inequivoco para detectar desplazamientos y referencias incorrectas.
- Registrar versions de Python, PyTorch, MNE, SciPy, scikit-learn y archivos de entrada mediante SHA-256.
- Separar claramente `source_time`, `feature_time`, `target_time` y `valid_mask`.

## 2. Matriz normativa

| ID | Objetivo | Procedimiento | Aceptacion / FAIL |
|---|---|---|---|
| V01 | Integridad | Leer development de S1/S2/S3, contrastar claves `.mat`, hashes, dtype, orientacion, tiempos y finitud | Forma coherente; originales inmutables; errores explicitos si clave/dimension no corresponde |
| V02 | QC | Aplicar `qc_decisions.json` desde datos crudos y verificar mapa de índices y máscaras | Acción aprobada correcta en todos los sujetos; sin deteccion aplicada por etiquetas o Pearson |
| V03 | Referencia espacial | Para matrices sintéticas, comparar `x - median(x, axis=canal)` instante por instante | Error numérico dentro de tolerancia FP especificada; sin estadísticas temporales globales |
| V04 | Splits y purga | Generar cinco bloques, segmentos train independientes, purga y validación por bloque | Índices validos disjuntos; cero ventanas que mezclen bloques o contexto prohibido |
| V05 | Escaladores | Ajustar train, perturbar solo val y recalcular escaladores/entradas train; inspeccionar percentiles | Escaladores y train identicos; `RobustScaler(10,90)` verdadero; test jamas `fit` |
| V06 | Retardo | Marcadores sintéticos espaciados a 100 Hz y registros artificiales por frontera | Para cada `t` valido, ECoG(t) -> flexión(t+20 muestras); cero pares que cruzan folds |
| V07 | Aislamiento val -> train | Inyectar impulso gigante en validación antes del procesamiento y regenerar ambos lados | Valores train admitidos inalterados dentro de tolerancia; sin contaminacion entre segmentos |
| V08 | Aislamiento train -> val | Inyectar impulso en train junto a frontera y regenerar val | Valores validos de val inalterados dentro de tolerancia y sin usar train para filtrar val |
| V09 | Filtrado, Morlet y resampling | Medir respuesta a impulso, frecuencia, tiempos, anti-alias y bordes por segmento | Soporte estimado registrado; muestreo 100 Hz consistente; máscaras y purga incluyen soporte real |
| V10 | Ventanas | Probar longitudes T<L, T=L, T=L+1, bloques disjuntos, máscaras y final de registro | Conteo exacto; ventana de T=L produce 1; sin ventanas intersegmento o sobre huecos |
| V11 | Finitud y dimensiones | Ejecutar conversiones para 3 sujetos/5 folds y verificar arrays, canales validos, rangos y máscaras | Finito donde corresponde; shapes (C,40,T), (5,T); persistencia y recarga consistente |
| V12 | Overlap-add | Modelo identidad/lineal sintético con stride 64, ultima ventana, huecos y bordes | Longitud original conservada; cobertura en todos los timestamps validos; reconstrucción correcta |
| V13 | Arquitectura | Ejecutar forward y backward para 27 (k,L,f) por sujeto en CPU o GPU con tensores validos | Entrada `(B,C,40,256)`, salida `(B,5,256)`, perdida y gradientes finitos; sin shape mismatch |
| V14 | Métricas y trazabilidad | Comparar Pearson, MSE, RMSE, CCC, coseno y macro frente a referencia numérica manual | Coincidencia dentro de tolerancia; Pearson indefinido NO se descarta silenciosamente; logs y hashes completos |

## 3. Diseño de pruebas por componente

### V01 — Integridad

- Cargar un archivo `.mat` de prueba con claves correctas y otro con una clave ausente: el segundo debe fallar con mensaje claro.
- Verificar que hay cinco columnas de `train_dg`, que sujetos tienen canales nominales según manifiesto (o cambio justificado) y que las longitudes temporales se pueden reconciliar.
- Comprobar el SHA-256 antes/después de lectura y que el archivo original no fue sobrescrito.
- **No** consultar archivo `*_testlabels.mat`; un test que lo intente antes de evaluación debe fallar por guardia de permisos o inyeccion de dependencias.

### V02 — Aplicación de QC

- Crear datos sintéticos con salto, pico breve y canal plano y ejecutar los detectores; generar marcas temporales en coordenadas de origen.
- Aplicar una lista aprobada de canales/máscaras a train/val/test, sin usar labels para decidirla.
- Verificar que la exclusion de canal cambia `C_validos`, reindexa correctamente y se aplica ANTES de referencia y Morlet.
- Si hay máscara temporal, ampliar por soporte de transformaciones y no concatenar segmentos sin continuidad.
- Un detector no puede excluir canales automáticamente por superar un umbral exploratorio sin decisión aprobada.

### V03 — Common Median Reference

Para `X` de forma `(C,T)`, comparar por instante:

```python
expected = X - np.median(X, axis=0, keepdims=True)
```

No aplicar `zscore` estimado sobre toda la serie antes de fijar folds. CMR espacial instantaneo no depende de tiempos futuros ni de etiquetas.

### V04 — Bloques temporales y purga

- Definir exactamente cinco bloques B1-B5 en índices de tiempo crudo/origen, conservando conversion a 100 Hz.
- Stage 1 valida B2 y B4; Stage 2 valida B1-B5.
- Para cada fold, registrar segmentos disjuntos, ventanas candidatas, intervalos usados por transformaciones y etiquetas desplazadas.
- Probar que la interseccion de sus contextos efectivos de train/val es vacia (salvo definicion explicitamente autorizada y documentada).
- Si `purge_samples` permanece `null` porque no se pudo justificar, marcar V04/V09 `BLOCKED` y no aprobar paso 3.

### V05 — Aislamiento de normalizacion

- Ajustar `RobustScaler` por cada feature canal-frecuencia con fold-train; percentiles **10 y 90**, `unit_variance=true`.
- Ajustar `MinMaxScaler` con cinco etiquetas de fold-train, sin influencia de val ni test.
- Guardar checksum de coeficientes y resultado sobre train.
- Introducir valores extremos SOLO en val y repetir: checksums del transformador y salida de train deben permanecer identicos dentro de la tolerancia declarada.
- Probar `transform(val)` y `inverse_transform(pred)` con shapes correctas; no guardar transformadores globales que puedan sobrescribirse entre folds.

### V06 — Alineación temporal de 200 ms

- Crear dos series sintéticas de marcadores con timestamps discretos; a 100 Hz, `delay_samples==20`.
- No desplazar señal respecto al objetivo en sentido contrario.
- Verificar `delay=0` por separado en test unitario de función para evitar errores de slicing vacio.
- Comprobar que el par (X[t],Y[t+20]) pertenece al mismo segmento cronológico valido y que los bordes se recortan consistentemente.
- No utilizar datos test para comprobar está condicion; sintéticos y development bastan.

### V07/V08 — Perturbaciones contrafactuales

- Congelar decisiones QC y particiones antes del experimento de perturbación.
- Inyectar un impulso artificial con amplitud extrema a ambos lados de cada frontera y volver a transformar.
- Comparar valores VALIDOS de la particion contraria; tolerancia de redondeo previamente declarada y documentada.
- El test debe usar las implementaciones reales de filtros/wavelets/decimacion, no una simulacion simplificada.
- Cualquier influencia no controlada implica aumentar guarda, reprocesar por segmentos o corregir pipeline, y repetir V04-V09.

### V09 — Soporte de filtros y wavelets

- Registrar MNE/SciPy: diseño efectivo del FIR bandpass/notch, orden/longitud, fase, padding y parámetros Morlet (40 freqs, 7 cycles, zero-mean).
- Medir soporte de propagacion de impulso de bandpass, cada notch, Morlet y antialias; definir tolerancia numérica explicita para el efecto residual, no solo el número de taps.
- Determinar margen total en segundos y muestras a 100 Hz; incluir 200 ms de alineación y contexto de ventanas cuando corresponda.
- Filtrar cada segmento independiente y marcar bordes invalidos. Probar equivalencia temporal, con tolerancias registradas, si se usa procesamiento fragmentado.
- No concluir que una guarda de 256 puntos (2.56 s) es suficiente sin medir las dependencias reales.

### V10 — Ventanas

Para segmento valido sin huecos, de T>=L:

```python
n_windows = 1 + (T - L) // stride
```

Con T<L devuelve cero. Para inferencia se debe incluir ademas una ventana anclada al final cuando el stride ordinario no cubre el ultimo instante. Una ventana no puede cruzar huecos de QC ni bordes entre segmentos temporales disjuntos.

### V11 — Sanidad de datos

- Comprobar para todos los sujetos y folds: formas, dtypes, canales, `isfinite`, memoria, mínimo/máximo y densidad de valores extremos.
- Rechazar matrices transformadas con divergencia de ordenes de magnitud no explicada por las decisiones de QC; no modificar etiquetas para corregir métricas.
- Serializar/deserializar arrays, escaladores y manifiestos; comprobar hashes y consistencia.

### V12 — Reconstrucción temporal

- Ejecutar overlap-add con predicciones sintéticas constantes e identidad, incluyendo el borde final.
- Mantener pesos/cuentas por índice temporal; cero cobertura en puntos declarados validos produce FAIL.
- No suavizar trayectorias de validación de forma diferente a las de test.
- Calcular Pearson del vector reconstruido por dedo, no media de Pearson por batch. Conservar exactamente cinco dedos y mismas máscaras por ejecución.

### V13 — Modelo parametrizable

- Baterias `{3,5,7} x {3,4,5} x {32,64,96}` = 27 configuraciones para cada sujeto.
- Kernels `[7,7]+[k]*(L-2)`, canales `[f,f,2f,4f,8f,8f][:L+1]`, dilataciones `[1,2,3,1,2][:L]`.
- Decoder interpola al tamaño temporal del skip y concatena sin recortes unilaterales arbitrarios.
- Probar varias longitudes de entrada opcionalmente, pero **obligatoria** 256. Forward + loss + backward con gradientes finitos.
- Un smoke test de pocos batches verifica formas, NO prueba memoria suficiente para un entrenamiento prolongado. Perfilado de GPU completo ocurre después del paso 3.

### V14 — Métricas, estado y seguimiento

- Pearson contra NumPy/SciPy sobre ejemplos positivos, negativos, perfectos y constantes.
- CCC contra cálculo manual y sensibilidad a escala/desplazamiento de medias; RMSE/MSE en unidades correctas.
- Evaluación indefinida por varianza cero o NaN debe invalidar la puntuacion y registrarse; no reemplazar `nanmean` silenciosamente.
- Comprobar precisión del valor de loss (coseno en eje tiempo por dedo) y su gradiente finito.
- Generar manifiesto con versión de código, configuración, entradas, seeds, folds y hash de escaladores.

## 4. Requisitos de entrega y formato de reporte

Archivos requeridos:

```text
results/verification/
  verification_report.json
  verification_summary.md
  test_results.xml
  filter_support.json
  approved_pipeline_manifest.json
  figures/
```

Formato mínimo de `verification_report.json`:

```json
{
  "schema_version": 1,
  "pipeline_git_commit": null,
  "config_sha256": null,
  "data_sha256_by_subject": {},
  "execution_timestamp_utc": null,
  "tests": {
    "V01": {"status": "PENDING", "evidence": [], "details": ""},
    "V02": {"status": "PENDING", "evidence": [], "details": ""},
    "V03": {"status": "PENDING", "evidence": [], "details": ""},
    "V04": {"status": "PENDING", "evidence": [], "details": ""},
    "V05": {"status": "PENDING", "evidence": [], "details": ""},
    "V06": {"status": "PENDING", "evidence": [], "details": ""},
    "V07": {"status": "PENDING", "evidence": [], "details": ""},
    "V08": {"status": "PENDING", "evidence": [], "details": ""},
    "V09": {"status": "PENDING", "evidence": [], "details": ""},
    "V10": {"status": "PENDING", "evidence": [], "details": ""},
    "V11": {"status": "PENDING", "evidence": [], "details": ""},
    "V12": {"status": "PENDING", "evidence": [], "details": ""},
    "V13": {"status": "PENDING", "evidence": [], "details": ""},
    "V14": {"status": "PENDING", "evidence": [], "details": ""}
  },
  "approved": false
}
```

**Estado inicial:** PENDING. El implementador debe actualizar el resultado solo después de ejecutar cada prueba real. Sin archivos originales, las comprobaciones de dataset/muestras reales quedaran `BLOCKED`.

## 5. Regla de congelacion

- **Si todos V01-V14 tienen PASS:** guardar hashes, cerrar el QC, preprocesamiento, división temporal, manejo de retardo, escalado, inferencia, métricas y arquitectura; pasar a factibilidad Colab y decisión posterior de épocas.
- **Si alguno falla:** identificar componente responsable, corregirlo, regenerar artefactos afectados y repetir todas las pruebas que dependan de el. Mantener el estado de congelacion desactivado mientras exista un FAIL o BLOCKED.
- **Si más tarde aparece un defecto genuino:** registrar incidencia y versión; invalidar/repetir solo resultados contaminados. No cambiar decisiones por puntaje de Pearson.

## 6. Restricciones de alcance

Estos tests no deciden cual LR, batch size o duración maximiza Pearson, ni demuestran ausencia de sobreajuste. Tampoco constituyen una comparación BatchNorm vs LayerNorm ni decoder vs transposed convolution. Su función es garantizar que los experimentos de entrenamiento posteriores sean metodologicamente coherentes.
