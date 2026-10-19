# Exclusión fija de canales

**Decisión:** excluir completos, en el preprocesamiento y **antes** de la referencia de mediana común, los canales de la tabla. La lista operativa vive solo en `configs/experiment_config.yaml` (`channels.excluded`); este documento explica la razón.

**Fecha:** 2026-10-18 · **Decidió:** la persona usuaria (propuesta técnica revisada en sesión) · **Protocolo:** v1.1

| Sujeto | Canal 0-based | Canal 1-based | Evidencia ECoG (EDA previo) | Figura de evidencia en la corrida EDA |
|---|---:|---:|---|---|
| S1 | 54 | 55 | Dos episodios transitorios (166–169 s y 195–196 s); STD local hasta 252× la típica del canal; salto máximo 646 311 u.o. Fuera de esos episodios el canal es normal | `fig/s02_S1_c054.png` |
| S2 | 20 | 21 | Transitorios recurrentes (8–12 episodios, 29–45 s de 400 s); STD local hasta 643× la típica; 99 % de la potencia de 40–200 Hz en ±1 Hz de las líneas de red | `fig/s02_S2_c020.png` |
| S2 | 37 | 38 | Transitorios recurrentes (9–10 episodios, 48–60 s de 400 s); STD local hasta 1817× la típica; STD basal anómala (376 frente a la mediana de 1794 del sujeto) | `fig/s02_S2_c037.png` |
| S3 | — | — | Sin canales atípicos (máximo z robusto de amplitud 14.5) | — |

Canales efectivos: **S1 = 61, S2 = 46, S3 = 64**.

## Por qué exclusión completa y no máscaras

- **S2-20 y S2-37:** los episodios ocupan del 7 al 15 % del registro; enmascararlos quitaría ese tiempo a **todos** los canales de S2 y partiría la señal en muchos segmentos.
- **S1-54:** solo tiene dos episodios, pero una máscara temporal exigiría una regla de detección para el test, que no puede inspeccionarse. Excluir el canal cuesta 1 de 62 y se aplica igual en desarrollo, validación y test.
- Sin máscaras temporales ni reparaciones locales.

## Lo que no se excluye

- **Ruido de línea de S2** (mediana del 67 % de la potencia de 40–200 Hz en las líneas): es común a casi todos los canales. Lo tratan la referencia de mediana común y el notch 60/120/180 Hz; las secciones s03 y s05 del EDA lo muestran.
- El canal 30 de S2 se mencionó en la conversación, pero es normal (z máximo 5.6); se refería al 20.

## Procedencia y límites

- Las cifras provienen de `results/eda_qc_review/20261018T230228Z/` (revisión de los CSV del primer EDA en Colab). Esa revisión **no** verificó los hashes de los `.mat` de origen (`raw_source_hashes_verified: false`). El EDA opcional (`results/eda/<run_id>/`) permite regenerar la evidencia con hashes verificados y la figura de cada canal. Desde v1.2 estas referencias son históricas: ni su carpeta ni sus figuras son requisitos de 02/03.
- La decisión usó solo ECoG de desarrollo: sin etiquetas, predicciones ni Pearson.
- Como la inspección observó todo el desarrollo, no debe presentarse como un criterio anidado por fold.
- Si la nueva corrida contradijera esta evidencia, se registra en `docs/experimental_log.md` y se pide autorización antes de cambiar la lista.
