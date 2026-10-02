# Datasets limpios

Salida de la limpieza de datos (notebook `02_limpieza.ipynb`, reglas R01–R13). Son 6 ficheros `.parquet` en `../Datos limpios/`. Contienen datos de pacientes seudonimizados: **no se suben a git ni salen del entorno**. Solo salen agregados (ver las tablas de `../Tablas/`).

```python
import pandas as pd
D = "../Datos limpios/"
pacientes = pd.read_parquet(D + "pacientes.parquet")
medidas   = pd.read_parquet(D + "medidas.parquet").query("~excluida")   # solo medidas válidas
estado    = pd.read_parquet(D + "estado_definicion.parquet")
capa_A = estado[estado.subject_id.isin(pacientes.loc[pacientes.incluible_A, "subject_id"])]
```

| Dataset | Filas | Una fila es… | Para qué sirve |
|---|---|---|---|
| `pacientes` | 9.809 | un paciente | Población, descripción y criterios de inclusión |
| `medidas` | 27.504 | una medida (paciente × visita × variable) | Trayectorias y modelos mixtos |
| `estado_definicion` | 9.809 | un paciente | **Entrada del clustering** (fenotipos) |
| `visitas_tenacity` | 861 | una visita prevista de TENACITY (287 × 3) | Abandono frente a "aún no le toca" |
| `registro_limpieza` | 31 | una regla aplicada | Trazabilidad: qué se limpió y a cuántos casos afectó |
| `flujo_consort` | 7 | un paso del flujo de pacientes | Cuántos pacientes llegan a cada capa |

## `pacientes`

Tabla del núcleo (`cohorte_unificada_nucleo`) con la limpieza aplicada y 13 columnas derivadas. Los códigos "se desconoce" (9) de las variables `nu_*` pasan a ausente.

Columnas añadidas:

| Columna | Significado |
|---|---|
| `cohorte_analisis` | Registro de análisis: cada paciente en **un solo** registro. Los 437 fusionados con Lleida van a `POSTCOVID_LLEIDA` y los 22 enriquecidos de Virgen del Rocío a `VIRGEN_DEL_ROCIO`. Úsala para replicar entre cohortes. Para describir un registro completo siguen valiendo las banderas `en_<registro>` |
| `grupo_edad` | `<50`, `50-64`, `65-74`, `>=75`. Vacío en los 39 pacientes con tramo ambiguo |
| `edad_tramo_ord`, `edad_lim_inf` | Tramo original ordenado por edad y su límite inferior |
| `visita_dias_valida` | `visita_dias` sin los 54 valores negativos |
| `superviviente` | Sin exitus hospitalario |
| `cmd50`, `cmd99`, `cmd_ok` | Cumplimentación de CIBERESUCICOVID (`IH_PorcCMD` ≥ 50 y ≥ 99). `cmd_ok` es verdadero fuera de CIBERESUCICOVID, donde el criterio no aplica |
| `con_seguimiento` | Tiene alguna medida de seguimiento válida |
| `inconsistente` | Exitus hospitalario **con** seguimiento (3 casos, pendientes de revisión) |
| `incluible_A` | Candidato a la capa A: superviviente, sin inconsistencia, `cmd_ok` y DLCO o FVC entre 60 y 210 días |
| `incluible_B` | Candidato a la capa B: `incluible_A` y además HADS o mMRC en esa ventana |

> **Ojo:** `nu_dlco_pct_seg`, `nu_fvc_pct_seg`, etc. son las variables originales de la primera visita y **no** pasan por la limpieza de medidas (duplicados, rangos, fechas). Para analizar función pulmonar usa `estado_definicion` o `medidas`.

## `medidas`

Tabla larga con todas las medidas de seguimiento. Las medidas descartadas **no se borran**: quedan con `excluida = True`. Filtra siempre con `.query("~excluida")`.

| Columna | Significado |
|---|---|
| `subject_id` | Paciente |
| `registro` | Registro **de origen de la medida** (distinto de `cohorte_analisis` en los pacientes compartidos) |
| `cohorte_analisis` | Registro de análisis del paciente |
| `visita`, `mes_nominal` | Visita (`M3`, `LV1`, …) y su mes nominal. El eje temporal real es `dias_alta` |
| `variable` | `dlco`, `fvc`, `fev1`, `pm6m`, `mmrc`, `hads_a`, `hads_d` (valores) y `fibrosis_estricta`, `fibrosis_amplia`, `tractos_fibrosos` (0/1) |
| `valor` | Valor de la medida (% del predicho, metros, puntuación o 0/1 en imagen) |
| `dias_alta`, `meses_alta` | Fecha de la prueba menos fecha de alta. Si falta la fecha, se usa `visita_dias` solo en la primera visita (`fecha_imputada = True`). Hay 334 medidas válidas sin tiempo: no entran en ninguna ventana temporal |
| `fuente_columna` | Columna de la tabla completa de la que sale el valor |
| `excluida`, `motivo_exclusion` | `dias_negativos`, `fuera_de_rango`, `duplicado_entre_registros` o `codigo_no_disponible` |

Las variables de imagen son **solo descriptivas**: la fibrosis estricta/amplia solo cuenta formularios de TAC completos, y "tractos fibrosos" (CIBERESUCICOVID) no es el mismo constructo, así que no se compara entre cohortes.

## `estado_definicion`

Una fila por paciente con su estado en la **ventana de definición** (60–210 días tras el alta): la primera medida válida de cada variable. Es la tabla de partida del clustering.

Para cada variable `v` entre `dlco`, `fvc`, `hads_a`, `hads_d`, `mmrc`, `pm6m` y `fev1`:

| Columna | Significado |
|---|---|
| `v` | Valor de la primera medida válida en la ventana |
| `v_dias` | Día de esa medida |
| `v_alterada` | Estado por umbral clínico: DLCO y FVC < 80 %, HADS ≥ 8, mMRC ≥ 2, PM6M < 400 m (orientativo), FEV1 < 80 % |
| `v_disp` | `medida`, `falta` (el registro la recoge pero este paciente no la tiene) o `no_recogida` (ausencia estructural: **nunca se imputa**) |

Capa A = DLCO y FVC, disponibles en los 4 registros. Capa B añade HADS, mMRC y PM6M, solo en Lleida y TENACITY. FEV1 es solo descriptivo (redundante con FVC).

## `visitas_tenacity`

Estado de cada visita prevista (M3, M6, A1) para los 287 pacientes de TENACITY, cuyo reclutamiento sigue abierto.

| `estado` | Significado |
|---|---|
| `realizada` | Hay fecha de visita |
| `abandono` | Marcada como pérdida de seguimiento |
| `no_le_toca` | Aún no ha llegado la fecha (alta + mes + 60 días > fecha de corte). **No es abandono** |
| `sin_dato` | Ya debida, sin visita ni pérdida declarada: se trata como abandono no declarado |
| `sin_fecha_alta` | No se puede calcular la fecha teórica (18 pacientes) |

La fecha de corte (última visita registrada) está en `visitas_tenacity.attrs["corte"]`.

## `registro_limpieza`

Una fila por regla aplicada: `regla` (R01–R13), `descripcion`, `tabla` afectada y `N` casos, en total y por registro (`N_CIBERESUCICOVID`, `N_POSTCOVID_LLEIDA`, …). Es la versión completa; la versión con recuentos < 10 suprimidos está en `../Tablas/02_registro_limpieza.csv`.

## `flujo_consort`

Flujo de pacientes por `cohorte_analisis` (cada paso exige los anteriores). El índice es el paso.

| Paso | Total |
|---|---|
| 1. Pacientes únicos | 9.809 |
| 2. Supervivientes al alta | 7.576 |
| 3. Sin inconsistencias | 7.576 |
| 4. Cumplimentación suficiente (CMD ≥ 50 en CIBERESUCICOVID) | 5.071 |
| 5. Con alguna medida de seguimiento válida | 2.751 |
| 6. Capa A: DLCO o FVC en la ventana | 2.114 |
| 7. Capa B: además HADS o mMRC | 710 |

La capa A se reparte en 1.352 (CIBERESUCICOVID), 556 (Lleida), 159 (TENACITY) y 47 (Virgen del Rocío). La capa B tiene 532 de Lleida, 155 de TENACITY y 23 de Virgen del Rocío; CIBERESUCICOVID no aporta porque no recoge HADS ni mMRC.

## Tablas (`../Tablas/`)

Agregados con recuentos < 10 suprimidos, aptos para compartir:

| Fichero | Contenido |
|---|---|
| `02_registro_limpieza.csv` | Reglas aplicadas y casos afectados |
| `02_flujo_consort.csv` | Flujo de pacientes por registro |
| `07_funcion_pulmonar_primera_visita.csv` | DLCO, FVC y FEV1 en la primera visita por registro (análisis exploratorio) |
| `11_indicadores_clave.csv` | Cifras clave del análisis exploratorio |

## Pendiente de validación clínica

- Que un TAC completo sin lesión marcada signifique "sin lesión".
- La asignación de los pacientes compartidos a Lleida y a Virgen del Rocío.
- Los 3 pacientes con exitus hospitalario y seguimiento.
- El 37 % de fumadores activos en TENACITY.
