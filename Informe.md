# Informe

## 1. Metodologia

Para poder hacer frente a este reto intentamos enfocarlo de diversas maneras, y se discutieron varios mètodos de trabajo para encontrar poder encontrar un objetivo con el que centrarnos.

### 1.1 Primera idea: predicción precoz de fenotipos y factores de riesgo

Nuestra primera aproximación al reto fue orientada a la **prevención**: en lugar de limitarnos a describir qué grupos de pacientes existen después de una infección vírica grave, queríamos ir un paso más allá e identificar qué factores permiten predecir, lo más pronto posible, cómo va a evolucionar un paciente.

La motivación de fondo era la siguiente: cuanto más tarde se recogen los datos, más información hay disponible y más precisa puede ser la identificación de fenotipos —pero también más tarde puede intervenir el médico. Por ello, nos planteamos que la clave del valor clínico real estaba en la **predicción precoz**: ser capaces de asignar a un paciente a un fenotipo de riesgo ya a los 3 meses del alta, antes de que la situación se consolide o empeore.

Para explorar esta idea, planteamos un **estudio longitudinal en tres momentos temporales**: a los 3 meses, a los 6 meses y al año del alta hospitalaria. La idea era construir un modelo de clustering independiente en cada uno de esos tres puntos, utilizando el gran dataset de CIBERESUCICOVID, y comparar cuál de los tres momentos ofrecía la mejor capacidad de predicción sobre la evolución posterior del paciente.
El dataset longitudinal de POSTCOVID-Lleida, con seguimiento de hasta 4 años, actuaría como **cohorte de validación externa**: los fenotipos descubiertos en CIBERESUCICOVID se comprobarían en Lleida para ver si se replicaban y si la evolución observada allí era coherente con lo predicho.
El criterio de evaluación de los tres modelos sería qué punto temporal permite **antes** identificar correctamente el fenotipo de riesgo del paciente, con el objetivo de que el médico pueda intervenir cuanto antes.

### 1.2 Preparación de datos

Antes de poder construir ningún modelo, fue necesario realizar una limpieza y preparación exhaustiva de los datasets. Los datos provenían de cuatro registros clínicos distintos (CIBERESUCICOVID, POSTCOVID-Lleida, TENACITY y Virgen del Rocío), cada uno con sus propios formatos, convenciones y momentos de recogida, lo que generaba importantes irregularidades que debían tratarse con criterio clínico.
Se definieron **13 reglas de limpieza**, documentadas formalmente, cada una con su justificación y la evidencia del análisis exploratorio previo que la motivó. A continuación se describen las más relevantes.

#### Corrección de valores especiales y codificación

En varias variables, el valor numérico `9` no representaba una categoría clínica real sino "se desconoce". Tratarlo sin corrección habría hecho que el algoritmo de clustering lo interpretara como un grupo de pacientes diferenciado, cuando en realidad es simplemente un dato ausente. Por ello, todos los valores `9` (y equivalentes como `9999` o `99999` usados en Virgen del Rocío) se convirtieron a valores nulos. De forma similar, la categoría `sexo = 2` ("no especificado") también se trató como ausente.

#### Tratamiento de las fechas y el eje temporal

El análisis no trabaja con el número de visita (visita 1, visita 2, visita 3) sino con los **días reales transcurridos desde el alta hospitalaria**, ya que dos pacientes etiquetados como "visita 1" podían estar a 45 o a 120 días del alta, respectivamente. Las fechas de cada registro se leyeron en el formato específico de cada cohorte (por ejemplo, Virgen del Rocío usaba el formato americano MM/DD/AAAA), y se calcularon los días exactos.
Las 54 medidas cuya fecha de visita era anterior a la fecha del alta (días negativos, un error de registro) se marcaron como inválidas pero no se eliminaron del fichero, para poder revisarlas.

#### Exclusión de valores fisiológicamente imposibles

Se definieron rangos plausibles para cada variable clínica: la DLCO, FVC y FEV1 debían estar entre el 20% y el 160% del valor predicho; la prueba de marcha de 6 minutos (PM6M) entre 100 y 900 metros; la escala de disnea mMRC entre 0 y 4; y los cuestionarios HADS entre 0 y 21. Cualquier medida fuera de estos rangos se consideró un error de registro y se excluyó del análisis.

#### Gestión de pacientes compartidos entre registros

437 pacientes aparecían simultáneamente en CIBERESUCICOVID y en POSTCOVID-Lleida, y 22 lo hacían en CIBERESUCICOVID y en Virgen del Rocío. Esto implicaba que la misma prueba clínica (por ejemplo, una DLCO) aparecía dos veces en el conjunto de datos. Para no duplicar el peso de estos pacientes, se eliminó la copia de CIBERESUCICOVID cuando el otro registro tenía la misma medida con una diferencia de menos de 30 días (el 97% de los casos tenía valores idénticos con diferencia ≤ 1 punto).

Además, para garantizar que la validación entre cohortes fuera legítima, cada paciente se asignó a **un único registro de análisis**: los fusionados con Lleida se contabilizaron del lado de POSTCOVID-Lleida, y los de Virgen del Rocío del lado de ese registro. Sin esta regla, estaríamos validando el modelo parcialmente con los mismos pacientes que se usaron para construirlo.

#### Calidad de datos en CIBERESUCICOVID

CIBERESUCICOVID disponía de un indicador de calidad por paciente (`IH_PorcCMD`): el porcentaje de campos del conjunto mínimo de datos que se habían rellenado. El análisis mostró que los 2.730 pacientes con menos del 50% de cumplimentación prácticamente no tenían seguimiento (menos del 0,5% tenía alguna DLCO registrada). Por ello, se estableció el umbral de **≥ 50% como criterio principal** de inclusión para el fenotipado, con ≥ 99% como análisis de sensibilidad para comprobar la robustez de los resultados.

#### Distinción entre "dato ausente" y "variable no recogida"

Una decisión metodológica fundamental fue distinguir entre dos tipos de ausencia de datos: que un paciente no tenga un valor para una variable que su registro sí recoge (dato ausente, potencialmente recuperable) y que su registro directamente no mida esa variable (variable no recogida, ausencia estructural). Esta segunda situación nunca se imputó, ya que hacerlo equivaldría a inventar información clínica. Fue el caso, por ejemplo, de los cuestionarios de salud mental (HADS) en CIBERESUCICOVID, que ese registro no recoge en absoluto.

#### Definición del estado de cada paciente: la ventana 60–210 días

Para el clustering, se tomó la **primera medida válida de cada paciente entre 60 y 210 días desde el alta**. Esta ventana fue la elegida porque es el período en que todos los registros con seguimiento tienen datos disponibles, permitiendo comparaciones válidas entre cohortes. Fuera de esta ventana, los registros no son comparables entre sí (Virgen del Rocío mide casi todo a las 6 semanas; Lleida tiene visitas hasta los 4 años, pero el resto no llega al año con datos suficientes).

Sobre estas medidas se aplicaron umbrales clínicos establecidos, como DLCO < 80%, HADS ≥ 8 o mMRC ≥ 2, para convertir los valores numéricos en estados clínicos interpretables. Se optó por umbrales clínicos en lugar de normalización estadística por cohorte (z-scores), porque esta última habría borrado diferencias reales entre registros: un paciente con DLCO del 71% no debería tratarse igual que uno con 80% solo porque están en cohortes distintas.
