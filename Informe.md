# Informe

## 1. Metodologia

Para poder hacer frente a este reto intentamos enfocarlo de diversas maneras, y se discutieron varios mètodos de trabajo para encontrar poder encontrar un objetivo con el que centrarnos.

### Primeres idees

Nuestra primera aproximación al reto fue orientada a la **prevención**: en lugar de limitarnos a describir qué grupos de pacientes existen después de una infección vírica grave, queríamos ir un paso más allá e identificar qué factores permiten predecir, lo más pronto posible, cómo va a evolucionar un paciente.

La motivación de fondo era la siguiente: cuanto más tarde se recogen los datos, más información hay disponible y más precisa puede ser la identificación de fenotipos —pero también más tarde puede intervenir el médico. Por ello, nos planteamos que la clave del valor clínico real estaba en la **predicción precoz**: ser capaces de asignar a un paciente a un fenotipo de riesgo ya a los 3 meses del alta, antes de que la situación se consolide o empeore.

Para explorar esta idea, planteamos un **estudio longitudinal en tres momentos temporales**: a los 3 meses, a los 6 meses y al año del alta hospitalaria. La idea era construir un modelo de clustering independiente en cada uno de esos tres puntos, utilizando el gran dataset de CIBERESUCICOVID, y comparar cuál de los tres momentos ofrecía la mejor capacidad de predicción sobre la evolución posterior del paciente.
El dataset longitudinal de POSTCOVID-Lleida, con seguimiento de hasta 4 años, actuaría como **cohorte de validación externa**: los fenotipos descubiertos en CIBERESUCICOVID se comprobarían en Lleida para ver si se replicaban y si la evolución observada allí era coherente con lo predicho.
El criterio de evaluación de los tres modelos sería qué punto temporal permite **antes** identificar correctamente el fenotipo de riesgo del paciente, con el objetivo de que el médico pueda intervenir cuanto antes.

## 2 Preparación de datos

Antes de poder construir ningún modelo, fue necesario realizar una limpieza y preparación exhaustiva de los datasets. Los datos provenían de cuatro registros clínicos distintos (CIBERESUCICOVID, POSTCOVID-Lleida, TENACITY y Virgen del Rocío), cada uno con sus propios formatos, convenciones y momentos de recogida, lo que generaba importantes irregularidades que debían tratarse con criterio clínico.
Se definieron **13 reglas de limpieza**, documentadas formalmente, cada una con su justificación y la evidencia del análisis exploratorio previo que la motivó. A continuación se describen las más relevantes.

### Corrección de valores especiales y codificación

En varias variables, el valor numérico `9` no representaba una categoría clínica real sino "se desconoce". Tratarlo sin corrección habría hecho que el algoritmo de clustering lo interpretara como un grupo de pacientes diferenciado, cuando en realidad es simplemente un dato ausente. Por ello, todos los valores `9` (y equivalentes como `9999` o `99999` usados en Virgen del Rocío) se convirtieron a valores nulos. De forma similar, la categoría `sexo = 2` ("no especificado") también se trató como ausente.

### Tratamiento de las fechas y el eje temporal

El análisis no trabaja con el número de visita (visita 1, visita 2, visita 3) sino con los **días reales transcurridos desde el alta hospitalaria**, ya que dos pacientes etiquetados como "visita 1" podían estar a 45 o a 120 días del alta, respectivamente. Las fechas de cada registro se leyeron en el formato específico de cada cohorte (por ejemplo, Virgen del Rocío usaba el formato americano MM/DD/AAAA), y se calcularon los días exactos.
Las 54 medidas cuya fecha de visita era anterior a la fecha del alta (días negativos, un error de registro) se marcaron como inválidas pero no se eliminaron del fichero, para poder revisarlas.

### Exclusión de valores fisiológicamente imposibles

Se definieron rangos plausibles para cada variable clínica: la DLCO, FVC y FEV1 debían estar entre el 20% y el 160% del valor predicho; la prueba de marcha de 6 minutos (PM6M) entre 100 y 900 metros; la escala de disnea mMRC entre 0 y 4; y los cuestionarios HADS entre 0 y 21. Cualquier medida fuera de estos rangos se consideró un error de registro y se excluyó del análisis.

### Gestión de pacientes compartidos entre registros

437 pacientes aparecían simultáneamente en CIBERESUCICOVID y en POSTCOVID-Lleida, y 22 lo hacían en CIBERESUCICOVID y en Virgen del Rocío. Esto implicaba que la misma prueba clínica (por ejemplo, una DLCO) aparecía dos veces en el conjunto de datos. Para no duplicar el peso de estos pacientes, se eliminó la copia de CIBERESUCICOVID cuando el otro registro tenía la misma medida con una diferencia de menos de 30 días (el 97% de los casos tenía valores idénticos con diferencia ≤ 1 punto).

Además, para garantizar que la validación entre cohortes fuera legítima, cada paciente se asignó a **un único registro de análisis**: los fusionados con Lleida se contabilizaron del lado de POSTCOVID-Lleida, y los de Virgen del Rocío del lado de ese registro. Sin esta regla, estaríamos validando el modelo parcialmente con los mismos pacientes que se usaron para construirlo.

### Calidad de datos en CIBERESUCICOVID

CIBERESUCICOVID disponía de un indicador de calidad por paciente (`IH_PorcCMD`): el porcentaje de campos del conjunto mínimo de datos que se habían rellenado. El análisis mostró que los 2.730 pacientes con menos del 50% de cumplimentación prácticamente no tenían seguimiento (menos del 0,5% tenía alguna DLCO registrada). Por ello, se estableció el umbral de **≥ 50% como criterio principal** de inclusión para el fenotipado, con ≥ 99% como análisis de sensibilidad para comprobar la robustez de los resultados.

### Distinción entre "dato ausente" y "variable no recogida"

Una decisión metodológica fundamental fue distinguir entre dos tipos de ausencia de datos: que un paciente no tenga un valor para una variable que su registro sí recoge (dato ausente, potencialmente recuperable) y que su registro directamente no mida esa variable (variable no recogida, ausencia estructural). Esta segunda situación nunca se imputó, ya que hacerlo equivaldría a inventar información clínica. Fue el caso, por ejemplo, de los cuestionarios de salud mental (HADS) en CIBERESUCICOVID, que ese registro no recoge en absoluto.

### Definición del estado de cada paciente: la ventana 60–210 días

Para el clustering, se tomó la **primera medida válida de cada paciente entre 60 y 210 días desde el alta**. Esta ventana fue la elegida porque es el período en que todos los registros con seguimiento tienen datos disponibles, permitiendo comparaciones válidas entre cohortes. Fuera de esta ventana, los registros no son comparables entre sí (Virgen del Rocío mide casi todo a las 6 semanas; Lleida tiene visitas hasta los 4 años, pero el resto no llega al año con datos suficientes).

Sobre estas medidas se aplicaron umbrales clínicos establecidos, como DLCO < 80%, HADS ≥ 8 o mMRC ≥ 2, para convertir los valores numéricos en estados clínicos interpretables. Se optó por umbrales clínicos en lugar de normalización estadística por cohorte (z-scores), porque esta última habría borrado diferencias reales entre registros: un paciente con DLCO del 71% no debería tratarse igual que uno con 80% solo porque están en cohortes distintas.

### Resultado de la limpieza: seis datasets estructurados

Como resultado del proceso de limpieza y preparación, los datos brutos se transformaron en **seis datasets especializados**, cada uno con un propósito concreto dentro del análisis:

| Dataset | Filas | Qué contiene | Para qué se usa |
| --- | --- | --- | --- |
| `pacientes` | 9.809 | Un paciente por fila, con todas las variables limpias y columnas derivadas (criterios de inclusión, grupo de edad, etc.) | Describir la población y aplicar los filtros de inclusión |
| `medidas` | 27.504 | Una medida por fila (paciente × visita × variable), incluyendo las excluidas marcadas | Estudiar las trayectorias de cada variable a lo largo del tiempo |
| `estado_definicion` | 9.809 | Un paciente por fila con su estado clínico en la ventana 60–210 días | **Tabla de entrada al clustering**: contiene si cada variable está alterada o no |
| `visitas_tenacity` | 861 | Una visita prevista por fila para los 287 pacientes de TENACITY | Distinguir abandono real de "aún no le ha tocado la visita" |
| `registro_limpieza` | 31 | Una regla de limpieza por fila, con cuántos pacientes afecta en cada registro | Trazabilidad y transparencia del proceso |
| `flujo_consort` | 7 | Un paso del flujo de pacientes | Mostrar cuántos pacientes llegan a cada etapa del análisis |

El flujo de pacientes a través de los filtros sucesivos fue el siguiente:

| Paso | Pacientes que pasan |
| --- | --- |
| Pacientes únicos en el dataset original | 9.809 |
| Supervivientes al alta hospitalaria | 7.576 |
| Con calidad suficiente de datos (CMD ≥ 50% en CIBERESUCICOVID) | 5.071 |
| Con alguna medida de seguimiento válida | 2.751 |
| **Capa A** — con DLCO o FVC en la ventana 60–210 días | **2.114** |
| **Capa B** — además con HADS o mMRC en esa ventana | **710** |

Esto significa que, de los 9.809 pacientes originales, el clustering de la Capa A trabajó con 2.114 (el 21.6% del total), y el de la Capa B con 710 (el 7.2%). La reducción es importante pero esperada: solo tienen datos de seguimiento válidos los pacientes que acudieron a las revisiones.

---

## 3. Clustering: evolución del enfoque

El diseño del clustering no fue lineal. A lo largo del proyecto exploramos diferentes aproximaciones antes de llegar a la solución final, aprendiendo algo relevante en cada paso.

### 3.1 Primera aproximación: Capa A y Capa B separadas

La primera idea fue organizar el clustering en dos capas diferenciadas según las variables disponibles. La **Capa A** trabajaría con variables respiratorias (DLCO y FVC), presentes en los cuatro registros, y descubriría los fenotipos en CIBERESUCICOVID para luego replicarlos en Lleida y TENACITY. La **Capa B** añadiría las variables de salud mental, calidad de vida y esfuerzo físico (HADS, mMRC, PM6M), que solo recogen Lleida y TENACITY, y descubriría sus fenotipos en Lleida para replicarlos en TENACITY.

**Punto fuerte:** La separación en capas era coherente con la realidad de los datos: no se puede incluir variables que solo recogen dos registros cuando el objetivo es descubrir fenotipos en los cuatro. Cada capa usa exactamente las variables que puede usar.

**Punto débil:** Al implementarlo vimos que la Capa A, con solo dos variables principales (DLCO y FVC), producía agrupaciones poco ricas —los grupos tendían a reflejar casi exclusivamente el nivel de DLCO, sin aportar mucha información adicional. La Capa B, por su parte, funcionaba con solo 710 pacientes, lo que limitaba la robustez estadística de los resultados.

---

### 3.2 Segunda aproximación: clustering solo con CIBERESUCICOVID

Dado el peso numérico de CIBERESUCICOVID (más del 95% de los pacientes), exploramos hacer el clustering únicamente sobre este registro, aprovechando su gran tamaño para obtener grupos más estables estadísticamente, y después proyectar los fenotipos encontrados sobre los otros registros.

**Punto fuerte:** Con 1.352 pacientes en la Capa A (frente a los ~160 de Lleida o los ~159 de TENACITY), el modelo tenía mucha más potencia estadística para encontrar grupos robustos. La estabilidad de los clústeres (medida por bootstrap) era significativamente mejor.

**Punto débil:** CIBERESUCICOVID no recoge las variables de salud mental ni de esfuerzo físico (HADS, PM6M, mMRC). Los fenotipos resultantes eran puramente respiratorios y no capturaban la dimensión multidisciplinar de las secuelas, que es precisamente uno de los aspectos más relevantes clínicamente.

---

### 3.3 Tercera aproximación: clustering específico para Lleida

Como alternativa, se planteó un clustering centrado en POSTCOVID-Lleida, el único registro con seguimiento largo (hasta 4 años) y con las variables de salud mental, calidad de vida, sueño y cognición. Esto permitiría un análisis verdaderamente multidimensional de las secuelas.

**Punto fuerte:** Lleida ofrece la visión más completa del paciente post-infeccioso: no solo función pulmonar, sino también estado emocional, sueño, cognición y calidad de vida. Los fenotipos descubiertos aquí serían los más ricos clínicamente y los más útiles para la práctica asistencial.

**Punto débil:** Lleida tiene 624 pacientes (556 en la Capa A), lo que es un tamaño razonable pero ajustado para clustering. Además, como cohorte de descubrimiento, deja solo TENACITY (287 pacientes) como cohorte de replicación, y ese tamaño es muy pequeño para validar resultados de forma robusta. El riesgo de sobreajuste al perfil específico de Lleida era real.

---

### 3.4 Aproximación final: clustering integrado con todos los datasets

Finalmente, se optó por una estrategia que combinase las ventajas de las aproximaciones anteriores: descubrir los fenotipos en CIBERESUCICOVID (por su tamaño), validarlos en Lleida y TENACITY, y usar las variables multidominio de Lleida como variables de **descripción** de los fenotipos (no de definición). De esta forma, los grupos se definen con variables comunes a todos los registros, pero se describen con toda la riqueza disponible en cada cohorte.

**Punto fuerte:** Combina el tamaño de CIBERESUCICOVID para el descubrimiento con la riqueza clínica de Lleida para la descripción. La replicación puede hacerse de forma rigurosa porque cada cohorte actúa en su rol adecuado.

**Punto débil:** La complejidad del diseño es mayor. Hay que gestionar cuidadosamente qué variables entran en la definición del fenotipo y cuáles solo en la descripción, y la interpretación de los resultados requiere mantener claro en todo momento qué se ha medido dónde.

---

## 4. Resultados

<!-- TODO equipo: completar con los resultados -->

## 5. Conclusiones

<!-- TODO equipo: completar con las conclusiones -->
