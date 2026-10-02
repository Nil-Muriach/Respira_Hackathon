# Reto 2 · Fenotipado digital de secuelas post-infecciosas

**Documento de apoyo para la presentación al jurado.** Explica y justifica todo el proyecto: datos, limpieza, fenotipado y modelos predictivos. Todas las cifras salen de los notebooks ejecutados (entre paréntesis, el notebook de origen).

---

## 0 · Resumen en una página

**La pregunta del reto no es si existen subgrupos (siempre salen), sino si sobreviven al cambio de cohorte.** Por eso todo el proyecto gira alrededor de una idea: un fenotipo solo cuenta si pasa un **"pasaporte"** de controles:
- tiene más estructura que unos datos al azar;
- es estable;
- se reproduce en una cohorte que no ha visto;
- tiene sentido clínico.

**Tres hallazgos:**

1. **Hay un fenotipo que sí sobrevive al cambio de cohorte.** A los 3 meses, con la función pulmonar que las tres cohortes miden igual (DLCO, FVC, FEV1), salen dos grupos:
   - **"función conservada"** (DLCO 79 %, FVC 96 %);
   - **"afectación funcional"** (DLCO 65 %, FVC 73 %).

   Las cifras que lo sostienen (nb 05):
   - es estable (Jaccard 0,92);
   - no depende de la cohorte (V de Cramér 0,12);
   - se reproduce al aprender con dos cohortes y probar en la tercera (ARI 0,50–0,73);
   - anticipa la DLCO baja al año: 47 % frente a 66 %, p < 0,0001.
2. **Los fenotipos más "ricos" no sobreviven, y lo demostramos.** Ni los sintomáticos de CIBERESUCICOVID (nb 03) ni los multidominio de Lleida (nb 04) se reproducen fuera de su cohorte. Es un resultado negativo honesto, y es justo lo que el reto pide comprobar.
3. **Mensaje clínico: el momento para predecir es la visita de los 3 meses, no el alta.**
   - Con los datos del alta, cualquier modelo (logística, EBM o red neuronal) se queda en un AUC de ≈ 0,63 (nb 06–09).
   - Al añadir la DLCO y la FVC de la primera visita, la predicción de la DLCO baja al año sube a un **AUC de 0,71**. Es la única mejora significativa del proyecto (+0,11, p = 0,016; nb 09).

**Traducción para la consulta:** la primera espirometría con DLCO a los 3 meses es la herramienta clave para decidir a quién seguir de cerca.

---

## 1 · El reto y cómo lo planteamos

- **Objetivo del reto:** un notebook reproducible que identifique fenotipos clínicos de las secuelas tras una infección vírica grave, **reproducibles entre cohortes**, interpretables y útiles en consulta.
- **Criterios de evaluación:** justificación (20 %), rigor científico (30 %), interpretabilidad (20 %) y presentación (30 %).
- **Nuestro criterio central:** antes de ver resultados fijamos qué tiene que cumplir un fenotipo para considerarlo real (el "contrato", en `config.yaml`):

| "Sello" del pasaporte | Cómo se mide | Umbral fijado de antemano |
|---|---|---|
| Hay estructura | Silueta real frente a la de 100 conjuntos con las columnas permutadas (prueba nula) | Supera el percentil 95 del nulo |
| Es estable | Jaccard bootstrap por clúster (200 remuestreos) | ≥ 0,75 estable; 0,5–0,75 patrón; < 0,5 se disuelve |
| Viaja | Transferencia al medoide frente a clustering propio en la otra cohorte (ARI, Jaccard con emparejamiento húngaro) | ARI con IC 95 % que excluye 0; Jaccard ≥ 0,5 |
| No separa cohortes | V de Cramér entre fenotipo y cohorte; perfiles por cohorte | Pequeña (< 0,3); perfiles parecidos |
| Sentido clínico | Fichas descriptivas con variables que **no** entran en la definición | Lo valora el equipo, no la IA |

Un fenotipo que no cumple se **presenta igualmente**, como no replicado. No se ajusta nada para que "salga".

---

## 2 · Los datos: cuatro estudios que no miden lo mismo

| Cohorte | Pacientes | Qué aporta | Limitación |
|---|---|---|---|
| **CIBERESUCICOVID** | 9.274 | Gran N, 76 centros, fase aguda de UCI muy completa | Seguimiento casi solo respiratorio y de síntomas; a 12 meses no pregunta síntomas |
| **POSTCOVID-Lleida** | 624 (437 también en CIBERESUCICOVID) | El seguimiento más rico y largo (hasta 4 años): función, PM6M, HADS, calidad de vida, sueño | Un solo centro |
| **TENACITY** | 287 | Post-UCI pospandémica (no solo COVID), función física | Pequeña; reclutamiento abierto; casuística distinta (18 % EPOC, 37 % fumadores activos) |
| **Virgen del Rocío** | 83 | Espirometría y DLCO **al mes** | Casi sin seguimiento posterior |

Hay **9.809 pacientes únicos** en una tabla de 4.956 columnas.

**El problema de fondo:** el **90,5 % de las celdas está vacío**. No es que falten datos: cada registro usó su propio cuaderno. Solo **13 variables clínicas son comunes a las cuatro cohortes**:
- **basales:** edad, sexo, tabaquismo, hipertensión, diabetes, cardiopatía, EPOC, enfermedad renal;
- **del ingreso:** UCI y estancia;
- **de seguimiento:** DLCO, FVC y FEV1.

Esta restricción explica muchas de las decisiones posteriores.

---

## 3 · Exploración (nb 01): lo que encontramos y cómo condicionó el diseño

| Hallazgo | Cifra | Decisión que provocó |
|---|---|---|
| El N útil es mucho menor que el N total | Solo ~2.300 pacientes tienen alguna DLCO; en CIBERESUCICOVID, el 14 % | Trabajar con N reales y declararlos siempre |
| Quién tiene DLCO depende del **centro** | 36 de 76 centros no la miden nunca | Validar dejando fuera cada centro |
| Las cohortes miden en momentos distintos | Virgen del Rocío, mediana de 44 días tras el alta; el resto, 86–120 días | Usar el tiempo real (`visita_dias`, fechas) y ventanas comunes; Virgen del Rocío solo como complemento |
| Aun en la misma ventana, TENACITY es distinta | DLCO ~10 puntos más alta a los 75–105 días | Usar **umbrales clínicos y rangos fijos**, no z-scores por cohorte (borrarían diferencias reales) |
| Los dominios están poco correlacionados | Pulmón, esfuerzo, disnea y ánimo: \|ρ\| ≤ 0,23 | Pesar los dominios por igual; esperar fenotipos multidominio difusos |
| El abandono es informativo | Quien no vuelve al año partía 6–7 puntos de DLCO más alto | Declararlo como limitación de los horizontes tardíos |
| Pruebas duplicadas | Los 437 pacientes compartidos tienen la misma DLCO registrada dos veces (97 % idéntica) | Contar cada prueba una vez; cada paciente en un solo lado de la replicación |
| El TAC se repite de forma selectiva | Se repite a quien tenía lesiones | La imagen solo **describe**, no define |

---

## 4 · Limpieza de datos (nb 02, `src/limpieza.py`)

### 4.1 Principios
1. **Nada se borra en silencio.** Cada regla deja una entrada en un registro con el N afectado, y las medidas descartadas se quedan marcadas como `excluida` con su motivo.
2. **No se imputa la ausencia estructural.** Si un registro no recoge una variable, es "no recogida", que es distinto de "falta".
3. **Ningún número mágico.** Umbrales, rangos, ventanas y columnas por visita están en `config.yaml`.
4. **Privacidad.** Los datos están seudonimizados, no anonimizados:
   - solo se muestran agregados y las celdas con N < 10 se suprimen;
   - los datos limpios se guardan fuera del repositorio.

### 4.2 Las reglas

| Regla | Qué hace | Por qué | Casos |
|---|---|---|---|
| R01 | Códigos de "se desconoce" (9, 9999, 99999; resolución = 3) → ausente | Si no, el 9 entraría como si fuera una categoría clínica | 552 en tabaquismo; 1.071 en resolución |
| R02 | Edad como tramo ordenado y como grupo amplio | Los tramos son desiguales por protección de datos | 39 tramos ambiguos |
| R03 | Fechas con formato explícito por registro; tiempo = fecha de la prueba − alta | El número de visita no es el eje temporal | — |
| R04 | Fechas anteriores al alta → fuera del análisis temporal | Son errores o reingresos | 54 `visita_dias` negativos |
| R05 | Rangos plausibles (p. ej., DLCO y FVC 20–160 %) | Errores de registro | ~40 medidas |
| R06 | Prueba repetida en pacientes compartidos → se queda la del registro de origen | 97 % idénticas; sin la regla, esos pacientes pesarían el doble | **953 medidas** |
| R07 | Cada paciente en una sola cohorte de análisis | Validar entre cohortes exige pacientes distintos | 437 + 22 |
| R08 | Exitus → no superviviente; exitus con seguimiento → inconsistente | Solo los supervivientes pueden tener secuelas | 2.233 exitus; 3 inconsistentes |
| R09 | Cumplimentación de CIBERESUCICOVID ≥ 50 % (≥ 99 % como sensibilidad) | Los registros < 50 % casi no tienen seguimiento | 2.729 |
| R10 | TAC: solo formularios completos; fibrosis estricta y amplia | Las casillas valen 0 aunque nadie las rellene | Miles de visitas sin TAC |
| R11 | Disponibilidad "medida", "falta" o "no recogida" | Distinguir el vacío estructural del dato perdido | — |
| R12 | TENACITY: separar el abandono de "aún no le toca la visita" | El reclutamiento sigue abierto | 78 visitas |
| R13 | Estado en la ventana de definición con umbrales clínicos | DLCO/FVC < 80 %, HADS ≥ 8, mMRC ≥ 2 | — |
| R14 | Preguntas condicionadas: resolución total → fatiga = 0 | El cuaderno solo pregunta la fatiga si los síntomas persisten (100 % vacía con resolución total) | **2.003 visitas** |

**Problemas de comparabilidad que no se pueden limpiar, y que declaramos:**
- **"Fatiga" no mide lo mismo entre cohortes:** 58 % en CIBERESUCICOVID, 2 % de astenia en Lleida y 25 % en TENACITY. Solo se usa dentro de CIBERESUCICOVID.
- **FEV1/FVC en CIBERESUCICOVID mezcla unidades:** el 35 % de los valores son > 100. Queda fuera.
- **A los 12 meses CIBERESUCICOVID no pregunta síntomas.**

### 4.3 Flujo de pacientes (nb 02)
| Paso | Total |
|---|---|
| Pacientes únicos | 9.809 |
| Supervivientes al alta | 7.576 |
| Cumplimentación suficiente | 5.071 |
| Con alguna medida de seguimiento válida | ~4.500 |
| Con DLCO o FVC en la ventana de definición (60–210 días) | ~2.100 |

---

## 5 · Cómo hacemos el fenotipado (método común a nb 03, 04 y 05)

| Decisión | Qué hicimos | Justificación |
|---|---|---|
| **Horizontes acumulados** | H3: visita de ~3 meses. H6: 3 + 6 meses, con el cambio entre visitas. H12: 3 + 6 + 12 meses | Responde a "¿qué fenotipo tiene el paciente con lo que sabemos hasta ese momento?" sin mirar el futuro (comprobado con tests) |
| **Ancla** | Para entrar en un horizonte hay que tener medidas en su última visita | Que el fenotipo describa el estado en ese momento, no solo el pasado |
| **Variables de definición ≠ de descripción** | Se agrupa con el estado de seguimiento; el ingreso, la imagen y los eventos solo describen | Separar el fenotipo de sus causas y consecuencias |
| **Distancia de Gower** | Admite variables mixtas y datos ausentes por pares, sin imputar | Los datos son mixtos y con muchos huecos estructurales |
| **Rangos fijos (P2,5–P97,5) idénticos en todas las cohortes** | p. ej., DLCO 40–110, FVC 55–125 | Con z-scores por cohorte se borrarían diferencias reales. Con el rango plausible (20–160), 20 puntos de DLCO pesaban 0,14 frente a 1 de un síntoma distinto (lo corregimos) |
| **Peso igual por dominio** | p. ej., difusión = espirometría | Que un dominio con más columnas (FVC y FEV1) no domine |
| **PAM (k-medoides)** | FasterPAM | Funciona con cualquier distancia; el medoide es un paciente real, transferible e interpretable |
| **k por prueba nula** | Mayor ventaja sobre el nulo entre los k que superan su p95 | Evita "encontrar" grupos en ruido; siluetas de 0,2–0,3 son normales en datos clínicos |
| **Replicación** | Etiquetas transferidas (medoide más cercano) frente a etiquetas nativas (PAM propio), con emparejamiento húngaro, ARI y Jaccard con IC bootstrap | Lo que debe replicarse es el **perfil**, no el tamaño del grupo |
| **Techo de comparación** | En la cohorte de descubrimiento, reasignar ocultando las variables que la otra no recoge | Saber de antemano cuánto podría replicarse como máximo |
| **Sensibilidad** | k = 3 fijo; un solo dominio; sin cambios; cumplimentación ≥ 99 % | Comprobar que la conclusión no depende de una decisión discutible |

---

## 6 · Resultados del fenotipado: qué sobrevive y qué no

### 6.1 Tres estrategias y su pasaporte

| | **nb 03 · CIBERESUCICOVID** | **nb 04 · Lleida multidominio** | **nb 05 · Las 3 cohortes, variables comunes** |
|---|---|---|---|
| Definición | DLCO, FVC + fatiga, resolución | DLCO, FVC, PM6M, mMRC, HADS-A, HADS-D | DLCO, FVC, FEV1 |
| N (H3 / H6 / H12) | 913 / 799 / 566 | 436 / 368 / 298 | **1.574 / 1.304 / 934** |
| ¿Estructura frente al nulo? | Sí, modesta (+0,06 a +0,14) | **No en H3**; mínima en H6 y H12 | **Sí, clara en H3** (silueta 0,36 frente a 0,11) |
| Estabilidad | Muy alta en H3 y H6 (≥ 0,98) | Patrón (0,52–0,78) | **Alta en H3** (0,92) |
| ¿Viaja? | **No** (ARI 0,17–0,29; techo ≈ 0) | **No**, ni en TENACITY con las mismas variables (ARI 0,15–0,31) | **Sí en H3**: ARI 0,50–0,73 dejando fuera cada cohorte |
| Veredicto | Fenotipos internos de CIBERESUCICOVID | Mejor como gradiente que como tipos | **Fenotipo principal del proyecto** |

### 6.2 nb 03 · CIBERESUCICOVID: fenotipos sintomáticos (no viajan)
- **Tres grupos estables:** **resolución completa**, **persistente con fatiga** y **persistente sin fatiga** (este último con la peor función, estancias más largas y más traqueotomías).
- **Los definen los síntomas, no la función:** con solo DLCO y FVC se reconocen al azar (ARI ≈ 0). Como Lleida y TENACITY solo comparten DLCO y FVC, **no pueden replicarse**.
- **La regla R14 bajó la ventaja frente al nulo** de +0,25 a +0,06 en H3. Parte de la "estructura" inicial era un artefacto de datos ausentes. Lo detectamos y lo corregimos.
- **No predicen el año:** DLCO baja (p = 0,44), reingreso (p = 0,83).

### 6.3 nb 04 · Lleida multidominio: los dominios son independientes
- **Cada dominio por separado sí forma grupos** (sobre todo el ánimo y la función). Al combinarlos, la estructura se diluye: una DLCO baja no implica ansiedad, ni al revés.
- **Hay un perfil "afectación multidominio persistente"** (peor función y marcha, disnea, HADS en rango de caso), pero es pequeño e inestable.
- **Lo valioso:** el perfil a 3 meses **ordena por gravedad** y predice la DLCO baja al año en las tres cohortes (Lleida 35 % → 79 %, p = 0,0003; CIBERESUCICOVID 58 % → 85 %, p = 0,005; TENACITY p = 0,035).

### 6.4 nb 05 · Las tres cohortes juntas: el fenotipo que sobrevive
- **H3, k = 2:**
  - **F1 "función conservada"** (821 pacientes): DLCO 79 %, FVC 96 %, FEV1 99 %;
  - **F2 "afectación funcional"** (753): DLCO 65 %, FVC 73 %, FEV1 76 %.
- **No separa cohortes:** V de Cramér 0,12. El perfil de cada grupo es casi idéntico en las tres (F2: DLCO 64 % y FVC 73 % tanto en CIBERESUCICOVID como en Lleida).
- **Viaja:** aprendido con dos cohortes y comprobado en la tercera, ARI 0,50 (CIBERESUCICOVID fuera), 0,61 (Lleida fuera) y 0,73 (TENACITY fuera).
- **Pronóstico:** DLCO < 80 % al año en el 47 % de F1 frente al 66 % de F2 (p < 0,0001). No anticipa ansiedad, disnea ni reingresos: es un fenotipo respiratorio.
- **¿Por qué solo 2 grupos?** Con 3 variables muy correlacionadas, los pacientes forman un continuo de gravedad. El corte en dos es el único robusto: con k = 3 la ventaja baja a +0,17 y la estabilidad a 0,68.
- **H6 y H12 son más débiles** (H12 viaja peor: ARI 0,05–0,34).

**Conclusión del fenotipado:** con lo que las cohortes miden igual, la distinción **reproducible** es *función conservada frente a afectación funcional* a los 3 meses. Los matices sintomáticos y emocionales existen, pero dependen de la cohorte o de cómo se pregunta.

---

## 7 · Modelos predictivos: ¿podemos anticipar el fenotipo o la evolución?

### 7.1 Diseño común
- **Sin fuga de información:** los predictores son anteriores a lo que se predice. El alta y la fase aguda predicen el estado a 3 o 12 meses; la función a 3 meses solo se usa para predecir el año.
- **Partición 80/20 estratificada**, más la validación que pide el reto: **dejando fuera cada cohorte** (nb 06–07) o **cada grupo de centros** (nb 08–09).
- **Métricas:**

  | Métrica | Qué mide |
  |---|---|
  | ROC-AUC | Discriminación, sin depender del umbral: 0,5 = azar, 1 = perfecto |
  | PR-AUC | Precisión frente a sensibilidad |
  | Brier y curva de calibración | Si las probabilidades predichas son fiables |
  | Sensibilidad y especificidad | Con el umbral de Youden elegido **en train** |
- **Modelo principal, EBM (Explainable Boosting Machine):** aditivo, con una curva por variable que se puede mirar. Es tan interpretable como una logística, pero capta relaciones no lineales. Se compara siempre con una **regresión logística**.

### 7.2 Los modelos

| nb | Cohortes | Diana | Predictores | AUC test | Validación externa | Lectura |
|---|---|---|---|---|---|---|
| 06 | 3 | Fenotipo funcional H3 (F2 frente a F1) | 10 del alta comunes | **0,634** [0,57–0,70] | 0,59–0,62 (cohorte no vista) | Modesto pero **transferible**; = logística (0,633) |
| 07 | 3 | Ídem | Ídem, **red neuronal** (MLP 64-32, 60 configuraciones probadas) | 0,626 | 0,62–0,66 | **No mejora al EBM** (Δ −0,008 [−0,04 a +0,03]): el límite son los datos, no el modelo |
| 08 | CIBERESUCICOVID | Fenotipo **sintomático** (3 clases) | 10 alta + **20 de fase aguda** | 0,575 (macro) | 0,58 (centros no vistos) | ≈ azar; la fase aguda no aporta nada |
| 09-A | CIBERESUCICOVID | Fenotipo funcional H3 | 10 alta + 20 aguda | 0,648 | 0,63 | La fase aguda suma +0,06 en test (p = 0,08), pero no en centros no vistos |
| 09-B | CIBERESUCICOVID | **DLCO < 80 % a 12 meses** | 10 alta + 20 aguda | 0,603 (logística 0,696) | 0,55–0,63 | Con 378 pacientes de train, el EBM se sobreajusta |
| **09-B+** | CIBERESUCICOVID | **DLCO < 80 % a 12 meses** | **+ DLCO, FVC y FEV1 a 3 meses** | **0,714** [0,59–0,82] | **0,66** | **Única mejora significativa** (+0,11, p = 0,016) |

### 7.3 Qué aprenden los modelos (interpretabilidad)
- **Estancia hospitalaria:** es el predictor principal en todos. El riesgo sube casi linealmente con los días de ingreso; en la red, P(afectación) pasa de 0,40 a 0,70 entre 7 y 85 días.
- **Comorbilidad:** EPOC, enfermedad renal, diabetes, hipertensión y tabaquismo activo aumentan el riesgo.
- **Fase aguda (nb 09):** marcadores de inflamación (PCR, leucocitos, linfocitos, dímero D), SOFA, días de UCI y de ventilación, PaO2/FiO2.
- **DLCO a 3 meses:** domina la predicción del año, con una importancia el doble que la del siguiente término.
- **Dos efectos que NO hay que leer como causales:**
  - **Más jóvenes, más riesgo.** Probablemente es un sesgo de selección: los mayores con espirometría de seguimiento son los más sanos.
  - **"Sin UCI", más riesgo.** Es un indicador de cohorte: casi todos los pacientes sin UCI son de Lleida, que tiene más afectación.

### 7.4 Qué modelo nos quedamos
- **Al alta, para cualquier hospital: el EBM del nb 06.** Usa variables comunes, está validado en cohortes no vistas y es interpretable. Sirve para **estratificar** a quién priorizar en la primera visita, no para diagnosticar.
- **En la primera visita: el EBM 09-B+.** Es el mejor resultado, aunque solo se ha validado dentro de CIBERESUCICOVID. **Siguiente paso:** validarlo en Lleida y TENACITY, que también tienen DLCO a 3 y 12 meses.
- **Descartados:** la red neuronal (no gana y no se puede interpretar) y el modelo sintomático (≈ azar).

---

## 8 · Mensaje clínico

1. **Tras una neumonía grave, la secuela que se reproduce entre hospitales es la funcional respiratoria.** Hay un grupo con función conservada y otro con afectación mixta de difusión y volumen (≈ la mitad de los pacientes a 3 meses).
2. **Al alta solo se puede estratificar de forma grosera** (estancia larga + EPOC, enfermedad renal o diabetes, más riesgo). Ningún algoritmo supera un AUC de ≈ 0,65 con esos datos.
3. **La primera espirometría con DLCO a los 3 meses es la decisión clave:** anticipa la difusión al año mucho mejor que todo lo que se sabe al alta.
4. **Los síntomas (fatiga) y el ánimo siguen su propio curso.** No se predicen por la gravedad del ingreso y requieren su propia evaluación.

---

## 9 · Limitaciones (las decimos nosotros antes que el jurado)

- **Pocas variables comparables:** 13 comunes a las cuatro cohortes. Por eso el fenotipo replicable es solo respiratorio.
- **Abandono informativo:** los horizontes tardíos sobrerrepresentan a quien sigue en seguimiento (que suele estar peor).
- **Muestras pequeñas** en TENACITY (H6/H12 ≈ 60) y en el modelo 09-B (95 pacientes de test): intervalos anchos.
- **Selección por centro:** en CIBERESUCICOVID, la DLCO solo se mide en algunos hospitales.
- **El fenotipo se deriva de un clustering** y hereda su incertidumbre.
- **No se imputa:** quien tiene pocas medidas aporta menos información. Lo preferimos a inventar datos.
- **Pendiente de validación clínica:**
  - nombres de los fenotipos;
  - "TAC completo sin lesión marcada = sin lesión";
  - asignación de los pacientes compartidos;
  - 3 exitus con seguimiento;
  - 37 % de fumadores activos en TENACITY.

---

## 10 · Siguientes pasos
1. **Validar el modelo de la primera visita** (alta + función a 3 meses → DLCO al año) en Lleida y TENACITY, con las variables comunes.
2. **Ampliar la definición multidominio** con calidad de vida (SF-12), sueño (Epworth) y cognición (MoCA, BC-CCI), aún sin limpiar.
3. **Corregir el abandono informativo** (pesos IPW) en las trayectorias.
4. **Herramienta para consulta:** introducir unos pocos datos y obtener el fenotipo, el riesgo a 12 meses y la ficha.

---

## 11 · Reproducibilidad y uso de IA

### Mapa del repositorio
| Notebook | Contenido |
|---|---|
| `01_analisis_exploratorio` | Exploración de las 4 cohortes |
| `02_limpieza` | Reglas R01–R14, flujo de pacientes, tablas limpias |
| `03_fenotipado_ciberes` | Fenotipos sintomáticos en CIBERESUCICOVID |
| `04_fenotipado_lleida` | Fenotipos multidominio en Lleida |
| `05_fenotipado_comun` | **Fenotipo común a las 3 cohortes (principal)** |
| `06_modelo_ebm_h3` | EBM al alta, 3 cohortes |
| `07_modelo_red_h3` | Red neuronal (comparación) |
| `08_modelo_ebm_ciberes` | EBM con fase aguda, diana sintomática |
| `09_modelo_ebm_ciberes_funcional` | EBM con fase aguda, dianas funcionales y función a 3 meses |

- **Código:** `src/` (carga, limpieza, fenotipado, pasaporte, privacidad) con tests en `tests/`. Cada notebook se genera desde `scripts/` y se ejecuta de principio a fin.
- **Parámetros:** todo en `config.yaml`, con semilla fija (2026).
- **Decisiones:** `docs/registro_decisiones.md`, con una casilla de validación clínica por regla.

### Declaración de uso de IA
- **Herramienta:** Claude (Anthropic) a través de Claude Code.
- **En qué ayudó:** escribir el código, ejecutar los análisis y redactar la documentación.
- **Lo que decidió el equipo:**
  - la estrategia (cohortes, horizontes, dianas, qué fenotipo usar);
  - la elección final de modelos;
  - la validación de las decisiones clínicas.
- **Cómo evitamos conclusiones a medida:** los criterios de validez se fijaron antes de ver los resultados, y los resúmenes de cada notebook se redactaron **después** de ejecutarlo.

---

## Anexo A · Preguntas probables del jurado

**¿Por qué no usasteis todas las variables?**
Porque el 90 % de la tabla son vacíos estructurales: cada cohorte usó un cuaderno distinto. Al agrupar con variables que solo tiene una cohorte, el clustering separa cohortes, no pacientes. Imputarlas sería inventar datos. Usamos las variables ricas para **describir** los fenotipos en cada cohorte.

**¿Por qué solo 2 fenotipos?**
Lo decide la prueba frente al azar, no nosotros. Con la función pulmonar común, los pacientes forman un continuo de gravedad. Dos grupos es el único corte estable (0,92) y reproducible entre cohortes. Con k = 3 la estabilidad cae a 0,68.

**Un AUC de 0,63 es bajo. ¿No ha fallado el modelo?**
Lo comprobamos con tres familias de modelos (logística, EBM y red neuronal) y todas dan 0,63 ± 0,01. El límite es la **información disponible al alta**, no el algoritmo. Al añadir la DLCO a 3 meses sube a 0,71. Eso también es un resultado: dice cuándo tiene sentido predecir.

**¿Por qué EBM y no XGBoost o deep learning?**
Rinde igual que modelos más complejos (lo comprobamos con la red neuronal) y es interpretable: cada variable tiene su curva, y eso es el 20 % de la nota y lo que necesita un médico.

**¿Cómo sabéis que los fenotipos no son ruido?**
Por cuatro comprobaciones:
- prueba nula con 100 permutaciones;
- estabilidad bootstrap;
- validación dejando fuera cohortes y centros;
- sensibilidad a decisiones discutibles.

Además publicamos lo que **no** pasó el pasaporte.

**¿Por qué no usar z-scores para normalizar entre cohortes?**
Porque borrarían diferencias reales: TENACITY tiene la DLCO más alta incluso en la misma ventana temporal. Usamos rangos clínicos fijos e idénticos para todas.

**¿Cómo tratasteis a los pacientes que están en dos registros?**
Cada paciente cuenta en una sola cohorte de análisis, y las pruebas duplicadas (953) se cuentan una vez. Sin esto, la "replicación" usaría parcialmente a los mismos pacientes.

**¿Y los datos ausentes?**
Distinguimos tres situaciones:
- **"No recogido"** (estructural): nunca se imputa.
- **"Falta"**: Gower compara solo las variables disponibles y el EBM trata el vacío como una categoría.
- **"Pregunta condicionada"** (fatiga cuando la resolución es total): se **deduce**, no se imputa.

**¿Hay fuga de información en los modelos?**
No. Los predictores son siempre anteriores al momento de la diana, y la función a 3 meses nunca se usa para predecir el fenotipo a 3 meses (que se define con ella). Los umbrales se eligen en train y los tests comprueban que no se usan datos posteriores al horizonte.

**¿Qué aporta esto a la consulta?**
Una regla práctica: estratificar al alta con un modelo sencillo e interpretable, y **decidir el seguimiento con la DLCO de los 3 meses**, que es lo que mejor anticipa la difusión al año.

## Anexo B · Glosario rápido
- **DLCO:** capacidad de difusión del monóxido de carbono; cuánto oxígeno pasa del pulmón a la sangre. Alterada si es < 80 % del predicho.
- **FVC / FEV1:** capacidad vital forzada y volumen espirado en el primer segundo (espirometría).
- **Gower:** distancia que mezcla variables numéricas y categóricas, y admite datos ausentes.
- **PAM / medoide:** clustering cuyo "centro" es un paciente real.
- **Silueta:** cuánto se parece cada paciente a su grupo frente al grupo vecino (de −1 a 1).
- **Jaccard / ARI:** coincidencia entre dos agrupaciones (1 = idénticas; ARI 0 = azar).
- **ROC-AUC:** probabilidad de que el modelo puntúe más alto a un caso positivo que a uno negativo (0,5 = azar).
- **Sensibilidad (= recall):** de los que tienen la condición, cuántos detecta el modelo.
- **Especificidad:** de los que no la tienen, cuántos clasifica bien.
- **Precisión:** de los que el modelo marca como positivos, cuántos lo son.
- **F1:** media armónica de precisión y sensibilidad.
- **EBM:** Explainable Boosting Machine, un modelo aditivo de árboles con una curva interpretable por variable.

## Anexo C · Figuras recomendadas para las diapositivas
| Mensaje | Figura (`outputs/figuras/`) |
|---|---|
| El problema de los datos | `03_cobertura_dominios.png`, `03_mapa_ausencias_nucleo.png` |
| Limpieza y flujo de pacientes | `02_flujo_consort.png` |
| El fenotipo que sobrevive | `05_comun_perfiles.png`, `05_comun_validacion_loco.png`, `05_comun_eleccion_k.png` |
| Lo que no se replica | `03_replicacion_ari.png`, `04_lleida_replicacion_ari.png` |
| Modelo al alta | `06_ebm_roc_pr.png`, `06_ebm_formas.png` |
| El límite está en los datos | `07_red_roc_pr.png` |
| El mensaje clave: la visita de 3 meses | `09_ebm_func_dlco_12m_roc_pr_calibracion.png`, `09_ebm_func_comparacion_global.png` |
