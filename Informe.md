# Reto 2 · Fenotipado digital de secuelas post-infecciosas — Informe

Documento completo del proyecto: explica y justifica los datos, la limpieza, el fenotipado, las trayectorias y los modelos predictivos. Todas las cifras salen de los notebooks ejecutados; entre paréntesis se indica el notebook de origen. El resumen ejecutable para el jurado es `main.ipynb`.

---

## 0 · Resumen en una página

**La pregunta del reto no es si existen subgrupos (siempre salen), sino si sobreviven al cambio de cohorte.** Por eso todo el proyecto gira alrededor de una idea: un fenotipo solo cuenta si pasa un **"pasaporte"** de controles. Tiene que:
- tener más estructura que unos datos al azar;
- ser estable;
- reproducirse en una cohorte que no ha visto;
- evolucionar distinto;
- tener sentido clínico.

**Tres hallazgos:**

1. **Hay un fenotipo que sí sobrevive al cambio de cohorte (nb 04).** A los 3 meses, con la función pulmonar que las tres cohortes miden igual (DLCO, FVC y FEV1), salen dos grupos:
   - **F1 · "función conservada"**: DLCO 79 %, FVC 96 %;
   - **F2 · "afectación funcional"**: DLCO 65 %, FVC 73 %.

   Es estable (Jaccard 0,92), no depende de la cohorte (V de Cramér 0,12) y se reproduce al aprender con dos cohortes y probar en la tercera (ARI 0,50–0,73). En cambio, los fenotipos de síntomas de CIBERESUCICOVID no se reproducen fuera de su cohorte (nb 03).
2. **Los dos fenotipos evolucionan distinto (nb 05).**
   - Ambos mejoran, pero la distancia entre ellos solo se cierra de 14,0 a **11,5 puntos de DLCO al año**, y F2 sigue de media por debajo del 80 %.
   - Se repite en CIBERESUCICOVID (−11,1) y en Lleida (−11,0).
   - Resiste la corrección del abandono (IPW: −10,4) y la regresión a la media (−10,9). En Lleida se mantiene hasta los 4 años.
3. **El momento para predecir es la visita de los 3 meses, no el alta.**
   - Con los datos del alta, cualquier modelo se queda en un AUC ≈ 0,63; con la fase aguda, ≈ 0,65 (nb 06–07).
   - A los 3 meses, **la DLCO sola** predice quién seguirá alterado al año: AUC 0,87 en todos los pacientes y 0,79 entre los alterados, dejando fuera cada cohorte. **Ningún modelo más complejo la mejora** (nb 08).

**Traducción para la consulta:**
- La espirometría con DLCO de los 3 meses decide el seguimiento.
- F1 se normaliza hacia los 6 meses.
- F2 necesita seguimiento más allá del año, y dentro de F2 la DLCO indica quién se recuperará: se recupera el 7 % con DLCO < 60 %, el 24 % con 60–69 % y el 62 % con 70–79 %.

---

## 1 · El reto y cómo lo planteamos

- **Objetivo del reto:** un notebook reproducible que identifique fenotipos clínicos de las secuelas tras una infección vírica grave. Deben ser **reproducibles entre cohortes**, interpretables y útiles en consulta.
- **Criterios de evaluación:** justificación (20 %), rigor científico (30 %), interpretabilidad (20 %) y presentación (30 %).
- **Nuestro criterio central:** fijamos qué tiene que cumplir un fenotipo para considerarlo real (`config.yaml`):

| "Sello" del pasaporte | Cómo se mide | Umbral | Resultado (fenotipo H3, nb 04–05) |
|---|---|---|---|
| Hay estructura | Silueta real frente a la de 100 conjuntos con las columnas permutadas | Supera el percentil 95 del nulo | ✅ 0,36 frente a 0,11 |
| Es estable | Jaccard bootstrap por clúster (200 remuestreos) | ≥ 0,75 estable; 0,5–0,75 patrón; < 0,5 se disuelve | ✅ 0,92 |
| Viaja | Transferencia al medoide frente a clustering propio en la cohorte excluida (ARI, Jaccard con emparejamiento húngaro) | ARI con IC 95 % que excluye 0 | ✅ ARI 0,50–0,73 |
| Evoluciona distinto | Modelo mixto DLCO ~ fenotipo × log(tiempo) | Diferencia a 12 meses con IC que excluye 0 en el conjunto y en cada cohorte grande | ✅ −11,5 [−13,6 a −9,3] |
| Se reconoce con pocas variables | Árbol de 1–2 preguntas validado dejando fuera cada cohorte (acierto, AUC-PR, calibración) | Regla corta que reproduzca el fenotipo en cohortes no vistas | ✅ «¿FVC a 3 m ≤ 83 %?» acierta el 95–97 % (`main`, apartado 2.2.6) |
| Sentido clínico | Fichas con variables que **no** entran en la definición | Lo valora el equipo, no la IA | Pendiente de validación del equipo |

Un fenotipo que no cumple se **presenta igualmente**, como no replicado. No se ajusta nada para que "salga".

### Cómo llegamos aquí

**La idea inicial era la prevención.** No queríamos limitarnos a describir qué grupos de pacientes existen: queríamos saber qué permite predecir, lo antes posible, cómo va a evolucionar un paciente. Cuanto más tarde se recogen los datos, más precisa es la identificación del fenotipo, pero más tarde puede intervenir el médico. Por eso planteamos un estudio en tres momentos (3, 6 y 12 meses tras el alta).

**El diseño del clustering evolucionó en cuatro pasos:**
1. **Capas A y B:**
   - **A (respiratoria):** descubrir en CIBERESUCICOVID y replicar en Lleida y TENACITY.
   - **B (multidominio, con ánimo, disnea y esfuerzo):** descubrir en Lleida.

   Con solo DLCO y FVC, la capa A reflejaba casi únicamente el nivel de DLCO; la capa B tenía solo 710 pacientes.
2. **Solo CIBERESUCICOVID, con síntomas y función (nb 03):** tres grupos muy estables, pero definidos por los síntomas, que las otras cohortes no recogen igual. **No viajan.**
3. **Solo Lleida, multidominio:** cada dominio por separado tiene estructura, pero juntos se diluyen. Los dominios son casi independientes (\|ρ\| ≤ 0,23, nb 01), y los grupos no se reproducían ni en TENACITY. Fue una exploración descartada y no se incluye en la entrega.
4. **Final: las tres cohortes juntas con las variables que miden igual (nb 04).** Es el fenotipo que pasa el pasaporte. La riqueza de cada cohorte (síntomas, ánimo, TAC, fase aguda) se usa para **describir** los fenotipos, no para definirlos.

Después, las trayectorias (nb 05) comprobaron que el fenotipo tiene sentido en el tiempo. La predicción se trasladó del alta (nb 06–07) a la visita de los 3 meses (nb 08).

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
| Quién tiene DLCO depende del **centro** | 36 de 76 centros no la miden nunca | Validar dejando fuera cada centro o cohorte |
| Las cohortes miden en momentos distintos | Virgen del Rocío, mediana de 44 días tras el alta; el resto, 86–120 días | Usar el tiempo real (fechas) y ventanas comunes; Virgen del Rocío solo como complemento |
| Aun en la misma ventana, TENACITY es distinta | DLCO ~10 puntos más alta a los 75–105 días | Usar **umbrales clínicos y rangos fijos**, no z-scores por cohorte (borrarían diferencias reales) |
| Los dominios están poco correlacionados | Pulmón, esfuerzo, disnea y ánimo: \|ρ\| ≤ 0,23 | Pesar los dominios por igual; esperar fenotipos multidominio difusos |
| El abandono es informativo | Quien no vuelve al año partía 6–7 puntos de DLCO más alto | Corregirlo en las trayectorias (nb 05) |
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

La justificación detallada de cada regla, con su casilla de validación clínica, está en `docs/registro_decisiones.md`.

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
| Con alguna medida de seguimiento válida | 4.494 |
| Con DLCO o FVC en la ventana de definición (60–210 días) | 2.114 |

---

## 5 · Cómo hacemos el fenotipado (método común a nb 03 y 04)

| Decisión | Qué hicimos | Justificación |
|---|---|---|
| **Horizontes acumulados** | H3: visita de ~3 meses. H6: 3 + 6 meses, con el cambio entre visitas. H12: 3 + 6 + 12 meses | Responde a "¿qué fenotipo tiene el paciente con lo que sabemos hasta ese momento?" sin mirar el futuro (comprobado con un `assert` en el código) |
| **Ancla** | Para entrar en un horizonte hay que tener medidas en su última visita | Que el fenotipo describa el estado en ese momento, no solo el pasado |
| **Variables de definición ≠ de descripción** | Se agrupa con el estado de seguimiento; el ingreso, la imagen y los eventos solo describen | Separar el fenotipo de sus causas y consecuencias |
| **Distancia de Gower** | Admite variables mixtas y datos ausentes por pares, sin imputar | Los datos son mixtos y con muchos huecos estructurales |
| **Rangos fijos (P2,5–P97,5) idénticos en todas las cohortes** | p. ej., DLCO 40–110, FVC 55–125 | Con z-scores por cohorte se borrarían diferencias reales. Con el rango plausible (20–160), 20 puntos de DLCO pesaban 0,14 frente a 1 de un síntoma distinto (lo corregimos) |
| **Peso igual por dominio** | p. ej., difusión = espirometría | Que un dominio con más columnas (FVC y FEV1) no domine |
| **PAM (k-medoides)** | FasterPAM | Funciona con cualquier distancia; el medoide es un paciente real, transferible e interpretable |
| **k por prueba nula** | Mayor ventaja sobre el nulo entre los k que superan su p95 | Evita "encontrar" grupos en ruido; siluetas de 0,2–0,3 son normales en datos clínicos |
| **Replicación** | Etiquetas transferidas (medoide más cercano) frente a etiquetas nativas (PAM propio), con emparejamiento húngaro, ARI y Jaccard con IC bootstrap | Lo que debe replicarse es el **perfil**, no el tamaño del grupo |
| **Sensibilidad** | k = 3 fijo; un solo dominio; sin cambios; cumplimentación ≥ 99 % | Comprobar que la conclusión no depende de una decisión discutible |

---

## 6 · Resultados del fenotipado: qué sobrevive y qué no

| | **nb 03 · CIBERESUCICOVID** | **nb 04 · Las 3 cohortes, variables comunes** |
|---|---|---|
| Definición | DLCO, FVC + fatiga, resolución | DLCO, FVC, FEV1 |
| N (H3 / H6 / H12) | 913 / 799 / 566 | **1.574 / 1.304 / 934** |
| ¿Estructura frente al nulo? | Sí, modesta (+0,06 a +0,14) | **Sí, clara en H3** (silueta 0,36 frente a 0,11) |
| Estabilidad | Muy alta en H3 y H6 (≥ 0,98) | **Alta en H3** (0,92) |
| ¿Viaja? | **No** (ARI 0,17–0,29; con solo DLCO y FVC, ≈ 0) | **Sí en H3**: ARI 0,50–0,73 dejando fuera cada cohorte |
| Veredicto | Fenotipos internos de CIBERESUCICOVID | **Fenotipo principal del proyecto** |

### 6.1 nb 03 · CIBERESUCICOVID: fenotipos sintomáticos (no viajan)
- **Tres grupos estables:** **resolución completa**, **persistente con fatiga** y **persistente sin fatiga**. El último tiene la peor función, las estancias más largas y más traqueotomías.
- **Los definen los síntomas, no la función:** con solo DLCO y FVC se reconocen al azar (ARI ≈ 0). Como Lleida y TENACITY solo comparten DLCO y FVC, **no pueden replicarse**.
- **La regla R14 bajó la ventaja frente al nulo** de +0,25 a +0,06 en H3. Parte de la "estructura" inicial era un artefacto de datos ausentes. Lo detectamos y lo corregimos.
- **No predicen el año:** DLCO baja (p = 0,44), reingreso (p = 0,83).

### 6.2 nb 04 · Las tres cohortes juntas: el fenotipo que sobrevive
- **H3, k = 2:**
  - **F1 "función conservada"** (821 pacientes): DLCO 79 %, FVC 96 %, FEV1 99 %;
  - **F2 "afectación funcional"** (753): DLCO 65 %, FVC 73 %, FEV1 76 %.
- **No separa cohortes:** V de Cramér 0,12. El perfil de cada grupo es casi idéntico en las tres (F2: DLCO 64 % y FVC 73 % tanto en CIBERESUCICOVID como en Lleida).
- **Viaja:** aprendido con dos cohortes y comprobado en la tercera, ARI 0,50 (CIBERESUCICOVID fuera), 0,61 (Lleida fuera) y 0,73 (TENACITY fuera).
- **Qué lo acompaña a los 3 meses** (variables que no entran en el clustering; `main`, apartado 2.2.7, `src/fichas.py`): las diferencias son pequeñas (SMD 0,2–0,5) pero todas en el mismo sentido. F2 tiene más lesiones fibróticas en el TAC (27 % frente a 12 %; con las reticulares, 76 % frente a 54 %), más tractos fibrosos en CIBERESUCICOVID (30 % frente a 16 %), más disnea mMRC ≥ 2 (27 % frente a 13 %), estancias más largas (26 frente a 21 días) y más traqueostomías (33 % frente a 22 %). No hay diferencias en edad, HADS, astenia, reingresos ni urgencias. El TAC solo se hace a una parte de los pacientes, así que esos % son sobre quien lo tiene.
- **Pronóstico:** DLCO < 80 % al año en el 47 % de F1 frente al 66 % de F2 (p < 0,0001). No anticipa ansiedad, disnea ni reingresos: es un fenotipo respiratorio.
- **¿Por qué solo 2 grupos?** Con 3 variables muy correlacionadas, los pacientes forman un continuo de gravedad. El corte en dos es el único robusto: con k = 3, la ventaja baja a +0,17 y la estabilidad a 0,68.
- **Quién sostiene la partición:** la espirometría. La versión "solo espirometría" coincide casi del todo con la principal (ARI 0,97), y en la práctica el fenotipo equivale a un corte de FVC en torno al 83 %. Es coherente con el umbral clínico del 80 %: el clustering lo redescubre sin habérselo dado.
- **H6 y H12 son más débiles:** H12 viaja peor (ARI 0,05–0,34).

---

## 7 · Trayectorias (nb 05): ¿los fenotipos evolucionan distinto?

**Diseño:**
- **Datos:** todas las medidas válidas de DLCO entre 30 y 480 días tras el alta. Son 2.512 medidas de 1.429 pacientes; 763 tienen al menos dos.
- **Modelo mixto:** `DLCO ~ fenotipo × log(meses/3) + cohorte`, con intercepto y pendiente aleatorios por paciente (`statsmodels`). Con el tiempo centrado en 3 meses, el intercepto es la DLCO a los 3 meses.

| | 3 meses | 6 meses | 12 meses | Ganancia 3 → 12 |
|---|---|---|---|---|
| F1 · función conservada | 79,4 % | 82,7 % | 86,1 % | +6,7 |
| F2 · afectación funcional | 65,4 % | 70,0 % | 74,6 % | +9,2 |
| **Diferencia F2 − F1** | −14,0 | −12,7 | **−11,5** [−13,6 a −9,3] | +2,5 [0,7 a 4,3] |

- **Se replica en cada cohorte:** −11,1 en CIBERESUCICOVID, −11,0 en Lleida y −16,5 en TENACITY (N pequeña). **Sello 3 superado.**
- **Abandono informativo:**
  - Quien solo tiene DLCO a 3 meses partía de 76,6 %; quien hizo las tres visitas, de 67,7 %.
  - Las medias brutas de los que vuelven infravaloran la recuperación: la ganancia de F1 es +3,8 sin pesos, +5,6 con IPW y +6,7 en el modelo mixto.
  - La diferencia entre fenotipos apenas cambia: −10,0 a −11,5 según el método.
  - En TENACITY, 39 pacientes aún no habían llegado a la visita anual: no se cuentan como abandono.
- **Regresión a la media:** sin la medida de 3 meses (que definió el fenotipo), la diferencia a 12 meses sigue en −10,9. Lo que sí se debía en parte a ese efecto es la recuperación "extra" de F2.
- **Transiciones:** asignando cada visita al medoide fijo de H3:
  - al año, el 76 % sigue en su fenotipo;
  - un 38 % de F2 pasa a F1 y solo un 8 % de F1 pasa a F2.

  Los medoides no tienen DLCO (el de F2 solo tiene FVC), así que la reasignación refleja sobre todo la FVC.
- **Más allá del año (solo Lleida, un centro):** la diferencia se mantiene hasta los 4 años (−10,9 a 48 meses). F2 se acerca al 80 % hacia los 2–3 años.

---

## 8 · Modelos predictivos: ¿podemos anticipar la evolución?

### 8.1 Diseño común
- **Sin fuga de información:** los predictores son anteriores a lo que se predice. El alta y la fase aguda predicen el estado a 3 o 12 meses; la función a 3 meses solo se usa para predecir el año.
- **Validación:** dejando fuera cada cohorte (nb 06 y 08) o cada grupo de centros (nb 07).
- **Métricas:**

  | Métrica | Qué mide |
  |---|---|
  | ROC-AUC | Discriminación, sin depender del umbral: 0,5 = azar, 1 = perfecto |
  | PR-AUC | Precisión frente a sensibilidad |
  | Brier, curva y pendiente de calibración | Si las probabilidades predichas son fiables |
  | Curva de decisión | Si usar el modelo mejora la decisión frente a "seguir a todos" |

### 8.2 Los modelos

| nb | Cohortes | Diana | Predictores | AUC | Validación externa | Lectura |
|---|---|---|---|---|---|---|
| 06 | 3 | Fenotipo H3 (F2 frente a F1) | 10 del alta comunes (EBM) | 0,634 [0,57–0,70] | 0,59–0,62 (cohorte no vista) | Modesto pero transferible; igual que la logística (0,633) |
| 07-A | CIBERESUCICOVID | Fenotipo H3 | 10 alta + 20 de fase aguda (EBM) | 0,648 | 0,63 (centros no vistos) | La fase aguda suma poco y no de forma robusta entre centros |
| 07-B | CIBERESUCICOVID | DLCO < 80 % a 12 meses | alta + aguda (+ función a 3 m) | 0,603 (0,714 con función a 3 m) | 0,55–0,66 | Con 378 pacientes de entrenamiento, el EBM se sobreajusta |
| **08** | **3** | **Entre los alterados a 3 m: ¿sigue alterado al año?** | **DLCO a 3 m sola** | **0,79** [0,74–0,84] | **0,79 / 0,84 / 0,83 por cohorte no vista** | **Ni la espirometría ni el alta la mejoran** |

**Notebook 08 en detalle:**
- **Población:** 361 pacientes con DLCO < 80 % a 3 meses y DLCO al año; el 70 % sigue alterado.
- **Contrato fijado antes de ajustar:** un modelo más complejo solo se adopta si mejora a la DLCO sola con un IC del ΔAUC que excluya 0.
- **Ninguno lo consigue:**
  - función a 3 m: +0,003 [−0,006 a +0,011];
  - árbol: −0,030;
  - logística L1 con función + alta: −0,016 [−0,034 a −0,002].
- **La ordenación viaja, el nivel absoluto no:** CIBERESUCICOVID persiste 12 puntos más de lo predicho y Lleida 10 menos. Las probabilidades deben recalibrarse localmente.
- **Tabla de bolsillo** (de los alterados a 3 meses, cuántos se recuperan al año):

  | DLCO a 3 meses | Se recupera al año |
  |---|---|
  | < 60 % | 7 % [4–13] |
  | 60–69 % | 24 % [17–33] |
  | 70–79 % | 62 % [53–71] |

### 8.3 Qué aprenden los modelos (interpretabilidad)
- **Al alta (nb 06–07):** el predictor principal es la estancia hospitalaria; luego la comorbilidad (EPOC, enfermedad renal, diabetes, hipertensión, tabaquismo activo) y, con fase aguda, la inflamación (PCR, linfocitos, dímero D), el SOFA y los días de UCI y de ventilación.
- **Dos efectos que NO hay que leer como causales:**
  - "Más jóvenes, más riesgo": probablemente sesgo de selección (los mayores con espirometría de seguimiento son los más sanos).
  - "Sin UCI, más riesgo": es un indicador de cohorte (casi todos los pacientes sin UCI son de Lleida).
- **En la primera visita (nb 08):** la DLCO a 3 meses domina (OR 0,23 por cada 11 puntos). La logística L1 descarta la FVC, la edad, la estancia y las comorbilidades.

### 8.4 Qué modelo nos quedamos
- **Al alta: el EBM del nb 06**, como estratificación grosera (a quién priorizar en la primera visita), no para decidir sobre un paciente.
- **En la visita de los 3 meses: la regla de la DLCO (nb 08)**, presentada como tabla de bolsillo. Es la más simple, está validada en cohortes no vistas y ningún modelo la supera.

---

## 9 · Mensaje clínico

1. **Tras una neumonía grave, la secuela que se reproduce entre hospitales es la funcional respiratoria.** Hay un grupo con función conservada y otro con afectación mixta de difusión y volumen (≈ la mitad de los pacientes a 3 meses).
2. **Los dos grupos mejoran, pero el de afectación no alcanza al otro:** al año sigue alterado de media, y en Lleida tarda 2–3 años en acercarse a la normalidad.
3. **Al alta solo se puede estratificar de forma grosera:** con estancia larga y EPOC, enfermedad renal o diabetes, el riesgo es mayor.
4. **La visita de los 3 meses es la decisión clave:** la DLCO indica quién seguirá alterado al año, mucho mejor que todo lo que se sabe al alta.
5. **Los síntomas (fatiga) y el ánimo siguen su propio curso:** no se mueven con la función pulmonar (nb 01) y requieren su propia evaluación.

---

## 10 · Limitaciones

- **Pocas variables comparables:** 13 comunes a las cuatro cohortes. Por eso el fenotipo replicable es solo respiratorio.
- **Abandono informativo:** se corrige con modelo mixto e IPW (supuesto MAR), pero no se puede descartar un abandono que dependa de la DLCO futura no observada.
- **Muestras pequeñas** en TENACITY (H6/H12 ≈ 60; 24 pacientes en el nb 08) y en el modelo 07-B (95 de test): intervalos anchos.
- **Selección por centro:** en CIBERESUCICOVID, la DLCO solo se mide en algunos hospitales.
- **El fenotipo se deriva de un clustering** y hereda su incertidumbre. Además, sus medoides tienen pruebas ausentes (el de F2 solo tiene FVC).
- **Umbral fijo del 80 %**, no el límite inferior de la normalidad. Cerca del umbral, parte de la "recuperación" puede ser variabilidad de la prueba.
- **No se imputa:** quien tiene pocas medidas aporta menos información. Lo preferimos a inventar datos.
- **Pendiente de validación clínica:**
  - nombres de los fenotipos;
  - "TAC completo sin lesión marcada = sin lesión";
  - asignación de los pacientes compartidos;
  - 3 exitus con seguimiento;
  - 37 % de fumadores activos en TENACITY.

---

## 11 · Siguientes pasos
1. **Medoides completos:** restringir los candidatos a medoide a pacientes con las tres pruebas.
2. **Recalibrar localmente** la regla de la DLCO en cada hospital antes de usar sus probabilidades.
3. **Ampliar la descripción multidominio** con calidad de vida (SF-12), sueño (Epworth) y cognición (MoCA, BC-CCI), aún sin limpiar.
4. **Herramienta para consulta:** introducir la espirometría de los 3 meses y obtener el fenotipo, la trayectoria esperada y el riesgo a 12 meses.

---

## 12 · Reproducibilidad

- **Mapa del repositorio y orden de ejecución:** `README.md`.
- **Código:** `src/` (carga, limpieza, fenotipado, pasaporte, trayectorias, fichas, privacidad); notebooks de detalle en `Workflow/`.
- **Parámetros:** todo en `config.yaml`, con semilla fija (2026).
- **Decisiones:** `docs/registro_decisiones.md`, con una casilla de validación clínica por regla.

---

## Anexo A · Preguntas probables del jurado

**¿Por qué no usasteis todas las variables?**
Porque el 90 % de la tabla son vacíos estructurales: cada cohorte usó un cuaderno distinto. Al agrupar con variables que solo tiene una cohorte, el clustering separa cohortes, no pacientes, e imputarlas sería inventar datos. Usamos las variables ricas para **describir** los fenotipos en cada cohorte.

**¿Por qué solo 2 fenotipos?**
Lo decide la prueba frente al azar, no nosotros. Con la función pulmonar común, los pacientes forman un continuo de gravedad. Dos grupos es el único corte estable (0,92) y reproducible entre cohortes; con k = 3, la estabilidad cae a 0,68.

**¿El fenotipo no es solo un corte de FVC?**
En la práctica, casi: la regla «¿FVC a 3 meses ≤ 83 %?» reproduce el 95–97 % de las etiquetas en cohortes no vistas (`main`, apartado 2.2.6). Lo vemos como una fortaleza:
- el clustering no supervisado **redescubre el umbral clínico** sin habérselo dado;
- el fenotipo se reconoce con una sola prueba;
- su valor está en lo que dice de la evolución: F2 sigue alterado al año.

**¿No es trivial que la DLCO a 3 meses prediga la DLCO al año?**
Sí, en parte, y por eso la pregunta útil es otra: **entre los alterados a 3 meses, ¿quién se recuperará?** Ahí la DLCO sola sigue dando un AUC de 0,79 en cohortes no vistas, y ni la espirometría ni los datos del alta la mejoran.

**Un AUC de 0,63 al alta es bajo. ¿No ha fallado el modelo?**
Lo comprobamos con logística y con EBM, y ambos dan 0,63. Añadir la fase aguda no lo mejora de forma robusta entre centros. El límite es la **información disponible al alta**, no el algoritmo. Eso también es un resultado: dice cuándo tiene sentido predecir.

**¿Por qué EBM?**
Rinde igual que la logística y es interpretable: cada variable tiene su curva. Eso es el 20 % de la nota y lo que necesita un médico. En la primera visita gana la regla más simple de todas.

**¿Cómo sabéis que los fenotipos no son ruido?**
Por cinco comprobaciones:
- prueba nula con 100 permutaciones;
- estabilidad bootstrap;
- validación dejando fuera cohortes y centros;
- sensibilidad a decisiones discutibles;
- trayectorias distintas, replicadas por cohorte.

Además, publicamos lo que **no** pasó el pasaporte.

**¿Y el abandono? Los que vuelven a revisión son los que están peor.**
Es verdad y lo medimos (nb 05). Lo corregimos con un modelo mixto y con pesos IPW. La diferencia entre fenotipos se mantiene (−10,4 a −11,5).

**¿Por qué no usar z-scores para normalizar entre cohortes?**
Porque borrarían diferencias reales: TENACITY tiene la DLCO más alta incluso en la misma ventana temporal. Usamos rangos clínicos fijos e idénticos para todas.

**¿Cómo tratasteis a los pacientes que están en dos registros?**
Cada paciente cuenta en una sola cohorte de análisis, y las pruebas duplicadas (953) se cuentan una vez. Sin esto, la "replicación" usaría parcialmente a los mismos pacientes.

**¿Y los datos ausentes?**
Distinguimos tres situaciones:
- **"No recogido"** (estructural): nunca se imputa.
- **"Falta":** Gower compara solo las variables disponibles; en los modelos se imputa la mediana del entrenamiento o el EBM trata el vacío como una categoría.
- **"Pregunta condicionada"** (fatiga cuando la resolución es total): se **deduce**, no se imputa.

**¿Hay fuga de información en los modelos?**
No:
- los predictores son siempre anteriores al momento de la diana;
- la función a 3 meses nunca se usa para predecir el fenotipo a 3 meses, que se define con ella;
- los umbrales y la regularización se eligen solo con los datos de entrenamiento;
- el código comprueba (`assert`) que no se usan datos posteriores al horizonte.

## Anexo B · Glosario rápido
- **DLCO:** capacidad de difusión del monóxido de carbono; cuánto gas pasa del pulmón a la sangre. Alterada si es < 80 % del predicho.
- **FVC / FEV1:** capacidad vital forzada y volumen espirado en el primer segundo (espirometría).
- **Gower:** distancia que mezcla variables numéricas y categóricas, y admite datos ausentes.
- **PAM / medoide:** clustering cuyo "centro" es un paciente real.
- **Silueta:** cuánto se parece cada paciente a su grupo frente al grupo vecino (de −1 a 1).
- **Jaccard / ARI:** coincidencia entre dos agrupaciones (1 = idénticas; ARI 0 = azar).
- **Modelo mixto:** regresión con un intercepto y una pendiente propios de cada paciente; aprovecha también a quien tiene una sola medida.
- **IPW:** pesos inversos de la probabilidad de seguir en seguimiento; corrigen el abandono que depende de lo observado.
- **Regresión a la media:** quien se selecciona por un valor bajo en una prueba ruidosa tiende a subir en la siguiente medida aunque no cambie nada.
- **ROC-AUC:** probabilidad de que el modelo puntúe más alto a un caso positivo que a uno negativo (0,5 = azar).
- **Calibración:** si las probabilidades predichas coinciden con lo observado.
- **Curva de decisión (beneficio neto):** cuánto mejora la decisión usar el modelo frente a tratar a todos o a nadie.
- **EBM:** Explainable Boosting Machine, un modelo aditivo de árboles con una curva interpretable por variable.

## Anexo C · Figuras recomendadas para las diapositivas
Se generan al ejecutar los notebooks, en `outputs/figuras/`; el prefijo es el número del notebook.

| Mensaje | Figura |
|---|---|
| El problema de los datos | `01_eda_03_cobertura_dominios.png`, `01_eda_03_mapa_ausencias_nucleo.png` |
| Limpieza y flujo de pacientes | `02_flujo_consort.png` |
| Lo que no se replica | `03_replicacion_ari.png` |
| El fenotipo que sobrevive | `04_comun_perfiles.png`, `04_comun_validacion_loco.png`, `04_comun_eleccion_k.png` |
| Evolucionan distinto | `05_tray_dlco_fenotipo.png`, `05_tray_extension_lleida.png`, `05_tray_transiciones.png` |
| Al alta no se puede | `06_ebm_roc_loco.png`, `06_ebm_formas.png` |
| La visita de los 3 meses decide | `08_pv_tramos_todos.png`, `08_pv_auc_loco.png`, `08_pv_interpretacion.png` |
