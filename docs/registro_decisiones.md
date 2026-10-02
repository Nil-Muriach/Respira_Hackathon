# Registro de decisiones — limpieza de datos

Cada regla de `src/limpieza.py` con su justificación y la evidencia del análisis exploratorio (`01_analisis_exploratorio.ipynb`, sección §). Los parámetros están en `config.yaml`. Las cifras de casos afectados se generan en `02_limpieza.ipynb` (tabla `registro_limpieza`).

La columna **Validación** la rellena el equipo clínico. La IA propone; el equipo decide.

| Regla | Decisión | Justificación | Evidencia | Validación |
|---|---|---|---|---|
| R01 | El código 9 ("se desconoce") pasa a ausente en las variables del núcleo. 9999 y 99999 ("no disponible", Virgen del Rocío) también. Sexo = 2 → ausente | Si no, un 9 entraría en la distancia como si fuera una categoría clínica | §2.2: 552 en tabaquismo | ☐ |
| R02 | La edad se usa como tramo ordinal (ordenado por límite inferior) y como grupo amplio (<50, 50–64, 65–74, ≥75). Los tramos que cruzan grupos (50-69, 60-69, 70+) quedan sin grupo | Los tramos son desiguales por protección de datos, y ordenarlos alfabéticamente sería un error | §4: 39 pacientes | ☐ |
| R03 | Las fechas se leen con formato explícito por registro (MM/DD/YYYY en Virgen del Rocío). `dias_alta` = fecha de la prueba − alta. Si falta la fecha, se usa `visita_dias` solo en la primera visita. En Lleida y TENACITY, la fecha de consulta tiene como respaldo la de la prueba funcional | Los intervalos son exactos aunque las fechas estén desplazadas. El número de visita no es el eje temporal | §6, §8 | ☐ |
| R04 | Las medidas con fecha anterior al alta y los `visita_dias < 0` quedan fuera del análisis temporal (se marcan, no se borran) | Son errores de fecha o reingresos | §2.3: 54 pacientes | ☐ |
| R05 | Rangos plausibles: DLCO, FVC y FEV1 entre 20 y 160 %; PM6M entre 100 y 900 m; mMRC entre 0 y 4; HADS entre 0 y 21. Fuera de rango → medida excluida | Los valores extremos son errores de registro. Una PM6M < 100 m es probablemente un % del predicho (3 casos) | §8 | ☐ |
| R06 | En los pacientes compartidos, la medida de CIBERESUCICOVID se excluye si el registro socio tiene la misma variable a ≤ 30 días (o en el mismo mes nominal si falta la fecha) | Es la misma prueba registrada dos veces (97 % con valor idéntico ±1). Contarla dos veces duplica el peso del paciente | §8 | ☐ |
| R07 | `cohorte_analisis`: los fusionados con Lleida (437) van a POSTCOVID_LLEIDA y los enriquecidos de Virgen del Rocío (22) a VIRGEN_DEL_ROCIO | Cada paciente tiene que estar en un solo lado de la replicación, y su seguimiento procede del registro socio | §1 | ☐ |
| R08 | Superviviente = sin exitus hospitalario. Un exitus con medidas de seguimiento se marca como inconsistente y queda fuera del fenotipado | Solo los supervivientes pueden tener secuelas. Exitus con seguimiento es imposible | §5: 3 casos | ☐ |
| R09 | En CIBERESUCICOVID, `IH_PorcCMD` ≥ 50 es el criterio principal y ≥ 99 el de sensibilidad. Son banderas, no se borran registros | Los registros < 50 % casi no tienen seguimiento (< 1 %) y no se conoce su estado vital | §2.1 | ☐ |
| R10 | TAC de Lleida y TENACITY: solo formularios completos (`*_tacimagen_complete == 2`); un formulario completo sin lesión marcada = sin lesión. Fibrosis estricta = fibrótica; amplia = fibrótica o reticular. CIBERESUCICOVID: tractos fibrosos solo con TAC realizado, como constructo aparte. La imagen se usa **solo para describir** | Las casillas valen 0 aunque nadie rellene el formulario. Los TAC tardíos son selectivos y en TENACITY casi no hay | §10 | ☐ (confirmar que "completo sin lesión = sin lesión") |
| R11 | Disponibilidad por paciente y variable: `medida` / `falta` / `no_recogida`. Nunca se imputa lo `no_recogida` | Que una cohorte no recoja una variable no es un dato perdido | §3 | ☐ |
| R12 | TENACITY: una visita no realizada cuya fecha teórica (alta + mes + 60 días) es posterior al corte es "no le toca", no abandono. Sin visita ni `perdida_seg` = "sin dato" | El reclutamiento sigue abierto. Contarlo como abandono sesgaría la corrección IPW | §8 | ☐ |
| R13 | Estado de definición: primera medida válida entre 60 y 210 días, con umbrales clínicos (DLCO < 80, FVC < 80, HADS ≥ 8, mMRC ≥ 2, PM6M < 400 orientativo). Capa A = DLCO o FVC. Capa B = además HADS o mMRC. FEV1 solo describe | Ventana común entre registros. Umbrales en lugar de z-scores por cohorte, para no borrar diferencias reales. FEV1 es redundante con FVC (ρ = 0,88) | §6, §7 | ☐ |
| R14 | Preguntas condicionadas: en CIBERESUCICOVID, si la resolución clínica es **total**, se deduce **fatiga = 0** (medida añadida con `deducida = True`) | El cuaderno solo pregunta por la fatiga si los síntomas no se han resuelto: los 871 pacientes con resolución total a 3 meses tienen la fatiga vacía (100 %). No es imputar, es deducir lo que el cuaderno da por hecho. Sin la regla, Gower comparaba a esos pacientes sin la variable fatiga | Tabla cruzada resolución × fatiga (CIBERESUCICOVID, M3 y M6) | ☐ |

## Pendientes

- [ ] Confirmar con el equipo de TENACITY el 37 % de fumadores activos (coincide con `IH_fumador`).
- [ ] Revisar los 3 exitus hospitalarios con seguimiento.
- [ ] Decidir si se imputa algo para el clasificador. Para el clustering no se imputa: Gower usa los pares disponibles.
