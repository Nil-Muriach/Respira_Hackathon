# CLAUDE.md — Reto 2: Fenotipado digital de secuelas post-infecciosas

Este archivo da contexto a Claude Code sobre el proyecto. Léelo entero antes de escribir código.

## Objetivo del reto

Construir un **notebook reproducible** que identifique automáticamente fenotipos clínicos de las secuelas tras una infección vírica grave, de forma:
- **reproducible entre cohortes** (lo más valorado),
- **interpretable** y **útil en la consulta**.

La pregunta no es si existen subgrupos (siempre salen), sino **si sobreviven al cambio de cohorte**.

Pasos mínimos del notebook: recibir la tabla → limpiar y seleccionar variables con criterio clínico → identificar fenotipos (clustering) → describir cada fenotipo → mostrar su evolución temporal → generar un informe breve.

Criterios de evaluación: justificación de objetivos (20 %), rigor científico (30 %), interpretabilidad (20 %), presentación (30 %).

## Conceptos

- **Cohorte**: grupo de pacientes de un estudio/registro concreto.
- **Fenotipo**: tipo de paciente según su estado en varios dominios (pulmón, síntomas, esfuerzo, ánimo, sueño, cognición) en un momento dado.
- **Trayectoria**: cómo cambia ese estado con el tiempo.

## Datos

### Cohortes

| Cohorte | N | Seguimiento | Qué aporta |
|---|---|---|---|
| CIBERESUCICOVID | 9.274 | 3 m, 6 m, 1 año (~53 var./visita) | Gran N, fase aguda UCI completa, mortalidad. Seguimiento solo respiratorio |
| POSTCOVID-Lleida | 624 (456 UCI, 166 planta) | ~3,5 m, ~7 m, 1 a, 18 m, 2 a, 3 a, 4 a | Seguimiento más largo y multidominio |
| TENACITY | 287 | 3 m, 6 m, 1 año (~300 var./visita) | Post-UCI (SDRA/neumonía, no solo COVID), única pospandémica, única con función física. **Sigue reclutando** |
| Virgen del Rocío | 83 | 1 m (casi nada después) | Espirometría y DLCO al mes. No permite trayectorias |

Total: **9.809 pacientes únicos** (una fila por paciente). 459 pacientes están en dos registros (437 Lleida + 22 Virgen del Rocío también en CIBERESUCICOVID) y se fusionaron en la fila de CIBERESUCICOVID.

### Ficheros (usar en este orden)

1. `cohorte_unificada_nucleo` — 9.809 × 39: 24 variables armonizadas + trazabilidad (`cohorte`, `origen`, `en_<registro>`, `visita_dias`). **Empezar aquí.**
2. `diccionario_unificado_full` y `dominios_variables` — qué es cada variable, qué cohortes la recogen, cuántos pacientes, qué dominio clínico.
3. `cohorte_unificada` — 9.809 × 4.956, tabla completa. Solo cuando ya se sepa qué columnas hacen falta.

### Prefijos de columnas por visita

- CIBERESUCICOVID y TENACITY: ingreso, `M3_`, `M6_`, `A1_`
- Lleida: `LFA_` (fase aguda), `LV1_`, `LV2_`, `LA1_`, `LM18_`, `LM24_`, `LA3_`, `LA4_`
- Virgen del Rocío: ingreso, visita 1, visita 2

> Verificar los nombres exactos con el diccionario antes de usarlos. No asumir.

### Columnas clave conocidas

- `cohorte` — cohorte de la fila (¡no sirve para filtrar pertenencia!).
- `en_POSTCOVID_LLEIDA`, `en_VIRGEN_DEL_ROCIO` (y demás `en_<registro>`) — **usar estas para filtrar** a todos los pacientes de un registro.
- `fusionada_lleida == 1` — los 437 pacientes compartidos CIBERESUCICOVID/Lleida.
- `visita_dias` — días reales desde el alta. **Es el eje temporal**, no el número de visita.
- `IH_PorcCMD` — % de cumplimentación del conjunto mínimo de datos en CIBERESUCICOVID (2.730 registros < 50 %).
- `*_tacimagen_complete` — bandera de formulario de TAC cumplimentado (valor `2` = completo).

### Cobertura por dominio

- Fase aguda y función respiratoria: en las 4 cohortes (base común).
- Calidad de vida, HADS, cognición, sueño: solo Lleida y TENACITY.
- Función física: solo TENACITY.
- PM6M: Lleida, TENACITY, Virgen del Rocío (no CIBERESUCICOVID).

## Diseño del análisis

### Idea central: el "pasaporte de fenotipo"

Cada fenotipo candidato debe superar cinco comprobaciones ("sellos"):
1. **Estable**: Jaccard bootstrap por clúster dentro de la cohorte de descubrimiento.
2. **Viaja**: se replica en otra cohorte no usada para descubrirlo.
3. **Evoluciona distinto**: trayectoria diferenciada.
4. **Se reconoce con pocas variables**: clasificador corto validado por cohorte/centro.
5. **Sentido clínico**: lo valora el equipo (no la IA).

### Dos capas

| Capa | Descubre | Replica | Variables |
|---|---|---|---|
| A — respiratoria | CIBERESUCICOVID | Lleida, TENACITY | Función pulmonar, imagen, síntomas |
| B — multidominio | Lleida | TENACITY (y al revés) | + HADS, calidad de vida, cognición, sueño |

Virgen del Rocío: solo uso complementario (su visita es al mes, no comparable).

### Principios

- **Variables de definición ≠ variables de descripción.** El clustering usa el estado tras el alta (ventana ~60–210 días según `visita_dias`). Las variables del ingreso se usan para describir y predecir, no para agrupar.
- **Umbrales clínicos en vez de z-scores por cohorte** (p. ej. DLCO < 80 %, HADS ≥ 8, mMRC ≥ 2). Los z-scores por cohorte borran diferencias reales.
- **Ausencia estructural ≠ ausencia por no rellenar.** Lo que una cohorte no recoge no se imputa nunca. Un 0 en un TAC sin bandera completa es "sin rellenar", no "sin hallazgos".
- **Prueba nula**: comparar la silueta real con la de datos permutados columna a columna antes de elegir k. Siluetas de 0,20–0,30 son normales en datos clínicos.

### Protocolo de replicación (cohorte A → cohorte B)

1. Clustering en A (Gower + PAM) → medoides.
2. En B: (a) asignar cada paciente al medoide de A más cercano ("etiquetas transferidas"); (b) clustering independiente en B con el mismo k ("etiquetas nativas").
3. Emparejar clústeres con el algoritmo húngaro (`scipy.optimize.linear_sum_assignment`) y calcular Jaccard por par + ARI global. Opcional: prediction strength (Tibshirani & Walther).
4. Comparar perfiles de clústeres emparejados (diferencias estandarizadas).
5. Repetir B → A.
6. Dar siempre **intervalos bootstrap**, no solo valores puntuales (las cohortes pequeñas dan estimaciones ruidosas).
7. El tamaño relativo de un fenotipo puede variar entre cohortes; lo que debe replicarse es el perfil.

### Trayectorias

- Variable principal: DLCO (% predicho). 4.420 medidas en 2.322 pacientes; 1.088 con ≥ 2.
- Modelo mixto: `DLCO ~ fenotipo * log(tiempo desde alta)` con intercepto y pendiente aleatorios (`statsmodels` MixedLM).
- **Comparación entre cohortes solo en la ventana común (0–12 meses).** Más allá del año solo hay datos de Lleida: presentarlo aparte como extensión, sin extrapolar las otras cohortes.
- Mostrar el N en cada punto temporal.
- **Abandono informativo**: quien no vuelve al año tenía mejor DLCO (73,4 % vs 65,2 %). Análisis de sensibilidad: pesos inversos de probabilidad de permanencia (IPW) y curvas por patrón de abandono.
- TENACITY: distinguir abandono de "aún no le toca la visita" (reclutamiento en curso).
- Lleida: reasignar cada visita al medoide más cercano y mostrar transiciones entre fenotipos (diagrama aluvial).

### Clasificador

- Corto (árbol de profundidad ≤ 3 o logística multinomial L1, ~5 variables de la primera visita).
- Exploratorio con variables del alta.
- Validación **leave-one-cohort-out / leave-one-center-out**, nunca partición aleatoria.
- Métricas: AUC-PR y calibración, no solo AUC-ROC.

## Trampas conocidas (comprobar en el código)

- [ ] Excluir `fusionada_lleida == 1` de uno de los dos lados al validar Lleida ↔ CIBERESUCICOVID (y los 22 de Virgen del Rocío).
- [ ] Filtrar TAC por `*_tacimagen_complete == 2`.
- [ ] Fibrosis: definición estricta (lesiones fibróticas) como principal; amplia (fibróticas o reticulares) como sensibilidad. CIBERESUCICOVID ("tractos fibrosos") no es el mismo constructo: comparar dentro de cada cohorte.
- [ ] Prevalencias de TAC en visitas tardías no son prevalencias de cohorte (el TAC se repite a quien tiene lesiones).
- [ ] Declarar umbral de `IH_PorcCMD` (≥ 50 principal, ≥ 99 sensibilidad).
- [ ] Usar `visita_dias`, no el número de visita.
- [ ] Matriz de Gower de ~9.000 × 9.000 ≈ 350 MB en float32: usar submuestra estratificada (estilo CLARA) para descubrimiento y asignar el resto al medoide.
- [ ] Toda figura lleva su N y su cohorte.

## Estructura del repositorio

```
reto2_fenotipos/
├── reto2_fenotipos.ipynb        # entregable principal
├── config.yaml                  # umbrales, ventanas, k, semillas
├── src/
│   ├── datos_sinteticos.py      # genera datos falsos con el mismo esquema (para desarrollar)
│   ├── carga.py                 # lectura, trazabilidad, exclusión de solapes, flujo CONSORT
│   ├── estados_clinicos.py      # variables → estados con umbrales clínicos
│   ├── fenotipado.py            # Gower, PAM, prueba nula, selección de k
│   ├── pasaporte.py             # bootstrap, transferencia, Jaccard, ARI, prediction strength
│   ├── trayectorias.py          # modelos mixtos, IPW, patrones de abandono, transiciones
│   ├── clasificador.py          # validación por cohorte/centro, calibración
│   ├── privacidad.py            # supresión de celdas N < 10, utilidades de salida segura
│   └── fichas.py                # fichas de fenotipo (solo agregados)
├── outputs/figuras/  outputs/tablas/  outputs/fichas/
├── docs/registro_decisiones.md  # cada decisión clínica con su justificación (lo escribe el equipo)
├── docs/declaracion_IA.md
└── requirements.txt
```

Si el entorno no permite importar `src/`, el notebook debe poder funcionar con el código pegado en celdas.

## Estructura del notebook

0. Pregunta y contrato de replicación (hipótesis y criterios fijados antes de ver resultados; leídos de `config.yaml`)
1. Recibe la tabla (carga, trazabilidad, flujo de pacientes)
2. Limpia y selecciona (mapa de ausencias por cohorte y dominio, estados clínicos)
3. Identifica fenotipos (capa A y B, prueba nula, k, estabilidad)
4. Pasaporte: replicación entre cohortes (mapa de calor fenotipos × cohortes)
5. Describe cada fenotipo (fichas)
6. Evolución (modelos mixtos, sensibilidad al abandono, transiciones)
7. Aplicación (clasificadores)
8. Informe breve, limitaciones y declaración de IA

## Convenciones de código

- Python 3, `pandas`, `numpy`, `scikit-learn`, `scikit-learn-extra` (KMedoids, `method="pam"`, `metric="precomputed"`), `gower`, `statsmodels`, `scipy`, `matplotlib`/`plotly`.
- Comprobar al inicio qué librerías hay en el entorno; si falta alguna, proponer alternativa.
- Todos los parámetros en `config.yaml`; nada de números mágicos en el código.
- Semilla fija en todo lo aleatorio.
- Funciones pequeñas, con docstring y type hints; que devuelvan objetos, no que impriman.
- Toda salida pasa por `privacidad.py` antes de mostrarse o guardarse.
- Tests básicos con los datos sintéticos.
- Código y comentarios en español.
