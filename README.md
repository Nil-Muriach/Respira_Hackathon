# Reto 2 · Fenotipado digital de secuelas post-infecciosas

Identificamos fenotipos de las secuelas tras una infección respiratoria grave que **se mantienen al cambiar de cohorte**, describimos **cómo evolucionan** y estudiamos **cuándo se pueden predecir**. Datos: cuatro cohortes (CIBERESUCICOVID, POSTCOVID-Lleida, TENACITY y Virgen del Rocío), 9.809 pacientes únicos.

**El notebook a presentar es [`main.ipynb`](main.ipynb).** Resume el proyecto de principio a fin. Los notebooks numerados desarrollan cada paso con más detalle.

## Resultados en tres frases

1. **Un fenotipo que viaja.** Con lo que las tres cohortes miden igual (DLCO, FVC y FEV1), a los 3 meses salen dos fenotipos: **F1 · función conservada** y **F2 · afectación funcional**.
   - Son estables (Jaccard 0,92).
   - Se reproducen al dejar fuera cada cohorte (ARI 0,50–0,73).
   - Los fenotipos de síntomas de una sola cohorte no se reproducen.
2. **Evolucionan distinto.** Los dos mejoran, pero al año F2 sigue **11,5 puntos de DLCO por debajo** de F1 y por debajo del 80 %, en todas las cohortes.
3. **Se predice en la visita de los 3 meses, no al alta.**
   - Con los datos del alta, AUC ≈ 0,63 con cualquier modelo.
   - A los 3 meses, **la DLCO sola** predice quién seguirá alterado al año: AUC 0,79 en cohortes no vistas. Ningún modelo más complejo la mejora.

## Estructura

```
├── main.ipynb                      ← notebook de la presentación (resumen ejecutable)
├── 01_analisis_exploratorio.ipynb  ← qué datos hay, de quién, cuándo y con qué calidad
├── 02_limpieza.ipynb               ← 14 reglas de limpieza, flujo de pacientes, tablas limpias
├── 03_fenotipado_ciberes.ipynb     ← fenotipos en CIBERESUCICOVID (síntomas + función): no viajan
├── 04_fenotipado_comun.ipynb       ← fenotipos con las 3 cohortes juntas: el fenotipo principal
├── 05_trayectorias.ipynb           ← evolución de la DLCO por fenotipo (modelo mixto, abandono, transiciones)
├── 06_modelo_alta.ipynb            ← ¿se predice el fenotipo con los datos del alta? (EBM, 3 cohortes)
├── 07_modelo_fase_aguda.ipynb      ← ¿y añadiendo la fase aguda? (EBM, solo CIBERESUCICOVID)
├── 08_modelo_primera_visita.ipynb  ← en la visita de 3 meses: ¿quién sigue alterado al año?
├── config.yaml                     ← todos los parámetros (umbrales, ventanas, k, semillas, rutas)
├── requirements.txt
├── src/                            ← funciones reutilizables (con docstrings)
│   ├── carga.py                    lectura de datos y tablas largas de seguimiento
│   ├── limpieza.py                 reglas R01–R14 y registro de limpieza
│   ├── fenotipado.py               distancia de Gower, PAM, prueba nula, estabilidad
│   ├── fenotipado_perfil.py        fenotipado a partir de un perfil de config.yaml
│   ├── pasaporte.py                replicación entre cohortes (húngaro, Jaccard, ARI)
│   ├── trayectorias.py             modelos mixtos, contrastes, IPW
│   └── privacidad.py               supresión de celdas con N < 10
├── tests/                          ← tests con datos sintéticos (pytest)
├── outputs/figuras/                ← figuras; el prefijo es el número del notebook (main/ para el principal)
├── outputs/tablas/                 ← tablas agregadas (no se versionan)
└── docs/
    ├── informe.md                  ← informe completo: métodos, resultados, limitaciones, preguntas del jurado
    ├── registro_decisiones.md      ← cada regla de limpieza con su justificación (validación clínica del equipo)
    ├── datos_limpios.md            ← qué contiene cada tabla limpia
    └── declaracion_IA.md           ← uso de inteligencia artificial
```

## Qué produce cada notebook

Los notebooks se ejecutan **en orden**. Cada uno lee lo que generan los anteriores (en `../Datos limpios/`, fuera del repositorio).

| Notebook | Lee | Genera |
|---|---|---|
| 01 Exploración | datos crudos | figuras `01_eda_*` |
| 02 Limpieza | datos crudos | `pacientes`, `medidas`, `estado_definicion`, `visitas_tenacity`, `registro_limpieza`, `flujo_consort` |
| 03 Fenotipado CIBERESUCICOVID | tablas limpias | `fenotipos.parquet` |
| 04 Fenotipado común | tablas limpias | `fenotipos_comun.parquet`, `base_modelo.parquet` |
| 05 Trayectorias | `fenotipos_comun` | figuras y tablas `05_tray_*` |
| 06 Modelo al alta | `base_modelo` | `modelo_ebm_h3.pkl` |
| 07 Modelo con fase aguda | `base_modelo` + tabla completa | `modelos_ebm_ciberes_funcional.pkl` |
| 08 Modelo de la primera visita | tablas limpias | `modelo_primera_visita.pkl` |
| `main` | datos crudos | resumen; regenera lo que necesita (tablas limpias y fenotipos) |

## Cómo ejecutarlo

1. **Datos.** No están en el repositorio: son datos de pacientes. La ruta se fija en `config.yaml → rutas.datos`. Por defecto es `../SECUELAS-Challenge/DATOS COVID Y OTROS VIRUS RESPIRATORIOS/bases_reto2/`, con `cohorte_unificada_nucleo.csv`, `cohorte_unificada.csv` y el diccionario. Las tablas limpias se guardan en `../Datos limpios/`.
2. **Entorno** (Python 3.13):
   ```bash
   python -m venv .venv
   .venv\Scripts\activate          # Windows
   pip install -r requirements.txt
   ```
3. **Ejecución:** abrir los notebooks en orden (01 → 08) y ejecutar todo, o directamente `main.ipynb`. Desde la terminal:
   ```bash
   python -m nbconvert --to notebook --execute --inplace 01_analisis_exploratorio.ipynb
   ```
   Los clustering con permutaciones y bootstrap (notebooks 03, 04 y `main`) tardan varios minutos.
4. **Tests:** `python -m pytest -q`.

## Principios

- **Reproducible:** todos los parámetros están en `config.yaml`, con semilla fija (2026). Nada de números mágicos en el código.
- **Replicación antes que descubrimiento:** un fenotipo solo cuenta si pasa el "pasaporte":
  - tiene estructura frente a datos permutados;
  - es estable (bootstrap);
  - se reproduce en una cohorte que no se usó para descubrirlo;
  - evoluciona distinto;
  - tiene sentido clínico.
- **Sin imputar la ausencia estructural:** lo que una cohorte no recoge nunca se inventa.
- **Privacidad:** los datos están seudonimizados y no salen del entorno. Solo se muestran agregados, y las celdas con N < 10 se suprimen (`src/privacidad.py`).
