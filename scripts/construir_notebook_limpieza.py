"""Genera `02_limpieza.ipynb`.

Uso: python scripts/construir_notebook_limpieza.py
(después: python -m nbconvert --to notebook --execute --inplace 02_limpieza.ipynb)
"""
from pathlib import Path

import nbformat as nbf

celdas: list = []
md = lambda texto: celdas.append(nbf.v4.new_markdown_cell(texto.strip()))
code = lambda texto: celdas.append(nbf.v4.new_code_cell(texto.strip()))

md("""
# Reto 2 · Limpieza de datos

**Notebook 02.** Convierte la tabla cruda en tablas limpias, trazables y reproducibles para el fenotipado (capas A y B), las trayectorias y el clasificador. Los problemas que resuelve salen del notebook 01 (análisis exploratorio).

Principios:
- **Nada se borra en silencio.** Cada regla deja una entrada en `registro_limpieza` con el N afectado por registro. Las medidas descartadas siguen en la tabla, con `excluida = True` y su motivo.
- **La ausencia estructural no se imputa.** Si un registro no recoge una variable, se marca como `no_recogida`, que es distinto de `falta`.
- **Ningún número mágico.** Umbrales, ventanas, rangos y columnas por visita están en `config.yaml`. La justificación de cada regla está en `docs/registro_decisiones.md`.

| Regla | Qué hace |
|---|---|
| R01 | Códigos "se desconoce" (9, 9999, 99999) y sexo no especificado → ausente |
| R02 | Edad: tramo ordinal y grupo amplio; tramos que cruzan grupos → grupo ausente |
| R03 | Fechas con formato explícito por registro; `dias_alta` = fecha de la prueba − alta |
| R04 | Fechas anteriores al alta → fuera del análisis temporal |
| R05 | Valores fuera del rango plausible → medida excluida |
| R06 | Prueba duplicada en pacientes compartidos → se queda la del registro socio |
| R07 | `cohorte_analisis`: cada paciente en un solo lado de la replicación |
| R08 | Supervivencia al alta; exitus con seguimiento → inconsistente |
| R09 | Cumplimentación de CIBERESUCICOVID (`IH_PorcCMD` ≥ 50; ≥ 99 como sensibilidad) |
| R10 | TAC: solo formularios completos o TAC realizados; fibrosis estricta y amplia |
| R11 | Ausencia estructural (`no_recogida`) frente a `falta` |
| R12 | TENACITY: abandono frente a "aún no le toca" la visita |
| R13 | Estado en la ventana de definición (60–210 días) y banderas de inclusión de las capas A y B |

> **Privacidad.** Solo se muestran agregados, con supresión de recuentos < 10. Las tablas limpias se guardan en `datos_procesados/`, fuera del control de versiones.
""")

md("## 0 · Entorno")
code("""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

RAIZ = Path.cwd()
sys.path.insert(0, str(RAIZ))
from src import carga, limpieza, privacidad

cfg = carga.cargar_config()
N_MIN = cfg["privacidad"]["n_minimo_celda"]
REG = list(cfg["registros"])
COLOR = cfg["colores_registro"]
ETQ = {"CIBERESUCICOVID": "CIBERESUCICOVID", "POSTCOVID_LLEIDA": "POSTCOVID-Lleida",
       "TENACITY": "TENACITY", "VIRGEN_DEL_ROCIO": "Virgen del Rocío"}
FIG = RAIZ / cfg["rutas"]["figuras"]; FIG.mkdir(parents=True, exist_ok=True)
TAB = RAIZ / cfg["rutas"]["tablas"]; TAB.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({"figure.dpi": 100, "axes.titleweight": "bold", "axes.titlesize": 12,
                     "axes.spines.top": False, "axes.spines.right": False})
pd.set_option("display.max_columns", 50); pd.set_option("display.width", 220); pd.set_option("display.max_colwidth", 110)


def nota(ax, texto: str) -> None:
    \"\"\"Pie de figura con N y cohortes.\"\"\"
    ax.figure.text(0.01, -0.01, texto, fontsize=8.5, color="0.35", va="top", ha="left")
""")

md("## 1 · Aplicar la limpieza")
code("""
tablas = limpieza.limpiar(cfg)
pacientes, medidas, estado = tablas["pacientes"], tablas["medidas"], tablas["estado_definicion"]
for nombre, t in tablas.items():
    print(f"{nombre:<20} {t.shape[0]:>7,} filas × {t.shape[1]:>3} columnas")
""")

md("""
## 2 · Registro de reglas

Cada fila es una regla aplicada y el número de casos afectados, por registro de análisis (en las tablas de pacientes) o por registro de origen (en las medidas).
""")
code("""
reg = tablas["registro_limpieza"].copy()
reg.columns = [ETQ.get(c.removeprefix("N_"), c) for c in reg.columns]
cols_n = ["N", *[ETQ[r] for r in REG if ETQ[r] in reg.columns]]
reg = privacidad.suprimir_celdas(reg, N_MIN, cols_n)
reg.to_csv(TAB / "02_registro_limpieza.csv", index=False)
reg.set_index(["regla", "tabla"])
""")
md("""
**Lectura.**
- **R06 es la regla de más impacto sobre las medidas.** Se eliminan cerca de 950 pruebas de CIBERESUCICOVID que son la misma prueba registrada en Lleida o en Virgen del Rocío (el 97 % con valor idéntico ±1). Sin esta regla, los pacientes compartidos pesarían el doble en las trayectorias.
- **R10 descarta miles de visitas sin TAC o con el formulario incompleto.** Son casillas a 0 que no significan "sin hallazgos".
- **R09 y R08 definen la población de CIBERESUCICOVID:** 2.233 exitus hospitalarios y unos 2.700 registros con poca cumplimentación (casi sin seguimiento, ver el notebook 01, §2.1).
- **R08:** 3 pacientes tienen exitus hospitalario y medidas de seguimiento. Quedan fuera del fenotipado hasta que el equipo lo revise.
""")

md("## 3 · Flujo de pacientes (CONSORT)")
code("""
consort = tablas["flujo_consort"].rename(columns=ETQ)
consort.to_csv(TAB / "02_flujo_consort.csv")
privacidad.suprimir_celdas(consort, N_MIN, list(consort.columns))
""")
code("""
fig, ax = plt.subplots(figsize=(12, 5))
datos = tablas["flujo_consort"][REG]
y = np.arange(len(datos))
izquierda = np.zeros(len(datos))
for r in REG:
    ax.barh(y, datos[r], left=izquierda, color=COLOR[r], label=ETQ[r], edgecolor="white")
    izquierda += datos[r].values
for i, total in enumerate(izquierda):
    ax.text(total + 80, i, f"{int(total):,}", va="center", fontsize=9.5)
ax.set_yticks(y, [p.split(". ", 1)[1] for p in datos.index]); ax.invert_yaxis()
ax.set_xlabel("Pacientes"); ax.set_xlim(0, izquierda.max() * 1.12)
ax.set_title("Flujo de pacientes por registro de análisis")
ax.legend(frameon=False, loc="lower right")
nota(ax, "Cada paso exige los anteriores. Registro de análisis: los pacientes compartidos cuentan en un solo registro "
         "(fusionados → Lleida; enriquecidos → Virgen del Rocío).")
plt.tight_layout(); fig.savefig(FIG / "02_flujo_consort.png", bbox_inches="tight", dpi=150); plt.show()
""")
md("""
**Lectura.**
- **CIBERESUCICOVID se queda en unos 1.350 pacientes** para la capa A (de 8.815). Las mayores pérdidas vienen de la mortalidad, de la falta de seguimiento y de que la DLCO no se mide en todos los centros.
- **Capa B (multidominio): unos 530 pacientes de Lleida y 155 de TENACITY.** CIBERESUCICOVID no aporta a la capa B porque no recoge HADS ni mMRC. Virgen del Rocío aporta unos 20 pacientes con mMRC, pero se usará solo como complemento.
""")

md("## 4 · Medidas: qué se conserva y qué se excluye")
code("""
resumen = (medidas.assign(motivo=medidas["motivo_exclusion"].fillna("válida"))
           .groupby(["variable", "motivo"]).size().unstack(fill_value=0))
resumen = resumen[["válida"] + [c for c in limpieza.MOTIVOS if c in resumen.columns]]
resumen["% excluidas"] = (100 * (1 - resumen["válida"] / resumen.sum(axis=1))).round(1)
privacidad.suprimir_celdas(resumen, N_MIN, [c for c in resumen.columns if c != "% excluidas"])
""")
code("""
validas = medidas[~medidas["excluida"]]
n_pac = validas.groupby(["variable", "registro"], observed=True)["subject_id"].nunique().unstack(fill_value=0)
n_pac = n_pac.rename(columns=ETQ)
print("Pacientes con al menos una medida válida (registro de origen de la medida):")
privacidad.suprimir_celdas(n_pac, N_MIN, list(n_pac.columns))
""")
code("""
dl = validas[validas["variable"] == "dlco"]
print(f"DLCO válida: {len(dl):,} medidas en {dl['subject_id'].nunique():,} pacientes; "
      f"con ≥ 2 medidas: {(dl.groupby('subject_id').size() >= 2).sum():,}")
print(f"Medidas sin tiempo (dias_alta ausente): {int(validas['dias_alta'].isna().sum())}")
""")

md("""
## 5 · Estado en la ventana de definición

Para cada paciente y variable se toma la **primera medida válida entre 60 y 210 días** tras el alta y se clasifica con el umbral clínico. La disponibilidad distingue:
- `medida`: hay valor en la ventana.
- `falta`: el registro recoge la variable, pero este paciente no tiene valor en la ventana.
- `no_recogida`: el registro no recoge la variable (ausencia estructural, nunca se imputa).
""")
code("""
variables = cfg["variables_definicion"]["capa_A"] + cfg["variables_definicion"]["capa_B_extra"] + ["fev1"]
disp = (estado.melt(id_vars="cohorte_analisis", value_vars=[f"{v}_disp" for v in variables], var_name="variable", value_name="disp")
        .assign(variable=lambda d: d["variable"].str.replace("_disp", "")))
pct = (disp.groupby(["variable", "cohorte_analisis"], observed=True)["disp"].value_counts(normalize=True)
       .unstack(fill_value=0) * 100)

fig, axes = plt.subplots(1, len(REG), figsize=(16, 4.2), sharey=True)
colores = {"medida": "#4daf4a", "falta": "#fdae61", "no_recogida": "#d9d9d9"}
for ax, r in zip(axes, REG):
    d = pct.xs(r, level="cohorte_analisis").reindex(variables).reindex(columns=list(colores), fill_value=0)
    d.plot.barh(stacked=True, ax=ax, color=list(colores.values()), width=0.75, legend=False, edgecolor="white")
    n = int((pacientes["cohorte_analisis"] == r).sum())
    ax.set_title(f"{ETQ[r]} (N={n:,})", fontsize=11); ax.set_xlabel("% de pacientes"); ax.set_xlim(0, 100)
axes[0].invert_yaxis(); axes[0].set_ylabel("")
axes[-1].legend(list(colores), frameon=False, bbox_to_anchor=(1, 1))
nota(axes[0], f"Ventana de definición {cfg['tiempo']['ventana_definicion_dias'][0]}–{cfg['tiempo']['ventana_definicion_dias'][1]} días. "
              "Gris = el registro no recoge la variable (ausencia estructural, no se imputa).")
plt.tight_layout(); fig.savefig(FIG / "02_disponibilidad_ventana.png", bbox_inches="tight", dpi=150); plt.show()
""")
code("""
# Prevalencia de alteración en la ventana, solo pacientes incluibles en la capa A
inc = estado.merge(pacientes[["subject_id", "incluible_A", "incluible_B"]], on="subject_id")
filas = []
for v in variables:
    for r in REG:
        s = inc.loc[(inc["cohorte_analisis"] == r) & inc["incluible_A"], f"{v}_alterada"].dropna()
        k = int(s.sum())
        if len(s) >= N_MIN and (k == 0 or k >= N_MIN):
            filas.append({"variable": v, "registro": ETQ[r], "N": len(s), "% alterada": round(100 * k / len(s), 1)})
prev = pd.DataFrame(filas).pivot(index="variable", columns="registro", values="% alterada").reindex(variables)
prev
""")
md("""
Estas prevalencias son **el punto de partida del fenotipado**. En la misma ventana temporal y con los mismos umbrales clínicos, las diferencias entre registros son diferencias de población, no de momento de medida. Por ejemplo, TENACITY tiene menos DLCO alterada que CIBERESUCICOVID y Lleida.
""")

md("## 6 · TENACITY: abandono frente a \"aún no le toca\"")
code("""
vt = tablas["visitas_tenacity"]
print(f"Fecha de corte estimada (última visita registrada, fechas desplazadas ±30 d): {vt.attrs.get('corte', 'n/d')}")
tab_vt = pd.crosstab(vt["visita"], vt["estado"]).reindex(["M3", "M6", "A1"])
privacidad.suprimir_celdas(tab_vt, N_MIN, list(tab_vt.columns))
""")
md("""
- **No le toca:** en las visitas de 6 y 12 meses, una parte de los pacientes "sin visita" todavía no ha llegado a la fecha. Contarlos como abandono inflaría la pérdida de seguimiento y sesgaría la corrección por abandono informativo (IPW).
- **Sin dato:** pacientes ya debidos, sin visita y sin `perdida_seg`. Lo prudente es tratarlos como abandono no declarado.
""")

md("## 7 · Guardar")
code("""
carpeta = limpieza.guardar(tablas, cfg)
print(f"Tablas guardadas en {carpeta.relative_to(RAIZ)}/ (ignorado por git):")
for p in sorted(carpeta.glob("*.parquet")):
    print(f"  {p.name}")
""")
md("""
### Uso en los siguientes notebooks

```python
import pandas as pd
pacientes = pd.read_parquet("datos_procesados/pacientes.parquet")
estado = pd.read_parquet("datos_procesados/estado_definicion.parquet")
medidas = pd.read_parquet("datos_procesados/medidas.parquet").query("~excluida")
capa_A = estado[estado["subject_id"].isin(pacientes.loc[pacientes["incluible_A"], "subject_id"])]
```

**Pendiente de validación clínica** (`docs/registro_decisiones.md`):
- Que un formulario de TAC completo sin lesión marcada signifique "sin lesión".
- La asignación de los pacientes compartidos a Lleida y Virgen del Rocío.
- Los 3 casos de exitus con seguimiento.
- Que la PM6M < 100 m se considere no plausible.
""")

nb = nbf.v4.new_notebook()
nb["cells"] = celdas
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                  "language_info": {"name": "python"}}
salida = Path(__file__).resolve().parents[1] / "02_limpieza.ipynb"
nbf.write(nb, salida)
print(f"Escrito {salida} ({len(celdas)} celdas)")
