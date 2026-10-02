"""Genera `04_fenotipado_lleida.ipynb` (mismo procedimiento que el notebook 03, descubriendo en Lleida).

Uso: python scripts/construir_notebook_lleida.py
(después: python -m nbconvert --to notebook --execute --inplace 04_fenotipado_lleida.ipynb)
"""
from pathlib import Path

import nbformat as nbf

celdas: list = []
md = lambda texto: celdas.append(nbf.v4.new_markdown_cell(texto.strip()))
code = lambda texto: celdas.append(nbf.v4.new_code_cell(texto.strip()))

md("""
# Reto 2 · Fenotipos multidominio descubiertos en POSTCOVID-Lleida (3, 6 y 12 meses)

**Notebook 04.** Clustering de fenotipos **multidominio en POSTCOVID-Lleida**, la cohorte con el seguimiento más rico, y comparación con TENACITY y CIBERESUCICOVID. Sigue el **mismo procedimiento que el notebook 03** (que descubre en CIBERESUCICOVID), para que los dos resultados sean comparables.

| | |
|---|---|
| Definición | **Función pulmonar** (DLCO, FVC), **esfuerzo** (PM6M), **disnea** (mMRC) y **ánimo** (HADS-A, HADS-D), con peso igual por dominio. Son las variables que Lleida y TENACITY recogen de forma comparable |
| Horizontes (acumulados) | H3 = visita 1 (~3,5 meses); H6 = + visita 2 (~7 meses); H12 = + visita anual. Cada horizonte suma el cambio de cada variable entre la primera y la última medida |
| Inclusión | Superviviente, sin inconsistencias y, en la última visita, **función y (HADS o mMRC)** |
| Descripción (no define) | Astenia, fatiga muscular, FEV1, fibrosis en el TAC (formularios completos), edad, sexo, gravedad y comorbilidad |
| Comparación | **TENACITY comparte las seis variables**: replicación completa. **CIBERESUCICOVID solo comparte DLCO y FVC**: se transfiere la parte respiratoria y los fenotipos se describen con sus síntomas, reingresos y TAC |
| Pacientes compartidos | Los 437 pacientes fusionados con CIBERESUCICOVID cuentan **solo en Lleida** (`cohorte_analisis`), así que no contaminan la comparación con CIBERESUCICOVID |

**Método** (el mismo que en el notebook 03; código en `src/fenotipado_perfil.py`, configuración en `config.yaml → fenotipado_lleida`):
- Distancia de Gower: numéricas normalizadas por un rango fijo (P2,5–P97,5, idéntico en todas las cohortes), sin imputar.
- PAM, con k elegido frente a una prueba nula de permutación.
- Estabilidad bootstrap.
- Replicación: transferencia al medoide frente a clustering nativo, con emparejamiento húngaro.

> **Privacidad.** Solo agregados; celdas < 10 suprimidas. Etiquetas por paciente en `Datos limpios/fenotipos_lleida.parquet`.
""")

md("## 0 · Entorno y contrato")
code("""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import adjusted_rand_score

RAIZ = Path.cwd()
sys.path.insert(0, str(RAIZ))
from src import carga, fenotipado_perfil as FP, pasaporte as P, privacidad

cfg = carga.cargar_config()
PF = FP.perfil(cfg, "fenotipado_lleida")
DESC = PF["descubre"]
REPLICAS = PF["replica"]
SEMILLA = cfg["semilla"]
N_MIN = cfg["privacidad"]["n_minimo_celda"]
COLOR = cfg["colores_registro"]
ETQ = {"CIBERESUCICOVID": "CIBERESUCICOVID", "POSTCOVID_LLEIDA": "POSTCOVID-Lleida", "TENACITY": "TENACITY"}
HORIZONTES = list(PF["horizontes"])
KS = list(range(PF["k_rango"][0], PF["k_rango"][1] + 1))
DATOS = carga.RAIZ / cfg["rutas"]["procesados"]
FIG = RAIZ / cfg["rutas"]["figuras"]; FIG.mkdir(parents=True, exist_ok=True)
TAB = RAIZ / cfg["rutas"]["tablas"]; TAB.mkdir(parents=True, exist_ok=True)
PFX = "04_lleida"

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({"figure.dpi": 100, "axes.titleweight": "bold", "axes.titlesize": 11.5,
                     "axes.spines.top": False, "axes.spines.right": False})
pd.set_option("display.max_columns", 60); pd.set_option("display.width", 220)


def nota(fig, texto: str) -> None:
    fig.text(0.01, -0.01, texto, fontsize=8.5, color="0.35", va="top", ha="left")


def guardar(fig, nombre: str) -> None:
    fig.savefig(FIG / f"{PFX}_{nombre}.png", bbox_inches="tight", dpi=150)


def ic_wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    c = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    m = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return (100 * (c - m), 100 * (c + m))


def nombre_f(c: int) -> str:
    return f"F{c + 1}"


def pct_seguro(serie: pd.Series) -> float:
    \"\"\"% de 1 en una serie 0/1 (NaN si N < N_MIN o si hay entre 1 y N_MIN−1 casos).\"\"\"
    s = serie.dropna(); k = int(s.sum())
    return 100 * k / len(s) if len(s) >= N_MIN and (k == 0 or k >= N_MIN) else np.nan
""")
code("""
pd.DataFrame([
    ("Descubre / replica", f"{DESC} → {', '.join(REPLICAS)}"),
    ("Visitas (tolerancia en días)", {t: d["ventana"] for t, d in PF["visitas"].items()}),
    ("Horizontes", PF["horizontes"]),
    ("Dominios de definición", PF["dominios"]),
    ("Ancla en la última visita", PF["ancla"]),
    ("Rango de normalización", PF["rango_gower"]),
    ("Elección de k", f"k en {KS}: mayor (silueta real − media nula) entre los que superan el p95 ({PF['n_permutaciones']} permutaciones)"),
    ("Estable", f"Jaccard bootstrap ({PF['n_bootstrap']}): ≥ 0,75 estable; 0,5–0,75 patrón; < 0,5 se disuelve"),
    ("Viaja", "ARI transferidas frente a nativas con IC 95 % que excluye 0; Jaccard por par ≥ 0,5"),
], columns=["criterio", "valor"]).set_index("criterio")
""")
md("Criterios fijados **antes** de ver los resultados (`config.yaml → fenotipado_lleida`). Un fenotipo que no los cumple se presenta igualmente, como no replicado o inestable.")

md("## 1 · Pacientes por horizonte")
code("""
pacientes = pd.read_parquet(DATOS / "pacientes.parquet")
medidas = pd.read_parquet(DATOS / "medidas.parquet")
cohortes = [DESC, *REPLICAS]
tablas_var = {(h, c): FP.construir_variables(medidas, pacientes, cfg, PF, h, c) for h in HORIZONTES for c in cohortes}

n_tabla = pd.DataFrame({h: {ETQ[c]: len(tablas_var[(h, c)][0]) for c in cohortes} for h in HORIZONTES})
for (h, c), (t, _) in tablas_var.items():
    limite = PF["visitas"][PF["horizontes"][h][-1]]["ventana"][1]
    assert t["dias_max"].max() <= limite, "fuga de datos futuros"
print("Comprobado: ninguna variable usa datos posteriores a su horizonte.")
print("Variables por horizonte:", {h: len(tablas_var[(h, DESC)][1]) for h in HORIZONTES})
privacidad.suprimir_celdas(n_tabla, N_MIN, HORIZONTES)
""")
code("""
cobertura = pd.concat({h: tablas_var[(h, DESC)][0][tablas_var[(h, DESC)][1]].notna().mean().mul(100).round(0)
                       for h in HORIZONTES}, axis=1)
print(f"% de pacientes de {DESC} con dato en cada variable (Gower usa los pares disponibles; no se imputa):")
cobertura
""")

md("## 2 · Los tres clusterings")
code("""
def ejecutar(h: str, k_forzado: int | None = None, columnas: list[str] | None = None, filtro=None,
             n_perm: int | None = None, n_boot: int | None = None, replicar: bool = True) -> dict:
    \"\"\"Clustering de un horizonte en Lleida + estabilidad + replicación en TENACITY y CIBERESUCICOVID.\"\"\"
    t, cols = tablas_var[(h, DESC)]
    cols = columnas or cols
    if filtro is not None:
        t = t[filtro(t)]
    t = t[t[cols].notna().any(axis=1)]
    X = t[cols].to_numpy(float)
    R, w, cat = FP.parametros_gower(cols, PF)
    D = FP.gower(X, None, R, w, cat)
    nulo = FP.seleccion_k_nulo(X, R, w, cat, KS, n_perm or PF["n_permutaciones"], SEMILLA, D)
    k, estructura = (k_forzado, bool(nulo.set_index("k").loc[k_forzado, "supera_p95"])) if k_forzado else FP.elegir_k(nulo)
    lab, med = FP.pam(D, k, SEMILLA)
    ultima = PF["horizontes"][h][-1]
    ref = t[f"dlco_{ultima}"] if f"dlco_{ultima}" in t else t[[c for c in t.columns if c.startswith("dlco_T")]].mean(axis=1)
    lab, med = FP.ordenar_por_dlco(lab, med, ref.to_numpy(float))
    r = {"h": h, "cols": cols, "R": R, "w": w, "cat": cat, "desc": t, "X": X, "D": D, "nulo": nulo, "k": k,
         "estructura": estructura, "lab": lab, "med": med, "silueta": FP.silueta(D, lab),
         "estab": FP.estabilidad_bootstrap(D, lab, k, n_boot or PF["n_bootstrap"], SEMILLA),
         "techo": {}, "replicas": {}}
    for rep in REPLICAS:
        # ¿Se reconocen los fenotipos solo con lo que la réplica comparte? (reasignar ocultando el resto)
        comp = FP.compartidas(cfg, PF, cols, rep)
        Xc = np.where(comp, X, np.nan)
        r["techo"][rep] = adjusted_rand_score(lab, FP.asignar_medoides(FP.gower(Xc, X[med], R, w, cat)))
        if not replicar:
            continue
        tr = tablas_var[(h, rep)][0]
        if len(tr) < 2 * k:
            continue
        out = P.replicar(X, med, tr.reindex(columns=cols).to_numpy(float), R, w, cat, k,
                         n_boot or PF["n_bootstrap"], SEMILLA)
        out["datos"] = tr
        out["dif"] = P.diferencias_perfil(t, lab, tr.reindex(columns=cols), out["transferidas"],
                                          [c for c, m in zip(cols, comp) if m], k)
        r["replicas"][rep] = out
    return r


resultados = {h: ejecutar(h) for h in HORIZONTES}
for h, r in resultados.items():
    techo = ", ".join(f"{ETQ[c]} {v:.2f}" for c, v in r["techo"].items())
    print(f"{h}: N = {len(r['desc'])}, k = {r['k']}, estructura frente al nulo = {'sí' if r['estructura'] else 'NO'}, "
          f"silueta = {r['silueta']:.2f}, techo de reconocimiento (ARI): {techo}")
""")

md("## 3 · Elección de k: silueta real frente a la nula")
code("""
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
for ax, h in zip(axes, HORIZONTES):
    r = resultados[h]; nl = r["nulo"]
    ax.fill_between(nl["k"], nl["nulo_media"], nl["nulo_p95"], color="0.8", alpha=0.6, label="nulo (media–p95)")
    ax.plot(nl["k"], nl["nulo_media"], color="0.5", lw=1)
    ax.plot(nl["k"], nl["silueta_real"], "o-", color=COLOR[DESC], lw=2, label="real")
    ax.axvline(r["k"], color="0.2", ls="--", lw=1)
    ax.set_title(f"{h} (N={len(r['desc'])}, k={r['k']})"); ax.set_xticks(KS); ax.set_xlabel("k")
axes[0].set_ylabel("Silueta"); axes[0].legend(frameon=False, fontsize=9)
nota(fig, f"POSTCOVID-Lleida. Banda gris: silueta de {PF['n_permutaciones']} conjuntos con columnas permutadas (sin estructura conjunta). "
          "Línea discontinua: k elegido.")
plt.tight_layout(); guardar(fig, "eleccion_k"); plt.show()
""")

md("## 4 · Perfiles de los fenotipos")
code("""
filas_hm = max(len([c for c in r["cols"] if not c.endswith("_cambio")]) for r in resultados.values())
fig, axes = plt.subplots(1, 3, figsize=(16, 1.5 + 0.38 * filas_hm), gridspec_kw={"width_ratios": [0.8, 1.2, 1.5]})
for ax, h in zip(axes, HORIZONTES):
    r = resultados[h]
    pct, n_alt = FP.porcentaje_alterado(r["desc"], r["cols"], r["lab"], cfg)
    tam = pd.Series(r["lab"]).value_counts().sort_index()
    oculta = (n_alt > 0) & (n_alt < N_MIN)
    oculta.loc[:, tam[tam < N_MIN].index] = True
    etiquetas = pct.round(0).astype("Int64").astype(str).mask(oculta, "<10")
    visible = pct.mask(oculta)
    visible.columns = [f"{nombre_f(c)}\\nN={tam[c] if tam[c] >= N_MIN else '<10'}" for c in pct.columns]
    sns.heatmap(visible, annot=etiquetas.values, fmt="", cmap="Reds", vmin=0, vmax=100, cbar=False, ax=ax, linewidths=0.5)
    ax.set_title(h); ax.set_ylabel(""); ax.tick_params(axis="x", rotation=0, labelsize=9); ax.tick_params(axis="y", labelsize=8.5)
nota(fig, "POSTCOVID-Lleida. % de pacientes con la variable alterada (DLCO/FVC < 80 %, PM6M < 400 m, mMRC ≥ 2, HADS ≥ 8). "
          "F1 = DLCO más alta en la última visita.")
plt.tight_layout(); guardar(fig, "perfiles"); plt.show()
""")
code("""
filas = []
for h, r in resultados.items():
    med_ = r["desc"][r["cols"]].groupby(r["lab"]).median()
    for c, fila in med_.iterrows():
        if (r["lab"] == c).sum() >= N_MIN:
            filas.append({"horizonte": h, "fenotipo": nombre_f(c), "N": int((r["lab"] == c).sum()), **fila.round(1).to_dict()})
medianas = pd.DataFrame(filas).set_index(["horizonte", "fenotipo"])
medianas.to_csv(TAB / f"{PFX}_medianas_fenotipos.csv")
print("Medianas por fenotipo:")
medianas
""")

md("""
### Fichas: descripción con variables que no definen el fenotipo

Cada cohorte se describe con lo que recoge:
- **"Alguna"** (eventos, hallazgos de imagen): % que lo presenta en alguna visita hasta el horizonte; la imagen, solo entre quienes tienen TAC.
- **"Última"** (FEV1, síntomas, cuestionarios): estado en la visita más reciente con medida.
- **Variables del ingreso.**
""")
code("""
VARS_DEF = set(FP.variables_definicion(PF))
IMAGEN = ("tac_", "tractos", "fibrosis")


def ficha(h: str, cohorte: str, datos: pd.DataFrame, etiquetas: np.ndarray, k: int) -> pd.DataFrame:
    visitas = PF["horizontes"][h]
    p = pacientes.set_index("subject_id").reindex(datos.index)
    seguimiento = {}
    for regla in ["alguna", "ultima"]:
        for v in PF["descripcion"][regla]:
            if v in VARS_DEF:
                continue
            if v.startswith(IMAGEN) or FP.visitas_que_recogen(cfg, PF, v, cohorte):
                seguimiento[v] = FP.estado_hasta(medidas, cfg, PF, cohorte, datos.index, v, visitas, regla)
    filas = {}
    for c in range(k):
        s = etiquetas == c
        sub = p[s]
        col = {"N": int(s.sum())}
        if s.sum() >= N_MIN:
            col["Edad ≥ 65 (%)"] = pct_seguro(sub["grupo_edad"].isin(["65-74", ">=75"]).astype(float).where(sub["grupo_edad"].notna()))
            col["Mujeres (%)"] = pct_seguro((sub["sexo"] == 1).astype(float).where(sub["sexo"].notna()))
            col["Estancia (mediana, días)"] = sub["estancia_hosp_dias"].median()
            for v in PF["descripcion"]["ingreso_binarias"]:
                col[f"{v.replace('nu_', '')} (%)"] = pct_seguro(sub[v])
            for v, serie in seguimiento.items():
                col[f"{v} (% sí / alterado)"] = pct_seguro(serie[s])
        filas[nombre_f(c)] = col
    return pd.DataFrame(filas).round(1)


fichas = {(h, DESC): ficha(h, DESC, r["desc"], r["lab"], r["k"]) for h, r in resultados.items()}
for h, r in resultados.items():
    print(f"\\n=== {h} · {DESC} ===")
    display(fichas[(h, DESC)].dropna(how="all"))
""")

md("## 5 · Estabilidad")
code("""
filas = []
for h, r in resultados.items():
    for _, e in r["estab"].iterrows():
        filas.append({"horizonte": h, "fenotipo": nombre_f(int(e["cluster"])), "N": int(e["n"]),
                      "Jaccard medio": round(e["jaccard_medio"], 2), "Jaccard p05": round(e["jaccard_p05"], 2)})
estab = pd.DataFrame(filas)
estab["veredicto"] = pd.cut(estab["Jaccard medio"], [0, 0.5, 0.75, 1.01], labels=["se disuelve", "patrón", "estable"])
estab.to_csv(TAB / f"{PFX}_estabilidad.csv", index=False)
print("Estabilidad entre centros: no aplica (POSTCOVID-Lleida es un solo centro).")
privacidad.suprimir_celdas(estab.set_index(["horizonte", "fenotipo"]), N_MIN, ["N"])
""")

md("""
## 6 · Comparación con TENACITY y CIBERESUCICOVID

### 6.1 ¿Se reconocen los fenotipos con lo que comparte cada cohorte?

En Lleida, cada paciente se reasigna a su medoide **ocultando las variables que la otra cohorte no recoge**. Ese ARI es el **techo** de lo que la transferencia puede reproducir:
- TENACITY lo comparte todo: techo 1.
- CIBERESUCICOVID solo comparte DLCO y FVC: su techo dice cuánto del fenotipo multidominio es reconocible con la función pulmonar.
""")
code("""
pd.DataFrame({h: {f"techo {ETQ[c]}": round(v, 2) for c, v in r["techo"].items()} for h, r in resultados.items()}).T
""")
md("### 6.2 Replicación: transferidas frente a nativas")
code("""
filas = []
for h, r in resultados.items():
    for rep, out in r["replicas"].items():
        fila = {"horizonte": h, "replica en": ETQ[rep], "N": out["n"], "k": r["k"], "techo": round(r["techo"][rep], 2),
                "ARI": round(out["ari"], 2), "ARI IC95": f"{out['ari_ic'][0]:.2f} a {out['ari_ic'][1]:.2f}"}
        for c in range(r["k"]):
            fila[f"Jaccard {nombre_f(c)}"] = f"{out['jaccard'][c]:.2f} ({out['jaccard_ic'][c][0]:.2f}–{out['jaccard_ic'][c][1]:.2f})"
        dif = out["dif"].abs()
        fila["% perfiles con |d| < 0,5"] = round(100 * (dif < 0.5).sum().sum() / max(dif.notna().sum().sum(), 1))
        filas.append(fila)
replicacion = pd.DataFrame(filas).set_index(["horizonte", "replica en"])
replicacion.to_csv(TAB / f"{PFX}_replicacion.csv")
replicacion
""")
code("""
fig, ax = plt.subplots(figsize=(8, 4))
for i, rep in enumerate(REPLICAS):
    xs, ys, lo, hi, tx = [], [], [], [], []
    for j, h in enumerate(HORIZONTES):
        out = resultados[h]["replicas"].get(rep)
        if out is None:
            continue
        x = j + (i - (len(REPLICAS) - 1) / 2) * 0.2
        xs.append(x); ys.append(out["ari"]); tx.append(resultados[h]["techo"][rep])
        lo.append(out["ari"] - out["ari_ic"][0]); hi.append(out["ari_ic"][1] - out["ari"])
    ax.errorbar(xs, ys, yerr=[lo, hi], fmt="o", ms=8, capsize=4, color=COLOR[rep], label=ETQ[rep])
    ax.plot(xs, tx, "_", color=COLOR[rep], ms=18, mew=2.5)
ax.axhline(0, color="0.3", lw=1); ax.set_xticks(range(len(HORIZONTES)), HORIZONTES); ax.set_ylim(-0.1, 1.05)
ax.set_ylabel("ARI transferidas frente a nativas (IC 95 %)"); ax.set_title("¿Viajan los fenotipos de Lleida?")
ax.legend(frameon=False, fontsize=8.5)
nota(fig, "Puntos: ARI de replicación. Rayas: techo (ARI máximo esperable con las variables que comparte esa cohorte, §6.1). "
          "ARI = 0: concordancia al azar.")
plt.tight_layout(); guardar(fig, "replicacion_ari"); plt.show()
""")
md("""
### 6.3 Los fenotipos de Lleida descritos en las otras cohortes

Los pacientes de TENACITY y CIBERESUCICOVID se asignan al fenotipo más cercano, con lo que comparten, y se describen con todo lo que cada cohorte recoge. En CIBERESUCICOVID eso incluye fatiga, resolución clínica, reingresos, urgencias y hallazgos del TAC.
""")
code("""
for h, r in resultados.items():
    bloques = {ETQ[DESC]: fichas[(h, DESC)]}
    for rep, out in r["replicas"].items():
        fichas[(h, rep)] = ficha(h, rep, out["datos"], out["transferidas"], r["k"])
        bloques[ETQ[rep]] = fichas[(h, rep)]
    print(f"\\n=== {h} ===")
    display(pd.concat(bloques, axis=1).dropna(how="all"))
pd.concat({f"{h}|{c}": f for (h, c), f in fichas.items()}, names=["horizonte|cohorte", "variable"]).to_csv(TAB / f"{PFX}_fichas.csv")
""")

md("""
## 7 · Comparación entre horizontes

### 7.1 Pasaporte conjunto
""")
code("""
pasaporte = pd.DataFrame([{
    "horizonte": h, "N": len(r["desc"]), "k": r["k"], "estructura": "sí" if r["estructura"] else "no",
    "silueta − nula": round(r["silueta"] - r["nulo"].set_index("k").loc[r["k"], "nulo_media"], 3),
    "Jaccard estabilidad (mín)": round(r["estab"]["jaccard_medio"].min(), 2),
    **{f"techo {ETQ[c]}": round(v, 2) for c, v in r["techo"].items()},
    **{f"ARI {ETQ[rep]}": f"{o['ari']:.2f} [{o['ari_ic'][0]:.2f}, {o['ari_ic'][1]:.2f}]" for rep, o in r["replicas"].items()},
} for h, r in resultados.items()]).set_index("horizonte")
pasaporte.to_csv(TAB / f"{PFX}_pasaporte.csv")
pasaporte
""")
md("### 7.2 Transiciones entre horizontes")
code("""
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
filas = []
for ax, (h1, h2) in zip(axes, [("H3", "H6"), ("H6", "H12")]):
    r1, r2 = resultados[h1], resultados[h2]
    e1 = pd.Series([nombre_f(c) for c in r1["lab"]], index=r1["desc"].index)
    e2 = pd.Series([nombre_f(c) for c in r2["lab"]], index=r2["desc"].index)
    tabla, ari, n = P.transiciones(e1, e2)
    filas.append({"transición": f"{h1} → {h2}", "pacientes comunes": n, "ARI": round(ari, 2)})
    if n:
        pct = tabla.div(tabla.sum(axis=1), axis=0) * 100
        anot = pct.round(0).astype(int).astype(str).mask(tabla < N_MIN, "<10")
        sns.heatmap(pct.mask(tabla < N_MIN), annot=anot.values, fmt="", cmap="Blues", vmin=0, vmax=100, cbar=False, ax=ax, linewidths=0.5)
    ax.set_xlabel(f"Fenotipo en {h2}"); ax.set_ylabel(f"Fenotipo en {h1}"); ax.set_title(f"{h1} → {h2} (N={n})")
nota(fig, "POSTCOVID-Lleida. % por fila: a qué fenotipo pasan los pacientes de cada fenotipo del horizonte previo.")
plt.tight_layout(); guardar(fig, "transiciones"); plt.show()
pd.DataFrame(filas).set_index("transición")
""")
md("""
### 7.3 ¿El fenotipo a 3 meses anticipa el año?

Para los pacientes fenotipados en H3 se calcula el % con cada desenlace a 12 meses (DLCO < 80 %, HADS-A ≥ 8, mMRC ≥ 2) según su fenotipo, en cada cohorte que recoge ese desenlace. En TENACITY y CIBERESUCICOVID se usan las etiquetas transferidas.
""")
code("""
r = resultados["H3"]
grupos = {DESC: (r["desc"], r["lab"])}
grupos.update({rep: (o["datos"], o["transferidas"]) for rep, o in r["replicas"].items()})
filas, pruebas = [], []
for coh, (datos, lab) in grupos.items():
    for nombre, v, visitas, regla in PF["desenlaces"]:
        if not FP.visitas_que_recogen(cfg, PF, v, coh) & set(visitas):
            continue
        y = FP.estado_hasta(medidas, cfg, PF, coh, datos.index, v, visitas, regla)
        ok = y.notna().values
        if ok.sum() < N_MIN:
            continue
        ct = pd.crosstab(lab[ok], y.values[ok])
        if ct.shape[1] == 2 and (ct.values > 0).all():
            pruebas.append({"cohorte": ETQ[coh], "desenlace": nombre, "N": int(ok.sum()),
                            "p (χ²)": f"{stats.chi2_contingency(ct)[1]:.2g}"})
        for c in range(r["k"]):
            s = y.values[ok & (lab == c)]
            k = int(np.nansum(s))
            if len(s) < N_MIN or 0 < k < N_MIN:
                continue
            lo, hi = ic_wilson(k, len(s))
            filas.append({"cohorte": ETQ[coh], "desenlace": nombre, "fenotipo H3": nombre_f(c), "N": len(s),
                          "%": round(100 * k / len(s), 1), "IC95": f"{lo:.0f}–{hi:.0f}"})
pronostico = pd.DataFrame(filas)
pronostico.to_csv(TAB / f"{PFX}_pronostico_h3.csv", index=False)
display(pd.DataFrame(pruebas).set_index(["cohorte", "desenlace"]) if pruebas else "sin pruebas evaluables")
pronostico.set_index(["cohorte", "desenlace", "fenotipo H3"]) if len(pronostico) else "sin desenlaces evaluables"
""")

md("""
## 8 · Sensibilidad

Se repite cada horizonte cambiando una sola cosa, con menos permutaciones y remuestreos para no alargar la ejecución:
- **k = 3 fijo**;
- **solo un dominio**, para cada dominio (función, esfuerzo, disnea, ánimo): ¿qué dominio sostiene la partición?;
- **sin función pulmonar** (solo esfuerzo, disnea y ánimo);
- **sin variables de cambio**.

La columna "ARI frente al principal" mide cuánto se parece la partición de cada escenario a la principal.
""")
code("""
rapido = dict(n_perm=20, n_boot=50, replicar=False)
filas = []
for h, base in resultados.items():
    cols = base["cols"]
    escenarios = {"principal": None, f"k = {PF['k_comparacion']} fijo": dict(k_forzado=PF["k_comparacion"])}
    for dom, vs in PF["dominios"].items():
        sub = [c for c in cols if c.rsplit("_", 1)[0] in vs]
        if sub:
            escenarios[f"solo {dom}"] = dict(columnas=sub)
    escenarios["sin función"] = dict(columnas=[c for c in cols if c.rsplit("_", 1)[0] not in PF["dominios"]["funcion"]])
    if any(c.endswith("_cambio") for c in cols):
        escenarios["sin cambios"] = dict(columnas=[c for c in cols if not c.endswith("_cambio")])
    for nombre, kw in escenarios.items():
        r = base if kw is None else ejecutar(h, **kw, **rapido)
        ari_vs = np.nan
        if kw is not None:
            comunes = r["desc"].index.intersection(base["desc"].index)
            ari_vs = adjusted_rand_score(pd.Series(base["lab"], index=base["desc"].index).loc[comunes],
                                         pd.Series(r["lab"], index=r["desc"].index).loc[comunes])
        filas.append({"horizonte": h, "escenario": nombre, "N": len(r["desc"]), "k": r["k"],
                      "estructura": "sí" if r["estructura"] else "no", "silueta": round(r["silueta"], 2),
                      "silueta − nula": round(r["silueta"] - r["nulo"].set_index("k").loc[r["k"], "nulo_media"], 2),
                      "Jaccard estab. (media)": round(r["estab"]["jaccard_medio"].mean(), 2),
                      "ARI frente al principal": round(ari_vs, 2)})
sensibilidad = pd.DataFrame(filas).set_index(["horizonte", "escenario"])
sensibilidad.to_csv(TAB / f"{PFX}_sensibilidad.csv")
privacidad.suprimir_celdas(sensibilidad, N_MIN, ["N"])
""")

md("## 9 · Guardar las etiquetas")
code("""
filas = []
for h, r in resultados.items():
    D_med = FP.gower(r["X"], r["X"][r["med"]], r["R"], r["w"], r["cat"])
    filas.append(pd.DataFrame({"subject_id": r["desc"].index, "horizonte": h, "cohorte_analisis": DESC,
                               "fenotipo": [nombre_f(c) for c in r["lab"]], "tipo": "descubrimiento",
                               "distancia_medoide": D_med[np.arange(len(r["lab"])), r["lab"]]}))
    for rep, out in r["replicas"].items():
        Xr = out["datos"].reindex(columns=r["cols"]).to_numpy(float)
        D_r = FP.gower(Xr, r["X"][r["med"]], r["R"], r["w"], r["cat"])
        for tipo, lab in [("transferida", out["transferidas"]), ("nativa", out["nativas"])]:
            filas.append(pd.DataFrame({"subject_id": out["datos"].index, "horizonte": h, "cohorte_analisis": rep,
                                       "fenotipo": [nombre_f(c) for c in lab], "tipo": tipo,
                                       "distancia_medoide": D_r[np.arange(len(lab)), lab]}))
fenotipos = pd.concat(filas, ignore_index=True)
fenotipos.to_parquet(DATOS / "fenotipos_lleida.parquet", index=False)
print(f"Guardado {DATOS.name}/fenotipos_lleida.parquet: {len(fenotipos):,} filas")
privacidad.suprimir_celdas(fenotipos.groupby(["horizonte", "cohorte_analisis", "tipo"]).size().unstack(fill_value=0),
                           N_MIN, ["descubrimiento", "transferida", "nativa"])
""")

md("## 10 · Resumen")
md("""
**Resultados (ejecución con semilla 2026; ver §3–§8).**

1. **Estructura multidominio débil.** En **H3 no hay estructura** frente al nulo: la silueta queda igual que la de datos permutados (−0,01), y el k = 4 elegido es un "mejor de lo peor" sin validez. En **H6** hay estructura modesta (+0,12, k = 3) y en **H12** es mínima (+0,06, k = 3). La estabilidad es de **patrón** (Jaccard 0,52–0,78): solo F1 de H6 es estable.
2. **Por qué: los dominios son casi independientes.** Por separado, cada dominio sí tiene estructura (sensibilidad §8): ánimo (+0,18 a +0,41), función (+0,15 a +0,18) y esfuerzo y disnea en H6 y H12. Al combinarlos, la estructura se diluye: un paciente puede tener la DLCO baja y el ánimo normal, o al revés, en cualquier combinación. Confirma lo que ya mostraba el análisis exploratorio (correlaciones |ρ| ≤ 0,23).
3. **Perfiles (H6 y H12).** Son nombres borradores para que los valide el equipo:
   - **F1 · "buena evolución":** la mejor función y PM6M, sin disnea ni ansiedad o depresión. Es el grupo más estable.
   - **F2 · "afectación leve mixta":** DLCO algo baja, disnea leve y HADS en el límite bajo.
   - **F3 · "afectación multidominio persistente":** FVC más baja, peor PM6M, disnea mMRC 2 y, en H12, HADS-A y HADS-D en rango de caso (8 y 6). Tiene más traqueotomías e hipertensión, y un 56 % de fatiga muscular a 12 meses. Es pequeño (42 pacientes en H12).
4. **Replicación débil incluso en TENACITY**, que comparte todas las variables (techo = 1): ARI 0,15–0,31 con IC muy anchos (N = 58–148). En CIBERESUCICOVID, el techo es ≈ 0,05: los fenotipos multidominio no son reconocibles solo con DLCO y FVC. **No se puede afirmar que estos fenotipos viajen.**
5. **Lo más útil: el gradiente de H3 anticipa el año, en las tres cohortes.** Aunque H3 no forma grupos discretos, la asignación ordena a los pacientes por gravedad, y esa ordenación predice los desenlaces a 12 meses:
   - **DLCO < 80 %:** Lleida 35 % → 53 % → 60 % → 79 % (p = 0,0003); CIBERESUCICOVID 58 % → 55 % → 68 % → 85 % (p = 0,005); TENACITY p = 0,035.
   - **HADS-A ≥ 8** (p = 0,0008) y **mMRC ≥ 2** (p = 0,003) en Lleida.

**Implicaciones**
- **Mejor como escala que como tipos.** En Lleida, las secuelas se describen mejor como un **gradiente de gravedad multidominio** (o como dominios separados) que como fenotipos discretos.
- **Valor práctico.** Aun sin fenotipos discretos sólidos, el perfil a 3 meses tiene **valor pronóstico** y se repite en CIBERESUCICOVID y TENACITY. Es el mejor punto de partida para el clasificador.
- **Comparación con el notebook 03.** En CIBERESUCICOVID, la estructura la daban los síntomas binarios; en Lleida, con variables continuas de cuatro dominios, no aparecen grupos nítidos. Ninguno de los dos fenotipados replica bien fuera de su cohorte.

**Limitaciones:**
- **Una sola cohorte de descubrimiento con N mediano** (unos 300–440 por horizonte) y un solo centro: no se puede valorar la estabilidad entre centros.
- **TENACITY es pequeña en H6 y H12** (unos 60): los IC de replicación serán anchos.
- **La comparación con CIBERESUCICOVID es parcial:** solo DLCO y FVC (techo en §6.1).
- **Calidad de vida, sueño y cognición todavía no están limpios ni armonizados.** Ampliarían la definición multidominio (SF-12, Epworth, BC-CCI/MoCA).
- **Abandono informativo** (notebook 01, §8): H12 sobrerrepresenta a quien sigue en seguimiento.
- **No se imputa:** quien tiene pocas medidas pesa con menos información.
""")

nb = nbf.v4.new_notebook()
nb["cells"] = celdas
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                  "language_info": {"name": "python"}}
salida = Path(__file__).resolve().parents[1] / "04_fenotipado_lleida.ipynb"
nbf.write(nb, salida)
print(f"Escrito {salida} ({len(celdas)} celdas)")
