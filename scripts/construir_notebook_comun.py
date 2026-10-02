"""Genera `05_fenotipado_comun.ipynb`: clustering de todas las cohortes juntas con sus variables comunes.

Uso: python scripts/construir_notebook_comun.py
(después: python -m nbconvert --to notebook --execute --inplace 05_fenotipado_comun.ipynb)
"""
from pathlib import Path

import nbformat as nbf

celdas: list = []
md = lambda texto: celdas.append(nbf.v4.new_markdown_cell(texto.strip()))
code = lambda texto: celdas.append(nbf.v4.new_code_cell(texto.strip()))

md("""
# Reto 2 · Fenotipos comunes a todas las cohortes (3, 6 y 12 meses)

**Notebook 05.** Clustering de **CIBERESUCICOVID, POSTCOVID-Lleida y TENACITY juntas**, usando solo las variables que **las tres miden igual**. Es la base para un modelo predictivo que sirva para pacientes de cualquier cohorte (el modelo se ajustará después; aquí solo se prepara la tabla).

| | |
|---|---|
| Definición | **Difusión** (DLCO) y **espirometría** (FVC, FEV1), con peso igual por dominio: FEV1 no duplica el peso de la espirometría (ρ ≈ 0,88 con FVC) |
| Horizontes (acumulados) | H3 = visita de ~3 meses; H6 = + ~6 meses; H12 = + ~12 meses. Con el cambio de cada variable entre la primera y la última medida |
| Inclusión | Superviviente, sin inconsistencias, cumplimentación ≥ 50 % (CIBERESUCICOVID) y función pulmonar en la última visita |
| Cohortes | Las tres juntas. Virgen del Rocío queda fuera: mide al mes y no tiene visitas a 3, 6 y 12 meses. Los pacientes compartidos cuentan una sola vez (`cohorte_analisis`) |
| Validación | **Dejando fuera cada cohorte**: se agrupa con las otras dos y se comprueba si la cohorte excluida reproduce los mismos grupos. También se comprueba que los grupos no se limitan a separar cohortes |
| Descripción (no define) | Con lo que recoge cada cohorte: síntomas, reingresos y TAC (CIBERESUCICOVID); HADS, mMRC, PM6M y fibrosis (Lleida, TENACITY); datos del ingreso |

**Método** (el mismo que en los notebooks 03 y 04; código en `src/fenotipado_perfil.py`, configuración en `config.yaml → fenotipado_comun`):
- Gower con rangos fijos (P2,5–P97,5 de las tres cohortes), sin imputar.
- PAM, con k elegido frente a una prueba nula de permutación.
- Estabilidad bootstrap.

> **Privacidad.** Solo agregados; celdas < 10 suprimidas. Etiquetas y tabla base del modelo en `Datos limpios/` (fuera del repositorio).
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
PF = FP.perfil(cfg, "fenotipado_comun")
COHORTES = PF["cohortes"]
SEMILLA = cfg["semilla"]
N_MIN = cfg["privacidad"]["n_minimo_celda"]
COLOR = cfg["colores_registro"]
ETQ = {"CIBERESUCICOVID": "CIBERESUCICOVID", "POSTCOVID_LLEIDA": "POSTCOVID-Lleida", "TENACITY": "TENACITY"}
HORIZONTES = list(PF["horizontes"])
KS = list(range(PF["k_rango"][0], PF["k_rango"][1] + 1))
DATOS = carga.RAIZ / cfg["rutas"]["procesados"]
FIG = RAIZ / cfg["rutas"]["figuras"]; FIG.mkdir(parents=True, exist_ok=True)
TAB = RAIZ / cfg["rutas"]["tablas"]; TAB.mkdir(parents=True, exist_ok=True)
PFX = "05_comun"

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


def cramer_v(tabla: pd.DataFrame) -> float:
    chi2 = stats.chi2_contingency(tabla)[0]
    return float(np.sqrt(chi2 / (tabla.values.sum() * (min(tabla.shape) - 1))))
""")
code("""
pd.DataFrame([
    ("Cohortes (juntas)", ", ".join(COHORTES)),
    ("Visitas (tolerancia en días)", {t: d["ventana"] for t, d in PF["visitas"].items()}),
    ("Horizontes", PF["horizontes"]),
    ("Dominios de definición", PF["dominios"]),
    ("Rango de normalización", PF["rango_gower"]),
    ("Elección de k", f"k en {KS}: mayor (silueta real − media nula) entre los que superan el p95 ({PF['n_permutaciones']} permutaciones)"),
    ("Estable", f"Jaccard bootstrap ({PF['n_bootstrap']}): ≥ 0,75 estable; 0,5–0,75 patrón; < 0,5 se disuelve"),
    ("Viaja (dejando fuera una cohorte)", "ARI transferidas frente a nativas en la cohorte excluida, con IC 95 % que excluye 0"),
    ("No separa cohortes", "V de Cramér fenotipo × cohorte pequeña (< 0,3) y perfiles parecidos entre cohortes"),
], columns=["criterio", "valor"]).set_index("criterio")
""")
md("Criterios fijados **antes** de ver los resultados (`config.yaml → fenotipado_comun`).")

md("## 1 · Pacientes y comparabilidad de las variables")
code("""
pacientes = pd.read_parquet(DATOS / "pacientes.parquet")
medidas = pd.read_parquet(DATOS / "medidas.parquet")
tablas_var = {}
for h in HORIZONTES:
    partes = [FP.construir_variables(medidas, pacientes, cfg, PF, h, c) for c in COHORTES]
    tablas_var[h] = (pd.concat([t for t, _ in partes]), partes[0][1])
    limite = PF["visitas"][PF["horizontes"][h][-1]]["ventana"][1]
    assert tablas_var[h][0]["dias_max"].max() <= limite, "fuga de datos futuros"
    assert tablas_var[h][0].index.is_unique, "paciente duplicado entre cohortes"
print("Comprobado: sin datos posteriores al horizonte y sin pacientes duplicados entre cohortes.")
n_tabla = pd.DataFrame({h: tablas_var[h][0]["cohorte_analisis"].astype(str).value_counts().rename(ETQ) for h in HORIZONTES})
n_tabla.loc["Total"] = n_tabla.sum()
privacidad.suprimir_celdas(n_tabla, N_MIN, HORIZONTES)
""")
code("""
# ¿Se miden igual en las tres cohortes? Distribución de cada variable en la última visita de H12
t12, _ = tablas_var["H12"]
fig, axes = plt.subplots(1, 3, figsize=(15, 3.8), sharey=True)
for ax, v in zip(axes, ["dlco", "fvc", "fev1"]):
    for c in COHORTES:
        s = t12.loc[t12["cohorte_analisis"] == c, f"{v}_T12"].dropna()
        if len(s) >= N_MIN:
            sns.kdeplot(s, ax=ax, color=COLOR[c], lw=2, label=f"{ETQ[c]} (N={len(s)}, mediana {s.median():.0f})", clip=(20, 160))
    ax.axvline(80, color="0.3", ls="--", lw=1); ax.set_xlabel(f"{v.upper()} a 12 meses (% predicho)"); ax.legend(frameon=False, fontsize=8)
axes[0].set_ylabel("Densidad")
nota(fig, "Pacientes de H12. Las diferencias de nivel entre cohortes son reales (casuística distinta), no de escala: "
          "las tres miden en % del predicho.")
plt.tight_layout(); guardar(fig, "distribuciones"); plt.show()
""")

md("## 2 · Los tres clusterings (todas las cohortes juntas)")
code("""
def ejecutar(h: str, k_forzado: int | None = None, columnas: list[str] | None = None, excluir: str | None = None,
             filtro=None, n_perm: int | None = None, n_boot: int | None = None) -> dict:
    \"\"\"Clustering conjunto de un horizonte: k contra el nulo, PAM y estabilidad bootstrap.\"\"\"
    t, cols = tablas_var[h]
    cols = columnas or cols
    if excluir is not None:
        t = t[t["cohorte_analisis"] != excluir]
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
    lab, med = FP.ordenar_por_dlco(lab, med, t[f"dlco_{ultima}"].to_numpy(float))
    return {"h": h, "cols": cols, "R": R, "w": w, "cat": cat, "desc": t, "X": X, "D": D, "nulo": nulo, "k": k,
            "estructura": estructura, "lab": lab, "med": med, "silueta": FP.silueta(D, lab),
            "estab": FP.estabilidad_bootstrap(D, lab, k, n_boot or PF["n_bootstrap"], SEMILLA)}


resultados = {h: ejecutar(h) for h in HORIZONTES}
for h, r in resultados.items():
    print(f"{h}: N = {len(r['desc'])}, k = {r['k']}, estructura frente al nulo = {'sí' if r['estructura'] else 'NO'}, "
          f"silueta = {r['silueta']:.2f} (nula {r['nulo'].set_index('k').loc[r['k'], 'nulo_media']:.2f})")
""")

md("## 3 · Elección de k: silueta real frente a la nula")
code("""
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
for ax, h in zip(axes, HORIZONTES):
    r = resultados[h]; nl = r["nulo"]
    ax.fill_between(nl["k"], nl["nulo_media"], nl["nulo_p95"], color="0.8", alpha=0.6, label="nulo (media–p95)")
    ax.plot(nl["k"], nl["nulo_media"], color="0.5", lw=1)
    ax.plot(nl["k"], nl["silueta_real"], "o-", color="#444", lw=2, label="real")
    ax.axvline(r["k"], color="0.2", ls="--", lw=1)
    ax.set_title(f"{h} (N={len(r['desc'])}, k={r['k']})"); ax.set_xticks(KS); ax.set_xlabel("k")
axes[0].set_ylabel("Silueta"); axes[0].legend(frameon=False, fontsize=9)
nota(fig, f"Tres cohortes juntas. Banda gris: silueta de {PF['n_permutaciones']} conjuntos con columnas permutadas. "
          "Línea discontinua: k elegido.")
plt.tight_layout(); guardar(fig, "eleccion_k"); plt.show()
""")

md("## 4 · Perfiles de los fenotipos")
code("""
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), gridspec_kw={"width_ratios": [0.8, 1.1, 1.3]})
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
    ax.set_title(h); ax.set_ylabel(""); ax.tick_params(axis="x", rotation=0, labelsize=9)
nota(fig, "Tres cohortes juntas. % de pacientes con DLCO, FVC o FEV1 < 80 % del predicho. F1 = DLCO más alta en la última visita.")
plt.tight_layout(); guardar(fig, "perfiles"); plt.show()
""")
code("""
filas = []
for h, r in resultados.items():
    for c, fila in r["desc"][r["cols"]].groupby(r["lab"]).median().iterrows():
        if (r["lab"] == c).sum() >= N_MIN:
            filas.append({"horizonte": h, "fenotipo": nombre_f(c), "N": int((r["lab"] == c).sum()), **fila.round(1).to_dict()})
medianas = pd.DataFrame(filas).set_index(["horizonte", "fenotipo"])
medianas.to_csv(TAB / f"{PFX}_medianas_fenotipos.csv")
medianas
""")

md("""
## 5 · ¿Los fenotipos separan pacientes o cohortes?

Si los grupos solo reflejaran la cohorte (por ejemplo, "los de TENACITY" frente a "los de CIBERESUCICOVID"), no servirían para predecir pacientes de cualquier cohorte. Se comprueban dos cosas:
- **Composición:** % de cada cohorte en cada fenotipo y V de Cramér.
- **Perfil por cohorte:** cómo es cada fenotipo dentro de cada cohorte. Si el fenotipo es real, las medianas deben ser parecidas.
""")
code("""
filas = []
for h, r in resultados.items():
    coh = r["desc"]["cohorte_analisis"].astype(str).map(ETQ).values
    ct = pd.crosstab(pd.Series([nombre_f(c) for c in r["lab"]], name="fenotipo"), pd.Series(coh, name="cohorte"))
    filas.append({"horizonte": h, "V de Cramér (fenotipo × cohorte)": round(cramer_v(ct), 2)})
    pct = (ct.div(ct.sum(axis=0), axis=1) * 100).round(1).mask(ct < N_MIN)
    print(f"\\n=== {h}: % de cada cohorte en cada fenotipo (columnas suman 100) ===")
    display(pct)
pd.DataFrame(filas).set_index("horizonte")
""")
code("""
filas = []
for h, r in resultados.items():
    d = r["desc"].assign(fenotipo=[nombre_f(c) for c in r["lab"]])
    ultima = PF["horizontes"][h][-1]
    for (fen, coh), g in d.groupby(["fenotipo", "cohorte_analisis"], observed=True):
        if len(g) >= N_MIN:
            filas.append({"horizonte": h, "fenotipo": fen, "cohorte": ETQ[str(coh)], "N": len(g),
                          **{f"{v}_{ultima} (mediana)": round(g[f"{v}_{ultima}"].median(), 1) for v in ["dlco", "fvc", "fev1"]}})
perfil_cohorte = pd.DataFrame(filas).set_index(["horizonte", "fenotipo", "cohorte"])
perfil_cohorte.to_csv(TAB / f"{PFX}_perfil_por_cohorte.csv")
perfil_cohorte
""")

md("## 6 · Estabilidad y validación dejando fuera cada cohorte")
code("""
filas = []
for h, r in resultados.items():
    for _, e in r["estab"].iterrows():
        filas.append({"horizonte": h, "fenotipo": nombre_f(int(e["cluster"])), "N": int(e["n"]),
                      "Jaccard medio": round(e["jaccard_medio"], 2), "Jaccard p05": round(e["jaccard_p05"], 2)})
estab = pd.DataFrame(filas)
estab["veredicto"] = pd.cut(estab["Jaccard medio"], [0, 0.5, 0.75, 1.01], labels=["se disuelve", "patrón", "estable"])
estab.to_csv(TAB / f"{PFX}_estabilidad.csv", index=False)
privacidad.suprimir_celdas(estab.set_index(["horizonte", "fenotipo"]), N_MIN, ["N"])
""")
md("""
**Dejar fuera una cohorte** (la prueba principal de este notebook). Para cada horizonte y cada cohorte:
1. Se agrupa solo con las **otras dos** cohortes (mismo k).
2. La cohorte excluida se **transfiere** a esos medoides.
3. Se compara:
   - con su **propio clustering nativo** (ARI con IC bootstrap): ¿la cohorte nueva tiene la misma estructura?;
   - con las **etiquetas del clustering conjunto** (ARI): ¿cambian sus etiquetas por haberla incluido?
""")
code("""
filas = []
loco = {}
for h, r in resultados.items():
    t = r["desc"]
    for c in COHORTES:
        dentro = (t["cohorte_analisis"] == c).values
        if dentro.sum() < 2 * r["k"]:
            continue
        X_otros, X_c = r["X"][~dentro], r["X"][dentro]
        D_otros = r["D"][np.ix_(~dentro, ~dentro)]
        lab_o, med_o = FP.pam(D_otros, r["k"], SEMILLA)
        lab_o, med_o = FP.ordenar_por_dlco(lab_o, med_o, t.loc[~dentro, f"dlco_{PF['horizontes'][h][-1]}"].to_numpy(float))
        out = P.replicar(X_otros, med_o, X_c, r["R"], r["w"], r["cat"], r["k"], PF["n_bootstrap"], SEMILLA)
        loco[(h, c)] = out
        filas.append({"horizonte": h, "cohorte excluida": ETQ[c], "N": int(dentro.sum()), "k": r["k"],
                      "ARI transferidas frente a nativas": round(out["ari"], 2),
                      "IC95": f"{out['ari_ic'][0]:.2f} a {out['ari_ic'][1]:.2f}",
                      "ARI frente al clustering conjunto": round(adjusted_rand_score(r["lab"][dentro], out["transferidas"]), 2),
                      "Jaccard por fenotipo": " / ".join(f"{j:.2f}" for j in out["jaccard"])})
validacion = pd.DataFrame(filas).set_index(["horizonte", "cohorte excluida"])
validacion.to_csv(TAB / f"{PFX}_validacion_loco.csv")
validacion
""")
code("""
fig, ax = plt.subplots(figsize=(8, 4))
for i, c in enumerate(COHORTES):
    xs, ys, lo, hi = [], [], [], []
    for j, h in enumerate(HORIZONTES):
        out = loco.get((h, c))
        if out is None:
            continue
        xs.append(j + (i - 1) * 0.18); ys.append(out["ari"])
        lo.append(out["ari"] - out["ari_ic"][0]); hi.append(out["ari_ic"][1] - out["ari"])
    ax.errorbar(xs, ys, yerr=[lo, hi], fmt="o", ms=8, capsize=4, color=COLOR[c], label=f"excluida: {ETQ[c]}")
ax.axhline(0, color="0.3", lw=1); ax.set_xticks(range(len(HORIZONTES)), HORIZONTES); ax.set_ylim(-0.1, 1.05)
ax.set_ylabel("ARI transferidas frente a nativas (IC 95 %)"); ax.set_title("¿Se reproducen los fenotipos en una cohorte no vista?")
ax.legend(frameon=False, fontsize=8.5)
nota(fig, "Cada punto: fenotipos aprendidos con las otras dos cohortes y comprobados en la excluida. ARI = 0: azar; 1: idénticos.")
plt.tight_layout(); guardar(fig, "validacion_loco"); plt.show()
""")

md("""
## 7 · Fichas: los fenotipos descritos en cada cohorte

Cada cohorte describe los fenotipos con lo que recoge y no entra en la definición:
- CIBERESUCICOVID: fatiga, resolución, reingresos y TAC.
- Lleida y TENACITY: HADS, mMRC, PM6M, astenia y fibrosis.
- Todas: datos del ingreso.
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
            if v not in VARS_DEF and (v.startswith(IMAGEN) or FP.visitas_que_recogen(cfg, PF, v, cohorte)):
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


fichas = {}
for h, r in resultados.items():
    bloques = {}
    for c in COHORTES:
        dentro = (r["desc"]["cohorte_analisis"] == c).values
        fichas[(h, c)] = ficha(h, c, r["desc"][dentro], r["lab"][dentro], r["k"])
        bloques[ETQ[c]] = fichas[(h, c)]
    print(f"\\n=== {h} ===")
    display(pd.concat(bloques, axis=1).dropna(how="all"))
pd.concat({f"{h}|{c}": f for (h, c), f in fichas.items()}, names=["horizonte|cohorte", "variable"]).to_csv(TAB / f"{PFX}_fichas.csv")
""")

md("""
## 8 · Comparación entre horizontes

### 8.1 Pasaporte conjunto
""")
code("""
pasaporte = pd.DataFrame([{
    "horizonte": h, "N": len(r["desc"]), "k": r["k"], "estructura": "sí" if r["estructura"] else "no",
    "silueta − nula": round(r["silueta"] - r["nulo"].set_index("k").loc[r["k"], "nulo_media"], 3),
    "Jaccard estabilidad (mín)": round(r["estab"]["jaccard_medio"].min(), 2),
    **{f"ARI dejando fuera {ETQ[c]}": f"{loco[(h, c)]['ari']:.2f} [{loco[(h, c)]['ari_ic'][0]:.2f}, {loco[(h, c)]['ari_ic'][1]:.2f}]"
       for c in COHORTES if (h, c) in loco},
} for h, r in resultados.items()]).set_index("horizonte")
pasaporte.to_csv(TAB / f"{PFX}_pasaporte.csv")
pasaporte
""")
md("### 8.2 Transiciones entre horizontes")
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
nota(fig, "Tres cohortes juntas. % por fila: a qué fenotipo pasan los pacientes de cada fenotipo del horizonte previo.")
plt.tight_layout(); guardar(fig, "transiciones"); plt.show()
pd.DataFrame(filas).set_index("transición")
""")
md("""
### 8.3 ¿El fenotipo a 3 meses anticipa el año?

% con cada desenlace según el fenotipo de H3, por cohorte y en las tres juntas (solo con los desenlaces que recoge cada cohorte).
""")
code("""
r = resultados["H3"]
filas, pruebas = [], []
for coh in [None, *COHORTES]:
    dentro = np.ones(len(r["lab"]), bool) if coh is None else (r["desc"]["cohorte_analisis"] == coh).values
    for nombre, v, visitas, regla in PF["desenlaces"]:
        cohs = COHORTES if coh is None else [coh]
        cohs = [c for c in cohs if FP.visitas_que_recogen(cfg, PF, v, c) & set(visitas)]
        if not cohs:
            continue
        y = pd.concat([FP.estado_hasta(medidas, cfg, PF, c, r["desc"].index[(r["desc"]["cohorte_analisis"] == c).values],
                                       v, visitas, regla) for c in cohs]).reindex(r["desc"].index)
        ok = y.notna().values & dentro
        if ok.sum() < N_MIN:
            continue
        etq = "Todas" if coh is None else ETQ[coh]
        ct = pd.crosstab(r["lab"][ok], y.values[ok])
        if ct.shape[1] == 2 and (ct.values > 0).all():
            pruebas.append({"cohorte": etq, "desenlace": nombre, "N": int(ok.sum()), "p (χ²)": f"{stats.chi2_contingency(ct)[1]:.2g}"})
        for c in range(r["k"]):
            s = y.values[ok & (r["lab"] == c)]
            k = int(np.nansum(s))
            if len(s) < N_MIN or 0 < k < N_MIN:
                continue
            lo, hi = ic_wilson(k, len(s))
            filas.append({"cohorte": etq, "desenlace": nombre, "fenotipo H3": nombre_f(c), "N": len(s),
                          "%": round(100 * k / len(s), 1), "IC95": f"{lo:.0f}–{hi:.0f}"})
pronostico = pd.DataFrame(filas)
pronostico.to_csv(TAB / f"{PFX}_pronostico_h3.csv", index=False)
display(pd.DataFrame(pruebas).set_index(["cohorte", "desenlace"]) if pruebas else "sin pruebas evaluables")
pronostico.set_index(["cohorte", "desenlace", "fenotipo H3"]) if len(pronostico) else "sin desenlaces evaluables"
""")

md("""
## 9 · Sensibilidad

Se repite cada horizonte cambiando una sola cosa, con menos permutaciones y remuestreos:
- **k = 3 fijo**;
- **sin FEV1** (solo DLCO y FVC);
- **solo difusión** y **solo espirometría**;
- **sin variables de cambio**;
- **`cmd99`** en CIBERESUCICOVID.
""")
code("""
cmd99 = set(pacientes.loc[pacientes["cmd99"] | (pacientes["cohorte_analisis"] != "CIBERESUCICOVID"), "subject_id"])
rapido = dict(n_perm=20, n_boot=50)
filas = []
for h, base in resultados.items():
    cols = base["cols"]
    escenarios = {"principal": None, f"k = {PF['k_comparacion']} fijo": dict(k_forzado=PF["k_comparacion"]),
                  "sin FEV1": dict(columnas=[c for c in cols if not c.startswith("fev1_")]),
                  "solo difusión": dict(columnas=[c for c in cols if c.startswith("dlco_")]),
                  "solo espirometría": dict(columnas=[c for c in cols if c.startswith(("fvc_", "fev1_"))]),
                  "cmd99": dict(filtro=lambda d: d.index.isin(cmd99))}
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

md("""
## 10 · Preparación para el modelo predictivo (sin ajustarlo)

Se guarda la **tabla base** del modelo futuro:
- **Diana:** fenotipo en H3, H6 y H12, y los desenlaces a 12 meses.
- **Predictores:** las variables del alta **comunes a las tres cohortes** (`fenotipado_comun.predictores_alta`).
- **Cohorte y centro:** para validar dejando fuera cada cohorte o cada centro.

Aquí solo se comprueba su disponibilidad por cohorte; **no se ajusta ningún modelo**.
""")
code("""
filas = []
for h, r in resultados.items():
    D_med = FP.gower(r["X"], r["X"][r["med"]], r["R"], r["w"], r["cat"])
    filas.append(pd.DataFrame({"subject_id": r["desc"].index, "horizonte": h,
                               "cohorte_analisis": r["desc"]["cohorte_analisis"].astype(str).values,
                               "fenotipo": [nombre_f(c) for c in r["lab"]],
                               "distancia_medoide": D_med[np.arange(len(r["lab"])), r["lab"]]}))
fenotipos = pd.concat(filas, ignore_index=True)
fenotipos.to_parquet(DATOS / "fenotipos_comun.parquet", index=False)

base = fenotipos.pivot(index="subject_id", columns="horizonte", values="fenotipo").add_prefix("fenotipo_")
pred = pacientes.set_index("subject_id")[["cohorte_analisis", "centro_id", *PF["predictores_alta"]]]
base = pred.join(base, how="inner")
for nombre, v, visitas, regla in PF["desenlaces"]:
    col = "y_" + nombre.split(" ")[0].lower().replace("-", "_")
    partes = [FP.estado_hasta(medidas, cfg, PF, c, base.index[base["cohorte_analisis"] == c], v, visitas, regla)
              for c in COHORTES if FP.visitas_que_recogen(cfg, PF, v, c) & set(visitas)]
    base[col] = pd.concat(partes).reindex(base.index) if partes else np.nan
base.reset_index().to_parquet(DATOS / "base_modelo.parquet", index=False)
print(f"Guardado {DATOS.name}/fenotipos_comun.parquet ({len(fenotipos):,} filas) y "
      f"{DATOS.name}/base_modelo.parquet ({len(base):,} pacientes × {base.shape[1]} columnas)")

disp = base.groupby("cohorte_analisis", observed=True).apply(lambda d: d.notna().mean() * 100, include_groups=False).round(0).T
disp.columns = [ETQ[str(c)] for c in disp.columns]
disp.loc["N pacientes"] = base["cohorte_analisis"].astype(str).value_counts().rename(ETQ)
print("% de pacientes con dato en cada columna de la tabla base:")
disp
""")
md("""
**Lectura para el modelo:**
- **Dianas.** Los fenotipos existen para todos los pacientes incluidos en cada horizonte. Los desenlaces dependen de lo que recoge cada cohorte: HADS y mMRC no existen en CIBERESUCICOVID, y reingreso no existe en Lleida ni en TENACITY.
- **Predictores.** Los del alta están en las tres cohortes con buena cobertura. La traqueotomía no se incluye porque Lleida no la recoge.
- **Validación prevista:** dejando fuera cada cohorte, y dejando fuera cada centro dentro de CIBERESUCICOVID. Nunca partición aleatoria.
""")

md("## 11 · Resumen")
md("""
**Resultados (ejecución con semilla 2026; ver §2–§10).**

1. **H3 da el fenotipo más sólido de todo el proyecto.** Tiene k = 2, una estructura clara frente al nulo (silueta 0,36 frente a 0,11) y es muy estable (Jaccard 0,92 en los dos grupos):
   - **F1 · "función conservada"** (821 pacientes): DLCO 79 %, FVC 96 %, FEV1 99 %.
   - **F2 · "afectación funcional"** (753): DLCO 65 %, FVC 73 %, FEV1 76 %. Es una alteración mixta, de difusión y de volumen.

   Los nombres son borradores para que los valide el equipo.
2. **No separa cohortes: separa pacientes.** La relación entre fenotipo y cohorte es pequeña (V de Cramér 0,10–0,12). Dentro de cada cohorte, el perfil del fenotipo es casi idéntico: F2 tiene DLCO 64 % y FVC 73 % tanto en CIBERESUCICOVID como en Lleida.
3. **Viaja, al menos a 3 meses.** Al aprender con dos cohortes y comprobar en la tercera:
   - ARI de 0,50 dejando fuera CIBERESUCICOVID, 0,61 dejando fuera Lleida y 0,73 dejando fuera TENACITY;
   - Jaccard por fenotipo de 0,68 a 0,87.

   Es el primer fenotipo que se reproduce razonablemente en cohortes no vistas, aunque los IC son anchos en las cohortes pequeñas.
4. **H6 y H12 son más débiles.**
   - **H6:** k = 4, y un grupo solo es "patrón". Al dejar fuera una cohorte, el ARI baja a ≈ 0,3.
   - **H12:** k = 2 y estable (Jaccard 0,89), pero viaja peor: ARI de 0,34, 0,21 y 0,05 según la cohorte excluida, con IC que incluyen valores bajos.
   - **Transiciones:** los fenotipos cambian entre horizontes (ARI H3 → H6 = 0,10).
5. **Quién sostiene la partición.** La **espirometría**: la versión "solo espirometría" coincide casi del todo con la principal en H3 (ARI 0,97). La difusión sola no tiene estructura en H3. Quitar FEV1 cambia poco (ARI 0,70).
6. **Valor pronóstico.** El fenotipo a 3 meses anticipa la **DLCO < 80 % al año**:
   - 47 % frente a 66 % en las tres cohortes juntas (p < 0,0001);
   - 58 % frente a 74 % en CIBERESUCICOVID (p = 0,02);
   - 38 % frente a 68 % en Lleida (p < 0,0001).

   **No** anticipa ansiedad, disnea ni reingresos: es un fenotipo respiratorio.

**Para el modelo predictivo**
- **Diana recomendada:** el **fenotipo H3** (binario, estable y transferible) o la **DLCO < 80 % al año**. H12 es una diana más débil.
- **Tabla base** en `Datos limpios/base_modelo.parquet`: 2.439 pacientes, con fenotipos, desenlaces y 10 predictores del alta comunes a las tres cohortes (cobertura ≥ 86 %).
- **Validación:** dejando fuera cada cohorte, y cada centro dentro de CIBERESUCICOVID.

**Limitaciones:**
- **Definición solo respiratoria** (DLCO, FVC, FEV1): es lo único que las tres cohortes miden igual. Síntomas, ánimo y esfuerzo describen, no definen.
- **Composición desigual.** CIBERESUCICOVID aporta unos dos tercios de los pacientes y pesa más en el descubrimiento; por eso la validación deja fuera cada cohorte.
- **TENACITY es pequeña en H6 y H12** (unos 60): los IC serán anchos.
- **Abandono informativo** (notebook 01, §8): H12 sobrerrepresenta a quien sigue en seguimiento.
- **No se imputa:** quien tiene pocas medidas pesa con menos información.
""")

nb = nbf.v4.new_notebook()
nb["cells"] = celdas
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                  "language_info": {"name": "python"}}
salida = Path(__file__).resolve().parents[1] / "05_fenotipado_comun.ipynb"
nbf.write(nb, salida)
print(f"Escrito {salida} ({len(celdas)} celdas)")
