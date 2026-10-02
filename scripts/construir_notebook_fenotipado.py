"""Genera `03_fenotipado_horizontes.ipynb`.

Uso: python scripts/construir_notebook_fenotipado.py
(después: python -m nbconvert --to notebook --execute --inplace 03_fenotipado_horizontes.ipynb)
"""
from pathlib import Path

import nbformat as nbf

celdas: list = []
md = lambda texto: celdas.append(nbf.v4.new_markdown_cell(texto.strip()))
code = lambda texto: celdas.append(nbf.v4.new_code_cell(texto.strip()))

md("""
# Reto 2 · Fenotipos a 3, 6 y 12 meses

**Notebook 03.** Seis clusterings de fenotipos post-infecciosos: **3 horizontes × 2 capas**.

| | Capa A · respiratoria | Capa B · multidominio |
|---|---|---|
| Variables | DLCO, FVC | DLCO, FVC, HADS-A, HADS-D, mMRC, PM6M |
| Descubre | CIBERESUCICOVID | POSTCOVID-Lleida |
| Replica | Lleida y TENACITY | TENACITY |

- **Horizonte acumulado.** H3 usa la ventana T3 (60–135 días). H6 usa T3 y T6 (136–270). H12 usa T3, T6 y T12 (271–480). Cada horizonte suma, además, el **cambio** de cada variable entre la primera y la última medida. Nunca se usan datos posteriores al horizonte.
- **Ancla de inclusión.** Para entrar en un horizonte, el paciente necesita medidas en su última ventana. Así el fenotipo describe el estado *en* ese momento.
- **Método.**
  - Distancia de Gower con rangos clínicos fijos e igual peso por dominio.
  - PAM.
  - k elegido contra una prueba nula de permutación.
  - Estabilidad bootstrap.
  - Replicación por transferencia al medoide frente a clustering nativo, con emparejamiento húngaro, ARI y Jaccard con IC bootstrap.

> **Privacidad.** Solo se muestran agregados. Los clústeres y celdas con N < 10 se suprimen. Las etiquetas por paciente se guardan en `Datos limpios/fenotipos.parquet`, fuera del repositorio.
""")

md("## 0 · Entorno y contrato de replicación")
code("""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

RAIZ = Path.cwd()
sys.path.insert(0, str(RAIZ))
from src import carga, fenotipado as F, pasaporte as P, privacidad

cfg = carga.cargar_config()
FC = cfg["fenotipado"]
SEMILLA = cfg["semilla"]
N_MIN = cfg["privacidad"]["n_minimo_celda"]
COLOR = cfg["colores_registro"]
ETQ = {"CIBERESUCICOVID": "CIBERESUCICOVID", "POSTCOVID_LLEIDA": "POSTCOVID-Lleida",
       "TENACITY": "TENACITY", "VIRGEN_DEL_ROCIO": "Virgen del Rocío"}
CAPAS, HORIZONTES = list(FC["capas"]), list(FC["horizontes"])
KS = list(range(FC["k_rango"][0], FC["k_rango"][1] + 1))
DATOS = carga.RAIZ / cfg["rutas"]["procesados"]
FIG = RAIZ / cfg["rutas"]["figuras"]; FIG.mkdir(parents=True, exist_ok=True)
TAB = RAIZ / cfg["rutas"]["tablas"]; TAB.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({"figure.dpi": 100, "axes.titleweight": "bold", "axes.titlesize": 11.5,
                     "axes.spines.top": False, "axes.spines.right": False})
pd.set_option("display.max_columns", 60); pd.set_option("display.width", 220)


def nota(fig, texto: str) -> None:
    \"\"\"Pie de figura con N y cohortes.\"\"\"
    fig.text(0.01, -0.01, texto, fontsize=8.5, color="0.35", va="top", ha="left")


def guardar(fig, nombre: str) -> None:
    fig.savefig(FIG / f"{nombre}.png", bbox_inches="tight", dpi=150)


def ic_wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    c = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    m = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return (100 * (c - m), 100 * (c + m))


def nombre_f(c: int) -> str:
    return f"F{c + 1}"
""")
code("""
contrato = pd.DataFrame([
    ("Ventanas (días desde el alta)", ", ".join(f"{k} = {v[0]}–{v[1]}" for k, v in FC["ventanas"].items())),
    ("Horizontes (acumulados)", ", ".join(f"{h}: {'+'.join(v)}" for h, v in FC["horizontes"].items())),
    ("Capa A", f"{FC['capas']['A']['dominios']} · descubre {FC['capas']['A']['descubre']} · replica {FC['capas']['A']['replica']}"),
    ("Capa B", f"{FC['capas']['B']['dominios']} · descubre {FC['capas']['B']['descubre']} · replica {FC['capas']['B']['replica']}"),
    ("Elección de k", f"k en {KS}: mayor (silueta real − media nula) entre los que superan el p95 nulo ({FC['n_permutaciones']} permutaciones)"),
    ("Estable", f"Jaccard bootstrap por clúster ({FC['n_bootstrap']} remuestreos): ≥ 0,75 estable, 0,5–0,75 patrón, < 0,5 se disuelve"),
    ("Viaja", "ARI transferidas frente a nativas con IC 95 % que excluye 0; Jaccard por par emparejado ≥ 0,5"),
    ("Perfil replicado", "|diferencia estandarizada| < 0,5 en las variables de definición"),
    ("k fijo de comparación", FC["k_comparacion"]),
], columns=["criterio", "valor"]).set_index("criterio")
contrato
""")
md("""
Estos criterios se fijan **antes** de ver los resultados y salen de `config.yaml`. Un fenotipo que no los cumple se presenta igualmente, como no replicado. No se ajusta nada para que "salga".
""")

md("## 1 · Pacientes por horizonte y capa")
code("""
pacientes = pd.read_parquet(DATOS / "pacientes.parquet")
medidas = pd.read_parquet(DATOS / "medidas.parquet")

tablas_var = {}
for capa in CAPAS:
    for h in HORIZONTES:
        tablas_var[(capa, h)] = F.construir_variables(medidas, pacientes, cfg, h, capa)

filas = []
for (capa, h), (t, cols) in tablas_var.items():
    usadas = [FC["capas"][capa]["descubre"], *FC["capas"][capa]["replica"]]
    for r in usadas:
        filas.append({"capa": capa, "horizonte": h, "cohorte": ETQ[r], "variables": len(cols),
                      "N": int((t["cohorte_analisis"] == r).sum()),
                      "papel": "descubre" if r == FC["capas"][capa]["descubre"] else "replica"})
n_tabla = pd.DataFrame(filas).pivot_table(index=["capa", "cohorte", "papel"], columns="horizonte", values="N").reindex(columns=HORIZONTES)
for (capa, h), (t, cols) in tablas_var.items():
    assert t["dias_max"].max() <= FC["ventanas"][FC["horizontes"][h][-1]][1], "fuga de datos futuros"
print("Comprobado: ninguna variable usa datos posteriores a su horizonte.")
print("Variables por clustering:", {f"{c}-{h}": len(v[1]) for (c, h), v in tablas_var.items()})
privacidad.suprimir_celdas(n_tabla.astype(int), N_MIN, HORIZONTES)
""")
md("""
- **H3 de la capa A tiene solo 2 variables** (DLCO y FVC a los 3 meses). Sus fenotipos serán cortes del plano difusión × volumen; la trayectoria aparece en H6 y H12.
- **Pocos pacientes de TENACITY en H6 y H12** (unos 65): ahí la replicación tendrá IC anchos.
- **Virgen del Rocío** no participa: mide al mes y casi no tiene seguimiento.
""")

md("## 2 · Ejecución de los 6 clusterings")
code("""
def ejecutar(capa: str, h: str, k_forzado: int | None = None, columnas: list[str] | None = None,
             filtro=None, n_perm: int | None = None, n_boot: int | None = None) -> dict:
    \"\"\"Clustering completo de (capa, horizonte): k contra el nulo, PAM, estabilidad y replicación.\"\"\"
    t, cols = tablas_var[(capa, h)]
    cols = columnas or cols
    if filtro is not None:
        t = t[filtro(t)]
    fc = FC["capas"][capa]
    desc = t[t["cohorte_analisis"] == fc["descubre"]]
    X = desc[cols].to_numpy(float)
    _, R, w = F.parametros_gower(cols, cfg, capa)
    D = F.gower(X, None, R, w)
    nulo = F.seleccion_k_nulo(X, R, w, KS, n_perm or FC["n_permutaciones"], SEMILLA, D)
    k, estructura = (k_forzado, bool(nulo.set_index("k").loc[k_forzado, "supera_p95"])) if k_forzado else F.elegir_k(nulo)
    lab, med = F.pam(D, k, SEMILLA)
    ultima = FC["horizontes"][h][-1]
    lab, med = F.ordenar_por_severidad(lab, med, desc[f"dlco_{ultima}"].to_numpy(float))
    res = {"capa": capa, "h": h, "cols": cols, "R": R, "w": w, "desc": desc, "X": X, "D": D, "nulo": nulo,
           "k": k, "estructura": estructura, "lab": lab, "med": med, "silueta": F.silueta(D, lab),
           "estab": F.estabilidad_bootstrap(D, lab, k, n_boot or FC["n_bootstrap"], SEMILLA), "replicas": {}}
    if capa == "A":
        res["centros"] = F.estabilidad_centros(D, lab, desc["centro_id"], k, FC["min_pacientes_centro"], SEMILLA)
    for r in fc["replica"]:
        rep = t[t["cohorte_analisis"] == r]
        if len(rep) < 2 * k:
            continue
        out = P.replicar(X, med, rep[cols].to_numpy(float), R, w, k, n_boot or FC["n_bootstrap"], SEMILLA)
        out["datos"] = rep
        out["dif"] = P.diferencias_perfil(desc, lab, rep, out["transferidas"], cols, k)
        res["replicas"][r] = out
    return res


resultados = {}
for capa in CAPAS:
    for h in HORIZONTES:
        resultados[(capa, h)] = ejecutar(capa, h)
        r = resultados[(capa, h)]
        print(f"Capa {capa} · {h}: N descubrimiento = {len(r['desc'])}, k = {r['k']}, "
              f"estructura frente al nulo = {'sí' if r['estructura'] else 'NO'}, silueta = {r['silueta']:.2f}")
""")

md("""
## 3 · Elección de k: silueta real frente a la nula

La línea gris es la silueta media de 100 conjuntos de datos con las columnas permutadas: tienen las mismas distribuciones, pero sin estructura conjunta. Solo hay fenotipos si la silueta real supera esa banda.
""")
code("""
fig, axes = plt.subplots(2, 3, figsize=(15, 7.5), sharey=True)
for i, capa in enumerate(CAPAS):
    for j, h in enumerate(HORIZONTES):
        r = resultados[(capa, h)]; nl = r["nulo"]; ax = axes[i, j]
        ax.fill_between(nl["k"], nl["nulo_media"], nl["nulo_p95"], color="0.8", alpha=0.6, label="nulo (media–p95)")
        ax.plot(nl["k"], nl["nulo_media"], color="0.5", lw=1)
        ax.plot(nl["k"], nl["silueta_real"], "o-", color=COLOR[FC["capas"][capa]["descubre"]], lw=2, label="real")
        ax.axvline(r["k"], color="0.2", ls="--", lw=1)
        ax.set_title(f"Capa {capa} · {h}  (N={len(r['desc'])}, k={r['k']})")
        ax.set_xticks(KS)
        if i == 1: ax.set_xlabel("k")
        if j == 0: ax.set_ylabel(f"Silueta · capa {capa}")
axes[0, 0].legend(frameon=False, fontsize=8.5)
nota(fig, "Descubrimiento: capa A en CIBERESUCICOVID, capa B en POSTCOVID-Lleida. Línea discontinua = k elegido.")
plt.tight_layout(); guardar(fig, "03_eleccion_k"); plt.show()
""")
code("""
resumen_k = pd.DataFrame([{
    "capa": c, "horizonte": h, "N": len(r["desc"]), "variables": len(r["cols"]), "k": r["k"],
    "silueta": round(r["silueta"], 3),
    "silueta nula (media)": round(r["nulo"].set_index("k").loc[r["k"], "nulo_media"], 3),
    "supera p95 nulo": r["estructura"],
} for (c, h), r in resultados.items()]).set_index(["capa", "horizonte"])
resumen_k
""")

md("## 4 · Perfiles de los fenotipos")
code("""
def heatmap_perfil(ax, r):
    \"\"\"% alterado por variable y ventana (filas) y fenotipo (columnas); celdas con < N_MIN casos suprimidas.\"\"\"
    pct, n_alt = F.porcentaje_alterado(r["desc"], r["cols"], r["lab"], cfg)
    tamanos = pd.Series(r["lab"]).value_counts().sort_index()
    oculta = (n_alt > 0) & (n_alt < N_MIN)
    oculta.loc[:, tamanos[tamanos < N_MIN].index] = True
    etiquetas = pct.round(0).astype("Int64").astype(str).mask(oculta, "<10")
    visible = pct.mask(oculta)                               # enmascarar ANTES de renombrar columnas
    visible.columns = [f"{nombre_f(c)}\\nN={tamanos[c] if tamanos[c] >= N_MIN else '<10'}" for c in pct.columns]
    sns.heatmap(visible, annot=etiquetas.values, fmt="", cmap="Reds", vmin=0, vmax=100, cbar=False,
                ax=ax, linewidths=0.5)
    ax.set_title(f"Capa {r['capa']} · {r['h']}", fontsize=11)
    ax.set_ylabel(""); ax.tick_params(axis="y", labelsize=8.5); ax.tick_params(axis="x", labelsize=8.5, rotation=0)


for capa in CAPAS:
    alto = 3.0 if capa == "A" else 7.5
    fig, axes = plt.subplots(1, 3, figsize=(16, alto), gridspec_kw={"width_ratios": [1, 1.2, 1.4]})
    for ax, h in zip(axes, HORIZONTES):
        heatmap_perfil(ax, resultados[(capa, h)])
    cohorte = ETQ[FC["capas"][capa]["descubre"]]
    nota(fig, f"Capa {capa}, descubrimiento en {cohorte}. Celdas: % de pacientes con la variable alterada "
              "(DLCO/FVC < 80 %, HADS ≥ 8, mMRC ≥ 2, PM6M < 400 m) en cada ventana. F1 = DLCO más alta en la última ventana.")
    plt.tight_layout(); guardar(fig, f"03_perfiles_capa_{capa}"); plt.show()
""")
code("""
# Mediana del cambio (último − primero) por fenotipo: la parte "trayectoria" del fenotipo
filas = []
for (capa, h), r in resultados.items():
    cols_c = [c for c in r["cols"] if c.endswith("_cambio")]
    if not cols_c:
        continue
    med_c = r["desc"][cols_c].groupby(r["lab"]).median().round(1)
    for c, fila in med_c.iterrows():
        if (r["lab"] == c).sum() >= N_MIN:
            filas.append({"capa": capa, "horizonte": h, "fenotipo": nombre_f(c), **fila.to_dict()})
cambios = pd.DataFrame(filas).set_index(["capa", "horizonte", "fenotipo"])
cambios.columns = [c.replace("_cambio", " Δ") for c in cambios.columns]
cambios
""")

md("""
### Fichas: descripción con variables que no definen el fenotipo

Las fichas describen cada fenotipo con datos del ingreso (edad, sexo, gravedad, comorbilidad) y con variables de seguimiento que no entran en su definición, medidas en la ventana ancla. En la capa A se describe en Lleida y TENACITY (etiquetas transferidas), porque solo esas cohortes recogen HADS, mMRC y PM6M.
""")
code("""
def variable_en_ventana(sujetos: pd.Index, variable: str, ventana: str) -> pd.Series:
    \"\"\"Primera medida válida de `variable` en la ventana, para los sujetos indicados.\"\"\"
    lo, hi = FC["ventanas"][ventana]
    m = medidas[~medidas["excluida"] & (medidas["variable"] == variable) & medidas["dias_alta"].between(lo, hi)]
    return m.sort_values("dias_alta").groupby("subject_id")["valor"].first().reindex(sujetos)


def ficha(datos: pd.DataFrame, etiquetas: np.ndarray, ventana: str, k: int) -> pd.DataFrame:
    \"\"\"Descripción por fenotipo (solo agregados; columnas con N < N_MIN suprimidas).\"\"\"
    p = pacientes.set_index("subject_id").reindex(datos.index)
    filas = {}
    for c in range(k):
        s = etiquetas == c
        sub = p[s]
        n = int(s.sum())
        col = {"N": n}
        if n >= N_MIN:
            col["Edad ≥ 65 (%)"] = 100 * sub["grupo_edad"].isin(["65-74", ">=75"]).sum() / max(sub["grupo_edad"].notna().sum(), 1)
            col["Mujeres (%)"] = 100 * (sub["sexo"] == 1).sum() / max(sub["sexo"].notna().sum(), 1)
            col["Estancia (mediana, días)"] = sub["estancia_hosp_dias"].median()
            for v in FC["variables_descripcion"]["ingreso"]:
                ss = sub[v].dropna()
                col[f"{v.replace('nu_', '')} (%)"] = 100 * ss.mean() if len(ss) >= N_MIN and (ss.sum() == 0 or ss.sum() >= N_MIN) else np.nan
            for v in FC["variables_descripcion"]["seguimiento"]:
                val = variable_en_ventana(datos.index[s], v, ventana).dropna()
                clave, direc = cfg["estados"][v]; u = cfg["umbrales_clinicos"][clave]
                alt = (val < u) if direc == "<" else (val >= u)
                col[f"{v} alterado (%)"] = 100 * alt.mean() if len(val) >= N_MIN and (alt.sum() == 0 or alt.sum() >= N_MIN) else np.nan
        filas[nombre_f(c)] = col
    return pd.DataFrame(filas).round(1)


fichas = {}
for (capa, h), r in resultados.items():
    ventana = FC["horizontes"][h][-1]
    fichas[(capa, h, FC["capas"][capa]["descubre"])] = ficha(r["desc"], r["lab"], ventana, r["k"])
    for rep, out in r["replicas"].items():
        fichas[(capa, h, rep)] = ficha(out["datos"], out["transferidas"], ventana, r["k"])
pd.concat(fichas, names=["capa", "horizonte", "cohorte", "variable"]).to_csv(TAB / "03_fichas_fenotipos.csv")
print("Fichas guardadas en outputs/tablas/03_fichas_fenotipos.csv (vacío = no recogido o celda < 10)")
""")
code("""
for capa in CAPAS:
    for h in HORIZONTES:
        cohortes = [FC["capas"][capa]["descubre"], *FC["capas"][capa]["replica"]]
        bloque = pd.concat({ETQ[c]: fichas[(capa, h, c)] for c in cohortes if (capa, h, c) in fichas}, axis=1)
        bloque = bloque.dropna(how="all")
        print(f"\\n=== Capa {capa} · {h} ===")
        display(bloque)
""")

md("## 5 · Estabilidad (¿el fenotipo es estable?)")
code("""
filas = []
for (capa, h), r in resultados.items():
    for _, e in r["estab"].iterrows():
        filas.append({"capa": capa, "horizonte": h, "fenotipo": nombre_f(int(e["cluster"])),
                      "N": int(e["n"]), "Jaccard medio": round(e["jaccard_medio"], 2), "Jaccard p05": round(e["jaccard_p05"], 2)})
estab = pd.DataFrame(filas)
estab["veredicto"] = pd.cut(estab["Jaccard medio"], [0, 0.5, 0.75, 1.01], labels=["se disuelve", "patrón", "estable"])
estab.to_csv(TAB / "03_estabilidad.csv", index=False)
privacidad.suprimir_celdas(estab.set_index(["capa", "horizonte", "fenotipo"]), N_MIN, ["N"])
""")
code("""
filas = []
for h in HORIZONTES:
    c = resultados[("A", h)].get("centros")
    if c is not None and len(c):
        filas.append({"horizonte": h, "centros evaluados": len(c), "ARI mediano": round(c["ari"].median(), 2),
                      "ARI mínimo": round(c["ari"].min(), 2), "ARI máximo": round(c["ari"].max(), 2)})
print(f"Capa A: dejar fuera cada centro de CIBERESUCICOVID con ≥ {FC['min_pacientes_centro']} pacientes y reasignarlo")
pd.DataFrame(filas).set_index("horizonte")
""")

md("## 6 · Replicación entre cohortes (¿el fenotipo viaja?)")
code("""
filas = []
for (capa, h), r in resultados.items():
    for rep, out in r["replicas"].items():
        fila = {"capa": capa, "horizonte": h, "replica en": ETQ[rep], "N": out["n"], "k": r["k"],
                "ARI": round(out["ari"], 2), "ARI IC95": f"{out['ari_ic'][0]:.2f} a {out['ari_ic'][1]:.2f}"}
        for c in range(r["k"]):
            fila[f"Jaccard {nombre_f(c)}"] = f"{out['jaccard'][c]:.2f} ({out['jaccard_ic'][c][0]:.2f}–{out['jaccard_ic'][c][1]:.2f})"
        dif = out["dif"].abs()
        fila["% perfiles |d|<0,5"] = round(100 * (dif < 0.5).sum().sum() / dif.notna().sum().sum(), 0)
        filas.append(fila)
replicacion = pd.DataFrame(filas).set_index(["capa", "horizonte", "replica en"])
replicacion.to_csv(TAB / "03_replicacion.csv")
replicacion
""")
code("""
fig, axes = plt.subplots(1, 2, figsize=(14, 4.3), sharey=True)
for ax, capa in zip(axes, CAPAS):
    reps = FC["capas"][capa]["replica"]
    for i, rep in enumerate(reps):
        xs, ys, lo, hi = [], [], [], []
        for j, h in enumerate(HORIZONTES):
            out = resultados[(capa, h)]["replicas"].get(rep)
            if out is None:
                continue
            xs.append(j + (i - (len(reps) - 1) / 2) * 0.12); ys.append(out["ari"])
            lo.append(out["ari"] - out["ari_ic"][0]); hi.append(out["ari_ic"][1] - out["ari"])
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt="o", ms=8, capsize=4, color=COLOR[rep], label=f"replica en {ETQ[rep]}")
    ax.axhline(0, color="0.3", lw=1)
    ax.set_xticks(range(len(HORIZONTES)), HORIZONTES)
    ax.set_title(f"Capa {capa} · descubre {ETQ[FC['capas'][capa]['descubre']]}")
    ax.legend(frameon=False, fontsize=9)
axes[0].set_ylabel("ARI transferidas frente a nativas (IC 95 %)")
nota(fig, "ARI = 0: concordancia al azar; ARI = 1: la cohorte de réplica reproduce exactamente los mismos grupos.")
plt.tight_layout(); guardar(fig, "03_replicacion_ari"); plt.show()
""")
code("""
# Perfil replicado: diferencia estandarizada réplica − descubrimiento por fenotipo (etiquetas transferidas)
pares = [(c, h, rep) for (c, h), r in resultados.items() for rep in r["replicas"]]
fig, axes = plt.subplots(1, len(pares), figsize=(2.6 * len(pares), 7), sharey=False)
for ax, (capa, h, rep) in zip(axes, pares):
    dif = resultados[(capa, h)]["replicas"][rep]["dif"].copy()
    dif.columns = [nombre_f(c) for c in dif.columns]
    dif = dif.loc[[c for c in dif.index if not c.endswith("_cambio")]]
    sns.heatmap(dif, cmap="RdBu_r", vmin=-1.5, vmax=1.5, center=0, annot=True, fmt=".1f", cbar=False, ax=ax,
                annot_kws={"size": 7}, linewidths=0.3)
    ax.set_title(f"{capa}·{h}\\n→ {ETQ[rep][:9]}", fontsize=9.5)
    ax.tick_params(labelsize=7.5)
nota(fig, "Diferencia estandarizada de medias (réplica − descubrimiento). |d| < 0,5 = perfil replicado. "
          "Rojo: la réplica tiene valores más altos.")
plt.tight_layout(); guardar(fig, "03_diferencias_perfil"); plt.show()
""")

md("""
## 7 · Comparación entre horizontes

### 7.1 Pasaporte conjunto

Responde a la pregunta del equipo: **¿en qué horizonte son más sólidos los fenotipos?**
""")
code("""
filas = []
for (capa, h), r in resultados.items():
    fila = {"capa": capa, "horizonte": h, "N descubre": len(r["desc"]), "k": r["k"],
            "estructura (vs nulo)": "sí" if r["estructura"] else "no",
            "silueta − nula": round(r["silueta"] - r["nulo"].set_index("k").loc[r["k"], "nulo_media"], 3),
            "Jaccard estabilidad (mín)": round(r["estab"]["jaccard_medio"].min(), 2),
            "Jaccard estabilidad (media)": round(r["estab"]["jaccard_medio"].mean(), 2)}
    for rep, out in r["replicas"].items():
        fila[f"ARI {ETQ[rep]}"] = f"{out['ari']:.2f} [{out['ari_ic'][0]:.2f}, {out['ari_ic'][1]:.2f}]"
    filas.append(fila)
pasaporte = pd.DataFrame(filas).set_index(["capa", "horizonte"])
pasaporte.to_csv(TAB / "03_pasaporte_horizontes.csv")
pasaporte
""")

md("### 7.2 Transiciones entre horizontes")
code("""
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
filas = []
for i, capa in enumerate(CAPAS):
    for j, (h1, h2) in enumerate([("H3", "H6"), ("H6", "H12")]):
        r1, r2 = resultados[(capa, h1)], resultados[(capa, h2)]
        e1 = pd.Series([nombre_f(c) for c in r1["lab"]], index=r1["desc"].index)
        e2 = pd.Series([nombre_f(c) for c in r2["lab"]], index=r2["desc"].index)
        tabla, ari, n = P.transiciones(e1, e2)
        filas.append({"capa": capa, "transición": f"{h1} → {h2}", "pacientes comunes": n, "ARI": round(ari, 2)})
        ax = axes[i, j]
        pct = tabla.div(tabla.sum(axis=1), axis=0) * 100
        anot = pct.round(0).astype(int).astype(str).mask(tabla < N_MIN, "<10")
        sns.heatmap(pct.mask(tabla < N_MIN), annot=anot.values, fmt="", cmap="Blues", vmin=0, vmax=100, cbar=False,
                    ax=ax, linewidths=0.5)
        ax.set_xlabel(f"Fenotipo en {h2}"); ax.set_ylabel(f"Fenotipo en {h1}")
        ax.set_title(f"Capa {capa}: {h1} → {h2}  (N={n}, ARI={ari:.2f})")
nota(fig, "% por fila: de los pacientes en cada fenotipo del horizonte previo, a qué fenotipo pasan. "
          "Cohorte de descubrimiento; solo pacientes presentes en ambos horizontes. F1 = DLCO más alta.")
plt.tight_layout(); guardar(fig, "03_transiciones"); plt.show()
pd.DataFrame(filas).set_index(["capa", "transición"])
""")

md("""
### 7.3 Valor pronóstico temprano: ¿el fenotipo a 3 meses anticipa el estado al año?

Para los pacientes fenotipados en H3 que tienen medida en la ventana T12, se calcula el % con alteración a los 12 meses según su fenotipo de H3.
""")
code("""
def estado_12m(sujetos: pd.Index, variables: list[str]) -> pd.Series:
    \"\"\"True si alguna de `variables` está alterada en T12; NaN si ninguna tiene medida.\"\"\"
    alt = []
    for v in variables:
        val = variable_en_ventana(sujetos, v, "T12")
        clave, direc = cfg["estados"][v]; u = cfg["umbrales_clinicos"][clave]
        alt.append(((val < u) if direc == "<" else (val >= u)).astype(float).where(val.notna()))
    return pd.concat(alt, axis=1).max(axis=1)        # 1 = alguna alterada, 0 = ninguna, NaN = sin medida


desenlaces = {"DLCO < 80 % al año": ["dlco"], "HADS ≥ 8 al año": ["hads_a", "hads_d"], "mMRC ≥ 2 al año": ["mmrc"]}
filas = []
for capa in CAPAS:
    r = resultados[(capa, "H3")]
    grupos = {FC["capas"][capa]["descubre"]: (r["desc"], r["lab"])}
    grupos.update({rep: (o["datos"], o["transferidas"]) for rep, o in r["replicas"].items()})
    for cohorte, (datos, lab) in grupos.items():
        for nombre, vs in desenlaces.items():
            if capa == "A" and nombre != "DLCO < 80 % al año" and cohorte == "CIBERESUCICOVID":
                continue
            y = estado_12m(datos.index, vs)
            for c in range(r["k"]):
                s = y[lab == c].dropna()
                kk = int(s.sum())
                if len(s) < N_MIN or 0 < kk < N_MIN:
                    continue
                lo, hi = ic_wilson(kk, len(s))
                filas.append({"capa": capa, "cohorte": ETQ[cohorte], "desenlace": nombre, "fenotipo H3": nombre_f(c),
                              "N con dato": len(s), "%": round(100 * kk / len(s), 1), "IC95": f"{lo:.0f}–{hi:.0f}"})
pronostico = pd.DataFrame(filas)
pronostico.to_csv(TAB / "03_pronostico_h3.csv", index=False)
pronostico.set_index(["capa", "cohorte", "desenlace", "fenotipo H3"])
""")
code("""
sub = pronostico[pronostico["desenlace"] == "DLCO < 80 % al año"]
fig, axes = plt.subplots(1, 2, figsize=(14, 4.2), sharey=True)
for ax, capa in zip(axes, CAPAS):
    d = sub[sub["capa"] == capa]
    cohortes = list(dict.fromkeys(d["cohorte"]))
    for i, coh in enumerate(cohortes):
        dd = d[d["cohorte"] == coh]
        x = np.array([int(f[1:]) - 1 for f in dd["fenotipo H3"]]) + (i - (len(cohortes) - 1) / 2) * 0.15
        lo = dd["%"] - dd["IC95"].str.split("–").str[0].astype(float)
        hi = dd["IC95"].str.split("–").str[1].astype(float) - dd["%"]
        clave = [k for k, v in ETQ.items() if v == coh][0]
        ax.errorbar(x, dd["%"], yerr=[lo, hi], fmt="o", capsize=3, color=COLOR[clave], label=coh, ms=7)
    k = resultados[(capa, "H3")]["k"]
    ax.set_xticks(range(k), [nombre_f(c) for c in range(k)]); ax.set_xlabel("Fenotipo a los 3 meses (H3)")
    ax.set_title(f"Capa {capa}"); ax.legend(frameon=False, fontsize=9); ax.set_ylim(0, 105)
axes[0].set_ylabel("% con DLCO < 80 % al año (IC 95 %)")
nota(fig, "Pacientes fenotipados a los 3 meses con DLCO medida entre 271 y 480 días. Puntos con N < 10 o < 10 casos suprimidos.")
plt.tight_layout(); guardar(fig, "03_pronostico_h3"); plt.show()
""")

md("""
## 8 · Análisis de sensibilidad

¿Las conclusiones dependen de decisiones discutibles? Se repite cada clustering cambiando una sola cosa:
- **k = 3 fijo**, para comparar horizontes con el mismo número de grupos;
- **sin variables de cambio** (solo niveles);
- **solo pacientes con trayectoria completa** (medidas en todas las ventanas);
- **capa A con `cmd99`** en lugar de `cmd50`.

Se usan menos permutaciones y remuestreos para que el notebook no tarde demasiado.
""")
code("""
cmd99 = set(pacientes.loc[pacientes["cmd99"] | (pacientes["cohorte_analisis"] != "CIBERESUCICOVID"), "subject_id"])
rapido = dict(n_perm=20, n_boot=50)
filas = []
for (capa, h), base in resultados.items():
    t, cols = tablas_var[(capa, h)]
    escenarios = {"principal": None,
                  f"k = {FC['k_comparacion']} fijo": dict(k_forzado=FC["k_comparacion"])}
    if h != "H3":
        escenarios["sin variables de cambio"] = dict(columnas=[c for c in cols if not c.endswith("_cambio")])
        ventanas = FC["horizontes"][h]
        resp = [f"dlco_{v}" for v in ventanas]
        escenarios["trayectoria completa"] = dict(filtro=lambda d, resp=resp: d[resp].notna().all(axis=1))
    if capa == "A":
        escenarios["cmd99"] = dict(filtro=lambda d: d.index.isin(cmd99))
    for nombre, kw in escenarios.items():
        r = base if kw is None else ejecutar(capa, h, **kw, **rapido)
        fila = {"capa": capa, "horizonte": h, "escenario": nombre, "N": len(r["desc"]), "k": r["k"],
                "estructura": "sí" if r["estructura"] else "no",
                "Jaccard estab. (media)": round(r["estab"]["jaccard_medio"].mean(), 2)}
        for rep, out in r["replicas"].items():
            fila[f"ARI {ETQ[rep]}"] = round(out["ari"], 2)
        filas.append(fila)
sensibilidad = pd.DataFrame(filas).set_index(["capa", "horizonte", "escenario"])
sensibilidad.to_csv(TAB / "03_sensibilidad.csv")
privacidad.suprimir_celdas(sensibilidad, N_MIN, ["N"])
""")

md("## 9 · Guardar las etiquetas")
code("""
filas = []
for (capa, h), r in resultados.items():
    D_med = F.gower(r["X"], r["X"][r["med"]], r["R"], r["w"])
    filas.append(pd.DataFrame({"subject_id": r["desc"].index, "capa": capa, "horizonte": h,
                               "cohorte_analisis": r["desc"]["cohorte_analisis"].astype(str).values,
                               "fenotipo": [nombre_f(c) for c in r["lab"]], "tipo": "descubrimiento",
                               "distancia_medoide": D_med[np.arange(len(r["lab"])), r["lab"]]}))
    for rep, out in r["replicas"].items():
        Xr = out["datos"][r["cols"]].to_numpy(float)
        D_r = F.gower(Xr, r["X"][r["med"]], r["R"], r["w"])
        for tipo, lab in [("transferida", out["transferidas"]), ("nativa", out["nativas"])]:
            filas.append(pd.DataFrame({"subject_id": out["datos"].index, "capa": capa, "horizonte": h,
                                       "cohorte_analisis": rep, "fenotipo": [nombre_f(c) for c in lab], "tipo": tipo,
                                       "distancia_medoide": D_r[np.arange(len(lab)), lab]}))
fenotipos = pd.concat(filas, ignore_index=True)
fenotipos.to_parquet(DATOS / "fenotipos.parquet", index=False)
print(f"Guardado {DATOS.name}/fenotipos.parquet: {len(fenotipos):,} filas")
privacidad.suprimir_celdas(fenotipos.groupby(["capa", "horizonte", "tipo"]).size().unstack().fillna(0).astype(int), N_MIN,
                           ["descubrimiento", "transferida", "nativa"])
""")

md("## 10 · Resumen")
md("""
> **Pendiente:** este resumen se redacta después de ejecutar el notebook, con los resultados reales del pasaporte (§7.1), las transiciones (§7.2), el valor pronóstico (§7.3) y la sensibilidad (§8). No se escriben conclusiones antes de ver los datos.

Limitaciones conocidas de antemano:
- **Abandono informativo.** Los horizontes tardíos (H12) sobrerrepresentan a quien sigue en seguimiento, que suele estar peor (notebook 01, §8).
- **TENACITY con N pequeño en H6 y H12** (unos 65): las replicaciones tienen IC anchos.
- **H3 de la capa A** se define solo con 2 variables.
- **No se imputa.** Gower usa los pares disponibles, y quien tiene pocas medidas pesa con menos información.
""")

nb = nbf.v4.new_notebook()
nb["cells"] = celdas
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                  "language_info": {"name": "python"}}
salida = Path(__file__).resolve().parents[1] / "03_fenotipado_horizontes.ipynb"
nbf.write(nb, salida)
print(f"Escrito {salida} ({len(celdas)} celdas)")
