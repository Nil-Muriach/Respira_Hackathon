"""Genera `08_modelo_ebm_ciberes.ipynb`: EBM multiclase sobre CIBERESUCICOVID (fenotipos sintomáticos del notebook 03).

Uso: python scripts/construir_notebook_ciberes_ebm.py
(después: python -m nbconvert --to notebook --execute --inplace 08_modelo_ebm_ciberes.ipynb)
"""
from pathlib import Path

import nbformat as nbf

celdas: list = []
md = lambda texto: celdas.append(nbf.v4.new_markdown_cell(texto.strip()))
code = lambda texto: celdas.append(nbf.v4.new_code_cell(texto.strip()))

md("""
# Reto 2 · EBM en CIBERESUCICOVID: ¿qué fenotipo sintomático tendrá el paciente a los 3 meses?

**Notebook 08.** Modelo **solo con CIBERESUCICOVID**. Predice, con los datos del alta y de la **fase aguda**, cuál de los **3 fenotipos del notebook 03** tendrá el paciente a los 3 meses:
- **F1 · resolución completa:** sin síntomas, la mejor función.
- **F2 · persistente con fatiga.**
- **F3 · persistente sin fatiga:** la peor función, estancias más largas.

| | |
|---|---|
| Diana | Fenotipo H3 del notebook 03 (3 clases) |
| Predictores | Los **10 del alta** del notebook 06 **+ 20 de la fase aguda** que solo recoge CIBERESUCICOVID: ventilación invasiva y no invasiva, prono, ECMO, corticoides, vasopresores, días de UCI y de VMI, APACHE II, SOFA, PaO2/FiO2 y analítica al ingreso en UCI |
| Modelo | **EBM multiclase** (`interpret`): una curva por variable y clase, totalmente interpretable |
| Partición | **80 % train / 20 % test**, estratificada por fenotipo |
| Comparación | EBM solo con los 10 del alta (¿cuánto aporta la fase aguda?) y regresión logística multinomial |
| Robustez | Validación **agrupada por centro**: centros que el modelo no ha visto (§7) |
| Sin fuga | Todos los predictores son anteriores al alta; la diana se mide a los 3 meses (síntomas, DLCO y FVC) |

> **Privacidad.** Solo agregados. Dataset y modelo en `Datos limpios/`, fuera del repositorio.
""")

md("## 0 · Entorno")
code("""
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from interpret.glassbox import ExplainableBoostingClassifier
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, log_loss,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, label_binarize

RAIZ = Path.cwd()
sys.path.insert(0, str(RAIZ))
from src import carga, privacidad

warnings.filterwarnings("ignore", category=UserWarning)
cfg = carga.cargar_config()
MC = cfg["modelo_ciberes"]
SEMILLA = cfg["semilla"]
N_MIN = cfg["privacidad"]["n_minimo_celda"]
DATOS = carga.RAIZ / cfg["rutas"]["procesados"]
FIG = RAIZ / cfg["rutas"]["figuras"]; FIG.mkdir(parents=True, exist_ok=True)
TAB = RAIZ / cfg["rutas"]["tablas"]; TAB.mkdir(parents=True, exist_ok=True)
PFX = "08_ebm_ciberes"
CLASES = ["F1", "F2", "F3"]
NOMBRE_CLASE = MC["nombres_clase"]
COLOR_CLASE = {"F1": "#1b9e77", "F2": "#d95f02", "F3": "#7570b3"}
C_EBM, C_ALTA, C_LOG = "#6a3d9a", "#cab2d6", "#999999"

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({"figure.dpi": 100, "axes.titleweight": "bold", "axes.titlesize": 11.5,
                     "axes.spines.top": False, "axes.spines.right": False})
pd.set_option("display.max_columns", 40); pd.set_option("display.width", 200)

ALTA = MC["predictores_alta"]
AGU_BIN = MC["predictores_agudos"]["binarias"]
AGU_CONT = MC["predictores_agudos"]["continuas"]
NOMBRES = {"edad": "Edad", "estancia_hosp_dias": "Estancia (días)", "sexo": "Sexo (mujer)", "nu_ingreso_uci": "UCI",
           "nu_hta": "HTA", "nu_diabetes": "Diabetes", "nu_card_cronica": "Cardiopatía", "nu_epoc": "EPOC",
           "nu_renal_cronica": "Enf. renal", "nu_tabaquismo": "Tabaquismo",
           **AGU_BIN, **{k: v[0] for k, v in AGU_CONT.items()}}


def nota(fig, texto: str) -> None:
    fig.text(0.01, -0.01, texto, fontsize=8.5, color="0.35", va="top", ha="left")


def guardar(fig, nombre: str) -> None:
    fig.savefig(FIG / f"{PFX}_{nombre}.png", bbox_inches="tight", dpi=150)


def edad_aproximada(tramo) -> float:
    if not isinstance(tramo, str) or tramo.startswith("<"):
        return np.nan
    if tramo.endswith("+"):
        return float(tramo[:-1]) + 2
    lo, hi = tramo.split("-")
    return (float(lo) + float(hi)) / 2


def auc_macro(y, P):
    return roc_auc_score(y, P, multi_class="ovr", average="macro", labels=[0, 1, 2])


def ic_bootstrap(y, P, metrica, n, semilla):
    rng = np.random.default_rng(semilla)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) == 3:
            vals.append(metrica(y[idx], P[idx]))
    return tuple(np.percentile(vals, [2.5, 97.5]))
""")

md("""
## 1 · Dataset

Fenotipo H3 del notebook 03 (pacientes de descubrimiento de CIBERESUCICOVID), datos del alta (tabla `pacientes`) y fase aguda (tabla completa). Limpieza de las variables agudas:
- **código 2** ("se desconoce") → ausente;
- **valores fuera del rango plausible** (`config.yaml → modelo_ciberes`) → ausente;
- **sin imputación:** el EBM trata el valor ausente como una categoría más.
""")
code("""
fen = pd.read_parquet(DATOS / MC["fenotipos"])
fen = fen[(fen["horizonte"] == MC["horizonte"]) & (fen["tipo"] == "descubrimiento")].set_index("subject_id")
pacientes = pd.read_parquet(DATOS / "pacientes.parquet").set_index("subject_id")

d = pacientes.loc[fen.index, ["centro_id", "edad_tramo5", *[c for c in ALTA["continuas"] + ALTA["binarias"] + ALTA["categoricas"] if c != "edad"]]].copy()
d["edad"] = d.pop("edad_tramo5").map(edad_aproximada)
agudas = carga.cargar_columnas_completa(cfg, list(AGU_BIN) + list(AGU_CONT)).set_index("subject_id").reindex(fen.index)
limpieza = {}
for v in AGU_BIN:
    n_cod = int((agudas[v] == MC["codigo_desconocido_agudo"]).sum())
    agudas[v] = agudas[v].where(agudas[v].isin([0, 1]))
    limpieza[NOMBRES[v]] = {"código 2 → NaN": n_cod, "fuera de rango → NaN": 0}
for v, (_, lo, hi) in AGU_CONT.items():
    fuera = agudas[v].notna() & ~agudas[v].between(lo, hi)
    limpieza[NOMBRES[v]] = {"código 2 → NaN": 0, "fuera de rango → NaN": int(fuera.sum())}
    agudas[v] = agudas[v].where(~fuera)
d = d.join(agudas)
d["fenotipo"] = fen["fenotipo"]
d["y"] = d["fenotipo"].map({c: i for i, c in enumerate(CLASES)})

F_ALTA = ALTA["continuas"] + ALTA["binarias"] + ALTA["categoricas"]
F_AGU = list(AGU_BIN) + list(AGU_CONT)
FEATS = F_ALTA + F_AGU
d.reset_index().to_parquet(DATOS / "dataset_modelo_ciberes.parquet", index=False)
print(f"Dataset: {len(d)} pacientes de CIBERESUCICOVID × {len(FEATS)} predictores ({len(F_ALTA)} del alta + {len(F_AGU)} de la fase aguda)")
print("Clases:", {f"{c} · {NOMBRE_CLASE[c]}": int((d['fenotipo'] == c).sum()) for c in CLASES})
privacidad.suprimir_celdas(pd.DataFrame(limpieza).T, N_MIN, ["código 2 → NaN", "fuera de rango → NaN"])
""")
code("""
# Descripción por fenotipo: mediana (continuas) o % (binarias), cobertura y prueba de Kruskal-Wallis / χ²
filas = []
for v in FEATS:
    if v == "nu_tabaquismo":
        continue
    fila = {"variable": NOMBRES[v], "cobertura (%)": round(100 * d[v].notna().mean())}
    binaria = v in ALTA["binarias"] or v in AGU_BIN
    grupos = [d.loc[d["fenotipo"] == c, v].dropna() for c in CLASES]
    for c, g in zip(CLASES, grupos):
        k = int(g.sum()) if binaria else None
        fila[c] = (round(100 * g.mean(), 1) if (k == 0 or (k or 0) >= N_MIN) else np.nan) if binaria else round(g.median(), 2)
    try:
        p = (stats.chi2_contingency(pd.crosstab(d["fenotipo"], d[v]))[1] if binaria else stats.kruskal(*grupos).pvalue)
    except ValueError:
        p = np.nan
    fila["p"] = f"{p:.2g}"
    filas.append(fila)
desc = pd.DataFrame(filas).set_index("variable")
desc.columns = [f"{c} · {NOMBRE_CLASE[c]}" if c in CLASES else c for c in desc.columns]
print("Binarias: % con la característica; continuas: mediana. p: χ² (binarias) o Kruskal-Wallis (continuas).")
desc
""")

md("## 2 · Partición 80 / 20")
code("""
train, test = train_test_split(d, test_size=MC["test_size"], stratify=d["y"], random_state=SEMILLA)
y_tr, y_te = train["y"].to_numpy(), test["y"].to_numpy()
pd.DataFrame({"train": train["fenotipo"].value_counts(), "test": test["fenotipo"].value_counts()}).loc[CLASES] \\
    .rename(index=lambda c: f"{c} · {NOMBRE_CLASE[c]}").assign(**{"% test": lambda x: (100 * x["test"] / x["test"].sum()).round(1)})
""")

md("## 3 · Entrenamiento")
code("""
def ebm_para(cols):
    tipos = ["continuous" if (c in ALTA["continuas"] or c in AGU_CONT) else "nominal" for c in cols]
    return ExplainableBoostingClassifier(feature_names=[NOMBRES[c] for c in cols], feature_types=tipos,
                                        random_state=SEMILLA, **MC["ebm"])


ebm = ebm_para(FEATS).fit(train[FEATS], y_tr)            # principal: alta + fase aguda
ebm_alta = ebm_para(F_ALTA).fit(train[F_ALTA], y_tr)     # comparación: solo alta

num = ALTA["continuas"] + ALTA["binarias"] + F_AGU
logit = Pipeline([
    ("prep", ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)), ("esc", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore"))]),
         ALTA["categoricas"])])),
    ("lr", LogisticRegression(max_iter=5000, C=0.5))]).fit(train[FEATS], y_tr)

P = {"EBM alta + aguda": ebm.predict_proba(test[FEATS]),
     "EBM solo alta": ebm_alta.predict_proba(test[F_ALTA]),
     "Logística multinomial": logit.predict_proba(test[FEATS])}
COLOR_MOD = {"EBM alta + aguda": C_EBM, "EBM solo alta": C_ALTA, "Logística multinomial": C_LOG}
print(f"EBM principal: {len(ebm.term_names_)} términos ({len(FEATS)} efectos principales + {len(ebm.term_names_) - len(FEATS)} interacciones)")
""")

md("## 4 · Rendimiento en test")
code("""
filas = {}
for nombre, Pm in P.items():
    pred = Pm.argmax(axis=1)
    lo, hi = ic_bootstrap(y_te, Pm, auc_macro, MC["n_bootstrap_ic"], SEMILLA)
    fila = {"ROC-AUC macro (OvR)": auc_macro(y_te, Pm), "IC95": f"{lo:.3f}–{hi:.3f}"}
    for i, c in enumerate(CLASES):
        fila[f"AUC {c} vs resto"] = roc_auc_score((y_te == i).astype(int), Pm[:, i])
    fila.update({"Exactitud": accuracy_score(y_te, pred), "Exactitud balanceada": balanced_accuracy_score(y_te, pred),
                 "F1 macro": f1_score(y_te, pred, average="macro"), "Log-loss": log_loss(y_te, Pm, labels=[0, 1, 2])})
    filas[nombre] = fila
tabla = pd.DataFrame(filas).T
tabla.loc["Azar (referencia)"] = {"ROC-AUC macro (OvR)": 0.5, "Exactitud": np.bincount(y_te).max() / len(y_te),
                                   "Exactitud balanceada": 1 / 3}
tabla.to_csv(TAB / f"{PFX}_metricas_test.csv")
print(f"Test: N = {len(y_te)}")
tabla.round(3)
""")
code("""
fig, axes = plt.subplots(1, 3, figsize=(17, 5), sharey=True)
for i, (ax, c) in enumerate(zip(axes, CLASES)):
    yb = (y_te == i).astype(int)
    for nombre, Pm in P.items():
        f, t, _ = roc_curve(yb, Pm[:, i])
        ax.plot(f, t, color=COLOR_MOD[nombre], lw=2.3 if nombre.startswith("EBM alta +") else 1.5,
                label=f"{nombre}: AUC = {roc_auc_score(yb, Pm[:, i]):.3f}")
    ax.plot([0, 1], [0, 1], "--", color="0.6", lw=1)
    ax.set_title(f"{c} · {NOMBRE_CLASE[c]} (frente al resto)", color=COLOR_CLASE[c])
    ax.set_xlabel("1 − especificidad"); ax.legend(frameon=False, fontsize=8.5, loc="lower right")
axes[0].set_ylabel("Sensibilidad")
nota(fig, f"Test: N = {len(y_te)} pacientes de CIBERESUCICOVID (20 %, estratificado por fenotipo). Curvas ROC uno contra el resto.")
plt.tight_layout(); guardar(fig, "roc_por_clase"); plt.show()
""")
code("""
fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
ax = axes[0]
x = np.arange(len(CLASES) + 1)
for k, (nombre, Pm) in enumerate(P.items()):
    vals = [roc_auc_score((y_te == i).astype(int), Pm[:, i]) for i in range(3)] + [auc_macro(y_te, Pm)]
    ax.bar(x + (k - 1) * 0.26, vals, width=0.26, color=COLOR_MOD[nombre], label=nombre)
lo, hi = ic_bootstrap(y_te, P["EBM alta + aguda"], auc_macro, MC["n_bootstrap_ic"], SEMILLA)
ax.errorbar(x[-1] - 0.26, auc_macro(y_te, P["EBM alta + aguda"]),
            yerr=[[auc_macro(y_te, P["EBM alta + aguda"]) - lo], [hi - auc_macro(y_te, P["EBM alta + aguda"])]],
            color="k", capsize=4)
ax.axhline(0.5, color="0.4", ls="--", lw=1)
ax.set_xticks(x, [f"{c} vs resto" for c in CLASES] + ["Macro"]); ax.set_ylim(0.4, 0.9)
ax.set_ylabel("ROC-AUC (test)"); ax.set_title("¿Cuánto aporta la fase aguda?"); ax.legend(frameon=False, fontsize=8.5)

ax = axes[1]
cm = confusion_matrix(y_te, P["EBM alta + aguda"].argmax(axis=1), labels=[0, 1, 2])
pct = cm / cm.sum(axis=1, keepdims=True) * 100
anot = np.array([[f"{pct[i, j]:.0f} %\\n(n={cm[i, j]})" if cm[i, j] >= N_MIN else f"{pct[i, j]:.0f} %\\n(<10)"
                  for j in range(3)] for i in range(3)])
sns.heatmap(pct, annot=anot, fmt="", cmap="Purples", vmin=0, vmax=100, cbar=False, ax=ax,
            xticklabels=[f"Pred. {c}" for c in CLASES], yticklabels=[f"Real {c}" for c in CLASES])
ax.set_title("Matriz de confusión (EBM alta + aguda, % por fila)")
nota(fig, "Barra de error: IC 95 % bootstrap del AUC macro del EBM principal. Clase predicha = la de mayor probabilidad.")
plt.tight_layout(); guardar(fig, "comparacion_confusion"); plt.show()
""")
code("""
fig, axes = plt.subplots(1, 3, figsize=(16, 4.4), sharey=True)
Pm = P["EBM alta + aguda"]
for i, (ax, c) in enumerate(zip(axes, CLASES)):
    yb = (y_te == i).astype(int)
    fr, mp = calibration_curve(yb, Pm[:, i], n_bins=MC["n_bins_calibracion"], strategy="quantile")
    ax.plot(mp, fr, "o-", color=COLOR_CLASE[c], lw=2, label="EBM alta + aguda")
    ax.plot([0, 1], [0, 1], "--", color="0.6", lw=1, label="Perfecta")
    ax2 = ax.twinx(); ax2.hist(Pm[:, i], bins=np.linspace(0, 1, 21), color=COLOR_CLASE[c], alpha=0.15); ax2.set_yticks([]); ax2.grid(False)
    ax.set_title(f"Calibración · {c} {NOMBRE_CLASE[c]}", fontsize=10.5); ax.set_xlabel(f"P({c}) predicha"); ax.set_xlim(0, 1)
axes[0].set_ylabel("Proporción observada"); axes[0].legend(frameon=False, fontsize=8.5)
nota(fig, "Sombreado: distribución de las probabilidades predichas en test.")
plt.tight_layout(); guardar(fig, "calibracion"); plt.show()
""")

md("""
## 5 · Interpretación del EBM

### 5.1 Importancia global
Contribución media absoluta a los log-odds, promediada sobre las tres clases.
""")
code("""
imp = pd.Series(ebm.term_importances(), index=ebm.term_names_).sort_values()
top = imp.tail(20)
fig, ax = plt.subplots(figsize=(8.5, 0.32 * len(top) + 1.3))
alta_nombres = {NOMBRES[c] for c in F_ALTA}
colores = ["#cab2d6" if " & " in t else (C_ALTA if t in alta_nombres else C_EBM) for t in top.index]
ax.barh(top.index, top.values, color=colores)
ax.set_xlabel("Importancia (media |contribución| al log-odds)"); ax.set_title("EBM: los 20 términos más importantes")
nota(fig, "Morado oscuro: fase aguda; lila: datos del alta; rosa pálido: interacciones.")
plt.tight_layout(); guardar(fig, "importancia"); plt.show()
imp.sort_values(ascending=False).round(4).to_frame("importancia").to_csv(TAB / f"{PFX}_importancia.csv")
""")
md("""
### 5.2 Funciones de forma por clase
Para cada una de las variables más importantes, cómo cambia la contribución al log-odds de **cada fenotipo**. Valores por encima de 0 hacen más probable ese fenotipo.
""")
code("""
glob = ebm.explain_global()
principales = [i for i, t in enumerate(ebm.term_names_) if " & " not in t]
orden = sorted(principales, key=lambda i: -ebm.term_importances()[i])[:8]
fig, axes = plt.subplots(2, 4, figsize=(18, 7.5))
for ax, i in zip(axes.flat, orden):
    dat = glob.data(i)
    sc = np.array(dat["scores"])
    nombre = ebm.term_names_[i]
    if ebm.feature_types_in_[i] == "continuous":
        bordes = np.array(dat["names"], float)
        for k, c in enumerate(CLASES):
            ax.step(bordes, np.r_[sc[:, k], sc[-1, k]], where="post", color=COLOR_CLASE[c], lw=2, label=c)
        col = [k for k, v in NOMBRES.items() if v == nombre][0]
        ax.set_xlim(*np.nanpercentile(d[col], [1, 99]))
    else:
        etiquetas = [str(x).replace(".0", "") for x in dat["names"]]
        for k, c in enumerate(CLASES):
            ax.bar(np.arange(len(etiquetas)) + (k - 1) * 0.27, sc[:, k], width=0.27, color=COLOR_CLASE[c], label=c)
        ax.set_xticks(range(len(etiquetas)), etiquetas)
    ax.axhline(0, color="0.3", lw=0.8)
    ax.set_title(f"{nombre} (imp. {ebm.term_importances()[i]:.2f})", fontsize=10)
axes.flat[0].legend(title="Fenotipo", frameon=False, fontsize=8.5)
axes.flat[0].set_ylabel("Contribución al log-odds"); axes.flat[4].set_ylabel("Contribución al log-odds")
nota(fig, " · ".join(f"{c} = {NOMBRE_CLASE[c]}" for c in CLASES) + ". Binarias: 0 = no, 1 = sí. Eje X recortado entre P1 y P99.")
plt.tight_layout(); guardar(fig, "formas"); plt.show()
""")

md("""
## 6 · ¿Cuánto aporta cada bloque de variables?

Diferencia de AUC macro entre el EBM con fase aguda y el EBM solo con el alta, en el mismo test, con IC bootstrap emparejado.
""")
code("""
rng = np.random.default_rng(SEMILLA)
difs = []
for _ in range(MC["n_bootstrap_ic"]):
    idx = rng.integers(0, len(y_te), len(y_te))
    if len(np.unique(y_te[idx])) == 3:
        difs.append(auc_macro(y_te[idx], P["EBM alta + aguda"][idx]) - auc_macro(y_te[idx], P["EBM solo alta"][idx]))
difs = np.array(difs)
delta = auc_macro(y_te, P["EBM alta + aguda"]) - auc_macro(y_te, P["EBM solo alta"])
pd.DataFrame([{"comparación": "EBM alta + aguda − EBM solo alta", "Δ AUC macro": round(delta, 3),
               "IC95": f"{np.percentile(difs, 2.5):+.3f} a {np.percentile(difs, 97.5):+.3f}",
               "p (bootstrap)": round(2 * min((difs <= 0).mean(), (difs >= 0).mean()), 3)}]).set_index("comparación")
""")

md("""
## 7 · Robustez: centros no vistos

La partición 80/20 mezcla centros. Para estimar cómo funcionaría el modelo **en un hospital nuevo**, se hace validación cruzada **agrupada por centro**: en cada pliegue, todos los pacientes de un grupo de centros quedan fuera del entrenamiento. Es la validación que pide el `CLAUDE.md` para el clasificador.
""")
code("""
dc = d[d["centro_id"].notna()]
gkf = GroupKFold(n_splits=MC["pliegues_centro"])
filas = []
for k, (itr, ite) in enumerate(gkf.split(dc, dc["y"], groups=dc["centro_id"])):
    tr, te = dc.iloc[itr], dc.iloc[ite]
    if len(np.unique(te["y"])) < 3:
        continue
    m = ebm_para(FEATS).fit(tr[FEATS], tr["y"])
    m_a = ebm_para(F_ALTA).fit(tr[F_ALTA], tr["y"])
    filas.append({"pliegue": k + 1, "centros evaluados": te["centro_id"].nunique(), "N evaluación": len(te),
                  "AUC macro (alta + aguda)": auc_macro(te["y"].to_numpy(), m.predict_proba(te[FEATS])),
                  "AUC macro (solo alta)": auc_macro(te["y"].to_numpy(), m_a.predict_proba(te[F_ALTA]))})
centros = pd.DataFrame(filas).set_index("pliegue")
centros.loc["media"] = centros.mean()
centros.round(3).to_csv(TAB / f"{PFX}_validacion_centros.csv")
print(f"Pacientes con centro conocido: {len(dc)} de {len(d)}; {dc['centro_id'].nunique()} centros.")
centros.round(3)
""")

md("## 8 · Guardar el modelo")
code("""
with open(DATOS / "modelo_ebm_ciberes.pkl", "wb") as f:
    pickle.dump({"modelo": ebm, "predictores": FEATS, "clases": CLASES, "nombres_clase": NOMBRE_CLASE, "semilla": SEMILLA}, f)
print(f"Modelo guardado en {DATOS.name}/modelo_ebm_ciberes.pkl (fuera del repositorio)")
""")

md("## 9 · Resumen")
md("""
**Resultados (ejecución con semilla 2026).**

1. **El modelo apenas supera al azar.** En el test (183 pacientes):
   - **ROC-AUC macro 0,575** [0,509–0,635]. Por clase: F1 0,59, F2 0,60 y F3 0,54.
   - **Exactitud 39 %**, igual que la de predecir siempre la clase más frecuente (40 %).
   - Calibración pobre: las probabilidades predichas se agrupan cerca de 1/3.
2. **La fase aguda no aporta nada.** El EBM con alta + aguda (30 variables) da el mismo AUC que el que solo usa el alta (10 variables): Δ = 0,00 [−0,06 a +0,06].
3. **En centros no vistos se confirma:** AUC macro 0,58 con alta + aguda y 0,59 solo con alta, en 41 centros y 5 pliegues. El modelo no está sobreajustado: la señal es débil en todas partes.
4. **Hay diferencias entre fenotipos, pero son pequeñas y se solapan (§1).** F2 y F3 tienen más estancia, más días de UCI y de VMI, más ventilación invasiva y no invasiva y más vasopresores que F1. F2 tiene más mujeres. Las diferencias son significativas de uno en uno (p < 0,01), pero las distribuciones se solapan demasiado para clasificar pacientes.

**Por qué no funciona**
- **La diana es sintomática.** Los fenotipos del notebook 03 los definen la fatiga y la resolución clínica, que son síntomas autorreferidos a los 3 meses. La gravedad de la fase aguda predice mal si un paciente se sentirá cansado o resuelto. Ni la analítica, ni el SOFA, ni la PaO2/FiO2 lo anticipan.
- **Los fenotipos son poco transferibles** (notebook 03: estructura modesta) y se superponen en la función pulmonar: la diana es en sí misma difusa.
- **La analítica tiene una cobertura del 40–75 %.** El EBM la aprovecha sin imputar, pero con poca información.

**Implicaciones**
- **El cuello de botella es la diana, no los predictores ni el modelo.** Con la diana funcional del notebook 06 (fenotipo H3 común), el AUC era de 0,63; con la sintomática de CIBERESUCICOVID baja a 0,58, aun con 20 variables agudas más.
- **Mejor diana para un modelo de CIBERESUCICOVID:** un desenlace funcional (fenotipo H3 común, o DLCO < 80 % a 12 meses) en lugar del fenotipo sintomático.
""")

nb = nbf.v4.new_notebook()
nb["cells"] = celdas
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                  "language_info": {"name": "python"}}
salida = Path(__file__).resolve().parents[1] / "08_modelo_ebm_ciberes.ipynb"
nbf.write(nb, salida)
print(f"Escrito {salida} ({len(celdas)} celdas)")
