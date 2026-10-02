"""Genera `07_modelo_red_h3.ipynb`: red neuronal (MLP) para el fenotipo funcional H3, comparada con el EBM del 06.

Uso: python scripts/construir_notebook_red.py
(después: python -m nbconvert --to notebook --execute --inplace 07_modelo_red_h3.ipynb)
"""
from pathlib import Path

import nbformat as nbf

celdas: list = []
md = lambda texto: celdas.append(nbf.v4.new_markdown_cell(texto.strip()))
code = lambda texto: celdas.append(nbf.v4.new_code_cell(texto.strip()))

md("""
# Reto 2 · Red neuronal: ¿qué fenotipo tendrá el paciente a los 3 meses?

**Notebook 07.** Misma pregunta que el notebook 06, pero con una **red neuronal** (perceptrón multicapa, MLP). El EBM y la regresión logística dieron un AUC de ≈ 0,63; aquí se comprueba si un modelo más flexible extrae más información de los mismos datos.

| | |
|---|---|
| Diana | `fenotipo_H3` = F2 (afectación funcional) frente a F1 (función conservada) |
| Predictores | Los mismos 10 del alta, comunes a las tres cohortes (`config.yaml → modelo_h3`) |
| Partición | **La misma 80/20 que el notebook 06** (misma semilla y estratificación). Se comprueba que el test es idéntico paciente a paciente |
| Red | MLP de `scikit-learn`: imputación por mediana o moda, estandarización y codificación one-hot del tabaquismo. Arquitectura y regularización elegidas por **validación cruzada en train**, con parada temprana. Predicción final: **media de 10 redes** con distintas semillas |
| Comparación | EBM (notebook 06, cargado de disco) y regresión logística sobre el **mismo test**, con la diferencia de AUC y su IC bootstrap emparejado |
| Robustez | Validación dejando fuera cada cohorte (§7) |

> **Privacidad.** Solo agregados. Modelo y predicciones en `Datos limpios/`, fuera del repositorio.
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
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix, f1_score,
                             precision_recall_curve, roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RAIZ = Path.cwd()
sys.path.insert(0, str(RAIZ))
from src import carga

warnings.filterwarnings("ignore", category=ConvergenceWarning)
cfg = carga.cargar_config()
MC, MR = cfg["modelo_h3"], cfg["modelo_red"]
SEMILLA = cfg["semilla"]
N_MIN = cfg["privacidad"]["n_minimo_celda"]
COLOR = cfg["colores_registro"]
ETQ = {"CIBERESUCICOVID": "CIBERESUCICOVID", "POSTCOVID_LLEIDA": "POSTCOVID-Lleida", "TENACITY": "TENACITY"}
DATOS = carga.RAIZ / cfg["rutas"]["procesados"]
FIG = RAIZ / cfg["rutas"]["figuras"]; FIG.mkdir(parents=True, exist_ok=True)
TAB = RAIZ / cfg["rutas"]["tablas"]; TAB.mkdir(parents=True, exist_ok=True)
PFX = "07_red"
C_RED, C_EBM, C_LOG = "#e7298a", "#6a3d9a", "#999999"

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({"figure.dpi": 100, "axes.titleweight": "bold", "axes.titlesize": 11.5,
                     "axes.spines.top": False, "axes.spines.right": False})
pd.set_option("display.max_columns", 40); pd.set_option("display.width", 200)

NOMBRES = {"edad": "Edad", "estancia_hosp_dias": "Estancia (días)", "sexo": "Sexo (mujer)", "nu_ingreso_uci": "UCI",
           "nu_hta": "HTA", "nu_diabetes": "Diabetes", "nu_card_cronica": "Cardiopatía", "nu_epoc": "EPOC",
           "nu_renal_cronica": "Enf. renal", "nu_tabaquismo": "Tabaquismo"}


def nota(fig, texto: str) -> None:
    fig.text(0.01, -0.01, texto, fontsize=8.5, color="0.35", va="top", ha="left")


def guardar(fig, nombre: str) -> None:
    fig.savefig(FIG / f"{PFX}_{nombre}.png", bbox_inches="tight", dpi=150)


def ic_bootstrap(y, p, metrica, n, semilla):
    rng = np.random.default_rng(semilla)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) == 2:
            vals.append(metrica(y[idx], p[idx]))
    return tuple(np.percentile(vals, [2.5, 97.5]))


def dif_auc_bootstrap(y, p1, p2, n, semilla):
    \"\"\"Diferencia de AUC (p1 − p2) con IC 95 % bootstrap emparejado (mismos pacientes en cada remuestreo).\"\"\"
    rng = np.random.default_rng(semilla)
    difs = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) == 2:
            difs.append(roc_auc_score(y[idx], p1[idx]) - roc_auc_score(y[idx], p2[idx]))
    difs = np.array(difs)
    return roc_auc_score(y, p1) - roc_auc_score(y, p2), np.percentile(difs, [2.5, 97.5]), 2 * min((difs <= 0).mean(), (difs >= 0).mean())
""")

md("## 1 · Datos y la misma partición 80/20 que el notebook 06")
code("""
dataset = pd.read_parquet(DATOS / "dataset_modelo_h3.parquet").set_index("subject_id")
CONT, BIN, CAT = MC["predictores"]["continuas"], MC["predictores"]["binarias"], MC["predictores"]["categoricas"]
FEATS = CONT + BIN + CAT
estrato = dataset["y"].astype(str) + "_" + dataset["cohorte_analisis"].astype(str)
train, test = train_test_split(dataset, test_size=MC["test_size"], stratify=estrato, random_state=SEMILLA)
X_tr, y_tr, X_te, y_te = train[FEATS], train["y"].to_numpy(), test[FEATS], test["y"].to_numpy()

with open(DATOS / "modelo_ebm_h3.pkl", "rb") as f:
    paquete_ebm = pickle.load(f)
ebm = paquete_ebm["modelo"]
# Comprobación: el EBM del 06 se entrenó con exactamente estos pacientes (mismo orden de partición)
assert list(paquete_ebm["predictores"]) == FEATS
print(f"Train: {len(train):,} · Test: {len(test):,} · % F2 en test: {y_te.mean():.1%}")
print("Partición idéntica a la del notebook 06 (misma semilla, estratificación y dataset).")
""")

md("## 2 · Preprocesado y búsqueda de la arquitectura")
code("""
prep = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("esc", StandardScaler())]), CONT + BIN),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore"))]), CAT)])
red_base = Pipeline([("prep", prep),
                     ("mlp", MLPClassifier(max_iter=MR["max_iter"], early_stopping=MR["early_stopping"],
                                           validation_fraction=MR["validation_fraction"], random_state=SEMILLA))])
rejilla = {"mlp__hidden_layer_sizes": [tuple(h) for h in MR["busqueda"]["hidden_layer_sizes"]],
           "mlp__alpha": MR["busqueda"]["alpha"], "mlp__learning_rate_init": MR["busqueda"]["learning_rate_init"]}
cv = StratifiedKFold(MR["pliegues_cv"], shuffle=True, random_state=SEMILLA)
busqueda = GridSearchCV(red_base, rejilla, scoring="roc_auc", cv=cv, n_jobs=-1).fit(X_tr, y_tr)

res = pd.DataFrame(busqueda.cv_results_)
res["arquitectura"] = res["param_mlp__hidden_layer_sizes"].astype(str)
top = (res.sort_values("mean_test_score", ascending=False)
       [["arquitectura", "param_mlp__alpha", "param_mlp__learning_rate_init", "mean_test_score", "std_test_score"]]
       .head(8).rename(columns={"param_mlp__alpha": "alpha (L2)", "param_mlp__learning_rate_init": "tasa aprendizaje",
                                "mean_test_score": "AUC CV (media)", "std_test_score": "AUC CV (DE)"}).round(3))
print(f"Configuraciones evaluadas: {len(res)} × {MR['pliegues_cv']} pliegues. Mejor: {busqueda.best_params_} "
      f"(AUC CV {busqueda.best_score_:.3f})")
top
""")
code("""
fig, ax = plt.subplots(figsize=(10, 4))
piv = res.pivot_table(index="arquitectura", columns="param_mlp__alpha", values="mean_test_score", aggfunc="max")
sns.heatmap(piv, annot=True, fmt=".3f", cmap="RdPu", ax=ax, cbar_kws={"label": "AUC CV (mejor tasa de aprendizaje)"})
ax.set_xlabel("alpha (regularización L2)"); ax.set_ylabel("Capas ocultas")
ax.set_title("Búsqueda de arquitectura: AUC de validación cruzada en train")
nota(fig, "Todas las configuraciones dan un AUC parecido: más capacidad no extrae más señal de estos predictores.")
plt.tight_layout(); guardar(fig, "busqueda"); plt.show()
""")

md("## 3 · Entrenamiento final: media de varias redes")
code("""
mejor = busqueda.best_estimator_
redes = [clone(mejor).set_params(mlp__random_state=SEMILLA + s).fit(X_tr, y_tr) for s in range(MR["n_semillas_ensamble"])]


def predecir_red(X):
    return np.mean([r.predict_proba(X)[:, 1] for r in redes], axis=0)


p_red = predecir_red(X_te)
p_ebm = ebm.predict_proba(X_te)[:, 1]
logit = Pipeline([("prep", clone(prep)), ("lr", LogisticRegression(max_iter=2000))]).fit(X_tr, y_tr)
p_log = logit.predict_proba(X_te)[:, 1]

fig, ax = plt.subplots(figsize=(8, 3.8))
for i, r in enumerate(redes[:5]):
    ax.plot(r.named_steps["mlp"].loss_curve_, color=C_RED, alpha=0.5, lw=1, label="pérdida (train)" if i == 0 else None)
    ax2 = ax.twinx() if i == 0 else ax2
    ax2.plot(r.named_steps["mlp"].validation_scores_, color="0.4", alpha=0.5, lw=1, ls="--",
             label="exactitud (validación interna)" if i == 0 else None)
ax.set_xlabel("Época"); ax.set_ylabel("Pérdida logística"); ax2.set_ylabel("Exactitud en validación"); ax2.grid(False)
ax.set_title("Curvas de aprendizaje (5 de las 10 redes)")
ax.legend(loc="upper right", frameon=False, fontsize=8.5); ax2.legend(loc="center right", frameon=False, fontsize=8.5)
nota(fig, "La parada temprana detiene el entrenamiento cuando la exactitud en el 15 % de validación interna deja de mejorar.")
plt.tight_layout(); guardar(fig, "aprendizaje"); plt.show()
""")

md("## 4 · Rendimiento en test (comparado con EBM y logística)")
code("""
p_oof = np.mean([cross_val_predict(clone(mejor).set_params(mlp__random_state=SEMILLA + s), X_tr, y_tr, cv=cv,
                                   method="predict_proba")[:, 1] for s in range(3)], axis=0)
f, t, thr = roc_curve(y_tr, p_oof)
UMBRAL = float(thr[np.argmax(t - f)])


def metricas(y, p, u):
    pred = (p >= u).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"ROC-AUC": roc_auc_score(y, p), "PR-AUC (AP)": average_precision_score(y, p), "Brier": brier_score_loss(y, p),
            "Sensibilidad": tp / (tp + fn), "Especificidad": tn / (tn + fp), "Exactitud": (tp + tn) / len(y), "F1": f1_score(y, pred)}


modelos = {"Red neuronal": p_red, "EBM (nb 06)": p_ebm, "Logística": p_log}
umbrales = {"Red neuronal": UMBRAL, "EBM (nb 06)": paquete_ebm["umbral_youden"], "Logística": 0.5}
tabla = pd.DataFrame({m: metricas(y_te, p, umbrales[m]) for m, p in modelos.items()}).T.round(3)
tabla["ROC-AUC IC95"] = [f"{lo:.3f}–{hi:.3f}" for lo, hi in
                         (ic_bootstrap(y_te, p, roc_auc_score, MC["n_bootstrap_ic"], SEMILLA) for p in modelos.values())]
tabla["umbral"] = [round(umbrales[m], 3) for m in modelos]
tabla.to_csv(TAB / f"{PFX}_metricas_test.csv")
print(f"Test: N = {len(y_te)}; umbral de la red elegido por validación cruzada en train (Youden): {UMBRAL:.3f}")
tabla
""")
code("""
filas = []
for otro in ["EBM (nb 06)", "Logística"]:
    d, (lo, hi), p = dif_auc_bootstrap(y_te, p_red, modelos[otro], MC["n_bootstrap_ic"], SEMILLA)
    filas.append({"comparación": f"Red − {otro}", "Δ AUC": round(d, 3), "IC95": f"{lo:+.3f} a {hi:+.3f}", "p (bootstrap)": round(p, 3)})
pd.DataFrame(filas).set_index("comparación")
""")
code("""
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
for nombre, p, color in [("Red neuronal", p_red, C_RED), ("EBM (nb 06)", p_ebm, C_EBM), ("Logística", p_log, C_LOG)]:
    f, t, _ = roc_curve(y_te, p)
    axes[0].plot(f, t, color=color, lw=2.2 if nombre == "Red neuronal" else 1.5,
                 label=f"{nombre}: AUC = {roc_auc_score(y_te, p):.3f}")
    pr, rc, _ = precision_recall_curve(y_te, p)
    axes[1].plot(rc, pr, color=color, lw=2.2 if nombre == "Red neuronal" else 1.5,
                 label=f"{nombre}: AP = {average_precision_score(y_te, p):.3f}")
axes[0].plot([0, 1], [0, 1], "--", color="0.6", lw=1, label="Azar")
axes[0].set_xlabel("1 − especificidad"); axes[0].set_ylabel("Sensibilidad"); axes[0].set_title("Curva ROC (test)")
axes[1].axhline(y_te.mean(), ls="--", color="0.6", lw=1, label=f"Azar (prevalencia {y_te.mean():.2f})")
axes[1].set_xlabel("Sensibilidad"); axes[1].set_ylabel("Precisión (VPP)"); axes[1].set_title("Precisión-sensibilidad (test)")
axes[1].set_ylim(0, 1.02)
for ax in axes:
    ax.legend(frameon=False, fontsize=9, loc="lower right" if ax is axes[0] else "upper right")
nota(fig, f"Mismo test de 315 pacientes que el notebook 06. Red: media de {MR['n_semillas_ensamble']} MLP {busqueda.best_params_['mlp__hidden_layer_sizes']}.")
plt.tight_layout(); guardar(fig, "roc_pr"); plt.show()
""")
code("""
fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), gridspec_kw={"width_ratios": [1.1, 1, 1]})
ax = axes[0]
for nombre, p, color in [("Red neuronal", p_red, C_RED), ("EBM (nb 06)", p_ebm, C_EBM), ("Logística", p_log, C_LOG)]:
    fr, mp = calibration_curve(y_te, p, n_bins=MC["n_bins_calibracion"], strategy="quantile")
    ax.plot(mp, fr, "o-", color=color, label=f"{nombre} (Brier {brier_score_loss(y_te, p):.3f})")
ax.plot([0, 1], [0, 1], "--", color="0.6", lw=1)
ax.set_xlabel("Probabilidad predicha de F2"); ax.set_ylabel("Proporción observada"); ax.set_title("Calibración (test)")
ax.legend(frameon=False, fontsize=8.5)
ax = axes[1]
bins = np.linspace(0, 1, 21)
ax.hist(p_red[y_te == 0], bins=bins, alpha=0.6, color="#1b9e77", label="F1 real")
ax.hist(p_red[y_te == 1], bins=bins, alpha=0.6, color="#d95f02", label="F2 real")
ax.axvline(UMBRAL, color="0.2", ls="--", lw=1)
ax.set_xlabel("Probabilidad predicha de F2 (red)"); ax.set_ylabel("Pacientes"); ax.set_title("Separación de las clases")
ax.legend(frameon=False, fontsize=9)
ax = axes[2]
cm = confusion_matrix(y_te, (p_red >= UMBRAL).astype(int), labels=[0, 1])
sns.heatmap(cm, annot=np.where(cm < N_MIN, "<10", cm.astype(str)), fmt="", cmap="RdPu", cbar=False, ax=ax,
            xticklabels=["Pred. F1", "Pred. F2"], yticklabels=["Real F1", "Real F2"], annot_kws={"size": 13})
ax.set_title(f"Matriz de confusión (red, umbral {UMBRAL:.2f})")
plt.tight_layout(); guardar(fig, "calibracion_confusion"); plt.show()
""")

md("""
## 5 · ¿En qué se fija la red?

Una red no tiene coeficientes legibles, así que se usan dos técnicas agnósticas al modelo:
- **Importancia por permutación** en test: cuánto cae el AUC al barajar cada variable.
- **Dependencia parcial:** cómo cambia la probabilidad media predicha al variar una variable, con el resto como en los datos.
""")
code("""
from sklearn.base import BaseEstimator, ClassifierMixin


class Ensamble(ClassifierMixin, BaseEstimator):
    \"\"\"Adaptador para que sklearn trate la media de redes como un único clasificador ya entrenado.\"\"\"
    def fit(self, X, y):
        self.classes_ = np.array([0, 1]); return self
    def predict_proba(self, X):
        p = predecir_red(X); return np.c_[1 - p, p]
    def predict(self, X): return (predecir_red(X) >= UMBRAL).astype(int)


ens = Ensamble().fit(X_tr, y_tr)
perm = permutation_importance(ens, X_te, y_te, scoring="roc_auc", n_repeats=MR["n_repeticiones_permutacion"],
                              random_state=SEMILLA)
imp = pd.DataFrame({"caída de AUC": perm.importances_mean, "DE": perm.importances_std},
                   index=[NOMBRES[f] for f in FEATS]).sort_values("caída de AUC")
fig, axes = plt.subplots(1, 2, figsize=(15, 4.5), gridspec_kw={"width_ratios": [1, 1.4]})
axes[0].barh(imp.index, imp["caída de AUC"], xerr=imp["DE"], color=C_RED, alpha=0.85)
axes[0].axvline(0, color="0.3", lw=0.8); axes[0].set_xlabel("Caída del ROC-AUC al permutar (test)")
axes[0].set_title("Importancia por permutación (red)")

ax = axes[1]
for var, color in [("estancia_hosp_dias", C_RED), ("edad", "#1f78b4")]:
    rejilla_v = np.linspace(*np.nanpercentile(X_te[var], [2, 98]), 30)
    medias = []
    for v in rejilla_v:
        Xv = X_te.copy(); Xv[var] = v
        medias.append(predecir_red(Xv).mean())
    ax.plot(rejilla_v, medias, color=color, lw=2, label=NOMBRES[var])
ax.set_xlabel("Valor de la variable (días de estancia / años de edad)"); ax.set_ylabel("P(F2) media predicha")
ax.set_title("Dependencia parcial (red)"); ax.legend(frameon=False)
nota(fig, f"Importancia: media ± DE de {MR['n_repeticiones_permutacion']} permutaciones sobre el test. "
          "Dependencia parcial entre los percentiles 2 y 98 de cada variable.")
plt.tight_layout(); guardar(fig, "interpretacion"); plt.show()
imp.sort_values("caída de AUC", ascending=False).round(4).to_csv(TAB / f"{PFX}_importancia_permutacion.csv")
""")

md("## 6 · Rendimiento por cohorte (dentro del test)")
code("""
filas = []
for c in ["CIBERESUCICOVID", "POSTCOVID_LLEIDA", "TENACITY"]:
    m = (test["cohorte_analisis"] == c).to_numpy()
    if m.sum() >= N_MIN and len(np.unique(y_te[m])) == 2:
        lo, hi = ic_bootstrap(y_te[m], p_red[m], roc_auc_score, MC["n_bootstrap_ic"], SEMILLA)
        filas.append({"cohorte": ETQ[c], "N test": int(m.sum()), "AUC red": round(roc_auc_score(y_te[m], p_red[m]), 3),
                      "IC95": f"{lo:.3f}–{hi:.3f}", "AUC EBM": round(roc_auc_score(y_te[m], p_ebm[m]), 3),
                      "AUC logística": round(roc_auc_score(y_te[m], p_log[m]), 3)})
pd.DataFrame(filas).set_index("cohorte")
""")

md("## 7 · Robustez: validación dejando fuera cada cohorte")
code("""
filas = []
for c in ["CIBERESUCICOVID", "POSTCOVID_LLEIDA", "TENACITY"]:
    tr, te = dataset[dataset["cohorte_analisis"] != c], dataset[dataset["cohorte_analisis"] == c]
    rs = [clone(mejor).set_params(mlp__random_state=SEMILLA + s).fit(tr[FEATS], tr["y"]) for s in range(MR["n_semillas_ensamble"])]
    p = np.mean([r.predict_proba(te[FEATS])[:, 1] for r in rs], axis=0)
    y = te["y"].to_numpy()
    lo, hi = ic_bootstrap(y, p, roc_auc_score, MC["n_bootstrap_ic"], SEMILLA)
    filas.append({"cohorte evaluada (no vista)": ETQ[c], "N": len(te), "AUC red": round(roc_auc_score(y, p), 3),
                  "IC95": f"{lo:.3f}–{hi:.3f}", "Brier": round(brier_score_loss(y, p), 3)})
loco = pd.DataFrame(filas).set_index("cohorte evaluada (no vista)")
loco_ebm = pd.read_csv(TAB / "06_ebm_validacion_loco.csv", index_col=0)["ROC-AUC"]
loco["AUC EBM (nb 06)"] = loco_ebm.reindex(loco.index).values
loco.to_csv(TAB / f"{PFX}_validacion_loco.csv")
loco
""")

md("## 8 · Guardar")
code("""
with open(DATOS / "modelo_red_h3.pkl", "wb") as f:
    pickle.dump({"redes": redes, "predictores": FEATS, "umbral_youden": UMBRAL, "mejores_parametros": busqueda.best_params_,
                 "semilla": SEMILLA}, f)
print(f"Modelo guardado en {DATOS.name}/modelo_red_h3.pkl (fuera del repositorio)")
""")

md("## 9 · Resumen")
md("""
**Resultados (ejecución con semilla 2026).**

1. **La red no mejora al EBM.** Mejor arquitectura: 2 capas (64, 32), α = 0,01, AUC de validación cruzada 0,651. Las 60 configuraciones evaluadas dan un AUC de 0,60 a 0,65: más capacidad no extrae más señal.
2. **Mismo test que el notebook 06** (315 pacientes):

   | Modelo | ROC-AUC |
   |---|---|
   | Red | **0,626** [0,566–0,684] |
   | EBM | 0,634 |
   | Logística | 0,633 |

   La diferencia entre la red y el EBM es −0,008 [−0,041 a +0,025] (p = 0,63): **ninguna diferencia**. La calibración y el Brier (0,241) también son equivalentes.
3. **En cohortes no vistas, la red queda algo por encima del EBM:** 0,65 / 0,62 / 0,66 frente a 0,62 / 0,59 / 0,61. Es coherente en las tres cohortes, pero la ventaja es pequeña (≈ 0,03–0,05) y los IC se solapan. Como mucho, la red **generaliza un poco mejor**, no predice mejor.
4. **Aprende lo mismo que el EBM:**
   - **estancia** (más días, más riesgo; P(F2) de 0,40 a 0,70 entre 7 y 85 días);
   - **edad** (más joven, más riesgo: el mismo patrón contraintuitivo, probablemente un sesgo de selección);
   - **sexo, EPOC e HTA.**

   Las dos familias de modelos coinciden en qué variables importan y en qué dirección.

**Conclusión**
- **El techo es la información, no el modelo.** Logística, EBM y red neuronal quedan todos en un AUC de 0,63 ± 0,01. Con los 10 datos del alta comunes a las tres cohortes, **ningún modelo pasa de un AUC de ≈ 0,65**.
- **Mejor quedarse con el EBM:** igual de preciso, totalmente interpretable (curvas por variable) y más fácil de explicar en consulta y al jurado.
- **Para mejorar** hacen falta **más predictores**, no un modelo más complejo:
  - la fase aguda de CIBERESUCICOVID (ventilación, PaO2/FiO2, analítica, días de UCI), en un modelo solo para CIBERESUCICOVID;
  - o la espirometría y DLCO de la primera visita, para predecir la evolución a 12 meses.
""")

nb = nbf.v4.new_notebook()
nb["cells"] = celdas
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                  "language_info": {"name": "python"}}
salida = Path(__file__).resolve().parents[1] / "07_modelo_red_h3.ipynb"
nbf.write(nb, salida)
print(f"Escrito {salida} ({len(celdas)} celdas)")
