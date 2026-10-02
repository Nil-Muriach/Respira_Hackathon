"""Trayectorias de la función pulmonar según el fenotipo (notebook 05).

Diseño (CLAUDE.md, config.yaml -> trayectorias):
- Tiempo = días reales desde el alta (no el número de visita), en escala log(meses / mes_ref): la recuperación
  tras el alta es rápida al principio y se frena después, y con el tiempo centrado el intercepto es el valor
  a los `mes_ref` meses.
- Modelo mixto `valor ~ fenotipo * lt (+ cohorte)` con intercepto y pendiente aleatorios por paciente.
- Abandono informativo: pesos inversos de la probabilidad de seguir en seguimiento (IPW) en un GEE, y medias
  observadas por patrón de seguimiento.
- Solo se usan medidas del registro de la cohorte de análisis del paciente (las duplicadas ya se excluyeron en
  la limpieza, regla R06).
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Datos

def tabla_larga(medidas: pd.DataFrame, fenotipo: pd.Series, cohorte: pd.Series, variable: str,
                ventana_dias: tuple[float, float], dias_por_mes: float, mes_ref: float) -> pd.DataFrame:
    """Una fila por medida válida de `variable` de los pacientes con fenotipo, dentro de la ventana.

    `fenotipo` y `cohorte` son series indexadas por `subject_id`. Solo se conservan las medidas del registro
    de la cohorte de análisis del paciente y con fecha (`dias_alta`) dentro de `ventana_dias` (inclusive).
    Columnas: subject_id, cohorte, fenotipo, visita, dias, meses, lt = log(meses / mes_ref), valor.
    """
    m = medidas[~medidas["excluida"] & (medidas["variable"] == variable)
                & medidas["subject_id"].isin(fenotipo.index)]
    coh = m["subject_id"].map(cohorte).astype(str)
    m = m[(m["registro"].astype(str) == coh) & m["dias_alta"].between(*ventana_dias)]
    meses = m["dias_alta"] / dias_por_mes
    return pd.DataFrame({
        "subject_id": m["subject_id"].values,
        "cohorte": m["subject_id"].map(cohorte).astype(str).values,
        "fenotipo": m["subject_id"].map(fenotipo).astype(str).values,
        "visita": m["visita"].values,
        "dias": m["dias_alta"].values,
        "meses": meses.values,
        "lt": np.log(meses.values / mes_ref),
        "valor": m["valor"].astype(float).values,
    }).sort_values(["subject_id", "dias"], ignore_index=True)


def tramo(dias: pd.Series, cortes: list[float], etiquetas: list[str]) -> pd.Series:
    """Tramo temporal (p. ej., T3, T6, T12) de cada medida según sus días; el primer corte es inclusivo."""
    return pd.cut(dias, cortes, labels=etiquetas, include_lowest=True)


def patron_seguimiento(presencia: pd.DataFrame) -> pd.Series:
    """Patrón de seguimiento de cada paciente: tramos con medida unidos por ' + ' ('ninguno' si no hay)."""
    return presencia.apply(lambda fila: " + ".join(c for c, v in fila.items() if v) or "ninguno", axis=1)


# ---------------------------------------------------------------------------
# Modelos

def ajustar_mixto(datos: pd.DataFrame, formula: str, re_formula: str | None = "~lt",
                  grupo: str = "subject_id") -> tuple:
    """Modelo lineal mixto (REML). Intenta intercepto + pendiente aleatorios (`re_formula`); si no converge o
    falla, solo intercepto aleatorio. Devuelve (resultado, descripción de los efectos aleatorios)."""
    if re_formula is not None:
        try:
            res = smf.mixedlm(formula, datos, groups=datos[grupo], re_formula=re_formula).fit(reml=True, method=["lbfgs"])
            if res.converged:
                return res, "intercepto y pendiente aleatorios"
        except (np.linalg.LinAlgError, ValueError):
            pass
    res = smf.mixedlm(formula, datos, groups=datos[grupo]).fit(reml=True, method=["lbfgs"])
    return res, "solo intercepto aleatorio"


def ajustar_gee(datos: pd.DataFrame, formula: str, pesos: np.ndarray | None = None, grupo: str = "subject_id"):
    """GEE gaussiano con correlación de trabajo independiente y errores robustos (sándwich).

    Con `pesos` = IPW por observación, estima la trayectoria media que se vería sin abandono, si este depende
    solo de lo observado (supuesto MAR dado las covariables del modelo de pesos).
    """
    return smf.gee(formula, grupo, datos, cov_struct=sm.cov_struct.Independence(),
                   family=sm.families.Gaussian(), weights=pesos).fit()


def _valor_termino(termino: str, fila: dict, pesos_cohorte: dict[str, float] | None) -> float:
    """Valor de un término de la fórmula ('Intercept', 'lt', 'fenotipo[T.F2]', 'fenotipo[T.F2]:lt', …)."""
    valor = 1.0
    for factor in termino.split(":"):
        if factor == "Intercept":
            continue
        m = re.fullmatch(r"(\w+)\[T\.(.+)\]", factor)
        if m is None:
            valor *= float(fila[factor])
        elif m.group(1) == "cohorte" and pesos_cohorte is not None:
            valor *= pesos_cohorte.get(m.group(2), 0.0)
        else:                                  # categoría ausente en `fila` -> nivel de referencia
            valor *= float(str(fila.get(m.group(1))) == m.group(2))
    return valor


def _parametros(res) -> tuple[pd.Series, pd.DataFrame]:
    """Efectos fijos y su matriz de covarianzas (MixedLM o GEE)."""
    fe = res.fe_params if hasattr(res, "fe_params") else res.params
    return fe, res.cov_params().loc[fe.index, fe.index]


def vector_contraste(res, fila: dict, pesos_cohorte: dict[str, float] | None = None) -> np.ndarray:
    """Vector x tal que x·β es la predicción del modelo para `fila` (fenotipo, lt…).

    Si el modelo incluye la cohorte y se dan `pesos_cohorte`, la predicción es la media ponderada entre cohortes
    (predicción marginal); los pesos deben sumar 1 e incluir la cohorte de referencia.
    """
    fe, _ = _parametros(res)
    return np.array([_valor_termino(t, fila, pesos_cohorte) for t in fe.index])


def combinacion(res, x: np.ndarray, alfa: float = 0.05) -> dict:
    """Estimación, IC y p de la combinación lineal x·β."""
    fe, V = _parametros(res)
    est = float(x @ fe.to_numpy())
    ee = float(np.sqrt(x @ V.to_numpy() @ x))
    z = stats.norm.ppf(1 - alfa / 2)
    return {"estimacion": est, "ee": ee, "ic_inf": est - z * ee, "ic_sup": est + z * ee,
            "p": float(2 * stats.norm.sf(abs(est / ee))) if ee > 0 else np.nan}


def predicciones(res, fenotipos: list[str], meses: list[float], mes_ref: float,
                 pesos_cohorte: dict[str, float] | None = None) -> pd.DataFrame:
    """Valor medio predicho por fenotipo y tiempo, con IC 95 % (solo efectos fijos)."""
    filas = []
    for f in fenotipos:
        for t in meses:
            x = vector_contraste(res, {"fenotipo": f, "lt": np.log(t / mes_ref)}, pesos_cohorte)
            filas.append({"fenotipo": f, "meses": t, **combinacion(res, x)})
    return pd.DataFrame(filas)


def diferencias(res, a: str, b: str, meses: list[float], mes_ref: float,
                pesos_cohorte: dict[str, float] | None = None) -> pd.DataFrame:
    """Diferencia predicha b − a en cada tiempo, con IC 95 % y p."""
    filas = []
    for t in meses:
        fila = {"lt": np.log(t / mes_ref)}
        x = vector_contraste(res, {**fila, "fenotipo": b}, pesos_cohorte) - \
            vector_contraste(res, {**fila, "fenotipo": a}, pesos_cohorte)
        filas.append({"meses": t, **combinacion(res, x)})
    return pd.DataFrame(filas)


def ganancia(res, fenotipo: str, desde: float, hasta: float, mes_ref: float) -> dict:
    """Cambio medio predicho entre dos tiempos (meses) para un fenotipo, con IC 95 %."""
    x = vector_contraste(res, {"fenotipo": fenotipo, "lt": np.log(hasta / mes_ref)}) - \
        vector_contraste(res, {"fenotipo": fenotipo, "lt": np.log(desde / mes_ref)})
    return combinacion(res, x)


# ---------------------------------------------------------------------------
# Abandono

def pesos_ipw(X: pd.DataFrame, observado: pd.Series, recorte_percentil: float = 99) -> pd.Series:
    """Pesos estabilizados P(observado) / P(observado | X) para los pacientes observados (NaN en el resto).

    La probabilidad se estima con una logística (imputación por mediana y estandarización). Los pesos se
    recortan en el percentil indicado para que unos pocos pacientes no dominen la estimación.
    """
    modelo = Pipeline([("imp", SimpleImputer(strategy="median")), ("esc", StandardScaler()),
                       ("lr", LogisticRegression(max_iter=5000))])
    obs = observado.astype(int).to_numpy()
    p = modelo.fit(X, obs).predict_proba(X)[:, 1]
    w = obs.mean() / p
    w = np.minimum(w, np.percentile(w[obs == 1], recorte_percentil))
    return pd.Series(np.where(obs == 1, w, np.nan), index=X.index)
