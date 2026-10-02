"""Fenotipado: variables acumuladas por horizonte, distancia de Gower, PAM, prueba nula y estabilidad.

Diseño (docs/registro_decisiones.md, notebook 03):
- Horizonte acumulado: H6 usa las ventanas T3 y T6; H12, T3, T6 y T12. Nunca datos posteriores.
- Gower con rangos clínicos FIJOS (no z-scores por cohorte): misma escala en descubrimiento
  y replicación. Ausencias por pares disponibles; no se imputa.
- Peso igual por dominio clínico, repartido entre las columnas del dominio.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import kmedoids
from sklearn.metrics import adjusted_rand_score, silhouette_score

from src import carga


# ---------------------------------------------------------------------------
# Variables

def variables_capa(cfg: dict, capa: str) -> list[str]:
    """Variables de definición de una capa, en orden de dominio."""
    return [v for vs in cfg["fenotipado"]["capas"][capa]["dominios"].values() for v in vs]


def asignar_ventana(dias: pd.Series, cfg: dict) -> pd.Series:
    """Etiqueta de ventana (T3, T6, T12) de cada medida según `dias_alta`; NaN fuera de ventana."""
    salida = pd.Series(pd.NA, index=dias.index, dtype="object")
    for nombre, (lo, hi) in cfg["fenotipado"]["ventanas"].items():
        salida[dias.between(lo, hi)] = nombre
    return salida


def construir_variables(medidas: pd.DataFrame, pacientes: pd.DataFrame, cfg: dict,
                        horizonte: str, capa: str) -> tuple[pd.DataFrame, list[str]]:
    """Tabla de pacientes incluidos en (horizonte, capa) con sus variables de definición.

    Columnas: `v_Tk` (primera medida válida de v en la ventana Tk ≤ horizonte) y, si hay más
    de una ventana, `v_cambio` = último valor disponible − primero (NaN si solo hay uno).
    Incluye metadatos (`cohorte_analisis`, `centro_id`, `dias_max`) fuera de la lista de columnas.
    Inclusión: base (superviviente, sin inconsistencia, cmd_ok) + ancla en la última ventana.
    """
    fcfg = cfg["fenotipado"]
    ventanas = fcfg["horizontes"][horizonte]
    variables = variables_capa(cfg, capa)

    m = medidas[~medidas["excluida"] & medidas["variable"].isin(variables) & medidas["dias_alta"].notna()].copy()
    m["ventana"] = asignar_ventana(m["dias_alta"], cfg)
    m = m[m["ventana"].isin(ventanas)]
    primera = m.sort_values("dias_alta").groupby(["subject_id", "variable", "ventana"]).first().reset_index()

    valores = primera.pivot_table(index="subject_id", columns=["variable", "ventana"], values="valor", aggfunc="first")
    columnas = []
    tabla = pd.DataFrame(index=valores.index)
    for v in variables:
        cols_v = []
        for t in ventanas:
            nombre = f"{v}_{t}"
            tabla[nombre] = valores[(v, t)] if (v, t) in valores.columns else np.nan
            cols_v.append(nombre)
        columnas += cols_v
        if len(ventanas) > 1:
            serie = tabla[cols_v]
            ultimo = serie.ffill(axis=1).iloc[:, -1]
            primero = serie.bfill(axis=1).iloc[:, 0]
            cambio = (ultimo - primero).where(serie.notna().sum(axis=1) >= 2)
            tabla[f"{v}_cambio"] = cambio
            columnas.append(f"{v}_cambio")
    tabla["dias_max"] = primera.groupby("subject_id")["dias_alta"].max()

    # Ancla: en la última ventana, al menos una variable de cada grupo
    ultima = ventanas[-1]
    ancla = pd.Series(True, index=tabla.index)
    for grupo in fcfg["capas"][capa]["ancla"]:
        ancla &= tabla[[f"{v}_{ultima}" for v in grupo]].notna().any(axis=1)

    base = pacientes.set_index("subject_id")
    base = base[base["superviviente"] & ~base["inconsistente"] & base["cmd_ok"]]
    tabla = tabla[ancla & tabla.index.isin(base.index)]
    tabla = tabla.join(base[["cohorte_analisis", "centro_id"]])
    return tabla, columnas


def parametros_gower(columnas: list[str], cfg: dict, capa: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mínimo, rango y peso de cada columna. Peso igual por dominio, repartido entre sus columnas."""
    fcfg = cfg["fenotipado"]
    dominio_de = {v: d for d, vs in fcfg["capas"][capa]["dominios"].items() for v in vs}
    minimos, rangos, dominios = [], [], []
    for c in columnas:
        v, sufijo = c.rsplit("_", 1)
        lo, hi = fcfg["rango_cambio"][v] if sufijo == "cambio" else cfg["rangos_plausibles"][v]
        minimos.append(lo); rangos.append(hi - lo); dominios.append(dominio_de[v])
    dominios = np.array(dominios)
    n_dom = len(set(dominios))
    pesos = np.array([1.0 / n_dom / (dominios == d).sum() for d in dominios])
    return np.array(minimos, float), np.array(rangos, float), pesos


# ---------------------------------------------------------------------------
# Distancia y PAM

def gower(X: np.ndarray, Y: np.ndarray | None, rangos: np.ndarray, pesos: np.ndarray) -> np.ndarray:
    """Distancia de Gower (variables numéricas) entre filas de X e Y, con ausencias.

    d(i, j) = Σ_k w_k δ_ijk min(|x_ik − y_jk| / R_k, 1) / Σ_k w_k δ_ijk, con δ = 1 si ambas están.
    Si dos pacientes no comparten ninguna variable, la distancia es 1 (máxima, conservadora).
    """
    Y = X if Y is None else Y
    num = np.zeros((X.shape[0], Y.shape[0]))
    den = np.zeros_like(num)
    for k in range(X.shape[1]):
        x, y = X[:, k][:, None], Y[:, k][None, :]
        presente = ~np.isnan(x) & ~np.isnan(y)
        diff = np.minimum(np.abs(x - y) / rangos[k], 1.0)
        num += pesos[k] * np.where(presente, diff, 0.0)
        den += pesos[k] * presente
    with np.errstate(invalid="ignore", divide="ignore"):
        D = np.where(den > 0, num / den, 1.0)
    if Y is X:
        np.fill_diagonal(D, 0.0)
    return D


def pam(D: np.ndarray, k: int, semilla: int) -> tuple[np.ndarray, np.ndarray]:
    """PAM (FasterPAM con inicialización BUILD). Devuelve (etiquetas 0..k-1, índices de medoides)."""
    r = kmedoids.fasterpam(D, k, init="build", random_state=semilla, n_cpu=1)
    return np.asarray(r.labels), np.asarray(r.medoids)


def asignar_medoides(D_a_medoides: np.ndarray) -> np.ndarray:
    """Etiqueta de cada paciente = medoide más cercano (columnas en el orden de las etiquetas)."""
    return D_a_medoides.argmin(axis=1)


def silueta(D: np.ndarray, etiquetas: np.ndarray) -> float:
    """Silueta media con distancias precalculadas."""
    if len(np.unique(etiquetas)) < 2:
        return np.nan
    return float(silhouette_score(D, etiquetas, metric="precomputed"))


def ordenar_por_severidad(etiquetas: np.ndarray, medoides: np.ndarray, valores_dlco: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Renumera los clústeres 0..k-1 por DLCO mediana descendente (0 = menos afectado)."""
    k = len(medoides)
    mediana = [np.nanmedian(valores_dlco[etiquetas == c]) if np.any(etiquetas == c) else np.nan for c in range(k)]
    orden = np.argsort(-np.nan_to_num(np.array(mediana), nan=-np.inf))
    mapa = np.empty(k, int); mapa[orden] = np.arange(k)
    return mapa[etiquetas], medoides[orden]


# ---------------------------------------------------------------------------
# Elección de k y estabilidad

def seleccion_k_nulo(X: np.ndarray, rangos: np.ndarray, pesos: np.ndarray, ks: list[int],
                     n_perm: int, semilla: int, D: np.ndarray | None = None) -> pd.DataFrame:
    """Silueta real frente a la de datos permutados columna a columna (destruye la estructura conjunta).

    Devuelve por k: silueta real, media y percentil 95 del nulo y la diferencia real − media nula.
    """
    rng = np.random.default_rng(semilla)
    D = gower(X, None, rangos, pesos) if D is None else D
    real = {k: silueta(D, pam(D, k, semilla)[0]) for k in ks}
    nulo = {k: [] for k in ks}
    for _ in range(n_perm):
        Xp = np.column_stack([rng.permutation(X[:, j]) for j in range(X.shape[1])])
        Dp = gower(Xp, None, rangos, pesos)
        for k in ks:
            nulo[k].append(silueta(Dp, pam(Dp, k, semilla)[0]))
    tabla = pd.DataFrame({
        "k": ks,
        "silueta_real": [real[k] for k in ks],
        "nulo_media": [np.nanmean(nulo[k]) for k in ks],
        "nulo_p95": [np.nanpercentile(nulo[k], 95) for k in ks],
    })
    tabla["diferencia"] = tabla["silueta_real"] - tabla["nulo_media"]
    tabla["supera_p95"] = tabla["silueta_real"] > tabla["nulo_p95"]
    return tabla


def elegir_k(tabla_nulo: pd.DataFrame) -> tuple[int, bool]:
    """k con mayor diferencia frente al nulo entre los que superan el p95; si ninguno, el de mayor
    diferencia y `False` (sin estructura demostrable)."""
    validos = tabla_nulo[tabla_nulo["supera_p95"]]
    if len(validos):
        return int(validos.loc[validos["diferencia"].idxmax(), "k"]), True
    return int(tabla_nulo.loc[tabla_nulo["diferencia"].idxmax(), "k"]), False


def estabilidad_bootstrap(D: np.ndarray, etiquetas: np.ndarray, k: int, n_boot: int, semilla: int) -> pd.DataFrame:
    """Jaccard bootstrap por clúster (Hennig): media del mejor Jaccard en cada remuestreo.

    Orientación habitual: < 0,5 disuelto; 0,6–0,75 patrón; > 0,85 muy estable.
    """
    rng = np.random.default_rng(semilla)
    n = len(etiquetas)
    jacc = {c: [] for c in range(k)}
    for b in range(n_boot):
        idx = np.unique(rng.integers(0, n, n))
        lab_b, _ = pam(D[np.ix_(idx, idx)], k, semilla + b)
        for c in range(k):
            orig = set(idx[etiquetas[idx] == c])
            if not orig:
                continue
            jacc[c].append(max(len(orig & set(idx[lab_b == cb])) / len(orig | set(idx[lab_b == cb])) for cb in range(k)))
    return pd.DataFrame({"cluster": range(k), "n": [int((etiquetas == c).sum()) for c in range(k)],
                         "jaccard_medio": [np.mean(jacc[c]) if jacc[c] else np.nan for c in range(k)],
                         "jaccard_p05": [np.percentile(jacc[c], 5) if jacc[c] else np.nan for c in range(k)]})


def estabilidad_centros(D: np.ndarray, etiquetas: np.ndarray, centros: pd.Series, k: int,
                        min_n: int, semilla: int) -> pd.DataFrame:
    """Deja fuera cada centro grande, reagrupa sin él y asigna sus pacientes a los nuevos medoides.

    ARI entre esa asignación y la etiqueta original de los pacientes del centro.
    """
    centros = centros.reset_index(drop=True)
    filas = []
    for centro, n in centros.value_counts().items():
        if n < min_n:
            continue
        dentro = np.where(centros == centro)[0]
        fuera = np.where(centros != centro)[0]
        lab, med = pam(D[np.ix_(fuera, fuera)], k, semilla)
        asignado = asignar_medoides(D[np.ix_(dentro, fuera[med])])
        filas.append({"centro": centro, "n": int(n), "ari": adjusted_rand_score(etiquetas[dentro], asignado)})
    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# Descripción

def perfiles(datos: pd.DataFrame, columnas: list[str], etiquetas: np.ndarray) -> pd.DataFrame:
    """Mediana de cada variable de definición por clúster (más N)."""
    p = datos[columnas].groupby(etiquetas).median().T
    p.loc["N"] = pd.Series(etiquetas).value_counts().sort_index().values
    return p


def porcentaje_alterado(datos: pd.DataFrame, columnas: list[str], etiquetas: np.ndarray,
                        cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """% de pacientes con la variable alterada (umbral clínico) por clúster y nº de alterados por celda."""
    filas, ns = {}, {}
    for c in columnas:
        v, sufijo = c.rsplit("_", 1)
        if sufijo == "cambio" or v not in cfg["estados"]:
            continue
        clave, direccion = cfg["estados"][v]
        u = cfg["umbrales_clinicos"][clave]
        alt = ((datos[c] < u) if direccion == "<" else (datos[c] >= u)).astype(float)
        alt = alt.where(datos[c].notna())
        filas[c] = 100 * alt.groupby(etiquetas).mean()
        ns[c] = alt.groupby(etiquetas).sum()
    return pd.DataFrame(filas).T, pd.DataFrame(ns).T
