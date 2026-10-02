"""Fenotipado de secuelas en CIBERESUCICOVID y comparación con Lleida y TENACITY.

Diseño (docs/registro_decisiones.md, notebook 03):
- Descubrimiento solo en CIBERESUCICOVID: un único cuaderno, sin armonización entre registros.
- Variables de definición: función pulmonar (DLCO, FVC) y síntomas (fatiga, resolución clínica),
  peso igual por dominio. Horizonte acumulado: H6 usa M3 y M6; H12, M3, M6 y A1.
- Las medidas se asignan por VISITA (muchos síntomas de CIBERESUCICOVID no tienen fecha); si hay
  fecha, debe caer en la ventana de tolerancia de la visita.
- Gower con rangos clínicos fijos; binarias por coincidencia; ausencias por pares disponibles.
- Lleida y TENACITY solo comparten DLCO y FVC: la transferencia usa esas variables (sus
  columnas de síntomas quedan vacías y Gower las ignora).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import kmedoids
from sklearn.metrics import adjusted_rand_score, silhouette_score


# ---------------------------------------------------------------------------
# Variables

def variables_definicion(cfg: dict) -> list[str]:
    """Variables de definición en orden de dominio."""
    return [v for vs in cfg["fenotipado"]["dominios"].values() for v in vs]


def visitas_que_recogen(cfg: dict, variable: str) -> set[str]:
    """Visitas de CIBERESUCICOVID en las que se recoge la variable (según `variables_visita`)."""
    filas = cfg["variables_visita"].get(variable, {}).get(cfg["fenotipado"]["descubre"], [])
    return {f[0] for f in filas}


def medidas_por_visita(medidas: pd.DataFrame, cfg: dict, cohorte: str, variables: list[str]) -> pd.DataFrame:
    """Medidas válidas de `cohorte` etiquetadas con la visita equivalente de CIBERESUCICOVID (M3, M6, A1).

    Se excluyen las que tienen fecha fuera de la ventana de tolerancia de su visita.
    """
    fc = cfg["fenotipado"]
    mapa = {(v if cohorte == fc["descubre"] else d.get(cohorte)): v for v, d in fc["visitas"].items()}
    mapa.pop(None, None)
    m = medidas[~medidas["excluida"] & (medidas["registro"].astype(str) == cohorte)
                & medidas["variable"].isin(variables) & medidas["visita"].isin(mapa)].copy()
    m["visita_c"] = m["visita"].map(mapa)
    lo = m["visita_c"].map(lambda v: fc["visitas"][v]["ventana"][0])
    hi = m["visita_c"].map(lambda v: fc["visitas"][v]["ventana"][1])
    return m[m["dias_alta"].isna() | m["dias_alta"].between(lo, hi)]


def construir_variables(medidas: pd.DataFrame, pacientes: pd.DataFrame, cfg: dict,
                        horizonte: str, cohorte: str) -> tuple[pd.DataFrame, list[str]]:
    """Pacientes de `cohorte` incluidos en el horizonte, con sus variables de definición.

    Columnas `v_VISITA` para cada variable y visita ≤ horizonte, y `v_cambio` (último − primero)
    para las variables con rango de cambio. Inclusión: superviviente, sin inconsistencia, cmd_ok y
    ancla en la última visita (un valor de cada grupo, si la visita recoge alguna variable del grupo).
    """
    fc = cfg["fenotipado"]
    visitas = fc["horizontes"][horizonte]
    variables = variables_definicion(cfg)
    m = medidas_por_visita(medidas, cfg, cohorte, variables)
    m = m[m["visita_c"].isin(visitas)]
    primera = m.sort_values("dias_alta").groupby(["subject_id", "variable", "visita_c"]).first().reset_index()
    valores = primera.pivot_table(index="subject_id", columns=["variable", "visita_c"], values="valor", aggfunc="first")

    tabla = pd.DataFrame(index=valores.index)
    columnas = []
    for v in variables:
        cols_v = []
        for vis in [vis for vis in visitas if vis in visitas_que_recogen(cfg, v)]:   # nunca columnas no recogidas
            nombre = f"{v}_{vis}"
            tabla[nombre] = valores[(v, vis)] if (v, vis) in valores.columns else np.nan
            cols_v.append(nombre)
        columnas += cols_v
        if v in fc["rango_cambio"] and len(visitas) > 1:
            serie = tabla[cols_v]
            cambio = (serie.ffill(axis=1).iloc[:, -1] - serie.bfill(axis=1).iloc[:, 0]).where(serie.notna().sum(axis=1) >= 2)
            tabla[f"{v}_cambio"] = cambio
            columnas.append(f"{v}_cambio")

    ultima = visitas[-1]
    ancla = pd.Series(True, index=tabla.index)
    for grupo in fc["ancla"]:
        recogidas = [v for v in grupo if ultima in visitas_que_recogen(cfg, v)]
        if cohorte != fc["descubre"]:
            recogidas = [v for v in recogidas if v in fc["compartidas"]]
        if recogidas:
            ancla &= tabla[[f"{v}_{ultima}" for v in recogidas]].notna().any(axis=1)

    base = pacientes.set_index("subject_id")
    base = base[base["superviviente"] & ~base["inconsistente"] & base["cmd_ok"] & (base["cohorte_analisis"] == cohorte)]
    tabla = tabla[ancla & tabla.index.isin(base.index)]
    tabla["dias_max"] = m[m["subject_id"].isin(tabla.index)].groupby("subject_id")["dias_alta"].max()
    tabla = tabla.join(base[["cohorte_analisis", "centro_id"]])
    return tabla, columnas


def parametros_gower(columnas: list[str], cfg: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rango, peso (igual por dominio, repartido entre sus columnas) y bandera categórica por columna."""
    fc = cfg["fenotipado"]
    dominio_de = {v: d for d, vs in fc["dominios"].items() for v in vs}
    rangos, dominios, categ = [], [], []
    for c in columnas:
        v, sufijo = c.rsplit("_", 1)
        lo, hi = fc["rango_cambio"][v] if sufijo == "cambio" else fc["rango_gower"][v]
        rangos.append(hi - lo); dominios.append(dominio_de[v]); categ.append(v in fc["categoricas"])
    dominios = np.array(dominios)
    n_dom = len(set(dominios))
    pesos = np.array([1.0 / n_dom / (dominios == d).sum() for d in dominios])
    return np.array(rangos, float), pesos, np.array(categ)


# ---------------------------------------------------------------------------
# Distancia y PAM

def gower(X: np.ndarray, Y: np.ndarray | None, rangos: np.ndarray, pesos: np.ndarray,
          categoricas: np.ndarray | None = None) -> np.ndarray:
    """Distancia de Gower entre filas de X e Y, con ausencias.

    d(i, j) = Σ_k w_k δ_ijk s_ijk / Σ_k w_k δ_ijk, con δ = 1 si ambas están, s = min(|x − y| / R, 1)
    en numéricas y s = 1(x ≠ y) en categóricas. Sin variables en común: distancia 1 (conservadora).
    """
    Y = X if Y is None else Y
    categoricas = np.zeros(X.shape[1], bool) if categoricas is None else np.asarray(categoricas, bool)
    num = np.zeros((X.shape[0], Y.shape[0]))
    den = np.zeros_like(num)
    for k in range(X.shape[1]):
        x, y = X[:, k][:, None], Y[:, k][None, :]
        presente = ~np.isnan(x) & ~np.isnan(y)
        diff = (x != y).astype(float) if categoricas[k] else np.minimum(np.abs(x - y) / rangos[k], 1.0)
        num += pesos[k] * np.where(presente, diff, 0.0)
        den += pesos[k] * presente
    with np.errstate(invalid="ignore", divide="ignore"):
        D = np.where(den > 0, num / den, 1.0)
    if Y is X:
        np.fill_diagonal(D, 0.0)
    return D


def pam(D: np.ndarray, k: int, semilla: int) -> tuple[np.ndarray, np.ndarray]:
    """PAM (FasterPAM, inicialización BUILD). Devuelve (etiquetas 0..k-1, índices de medoides)."""
    r = kmedoids.fasterpam(D, k, init="build", random_state=semilla, n_cpu=1)
    return np.asarray(r.labels), np.asarray(r.medoids)


def asignar_medoides(D_a_medoides: np.ndarray) -> np.ndarray:
    """Etiqueta = medoide más cercano (columnas en el orden de las etiquetas)."""
    return D_a_medoides.argmin(axis=1)


def silueta(D: np.ndarray, etiquetas: np.ndarray) -> float:
    if len(np.unique(etiquetas)) < 2:
        return np.nan
    return float(silhouette_score(D, etiquetas, metric="precomputed"))


def ordenar_por_dlco(etiquetas: np.ndarray, medoides: np.ndarray, dlco: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Renumera los clústeres por DLCO mediana descendente (0 = mejor función)."""
    k = len(medoides)
    mediana = np.array([np.nanmedian(dlco[etiquetas == c]) if np.any(~np.isnan(dlco[etiquetas == c])) else -np.inf
                        for c in range(k)])
    orden = np.argsort(-mediana, kind="stable")
    mapa = np.empty(k, int); mapa[orden] = np.arange(k)
    return mapa[etiquetas], medoides[orden]


# ---------------------------------------------------------------------------
# Elección de k y estabilidad

def seleccion_k_nulo(X: np.ndarray, rangos: np.ndarray, pesos: np.ndarray, categoricas: np.ndarray,
                     ks: list[int], n_perm: int, semilla: int, D: np.ndarray | None = None) -> pd.DataFrame:
    """Silueta real frente a la de datos permutados columna a columna (sin estructura conjunta)."""
    rng = np.random.default_rng(semilla)
    D = gower(X, None, rangos, pesos, categoricas) if D is None else D
    real = {k: silueta(D, pam(D, k, semilla)[0]) for k in ks}
    nulo = {k: [] for k in ks}
    for _ in range(n_perm):
        Xp = np.column_stack([rng.permutation(X[:, j]) for j in range(X.shape[1])])
        Dp = gower(Xp, None, rangos, pesos, categoricas)
        for k in ks:
            nulo[k].append(silueta(Dp, pam(Dp, k, semilla)[0]))
    tabla = pd.DataFrame({"k": ks, "silueta_real": [real[k] for k in ks],
                          "nulo_media": [np.nanmean(nulo[k]) for k in ks],
                          "nulo_p95": [np.nanpercentile(nulo[k], 95) for k in ks]})
    tabla["diferencia"] = tabla["silueta_real"] - tabla["nulo_media"]
    tabla["supera_p95"] = tabla["silueta_real"] > tabla["nulo_p95"]
    return tabla


def elegir_k(tabla_nulo: pd.DataFrame) -> tuple[int, bool]:
    """k con mayor diferencia frente al nulo entre los que superan el p95 (si ninguno: el mejor y False)."""
    validos = tabla_nulo[tabla_nulo["supera_p95"]]
    if len(validos):
        return int(validos.loc[validos["diferencia"].idxmax(), "k"]), True
    return int(tabla_nulo.loc[tabla_nulo["diferencia"].idxmax(), "k"]), False


def estabilidad_bootstrap(D: np.ndarray, etiquetas: np.ndarray, k: int, n_boot: int, semilla: int) -> pd.DataFrame:
    """Jaccard bootstrap por clúster (Hennig): < 0,5 se disuelve; 0,5–0,75 patrón; > 0,75 estable."""
    rng = np.random.default_rng(semilla)
    n = len(etiquetas)
    jacc = {c: [] for c in range(k)}
    for b in range(n_boot):
        idx = np.unique(rng.integers(0, n, n))
        lab_b, _ = pam(D[np.ix_(idx, idx)], k, semilla + b)
        for c in range(k):
            orig = set(idx[etiquetas[idx] == c])
            if orig:
                jacc[c].append(max(len(orig & set(idx[lab_b == cb])) / len(orig | set(idx[lab_b == cb])) for cb in range(k)))
    return pd.DataFrame({"cluster": range(k), "n": [int((etiquetas == c).sum()) for c in range(k)],
                         "jaccard_medio": [np.mean(jacc[c]) if jacc[c] else np.nan for c in range(k)],
                         "jaccard_p05": [np.percentile(jacc[c], 5) if jacc[c] else np.nan for c in range(k)]})


def estabilidad_centros(D: np.ndarray, etiquetas: np.ndarray, centros: pd.Series, k: int,
                        min_n: int, semilla: int) -> pd.DataFrame:
    """Deja fuera cada centro grande, reagrupa sin él y asigna sus pacientes a los nuevos medoides (ARI)."""
    centros = centros.reset_index(drop=True)
    filas = []
    for centro, n in centros.value_counts().items():
        if n < min_n:
            continue
        dentro = np.where(centros == centro)[0]
        fuera = np.where(centros != centro)[0]
        _, med = pam(D[np.ix_(fuera, fuera)], k, semilla)
        asignado = asignar_medoides(D[np.ix_(dentro, fuera[med])])
        filas.append({"centro": centro, "n": int(n), "ari": adjusted_rand_score(etiquetas[dentro], asignado)})
    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# Descripción

def porcentaje_alterado(datos: pd.DataFrame, columnas: list[str], etiquetas: np.ndarray,
                        cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """% con la variable alterada (umbral clínico) por clúster y nº de alterados por celda."""
    filas, ns = {}, {}
    for c in columnas:
        v, sufijo = c.rsplit("_", 1)
        if sufijo == "cambio" or v not in cfg["estados"]:
            continue
        clave, direccion = cfg["estados"][v]
        u = cfg["umbrales_clinicos"][clave]
        alt = ((datos[c] < u) if direccion == "<" else (datos[c] >= u)).astype(float).where(datos[c].notna())
        filas[c] = 100 * alt.groupby(etiquetas).mean()
        ns[c] = alt.groupby(etiquetas).sum()
    return pd.DataFrame(filas).T, pd.DataFrame(ns).T


def estado_hasta(medidas: pd.DataFrame, cfg: dict, cohorte: str, sujetos: pd.Index,
                 variable: str, visitas: list[str], regla: str = "alguna") -> pd.Series:
    """Estado de una variable descriptiva en las visitas indicadas (equivalentes de CIBERESUCICOVID).

    regla "alguna": 1 si alterada en alguna visita (eventos, hallazgos de imagen);
    "ultima": estado en la visita más reciente con medida (estado en el horizonte).
    Devuelve 1/0 (alterado o no) o NaN si no hay medida.
    """
    m = medidas_por_visita(medidas, cfg, cohorte, [variable])
    m = m[m["visita_c"].isin(visitas) & m["subject_id"].isin(sujetos)]
    clave, direccion = cfg["estados"][variable]
    u = cfg["umbrales_clinicos"][clave]
    m = m.assign(alt=((m["valor"] < u) if direccion == "<" else (m["valor"] >= u)).astype(float))
    if regla == "alguna":
        return m.groupby("subject_id")["alt"].max().reindex(sujetos)
    orden = {v: i for i, v in enumerate(visitas)}
    m = m.assign(o=m["visita_c"].map(orden)).sort_values("o", ascending=False)
    return m.groupby("subject_id")["alt"].first().reindex(sujetos)
