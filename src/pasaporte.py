"""Pasaporte de fenotipo: replicación entre cohortes, emparejamiento y transiciones."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score

from src import fenotipado as F


def matriz_jaccard(a: np.ndarray, b: np.ndarray, k: int) -> np.ndarray:
    """Jaccard entre cada clúster i de `a` y cada clúster j de `b` (mismos pacientes)."""
    J = np.zeros((k, k))
    for i in range(k):
        A = a == i
        for j in range(k):
            B = b == j
            union = (A | B).sum()
            J[i, j] = (A & B).sum() / union if union else 0.0
    return J


def emparejar_hungaro(a: np.ndarray, b: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Empareja los clústeres de `b` con los de `a` maximizando el Jaccard total.

    Devuelve (mapa: clúster de b -> clúster de a, Jaccard de cada par en el orden de a).
    """
    J = matriz_jaccard(a, b, k)
    fila, col = linear_sum_assignment(-J)
    mapa = np.empty(k, int); mapa[col] = fila
    jacc = np.empty(k); jacc[fila] = J[fila, col]
    return mapa, jacc


def replicar(X_desc: np.ndarray, medoides: np.ndarray, X_rep: np.ndarray, rangos: np.ndarray,
             pesos: np.ndarray, k: int, n_boot: int, semilla: int) -> dict:
    """Replicación de un clustering en otra cohorte.

    - Transferidas: cada paciente de la réplica al medoide más cercano del descubrimiento.
    - Nativas: PAM en la réplica con el mismo k, emparejadas con el algoritmo húngaro.
    - Concordancia: ARI global y Jaccard por par, con IC bootstrap (reagrupando la réplica).
    """
    D_rep = F.gower(X_rep, None, rangos, pesos)
    D_med = F.gower(X_rep, X_desc[medoides], rangos, pesos)
    transferidas = F.asignar_medoides(D_med)
    nativas_crudas, _ = F.pam(D_rep, k, semilla)
    mapa, jacc = emparejar_hungaro(transferidas, nativas_crudas, k)
    nativas = mapa[nativas_crudas]
    ari = adjusted_rand_score(transferidas, nativas)

    rng = np.random.default_rng(semilla)
    n = len(X_rep)
    aris, jaccs = [], []
    for b in range(n_boot):
        idx = np.unique(rng.integers(0, n, n))
        nat_b, _ = F.pam(D_rep[np.ix_(idx, idx)], k, semilla + b)
        m_b, j_b = emparejar_hungaro(transferidas[idx], nat_b, k)
        aris.append(adjusted_rand_score(transferidas[idx], m_b[nat_b]))
        jaccs.append(j_b)
    jaccs = np.array(jaccs)
    return {
        "transferidas": transferidas, "nativas": nativas, "n": n,
        "ari": ari, "ari_ic": tuple(np.percentile(aris, [2.5, 97.5])),
        "jaccard": jacc, "jaccard_ic": np.percentile(jaccs, [2.5, 97.5], axis=0).T,
    }


def diferencias_perfil(X_desc: pd.DataFrame, lab_desc: np.ndarray, X_rep: pd.DataFrame,
                       lab_rep: np.ndarray, columnas: list[str], k: int) -> pd.DataFrame:
    """Diferencia estandarizada de medias (réplica − descubrimiento) por clúster y variable.

    |d| < 0,2 despreciable; 0,2–0,5 pequeña; > 0,5 relevante. Lo que debe replicarse es el perfil.
    """
    filas = {}
    for c in range(k):
        a, b = X_desc.loc[lab_desc == c, columnas], X_rep.loc[lab_rep == c, columnas]
        sd = np.sqrt((a.var() + b.var()) / 2).replace(0, np.nan)
        filas[c] = (b.mean() - a.mean()) / sd
    return pd.DataFrame(filas)


def transiciones(et_1: pd.Series, et_2: pd.Series) -> tuple[pd.DataFrame, float, int]:
    """Matriz de transición (filas: horizonte previo, % por fila) y ARI en los pacientes comunes."""
    comunes = et_1.index.intersection(et_2.index)
    if len(comunes) == 0:
        return pd.DataFrame(), np.nan, 0
    a, b = et_1.loc[comunes], et_2.loc[comunes]
    tabla = pd.crosstab(a, b)
    return tabla, adjusted_rand_score(a, b), len(comunes)
