"""Fenotipado a partir de un perfil de `config.yaml` (notebook 04: las tres cohortes juntas).

Generaliza la construcción de variables del notebook 03 a un PERFIL leído de `config.yaml`
(p. ej., `fenotipado_comun`): dominios y variables de definición, cohortes de réplica, ancla,
desenlaces y rangos. Distancia, PAM, prueba nula y estabilidad se reutilizan de `src/fenotipado.py`
sin modificarlo (el notebook 03 sigue usando ese módulo tal cual).

- Visitas genéricas T3, T6, T12 con su código en cada cohorte; horizonte acumulado (H6 = T3 + T6…).
- Las columnas son las de la cohorte de descubrimiento; en otra cohorte, lo que no recoge queda
  vacío y Gower lo ignora (la transferencia usa solo lo compartido).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.fenotipado import (  # noqa: F401  (reexportadas para el notebook)
    asignar_medoides, elegir_k, estabilidad_bootstrap, estabilidad_centros, gower, ordenar_por_dlco,
    pam, porcentaje_alterado, seleccion_k_nulo, silueta,
)


def perfil(cfg: dict, clave: str) -> dict:
    """Perfil de fenotipado (`cfg[clave]`)."""
    return {**cfg[clave], "clave": clave}


def variables_definicion(pf: dict) -> list[str]:
    """Variables de definición del perfil, en orden de dominio."""
    return [v for vs in pf["dominios"].values() for v in vs]


def visitas_que_recogen(cfg: dict, pf: dict, variable: str, cohorte: str) -> set[str]:
    """Visitas genéricas (T3, T6, T12) en las que `cohorte` recoge la variable (según `variables_visita`)."""
    codigos = {f[0] for f in cfg["variables_visita"].get(variable, {}).get(cohorte, [])}
    return {t for t, d in pf["visitas"].items() if d.get(cohorte) in codigos}


def medidas_por_visita(medidas: pd.DataFrame, pf: dict, cohorte: str, variables: list[str]) -> pd.DataFrame:
    """Medidas válidas de `cohorte` etiquetadas con la visita genérica (`visita_c`).

    Se excluyen las que tienen fecha fuera de la ventana de tolerancia de su visita.
    """
    vis = pf["visitas"]
    mapa = {d[cohorte]: t for t, d in vis.items() if cohorte in d}
    m = medidas[~medidas["excluida"] & (medidas["registro"].astype(str) == cohorte)
                & medidas["variable"].isin(variables) & medidas["visita"].isin(mapa)].copy()
    m["visita_c"] = m["visita"].map(mapa)
    lo = m["visita_c"].map(lambda t: vis[t]["ventana"][0])
    hi = m["visita_c"].map(lambda t: vis[t]["ventana"][1])
    return m[m["dias_alta"].isna() | m["dias_alta"].between(lo, hi)]


def columnas_definicion(cfg: dict, pf: dict, horizonte: str) -> list[str]:
    """Columnas de definición: `v_T` si la cohorte de descubrimiento recoge v en T, y `v_cambio`."""
    visitas = pf["horizontes"][horizonte]
    columnas = []
    for v in variables_definicion(pf):
        columnas += [f"{v}_{t}" for t in visitas if t in visitas_que_recogen(cfg, pf, v, pf["descubre"])]
        if v in pf["rango_cambio"] and len(visitas) > 1:
            columnas.append(f"{v}_cambio")
    return columnas


def construir_variables(medidas: pd.DataFrame, pacientes: pd.DataFrame, cfg: dict, pf: dict,
                        horizonte: str, cohorte: str) -> tuple[pd.DataFrame, list[str]]:
    """Pacientes de `cohorte` incluidos en el horizonte, con las columnas de definición del perfil.

    `v_T`: primera medida válida de v en T (≤ horizonte); `v_cambio`: último − primero (≥ 2 medidas).
    Inclusión: superviviente, sin inconsistencia, cmd_ok y ancla en la última visita (un valor de cada
    grupo del ancla, si la cohorte recoge alguna variable del grupo en esa visita).
    """
    visitas = pf["horizontes"][horizonte]
    variables = variables_definicion(pf)
    columnas = columnas_definicion(cfg, pf, horizonte)
    m = medidas_por_visita(medidas, pf, cohorte, variables)
    m = m[m["visita_c"].isin(visitas)]
    primera = m.sort_values("dias_alta").groupby(["subject_id", "variable", "visita_c"]).first().reset_index()
    valores = primera.pivot_table(index="subject_id", columns=["variable", "visita_c"], values="valor", aggfunc="first")

    tabla = pd.DataFrame(index=valores.index)
    for c in columnas:
        v, t = c.rsplit("_", 1)
        if t != "cambio":
            tabla[c] = valores[(v, t)] if (v, t) in valores.columns else np.nan
    for v in variables:
        if f"{v}_cambio" in columnas:
            serie = tabla[[f"{v}_{t}" for t in visitas if f"{v}_{t}" in tabla]]
            tabla[f"{v}_cambio"] = (serie.ffill(axis=1).iloc[:, -1] - serie.bfill(axis=1).iloc[:, 0]) \
                .where(serie.notna().sum(axis=1) >= 2)

    ultima = visitas[-1]
    ancla = pd.Series(True, index=tabla.index)
    for grupo in pf["ancla"]:
        recogidas = [v for v in grupo if ultima in visitas_que_recogen(cfg, pf, v, cohorte) and f"{v}_{ultima}" in tabla]
        if recogidas:
            ancla &= tabla[[f"{v}_{ultima}" for v in recogidas]].notna().any(axis=1)

    base = pacientes.set_index("subject_id")
    base = base[base["superviviente"] & ~base["inconsistente"] & base["cmd_ok"] & (base["cohorte_analisis"] == cohorte)]
    tabla = tabla[ancla & tabla.index.isin(base.index)]
    tabla["dias_max"] = m[m["subject_id"].isin(tabla.index)].groupby("subject_id")["dias_alta"].max()
    tabla = tabla.join(base[["cohorte_analisis", "centro_id"]])
    return tabla, columnas


def compartidas(cfg: dict, pf: dict, columnas: list[str], cohorte: str) -> np.ndarray:
    """Máscara de columnas que `cohorte` también recoge (para transferir y medir el techo de comparación)."""
    mascara = []
    for c in columnas:
        v, t = c.rsplit("_", 1)
        recoge = visitas_que_recogen(cfg, pf, v, cohorte)
        mascara.append(bool(recoge) if t == "cambio" else t in recoge)
    return np.array(mascara)


def parametros_gower(columnas: list[str], pf: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rango, peso (igual por dominio, repartido entre sus columnas) y bandera categórica por columna."""
    dominio_de = {v: d for d, vs in pf["dominios"].items() for v in vs}
    rangos, dominios, categ = [], [], []
    for c in columnas:
        v, sufijo = c.rsplit("_", 1)
        lo, hi = pf["rango_cambio"][v] if sufijo == "cambio" else pf["rango_gower"][v]
        rangos.append(hi - lo); dominios.append(dominio_de[v]); categ.append(v in pf["categoricas"])
    dominios = np.array(dominios)
    n_dom = len(set(dominios))
    pesos = np.array([1.0 / n_dom / (dominios == d).sum() for d in dominios])
    return np.array(rangos, float), pesos, np.array(categ)


def estado_hasta(medidas: pd.DataFrame, cfg: dict, pf: dict, cohorte: str, sujetos: pd.Index,
                 variable: str, visitas: list[str], regla: str = "alguna") -> pd.Series:
    """Estado de una variable descriptiva en las visitas genéricas indicadas.

    regla "alguna": 1 si alterada en alguna visita (eventos, imagen); "ultima": estado en la visita más
    reciente con medida. Devuelve 1/0 o NaN si no hay medida.
    """
    m = medidas_por_visita(medidas, pf, cohorte, [variable])
    m = m[m["visita_c"].isin(visitas) & m["subject_id"].isin(sujetos)]
    clave, direccion = cfg["estados"][variable]
    u = cfg["umbrales_clinicos"][clave]
    m = m.assign(alt=((m["valor"] < u) if direccion == "<" else (m["valor"] >= u)).astype(float))
    if regla == "alguna":
        return m.groupby("subject_id")["alt"].max().reindex(sujetos)
    orden = {v: i for i, v in enumerate(visitas)}
    m = m.assign(o=m["visita_c"].map(orden)).sort_values("o", ascending=False)
    return m.groupby("subject_id")["alt"].first().reindex(sujetos)
