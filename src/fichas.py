"""Tabla de diferencias entre fenotipos: perfil, tamaño del efecto e interpretación (solo agregados).

Cada fila es una variable y cada columna un fenotipo. Las variables continuas se resumen con
mediana [P25–P75] y las binarias con %. La diferencia se mide con la diferencia estandarizada de
medias (SMD) entre el par de fenotipos más separado. Es independiente del N, así que sirve para
comparar cohortes de tamaños muy distintos. La p (Kruskal-Wallis o χ², ajustada por Benjamini-
Hochberg) solo se calcula para las variables que describen: las que definen el fenotipo difieren
por construcción.

Privacidad: un fenotipo con N < n_min no se describe. Si un % tiene entre 1 y n_min − 1 casos, se
muestra solo la cota «< x %» (x = n_min / N) y la comparación usa esa cota, sin afirmar nunca que
ese fenotipo tenga más que otro.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


def etiqueta(v: str, cfg: dict) -> str:
    """Nombre legible de una variable (`fichas.etiquetas` de config.yaml)."""
    return cfg["fichas"]["etiquetas"].get(v, v)


def etiqueta_estado(v: str, cfg: dict) -> str:
    """Nombre del estado alterado con su umbral clínico, p. ej. «DLCO < 80 %» o «Disnea mMRC ≥ 2»."""
    clave, direccion = cfg["estados"][v]
    if clave == "presente":
        return etiqueta(v, cfg)
    unidad = " %" if clave.endswith("_pct") else " m" if clave.endswith("_metros") else ""
    return f"{etiqueta(v, cfg)} {'<' if direccion == '<' else '≥'} {cfg['umbrales_clinicos'][clave]}{unidad}"


def alterado(serie: pd.Series, v: str, cfg: dict) -> pd.Series:
    """1/0 según el umbral clínico de `v` (NaN se mantiene)."""
    clave, direccion = cfg["estados"][v]
    u = cfg["umbrales_clinicos"][clave]
    return ((serie < u) if direccion == "<" else (serie >= u)).astype(float).where(serie.notna())


def variables_definicion(desc: pd.DataFrame, columnas: list[str], cfg: dict,
                         bloque: str = "Define el fenotipo") -> tuple[pd.DataFrame, list[dict]]:
    """Variables de definición: valor continuo y % alterado (umbral clínico) de cada `v_VISITA`.

    Las variables 0/1 u ordinales con umbral (fatiga, resolución) solo dan el % alterado.
    """
    datos, info = {}, []
    for c in columnas:
        v, vis = c.rsplit("_", 1)
        if vis == "cambio":
            datos[c] = desc[c]
            info.append({"col": c, "bloque": bloque, "variable": f"Cambio de {etiqueta(v, cfg)} (puntos)", "binaria": False})
            continue
        if cfg["estados"][v][0] != "presente" and desc[c].dropna().nunique() > 3:
            datos[c] = desc[c]
            info.append({"col": c, "bloque": bloque, "variable": f"{etiqueta(v, cfg)} {vis}", "binaria": False, "base": c})
        datos[f"{c}_alt"] = alterado(desc[c], v, cfg)
        info.append({"col": f"{c}_alt", "bloque": bloque, "variable": f"{etiqueta_estado(v, cfg)} ({vis})", "binaria": True,
                     "base": c})   # en el retrato, solo la de las dos filas de c que más separa
    return pd.DataFrame(datos, index=desc.index), info


def variables_ingreso(pac: pd.DataFrame, binarias: list[str], cfg: dict,
                      bloque: str = "Describe · ingreso") -> tuple[pd.DataFrame, list[dict]]:
    """Edad ≥ 65, sexo, estancia y variables 0/1 del ingreso (`pac` indexado por paciente)."""
    fc = cfg["fichas"]
    datos = {
        "edad_mayor": pac["grupo_edad"].isin(fc["edad_mayor"]).astype(float).where(pac["grupo_edad"].notna()),
        "mujer": (pac["sexo"] == fc["codigo_mujer"]).astype(float).where(pac["sexo"].notna()),
        "estancia_hosp_dias": pac["estancia_hosp_dias"],
    }
    info = [
        {"col": "edad_mayor", "bloque": bloque, "variable": f"Edad ≥ {fc['edad_mayor'][0][:2]} años", "binaria": True},
        {"col": "mujer", "bloque": bloque, "variable": "Mujer", "binaria": True},
        {"col": "estancia_hosp_dias", "bloque": bloque, "variable": "Estancia hospitalaria (días)", "binaria": False},
    ]
    for v in binarias:
        datos[v] = pac[v]
        info.append({"col": v, "bloque": bloque, "variable": etiqueta(v, cfg), "binaria": True})
    return pd.DataFrame(datos, index=pac.index), info


def magnitud(d: float, cfg: dict) -> str:
    """Etiqueta de Cohen para |SMD|."""
    cortes, nombres = cfg["fichas"]["cortes_smd"], cfg["fichas"]["etiquetas_smd"]
    return nombres[int(np.searchsorted(cortes, abs(d), side="right"))]


def resumen_grupo(s: pd.Series, binaria: bool, n_min: int) -> dict | None:
    """Resumen de un fenotipo para una variable: texto de la celda, valor corto, media y varianza.

    None si hay menos de `n_min` pacientes con dato. Si una binaria tiene entre 1 y n_min − 1 casos,
    se muestra solo la cota «< x %» y la comparación usa esa cota (`cota=True`).
    """
    s = s.dropna()
    n = len(s)
    if n < n_min:
        return None
    if binaria:
        k = int(s.sum())
        cota = 0 < k < n_min
        prop = n_min / n if cota else k / n
        texto = f"{'<' if cota else ''}{100 * prop:.0f} %"
        return {"texto": texto, "corto": texto, "media": prop, "var": prop * (1 - prop), "cota": cota}
    q1, med, q3 = s.quantile([0.25, 0.5, 0.75])
    dec = 1 if q3 - q1 < 10 else 0
    return {"texto": f"{med:.{dec}f} [{q1:.{dec}f}–{q3:.{dec}f}]", "corto": f"{med:.{dec}f}",
            "media": float(s.mean()), "var": float(s.var(ddof=1)), "cota": False}


def smd(alto: dict, bajo: dict) -> float:
    """SMD (media alto − media bajo) / DE combinada, a partir de dos resúmenes de grupo.

    Separación total (varianzas 0 y medias distintas) = ∞. Si el grupo alto es una cota (< x %), su
    valor real puede ser menor y la diferencia no se puede afirmar: NaN.
    """
    if alto["cota"]:
        return np.nan
    dif = alto["media"] - bajo["media"]
    de = np.sqrt((alto["var"] + bajo["var"]) / 2)
    return float(dif / de) if de > 0 else (np.inf if dif > 0 else 0.0)


def _interpretar(g: dict, binaria: bool, d: float, cfg: dict) -> str:
    """Frase de lectura de una fila.

    Un fenotipo destaca si se separa del siguiente con al menos una diferencia pequeña; si ninguno
    destaca, se describe el gradiente.
    """
    if magnitud(d, cfg) == cfg["fichas"]["etiquetas_smd"][0]:
        return "Sin diferencias relevantes"
    orden = sorted(g, key=lambda f: -g[f]["media"])
    sube, baja = ("Más frecuente", "Menos frecuente") if binaria else ("Más alta", "Más baja")
    corte = cfg["fichas"]["cortes_smd"][0]
    for f, texto, otros, d_vecino in [(orden[0], sube, orden[1:], smd(g[orden[0]], g[orden[1]])),
                                      (orden[-1], baja, orden[:-1], smd(g[orden[-2]], g[orden[-1]]))]:
        if d_vecino >= corte:
            resto = " y ".join(f"{o} ({g[o]['corto']})" for o in sorted(otros))
            return f"{texto} en {f} ({g[f]['corto']}) que en {resto}"
    return "Gradiente: " + " > ".join(f"{f} ({g[f]['corto']})" for f in orden)


def tabla_diferencias(datos: pd.DataFrame, info: list[dict], fenotipo: pd.Series, cfg: dict,
                      cohorte: pd.Series | None = None, nombres_cohorte: dict | None = None
                      ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tabla de perfil y diferencias entre fenotipos, y un retrato de cada fenotipo frente al resto.

    `datos`: una columna por variable (índice = paciente). `info`: lista de dicts con `col`, `bloque`,
    `variable` y `binaria` (y `base`: filas con la misma base cuentan una sola vez en el retrato), en el
    orden de la tabla; las variables con bloque que empieza por «Define» no llevan p. `fenotipo`: etiqueta de cada
    paciente. `cohorte` (opcional): añade la columna con las cohortes que aportan dato a cada variable.
    """
    fc = cfg["fichas"]
    n_min = cfg["privacidad"]["n_minimo_celda"]
    fenotipo = fenotipo.reindex(datos.index)
    n_fen = fenotipo.value_counts()
    fens = sorted(n_fen.index)
    validos = [f for f in fens if n_fen[f] >= n_min]
    cab = {f: f"{f} (N={n_fen[f]})" if f in validos else f"{f} (N<{n_min})" for f in fens}

    filas, rasgos = [], {(f, define): [] for f in validos for define in (True, False)}
    for it in info:
        s = datos[it["col"]]
        define = it["bloque"].startswith("Define")
        g = {f: resumen_grupo(s[fenotipo == f], it["binaria"], n_min) for f in validos}
        g = {f: r for f, r in g.items() if r is not None}
        fila = {"bloque": it["bloque"], "variable": it["variable"],
                **{cab[f]: g[f]["texto"] if f in g else "—" for f in fens}}
        n = int(s.notna().sum())
        fila["N con dato"] = n if n >= n_min else f"<{n_min}"
        if cohorte is not None:
            por_coh = s.notna().groupby(cohorte.reindex(datos.index)).sum()
            fila["Cohortes con dato"] = ", ".join((nombres_cohorte or {}).get(c, c) for c in por_coh.index
                                                  if por_coh[c] >= n_min)

        # Mayor SMD entre pares (el grupo con media más alta primero; un par con cota arriba no cuenta)
        pares = [smd(g[a], g[b]) if g[a]["media"] >= g[b]["media"] else smd(g[b], g[a]) for a in g for b in g if a < b]
        pares = [x for x in pares if not np.isnan(x)]
        d = max(pares) if pares else np.nan
        texto = "Datos insuficientes" if np.isnan(d) else _interpretar(g, it["binaria"], d, cfg)

        # Retrato: cada fenotipo frente al resto de pacientes juntos (uno contra el resto)
        minimo = fc["cortes_smd"][fc["etiquetas_smd"].index(fc["retrato_min"]) - 1]
        for f in g:
            resto = resumen_grupo(s[fenotipo != f], it["binaria"], n_min)
            if resto is None:
                continue
            d_f = smd(g[f], resto) if g[f]["media"] >= resto["media"] else -smd(resto, g[f])
            if abs(d_f) >= minimo:
                rasgos[(f, define)].append((abs(d_f), it.get("base", it["col"]),
                                            f"{'↑' if d_f > 0 else '↓'} {it['variable']}: {g[f]['corto']} vs {resto['corto']}"))

        fila["SMD máx."] = "—" if np.isnan(d) else "∞" if np.isinf(d) else f"{d:.2f}"
        fila["Magnitud"] = "—" if np.isnan(d) else magnitud(d, cfg)
        fila["_p"] = np.nan
        if not define and len(g) >= 2:
            dentro = s.notna() & fenotipo.isin(list(g))
            if it["binaria"]:
                tabla = pd.crosstab(fenotipo[dentro], s[dentro])
                fila["_p"] = stats.chi2_contingency(tabla)[1] if tabla.shape[1] == 2 else np.nan
            else:
                fila["_p"] = stats.kruskal(*[s[dentro & (fenotipo == f)] for f in g]).pvalue
        fila["Interpretación"] = texto
        filas.append(fila)

    tabla = pd.DataFrame(filas)
    tabla["p ajustada"] = np.where(tabla["bloque"].str.startswith("Define"), "— (define)", "—")
    hay_p = tabla["_p"].notna()
    if hay_p.any():
        p_adj = multipletests(tabla.loc[hay_p, "_p"], method="fdr_bh")[1]
        tabla.loc[hay_p, "p ajustada"] = [f"<{fc['p_min']}" if p < fc["p_min"] else f"{p:.3f}" for p in p_adj]
    orden = ["bloque", "variable", *cab.values(), "N con dato", *(["Cohortes con dato"] if cohorte is not None else []),
             "SMD máx.", "Magnitud", "p ajustada", "Interpretación"]
    tabla = tabla[orden].set_index(["bloque", "variable"])

    def resumir(r: list) -> str:
        vistos, textos = set(), []
        for _, base, t in sorted(r, key=lambda x: -x[0]):
            if base not in vistos:
                vistos.add(base); textos.append(t)
        return " · ".join(textos[:fc["max_rasgos_retrato"]]) or "—"

    retratos = pd.DataFrame({
        "Lo que lo define (frente al resto)":
            {cab[f]: resumir(rasgos[(f, True)]) for f in validos},
        "Lo que lo acompaña (frente al resto)":
            {cab[f]: resumir(rasgos[(f, False)]) for f in validos},
    })
    retratos.index.name = "Fenotipo"
    return tabla, retratos
