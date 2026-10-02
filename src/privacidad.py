"""Utilidades de salida segura: solo agregados y supresión de celdas pequeñas."""
from __future__ import annotations

import pandas as pd


def suprimir_celdas(tabla: pd.DataFrame, n_min: int, columnas_n: list[str] | None = None) -> pd.DataFrame:
    """Sustituye por '<n_min' los recuentos menores que `n_min`.

    Si se indican `columnas_n`, solo se revisan esas columnas; en otro caso,
    todas las columnas enteras. Devuelve una copia (tipo objeto en las columnas tocadas).
    """
    salida = tabla.copy()
    cols = columnas_n or [c for c in salida.columns if pd.api.types.is_integer_dtype(salida[c])]
    for c in cols:
        pequenas = salida[c].notna() & (salida[c] < n_min) & (salida[c] > 0)
        if pequenas.any():
            salida[c] = salida[c].astype(object)
            salida.loc[pequenas, c] = f"<{n_min}"
    return salida


def resumen_seguro(serie: pd.Series, n_min: int) -> dict:
    """Resumen descriptivo de una variable numérica, vacío si N < n_min."""
    s = serie.dropna()
    if len(s) < n_min:
        return {"N": f"<{n_min}"}
    return {
        "N": len(s),
        "media": round(s.mean(), 1),
        "DE": round(s.std(), 1),
        "mediana": round(s.median(), 1),
        "P25": round(s.quantile(0.25), 1),
        "P75": round(s.quantile(0.75), 1),
    }


def proporcion_segura(numerador: int, denominador: int, n_min: int) -> str:
    """Texto 'x/N (p %)' o suprimido si el numerador o el denominador son pequeños."""
    if denominador < n_min:
        return f"N<{n_min}"
    if 0 < numerador < n_min:
        return f"<{n_min}/{denominador}"
    return f"{numerador}/{denominador} ({100 * numerador / denominador:.1f} %)"
