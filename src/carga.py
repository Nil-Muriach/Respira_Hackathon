"""Lectura de datos, trazabilidad y tablas largas de seguimiento."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

RAIZ = Path(__file__).resolve().parents[1]

# Variables del núcleo con código 9 = "se desconoce"
VARIABLES_CODIGO_9 = [
    "nu_tabaquismo", "nu_hta", "nu_diabetes", "nu_card_cronica", "nu_pulm_cronica",
    "nu_epoc", "nu_asma", "nu_renal_cronica", "nu_ictus_previo", "nu_sahs",
    "nu_sdra", "nu_iot", "nu_traqueo", "nu_exitus_hosp",
]

# Las columnas de seguimiento por registro y visita están en config.yaml (`variables_visita`).


def cargar_config(ruta: str | Path = RAIZ / "config.yaml") -> dict:
    """Lee `config.yaml`."""
    with open(ruta, encoding="utf-8") as f:
        return yaml.safe_load(f)


def ruta_datos(cfg: dict, clave: str) -> Path:
    """Ruta absoluta a un fichero de datos declarado en `cfg['rutas']`."""
    return (RAIZ / cfg["rutas"]["datos"] / cfg["rutas"][clave]).resolve()


def cargar_nucleo(cfg: dict, recodificar: bool = True) -> pd.DataFrame:
    """Tabla núcleo con fechas parseadas y, si `recodificar`, código 9 convertido en ausente."""
    df = pd.read_csv(ruta_datos(cfg, "nucleo"), low_memory=False)
    for c in ["nu_fecha_ingreso", "nu_fecha_alta", "nu_fecha_visita1"]:
        df[c] = pd.to_datetime(df[c], errors="coerce")
    if not recodificar:
        return df
    return recodificar_desconocidos(df, VARIABLES_CODIGO_9, cfg["calidad"]["codigo_desconocido"])


def recodificar_desconocidos(df: pd.DataFrame, columnas: list[str], codigo: int) -> pd.DataFrame:
    """Sustituye `codigo` ("se desconoce") por NaN en las columnas indicadas."""
    df = df.copy()
    for c in columnas:
        if c in df:
            df[c] = df[c].where(df[c] != codigo)
    return df


def cargar_columnas_completa(cfg: dict, columnas: list[str]) -> pd.DataFrame:
    """Lee solo `columnas` (más `subject_id`) de la tabla completa."""
    cols = list(dict.fromkeys(["subject_id", *columnas]))
    return pd.read_csv(ruta_datos(cfg, "completa"), usecols=cols, low_memory=False)


def cargar_diccionario(cfg: dict) -> dict[str, pd.DataFrame]:
    """Hojas del diccionario: núcleo, todas las variables, cobertura por dominio y visitas."""
    hojas = pd.read_excel(ruta_datos(cfg, "diccionario"), sheet_name=None)
    nombres = {k.lower(): k for k in hojas}
    buscar = lambda patron: hojas[next(v for k, v in nombres.items() if patron in k)]
    return {
        "nucleo": buscar("cleo"),
        "todas": buscar("todas"),
        "cobertura": buscar("cobertura"),
        "visitas": buscar("visitas"),
    }


def pertenencia_larga(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Una fila por (paciente, registro) usando las banderas `en_<registro>`.

    Un paciente en dos registros aparece dos veces: es la vista correcta para
    describir cada registro completo, pero no para sumar entre registros.
    """
    partes = []
    for registro, bandera in cfg["registros"].items():
        sub = df[df[bandera] == 1].copy()
        sub["registro"] = registro
        partes.append(sub)
    salida = pd.concat(partes, ignore_index=True)
    salida["registro"] = pd.Categorical(salida["registro"], categories=list(cfg["registros"]), ordered=True)
    return salida


def limite_inferior_tramo(tramo: str | float) -> float:
    """Límite inferior numérico de un tramo de edad ('60-64' -> 60, '<50' -> 0, '70+' -> 70)."""
    if not isinstance(tramo, str):
        return np.nan
    if tramo.startswith("<"):
        return 0.0
    m = re.match(r"(\d+)", tramo)
    return float(m.group(1)) if m else np.nan


def edad_ordenada(serie: pd.Series) -> pd.Series:
    """`edad_tramo5` como categórica ordenada por el límite inferior del tramo (no alfabética)."""
    tramos = sorted(serie.dropna().unique(), key=lambda t: (limite_inferior_tramo(t), len(t)))
    return pd.Categorical(serie, categories=tramos, ordered=True)


def edad_agrupada(serie: pd.Series, grupos: dict[str, list[str]]) -> pd.Series:
    """Reagrupa los tramos de 5 años en grupos amplios; los tramos que cruzan grupos quedan NaN."""
    mapa = {tramo: grupo for grupo, tramos in grupos.items() for tramo in tramos}
    return pd.Categorical(serie.map(mapa), categories=list(grupos), ordered=True)


def fuentes_variable(cfg: dict, variable: str) -> list[tuple[str, str, str, list[str], int]]:
    """Fuentes de una variable de seguimiento: (registro, visita, columna, columnas_fecha, mes)."""
    return [(registro, visita, col, columnas_fecha(fecha), mes)
            for registro, filas in cfg["variables_visita"][variable].items()
            for visita, col, fecha, mes in filas]


def columnas_fecha(fecha: str | list[str] | None) -> list[str]:
    """Normaliza la columna de fecha de config (texto, lista de respaldo o nada) a una lista."""
    if fecha is None:
        return []
    return [fecha] if isinstance(fecha, str) else list(fecha)


def parsear_fecha(serie: pd.Series, formato: str) -> pd.Series:
    """Convierte texto a fecha con formato explícito; lo que no encaja queda NaT."""
    return pd.to_datetime(serie, format=formato, errors="coerce")


def fecha_coalescida(df: pd.DataFrame, columnas: list[str], formato: str) -> pd.Series:
    """Primera fecha disponible entre `columnas` (en orden de preferencia)."""
    fecha = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    for c in columnas:
        fecha = fecha.fillna(parsear_fecha(df[c], formato))
    return fecha


def medidas_larga(cfg: dict, nucleo: pd.DataFrame, variables: list[str] | None = None,
                  completa: pd.DataFrame | None = None) -> pd.DataFrame:
    """Tabla larga sin limpiar: una fila por (paciente, registro de origen, visita, variable).

    `dias_alta` = fecha de la prueba - fecha de alta (fechas desplazadas por paciente, pero
    el intervalo es exacto). Si falta la fecha, se usa `visita_dias` solo en la primera visita
    de cada registro y se marca `fecha_imputada`. No aplica ninguna regla de limpieza.
    """
    variables = variables or list(cfg["variables_visita"])
    fuentes = {v: fuentes_variable(cfg, v) for v in variables}
    if completa is None:
        columnas = sorted({c for fs in fuentes.values() for _, _, col, fechas, _ in fs for c in [col, *fechas]})
        completa = cargar_columnas_completa(cfg, columnas)
    base = nucleo[["subject_id", "nu_fecha_alta", "visita_dias"]]
    completa = completa.merge(base, on="subject_id", how="left")

    partes = []
    for variable, fs in fuentes.items():
        primera = {}
        for registro, visita, col, col_f, mes in fs:
            primera.setdefault(registro, visita)
            sub = completa[completa[col].notna()]
            fecha = fecha_coalescida(sub, col_f, cfg["formato_fecha"][registro])
            dias = (fecha - sub["nu_fecha_alta"]).dt.days
            imputada = fecha.isna()
            if primera[registro] == visita:
                dias = dias.fillna(sub["visita_dias"])
            partes.append(pd.DataFrame({
                "subject_id": sub["subject_id"].values,
                "registro": registro,
                "visita": visita,
                "mes_nominal": mes,
                "variable": variable,
                "valor": sub[col].astype(float).values,
                "dias_alta": dias.values,
                "fecha_imputada": (imputada & dias.notna()).values,
                "fuente_columna": col,
            }))
    larga = pd.concat(partes, ignore_index=True)
    larga["registro"] = pd.Categorical(larga["registro"], categories=list(cfg["registros"]), ordered=True)
    larga["meses_alta"] = larga["dias_alta"] / cfg["limpieza"]["dias_por_mes"]
    return larga


def dlco_larga(cfg: dict, nucleo: pd.DataFrame) -> pd.DataFrame:
    """Tabla larga de DLCO en el formato del notebook 01 (envoltorio de `medidas_larga`)."""
    larga = medidas_larga(cfg, nucleo, ["dlco"])
    larga = larga.merge(nucleo[["subject_id", "fusionada_lleida"]], on="subject_id", how="left")
    return larga.rename(columns={"valor": "dlco", "fecha_imputada": "fecha_imputada_visita_dias"})[
        ["subject_id", "registro", "visita", "mes_nominal", "dlco", "dias_alta",
         "fecha_imputada_visita_dias", "fusionada_lleida", "meses_alta"]]
