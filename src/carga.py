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

# Medidas de DLCO (% predicho) por registro y visita: (registro, visita, columna DLCO,
# columna de fecha de la prueba, mes nominal). Verificado contra el diccionario.
FUENTES_DLCO: list[tuple[str, str, str, str | None, int]] = [
    ("CIBERESUCICOVID", "M3", "M3_followup_dlco", "M3_followup_spirometry_date", 3),
    ("CIBERESUCICOVID", "M6", "M6_followup_dlco", "M6_followup_spirometry_date", 6),
    ("CIBERESUCICOVID", "A1", "A1_followup_dlco_3", "A1_followup_spirometry_date_3", 12),
    ("TENACITY", "M3", "M3_dlco", "M3_fecha_pfr", 3),
    ("TENACITY", "M6", "M6_dlco", "M6_fecha_pfr", 6),
    ("TENACITY", "A1", "A1_dlco", "A1_fecha_pfr", 12),
    ("POSTCOVID_LLEIDA", "LV1", "LV1_dlco_porcentage", "LV1_fecha_pfr", 3),
    ("POSTCOVID_LLEIDA", "LV2", "LV2_dlco_porcentage", "LV2_fecha_pfr", 6),
    ("POSTCOVID_LLEIDA", "LA1", "LA1_dlco_porcentage", "LA1_fecha_pfr", 12),
    ("POSTCOVID_LLEIDA", "LM18", "LM18_dlco_porcentage", "LM18_fecha_pfr", 18),
    ("POSTCOVID_LLEIDA", "LM24", "LM24_dlco_porcentage", "LM24_fecha_pfr", 24),
    ("POSTCOVID_LLEIDA", "LA3", "LA3_dlco_porcentage", "LA3_fecha_pfr", 36),
    ("POSTCOVID_LLEIDA", "LA4", "LA4_dlco_porcentage", "LA4_fecha_pfr", 48),
    ("VIRGEN_DEL_ROCIO", "V1", "DLCO_porcentaje", "Fecha_DLCO", 1),
    ("VIRGEN_DEL_ROCIO", "V2", "DLCO_porc2", None, 6),
]


def cargar_config(ruta: str | Path = RAIZ / "config.yaml") -> dict:
    """Lee `config.yaml`."""
    with open(ruta, encoding="utf-8") as f:
        return yaml.safe_load(f)


def ruta_datos(cfg: dict, clave: str) -> Path:
    """Ruta absoluta a un fichero de datos declarado en `cfg['rutas']`."""
    return (RAIZ / cfg["rutas"]["datos"] / cfg["rutas"][clave]).resolve()


def cargar_nucleo(cfg: dict) -> pd.DataFrame:
    """Tabla núcleo con fechas parseadas y código 9 convertido en ausente."""
    df = pd.read_csv(ruta_datos(cfg, "nucleo"), low_memory=False)
    for c in ["nu_fecha_ingreso", "nu_fecha_alta", "nu_fecha_visita1"]:
        df[c] = pd.to_datetime(df[c], errors="coerce")
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


def dlco_larga(cfg: dict, nucleo: pd.DataFrame) -> pd.DataFrame:
    """Tabla larga de DLCO: una fila por medida (paciente, registro, visita).

    `dias_alta` = fecha de la prueba - fecha de alta (las fechas están desplazadas por
    paciente, pero el intervalo es exacto). Si falta la fecha de la prueba se usa
    `visita_dias` en la primera visita de cada registro; si no, queda NaN y se
    conserva el mes nominal.
    """
    columnas = [c for _, _, v, f, _ in FUENTES_DLCO for c in (v, f) if c]
    completa = cargar_columnas_completa(cfg, columnas)
    base = nucleo[["subject_id", "cohorte", "nu_fecha_alta", "visita_dias", "fusionada_lleida",
                   "enriquecida_vrocio", *cfg["registros"].values()]]
    completa = completa.merge(base, on="subject_id", how="left")

    primera_visita = {}
    partes = []
    for registro, visita, col_v, col_f, mes in FUENTES_DLCO:
        sub = completa[completa[col_v].notna()]
        fecha = pd.to_datetime(sub[col_f], errors="coerce", format="mixed") if col_f else pd.Series(pd.NaT, index=sub.index)
        dias = (fecha - sub["nu_fecha_alta"]).dt.days
        if registro not in primera_visita:
            primera_visita[registro] = visita
            dias = dias.fillna(sub["visita_dias"])
        partes.append(pd.DataFrame({
            "subject_id": sub["subject_id"],
            "registro": registro,
            "visita": visita,
            "mes_nominal": mes,
            "dlco": sub[col_v].astype(float),
            "dias_alta": dias,
            "fecha_imputada_visita_dias": fecha.isna().values & dias.notna().values,
            "fusionada_lleida": sub["fusionada_lleida"],
        }))
    larga = pd.concat(partes, ignore_index=True)
    larga["registro"] = pd.Categorical(larga["registro"], categories=list(cfg["registros"]), ordered=True)
    larga["meses_alta"] = larga["dias_alta"] / 30.44
    return larga
