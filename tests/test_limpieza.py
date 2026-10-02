"""Tests de las reglas de limpieza con datos sintéticos mínimos (sin datos reales)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import carga, limpieza  # noqa: E402


@pytest.fixture
def cfg() -> dict:
    return carga.cargar_config()


def _pacientes(**extra) -> pd.DataFrame:
    """Tres pacientes: uno de CIBERESUCICOVID, uno fusionado con Lleida y uno de TENACITY."""
    base = pd.DataFrame({
        "subject_id": ["A", "B", "C"],
        "cohorte": ["CIBERESUCICOVID", "CIBERESUCICOVID", "TENACITY"],
        "fusionada_lleida": [0, 1, 0],
        "enriquecida_vrocio": [0, 0, 0],
        "en_TENACITY": [0, 0, 1],
        "nu_fecha_alta": pd.to_datetime(["2020-05-01", "2020-05-01", "2026-05-01"]),
        "visita_dias": [90, -10, 100],
        "nu_exitus_hosp": [0, 0, np.nan],
        "IH_PorcCMD": [30, 100, np.nan],
    })
    for k, v in extra.items():
        base[k] = v
    return base


def _medida(sid, registro, variable, valor, dias, mes=3, visita="M3", imputada=False):
    return {"subject_id": sid, "registro": registro, "visita": visita, "mes_nominal": mes, "variable": variable,
            "valor": valor, "dias_alta": dias, "fecha_imputada": imputada, "fuente_columna": "x"}


def test_codigo_9_y_sexo_a_nan(cfg):
    nucleo = _pacientes(nu_tabaquismo=[9, 1, 0], sexo=[2, 0, 1], nu_sexo=[2, 0, 1])
    for c in carga.VARIABLES_CODIGO_9:
        if c not in nucleo:
            nucleo[c] = 0
    df, entradas = limpieza.recodificar_pacientes(nucleo, cfg)
    assert np.isnan(df.loc[0, "nu_tabaquismo"]) and df.loc[1, "nu_tabaquismo"] == 1
    assert np.isnan(df.loc[0, "sexo"])
    assert any(e["N"] == 1 and "nu_tabaquismo" in e["descripcion"] for e in entradas)


def test_cohorte_analisis_y_banderas(cfg):
    df, _ = limpieza.asignar_cohorte_analisis(_pacientes(), cfg)
    assert list(df["cohorte_analisis"].astype(str)) == ["CIBERESUCICOVID", "POSTCOVID_LLEIDA", "TENACITY"]
    df, _ = limpieza.banderas_pacientes(df, cfg)
    assert np.isnan(df.loc[1, "visita_dias_valida"])          # visita_dias < 0
    assert not df.loc[0, "cmd_ok"]                            # CIBERESUCICOVID con CMD 30
    assert df.loc[1, "cmd_ok"] and df.loc[2, "cmd_ok"]        # el criterio no aplica fuera de CIBERESUCICOVID


def test_medidas_negativas_rango_y_duplicado(cfg):
    pac, _ = limpieza.asignar_cohorte_analisis(_pacientes(), cfg)
    med = pd.DataFrame([
        _medida("A", "CIBERESUCICOVID", "dlco", 70, -5),          # fecha anterior al alta
        _medida("A", "CIBERESUCICOVID", "dlco", 300, 90),         # fuera de rango
        _medida("B", "CIBERESUCICOVID", "dlco", 65, 95),          # duplicado de la de Lleida (Δ 5 días)
        _medida("B", "POSTCOVID_LLEIDA", "dlco", 65, 100, visita="LV1"),
        _medida("B", "CIBERESUCICOVID", "dlco", 75, 300, mes=12, visita="A1"),  # sin pareja en Lleida: se queda
    ])
    m, _ = limpieza.limpiar_medidas(med, pac, cfg)
    assert list(m["motivo_exclusion"].fillna("ok")) == [
        "dias_negativos", "fuera_de_rango", "duplicado_entre_registros", "ok", "ok"]


def test_tac_incompleto_no_cuenta(cfg):
    pac = _pacientes()
    completa = pd.DataFrame({
        "subject_id": ["C"],
        "M3_tacimagen_complete": [0], "M3_fibetic_reticular_lesions": [np.nan], "M3_fecha_tac": ["2026-08-01"],
        "M6_tacimagen_complete": [2], "M6_fibetic_reticular_lesions": [2], "M6_fecha_tac": ["2026-11-01"],
    })
    hallazgos = cfg["imagen"]["tac_ciberes"]["hallazgos"].values()
    fuentes = [("fib", *f) for fs in cfg["imagen"]["fibrosis"].values() for f in fs] + \
              [("tac", *f) for f in cfg["imagen"]["tac_ciberes"]["visitas"]]
    for tipo, _, bandera, var, fecha, _ in fuentes:
        vars_ = [f"{var}___{c}" for c in hallazgos] if tipo == "tac" else [var]
        for c in [bandera, *vars_, *carga.columnas_fecha(fecha)]:
            if c not in completa:
                completa[c] = np.nan
    img, _ = limpieza.medidas_imagen(cfg, pac, completa)
    assert set(img["visita"]) == {"M6"}                        # el formulario incompleto de M3 no entra
    fila = img.set_index("variable")["valor"]
    assert fila["fibrosis_estricta"] == 0 and fila["fibrosis_amplia"] == 1   # reticular = solo amplia


def test_tenacity_no_le_toca(cfg):
    pac = _pacientes()
    completa = pd.DataFrame({
        "subject_id": ["C"],
        "M3_fecha_visita": ["2026-08-10"], "M3_perdida_seg": [0],
        "M6_fecha_visita": [np.nan], "M6_perdida_seg": [np.nan],
        "A1_fecha_visita": [np.nan], "A1_perdida_seg": [np.nan],
    })
    vis, _ = limpieza.visitas_tenacity(cfg, pac, completa)
    estados = vis.set_index("visita")["estado"]
    assert estados["M3"] == "realizada"
    assert estados["A1"] == "no_le_toca"                      # alta 2026-05 + 12 meses > corte 2026-08


def test_resolucion_desconocida_excluida(cfg):
    pac, _ = limpieza.asignar_cohorte_analisis(_pacientes(), cfg)
    med = pd.DataFrame([_medida("A", "CIBERESUCICOVID", "resolucion", 3, 90),
                        _medida("A", "CIBERESUCICOVID", "resolucion", 1, 180, mes=6, visita="M6")])
    m, _ = limpieza.limpiar_medidas(med, pac, cfg)
    assert list(m["excluida"]) == [True, False]


def test_fatiga_deducida_si_resolucion_total(cfg):
    pac, _ = limpieza.asignar_cohorte_analisis(_pacientes(), cfg)
    med = pd.DataFrame([
        _medida("A", "CIBERESUCICOVID", "resolucion", 0, 90),                       # total, sin fatiga -> deducir 0
        _medida("A", "CIBERESUCICOVID", "resolucion", 1, 180, mes=6, visita="M6"),  # parcial -> no deducir
        _medida("B", "CIBERESUCICOVID", "resolucion", 0, 95),
        _medida("B", "CIBERESUCICOVID", "fatiga", 1, 95),                           # ya existe: no se toca
        _medida("C", "CIBERESUCICOVID", "resolucion", 3, 90),                       # se desconoce -> no deducir
    ])
    m, _ = limpieza.limpiar_medidas(med, pac, cfg)
    m, entradas = limpieza.deducir_condicionadas(m, cfg)
    ded = m[m["deducida"]]
    assert list(zip(ded["subject_id"], ded["visita"], ded["valor"])) == [("A", "M3", 0.0)]
    assert m[(m["subject_id"] == "B") & (m["variable"] == "fatiga")]["valor"].tolist() == [1]
    assert entradas[0]["N"] == 1


def test_estado_definicion_estructural(cfg):
    pac, _ = limpieza.asignar_cohorte_analisis(_pacientes(), cfg)
    med = pd.DataFrame([
        _medida("A", "CIBERESUCICOVID", "dlco", 70, 90),
        _medida("A", "CIBERESUCICOVID", "dlco", 90, 150),          # segunda en ventana: se usa la primera
        _medida("C", "TENACITY", "hads_a", 10, 100),
    ]).assign(excluida=False)
    est = limpieza.estado_definicion(med, pac, cfg).set_index("subject_id")
    assert est.loc["A", "dlco"] == 70 and bool(est.loc["A", "dlco_alterada"])
    assert est.loc["A", "hads_a_disp"] == "no_recogida"        # CIBERESUCICOVID no recoge HADS
    assert est.loc["C", "dlco_disp"] == "falta"                # TENACITY sí recoge DLCO
    assert bool(est.loc["C", "hads_a_alterada"])
