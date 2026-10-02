"""Tests del fenotipado por perfil (notebook 04, Lleida) con datos sintéticos."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import carga, fenotipado_perfil as FP  # noqa: E402


@pytest.fixture
def cfg() -> dict:
    return carga.cargar_config()


@pytest.fixture
def pf(cfg) -> dict:
    return FP.perfil(cfg, "fenotipado_lleida")


def _m(sid, reg, var, valor, dias, visita):
    return {"subject_id": sid, "registro": reg, "variable": var, "valor": valor, "dias_alta": dias,
            "visita": visita, "excluida": False}


def test_columnas_y_compartidas(cfg, pf):
    cols = FP.columnas_definicion(cfg, pf, "H6")
    assert "hads_a_T6" in cols and "pm6m_cambio" in cols and "dlco_T3" in cols
    m_cib = FP.compartidas(cfg, pf, cols, "CIBERESUCICOVID")      # CIBERESUCICOVID solo comparte DLCO/FVC
    assert {c.rsplit("_", 1)[0] for c, m in zip(cols, m_cib) if m} == {"dlco", "fvc"}
    assert FP.compartidas(cfg, pf, cols, "TENACITY").all()        # TENACITY lo comparte todo


def test_pesos_iguales_por_dominio(cfg, pf):
    cols = FP.columnas_definicion(cfg, pf, "H3")                  # 6 variables, 4 dominios
    _, pesos, cat = FP.parametros_gower(cols, pf)
    por_dom = {d: sum(p for c, p in zip(cols, pesos) if c.rsplit("_", 1)[0] in vs) for d, vs in pf["dominios"].items()}
    assert pesos.sum() == pytest.approx(1) and np.allclose(list(por_dom.values()), 0.25) and not cat.any()


def test_construccion_ancla_y_sin_fuga(cfg, pf):
    L = "POSTCOVID_LLEIDA"
    pac = pd.DataFrame({"subject_id": ["P", "Q", "C"], "superviviente": True, "inconsistente": False, "cmd_ok": True,
                        "cohorte_analisis": [L, L, "CIBERESUCICOVID"], "centro_id": "X"})
    med = pd.DataFrame([
        _m("P", L, "dlco", 60, 100, "LV1"), _m("P", L, "hads_a", 9, 100, "LV1"),
        _m("P", L, "dlco", 70, 200, "LV2"), _m("P", L, "mmrc", 1, 200, "LV2"),
        _m("P", L, "dlco", 95, 370, "LA1"),                                       # futuro para H3/H6
        _m("Q", L, "dlco", 50, 100, "LV1"),                                       # sin HADS ni mMRC: fuera
        _m("C", "CIBERESUCICOVID", "dlco", 65, 90, "M3"),                         # CIBERES: ancla solo DLCO/FVC
    ])
    h3, _ = FP.construir_variables(med, pac, cfg, pf, "H3", L)
    assert list(h3.index) == ["P"] and h3["dias_max"].max() <= 150
    h6, _ = FP.construir_variables(med, pac, cfg, pf, "H6", L)
    assert h6.loc["P", "dlco_cambio"] == 10 and h6["dias_max"].max() <= 270
    c3, _ = FP.construir_variables(med, pac, cfg, pf, "H3", "CIBERESUCICOVID")
    assert list(c3.index) == ["C"] and np.isnan(c3.loc["C", "hads_a_T3"])
