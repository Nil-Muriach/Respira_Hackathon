"""Tests del fenotipado por perfil (notebook 04, las tres cohortes juntas) con datos sintéticos."""
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
    return FP.perfil(cfg, "fenotipado_comun")


def _m(sid, reg, var, valor, dias, visita):
    return {"subject_id": sid, "registro": reg, "variable": var, "valor": valor, "dias_alta": dias,
            "visita": visita, "excluida": False}


def test_columnas_y_compartidas(cfg, pf):
    cols = FP.columnas_definicion(cfg, pf, "H6")
    assert "dlco_T3" in cols and "fev1_T6" in cols and "dlco_cambio" in cols
    assert FP.compartidas(cfg, pf, cols, "CIBERESUCICOVID").all()   # las tres cohortes recogen DLCO, FVC y FEV1
    assert FP.compartidas(cfg, pf, cols, "TENACITY").all()
    assert not FP.compartidas(cfg, pf, cols, "VIRGEN_DEL_ROCIO").any()   # sin visitas a 3-6-12 meses en este perfil


def test_pesos_iguales_por_dominio(cfg, pf):
    cols = FP.columnas_definicion(cfg, pf, "H3")                  # DLCO (difusión) y FVC + FEV1 (espirometría)
    _, pesos, cat = FP.parametros_gower(cols, pf)
    por_dom = {d: sum(p for c, p in zip(cols, pesos) if c.rsplit("_", 1)[0] in vs) for d, vs in pf["dominios"].items()}
    assert pesos.sum() == pytest.approx(1) and np.allclose(list(por_dom.values()), 0.5) and not cat.any()


def test_construccion_ancla_y_sin_fuga(cfg, pf):
    L = "POSTCOVID_LLEIDA"
    pac = pd.DataFrame({"subject_id": ["P", "Q", "C"], "superviviente": True, "inconsistente": False, "cmd_ok": True,
                        "cohorte_analisis": [L, L, "CIBERESUCICOVID"], "centro_id": "X"})
    med = pd.DataFrame([
        _m("P", L, "dlco", 60, 100, "LV1"),
        _m("P", L, "dlco", 70, 200, "LV2"),
        _m("P", L, "dlco", 95, 370, "LA1"),                                       # futuro para H3/H6
        _m("Q", L, "hads_a", 9, 100, "LV1"),                                      # sin función pulmonar: fuera
        _m("C", "CIBERESUCICOVID", "fvc", 75, 90, "M3"),                          # basta una de las tres
    ])
    h3, _ = FP.construir_variables(med, pac, cfg, pf, "H3", L)
    assert list(h3.index) == ["P"] and h3["dias_max"].max() <= 150
    h6, _ = FP.construir_variables(med, pac, cfg, pf, "H6", L)
    assert h6.loc["P", "dlco_cambio"] == 10 and h6["dias_max"].max() <= 270
    c3, _ = FP.construir_variables(med, pac, cfg, pf, "H3", "CIBERESUCICOVID")
    assert list(c3.index) == ["C"] and np.isnan(c3.loc["C", "dlco_T3"]) and c3.loc["C", "fvc_T3"] == 75
