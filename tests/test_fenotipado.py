"""Tests del fenotipado con datos sintéticos (sin datos reales)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import carga, fenotipado as F, pasaporte as P  # noqa: E402


@pytest.fixture
def cfg() -> dict:
    return carga.cargar_config()


def test_gower_valores_a_mano():
    rangos, pesos = np.array([10.0, 10.0]), np.array([0.5, 0.5])
    X = np.array([[0.0, 0.0], [5.0, 10.0], [0.0, np.nan], [np.nan, 3.0]])
    D = F.gower(X, None, rangos, pesos)
    assert D[0, 1] == pytest.approx((0.5 * 0.5 + 0.5 * 1.0) / 1.0)   # 0,75
    assert D[1, 2] == pytest.approx(0.5)                              # solo comparten la 1.ª variable
    assert D[2, 3] == 1.0                                             # sin variables en común: máxima
    assert np.allclose(np.diag(D), 0) and np.allclose(D, D.T)


def test_gower_recorta_al_rango():
    D = F.gower(np.array([[0.0], [50.0]]), None, np.array([10.0]), np.array([1.0]))
    assert D[0, 1] == 1.0


def test_pam_recupera_grupos_evidentes():
    rng = np.random.default_rng(0)
    X = np.r_[rng.normal(0, 1, (40, 2)), rng.normal(20, 1, (40, 2))]
    D = F.gower(X, None, np.array([40.0, 40.0]), np.array([0.5, 0.5]))
    lab, med = F.pam(D, 2, 0)
    assert len(set(lab[:40])) == 1 and len(set(lab[40:])) == 1 and lab[0] != lab[-1]
    assert F.silueta(D, lab) > 0.8


def test_hungaro_con_etiquetas_permutadas():
    a = np.array([0, 0, 1, 1, 2, 2])
    b = np.array([2, 2, 0, 0, 1, 1])
    mapa, jacc = P.emparejar_hungaro(a, b, 3)
    assert np.array_equal(mapa[b], a) and np.allclose(jacc, 1.0)


def test_pesos_iguales_por_dominio(cfg):
    _, rangos, pesos = F.parametros_gower(["dlco_T3", "fvc_T3", "hads_a_T3", "hads_d_T3", "mmrc_T3", "pm6m_T3"], cfg, "B")
    assert pesos.sum() == pytest.approx(1.0)
    assert pesos[0] + pesos[1] == pytest.approx(pesos[4]) == pytest.approx(pesos[5])
    assert rangos[0] == 140 and rangos[4] == 4


def _medida(sid, variable, valor, dias):
    return {"subject_id": sid, "variable": variable, "valor": valor, "dias_alta": dias, "excluida": False}


def test_variables_acumuladas_sin_fuga(cfg):
    pac = pd.DataFrame({"subject_id": ["P", "Q"], "superviviente": True, "inconsistente": False, "cmd_ok": True,
                        "cohorte_analisis": "CIBERESUCICOVID", "centro_id": "C1"})
    med = pd.DataFrame([
        _medida("P", "dlco", 60, 100),     # T3
        _medida("P", "dlco", 70, 200),     # T6
        _medida("P", "dlco", 90, 400),     # T12 (futuro para H3 y H6)
        _medida("Q", "dlco", 50, 100),     # Q no tiene T6: fuera de H6 (ancla)
    ])
    h3, cols3 = F.construir_variables(med, pac, cfg, "H3", "A")
    assert cols3 == ["dlco_T3", "fvc_T3"] and h3["dias_max"].max() <= 135
    assert set(h3.index) == {"P", "Q"}
    h6, cols6 = F.construir_variables(med, pac, cfg, "H6", "A")
    assert set(h6.index) == {"P"}                                   # ancla en T6
    assert h6.loc["P", "dlco_cambio"] == 10 and h6["dias_max"].max() <= 270
    h12, _ = F.construir_variables(med, pac, cfg, "H12", "A")
    assert h12.loc["P", "dlco_cambio"] == 30                        # último − primero


def test_ordenar_por_severidad():
    lab = np.array([0, 0, 1, 1])
    lab2, med2 = F.ordenar_por_severidad(lab, np.array([0, 2]), np.array([50.0, 55, 90, 95]))
    assert list(lab2) == [1, 1, 0, 0] and list(med2) == [2, 0]      # 0 = DLCO más alta


def test_porcentaje_alterado(cfg):
    datos = pd.DataFrame({"dlco_T3": [70, 90, np.nan, 60], "hads_a_T3": [10, 2, 9, 1], "dlco_cambio": [1, 2, 3, 4]})
    pct, n = F.porcentaje_alterado(datos, list(datos.columns), np.array([0, 0, 1, 1]), cfg)
    assert "dlco_cambio" not in pct.index                         # el cambio no tiene umbral
    assert pct.loc["dlco_T3", 0] == 50 and pct.loc["dlco_T3", 1] == 100   # el NaN no cuenta
    assert n.loc["hads_a_T3", 1] == 1
