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


def test_gower_numericas_y_ausencias():
    rangos, pesos = np.array([10.0, 10.0]), np.array([0.5, 0.5])
    X = np.array([[0.0, 0.0], [5.0, 10.0], [0.0, np.nan], [np.nan, 3.0]])
    D = F.gower(X, None, rangos, pesos)
    assert D[0, 1] == pytest.approx(0.75)
    assert D[1, 2] == pytest.approx(0.5)          # solo comparten la primera variable
    assert D[2, 3] == 1.0                         # nada en común: máxima
    assert np.allclose(np.diag(D), 0) and np.allclose(D, D.T)


def test_gower_categoricas_y_recorte():
    X = np.array([[0.0, 1.0], [50.0, 0.0]])
    D = F.gower(X, None, np.array([10.0, 1.0]), np.array([0.5, 0.5]), np.array([False, True]))
    assert D[0, 1] == pytest.approx(1.0)          # numérica recortada a 1 + categoría distinta


def test_pam_y_silueta():
    rng = np.random.default_rng(0)
    X = np.r_[rng.normal(0, 1, (40, 2)), rng.normal(20, 1, (40, 2))]
    D = F.gower(X, None, np.array([40.0, 40.0]), np.array([0.5, 0.5]))
    lab, _ = F.pam(D, 2, 0)
    assert len(set(lab[:40])) == 1 and lab[0] != lab[-1] and F.silueta(D, lab) > 0.8


def test_hungaro_etiquetas_permutadas():
    a, b = np.array([0, 0, 1, 1, 2, 2]), np.array([2, 2, 0, 0, 1, 1])
    mapa, jacc = P.emparejar_hungaro(a, b, 3)
    assert np.array_equal(mapa[b], a) and np.allclose(jacc, 1.0)


def test_ordenar_por_dlco():
    lab, med = F.ordenar_por_dlco(np.array([0, 0, 1, 1]), np.array([0, 2]), np.array([50.0, 55, 90, 95]))
    assert list(lab) == [1, 1, 0, 0] and list(med) == [2, 0]


def test_pesos_iguales_por_dominio(cfg):
    cols = ["dlco_M3", "fvc_M3", "fatiga_M3", "resolucion_M3"]
    rangos, pesos, cat = F.parametros_gower(cols, cfg)
    assert pesos.sum() == pytest.approx(1) and pesos[0] + pesos[1] == pytest.approx(pesos[2] + pesos[3])
    assert list(cat) == [False, False, True, False]


def _m(sid, reg, var, valor, dias, visita):
    return {"subject_id": sid, "registro": reg, "variable": var, "valor": valor, "dias_alta": dias,
            "visita": visita, "excluida": False}


def test_variables_por_visita_sin_fuga_y_ancla(cfg):
    pac = pd.DataFrame({"subject_id": ["P", "Q", "R"], "superviviente": True, "inconsistente": False, "cmd_ok": True,
                        "cohorte_analisis": "CIBERESUCICOVID", "centro_id": "C1"})
    C = "CIBERESUCICOVID"
    med = pd.DataFrame([
        _m("P", C, "dlco", 60, 90, "M3"), _m("P", C, "resolucion", 1, np.nan, "M3"),   # síntoma sin fecha: vale
        _m("P", C, "dlco", 70, 180, "M6"), _m("P", C, "fatiga", 1, np.nan, "M6"),
        _m("P", C, "dlco", 90, 360, "A1"),                                             # futuro para H3/H6
        _m("Q", C, "dlco", 50, 90, "M3"),                                              # sin síntomas: fuera de H3
        _m("R", C, "dlco", 80, 400, "M3"), _m("R", C, "fatiga", 0, 95, "M3"),          # DLCO fuera de ventana M3
    ])
    h3, cols3 = F.construir_variables(med, pac, cfg, "H3", C)
    assert cols3 == ["dlco_M3", "fvc_M3", "fatiga_M3", "resolucion_M3"]
    assert set(h3.index) == {"P"}                  # Q sin síntomas; R sin función válida en M3
    h6, _ = F.construir_variables(med, pac, cfg, "H6", C)
    assert h6.loc["P", "dlco_cambio"] == 10 and h6["dias_max"].max() <= 270
    h12, cols12 = F.construir_variables(med, pac, cfg, "H12", C)
    assert "fatiga_A1" not in cols12 and h12.loc["P", "dlco_cambio"] == 30   # A1 no recoge síntomas


def test_replica_solo_exige_compartidas(cfg):
    pac = pd.DataFrame({"subject_id": ["L"], "superviviente": True, "inconsistente": False, "cmd_ok": True,
                        "cohorte_analisis": "POSTCOVID_LLEIDA", "centro_id": "C2"})
    med = pd.DataFrame([_m("L", "POSTCOVID_LLEIDA", "dlco", 70, 100, "LV1")])
    t, cols = F.construir_variables(med, pac, cfg, "H3", "POSTCOVID_LLEIDA")
    assert list(t.index) == ["L"] and t.loc["L", "dlco_M3"] == 70 and np.isnan(t.loc["L", "fatiga_M3"])
