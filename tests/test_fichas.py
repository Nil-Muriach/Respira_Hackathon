"""Tests de la tabla de diferencias entre fenotipos (src/fichas.py) con datos sintéticos."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import carga, fichas  # noqa: E402


@pytest.fixture
def cfg() -> dict:
    return carga.cargar_config()


@pytest.fixture
def sinteticos() -> tuple[pd.DataFrame, pd.Series]:
    """Dos fenotipos de 100: DLCO muy distinta, sexo igual y un evento raro (5 casos) solo en F2."""
    rng = np.random.default_rng(0)
    idx = pd.Index([f"p{i}" for i in range(200)])
    fen = pd.Series(["F1"] * 100 + ["F2"] * 100, index=idx)
    datos = pd.DataFrame({
        "dlco_T3": np.r_[rng.normal(90, 8, 100), rng.normal(60, 8, 100)],
        "mujer": np.tile([0.0, 1.0], 100),
        "raro": np.r_[np.zeros(100), np.ones(5), np.zeros(95)],
    }, index=idx)
    return datos, fen


def _info() -> list[dict]:
    return [{"col": "dlco_T3", "bloque": "Define el fenotipo", "variable": "DLCO T3", "binaria": False},
            {"col": "mujer", "bloque": "Describe · ingreso", "variable": "Mujer", "binaria": True},
            {"col": "raro", "bloque": "Describe · eventos", "variable": "Evento raro", "binaria": True}]


def test_diferencia_grande_y_despreciable(cfg, sinteticos):
    datos, fen = sinteticos
    tabla, retratos = fichas.tabla_diferencias(datos, _info(), fen, cfg)
    dlco = tabla.xs("DLCO T3", level="variable").iloc[0]
    assert dlco["Magnitud"] == "grande" and dlco["p ajustada"] == "— (define)"   # define: sin p
    assert dlco["Interpretación"].startswith("Más alta en F1")
    assert tabla.xs("Mujer", level="variable").iloc[0]["Interpretación"] == "Sin diferencias relevantes"
    assert "↑ DLCO T3" in retratos.iloc[0, 0] and "↓ DLCO T3" in retratos.iloc[1, 0]


def test_celdas_pequenas_se_suprimen_y_no_se_afirman(cfg, sinteticos):
    datos, fen = sinteticos
    tabla, _ = fichas.tabla_diferencias(datos, _info(), fen, cfg)
    raro = tabla.xs("Evento raro", level="variable").iloc[0]
    n_min = cfg["privacidad"]["n_minimo_celda"]
    assert raro.filter(like="F2").iloc[0] == f"<{n_min} %"          # 5 casos de 100: solo la cota
    assert "Más frecuente en F2" not in raro["Interpretación"]        # la cota nunca se da como «más»


def test_separacion_total_es_infinita(cfg):
    a = {"media": 1.0, "var": 0.0, "cota": False}
    b = {"media": 0.0, "var": 0.0, "cota": False}
    assert np.isinf(fichas.smd(a, b)) and fichas.magnitud(np.inf, cfg) == "grande"
    assert np.isnan(fichas.smd({**a, "cota": True}, b))


def test_fenotipo_pequeno_no_se_describe(cfg, sinteticos):
    datos, fen = sinteticos
    fen = fen.copy()
    fen.iloc[:5] = "F3"                                             # 5 pacientes < n_min
    tabla, retratos = fichas.tabla_diferencias(datos, _info(), fen, cfg)
    col_f3 = [c for c in tabla.columns if c.startswith("F3")][0]
    assert col_f3.endswith(f"(N<{cfg['privacidad']['n_minimo_celda']})") and (tabla[col_f3] == "—").all()
    assert not any(i.startswith("F3") for i in retratos.index)
