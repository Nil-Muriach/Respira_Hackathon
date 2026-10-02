"""Tests de las trayectorias con datos sintéticos (sin datos reales)."""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import trayectorias as T  # noqa: E402

warnings.filterwarnings("ignore")


def _sinteticos(n: int = 600, semilla: int = 0) -> pd.DataFrame:
    """Trayectorias con efectos conocidos: F2 empieza 14 puntos más abajo y gana 2 puntos más por unidad de lt."""
    rng = np.random.default_rng(semilla)
    fen = rng.choice(["F1", "F2"], n)
    coh = rng.choice(["A", "B"], n)
    b0, b1 = rng.normal(0, 6, n), rng.normal(0, 1.5, n)
    filas = []
    for i in range(n):
        for meses in rng.choice([3, 6, 12], rng.integers(1, 4), replace=False):
            lt = np.log(meses / 3)
            f2 = fen[i] == "F2"
            valor = 78 - 14 * f2 + 3 * (coh[i] == "B") + (5 + 2 * f2) * lt + b0[i] + b1[i] * lt + rng.normal(0, 3)
            filas.append({"subject_id": i, "fenotipo": fen[i], "cohorte": coh[i], "lt": lt, "valor": valor})
    return pd.DataFrame(filas)


def test_mixto_recupera_efectos_y_contrastes():
    d = _sinteticos()
    res, tipo = T.ajustar_mixto(d, "valor ~ fenotipo * lt + cohorte")
    assert tipo == "intercepto y pendiente aleatorios"
    assert res.fe_params["fenotipo[T.F2]"] == pytest.approx(-14, abs=1.5)
    assert res.fe_params["fenotipo[T.F2]:lt"] == pytest.approx(2, abs=1.0)
    dif = T.diferencias(res, "F1", "F2", [3, 12], 3)
    assert dif.loc[0, "estimacion"] == pytest.approx(res.fe_params["fenotipo[T.F2]"])          # a 3 meses, lt = 0
    assert dif.loc[1, "estimacion"] == pytest.approx(res.fe_params["fenotipo[T.F2]"] + np.log(4) * res.fe_params["fenotipo[T.F2]:lt"])
    g = T.ganancia(res, "F1", 3, 12, 3)
    assert g["estimacion"] == pytest.approx(np.log(4) * res.fe_params["lt"])
    assert g["ic_inf"] < g["estimacion"] < g["ic_sup"]


def test_prediccion_marginal_promedia_cohortes():
    d = _sinteticos()
    res, _ = T.ajustar_mixto(d, "valor ~ fenotipo * lt + cohorte")
    pa = T.predicciones(res, ["F1"], [3], 3, {"A": 1.0, "B": 0.0}).loc[0, "estimacion"]
    pb = T.predicciones(res, ["F1"], [3], 3, {"A": 0.0, "B": 1.0}).loc[0, "estimacion"]
    pm = T.predicciones(res, ["F1"], [3], 3, {"A": 0.5, "B": 0.5}).loc[0, "estimacion"]
    assert pm == pytest.approx((pa + pb) / 2) and pb - pa == pytest.approx(res.fe_params["cohorte[T.B]"])


def test_gee_con_pesos_unitarios_igual_que_sin_pesos():
    d = _sinteticos(300)
    a = T.ajustar_gee(d, "valor ~ fenotipo * lt")
    b = T.ajustar_gee(d, "valor ~ fenotipo * lt", np.ones(len(d)))
    assert np.allclose(a.params, b.params)
    assert T.diferencias(a, "F1", "F2", [12], 3).loc[0, "estimacion"] == pytest.approx(
        a.params["fenotipo[T.F2]"] + np.log(4) * a.params["fenotipo[T.F2]:lt"])


def test_tabla_larga_filtra_registro_ventana_y_excluidas():
    fen = pd.Series({"P": "F1", "Q": "F2"})
    coh = pd.Series({"P": "LLEIDA", "Q": "CIBER"})
    med = pd.DataFrame([
        {"subject_id": "P", "registro": "LLEIDA", "variable": "dlco", "valor": 70, "dias_alta": 91, "visita": "LV1", "excluida": False},
        {"subject_id": "P", "registro": "CIBER", "variable": "dlco", "valor": 71, "dias_alta": 95, "visita": "M3", "excluida": False},   # otro registro
        {"subject_id": "P", "registro": "LLEIDA", "variable": "dlco", "valor": 80, "dias_alta": 900, "visita": "LM24", "excluida": False},  # fuera de ventana
        {"subject_id": "Q", "registro": "CIBER", "variable": "dlco", "valor": 60, "dias_alta": 365, "visita": "A1", "excluida": False},
        {"subject_id": "Q", "registro": "CIBER", "variable": "dlco", "valor": 300, "dias_alta": 100, "visita": "M3", "excluida": True},
        {"subject_id": "Q", "registro": "CIBER", "variable": "fvc", "valor": 90, "dias_alta": 100, "visita": "M3", "excluida": False},
        {"subject_id": "R", "registro": "CIBER", "variable": "dlco", "valor": 90, "dias_alta": 100, "visita": "M3", "excluida": False},     # sin fenotipo
    ])
    t = T.tabla_larga(med, fen, coh, "dlco", (30, 480), 30.44, 3)
    assert list(t["valor"]) == [70, 60]
    assert t.loc[0, "lt"] == pytest.approx(np.log(91 / 30.44 / 3)) and list(t["cohorte"]) == ["LLEIDA", "CIBER"]


def test_tramos_y_patrones():
    assert list(T.tramo(pd.Series([30, 150, 151, 480]), [30, 150, 270, 480], ["T3", "T6", "T12"]).astype(str)) == \
        ["T3", "T3", "T6", "T12"]
    pres = pd.DataFrame({"T3": [True, True, False], "T6": [False, True, False], "T12": [True, True, False]})
    assert list(T.patron_seguimiento(pres)) == ["T3 + T12", "T3 + T6 + T12", "ninguno"]


def test_pesos_ipw_mayores_si_es_improbable_volver():
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"x": rng.normal(size=2000)})
    obs = pd.Series(rng.random(2000) < 1 / (1 + np.exp(-2 * X["x"])))
    w = T.pesos_ipw(X, obs, recorte_percentil=100)
    assert w[~obs].isna().all() and (w[obs] > 0).all()
    assert w[obs & (X["x"] < -1)].mean() > w[obs & (X["x"] > 1)].mean()      # quien rara vez vuelve pesa más
    assert w[obs].mean() == pytest.approx(1, abs=0.15)                          # estabilizados: media ≈ 1
    # los observados están sesgados hacia x alto; ponderados, se parecen a la población completa
    assert X.loc[obs, "x"].mean() > 0.4
    assert np.average(X.loc[obs, "x"], weights=w[obs]) == pytest.approx(X["x"].mean(), abs=0.1)
