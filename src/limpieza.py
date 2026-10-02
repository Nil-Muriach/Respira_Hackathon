"""Limpieza de datos: de la tabla cruda a tablas analizables y trazables.

Cada regla es una función pequeña que devuelve el objeto transformado y una o varias
entradas para `registro_limpieza`. Nada se borra en silencio: las medidas descartadas se
quedan en la tabla con `excluida = True` y su `motivo_exclusion`. Las reglas y su
justificación están en docs/registro_decisiones.md.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src import carga

MOTIVOS = ["codigo_no_disponible", "dias_negativos", "fuera_de_rango", "duplicado_entre_registros"]


# ---------------------------------------------------------------------------
# Registro de reglas

def entrada(regla: str, descripcion: str, afectados: pd.Series, tabla: str) -> dict:
    """Entrada de `registro_limpieza`: N afectados en total y por registro.

    `afectados` es la serie con el registro de cada fila afectada (una fila = un caso).
    """
    conteo = afectados.astype(str).value_counts()
    return {"regla": regla, "descripcion": descripcion, "tabla": tabla, "N": int(len(afectados)),
            **{f"N_{r}": int(n) for r, n in conteo.items()}}


# ---------------------------------------------------------------------------
# Pacientes

def recodificar_pacientes(nucleo: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, list[dict]]:
    """R01: código 9 ("se desconoce") y sexo no especificado -> NaN."""
    df = nucleo.copy()
    entradas = []
    codigo = cfg["calidad"]["codigo_desconocido"]
    for c in carga.VARIABLES_CODIGO_9:
        mask = df[c] == codigo
        if mask.any():
            entradas.append(entrada("R01", f"{c}: código {codigo} -> NaN", df.loc[mask, "cohorte"], "pacientes"))
        df[c] = df[c].where(~mask)
    sexo_ne = cfg["limpieza"]["sexo_no_especificado"]
    for c in ["sexo", "nu_sexo"]:
        mask = df[c] == sexo_ne
        entradas.append(entrada("R01", f"{c}: {sexo_ne} (no especificado) -> NaN", df.loc[mask, "cohorte"], "pacientes"))
        df[c] = df[c].where(~mask)
    return df, entradas


def anadir_edad(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, list[dict]]:
    """R02: edad ordinal (orden por límite inferior) y grupo amplio; tramos ambiguos -> grupo NaN."""
    df = df.copy()
    df["edad_tramo_ord"] = carga.edad_ordenada(df["edad_tramo5"])
    df["edad_lim_inf"] = df["edad_tramo5"].map(carga.limite_inferior_tramo)
    df["grupo_edad"] = carga.edad_agrupada(df["edad_tramo5"], cfg["edad_grupos"])
    ambiguos = df["edad_tramo5"].notna() & df["grupo_edad"].isna()
    return df, [entrada("R02", "Tramo de edad que cruza grupos -> grupo_edad NaN (tramo original conservado)",
                        df.loc[ambiguos, "cohorte"], "pacientes")]


def asignar_cohorte_analisis(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, list[dict]]:
    """R07: cada paciente en un solo registro de análisis (un solo lado de la replicación)."""
    df = df.copy()
    df["cohorte_analisis"] = df["cohorte"]
    entradas = []
    for bandera, destino in cfg["limpieza"]["asignacion_compartidos"].items():
        mask = df[bandera] == 1
        df.loc[mask, "cohorte_analisis"] = destino
        entradas.append(entrada("R07", f"{bandera} == 1 -> cohorte_analisis = {destino}",
                                df.loc[mask, "cohorte_analisis"], "pacientes"))
    df["cohorte_analisis"] = pd.Categorical(df["cohorte_analisis"], categories=list(cfg["registros"]), ordered=True)
    return df, entradas


def banderas_pacientes(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, list[dict]]:
    """R04 (visita_dias < 0), R08 (supervivencia) y R09 (cumplimentación CIBERESUCICOVID)."""
    df = df.copy()
    neg = df["visita_dias"] < 0
    df["visita_dias_valida"] = df["visita_dias"].where(~neg)
    df["superviviente"] = df["nu_exitus_hosp"] != 1
    cib = df["cohorte_analisis"] == "CIBERESUCICOVID"
    df["cmd50"] = df["IH_PorcCMD"] >= cfg["calidad"]["ih_porccmd_principal"]
    df["cmd99"] = df["IH_PorcCMD"] >= cfg["calidad"]["ih_porccmd_sensibilidad"]
    df["cmd_ok"] = ~cib | df["cmd50"]          # el criterio solo aplica al cuaderno de CIBERESUCICOVID
    reg = df["cohorte_analisis"]
    return df, [
        entrada("R04", "visita_dias < 0 -> visita_dias_valida NaN (fuera del análisis temporal)", reg[neg], "pacientes"),
        entrada("R08", "Exitus hospitalario -> superviviente = False", reg[~df["superviviente"]], "pacientes"),
        entrada("R09", f"CIBERESUCICOVID con IH_PorcCMD < {cfg['calidad']['ih_porccmd_principal']} -> cmd_ok = False",
                reg[~df["cmd_ok"]], "pacientes"),
    ]


# ---------------------------------------------------------------------------
# Medidas

def limpiar_medidas(medidas: pd.DataFrame, pacientes: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, list[dict]]:
    """R01b, R04, R05 y R06 sobre la tabla larga. Marca `excluida` y `motivo_exclusion` (el primero que aplica)."""
    m = medidas.copy()
    m["excluida"] = False
    m["motivo_exclusion"] = pd.Series(pd.NA, index=m.index, dtype="string")
    entradas = []

    def marcar(mask: pd.Series, motivo: str, regla: str, descripcion: str) -> None:
        nuevo = mask & ~m["excluida"]
        m.loc[nuevo, "excluida"] = True
        m.loc[nuevo, "motivo_exclusion"] = motivo
        entradas.append(entrada(regla, descripcion, m.loc[nuevo, "registro"], "medidas"))

    # R03 (informativa): las medidas sin fecha se conservan, pero no pueden entrar en ninguna ventana temporal
    entradas.append(entrada("R03", "Fecha de la prueba ausente: se usa visita_dias (solo 1.ª visita)",
                            m.loc[m["fecha_imputada"], "registro"], "medidas"))
    entradas.append(entrada("R03", "Sin fecha ni visita_dias: válida, pero fuera de cualquier ventana temporal",
                            m.loc[m["dias_alta"].isna(), "registro"], "medidas"))
    codigos = cfg["limpieza"]["codigos_no_disponible"]
    marcar(m["valor"].isin(codigos), "codigo_no_disponible", "R01", f"Valor {codigos} (no disponible)")
    for variable, cods in cfg["limpieza"].get("codigos_desconocido_variable", {}).items():
        marcar((m["variable"] == variable) & m["valor"].isin(cods), "codigo_no_disponible", "R01",
               f"{variable}: código {cods} (se desconoce)")
    marcar(m["dias_alta"] < 0, "dias_negativos", "R04", "Medida con fecha anterior al alta")
    rangos = cfg["rangos_plausibles"]
    lo = m["variable"].map(lambda v: rangos.get(v, [-np.inf, np.inf])[0])
    hi = m["variable"].map(lambda v: rangos.get(v, [-np.inf, np.inf])[1])
    marcar((m["valor"] < lo) | (m["valor"] > hi), "fuera_de_rango", "R05", "Valor fuera del rango plausible")
    dup, descripcion = detectar_duplicados(m, pacientes, cfg)
    marcar(dup, "duplicado_entre_registros", "R06", descripcion)
    return m, entradas


def detectar_duplicados(m: pd.DataFrame, pacientes: pd.DataFrame, cfg: dict) -> tuple[pd.Series, str]:
    """R06: en pacientes compartidos, la medida de CIBERESUCICOVID es duplicada si el registro
    socio tiene la misma variable a |Δdías| <= tolerancia (o el mismo mes nominal si falta la fecha)."""
    tol = cfg["limpieza"]["tolerancia_duplicado_dias"]
    socio = pd.Series(pd.NA, index=pacientes.index, dtype="object")
    for bandera, destino in cfg["limpieza"]["asignacion_compartidos"].items():
        socio[pacientes[bandera] == 1] = destino
    socio = pd.Series(socio.values, index=pacientes["subject_id"]).dropna()

    validas = m[~m["excluida"]]
    cib = validas[(validas["registro"] == "CIBERESUCICOVID") & validas["subject_id"].isin(socio.index)].reset_index()
    if cib.empty:
        return pd.Series(False, index=m.index), "Medida de CIBERESUCICOVID repetida en el registro socio (sin casos)"
    cib["socio"] = cib["subject_id"].map(socio)
    otras = validas[validas["registro"] != "CIBERESUCICOVID"][["subject_id", "registro", "variable", "valor", "dias_alta", "mes_nominal"]]
    par = cib.merge(otras, left_on=["subject_id", "socio", "variable"], right_on=["subject_id", "registro", "variable"],
                    suffixes=("", "_socio"))
    cerca = np.where(par["dias_alta"].notna() & par["dias_alta_socio"].notna(),
                     (par["dias_alta"] - par["dias_alta_socio"]).abs() <= tol,
                     par["mes_nominal"] == par["mes_nominal_socio"])
    par = par[cerca]
    idx = par["index"].unique()
    iguales = (par.groupby("index").apply(lambda g: (g["valor"] - g["valor_socio"]).abs().min() <= 1,
                                          include_groups=False).mean() if len(par) else np.nan)
    descripcion = (f"Medida de CIBERESUCICOVID repetida en el registro socio (|Δdías| <= {tol}); "
                   f"valor idéntico (±1) en el {100 * iguales:.0f} %")
    return m.index.isin(idx), descripcion


def deducir_condicionadas(m: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, list[dict]]:
    """R14: preguntas condicionadas. Si el cuaderno solo pregunta `variable` cuando `si` no vale
    `igual_a` (p. ej., fatiga solo si la resolución no es total), su ausencia en ese caso no es un
    dato perdido: se añade la medida deducida (`valor`), marcada con `deducida = True`.

    Solo se deduce si no existe ya una medida válida de `variable` en esa visita.
    """
    m = m.copy()
    if "deducida" not in m:
        m["deducida"] = False
    entradas, nuevas = [], []
    for regla in cfg["limpieza"].get("preguntas_condicionadas", []):
        validas = m[~m["excluida"] & (m["registro"].astype(str) == regla["registro"])]
        condicion = validas[(validas["variable"] == regla["si"]) & (validas["valor"] == regla["igual_a"])]
        ya = validas.loc[validas["variable"] == regla["variable"], ["subject_id", "visita"]]
        clave = ["subject_id", "visita"]
        faltan = condicion.merge(ya.drop_duplicates(), on=clave, how="left", indicator=True)
        faltan = faltan[faltan["_merge"] == "left_only"].drop(columns="_merge")
        faltan = faltan.drop_duplicates(clave)
        faltan = faltan.assign(variable=regla["variable"], valor=float(regla["valor"]), deducida=True,
                               fuente_columna=f"deducida de {regla['si']} = {regla['igual_a']}")
        nuevas.append(faltan[m.columns])
        entradas.append(entrada("R14", f"{regla['variable']} = {regla['valor']} deducida cuando {regla['si']} = "
                                       f"{regla['igual_a']} (pregunta condicionada del cuaderno)",
                                faltan["registro"], "medidas"))
    if nuevas:
        m = pd.concat([m, *nuevas], ignore_index=True)
        m["registro"] = pd.Categorical(m["registro"], categories=list(cfg["registros"]), ordered=True)
    return m, entradas


def medidas_imagen(cfg: dict, pacientes: pd.DataFrame, completa: pd.DataFrame | None = None) -> tuple[pd.DataFrame, list[dict]]:
    """R10: imagen como tabla larga. Solo formularios completos (Lleida/TENACITY) y TAC realizados
    (CIBERESUCICOVID). Fibrosis estricta = fibrótica; amplia = fibrótica o reticular.
    CIBERESUCICOVID: una variable 0/1 por hallazgo del TAC (casillas `prefijo___código`)."""
    img = cfg["imagen"]
    hallazgos = img["tac_ciberes"]["hallazgos"]
    fuentes = [("fibrosis", r, *f) for r, fs in img["fibrosis"].items() for f in fs] + \
              [("tac", "CIBERESUCICOVID", *f) for f in img["tac_ciberes"]["visitas"]]
    if completa is None:
        cols = set()
        for tipo, _, _, bandera, var, fecha, _ in fuentes:
            vars_ = [f"{var}___{c}" for c in hallazgos.values()] if tipo == "tac" else [var]
            cols |= {bandera, *vars_, *carga.columnas_fecha(fecha)}
        completa = carga.cargar_columnas_completa(cfg, sorted(cols))
    completa = completa.merge(pacientes[["subject_id", "nu_fecha_alta"]], on="subject_id", how="left")

    partes, entradas = [], []
    for tipo, registro, visita, bandera, var, fecha_cols, mes in fuentes:
        con_form = completa[completa[bandera].notna()]
        if tipo == "fibrosis":
            valido = con_form[bandera] == img["formulario_completo"]
            descartados = "formulario de TAC no completo (ceros = sin rellenar)"
        else:
            valido = con_form[bandera] == 1
            descartados = "visita sin TAC realizado"
        entradas.append(entrada("R10", f"{registro} {visita}: {descartados}",
                                pd.Series(registro, index=con_form.index[~valido]), "medidas"))
        sub = con_form[valido]
        fecha = carga.fecha_coalescida(sub, carga.columnas_fecha(fecha_cols), cfg["formato_fecha"][registro])
        dias = (fecha - sub["nu_fecha_alta"]).dt.days
        if tipo == "fibrosis":
            lesion = sub[var]
            valores = {"fibrosis_estricta": (lesion == img["codigo_fibrotica"]).astype(float),
                       "fibrosis_amplia": lesion.isin([img["codigo_fibrotica"], img["codigo_reticular"]]).astype(float)}
        else:
            valores = {nombre: sub[f"{var}___{cod}"].astype(float) for nombre, cod in hallazgos.items()}
        for variable, valor in valores.items():
            partes.append(pd.DataFrame({
                "subject_id": sub["subject_id"].values, "registro": registro, "visita": visita, "mes_nominal": mes,
                "variable": variable, "valor": valor.values, "dias_alta": dias.values,
                "fecha_imputada": False, "fuente_columna": var}))
    tabla = pd.concat(partes, ignore_index=True)
    tabla["registro"] = pd.Categorical(tabla["registro"], categories=list(cfg["registros"]), ordered=True)
    tabla["meses_alta"] = tabla["dias_alta"] / cfg["limpieza"]["dias_por_mes"]
    return tabla, entradas


def marcar_inconsistentes(pacientes: pd.DataFrame, medidas: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """R08b: exitus hospitalario con medidas de seguimiento válidas -> inconsistente."""
    df = pacientes.copy()
    con_seg = set(medidas.loc[~medidas["excluida"], "subject_id"])
    df["con_seguimiento"] = df["subject_id"].isin(con_seg)
    df["inconsistente"] = ~df["superviviente"] & df["con_seguimiento"]
    return df, [entrada("R08", "Exitus hospitalario con seguimiento -> inconsistente (fuera del fenotipado)",
                        df.loc[df["inconsistente"], "cohorte_analisis"], "pacientes")]


# ---------------------------------------------------------------------------
# Ausencia estructural, TENACITY y estado de definición

def matriz_recogida(cfg: dict) -> pd.DataFrame:
    """R11: variable × registro -> True si el registro recoge la variable (según `variables_visita` e `imagen`)."""
    variables = {v: set(fs) for v, fs in cfg["variables_visita"].items()}
    variables["fibrosis_estricta"] = variables["fibrosis_amplia"] = set(cfg["imagen"]["fibrosis"])
    for hallazgo in cfg["imagen"]["tac_ciberes"]["hallazgos"]:
        variables[hallazgo] = {"CIBERESUCICOVID"}
    return pd.DataFrame({r: {v: r in regs for v, regs in variables.items()} for r in cfg["registros"]})


def visitas_tenacity(cfg: dict, pacientes: pd.DataFrame, completa: pd.DataFrame | None = None) -> tuple[pd.DataFrame, list[dict]]:
    """R12: estado de cada visita de TENACITY: realizada, abandono, no_le_toca o sin_dato.

    Corte = última fecha de visita registrada. Una visita no realizada cuya fecha teórica
    (alta + mes + margen) es posterior al corte no cuenta como abandono.
    """
    vis = cfg["visitas_tenacity"]
    formato = cfg["formato_fecha"]["TENACITY"]
    if completa is None:
        completa = carga.cargar_columnas_completa(cfg, [c for _, f, p, _ in vis for c in (f, p)])
    ten = pacientes.loc[pacientes["en_TENACITY"] == 1, ["subject_id", "nu_fecha_alta"]].merge(completa, on="subject_id")
    fechas = {v: carga.parsear_fecha(ten[f], formato) for v, f, _, _ in vis}
    corte = pd.concat(fechas.values()).max()
    margen = cfg["limpieza"]["margen_no_le_toca_dias"]
    filas = []
    for v, f, perdida, mes in vis:
        teorica = ten["nu_fecha_alta"] + pd.to_timedelta(mes * cfg["limpieza"]["dias_por_mes"] + margen, unit="D")
        estado = np.select(
            [fechas[v].notna(), ten[perdida] == 1, teorica > corte, ten["nu_fecha_alta"].isna()],
            ["realizada", "abandono", "no_le_toca", "sin_fecha_alta"], default="sin_dato")
        filas.append(pd.DataFrame({"subject_id": ten["subject_id"].values, "visita": v, "mes_nominal": mes,
                                   "estado": estado}))
    tabla = pd.concat(filas, ignore_index=True)
    tabla.attrs["corte"] = f"{corte:%Y-%m-%d}"   # texto: los metadatos de parquet deben ser JSON
    no_toca = tabla[tabla["estado"] == "no_le_toca"]
    return tabla, [entrada("R12", f"TENACITY: visita aún no debida (corte {corte:%Y-%m}) -> no_le_toca, no abandono",
                           pd.Series("TENACITY", index=no_toca.index), "visitas_tenacity")]


def estado_definicion(medidas: pd.DataFrame, pacientes: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """R13: primera medida válida de cada variable en la ventana de definición, con su estado
    clínico por umbral y su disponibilidad: medida / falta / no_recogida (estructural)."""
    v0, v1 = cfg["tiempo"]["ventana_definicion_dias"]
    defs = cfg["variables_definicion"]
    variables = defs["capa_A"] + defs["capa_B_extra"] + defs.get("sintomas_ciberes", []) + ["fev1"]
    en_ventana = medidas[~medidas["excluida"] & medidas["variable"].isin(variables)
                         & medidas["dias_alta"].between(v0, v1)]
    primera = en_ventana.sort_values("dias_alta").groupby(["subject_id", "variable"]).first().reset_index()
    valores = primera.pivot(index="subject_id", columns="variable", values="valor")
    dias = primera.pivot(index="subject_id", columns="variable", values="dias_alta").add_suffix("_dias")

    est = pacientes[["subject_id", "cohorte_analisis"]].set_index("subject_id")
    est = est.join(valores.reindex(columns=variables)).join(dias.reindex(columns=[f"{v}_dias" for v in variables]))
    recoge = matriz_recogida(cfg)
    umbrales = cfg["umbrales_clinicos"]
    for v in variables:
        clave, direccion = cfg["estados"][v]
        u = umbrales[clave]
        alterado = est[v] < u if direccion == "<" else est[v] >= u
        est[f"{v}_alterada"] = alterado.where(est[v].notna()).astype("boolean")
        recogida = est["cohorte_analisis"].astype(str).map(recoge.loc[v])
        est[f"{v}_disp"] = np.select([est[v].notna(), ~recogida.astype(bool)], ["medida", "no_recogida"], default="falta")
    return est.reset_index()


def anadir_inclusion(pacientes: pd.DataFrame, estado: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """R13: banderas incluible_A (DLCO o FVC en ventana) e incluible_B (+ HADS o mMRC)."""
    df = pacientes.merge(estado[["subject_id"] + [f"{v}_disp" for v in ["dlco", "fvc", "hads_a", "hads_d", "mmrc"]]],
                         on="subject_id", how="left")
    base = df["superviviente"] & ~df["inconsistente"] & df["cmd_ok"]
    resp = (df["dlco_disp"] == "medida") | (df["fvc_disp"] == "medida")
    otros = (df["hads_a_disp"] == "medida") | (df["hads_d_disp"] == "medida") | (df["mmrc_disp"] == "medida")
    df["incluible_A"] = base & resp
    df["incluible_B"] = df["incluible_A"] & otros
    return df.drop(columns=[c for c in df.columns if c.endswith("_disp")])


def flujo_consort(pacientes: pd.DataFrame) -> pd.DataFrame:
    """Diagrama de flujo por cohorte de análisis: cada paso exige los anteriores."""
    pasos = [
        ("1. Pacientes únicos", pd.Series(True, index=pacientes.index)),
        ("2. Supervivientes al alta", pacientes["superviviente"]),
        ("3. Sin inconsistencias", ~pacientes["inconsistente"]),
        ("4. Cumplimentación suficiente (CMD ≥ 50 en CIBERESUCICOVID)", pacientes["cmd_ok"]),
        ("5. Con alguna medida de seguimiento válida", pacientes["con_seguimiento"]),
        ("6. Capa A: DLCO o FVC en la ventana de definición", pacientes["incluible_A"]),
        ("7. Capa B: además HADS o mMRC en la ventana", pacientes["incluible_B"]),
    ]
    acumulado = pd.Series(True, index=pacientes.index)
    filas = {}
    for nombre, cond in pasos:
        acumulado = acumulado & cond.fillna(False).astype(bool)
        filas[nombre] = acumulado.groupby(pacientes["cohorte_analisis"], observed=False).sum()
    tabla = pd.DataFrame(filas).T
    tabla["Total"] = tabla.sum(axis=1)
    return tabla.astype(int)


# ---------------------------------------------------------------------------
# Orquestación

def limpiar(cfg: dict) -> dict[str, pd.DataFrame]:
    """Aplica todas las reglas en orden y devuelve las tablas limpias más el registro."""
    entradas: list[dict] = []
    nucleo = carga.cargar_nucleo(cfg, recodificar=False)
    nucleo = nucleo.merge(carga.cargar_columnas_completa(cfg, ["IH_PorcCMD"]), on="subject_id", how="left")

    pac, e = recodificar_pacientes(nucleo, cfg); entradas += e
    pac, e = anadir_edad(pac, cfg); entradas += e
    pac, e = asignar_cohorte_analisis(pac, cfg); entradas += e
    pac, e = banderas_pacientes(pac, cfg); entradas += e

    med = carga.medidas_larga(cfg, pac)
    med, e = limpiar_medidas(med, pac, cfg); entradas += e
    med, e = deducir_condicionadas(med, cfg); entradas += e
    img, e = medidas_imagen(cfg, pac); entradas += e
    img["excluida"] = img["dias_alta"] < 0
    img["motivo_exclusion"] = pd.Series(pd.NA, index=img.index, dtype="string").mask(img["excluida"], "dias_negativos")
    img["deducida"] = False
    entradas.append(entrada("R04", "Imagen con fecha anterior al alta", img.loc[img["excluida"], "registro"], "medidas"))
    medidas = pd.concat([med, img], ignore_index=True)
    medidas["registro"] = pd.Categorical(medidas["registro"], categories=list(cfg["registros"]), ordered=True)
    medidas = medidas.merge(pac[["subject_id", "cohorte_analisis"]], on="subject_id", how="left")

    pac, e = marcar_inconsistentes(pac, medidas[medidas["variable"].isin(cfg["variables_visita"])]); entradas += e
    vis_ten, e = visitas_tenacity(cfg, pac); entradas += e
    estado = estado_definicion(medidas, pac, cfg)
    pac = anadir_inclusion(pac, estado, cfg)

    registro = pd.DataFrame(entradas).fillna(0)
    cols_n = [c for c in registro.columns if c.startswith("N")]
    registro[cols_n] = registro[cols_n].astype(int)
    return {"pacientes": pac, "medidas": medidas, "estado_definicion": estado,
            "visitas_tenacity": vis_ten, "registro_limpieza": registro, "flujo_consort": flujo_consort(pac)}


def guardar(tablas: dict[str, pd.DataFrame], cfg: dict) -> Path:
    """Guarda las tablas en `rutas.procesados` (parquet). Devuelve la carpeta."""
    carpeta = carga.RAIZ / cfg["rutas"]["procesados"]
    carpeta.mkdir(parents=True, exist_ok=True)
    for nombre, tabla in tablas.items():
        tabla.to_parquet(carpeta / f"{nombre}.parquet", index=nombre == "flujo_consort")
    return carpeta
