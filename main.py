import os
import json
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from tensorflow import keras

# ------------------------------------------------------------------
# Rutas de los artefactos del modelo
# ------------------------------------------------------------------

CARPETA_MODELO = os.path.join(os.path.dirname(__file__), "modelo")

RUTA_MODELO = os.path.join(CARPETA_MODELO, "modelo_auditor_ips.keras")
RUTA_ESCALADOR = os.path.join(CARPETA_MODELO, "escalador_auditor_ips.pkl")
RUTA_CONFIGURACION = os.path.join(CARPETA_MODELO, "configuracion_auditor_ips.json")

# ------------------------------------------------------------------
# Cargar modelo, escalador y configuración UNA sola vez al iniciar
# ------------------------------------------------------------------

modelo_auditor = keras.models.load_model(RUTA_MODELO)
escalador_auditor = joblib.load(RUTA_ESCALADOR)

with open(RUTA_CONFIGURACION, "r", encoding="utf-8") as f:
    configuracion_auditor = json.load(f)

COLUMNAS_MODELO = configuracion_auditor["columnas_modelo"]
UMBRAL_REVISION = configuracion_auditor["umbral_revision_preventiva"]
UMBRAL_ALERTA = configuracion_auditor["umbral_generar_alerta"]
EDAD_MEDIANA = configuracion_auditor.get("edad_mediana", 40.0)

# ------------------------------------------------------------------
# App FastAPI
# ------------------------------------------------------------------

app = FastAPI(
    title="Auditor IPS API",
    description="API para auditoría automática de atenciones, historia clínica y prefactura",
    version="1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def salud():
    return {"status": "ok", "mensaje": "API Auditor IPS activa"}


def ejecutar_auditoria_ips(datos_atenciones, datos_historia, datos_prefactura):
    """Réplica exacta de la lógica de auditoría entrenada en el notebook."""
    try:
        atenciones_proceso = datos_atenciones.copy()
        historia_proceso = datos_historia.copy()
        prefactura_proceso = datos_prefactura.copy()

        atenciones_proceso["fecha_atencion"] = pd.to_datetime(
            atenciones_proceso["fecha_atencion"], errors="coerce"
        )
        historia_proceso["fecha_registro"] = pd.to_datetime(
            historia_proceso["fecha_registro"], errors="coerce"
        )
        prefactura_proceso["fecha_facturacion"] = pd.to_datetime(
            prefactura_proceso["fecha_facturacion"], errors="coerce"
        )

        historia_proceso["codigo_cups"] = (
            historia_proceso["codigo_cups"].astype("string").str.strip()
        )
        prefactura_proceso["codigo_cups_facturado"] = (
            prefactura_proceso["codigo_cups_facturado"].astype("string").str.strip()
        )

        historia_resumida = (
            historia_proceso
            .groupby(["id_atencion", "codigo_cups"], as_index=False, dropna=False)
            .agg(
                cantidad_realizada=("cantidad_realizada", "sum"),
                descripcion_clinica=("descripcion", "first"),
                soporte_clinico=("soporte_clinico", "first"),
                fecha_registro=("fecha_registro", "min"),
                id_detalle=("id_detalle", "first"),
            )
        )

        prefactura_resumida = (
            prefactura_proceso
            .groupby(["id_atencion", "codigo_cups_facturado"], as_index=False, dropna=False)
            .agg(
                cantidad_facturada=("cantidad_facturada", "sum"),
                descripcion_facturada=("descripcion_servicio_facturado", "first"),
                valor_unitario=("valor_unitario", "first"),
                valor_total=("valor_total", "sum"),
                fecha_facturacion=("fecha_facturacion", "min"),
                id_prefactura=("id_prefactura", "first"),
                id_paciente_prefactura=("id_paciente", "first"),
            )
        )

        resultado = historia_resumida.merge(
            prefactura_resumida,
            left_on=["id_atencion", "codigo_cups"],
            right_on=["id_atencion", "codigo_cups_facturado"],
            how="outer",
            indicator=True,
        )

        columnas_atencion = [
            "id_atencion", "id_paciente", "fecha_atencion", "tipo_atencion",
            "diagnostico_principal_cie10", "descripcion_diagnostico",
            "medico_tratante", "sede", "eps",
        ]

        resultado = resultado.merge(
            atenciones_proceso[columnas_atencion], on="id_atencion", how="left"
        )

        resultado["tiene_prefactura"] = resultado["_merge"].isin(["both", "right_only"]).astype(int)
        resultado["tiene_detalle_clinico"] = resultado["_merge"].isin(["both", "left_only"]).astype(int)
        resultado["codigo_coincide"] = resultado["_merge"].eq("both").astype(int)

        resultado["cantidad_realizada"] = pd.to_numeric(
            resultado["cantidad_realizada"], errors="coerce"
        ).fillna(0)
        resultado["cantidad_facturada"] = pd.to_numeric(
            resultado["cantidad_facturada"], errors="coerce"
        ).fillna(0)
        resultado["diferencia_cantidad"] = abs(
            resultado["cantidad_realizada"] - resultado["cantidad_facturada"]
        )
        resultado["cantidad_coincide"] = resultado["diferencia_cantidad"].eq(0).astype(int)

        resultado["tiene_soporte_clinico"] = (
            resultado["soporte_clinico"]
            .astype("string")
            .str.upper()
            .str.strip()
            .eq("SI")
            .fillna(False)
            .astype(int)
        )

        resultado["dias_hasta_facturacion"] = (
            resultado["fecha_facturacion"] - resultado["fecha_atencion"]
        ).dt.days
        resultado["dias_hasta_facturacion"] = resultado["dias_hasta_facturacion"].fillna(0)

        resultado["valor_unitario"] = pd.to_numeric(
            resultado["valor_unitario"], errors="coerce"
        ).fillna(0)
        resultado["valor_total"] = pd.to_numeric(
            resultado["valor_total"], errors="coerce"
        ).fillna(0)

        resultado["edad"] = EDAD_MEDIANA

        datos_modelo = resultado[COLUMNAS_MODELO].copy()
        for columna in datos_modelo.columns:
            datos_modelo[columna] = pd.to_numeric(datos_modelo[columna], errors="coerce")
        datos_modelo = datos_modelo.fillna(0).astype(float)

        datos_escalados = escalador_auditor.transform(datos_modelo)

        probabilidades = modelo_auditor.predict(datos_escalados, verbose=0).ravel()
        resultado["probabilidad_inconsistencia"] = (probabilidades * 100).round(2)

        resultado["regla_no_facturado"] = resultado["_merge"].eq("left_only")
        resultado["regla_facturado_sin_detalle"] = resultado["_merge"].eq("right_only")
        resultado["regla_cantidad_discordante"] = (
            resultado["_merge"].eq("both") & resultado["cantidad_coincide"].eq(0)
        )
        resultado["regla_fecha_invalida"] = resultado["dias_hasta_facturacion"] < 0

        resultado["regla_critica"] = resultado[[
            "regla_no_facturado", "regla_facturado_sin_detalle",
            "regla_cantidad_discordante", "regla_fecha_invalida",
        ]].any(axis=1)

        resultado["tipo_alerta"] = np.select(
            [
                resultado["regla_no_facturado"],
                resultado["regla_facturado_sin_detalle"],
                resultado["regla_cantidad_discordante"],
                resultado["regla_fecha_invalida"],
            ],
            ["NO_FACTURADO", "FACTURADO_SIN_REGISTRO_CLINICO", "CANTIDAD_DISCORDANTE", "FECHA_INVALIDA"],
            default="SIN_REGLA_CRITICA",
        )

        probabilidad_decimal = resultado["probabilidad_inconsistencia"] / 100

        resultado["decision_auditor"] = np.select(
            [
                resultado["regla_critica"],
                probabilidad_decimal >= UMBRAL_ALERTA,
                probabilidad_decimal >= UMBRAL_REVISION,
            ],
            ["GENERAR ALERTA", "GENERAR ALERTA", "REVISIÓN PREVENTIVA"],
            default="APROBAR",
        )

        resultado["nivel_riesgo"] = np.select(
            [
                resultado["decision_auditor"].eq("GENERAR ALERTA"),
                resultado["decision_auditor"].eq("REVISIÓN PREVENTIVA"),
            ],
            ["ALTO", "MEDIO"],
            default="BAJO",
        )

        def explicar_resultado(fila):
            motivos = []
            if fila["regla_no_facturado"]:
                motivos.append("Procedimiento realizado no incluido en la prefactura")
            if fila["regla_facturado_sin_detalle"]:
                motivos.append("Servicio facturado sin registro clínico equivalente")
            if fila["regla_cantidad_discordante"]:
                motivos.append("La cantidad realizada y la cantidad facturada no coinciden")
            if fila["regla_fecha_invalida"]:
                motivos.append("La fecha de facturación es anterior a la atención")
            if not motivos:
                if fila["decision_auditor"] == "GENERAR ALERTA":
                    motivos.append("La red neuronal detectó un riesgo elevado de inconsistencia")
                elif fila["decision_auditor"] == "REVISIÓN PREVENTIVA":
                    motivos.append("La red neuronal detectó un riesgo intermedio que requiere revisión")
                else:
                    motivos.append("No se detectaron inconsistencias administrativas evidentes")
            return "; ".join(motivos)

        resultado["motivo_alerta"] = resultado.apply(explicar_resultado, axis=1)

        tarifas_historicas_funcion = (
            prefactura_proceso
            .dropna(subset=["codigo_cups_facturado", "valor_unitario"])
            .groupby("codigo_cups_facturado", as_index=False)
            .agg(tarifa_estimada=("valor_unitario", "median"))
        )

        resultado = resultado.merge(
            tarifas_historicas_funcion,
            left_on="codigo_cups",
            right_on="codigo_cups_facturado",
            how="left",
            suffixes=("", "_tarifa"),
        )

        resultado["valor_potencial_no_facturado"] = np.where(
            resultado["tipo_alerta"].eq("NO_FACTURADO"),
            resultado["tarifa_estimada"].fillna(0) * resultado["cantidad_realizada"],
            0,
        )

        resultado["valor_en_riesgo_glosa"] = np.where(
            resultado["tipo_alerta"].isin([
                "FACTURADO_SIN_REGISTRO_CLINICO", "CANTIDAD_DISCORDANTE", "FECHA_INVALIDA",
            ]),
            resultado["valor_total"].fillna(0),
            0,
        )

        columnas_resultado = [
            "id_atencion", "id_paciente", "codigo_cups", "codigo_cups_facturado",
            "descripcion_clinica", "descripcion_facturada", "cantidad_realizada",
            "cantidad_facturada", "valor_unitario", "valor_total",
            "probabilidad_inconsistencia", "tipo_alerta", "nivel_riesgo",
            "decision_auditor", "motivo_alerta", "valor_potencial_no_facturado",
            "valor_en_riesgo_glosa",
        ]

        tabla_final = resultado[columnas_resultado].copy()

        orden_decision = {"GENERAR ALERTA": 1, "REVISIÓN PREVENTIVA": 2, "APROBAR": 3}
        tabla_final["orden"] = tabla_final["decision_auditor"].map(orden_decision)
        tabla_final = (
            tabla_final
            .sort_values(["orden", "probabilidad_inconsistencia"], ascending=[True, False])
            .drop(columns="orden")
            .reset_index(drop=True)
        )

        resumen = {
            "registros_procesados": len(tabla_final),
            "alertas": int((tabla_final["decision_auditor"] == "GENERAR ALERTA").sum()),
            "revisiones_preventivas": int((tabla_final["decision_auditor"] == "REVISIÓN PREVENTIVA").sum()),
            "aprobados": int((tabla_final["decision_auditor"] == "APROBAR").sum()),
            "valor_potencial_no_facturado": float(tabla_final["valor_potencial_no_facturado"].sum()),
            "valor_en_riesgo_glosa": float(tabla_final["valor_en_riesgo_glosa"].sum()),
        }

        return tabla_final, resumen

    except Exception as error:
        return pd.DataFrame(), {"error": str(error)}


@app.post("/auditar")
async def auditar(
    atenciones: UploadFile = File(..., description="CSV de atenciones"),
    historia_clinica: UploadFile = File(..., description="CSV de historia clínica"),
    prefactura: UploadFile = File(..., description="CSV de prefactura"),
):
    for archivo in (atenciones, historia_clinica, prefactura):
        if not archivo.filename.lower().endswith(".csv"):
            raise HTTPException(
                status_code=400, detail=f"{archivo.filename} debe ser un archivo .csv"
            )

    try:
        datos_atenciones = pd.read_csv(atenciones.file)
        datos_historia = pd.read_csv(historia_clinica.file)
        datos_prefactura = pd.read_csv(prefactura.file)
    except Exception as error:
        raise HTTPException(status_code=400, detail=f"Error leyendo los CSV: {error}")

    tabla_final, resumen = ejecutar_auditoria_ips(
        datos_atenciones, datos_historia, datos_prefactura
    )

    if "error" in resumen:
        raise HTTPException(status_code=500, detail=resumen["error"])

    tabla_final = tabla_final.replace({np.nan: None})

    return JSONResponse(content={
        "resumen": resumen,
        "resultados": tabla_final.to_dict(orient="records"),
        "generado_en": datetime.utcnow().isoformat(),
    })