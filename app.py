import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import tabpfn_client
from tabpfn_client import TabPFNClassifier

# CONFIGURACIÓN
st.set_page_config(
    page_title="Simulador de Variedades de Porotos",
    page_icon="🌱",
    layout="wide"
)

# TOKEN
if "TABPFN_TOKEN" not in st.secrets:
    st.error("TABPFN_TOKEN no está configurado.")
    st.stop()

os.environ["TABPFN_TOKEN"] = st.secrets["TABPFN_TOKEN"]
tabpfn_client.set_access_token(st.secrets["TABPFN_TOKEN"])

# TÍTULO
st.title("🌱 Simulador What-If de Clasificación de Porotos")
st.markdown("""
Modelo: **TabPFN ajustado** | Clase crítica: **SIRA**

Modifique las características físicas del poroto para observar cómo cambian
las probabilidades de clasificación.
""")

# CARGAR MODELO
@st.cache_resource
def cargar_modelo():
    X_train = joblib.load("X_train.joblib")
    y_train = joblib.load("y_train.joblib")
    stats = joblib.load("stats_train.joblib")

    modelo = TabPFNClassifier(
        n_estimators=4,
        balance_probabilities=False,
        random_state=2026,
        fit_mode="fit_with_cache"
    )

    modelo.fit(X_train, y_train)
    return modelo, stats

with st.spinner("Cargando modelo..."):
    modelo, stats = cargar_modelo()

# VARIABLES
variables = [
    "Area",
    "Perimeter",
    "MajorAxisLength",
    "MinorAxisLength",
    "roundness"
]

st.sidebar.header("⚙️ Variables ajustables")
st.sidebar.caption(
    "Las demás variables permanecen en la mediana del entrenamiento."
)

valores_usuario = {}

for variable in variables:
    p5 = stats["percentiles_variables"][variable]["p5"]
    p50 = stats["percentiles_variables"][variable]["p50"]
    p95 = stats["percentiles_variables"][variable]["p95"]

    valores_usuario[variable] = st.sidebar.slider(
        variable,
        float(p5),
        float(p95),
        float(p50),
        step=0.001 if variable == "roundness" else 1.0
    )

# UMBRAL SIRA
umbral_sira = st.sidebar.slider(
    "Umbral de P(SIRA)",
    0.10,
    0.90,
    0.50,
    0.05
)

# CREAR CASO
nuevo_caso = pd.DataFrame(
    [stats["valores_base"]],
    columns=stats["columnas_X"]
)

for variable, valor in valores_usuario.items():
    nuevo_caso[variable] = valor

# PREDICCIÓN
probabilidades = modelo.predict_proba(nuevo_caso)[0]
clases = modelo.classes_

clase_predicha = clases[np.argmax(probabilidades)]
prob_maxima = np.max(probabilidades)
indice_sira = list(clases).index("SIRA")
prob_sira = probabilidades[indice_sira]

# RESULTADOS
st.subheader("🔮 Resultado What-If")

col1, col2, col3 = st.columns(3)

col1.metric("🏆 Variedad predicha", str(clase_predicha))
col2.metric("🎯 Probabilidad máxima", f"{prob_maxima:.2%}")
col3.metric("⚠️ Probabilidad SIRA", f"{prob_sira:.2%}")

# DECISIÓN
if prob_sira < 0.20:
    st.success("🟢 Control habitual")
elif prob_sira < umbral_sira:
    st.warning("🟡 Revisión preventiva")
else:
    st.error("🔴 Seguimiento preferente SIRA")

# PROBABILIDADES
st.subheader("📊 Probabilidades por variedad")

df_probabilidades = pd.DataFrame({
    "Variedad": clases,
    "Probabilidad (%)": probabilidades * 100
}).set_index("Variedad")

st.bar_chart(df_probabilidades)

# RANGO JUSTIFICADO POR EL MODELO
st.subheader("🎯 Rango de referencia según el modelo")

st.caption(
    "Se modifica una variable a la vez entre P5 y P95, "
    "manteniendo las demás en su mediana. "
    "Se buscan valores donde P(SIRA) sea menor al umbral."
)

rangos = []

for variable in variables:
    p5 = stats["percentiles_variables"][variable]["p5"]
    p95 = stats["percentiles_variables"][variable]["p95"]
    valores_prueba = np.linspace(p5, p95, 10)

    escenarios = []

    for valor in valores_prueba:
        caso = stats["valores_base"].copy()
        caso[variable] = valor
        escenarios.append(caso)

    df_escenarios = pd.DataFrame(
        escenarios,
        columns=stats["columnas_X"]
    )

    probs = modelo.predict_proba(df_escenarios)
    probs_sira = probs[:, indice_sira]
    favorables = valores_prueba[probs_sira < umbral_sira]

    if len(favorables) > 0:
        rango = f"{favorables.min():.3f} - {favorables.max():.3f}"
    else:
        rango = "Sin rango bajo el umbral"

    rangos.append({
        "Variable": variable,
        "Rango favorable": rango,
        "Criterio": f"P(SIRA) < {umbral_sira:.0%}"
    })

st.dataframe(
    pd.DataFrame(rangos),
    hide_index=True,
    use_container_width=True
)

# RANGO HISTÓRICO DE SIRA
st.subheader("📏 Rango descriptivo observado de SIRA")

tabla_sira = pd.DataFrame(stats["sira_percentiles"]).rename(
    index={
        "p5": "Percentil 5",
        "p50": "Mediana",
        "p95": "Percentil 95"
    }
)

st.dataframe(
    tabla_sira.round(4),
    use_container_width=True
)

# LIMITACIÓN
st.info(
    "Los rangos son referencias obtenidas del modelo y no representan "
    "relaciones causales. La decisión final requiere validación humana."
)
