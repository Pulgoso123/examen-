import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

from tabpfn_client import TabPFNClassifier
import tabpfn_client


# ==========================================
# CONFIGURACIÓN
# ==========================================

st.set_page_config(
    page_title="Simulador de Porotos",
    page_icon="🌱",
    layout="wide"
)

st.title("🌱 Simulador What-If de Clasificación de Porotos")

st.write(
    "Modelo seleccionado: **TabPFN ajustado** | "
    "Clase crítica: **SIRA**"
)


# ==========================================
# TOKEN PRIOR LABS
# ==========================================

if "TABPFN_TOKEN" not in st.secrets:
    st.error("TABPFN_TOKEN no configurado.")
    st.stop()

os.environ["TABPFN_TOKEN"] = st.secrets["TABPFN_TOKEN"]

tabpfn_client.set_access_token(
    st.secrets["TABPFN_TOKEN"]
)


# ==========================================
# CARGAR MODELO Y DATOS
# ==========================================

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


# ==========================================
# VARIABLES DEL SIMULADOR
# ==========================================

variables = [
    "Area",
    "Perimeter",
    "MajorAxisLength",
    "MinorAxisLength",
    "roundness"
]

st.sidebar.header("⚙️ Variables")

valores = {}

for variable in variables:

    p5 = stats["percentiles_variables"][variable]["p5"]
    p50 = stats["percentiles_variables"][variable]["p50"]
    p95 = stats["percentiles_variables"][variable]["p95"]

    valores[variable] = st.sidebar.slider(
        variable,
        min_value=float(p5),
        max_value=float(p95),
        value=float(p50)
    )


# ==========================================
# CREAR CASO WHAT-IF
# ==========================================

nuevo_caso = pd.DataFrame(
    [stats["valores_base"]],
    columns=stats["columnas_X"]
)

for variable, valor in valores.items():
    nuevo_caso[variable] = valor


# ==========================================
# PREDICCIÓN
# ==========================================

probabilidades = modelo.predict_proba(
    nuevo_caso
)[0]

clases = modelo.classes_

resultado = pd.DataFrame({
    "Variedad": clases,
    "Probabilidad": probabilidades
})

resultado["Probabilidad (%)"] = (
    resultado["Probabilidad"] * 100
)

clase_predicha = clases[
    np.argmax(probabilidades)
]

indice_sira = list(clases).index("SIRA")

prob_sira = probabilidades[
    indice_sira
]


# ==========================================
# RESULTADOS
# ==========================================

col1, col2 = st.columns(2)

with col1:

    st.metric(
        "🏆 Clase predicha",
        clase_predicha
    )

with col2:

    st.metric(
        "⚠️ Probabilidad de SIRA",
        f"{prob_sira:.2%}"
    )


st.subheader("📊 Probabilidades por variedad")

st.bar_chart(
    resultado.set_index("Variedad")[
        "Probabilidad (%)"
    ]
)


# ==========================================
# DECISIÓN OPERACIONAL
# ==========================================

st.subheader("📋 Decisión operacional")

umbral = 0.50

if prob_sira < umbral:

    st.success(
        "Probabilidad de SIRA bajo el umbral de referencia."
    )

else:

    st.error(
        "Caso prioritario para revisión: "
        "P(SIRA) ≥ 50%."
    )


# ==========================================
# RANGO JUSTIFICADO POR EL MODELO
# ==========================================

st.markdown("---")

st.subheader(
    "🎯 Rango de referencia según el modelo"
)

st.write(
    "Se modifica una variable a la vez entre sus percentiles "
    "5 y 95. Las demás se mantienen en la mediana. "
    "Se consideran favorables los valores donde "
    "**P(SIRA) < 50%**."
)

rangos = []

for variable in variables:

    p5 = stats[
        "percentiles_variables"
    ][variable]["p5"]

    p95 = stats[
        "percentiles_variables"
    ][variable]["p95"]

    valores_prueba = np.linspace(
        p5,
        p95,
        10
    )

    escenarios = []

    for valor in valores_prueba:

        caso = stats[
            "valores_base"
        ].copy()

        caso[variable] = valor

        escenarios.append(caso)

    df_escenarios = pd.DataFrame(
        escenarios,
        columns=stats["columnas_X"]
    )

    probs = modelo.predict_proba(
        df_escenarios
    )

    probs_sira = probs[
        :,
        indice_sira
    ]

    favorables = valores_prueba[
        probs_sira < umbral
    ]

    if len(favorables) > 0:

        minimo = favorables.min()
        maximo = favorables.max()

        rango = (
            f"{minimo:.3f} - {maximo:.3f}"
        )

    else:

        rango = "Sin rango bajo el umbral"

    rangos.append({
        "Variable": variable,
        "Rango favorable aproximado": rango,
        "Criterio": "P(SIRA) < 50%"
    })


tabla_rangos = pd.DataFrame(
    rangos
)

st.dataframe(
    tabla_rangos,
    hide_index=True,
    use_container_width=True
)


# ==========================================
# LIMITACIÓN
# ==========================================

st.info(
    "Los rangos son una referencia del modelo. "
    "Se modifica una variable a la vez y no representan "
    "una relación causal. La decisión final requiere "
    "validación humana."
)
