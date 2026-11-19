import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

# ==========================================
# CONFIGURACIÓN DE LA PÁGINA
# ==========================================

st.set_page_config(
    page_title="Simulador de Variedades de Porotos",
    page_icon="🌱",
    layout="wide"
)

# ==========================================
# TOKEN DE PRIOR LABS
# ==========================================

if "TABPFN_TOKEN" in st.secrets:
    os.environ["TABPFN_TOKEN"] = st.secrets["TABPFN_TOKEN"]
else:
    st.error("⚠️ TABPFN_TOKEN no está configurado en Streamlit Secrets.")
    st.stop()


# ==========================================
# IMPORTAR TABPFN CLIENT
# ==========================================

try:
    import tabpfn_client
    from tabpfn_client import TabPFNClassifier

    # Entregar explícitamente el token al cliente
    tabpfn_client.set_access_token(
        st.secrets["TABPFN_TOKEN"]
    )

except Exception as e:
    st.error(
        f"Error al importar TabPFN Client: {e}"
    )
    st.stop()


# ==========================================
# TÍTULO
# ==========================================

st.title(
    "🌱 Herramienta de Apoyo Operacional y Clasificación de Porotos"
)

st.markdown("""
Esta aplicación permite analizar cómo cambian las probabilidades
predichas de las siete variedades de poroto
(*DERMASON, SIRA, SEKER, HOROZ, CALI, BARBUNYA y BOMBAY*)
a partir de la simulación de cinco características físicas.

La herramienta se basa en **TabPFN** y utiliza los mismos datos de
entrenamiento empleados en el análisis del proyecto.

La clase definida como crítica para el análisis operacional es **SIRA**.
""")


# ==========================================
# CARGAR DATOS Y MODELO
# ==========================================

@st.cache_resource
def cargar_modelo():

    try:

        # Cargar datos de entrenamiento
        X_train = joblib.load(
            "X_train.joblib"
        )

        y_train = joblib.load(
            "y_train.joblib"
        )

        # Cargar estadísticas del entrenamiento
        stats = joblib.load(
            "stats_train.joblib"
        )

        # Crear TabPFN usando la API de Prior Labs
        # n_estimators = 4 corresponde a la configuración
        # seleccionada en el proyecto.
        modelo = TabPFNClassifier(
            n_estimators=4,
            fit_mode="fit_with_cache"
        )

        # Guardar el conjunto de entrenamiento como contexto
        modelo.fit(
            X_train,
            y_train
        )

        return modelo, stats

    except Exception as e:

        st.error(
            f"Error al cargar o construir el modelo: {e}"
        )

        return None, None


modelo, stats = cargar_modelo()


# ==========================================
# CONTINUAR SOLO SI EL MODELO CARGÓ
# ==========================================

if modelo is not None and stats is not None:

    # Variables que puede modificar el usuario
    variables_simulador = [
        "Area",
        "Perimeter",
        "MajorAxisLength",
        "MinorAxisLength",
        "roundness"
    ]

    # ======================================
    # BARRA LATERAL
    # ======================================

    st.sidebar.header(
        "⚙️ Parámetros Físicos Ajustables"
    )

    st.sidebar.markdown("""
    Modifique las características físicas del poroto.

    Los límites corresponden a los percentiles
    **5 y 95 del conjunto de entrenamiento**.
    """)

    valores_usuario = {}

    for var in variables_simulador:

        p5 = stats[
            "percentiles_variables"
        ][var]["p5"]

        p50 = stats[
            "percentiles_variables"
        ][var]["p50"]

        p95 = stats[
            "percentiles_variables"
        ][var]["p95"]

        valores_usuario[var] = st.sidebar.slider(
            label=var,
            min_value=float(p5),
            max_value=float(p95),
            value=float(p50),
            step=(
                0.001
                if var == "roundness"
                else 1.0
            )
        )


    # ======================================
    # CREAR NUEVO CASO
    # ======================================

    datos_entrada = pd.DataFrame(
        [stats["valores_base"]],
        columns=stats["columnas_X"]
    )

    for var, valor in valores_usuario.items():
        datos_entrada[var] = valor


    # ======================================
    # PREDICCIÓN
    # ======================================

    st.markdown("---")

    with st.spinner(
        "Calculando probabilidades con TabPFN..."
    ):

        try:

            probabilidades = (
                modelo.predict_proba(
                    datos_entrada
                )[0]
            )

            clases = modelo.classes_

            dict_probs = {
                str(clase): float(prob)
                for clase, prob
                in zip(
                    clases,
                    probabilidades
                )
            }

            clase_ganadora = clases[
                np.argmax(probabilidades)
            ]

            prob_ganadora = float(
                np.max(probabilidades)
            )

            prob_sira = dict_probs.get(
                "SIRA",
                0.0
            )

        except Exception as e:

            st.error(
                f"Error durante la predicción: {e}"
            )

            st.stop()


    # ======================================
    # RESULTADOS PRINCIPALES
    # ======================================

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            label="🏆 Variedad predicha",
            value=str(clase_ganadora)
        )

    with col2:

        st.metric(
            label="🎯 Probabilidad máxima",
            value=f"{prob_ganadora:.2%}"
        )

    with col3:

        st.metric(
            label="⚠️ Probabilidad de SIRA",
            value=f"{prob_sira:.2%}"
        )


    # ======================================
    # DECISIÓN OPERACIONAL
    # ======================================

    st.subheader(
        "📋 Decisión operacional"
    )

    if prob_sira < 0.20:

        st.success(
            "🟢 **Control habitual**: "
            "la probabilidad estimada de SIRA "
            "es inferior al 20%."
        )

    elif prob_sira < 0.50:

        st.warning(
            "🟡 **Revisión preventiva**: "
            "la probabilidad estimada de SIRA "
            "se encuentra entre 20% y 50%."
        )

    else:

        st.error(
            "🔴 **Seguimiento preferente SIRA**: "
            "la probabilidad estimada de SIRA "
            "es igual o superior al 50%."
        )


    # ======================================
    # GRÁFICO DE PROBABILIDADES
    # ======================================

    st.markdown("---")

    st.subheader(
        "📊 Probabilidades de Clasificación"
    )

    df_grafico = pd.DataFrame({
        "Variedad": list(
            dict_probs.keys()
        ),
        "Probabilidad (%)": [
            valor * 100
            for valor
            in dict_probs.values()
        ]
    })

    df_grafico = df_grafico.sort_values(
        "Probabilidad (%)",
        ascending=True
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    colores = [
        "#d95f02"
        if variedad == "SIRA"
        else "#1f77b4"
        for variedad
        in df_grafico["Variedad"]
    ]

    ax.barh(
        df_grafico["Variedad"],
        df_grafico["Probabilidad (%)"],
        color=colores
    )

    ax.set_xlim(
        0,
        100
    )

    ax.set_xlabel(
        "Probabilidad (%)"
    )

    ax.set_ylabel(
        "Variedad"
    )

    ax.set_title(
        "Probabilidad estimada por variedad"
    )

    st.pyplot(
        fig
    )


    # ======================================
    # TABLA DE PROBABILIDADES
    # ======================================

    with st.expander(
        "Ver probabilidades completas"
    ):

        tabla_probabilidades = (
            df_grafico
            .sort_values(
                "Probabilidad (%)",
                ascending=False
            )
            .reset_index(drop=True)
        )

        st.dataframe(
            tabla_probabilidades,
            use_container_width=True
        )


    # ======================================
    # VENTANA DE REFERENCIA SIRA
    # ======================================

    st.markdown("---")

    st.subheader(
        "📏 Ventana de Referencia Operacional — SIRA"
    )

    st.markdown("""
    Los siguientes valores corresponden a los
    percentiles **5, 50 y 95** observados en las
    muestras de entrenamiento pertenecientes a
    la clase **SIRA**.

    Estos valores son descriptivos de los datos
    observados y no representan relaciones
    causales.
    """)

    tabla_sira = pd.DataFrame(
        stats["sira_percentiles"]
    ).rename(
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


    # ======================================
    # NOTA FINAL
    # ======================================

    st.info("""
    📌 **Nota de apoyo a las decisiones**

    Esta herramienta corresponde a un simulador
    académico de apoyo a la clasificación y
    priorización.

    Las predicciones deben complementarse con
    supervisión y validación técnica humana.
    """)

else:

    st.error(
        "No fue posible iniciar el simulador."
    )
