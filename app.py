import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

# Configurar página de Streamlit
st.set_page_config(
    page_title="Simulador de Variedades de Porotos",
    layout="wide"
)

# Título e Introducción en español
st.title("🌱 Herramienta de Apoyo Operacional y Clasificación de Porotos")
st.markdown("""
Esta aplicación permite analizar cómo cambian las probabilidades predichas de las siete variedades de poroto
(*DERMASON*, *SIRA*, *SEKER*, *HOROZ*, *CALI*, *BARBUNYA* y *BOMBAY*) a partir de la simulación de cinco características físicas clave.
La predicción utiliza el modelo **TabPFN Ajustado** obtenido del análisis del Hito 1.
""")

# Cargar token de TabPFN desde Streamlit Secrets
if "TABPFN_TOKEN" in st.secrets:
    os.environ["TABPFN_TOKEN"] = st.secrets["TABPFN_TOKEN"]
elif "TABPFN_TOKEN" not in os.environ:
    st.error("⚠️ TABPFN_TOKEN no configurado.")
    st.stop()
    
# Carga de recursos con manejo de errores
from tabpfn_client import TabPFNClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


@st.cache_resource
def cargar_artefactos():

    try:

        # Cargar datos de entrenamiento
        X_train = joblib.load("X_train.joblib")
        y_train = joblib.load("y_train.joblib")

        # Cargar estadísticas utilizadas por el simulador
        stats = joblib.load("stats_train.joblib")

        # Reconstruir el modelo TabPFN ajustado
        modelo = Pipeline(steps=[
            ("imputador", SimpleImputer(strategy="median")),
            ("modelo", TabPFNClassifier(
                n_estimators=4,
                balance_probabilities=False,
                random_state=2026
            ))
        ])

        # Ajustar usando los mismos datos de entrenamiento
        modelo.fit(X_train, y_train)

        return modelo, stats

    except Exception as e:

        st.error(
            f"Error al cargar o construir el modelo: {e}"
        )

        return None, None


modelo, stats = cargar_artefactos()

if modelo is not None and stats is not None:
    # Variables del simulador especificadas en el notebook
    variables_simulador = ["Area", "Perimeter", "MajorAxisLength", "MinorAxisLength", "roundness"]

    # Menú lateral para controles del simulador
    st.sidebar.header("⚙️ Parámetros Físicos Ajustables")
    st.sidebar.markdown("Defina los límites usando los percentiles 5 y 95 de entrenamiento.")

    valores_usuario = {}
    for var in variables_simulador:
        p5 = stats["percentiles_variables"][var]["p5"]
        p50 = stats["percentiles_variables"][var]["p50"]
        p95 = stats["percentiles_variables"][var]["p95"]

        # Crear slider con valor inicial mediana (p50) y límites p5 a p95
        valores_usuario[var] = st.sidebar.slider(
            label=f"{var}",
            min_value=float(p5),
            max_value=float(p95),
            value=float(p50),
            step=0.01 if var == "roundness" else 1.0
        )

    # Construir dataframe respetando el formato, orden y esquema de X_train
    datos_entrada = pd.DataFrame([stats["valores_base"]], columns=stats["columnas_X"])
    for var, val in valores_usuario.items():
        datos_entrada[var] = val

    # Sección principal
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("🔮 Predicciones del Modelo")
        try:
            # Obtener probabilidades estimadas
            probabilidades = modelo.predict_proba(datos_entrada)[0]
            clases = modelo.classes_

            # Crear diccionario ordenado de resultados
            dict_probs = {clases[i]: probabilidades[i] for i in range(len(clases))}
            clase_ganadora = clases[np.argmax(probabilidades)]
            prob_ganadora = max(probabilidades)

            # Mostrar clase ganadora con color de énfasis
            st.metric(label="🏆 Variedad de Poroto Predicha", value=str(clase_ganadora))

            prob_sira = dict_probs.get("SIRA", 0.0)

            # Destacar visualmente la probabilidad específica de la clase crítica SIRA
            st.markdown(f"### 📢 Probabilidad de Clase Crítica (SIRA): **{prob_sira * 100:.2f}%**")

            # Criterios operacionales propuestos para SIRA
            if prob_sira < 0.20:
                st.success("🟢 Nivel de seguimiento recomendado: **Control habitual** (Probabilidad inferior al 20%)")
            elif prob_sira < 0.50:
                st.warning("🟡 Nivel de seguimiento recomendado: **Revisión preventiva** (Probabilidad entre 20% y 50%)")
            else:
                st.error("🔴 Nivel de seguimiento recomendado: **Seguimiento preferente SIRA** (Probabilidad superior o igual al 50%)")

        except Exception as e:
            st.error(f"Error durante el proceso de inferencia del modelo: {e}")

    with col2:
        st.subheader("📊 Probabilidades de Clasificación por Clase")
        if 'dict_probs' in locals():
            df_grafico = pd.DataFrame({
                "Variedad": list(dict_probs.keys()),
                "Probabilidad (%)": [val * 100 for val in dict_probs.values()]
            }).sort_values("Probabilidad (%)", ascending=True)

            fig, ax = plt.subplots(figsize=(6, 4))
            colores = ["#d95f02" if x == "SIRA" else "#1f77b4" for x in df_grafico["Variedad"]]
            ax.barh(df_grafico["Variedad"], df_grafico["Probabilidad (%)"], color=colores)
            ax.set_xlim(0, 100)
            ax.set_xlabel("Probabilidad (%)")
            st.pyplot(fig)

    # Sección Ventana de Operación Recomendada
    st.markdown("---")
    st.subheader("📏 Ventana de Operación de Referencia Operacional (Clase Crítica SIRA)")
    st.markdown("""
    Los siguientes valores representan los percentiles 5, 50 y 95 extraídos exclusivamente de las muestras de entrenamiento etiquetadas
    como variedad **SIRA**. Estos rangos son **descriptivos de los datos observados** y no deben asumirse como recomendaciones de
    modificación o rangos de causalidad garantizados.
    """)

    tabla_sira_p = pd.DataFrame(stats["sira_percentiles"]).rename(
        index={"p5": "Mínimo (Percentil 5)", "p50": "Mediana (Percentil 50)", "p95": "Máximo (Percentil 95)"}
    )
    st.dataframe(tabla_sira_p.round(4))

    st.info("""
    📌 **Nota de Apoyo a las Decisiones:** Esta aplicación constituye una herramienta digital de exploración y priorización para agilizar
    los turnos de calidad empresarial. Sus predicciones no aseguran relaciones de causa-efecto directas y deben complementarse siempre
    con la validación técnica humana y de laboratorio.
    """)
