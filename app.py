import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

# ==========================================
# CONFIGURACIÓN DE STREAMLIT
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

    tabpfn_client.set_access_token(
        st.secrets["TABPFN_TOKEN"]
    )

except Exception as e:
    st.error(f"Error al importar TabPFN Client: {e}")
    st.stop()

# ==========================================
# TÍTULO
# ==========================================

st.title(
    "🌱 Herramienta de Apoyo Operacional y Clasificación de Porotos"
)

st.markdown("""
Esta aplicación permite analizar cómo cambian las probabilidades predichas
de las siete variedades de poroto:

**DERMASON, SIRA, SEKER, HOROZ, CALI, BARBUNYA y BOMBAY**

a partir de la modificación de cinco características físicas.

El modelo utilizado es **TabPFN ajustado**, seleccionado en el proyecto
a partir de su desempeño en validación cruzada.

La clase definida como crítica para el análisis operacional es **SIRA**.
""")

# ==========================================
# CARGAR DATOS Y CONSTRUIR MODELO
# ==========================================

@st.cache_resource
def cargar_modelo():

    try:

        # Datos utilizados en el entrenamiento original
        X_train = joblib.load("X_train.joblib")
        y_train = joblib.load("y_train.joblib")

        # Estadísticas utilizadas por el simulador
        stats = joblib.load("stats_train.joblib")

        # TabPFN ajustado
        modelo = TabPFNClassifier(
            n_estimators=4,
            balance_probabilities=False,
            random_state=2026,
            fit_mode="fit_with_cache"
        )

        # El entrenamiento se utiliza como contexto
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


with st.spinner("Cargando TabPFN y datos de entrenamiento..."):
    modelo, stats = cargar_modelo()


if modelo is None or stats is None:
    st.stop()

# ==========================================
# VARIABLES DEL SIMULADOR
# ==========================================

variables_simulador = [
    "Area",
    "Perimeter",
    "MajorAxisLength",
    "MinorAxisLength",
    "roundness"
]

st.sidebar.header(
    "⚙️ Parámetros físicos ajustables"
)

st.sidebar.markdown("""
Modifique las cinco características físicas para observar
cómo cambia la predicción del modelo.

Las demás variables permanecen en la **mediana del conjunto
de entrenamiento**.
""")

# ==========================================
# SLIDERS
# ==========================================

valores_usuario = {}

for variable in variables_simulador:

    p5 = stats["percentiles_variables"][variable]["p5"]
    p50 = stats["percentiles_variables"][variable]["p50"]
    p95 = stats["percentiles_variables"][variable]["p95"]

    if variable == "roundness":
        paso = 0.001
    else:
        paso = 1.0

    valores_usuario[variable] = st.sidebar.slider(
        label=variable,
        min_value=float(p5),
        max_value=float(p95),
        value=float(p50),
        step=paso
    )

# ==========================================
# UMBRAL OPERACIONAL
# ==========================================

st.sidebar.markdown("---")

st.sidebar.subheader(
    "🎯 Umbral de referencia"
)

umbral_sira = st.sidebar.slider(
    "Probabilidad máxima deseada de SIRA",
    min_value=0.10,
    max_value=0.90,
    value=0.50,
    step=0.05,
    format="%.2f"
)

st.sidebar.caption(
    "El 50% se utiliza como referencia operacional "
    "para la demostración y puede ajustarse según "
    "la capacidad y costos reales de inspección."
)

# ==========================================
# CREAR CASO WHAT-IF
# ==========================================

datos_entrada = pd.DataFrame(
    [stats["valores_base"]],
    columns=stats["columnas_X"]
)

for variable, valor in valores_usuario.items():
    datos_entrada[variable] = valor

# ==========================================
# PREDICCIÓN WHAT-IF
# ==========================================

try:

    with st.spinner(
        "Calculando probabilidades..."
    ):

        probabilidades = modelo.predict_proba(
            datos_entrada
        )[0]

    clases = modelo.classes_

    dict_probs = {
        str(clase): float(probabilidad)
        for clase, probabilidad
        in zip(clases, probabilidades)
    }

    clase_predicha = clases[
        np.argmax(probabilidades)
    ]

    prob_maxima = float(
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

# ==========================================
# RESULTADOS PRINCIPALES
# ==========================================

st.markdown("---")

st.subheader(
    "🔮 Resultado del escenario What-If"
)

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "🏆 Variedad predicha",
        str(clase_predicha)
    )

with col2:

    st.metric(
        "🎯 Probabilidad máxima",
        f"{prob_maxima:.2%}"
    )

with col3:

    st.metric(
        "⚠️ Probabilidad de SIRA",
        f"{prob_sira:.2%}"
    )

# ==========================================
# DECISIÓN OPERACIONAL
# ==========================================

st.subheader(
    "📋 Decisión operacional"
)

if prob_sira < 0.20:

    st.success(
        "🟢 **Control habitual:** "
        "la probabilidad estimada de SIRA es inferior al 20%."
    )

elif prob_sira < umbral_sira:

    st.warning(
        "🟡 **Revisión preventiva:** "
        "el caso presenta una probabilidad intermedia de SIRA."
    )

else:

    st.error(
        "🔴 **Seguimiento preferente:** "
        "la probabilidad estimada de SIRA supera "
        "el umbral operacional definido."
    )

# ==========================================
# GRÁFICO DE PROBABILIDADES
# ==========================================

st.markdown("---")

st.subheader(
    "📊 Probabilidades de clasificación por variedad"
)

df_probabilidades = pd.DataFrame({
    "Variedad": list(dict_probs.keys()),
    "Probabilidad (%)": [
        prob * 100
        for prob in dict_probs.values()
    ]
})

df_probabilidades = df_probabilidades.sort_values(
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
    for variedad in df_probabilidades["Variedad"]
]

ax.barh(
    df_probabilidades["Variedad"],
    df_probabilidades["Probabilidad (%)"],
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
    "Probabilidad estimada por TabPFN"
)

st.pyplot(fig)

with st.expander(
    "Ver probabilidades completas"
):

    tabla_probabilidades = (
        df_probabilidades
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

# ==========================================
# FUNCIÓN PARA ENCONTRAR INTERVALOS
# ==========================================

def encontrar_intervalos(
    valores,
    probabilidades_sira,
    umbral
):

    cumple = probabilidades_sira < umbral

    intervalos = []
    inicio = None

    for i, valor in enumerate(valores):

        if cumple[i] and inicio is None:
            inicio = valor

        if inicio is not None:

            ultimo = i == len(valores) - 1

            if not cumple[i] or ultimo:

                if cumple[i] and ultimo:
                    fin = valor
                else:
                    fin = valores[i - 1]

                intervalos.append(
                    (inicio, fin)
                )

                inicio = None

    return intervalos

# ==========================================
# RANGO JUSTIFICADO POR EL MODELO
# ==========================================

@st.cache_data(
    show_spinner=False
)
def calcular_rangos_modelo(
    _modelo,
    _stats,
    _variables,
    umbral
):

    filas = []
    metadata = []

    # Se prueban distintos valores para cada variable
    # manteniendo las demás en la mediana.

    for variable in _variables:

        p5 = _stats[
            "percentiles_variables"
        ][variable]["p5"]

        p95 = _stats[
            "percentiles_variables"
        ][variable]["p95"]

        valores_prueba = np.linspace(
            p5,
            p95,
            20
        )

        for valor in valores_prueba:

            caso = _stats[
                "valores_base"
            ].copy()

            caso[variable] = valor

            filas.append(
                caso
            )

            metadata.append(
                (variable, valor)
            )

    df_escenarios = pd.DataFrame(
        filas,
        columns=_stats["columnas_X"]
    )

    # Una sola consulta al modelo para todos
    # los escenarios.
    probabilidades = (
        _modelo.predict_proba(
            df_escenarios
        )
    )

    clases = list(
        _modelo.classes_
    )

    indice_sira = clases.index(
        "SIRA"
    )

    prob_sira_todos = (
        probabilidades[
            :,
            indice_sira
        ]
    )

    resultados = []

    posicion = 0

    for variable in _variables:

        p5 = _stats[
            "percentiles_variables"
        ][variable]["p5"]

        p95 = _stats[
            "percentiles_variables"
        ][variable]["p95"]

        valores_prueba = np.linspace(
            p5,
            p95,
            20
        )

        n = len(
            valores_prueba
        )

        probs_variable = (
            prob_sira_todos[
                posicion:
                posicion + n
            ]
        )

        posicion += n

        intervalos = encontrar_intervalos(
            valores_prueba,
            probs_variable,
            umbral
        )

        if len(intervalos) == 0:

            texto_intervalos = (
                "No se encontró un intervalo "
                "bajo el umbral"
            )

        else:

            textos = []

            for inicio, fin in intervalos:

                textos.append(
                    f"{inicio:.4f} – {fin:.4f}"
                )

            texto_intervalos = "; ".join(
                textos
            )

        resultados.append({
            "Variable": variable,
            "Intervalo(s) con menor riesgo": texto_intervalos,
            "Criterio": f"P(SIRA) < {umbral:.0%}",
            "P(SIRA) mínima": float(
                np.min(probs_variable)
            ),
            "P(SIRA) máxima": float(
                np.max(probs_variable)
            )
        })

    return pd.DataFrame(
        resultados
    )

# ==========================================
# MOSTRAR RANGO JUSTIFICADO
# ==========================================

st.markdown("---")

st.subheader(
    "🎯 Rango de referencia justificado por el modelo"
)

st.markdown("""
Para construir este rango se modifica **una variable a la vez**
entre sus percentiles 5 y 95 del conjunto de entrenamiento.

Las otras características permanecen en sus valores medianos.

Para cada escenario se calcula directamente la probabilidad de
la clase crítica **SIRA** mediante `predict_proba()`.

Se consideran favorables los intervalos donde la probabilidad
estimada de SIRA permanece bajo el umbral operacional seleccionado.
""")

try:

    with st.spinner(
        "Calculando rangos con TabPFN..."
    ):

        tabla_rangos_modelo = (
            calcular_rangos_modelo(
                modelo,
                stats,
                variables_simulador,
                umbral_sira
            )
        )

    st.dataframe(
        tabla_rangos_modelo,
        use_container_width=True,
        hide_index=True
    )

    st.info("""
    📌 **Interpretación:** estos intervalos son obtenidos
    directamente de las probabilidades entregadas por TabPFN.

    No indican causalidad. Representan escenarios en los que,
    manteniendo las demás variables en su mediana, el modelo
    estima una probabilidad de SIRA inferior al umbral definido.
    """)

except Exception as e:

    st.warning(
        f"No fue posible calcular los rangos del modelo: {e}"
    )

# ==========================================
# RANGO DESCRIPTIVO REAL DE SIRA
# ==========================================

st.markdown("---")

st.subheader(
    "📏 Rango descriptivo observado de la clase SIRA"
)

st.markdown("""
Esta segunda tabla es diferente a la anterior.

Aquí se muestran solamente los percentiles **5, 50 y 95**
observados en los datos de entrenamiento cuya clase real es
**SIRA**.

Por lo tanto, esta tabla describe los datos históricos,
pero no representa por sí sola una recomendación operacional.
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

# ==========================================
# LIMITACIONES
# ==========================================

st.markdown("---")

st.subheader(
    "⚠️ Limitaciones de la simulación"
)

st.markdown("""
- La simulación modifica **una variable a la vez**.
- Las variables restantes permanecen en la mediana del entrenamiento.
- Los rangos obtenidos representan asociaciones aprendidas por el modelo,
  no relaciones de causa y efecto.
- El umbral operacional debe ajustarse posteriormente a la capacidad real
  de inspección y a los costos de los errores.
- La decisión final debe complementarse con supervisión técnica humana.
""")

# ==========================================
# NOTA FINAL
# ==========================================

st.info("""
📌 **Uso académico**

Esta aplicación corresponde a una herramienta de apoyo a la
clasificación y priorización.

El simulador permite analizar escenarios What-If y observar
directamente cómo cambian las probabilidades entregadas por TabPFN.
""")
