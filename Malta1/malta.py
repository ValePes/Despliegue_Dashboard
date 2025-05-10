import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, confusion_matrix, accuracy_score, recall_score

# ----------------------------
#  Configuración de la página
# ----------------------------
st.set_page_config(
    page_title="Dashboard Multiciudades",
    page_icon=":sunny:",
    layout="wide",
    initial_sidebar_state="expanded"
)
# ----------------------------
#  Estilos neon para gráficas
# ----------------------------
st.markdown("""
<style>
.neon-chart .js-plotly-plot {
    border: 4px solid #39FF14 !important;
    box-shadow: 0 0 15px #39FF14 !important;
    border-radius: 8px !important;
}
</style>
""", unsafe_allow_html=True)

def render_neon_chart(fig):
    """Envuelve un Plotly-fig en un contenedor con clase .neon-chart"""
    st.markdown('<div class="neon-chart">', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------
#  Carga de datos
# ----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("Malta_limpio.csv")

df = load_data()
df.rename(columns={"region_name": "ciudad"}, inplace=True)
df["ciudad"] = df["ciudad"].str.strip()

# ----------------------------
#  Sidebar: Ciudad y Navegación
# ----------------------------
st.sidebar.title("Controles")
ciudades = df["ciudad"].unique().tolist()
ciudad_seleccionada = st.sidebar.selectbox("Elige ciudad", ciudades, key="select_ciudad")
pagina = st.sidebar.radio("Ir a", ["Inicio", "Univariado", "Predictivo"], key="radio_pagina")

df_city = df[df["ciudad"] == ciudad_seleccionada].copy()

# ============================
#  Página 1: Home / KPIs + Mapa
# ============================
def show_home(ciudad, df_city):
    st.header(f"📊 Resumen de {ciudad}")

    # — KPIs —
    num_cols = df_city.select_dtypes("number").columns.tolist()
    defaults = num_cols[:3] if len(num_cols) >= 3 else num_cols
    kpis = st.multiselect("Selecciona métricas a mostrar", num_cols, default=defaults)
    cols = st.columns(len(kpis) or 1)
    for metric, col in zip(kpis, cols):
        valor = round(df_city[metric].mean(), 2)
        col.metric(label=metric.capitalize(), value=valor)

    # — Mapa interactivo —
    st.markdown("---")
    st.subheader(f"🗺️ Mapa de {ciudad}")
    if {"latitude", "longitude"}.issubset(df_city.columns):
        df_mapa = df_city.dropna(subset=["latitude", "longitude"])
        fig = px.scatter_mapbox(
            df_mapa,
            lat="latitude",
            lon="longitude",
            hover_name="ciudad",
            hover_data=df_mapa.columns,
            zoom=10,
            height=500,
            color_discrete_sequence=["#002147"]
        )
        fig.update_layout(
            mapbox_style="open-street-map",
            margin={"l":0, "r":0, "t":0, "b":0}
        )
        render_neon_chart(fig)
    else:
        st.warning("No hay columnas latitude/longitude para mostrar el mapa.")

# ============================
#  Página 3: Univariado
# ============================
def show_univariate(ciudad, df_city):
    st.header(f"🔎 Análisis Univariado de {ciudad}")
    tipo = st.selectbox("Tipo de variable", ["Categórica", "Numérica"])
    if tipo == "Categórica":
        cats = df_city.select_dtypes("object").columns.tolist()
        sel = st.selectbox("Selecciona variable categórica", cats)
        vc = (df_city[sel].value_counts()
              .rename_axis(sel)
              .reset_index(name="count"))
        st.plotly_chart(px.bar(vc, x=sel, y="count", title=f"Frecuencia de {sel}"), use_container_width=True)
        st.plotly_chart(px.pie(vc, names=sel, values="count", title=f"Distribución de {sel}"), use_container_width=True)
        if st.checkbox("Mostrar tabla de frecuencias"):
            st.dataframe(vc)
    else:
        nums = df_city.select_dtypes("number").columns.tolist()
        sel = st.selectbox("Selecciona variable numérica", nums)
        st.plotly_chart(px.histogram(df_city, x=sel, nbins=30, title=f"Histograma de {sel}"), use_container_width=True)
        if st.checkbox("Mostrar boxplot"):
            st.plotly_chart(px.box(df_city, y=sel, title=f"Boxplot de {sel}"), use_container_width=True)

# ============================
#  Página 4: Predictivo
# ============================
def show_predictive(ciudad, df_city):
    st.header(f"🤖 Modelado Predictivo en {ciudad}")
    nums = df_city.select_dtypes("number").columns.tolist()
    cat_cols = df_city.select_dtypes("object").columns.tolist()

    features = st.multiselect("Variables predictoras", nums)
    if not features:
        st.info("Selecciona al menos una variable predictora.")
        return

    model_type = st.selectbox("Tipo de modelo", ["Regresión lineal", "Regresión logística"])

    if model_type == "Regresión logística":
        target = st.selectbox("Variable objetivo (categórica)", cat_cols)
    else:
        target = st.selectbox("Variable objetivo (numérica)", nums)

    if model_type == "Regresión lineal" and st.checkbox("Mostrar matriz de correlación"):
        corr = df_city[features + [target]].corr().round(2)
        fig_corr = px.imshow(corr, text_auto=True, aspect="auto",
                             color_continuous_scale="RdBu", title="Heatmap de correlaciones")
        render_neon_chart(fig_corr)

    if st.button("Ejecutar modelo"):
        X_train, X_test, y_train, y_test = train_test_split(df_city[features], df_city[target], random_state=0)
        if model_type == "Regresión lineal":
            model = LinearRegression().fit(X_train, y_train)
            y_pred = model.predict(X_test)
            st.metric("MSE", round(mean_squared_error(y_test, y_pred), 3))
            dfc = pd.DataFrame({"Real": y_test, "Predicho": y_pred})
            st.plotly_chart(px.scatter(dfc, x="Real", y="Predicho", title="Real vs Predicho"), use_container_width=True)
            st.dataframe(dfc)
            if st.checkbox("Mostrar heatmap de coeficientes"):
                coefs = pd.DataFrame(model.coef_, index=features, columns=["coef"])
                st.plotly_chart(px.imshow(coefs, text_auto=True, title="Coeficientes"), use_container_width=True)
        else:
            model = LogisticRegression(max_iter=1000).fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_true = y_test.values
            labels = np.unique(np.concatenate([y_true, y_pred]))
            cm = confusion_matrix(y_true, y_pred, labels=labels)
            df_cm = pd.DataFrame(cm, index=labels, columns=labels)
            fig_cm = px.imshow(df_cm, text_auto=True, aspect="auto",
                               color_continuous_scale="Blues",
                               labels={"x":"Predicho","y":"Real"},
                               title="Matriz de Confusión")
            render_neon_chart(fig_cm)
            st.metric("Exactitud", round(accuracy_score(y_test, y_pred), 3))
            st.metric("Sensibilidad", round(recall_score(y_test, y_pred, average="macro"), 3))

# ============================
#  Renderizado según página
# ============================

if pagina == "Inicio":
    show_home(ciudad_seleccionada, df_city)
elif pagina == "Univariado":
    show_univariate(ciudad_seleccionada, df_city)
else:
    show_predictive(ciudad_seleccionada, df_city)