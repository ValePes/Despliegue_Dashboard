import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from streamlit.components.v1 import html
from sklearn.metrics import (
    mean_squared_error,
    confusion_matrix,
    accuracy_score,
    recall_score,
    classification_report
)

# =============================================
# CONFIGURACIÓN INICIAL MEJORADA
# =============================================
st.set_page_config(
    page_title="Malta Analytics Dashboard",
    page_icon="🏝️",
    layout="wide",
    initial_sidebar_state="expanded"
)

def preprocess_data(df):
    """Convierte columnas comunes a formato numérico"""
    df_processed = df.copy()

    # Booleanos
    bool_cols = ['host_is_superhost', 'host_identity_verified', 'instant_bookable', 'has_availability']
    for col in bool_cols:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].map({'t': 1, 'f': 0})

    # Porcentajes
    if 'host_response_rate' in df_processed.columns:
        # 1) Quitamos el símbolo '%'
        # 2) Convertimos a numérico forzando errores → NaN
        df_processed['host_response_rate'] = pd.to_numeric(
            df_processed['host_response_rate']
                .str.replace('%', '', regex=False),
            errors='coerce'
        )
        # 3) (Opcional) Rellenar NaN con 0
        df_processed['host_response_rate'].fillna(0, inplace=True)

    return df_processed


# =============================================
# ESTILOS AVANZADOS CON GRADIENTES Y TRANSICIONES
# =============================================
st.markdown("""
<style>
:root {
    --primary: #FF7F50;
    --secondary: #4682B4;
    --dark: #2c3e50;
    --light: #f8f9fa;
    --success: #28a745;
}

.stApp {
    background: linear-gradient(135deg, #f5f7fa 0%, #e4e8eb 100%);
    font-family: 'Segoe UI', 'Roboto', sans-serif;
}

header[data-testid="stHeader"] {
    background: linear-gradient(90deg, var(--primary) 0%, var(--secondary) 100%);
    box-shadow: 0 4px 12px 0 rgba(0,0,0,0.1);
}

/* Sidebar mejorado */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #ffffff 0%, #f8f9fa 100%);
    border-right: 1px solid rgba(0,0,0,0.1);
}

/* Títulos */
h1, h2, h3, h4 {
    color: var(--dark);
    font-weight: 600;
    margin-bottom: 0.5rem;
}

h1 {
    border-bottom: 2px solid var(--primary);
    padding-bottom: 0.3rem;
}

/* Tarjetas y métricas */
.stMetric {
    border-left: 4px solid var(--primary);
    padding-left: 1rem;
    transition: all 0.3s ease;
}

.stMetric:hover {
    transform: translateY(-3px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}

/* Botones */
.stButton>button {
    border-radius: 8px;
    background: linear-gradient(90deg, var(--primary) 0%, var(--secondary) 100%);
    color: white;
    border: none;
    transition: all 0.3s;
}

.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
}

/* Gráficos con efecto vidrio */
.glass-card {
    background: rgba(255, 255, 255, 0.8);
    backdrop-filter: blur(10px);
    border-radius: 12px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    border: 1px solid rgba(255,255,255,0.3);
    transition: all 0.3s ease;
}

.glass-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 8px 25px rgba(0,0,0,0.15);
}

/* Animaciones */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

.animated {
    animation: fadeIn 0.6s ease forwards;
}

/* Tabs mejorados */
.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0 !important;
    padding: 10px 20px !important;
    transition: all 0.3s;
}

.stTabs [aria-selected="true"] {
    background-color: var(--primary) !important;
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

# =============================================
# FUNCIONES UTILITARIAS MEJORADAS
# =============================================
def render_glass_card(content):
    html(f"""
    <div class="glass-card animated">
        {content}
    </div>
    """)

def render_neon_chart(fig, title=None):
    if title:
        st.markdown(f"<h3 style='color: #2c3e50;'>{title}</h3>", unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True, 
                   config={'displayModeBar': True, 'scrollZoom': True})

def create_metric_card(title, value, delta=None, icon="📊"):
    return f"""
    <div class="stMetric glass-card">
        <div style="display: flex; align-items: center; gap: 10px;">
            <span style="font-size: 24px;">{icon}</span>
            <div>
                <h4 style="margin: 0; color: #6c757d;">{title}</h4>
                <h2 style="margin: 0; color: var(--dark);">{value}</h2>
                {f'<span style="color: var(--success);">{delta}</span>' if delta else ''}
            </div>
        </div>
    </div>
    """

# =============================================
# CARGA DE DATOS CON MEJOR MANEJO DE ERRORES
# =============================================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("Malta_limpio.csv")
        df.rename(columns={"region_name": "ciudad"}, inplace=True)
        df["ciudad"] = df["ciudad"].str.strip()
        return df
    except Exception as e:
        st.error(f"Error al cargar los datos: {str(e)}")
        return pd.DataFrame()

df = load_data()
df = preprocess_data(df)

# =============================================
# SIDEBAR MEJORADO
# =============================================
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 style="color: var(--primary);">🏝️ Malta Analytics</h1>
        <p style="color: #6c757d;">Exploración de datos avanzada</p>
    </div>
    """, unsafe_allow_html=True)

    ciudades = df["ciudad"].unique().tolist() if not df.empty else []
    ciudad_seleccionada = st.selectbox(
        "📍 Selecciona una ciudad", 
        ciudades, 
        key="select_ciudad",
        help="Selecciona la ciudad para filtrar los datos"
    )

    st.markdown("---")

    pagina = st.radio(
        "🔍 Navegación", 
        ["Inicio", "Modelado Explicativo", "Modelado Predictivo"], 
        key="radio_pagina",
        index=0
    )

    if "year" in df.columns:
        st.markdown("---")
        years = sorted(df["year"].dropna().unique())
        year_selected = st.select_slider(
            "📅 Filtrar por año", 
            options=years,
            help="Selecciona un año específico para filtrar los datos"
        )
        df = df[df["year"] == year_selected]

# Filtramos datos por ciudad seleccionada
df_city = df[df["ciudad"] == ciudad_seleccionada].copy() if not df.empty else pd.DataFrame()

# =============================================
# PÁGINA DE INICIO MEJORADA
# =============================================
def show_home(ciudad, df_city):
    st.title(f"🌴 Análisis de {ciudad}")

    # Sección de introducción con video
    col1, col2 = st.columns([1, 2])
    with col1:
        with st.container():
            st.video("https://www.youtube.com/watch?v=2kjoru5p0e4")
    with col2:
        render_glass_card("""
        <h2>Bienvenido/a al Dashboard de Malta</h2>
        <p>Explora datos turísticos, demográficos y económicos de las principales ciudades de Malta. 
        Utiliza las herramientas interactivas para descubrir patrones y tendencias.</p>
        """)

    st.markdown("---")

    # Indicadores Clave mejorados
    st.subheader("📊 Indicadores Clave")
    if not df_city.empty:
        num_cols = df_city.select_dtypes("number").columns.tolist()
        defaults = num_cols[:3] if len(num_cols) >= 3 else num_cols

        cols = st.columns(3)
        metrics = [
            ("Precio Promedio", f"€{df_city['price'].mean():.2f}", "💶"),
            ("Valoración Media", f"{df_city['review_scores_rating'].mean():.1f}/5" if 'review_scores_rating' in df_city else "N/A", "⭐"),
            ("Propiedades", len(df_city), "🏠")
        ]

        for (title, value, icon), col in zip(metrics, cols):
            with col:
                html(create_metric_card(title, value, icon=icon))
    else:
        st.warning("No hay datos disponibles para mostrar indicadores.")

    st.markdown("---")

    # Mapa Interactivo mejorado
    st.subheader("🗺️ Mapa Interactivo")
    if not df_city.empty and {"latitude", "longitude"}.issubset(df_city.columns):
        df_mapa = df_city.dropna(subset=["latitude", "longitude"])
        if not df_mapa.empty:
            fig = px.scatter_mapbox(
                df_mapa,
                lat="latitude",
                lon="longitude",
                hover_name="ciudad",
                hover_data=["price", "room_type"],
                color="price",
                color_continuous_scale="viridis",
                zoom=10,
                height=600,
                title=f"Distribución de propiedades en {ciudad}"
            )
            fig.update_layout(
                mapbox_style="carto-positron",
                margin={"l":0,"r":0,"t":40,"b":0},
                mapbox=dict(center=dict(
                    lat=df_mapa["latitude"].mean(),
                    lon=df_mapa["longitude"].mean()
                ))
            )
            render_neon_chart(fig, "Mapa de Propiedades")
        else:
            st.warning("No hay datos geográficos disponibles para mostrar el mapa.")
    else:
        st.warning("No se encontraron columnas de coordenadas (latitude/longitude).")

    # Sección de datos curiosos
    st.markdown("---")
    with st.expander("💡 Datos Curiosos sobre Malta"):
        st.markdown("""
        - Malta es uno de los países más pequeños y densamente poblados del mundo.
        - Tiene tres sitios declarados Patrimonio de la Humanidad por la UNESCO.
        - El maltés es el único idioma semítico escrito en alfabeto latino.
        """)

# =============================================
# PÁGINA DE MODELADO EXPLICATIVO MEJORADA
# =============================================
def show_explanatory(ciudad, df_city):
    st.title(f"📈 Análisis Exploratorio en {ciudad}")

    if df_city.empty:
        st.warning("No hay datos disponibles para análisis.")
        return

    tipo_analisis = st.selectbox(
        "Selecciona el tipo de análisis",
        ["Distribución", "Relación"],
        help="Elige qué tipo de análisis deseas realizar"
    )

    if tipo_analisis == "Distribución":
        st.subheader("📊 Análisis de Distribución")
        tipo = st.radio(
            "Tipo de variable a analizar",
            ["Categórica", "Numérica"],
            horizontal=True
        )

        if tipo == "Categórica":
            cats = df_city.select_dtypes("object").columns.tolist()
            if cats:
                sel = st.selectbox("Selecciona variable categórica", cats)
                with st.spinner("Generando visualizaciones..."):
                    vc = df_city[sel].value_counts().reset_index()
                    vc.columns = [sel, "count"]

                    col1, col2 = st.columns(2)

                    with col1:
                        fig_bar = px.bar(
                            vc, 
                            x=sel, 
                            y="count", 
                            title=f"Frecuencia de {sel}",
                            color=sel,
                            color_discrete_sequence=px.colors.qualitative.Pastel
                        )
                        render_neon_chart(fig_bar, "Distribución por Categorías")

                    with col2:
                        fig_pie = px.pie(
                            vc, 
                            names=sel, 
                            values="count", 
                            title=f"Proporción de {sel}",
                            hole=0.3
                        )
                        render_neon_chart(fig_pie, "Composición Porcentual")

                    if st.checkbox("Mostrar tabla de frecuencias", key="freq_table"):
                        st.dataframe(vc.style.background_gradient(cmap="Blues"))
            else:
                st.warning("No hay variables categóricas disponibles.")
        else:
            nums = df_city.select_dtypes("number").columns.tolist()
            if nums:
                sel = st.selectbox("Selecciona variable numérica", nums)

                col1, col2 = st.columns(2)
                with col1:
                    fig_hist = px.histogram(
                        df_city, 
                        x=sel, 
                        nbins=30, 
                        title=f"Distribución de {sel}",
                        marginal="box",
                        color_discrete_sequence=["#FF7F50"]
                    )
                    render_neon_chart(fig_hist, "Histograma y Boxplot")

                with col2:
                    fig_box = px.box(
                        df_city, 
                        y=sel, 
                        title=f"Distribución de {sel}",
                        color_discrete_sequence=["#4682B4"],
                        points="all"
                    )
                    render_neon_chart(fig_box, "Diagrama de Cajas")

                st.markdown("**Estadísticas Descriptivas**")
                st.table(df_city[sel].describe().to_frame().style.format("{:.2f}"))
            else:
                st.warning("No hay variables numéricas disponibles.")

    elif tipo_analisis == "Relación":
        st.subheader("🔗 Análisis de Relación")
        nums = df_city.select_dtypes("number").columns.tolist()
        if len(nums) >= 2:
            x_var = st.selectbox("Variable X", nums)
            y_var = st.selectbox("Variable Y", nums, index=1 if len(nums) > 1 else 0)

            fig_scatter = px.scatter(
                df_city,
                x=x_var,
                y=y_var,
                trendline="ols",
                title=f"Relación entre {x_var} y {y_var}",
                color_discrete_sequence=["#FF7F50"]
            )
            render_neon_chart(fig_scatter, "Diagrama de Dispersión")

            corr = df_city[[x_var, y_var]].corr().iloc[0,1]
            st.metric("Coeficiente de Correlación", f"{corr:.2f}")
        else:
            st.warning("Se necesitan al menos 2 variables numéricas para este análisis.")

# =============================================
# PÁGINA DE MODELADO PREDICTIVO MEJORADA
# =============================================
def show_predictive(ciudad, df_city):
    st.title(f"🔮 Modelado Predictivo en {ciudad}")

    if df_city.empty:
        st.warning("No hay datos disponibles para modelado.")
        return

    st.markdown("""
    <div class="glass-card">
        <h3>Configuración del Modelo</h3>
        <p>Selecciona las variables y parámetros para construir tu modelo predictivo.</p>
    </div>
    """, unsafe_allow_html=True)

    nums = df_city.select_dtypes("number").columns.tolist()
    cat_cols = df_city.select_dtypes("object").columns.tolist()

    model_type = st.selectbox(
        "Tipo de modelo",
        ["Regresión lineal", "Regresión logística"],
        help="Selecciona el tipo de modelo predictivo a utilizar"
    )

    if model_type == "Regresión logística":
        target = st.selectbox(
            "Variable objetivo (categórica)", 
            cat_cols,
            help="Selecciona la variable categórica que quieres predecir"
        )
        features = st.multiselect(
            "Variables predictoras (numéricas)", 
            nums,
            help="Selecciona las variables numéricas que servirán para predecir"
        )
    else:
        target = st.selectbox(
            "Variable objetivo (numérica)", 
            nums,
            help="Selecciona la variable numérica que quieres predecir"
        )
        features = st.multiselect(
            "Variables predictoras", 
            [col for col in nums if col != target],
            help="Selecciona las variables que servirán para predecir"
        )

        # Solo para Regresión Lineal
    if model_type == "Regresión lineal" and features and st.checkbox("Mostrar matriz de correlación", value=True):
        # target y features son todos numéricos
        corr = df_city[features + [target]].corr().round(2)
        fig_corr = px.imshow(
            corr, 
            text_auto=True, 
            aspect="auto",
            color_continuous_scale="YlOrRd", 
            title="Matriz de Correlaciones"
        )
        render_neon_chart(fig_corr, "Relación entre Variables")

    if st.button("🚀 Ejecutar Modelo", type="primary"):
        if not features:
            st.error("Por favor selecciona al menos una variable predictora.")
            return

        with st.spinner("Entrenando modelo..."):
            X_train, X_test, y_train, y_test = train_test_split(
                df_city[features], 
                df_city[target], 
                test_size=0.3, 
                random_state=42
            )

            if model_type == "Regresión lineal":
                model = LinearRegression().fit(X_train, y_train)
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                r2 = model.score(X_test, y_test)

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(create_metric_card(
                        "Error Cuadrático Medio", 
                        f"{mse:.2f}", 
                        icon="📉"
                    ), unsafe_allow_html=True)
                with col2:
                    st.markdown(create_metric_card(
                        "Coeficiente R²", 
                        f"{r2:.2f}", 
                        delta="+ Bueno" if r2 > 0.7 else "- Mejorable" if r2 > 0.3 else "-- Pobre",
                        icon="📊"
                    ), unsafe_allow_html=True)

                df_results = pd.DataFrame({
                    "Real": y_test,
                    "Predicho": y_pred,
                    "Diferencia": y_test - y_pred
                })

                fig_results = px.scatter(
                    df_results, 
                    x="Real", 
                    y="Predicho", 
                    title="Valores Reales vs Predichos",
                    trendline="ols",
                    color="Diferencia",
                    color_continuous_scale="bluered"
                )
                fig_results.add_shape(
                    type="line", 
                    x0=df_results["Real"].min(), 
                    y0=df_results["Real"].min(),
                    x1=df_results["Real"].max(), 
                    y1=df_results["Real"].max(),
                    line=dict(color="green", dash="dash")
                )
                render_neon_chart(fig_results, "Resultados del Modelo")

                st.markdown("**Coeficientes del Modelo**")
                coefs = pd.DataFrame({
                    "Variable": features,
                    "Coeficiente": model.coef_,
                    "Importancia Absoluta": np.abs(model.coef_)
                }).sort_values("Importancia Absoluta", ascending=False)
                st.dataframe(coefs.style.bar(color="#FF7F50"))

            else:  # Regresión logística
                model = LogisticRegression(max_iter=1000).fit(X_train, y_train)
                y_pred = model.predict(X_test)

                accuracy = accuracy_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred, average="weighted")

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(create_metric_card(
                        "Exactitud (Accuracy)", 
                        f"{accuracy:.2%}", 
                        delta="+ Bueno" if accuracy > 0.7 else "- Mejorable" if accuracy > 0.5 else "-- Pobre",
                        icon="🎯"
                    ), unsafe_allow_html=True)
                with col2:
                    st.markdown(create_metric_card(
                        "Sensibilidad (Recall)", 
                        f"{recall:.2%}", 
                        icon="📈"
                    ), unsafe_allow_html=True)

                labels = sorted(y_test.unique())
                cm = confusion_matrix(y_test, y_pred, labels=labels)
                df_cm = pd.DataFrame(cm, index=labels, columns=labels)

                fig_cm = px.imshow(
                    df_cm, 
                    text_auto=True, 
                    aspect="auto",
                    labels=dict(x="Predicho", y="Real", color="Conteo"),
                    title="Matriz de Confusión",
                    color_continuous_scale="Blues"
                )
                render_neon_chart(fig_cm, "Matriz de Confusión")

                st.markdown("**Reporte de Clasificación**")
                report = classification_report(y_test, y_pred, output_dict=True)
                st.table(pd.DataFrame(report).transpose().style.background_gradient(cmap="Blues"))

# =============================================
# ROUTING PRINCIPAL
# =============================================
if pagina == "Inicio":
    show_home(ciudad_seleccionada, df_city)
elif pagina == "Modelado Explicativo":
    show_explanatory(ciudad_seleccionada, df_city)
else:
    show_predictive(ciudad_seleccionada, df_city)

# =============================================
# FOOTER MEJORADO
# =============================================
st.markdown("""
<div style="text-align: center; padding: 20px; margin-top: 50px; background: rgba(255,255,255,0.7); border-radius: 8px;">
    <p style="color: #6c757d; margin: 0;">© 2025 Malta Analytics Dashboard | Desarrollado con Streamlit</p>
    <div style="display: flex; justify-content: center; gap: 15px; margin-top: 10px;">
        <a href="#" style="color: var(--primary); text-decoration: none;">Términos</a>
        <a href="#" style="color: var(--primary); text-decoration: none;">Privacidad</a>
        <a href="#" style="color: var(--primary); text-decoration: none;">Contacto</a>
    </div>
</div>
""", unsafe_allow_html=True)
