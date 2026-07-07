"""
Dashboard Streamlit - Modelado Predictivo de Severidad de Siniestros Viales
Proyecto Final - Universidad Nacional del Altiplano
Escuela Profesional de Ingenieria de Sistemas
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import joblib
from pathlib import Path
from PIL import Image
import base64
from io import BytesIO

# ─── Configuracion de pagina ─────────────────────────────────
st.set_page_config(
    page_title="Siniestros Viales - ONSV Peru",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Rutas ────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_PROCESSED = BASE_DIR / "data" / "processed"
FIGURES = BASE_DIR / "outputs" / "figures"
MODELS_DIR = BASE_DIR / "outputs" / "models"
TABLES = BASE_DIR / "outputs" / "tables"

# ─── Cargar datos ─────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_parquet(DATA_PROCESSED / "dataset_features.parquet")
    df['FECHA_SINIESTRO'] = pd.to_datetime(df['FECHA_SINIESTRO'], errors='coerce')
    severidad_map = {0: 'Baja (1 fallecido)', 1: 'Media (2 fallecidos)', 2: 'Alta (3+ fallecidos)'}
    df['SEVERIDAD_LABEL'] = df['severidad'].map(severidad_map)
    return df

@st.cache_resource
def load_model():
    model_path = MODELS_DIR / 'modelo_logistic_regression.pkl'
    scaler_path = MODELS_DIR / 'scaler.pkl'
    if model_path.exists() and scaler_path.exists():
        return joblib.load(model_path), joblib.load(scaler_path)
    return None, None

@st.cache_data
def load_comparison():
    csv_path = TABLES / 'comparacion_modelos.csv'
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return None

df = load_data()
model, scaler = load_model()
df_cmp = load_comparison()

# ─── Sidebar ──────────────────────────────────────────────────
st.sidebar.markdown("""
<div style="text-align: center; padding: 10px;">
    <h2 style="color: #1a5276;">🚦 ONSV Peru</h2>
    <p style="font-size: 12px; color: #666;">
        Observatorio Nacional de Seguridad Vial<br>
        2021-2025
    </p>
</div>
""", unsafe_allow_html=True)

pagina = st.sidebar.radio(
    "Navegacion",
    ["Inicio", "Resumen Nacional", "Mapa", "Analisis Temporal",
     "Perfil de Personas", "Factores de Riesgo",
     "Modelos Predictivos", "Prediccion Individual", "Conclusiones"]
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"""
**Datos cargados:**
- {len(df):,} siniestros
- {df['CANTIDAD_FALLECIDOS'].sum():,} fallecidos
- {df['CANTIDAD_LESIONADOS'].sum():,} lesionados
""")

st.sidebar.markdown("---")
st.sidebar.info(
    "Proyecto Final - Ingenieria de Sistemas\n"
    "Universidad Nacional del Altiplano"
)

# ─── FUNCIONES COMUNES ────────────────────────────────────────
def img_to_base64(path):
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except:
        return None

def mostrar_metricas():
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Siniestros", f"{len(df):,}",
                  help="Numero total de siniestros fatales registrados")
    with col2:
        st.metric("Total Fallecidos", f"{df['CANTIDAD_FALLECIDOS'].sum():,}",
                  help="Numero total de victimas fatales")
    with col3:
        st.metric("Total Lesionados", f"{df['CANTIDAD_LESIONADOS'].sum():,}",
                  help="Numero total de personas lesionadas")
    with col4:
        promedio = df['CANTIDAD_FALLECIDOS'].mean()
        st.metric("Prom. Fallecidos/Siniestro", f"{promedio:.2f}",
                  help="Promedio de fallecidos por siniestro")

# ─── PAGINA 1: INICIO ─────────────────────────────────────────
if pagina == "Inicio":
    st.title("🚦 Modelado Predictivo de la Severidad de Siniestros Viales Fatales en el Peru")
    st.markdown("---")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        ### 📋 Resumen del Proyecto

        **Objetivo:** Desarrollar un sistema analitico y predictivo capaz de identificar los
        factores de riesgo asociados a la severidad de los siniestros viales en el Peru
        mediante tecnicas de Ciencia de Datos y Machine Learning.

        **Alcance:**
        - Periodo: 2021-2025 (preliminar)
        - Cobertura: Nacional (24 departamentos)
        - Datos: ONSV - Observatorio Nacional de Seguridad Vial
        - Metodologia: CRISP-DM

        **Tecnologias Utilizadas:**
        - 🐍 Python: Preprocesamiento, ML, evaluacion
        - 📊 R: Estadistica descriptiva e inferencial
        - 🖥️ Streamlit: Dashboard interactivo
        """)

    with col2:
        st.image("https://www.onsv.gob.pe/images/logo_onsv.png",
                 width=250, caption="Observatorio Nacional de Seguridad Vial")
        st.markdown("""
        **Equipo:**
        - Universidad Nacional del Altiplano
        - Escuela Profesional de Ingenieria de Sistemas
        """)

    mostrar_metricas()

    st.markdown("---")
    st.markdown("""
    ### 🎯 Metodologia CRISP-DM
    | Fase | Estado |
    |------|--------|
    | 1. Business Understanding | ✅ Completado |
    | 2. Data Understanding | ✅ Completado |
    | 3. Data Preparation | ✅ Completado |
    | 4. Modeling | ✅ Completado (6 modelos) |
    | 5. Evaluation | ✅ Completado |
    | 6. Deployment | ✅ Dashboard desplegado |
    """)

# ─── PAGINA 2: RESUMEN NACIONAL ──────────────────────────────
elif pagina == "Resumen Nacional":
    st.title("📊 Resumen Nacional")
    st.markdown("---")

    mostrar_metricas()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Distribucion de Severidad")
        fig = px.pie(df, names='SEVERIDAD_LABEL',
                     title='Proporcion por Nivel de Severidad',
                     color='SEVERIDAD_LABEL',
                     color_discrete_map={
                         'Baja (1 fallecido)': '#2ecc71',
                         'Media (2 fallecidos)': '#f39c12',
                         'Alta (3+ fallecidos)': '#e74c3c'
                     })
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Severidad por Departamento (Top 10)")
        dpto_sev = df.groupby('DEPARTAMENTO').agg(
            Total=('severidad', 'count'),
            Promedio=('severidad', 'mean')
        ).sort_values('Total', ascending=False).head(10)

        fig = px.bar(dpto_sev, x=dpto_sev.index, y='Total',
                     color='Promedio', color_continuous_scale='RdYlGn_r',
                     title='Top 10 Departamentos con Mas Siniestros')
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Siniestros por Zona")
        zona_counts = df['ZONA'].value_counts()
        fig = px.bar(zona_counts, x=zona_counts.index, y=zona_counts.values,
                     color=zona_counts.index,
                     title='Distribucion Urbano/Rural',
                     labels={'y': 'Cantidad', 'x': 'Zona'})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Top 10 Causas Principales")
        causa_counts = df['CAUSA_FACTOR_PRINCIPAL'].value_counts().head(10)
        fig = px.bar(causa_counts, x=causa_counts.values, y=causa_counts.index,
                     orientation='h', title='Causas mas Frecuentes',
                     color=causa_counts.values, color_continuous_scale='Oranges')
        st.plotly_chart(fig, use_container_width=True)

# ─── PAGINA 3: MAPA ──────────────────────────────────────────
elif pagina == "Mapa":
    st.title("🗺️ Distribucion Geografica")
    st.markdown("---")

    df_map = df.dropna(subset=['LATITUD', 'LONGITUD']).copy()
    df_map = df_map[
        (df_map['LATITUD'].between(-20, 0)) &
        (df_map['LONGITUD'].between(-85, -68))
    ]

    st.subheader("Mapa de Siniestros Fatales")

    fig = px.scatter_mapbox(
        df_map.sample(min(3000, len(df_map))),
        lat='LATITUD', lon='LONGITUD',
        color='SEVERIDAD_LABEL',
        size='CANTIDAD_FALLECIDOS',
        hover_name='DEPARTAMENTO',
        hover_data={
            'CLASE_SINIESTRO': True,
            'CANTIDAD_FALLECIDOS': True,
            'CAUSA_FACTOR_PRINCIPAL': True,
            'LATITUD': False, 'LONGITUD': False
        },
        color_discrete_map={
            'Baja (1 fallecido)': '#2ecc71',
            'Media (2 fallecidos)': '#f39c12',
            'Alta (3+ fallecidos)': '#e74c3c'
        },
        zoom=4.5, center={"lat": -9.5, "lon": -75},
        mapbox_style='open-street-map',
        title='Distribucion Geografica de Siniestros Fatales (muestra de 3000)'
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top 10 Departamentos")
    dpto_stats = df.groupby('DEPARTAMENTO').agg(
        Siniestros=('CODIGO_SINIESTRO', 'count'),
        Fallecidos=('CANTIDAD_FALLECIDOS', 'sum'),
        Lesionados=('CANTIDAD_LESIONADOS', 'sum'),
        Severidad_Prom=('severidad', 'mean')
    ).sort_values('Siniestros', ascending=False).head(10)

    st.dataframe(dpto_stats.style.format({
        'Severidad_Prom': '{:.2f}',
        'Siniestros': '{:,.0f}',
        'Fallecidos': '{:,.0f}',
        'Lesionados': '{:,.0f}'
    }), use_container_width=True)

# ─── PAGINA 4: ANALISIS TEMPORAL ─────────────────────────────
elif pagina == "Analisis Temporal":
    st.title("📈 Analisis Temporal")
    st.markdown("---")

    df_temp = df.dropna(subset=['FECHA_SINIESTRO']).copy()
    df_temp['ANIO'] = df_temp['FECHA_SINIESTRO'].dt.year
    df_temp['MES'] = df_temp['FECHA_SINIESTRO'].dt.month
    df_temp['ANIO_MES'] = df_temp['FECHA_SINIESTRO'].dt.to_period('M').astype(str)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Siniestros por Anio")
        anio_counts = df_temp['ANIO'].value_counts().sort_index()
        fig = px.bar(anio_counts, x=anio_counts.index, y=anio_counts.values,
                     labels={'x': 'Anio', 'y': 'Cantidad'},
                     color=anio_counts.values,
                     color_continuous_scale='Blues',
                     title='Evolucion Anual de Siniestros')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Siniestros por Mes")
        mes_counts = df_temp['MES'].value_counts().sort_index()
        meses = ['Ene','Feb','Mar','Abr','May','Jun',
                 'Jul','Ago','Sep','Oct','Nov','Dic']
        fig = px.line(x=meses, y=[mes_counts.get(i, 0) for i in range(1, 13)],
                      markers=True,
                      labels={'x': 'Mes', 'y': 'Cantidad'},
                      title='Estacionalidad Mensual (todos los anios)')
        fig.update_traces(line_color='#2ecc71')
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Franja Horaria")
    franja_map = {0: 'Madrugada (0-6)', 1: 'Maniana (6-12)',
                  2: 'Tarde (12-18)', 3: 'Noche (18-24)'}
    df_temp['FRANJA'] = df_temp['FRANJA_HORARIA'].map(franja_map)
    franja_counts = df_temp['FRANJA'].value_counts()
    fig = px.bar(franja_counts, x=franja_counts.index, y=franja_counts.values,
                 color=franja_counts.index,
                 title='Distribucion por Franja Horaria',
                 labels={'y': 'Cantidad', 'x': 'Franja'})
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Tendencia Temporal Mensual")
    serie_mensual = df_temp.groupby('ANIO_MES').size().reset_index(name='Conteo')
    serie_mensual = serie_mensual.sort_values('ANIO_MES')
    fig = px.line(serie_mensual, x='ANIO_MES', y='Conteo',
                  title='Evolucion Mensual de Siniestros Fatales',
                  markers=True)
    fig.update_xaxes(tickangle=45, nticks=20)
    st.plotly_chart(fig, use_container_width=True)

# ─── PAGINA 5: PERFIL DE PERSONAS ────────────────────────────
elif pagina == "Perfil de Personas":
    st.title("👥 Perfil de Personas Involucradas")
    st.markdown("---")

    personas_path = DATA_PROCESSED / "personas_clean.parquet"
    if personas_path.exists():
        df_per = pd.read_parquet(personas_path)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Personas", f"{len(df_per):,}")
        with col2:
            fallecidos = (df_per['GRAVEDAD'].str.upper() == 'FALLECIDO').sum()
            st.metric("Fallecidos", f"{fallecidos:,}",
                      f"{fallecidos/len(df_per)*100:.1f}%")
        with col3:
            lesionados = (df_per['GRAVEDAD'].str.upper() == 'LESIONADO').sum()
            st.metric("Lesionados", f"{lesionados:,}",
                      f"{lesionados/len(df_per)*100:.1f}%")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Distribucion por Tipo de Persona")
            tipo_counts = df_per['TIPO_PERSONA'].value_counts()
            fig = px.pie(values=tipo_counts.values, names=tipo_counts.index,
                         title='Conductor / Peaton / Pasajero')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Distribucion por Sexo")
            sexo_counts = df_per['SEXO'].value_counts()
            fig = px.pie(values=sexo_counts.values, names=sexo_counts.index,
                         title='Masculino / Femenino',
                         color=sexo_counts.index,
                         color_discrete_map={
                             'MASCULINO': '#3498db', 'FEMENINO': '#e91e63'
                         })
            st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Distribucion de Edad")
            edad_valid = df_per.dropna(subset=['EDAD'])
            edad_valid = edad_valid[edad_valid['EDAD'].between(0, 110)]
            fig = px.histogram(edad_valid, x='EDAD', nbins=40,
                               title='Histograma de Edad',
                               color_discrete_sequence=['#9b59b6'])
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Gravedad por Tipo de Persona")
            cruzada = pd.crosstab(df_per['TIPO_PERSONA'], df_per['GRAVEDAD'])
            fig = px.bar(cruzada, barmode='group',
                         title='Gravedad segun Tipo de Persona')
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Licencia y Alcohol")
        col1, col2 = st.columns(2)
        with col1:
            lic_counts = df_per['POSEE_LICENCIA'].value_counts()
            fig = px.pie(values=lic_counts.values,
                         names=['Sin Licencia', 'Con Licencia'],
                         title='Posee Licencia de Conducir',
                         color=['Sin Licencia', 'Con Licencia'],
                         color_discrete_map={
                             'Sin Licencia': '#e74c3c', 'Con Licencia': '#2ecc71'
                         })
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            if 'ALCOHOL_POSITIVO' in df_per.columns:
                alc_counts = df_per['ALCOHOL_POSITIVO'].value_counts()
                fig = px.pie(values=alc_counts.values,
                             names=['Negativo', 'Positivo'],
                             title='Resultado Dosaje Etilico',
                             color=['Negativo', 'Positivo'],
                             color_discrete_map={
                                 'Negativo': '#3498db', 'Positivo': '#e74c3c'
                             })
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Datos de personas no disponibles. Ejecute primero 01_clean_datasets.py")

# ─── PAGINA 6: FACTORES DE RIESGO ────────────────────────────
elif pagina == "Factores de Riesgo":
    st.title("⚠️ Factores de Riesgo")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Condicion Climatica")
        clima_counts = df['CONDICION_CLIMATICA'].value_counts().head(8)
        fig = px.bar(clima_counts, x=clima_counts.index, y=clima_counts.values,
                     color=clima_counts.values, color_continuous_scale='Blues',
                     title='Siniestros por Condicion Climatica')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Clase de Siniestro")
        clase_counts = df['CLASE_SINIESTRO'].value_counts()
        fig = px.bar(clase_counts, x=clase_counts.index, y=clase_counts.values,
                     color=clase_counts.values, color_continuous_scale='Purples',
                     title='Distribucion por Clase')
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Tipo de Via")
        via_counts = df['TIPO_VIA'].value_counts().head(8)
        fig = px.bar(via_counts, x=via_counts.index, y=via_counts.values,
                     color=via_counts.values, color_continuous_scale='Greens',
                     title='Siniestros por Tipo de Via')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Riesgo de Infraestructura")
        if 'RIESGO_INFRAESTRUCTURA' in df.columns:
            fig = px.histogram(df, x='RIESGO_INFRAESTRUCTURA',
                               color='SEVERIDAD_LABEL', nbins=10,
                               title='Riesgo de Infraestructura vs Severidad',
                               color_discrete_map={
                                   'Baja (1 fallecido)': '#2ecc71',
                                   'Media (2 fallecidos)': '#f39c12',
                                   'Alta (3+ fallecidos)': '#e74c3c'
                               })
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Matriz de Correlacion")
    img_path = FIGURES / "matriz_correlacion.png"
    if img_path.exists():
        st.image(str(img_path), use_container_width=True)

# ─── PAGINA 7: MODELOS PREDICTIVOS ───────────────────────────
elif pagina == "Modelos Predictivos":
    st.title("🤖 Modelos Predictivos")
    st.markdown("---")

    if df_cmp is not None:
        st.subheader("Comparacion de Modelos")
        st.dataframe(df_cmp.style.highlight_max(color='#2ecc71', axis=0),
                     use_container_width=True)

        st.subheader("Comparacion Visual")
        fig_path = FIGURES / "comparacion_modelos.png"
        if fig_path.exists():
            st.image(str(fig_path), use_container_width=True)

        st.subheader("Feature Importance")
        fi_path = FIGURES / "feature_importance_top15.png"
        if fi_path.exists():
            st.image(str(fi_path), use_container_width=True)
        else:
            fi_path2 = FIGURES / "shap_feature_importance.png"
            if fi_path2.exists():
                st.image(str(fi_path2), use_container_width=True)

        st.subheader("Matrices de Confusion")
        cm_files = sorted(FIGURES.glob("cm_*.png"))
        cols = st.columns(3)
        for i, cm_path in enumerate(cm_files):
            with cols[i % 3]:
                st.image(str(cm_path), use_container_width=True)
    else:
        st.warning("Resultados de modelos no encontrados. Ejecute primero 04_modelamiento.py")

# ─── PAGINA 8: PREDICCION INDIVIDUAL ──────────────────────────
elif pagina == "Prediccion Individual":
    st.title("🔮 Prediccion Individual de Severidad")
    st.markdown("---")

    if model is not None and scaler is not None:
        st.markdown("""
        Ingrese las caracteristicas del siniestro para predecir su nivel de severidad.
        """)

        col1, col2, col3 = st.columns(3)
        with col1:
            hora = st.slider("Hora del dia", 0, 23, 12)
            mes = st.selectbox("Mes", range(1, 13),
                               format_func=lambda x: ['Ene','Feb','Mar','Abr','May','Jun',
                                                       'Jul','Ago','Sep','Oct','Nov','Dic'][x-1])
            anio = st.selectbox("Anio", [2021, 2022, 2023, 2024, 2025])

        with col2:
            latitud = st.number_input("Latitud", value=-12.0, format="%.6f")
            longitud = st.number_input("Longitud", value=-77.0, format="%.6f")
            zona = st.selectbox("Zona", ["URBANA", "RURAL"])

        with col3:
            clima = st.selectbox("Condicion Climatica",
                                 ["DESPEJADO", "LLUVIOSO", "SOLEADO", "NUBLADO", "NEBLINA"])
            lesionados = st.number_input("Cantidad de Lesionados", 0, 50, 0)
            vehiculos = st.number_input("Vehiculos Danados", 0, 20, 1)

        if st.button("🔮 Predecir Severidad", type="primary", use_container_width=True):
            try:
                # Construir vector de features
                franja = 0 if hora <= 5 else (1 if hora <= 11 else (2 if hora <= 17 else 3))
                es_noche = 1 if franja in [0, 3] else 0
                fin_semana = 1 if pd.Timestamp(f"{anio}-{mes:02d}-01").dayofweek >= 5 else 0
                temporada = 0 if mes in [12,1,2] else (1 if mes in [3,4,5] else (2 if mes in [6,7,8] else 3))

                # Riesgos
                riesgo_climatico = 1 if 'LLUV' in str(clima).upper() or 'NEBL' in str(clima).upper() else 0
                riesgo_infra = 1 if zona == "RURAL" else 0

                input_data = pd.DataFrame([{
                    'CANTIDAD_LESIONADOS': lesionados,
                    'CANTIDAD_VEHICULOS': vehiculos,
                    'LATITUD': latitud,
                    'LONGITUD': longitud,
                    'HORA': hora,
                    'ANIO': anio,
                    'MES': mes,
                    'DIA_SEMANA': 0,
                    'FIN_SEMANA': fin_semana,
                    'TEMPORADA': temporada,
                    'MACROREGION': 0,
                    'RIESGO_INFRAESTRUCTURA': riesgo_infra,
                    'RIESGO_CLIMATICO': riesgo_climatico,
                    'SENIALIZACION_DEFICIENTE': 0,
                    'EDAD_PROMEDIO': 35.0,
                    'EDAD_STD': 12.0,
                    'TOTAL_ALCOHOL': 0,
                    'TOTAL_LICENCIA': 1,
                    'TOTAL_PERSONAS': 2,
                    'PCT_ALCOHOL': 0.0,
                    'PCT_LICENCIA': 0.5,
                    'ES_NOCHE': es_noche
                }])

                for col in scaler.feature_names_in_:
                    if col not in input_data.columns:
                        input_data[col] = 0
                input_data = input_data[list(scaler.feature_names_in_)]

                input_scaled = scaler.transform(input_data)
                pred = model.predict(input_scaled)[0]
                proba = model.predict_proba(input_scaled)[0]

                severidad_map = {0: 'Baja (1 fallecido)', 1: 'Media (2 fallecidos)', 2: 'Alta (3+ fallecidos)'}
                color_map = {0: '#2ecc71', 1: '#f39c12', 2: '#e74c3c'}

                st.markdown("---")
                st.subheader("Resultado de la Prediccion")

                col_res1, col_res2, col_res3 = st.columns(3)
                for i in range(3):
                    with [col_res1, col_res2, col_res3][i]:
                        pct = proba[i] * 100
                        st.markdown(f"""
                        <div style="text-align: center; padding: 15px;
                             border-radius: 10px; border: 2px solid {color_map[i]};
                             background-color: {'#f0fff0' if i == pred else 'white'}">
                            <h3 style="color: {color_map[i]};">{severidad_map[i]}</h3>
                            <p style="font-size: 24px; font-weight: bold; color: {color_map[i]};">{pct:.1f}%</p>
                        </div>
                        """, unsafe_allow_html=True)

                st.success(f"**Prediccion:** {severidad_map[pred]} con {proba[pred]*100:.1f}% de probabilidad")

            except Exception as e:
                st.error(f"Error en la prediccion: {e}")
    else:
        st.warning("Modelo no disponible. Ejecute primero 04_modelamiento.py")

# ─── PAGINA 9: CONCLUSIONES ──────────────────────────────────
elif pagina == "Conclusiones":
    st.title("📝 Conclusiones y Recomendaciones")
    st.markdown("---")

    st.markdown("""
    ### 📌 Conclusiones

    1. **Distribucion de Severidad**
       - El 89.8% de los siniestros registran 1 fallecido (severidad baja)
       - Solo el 3.4% presenta 3 o mas fallecidos (severidad alta)
       - Esto implica un desbalance natural que requiere tecnicas de balanceo

    2. **Factores de Riesgo Identificados**
       - Las causas principales son: imprudencia del conductor, exceso de velocidad y conducir en estado de ebriedad
       - Lima concentra la mayor cantidad de siniestros, seguido de La Libertad y Cusco
       - Las carreteras nacionales concentran los siniestros mas severos
       - La mayoria ocurre en zonas rurales y en tramos rectos

    3. **Rendimiento del Modelo**
       - **Logistic Regression** fue el mejor modelo: F1=0.684, ROC-AUC=0.925
       - XGBoost obtuvo el mayor accuracy (0.914) pero menor F1 macro
       - La regresion logistica ofrece el mejor balance precision-recall
       - El ROC-AUC > 0.92 indica excelente capacidad discriminativa

    4. **Factores Predictivos mas Importantes**
       - Numero de personas involucradas
       - Hora del dia (especialmente nocturno y madrugada)
       - Condiciones de infraestructura vial
       - Presencia de alcohol en conductores

    ### 🎯 Recomendaciones

    1. **Politicas Publicas**
       - Reforzar controles de velocidad y alcoholemia en carreteras nacionales
       - Mejorar la senializacion en zonas rurales
       - Campanas de concientizacion en departamentos con mayor incidencia

    2. **Investigacion Futura**
       - Incorporar datos de flujo vehicular
       - Incluir variables de diseno geometrico de vias
       - Explorar tecnicas de deep learning con datos de mayor granularidad
       - Implementar sistema de alerta temprana basado en condiciones climaticas

    3. **Mejoras Tecnicas**
       - Recolectar datos mas completos de senializacion (94% de nulos actualmente)
       - Estandarizar la codificacion de causas a nivel nacional
       - Integrar con datos meteorologicos en tiempo real
    """)

    st.info("""
    **Trabajo Futuro:** Este proyecto sienta las bases para un sistema de monitoreo continuo
    que podria ser adoptado por el ONSV como herramienta de soporte a la decision en
    politicas de seguridad vial.
    """)

# ─── Footer ───────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="text-align: center; font-size: 11px; color: #999;">
    Desarrollado con ❤️ para la UNA-Puno<br>
    © 2025 - Proyecto Final
</div>
""", unsafe_allow_html=True)
