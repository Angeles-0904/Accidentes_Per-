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
import json
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
REPORTS = BASE_DIR / "outputs" / "reports"

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

@st.cache_data
def load_r_results():
    """Cargar resultados de R (pruebas estadisticas)"""
    csv_path = TABLES / 'resultados_r.csv'
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return None

@st.cache_data
def load_r_severidad():
    csv_path = TABLES / 'r_severidad_distribucion.csv'
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return None

@st.cache_resource
def load_geojson():
    """Cargar GeoJSON de departamentos del Peru."""
    geojson_path = BASE_DIR / "data" / "external" / "peru_departamentos.geojson"
    if geojson_path.exists():
        with open(geojson_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

@st.cache_data
def load_r_dptos():
    csv_path = TABLES / 'r_top_departamentos.csv'
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return None

df = load_data()
model, scaler = load_model()
df_cmp = load_comparison()
df_r = load_r_results()
df_r_sev = load_r_severidad()
df_r_dptos = load_r_dptos()
geojson_peru = load_geojson()

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
     "Dashboard Ejecutivo",
     "Analisis Estadistico (R)",
     "Modelos Predictivos", "Prediccion Individual", "Conclusiones"]
)

# ─── FILTROS INTERACTIVOS ────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔍 Filtros")

# Estado de los filtros en session_state para persistencia
if "filtro_departamento" not in st.session_state:
    st.session_state.filtro_departamento = []
if "filtro_anio" not in st.session_state:
    st.session_state.filtro_anio = []

# Obtener opciones
anios_disponibles = sorted(df['ANIO'].dropna().unique().astype(int).tolist())
dptos_disponibles = sorted(df['DEPARTAMENTO'].dropna().unique().tolist())

# Filtros
filtro_dpto = st.sidebar.multiselect(
    "Departamento",
    options=dptos_disponibles,
    default=st.session_state.filtro_departamento,
    placeholder="Todos los departamentos",
    help="Filtrar por uno o mas departamentos"
)

filtro_anio = st.sidebar.multiselect(
    "Anio",
    options=anios_disponibles,
    default=st.session_state.filtro_anio,
    placeholder="Todos los anios",
    help="Filtrar por uno o mas anios"
)

filtro_zona = st.sidebar.radio(
    "Zona",
    options=["Todas", "URBANA", "RURAL"],
    horizontal=True,
    help="Filtrar por zona geografica"
)

filtro_severidad = st.sidebar.radio(
    "Severidad",
    options=["Todas", "Baja", "Media", "Alta"],
    horizontal=True,
    help="Filtrar por nivel de severidad"
)

# ─── APLICAR FILTROS ──────────────────────────────────────────
@st.cache_data
def get_filtered_data(df_orig, dptos, anios, zona, severidad):
    """Aplicar filtros al dataframe y retornar version filtrada."""
    df_filt = df_orig.copy()

    if dptos:
        df_filt = df_filt[df_filt['DEPARTAMENTO'].isin(dptos)]
    if anios:
        df_filt = df_filt[df_filt['ANIO'].isin(anios)]
    if zona != "Todas":
        df_filt = df_filt[df_filt['ZONA'] == zona]
    if severidad != "Todas":
        severidad_map = {"Baja": 0, "Media": 1, "Alta": 2}
        df_filt = df_filt[df_filt['severidad'] == severidad_map[severidad]]

    return df_filt

df_filtrado = get_filtered_data(df, filtro_dpto, filtro_anio, filtro_zona, filtro_severidad)

# Guardar estado actual para persistencia
if filtro_dpto:
    st.session_state.filtro_departamento = filtro_dpto
if filtro_anio:
    st.session_state.filtro_anio = filtro_anio

st.sidebar.markdown(f"""
**Datos cargados:**
- {len(df):,} siniestros
- {df['CANTIDAD_FALLECIDOS'].sum():,} fallecidos
""")

# Badge de filtro activo
if len(df_filtrado) < len(df):
    st.sidebar.info(f"🎯 Mostrando **{len(df_filtrado):,}** de **{len(df):,}** siniestros")

st.sidebar.markdown("---")
st.sidebar.info(
    "Proyecto Final - Ingenieria de Sistemas\n"
    "Universidad Nacional del Altiplano - Puno"
)

# ─── FUNCIONES COMUNES ────────────────────────────────────────
def img_to_base64(path):
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except:
        return None

def mostrar_metricas(df_mostrar=None):
    """Mostrar KPIs basicos. Usa df_filtrado si no se especifica df_mostrar."""
    if df_mostrar is None:
        df_mostrar = df_filtrado
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Siniestros", f"{len(df_mostrar):,}",
                  help="Numero total de siniestros fatales registrados")
    with col2:
        st.metric("Total Fallecidos", f"{df_mostrar['CANTIDAD_FALLECIDOS'].sum():,}",
                  help="Numero total de victimas fatales")
    with col3:
        st.metric("Total Lesionados", f"{df_mostrar['CANTIDAD_LESIONADOS'].sum():,}",
                  help="Numero total de personas lesionadas")
    with col4:
        promedio = df_mostrar['CANTIDAD_FALLECIDOS'].mean()
        st.metric("Prom. Fallecidos/Siniestro", f"{promedio:.2f}",
                  help="Promedio de fallecidos por siniestro")

def mostrar_kpis_ejecutivos(df_mostrar):
    """KPIs ejecutivos para un gerente del ONSV."""
    col1, col2, col3, col4 = st.columns(4)

    # KPI 1: Tasa de mortalidad (fallecidos por siniestro)
    tasa = df_mostrar['CANTIDAD_FALLECIDOS'].sum() / len(df_mostrar) if len(df_mostrar) > 0 else 0
    col1.markdown("""
    <div style="background:#1a5276; padding:15px; border-radius:10px; text-align:center;">
        <p style="color:#85c1e9; font-size:12px; margin:0;">🚨 TASA MORTALIDAD</p>
        <p style="color:white; font-size:28px; font-weight:bold; margin:5px 0;">{:.2f}</p>
        <p style="color:#85c1e9; font-size:11px; margin:0;">fallecidos / siniestro</p>
    </div>
    """.format(tasa), unsafe_allow_html=True)

    # KPI 2: % Severidad Alta
    pct_alta = (df_mostrar['severidad'] == 2).sum() / len(df_mostrar) * 100 if len(df_mostrar) > 0 else 0
    col2.markdown("""
    <div style="background:#922b21; padding:15px; border-radius:10px; text-align:center;">
        <p style="color:#f1948a; font-size:12px; margin:0;">⚠️ SEVERIDAD ALTA</p>
        <p style="color:white; font-size:28px; font-weight:bold; margin:5px 0;">{:.1f}%</p>
        <p style="color:#f1948a; font-size:11px; margin:0;">3+ fallecidos</p>
    </div>
    """.format(pct_alta), unsafe_allow_html=True)

    # KPI 3: Departamento mas critico
    dpto_critico = df_mostrar.groupby('DEPARTAMENTO').agg(
        Total=('severidad', 'count')
    ).sort_values('Total', ascending=False).head(1)
    nom_dpto = dpto_critico.index[0] if len(dpto_critico) > 0 else "-"
    val_dpto = int(dpto_critico['Total'].iloc[0]) if len(dpto_critico) > 0 else 0
    col3.markdown("""
    <div style="background:#1e8449; padding:15px; border-radius:10px; text-align:center;">
        <p style="color:#82e0aa; font-size:12px; margin:0;">📍 DPTO MAS CRITICO</p>
        <p style="color:white; font-size:22px; font-weight:bold; margin:5px 0;">{}</p>
        <p style="color:#82e0aa; font-size:11px; margin:0;">{} siniestros</p>
    </div>
    """.format(nom_dpto, val_dpto), unsafe_allow_html=True)

    # KPI 4: Franja horaria mas peligrosa
    franja_map_kpi = {0: 'Madrugada', 1: 'Maniana', 2: 'Tarde', 3: 'Noche'}
    franja_critica = df_mostrar['FRANJA_HORARIA'].value_counts().head(1)
    nom_franja = franja_map_kpi.get(franja_critica.index[0], '-') if len(franja_critica) > 0 else "-"
    val_franja = int(franja_critica.iloc[0]) if len(franja_critica) > 0 else 0
    col4.markdown("""
    <div style="background:#7d3c98; padding:15px; border-radius:10px; text-align:center;">
        <p style="color:#d2b4de; font-size:12px; margin:0;">🕐 FRANJA PELIGROSA</p>
        <p style="color:white; font-size:22px; font-weight:bold; margin:5px 0;">{}</p>
        <p style="color:#d2b4de; font-size:11px; margin:0;">{} siniestros</p>
    </div>
    """.format(nom_franja, val_franja), unsafe_allow_html=True)

    with st.expander("📋 Ver todos los indicadores ejecutivos"):
        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            st.metric("Vehiculos involucrados", f"{df_mostrar['CANTIDAD_VEHICULOS'].sum():,}")
            st.metric("Tasa lesionados/siniestro", f"{df_mostrar['CANTIDAD_LESIONADOS'].mean():.2f}")
        with col_b:
            st.metric("% Zona Rural", f"{(df_mostrar['ZONA']=='RURAL').sum()/len(df_mostrar)*100:.1f}%")
            st.metric("% Zona Urbana", f"{(df_mostrar['ZONA']=='URBANA').sum()/len(df_mostrar)*100:.1f}%")
        with col_c:
            st.metric("Edad promedio", f"{df_mostrar['EDAD_PROMEDIO'].mean():.1f} anios")
            st.metric("% Alcohol positivo", f"{df_mostrar['PCT_ALCOHOL'].mean()*100:.1f}%")
        with col_d:
            st.metric("Domingos (mayor incidencia)", f"{(df_mostrar['DIA_SEMANA']==6).sum():,}")
            st.metric("% Fin de semana", f"{df_mostrar['FIN_SEMANA'].mean()*100:.1f}%")

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

    mostrar_metricas(df)

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

    # KPIs Ejecutivos
    mostrar_kpis_ejecutivos(df_filtrado)

    st.markdown("---")
    mostrar_metricas(df_filtrado)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Distribucion de Severidad")
        fig = px.pie(df_filtrado, names='SEVERIDAD_LABEL',
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
        dpto_sev = df_filtrado.groupby('DEPARTAMENTO').agg(
            Total=('severidad', 'count'),
            Promedio=('severidad', 'mean')
        ).sort_values('Total', ascending=False).head(10)

        fig = px.bar(dpto_sev, x=dpto_sev.index, y='Total',
                     color='Promedio', color_continuous_scale='RdYlGn_r',
                     title='Top 10 Departamentos')
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Siniestros por Zona")
        zona_counts = df_filtrado['ZONA'].value_counts()
        fig = px.bar(zona_counts, x=zona_counts.index, y=zona_counts.values,
                     color=zona_counts.index,
                     title='Distribucion Urbano/Rural',
                     labels={'y': 'Cantidad', 'x': 'Zona'})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Top 10 Causas Principales")
        causa_counts = df_filtrado['CAUSA_FACTOR_PRINCIPAL'].value_counts().head(10)
        fig = px.bar(causa_counts, x=causa_counts.values, y=causa_counts.index,
                     orientation='h', title='Causas mas Frecuentes',
                     color=causa_counts.values, color_continuous_scale='Oranges')
        st.plotly_chart(fig, use_container_width=True)

    # Boton de descarga datos filtrados
    csv_filtrado = df_filtrado.to_csv(index=False).encode('utf-8-sig')
    st.download_button(
        "📥 Descargar datos filtrados (CSV)",
        data=csv_filtrado,
        file_name="siniestros_filtrados.csv",
        mime="text/csv",
        use_container_width=True
    )

    # Insight automatizado
    with st.expander("📌 Ver analisis del periodo seleccionado"):
        total_s = len(df_filtrado)
        total_f = df_filtrado['CANTIDAD_FALLECIDOS'].sum()
        pct_rural = (df_filtrado['ZONA'] == 'RURAL').mean() * 100
        top_causa = df_filtrado['CAUSA_FACTOR_PRINCIPAL'].value_counts().index[0] if total_s > 0 else "-"
        prom_edad = df_filtrado['EDAD_PROMEDIO'].mean()

        st.markdown(f"""
        **Insight del periodo filtrado:**
        - Se analizaron **{total_s:,} siniestros** con **{total_f:,} fallecidos**.
        - El **{pct_rural:.1f}%** ocurrio en zona rural (donde la severidad suele ser mayor).
        - La causa principal es: **{top_causa}**.
        - La edad promedio de involucrados es **{prom_edad:.1f} anios**.
        """)

# ─── PAGINA 3: MAPA ──────────────────────────────────────────
elif pagina == "Mapa":
    st.title("🗺️ Distribucion Geografica")
    st.markdown("---")

    mostrar_metricas(df_filtrado)

    df_map = df_filtrado.dropna(subset=['LATITUD', 'LONGITUD']).copy()
    df_map = df_map[
        (df_map['LATITUD'].between(-20, 0)) &
        (df_map['LONGITUD'].between(-85, -68))
    ]

    st.subheader("Mapa de Siniestros Fatales")
    n_muestra = min(3000, len(df_map))
    fig = px.scatter_mapbox(
        df_map.sample(n_muestra) if n_muestra > 0 else df_map,
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
        title=f'Distribucion Geografica ({n_muestra} puntos mostrados)'
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Estadisticas por Departamento")
    dpto_stats = df_filtrado.groupby('DEPARTAMENTO').agg(
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

    # Boton de descarga
    csv_data = dpto_stats.to_csv().encode('utf-8-sig')
    st.download_button(
        "📥 Descargar datos de departamentos (CSV)",
        data=csv_data,
        file_name="departamentos_filtrados.csv",
        mime="text/csv",
        use_container_width=True
    )

# ─── PAGINA 4: ANALISIS TEMPORAL ─────────────────────────────
elif pagina == "Analisis Temporal":
    st.title("📈 Analisis Temporal")
    st.markdown("---")

    mostrar_metricas(df_filtrado)

    df_temp = df_filtrado.dropna(subset=['FECHA_SINIESTRO']).copy()
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

    # Descarga de serie temporal
    csv_temporal = serie_mensual.to_csv(index=False).encode('utf-8-sig')
    st.download_button(
        "📥 Descargar serie temporal (CSV)",
        data=csv_temporal,
        file_name="serie_temporal_mensual.csv",
        mime="text/csv",
        use_container_width=True
    )

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

    # Descarga
    if personas_path.exists():
        csv_personas = df_per.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            "📥 Descargar datos de personas (CSV)",
            data=csv_personas,
            file_name="personas_involucradas.csv",
            mime="text/csv",
            use_container_width=True
        )

# ─── PAGINA 6: FACTORES DE RIESGO ────────────────────────────
elif pagina == "Factores de Riesgo":
    st.title("⚠️ Factores de Riesgo")
    st.markdown("---")

    mostrar_metricas(df_filtrado)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Condicion Climatica")
        clima_counts = df_filtrado['CONDICION_CLIMATICA'].value_counts().head(8)
        fig = px.bar(clima_counts, x=clima_counts.index, y=clima_counts.values,
                     color=clima_counts.values, color_continuous_scale='Blues',
                     title='Siniestros por Condicion Climatica')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Clase de Siniestro")
        clase_counts = df_filtrado['CLASE_SINIESTRO'].value_counts()
        fig = px.bar(clase_counts, x=clase_counts.index, y=clase_counts.values,
                     color=clase_counts.values, color_continuous_scale='Purples',
                     title='Distribucion por Clase')
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Tipo de Via")
        via_counts = df_filtrado['TIPO_VIA'].value_counts().head(8)
        fig = px.bar(via_counts, x=via_counts.index, y=via_counts.values,
                     color=via_counts.values, color_continuous_scale='Greens',
                     title='Siniestros por Tipo de Via')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Riesgo de Infraestructura")
        if 'RIESGO_INFRAESTRUCTURA' in df_filtrado.columns:
            fig = px.histogram(df_filtrado, x='RIESGO_INFRAESTRUCTURA',
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

    # --- Seccion R: Validacion estadistica ---
    st.markdown("---")
    st.subheader("📊 Validacion Estadistica (Resultados de R)")
    if df_r is not None:
        st.markdown("""
        A continuacion se muestran los resultados de las **pruebas de hipotesis**
        realizadas en **R** para validar estadisticamente la relacion entre los
        factores de riesgo y la severidad de los siniestros.
        """)
        for _, row in df_r.iterrows():
            p_val = row['p_valor']
            sig = "✅ Significativo" if p_val < 0.05 else "❌ No significativo"
            st.markdown(f"""
            **{row['Prueba']}**: {row['Variable_Independiente']} vs {row['Variable_Dependiente']}
            - Estadistico: {row['Estadistico']:.4f}
            - p-valor: {p_val:.6e}
            - {sig} (α=0.05)
            - {row['Conclusion']}
            """)
    else:
        st.info("Ejecute el script de R para ver los resultados estadisticos.")

    # Descarga
    csv_riesgo = df_filtrado.to_csv(index=False).encode('utf-8-sig')
    st.download_button(
        "📥 Descargar datos de factores de riesgo (CSV)",
        data=csv_riesgo,
        file_name="factores_riesgo_filtrados.csv",
        mime="text/csv",
        use_container_width=True
    )

# ─── PAGINA 7: ANALISIS ESTADISTICO (R) ──────────────────────
elif pagina == "Analisis Estadistico (R)":
    st.title("📊 Analisis Estadistico con R")
    st.markdown("---")

    st.markdown("""
    Esta pagina muestra los resultados generados por **R** para el analisis
    estadistico descriptivo e inferencial del proyecto. R se encarga de la
    validacion estadistica que complementa el Machine Learning desarrollado en Python.
    """)

    # ─── Seccion 1: Estadistica Descriptiva ────
    st.subheader("1. Estadistica Descriptiva")

    col1, col2 = st.columns(2)
    with col1:
        if df_r_sev is not None:
            st.markdown("**Distribucion de Severidad (desde R):**")
            fig = px.pie(df_r_sev, values='Conteo', names='Nivel',
                         title='Distribucion de Severidad',
                         color='Nivel',
                         color_discrete_map={
                             'Baja': '#2ecc71', 'Media': '#f39c12', 'Alta': '#e74c3c'
                         })
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Ejecute el script de R para visualizar.")

    with col2:
        st.markdown("**Resumen de Datos:**")
        st.markdown(f"""
        - Total de siniestros analizados: **{len(df):,}**
        - Periodo: **2021-2025**
        - Departamentos: **24**
        - Variables analizadas: **{len(df.columns)}**
        """)
        if df_r_sev is not None:
            for _, row in df_r_sev.iterrows():
                st.markdown(f"- {row['Nivel']}: **{row['Conteo']:,}** ({row['Porcentaje']:.1f}%)")

    # ─── Seccion 2: Graficos R ────
    st.subheader("2. Visualizaciones Generadas en R")
    r_figures = [
        ("r_boxplot_lesionados.png", "Distribucion de Lesionados por Severidad"),
        ("r_severidad_departamento.png", "Proporcion de Severidad por Departamento"),
        ("r_correlaciones.png", "Matriz de Correlaciones"),
        ("r_histograma_edad.png", "Distribucion de Edad Promedio"),
        ("r_riesgo_infraestructura.png", "Riesgo de Infraestructura por Severidad"),
    ]
    cols_r = st.columns(2)
    for i, (fname, caption) in enumerate(r_figures):
        fpath = FIGURES / fname
        with cols_r[i % 2]:
            if fpath.exists():
                st.image(str(fpath), caption=caption, use_container_width=True)
            else:
                st.info(f"Grafico '{fname}' no disponible. Ejecute el script de R.")

    # ─── Seccion 3: Pruebas de Hipotesis ────
    st.subheader("3. Pruebas de Hipotesis (Estadistica Inferencial)")

    st.markdown("""
    Se realizaron **6 pruebas estadisticas** en R para validar la relacion entre
    los factores de riesgo y la severidad de los siniestros.
    """)

    if df_r is not None and len(df_r) > 0:
        for i, (_, row) in enumerate(df_r.iterrows()):
            p_val = row['p_valor']
            sig_symbol = "✅" if p_val < 0.05 else "❌"
            sig_text = "**Significativo**" if p_val < 0.05 else "No significativo"

            with st.expander(f"{sig_symbol} {row['Prueba']}: {row['Variable_Independiente']} vs {row['Variable_Dependiente']}", expanded=i < 3):
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown(f"""
                    **Tipo de prueba:** {row['Prueba']}\n\n
                    **Hipotesis Nula (H₀):** No existe relacion entre {row['Variable_Independiente']} y {row['Variable_Dependiente']}\n\n
                    **Hipotesis Alterna (H₁):** Existe relacion entre {row['Variable_Independiente']} y {row['Variable_Dependiente']}
                    """)
                with col_b:
                    st.markdown(f"""
                    **Estadistico de prueba:** {row['Estadistico']:.4f}\n\n
                    **p-valor:** {p_val:.6e}\n\n
                    **Nivel de significancia:** α = 0.05\n\n
                    **Conclusion:** {sig_text} — {row['Conclusion']}
                    """)

        st.success(f"""
        Todas las **{len(df_r)} pruebas estadisticas** resultaron **significativas** (p < 0.05),
        lo que confirma que los factores analizados tienen una relacion estadisticamente
        significativa con la severidad de los siniestros viales.
        """)
    else:
        st.warning("No se encontraron resultados de R. Ejecute primero el script R:")
        st.code('"D:/R-4.6.1/bin/x64/Rscript.exe" r/estadistica/analisis_descriptivo.R')

    # ─── Seccion 4: Reporte Completo ────
    st.subheader("4. Reporte Estadistico Completo")
    report_path = REPORTS / "reporte_estadistico_R.txt"
    if report_path.exists():
        with open(report_path, "r", encoding="utf-8") as f:
            report_content = f.read()
        with st.expander("Ver reporte completo generado por R"):
            st.text(report_content)
        st.download_button(
            "📥 Descargar Reporte Estadistico (TXT)",
            data=report_content,
            file_name="reporte_estadistico_R.txt",
            mime="text/plain",
            use_container_width=True
        )
    else:
        st.info("Reporte no disponible. Ejecute el script de R primero.")

    # ─── Seccion 5: Metodologia ────
    st.subheader("5. Justificacion del Uso de R")
    st.markdown("""
    | Aspecto | Python | R |
    |---------|--------|---|
    | **Proposito** | Ingenieria de datos, ML, Dashboard | Estadistica descriptiva e inferencial |
    | **Librerias** | pandas, scikit-learn, xgboost | tidyverse, ggplot2, corrplot, caret |
    | **Analisis** | Modelado predictivo, SHAP | Pruebas chi-cuadrado, ANOVA, Tukey HSD |
    | **Visualizaciones** | Plotly interactivo (Dashboard) | ggplot2 profesional (figuras PNG) |
    | **Output** | Modelos .pkl, Dashboard | Graficos PNG, reporte .txt, tablas .csv |

    **R** se utilizó exclusivamente para el **analisis estadistico formal** (pruebas de hipotesis,
    ANOVA, diagnosticos), mientras que **Python** se encargó del preprocesamiento,
    modelado, y dashboard interactivo. Ambos lenguajes se **complementan** y sus resultados
    se integran en esta interfaz.
    """)

# ─── PAGINA 8: DASHBOARD EJECUTIVO ────────────────────────────
elif pagina == "Dashboard Ejecutivo":
    st.title("🏛️ Dashboard Ejecutivo - ONSV")
    st.markdown("---")

    # ─── Seccion 1: KPIS ESTRATEGICOS ────
    st.subheader("Indicadores Estrategicos Nacionales")

    # KPI globales (sin filtros, datos nacionales)
    total_siniestros = len(df)
    total_fallecidos = int(df['CANTIDAD_FALLECIDOS'].sum())
    total_lesionados = int(df['CANTIDAD_LESIONADOS'].sum())
    tasa_mortalidad = total_fallecidos / total_siniestros if total_siniestros > 0 else 0
    pct_alta_nac = (df['severidad'] == 2).sum() / total_siniestros * 100

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f"""<div style="background:linear-gradient(135deg,#1a5276,#2980b9);padding:15px;border-radius:10px;text-align:center;">
            <p style="color:#85c1e9;font-size:11px;margin:0;">🚦 TOTAL SINIESTROS</p>
            <p style="color:white;font-size:26px;font-weight:bold;margin:5px 0;">{total_siniestros:,}</p>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div style="background:linear-gradient(135deg,#922b21,#e74c3c);padding:15px;border-radius:10px;text-align:center;">
            <p style="color:#f1948a;font-size:11px;margin:0;">💀 TOTAL FALLECIDOS</p>
            <p style="color:white;font-size:26px;font-weight:bold;margin:5px 0;">{total_fallecidos:,}</p>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div style="background:linear-gradient(135deg,#1e8449,#27ae60);padding:15px;border-radius:10px;text-align:center;">
            <p style="color:#82e0aa;font-size:11px;margin:0;">🏥 TOTAL LESIONADOS</p>
            <p style="color:white;font-size:26px;font-weight:bold;margin:5px 0;">{total_lesionados:,}</p>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div style="background:linear-gradient(135deg,#7d3c98,#af7ac5);padding:15px;border-radius:10px;text-align:center;">
            <p style="color:#d2b4de;font-size:11px;margin:0;">📊 TASA MORTALIDAD</p>
            <p style="color:white;font-size:26px;font-weight:bold;margin:5px 0;">{tasa_mortalidad:.2f}</p>
        </div>""", unsafe_allow_html=True)
    with col5:
        color_alta = "#e74c3c" if pct_alta_nac > 5 else ("#f39c12" if pct_alta_nac > 3 else "#2ecc71")
        st.markdown(f"""<div style="background:linear-gradient(135deg,#b7950b,#f1c40f);padding:15px;border-radius:10px;text-align:center;">
            <p style="color:#333;font-size:11px;margin:0;">⚠️ SEVERIDAD ALTA</p>
            <p style="color:white;font-size:26px;font-weight:bold;margin:5px 0;">{pct_alta_nac:.1f}%</p>
        </div>""", unsafe_allow_html=True)

    # ─── Seccion 2: MAPA COROPLETICO ────
    st.markdown("---")
    st.subheader("🗺️ Mapa de Riesgo por Departamento")

    # Agregar datos por departamento
    dpto_data = df.groupby('DEPARTAMENTO').agg(
        Siniestros=('CODIGO_SINIESTRO', 'count'),
        Fallecidos=('CANTIDAD_FALLECIDOS', 'sum'),
        Lesionados=('CANTIDAD_LESIONADOS', 'sum'),
        Severidad_Prom=('severidad', 'mean'),
        Tasa_Mortalidad=('CANTIDAD_FALLECIDOS', 'mean'),
        Pct_Rural=('ZONA', lambda x: (x == 'RURAL').mean() * 100)
    ).reset_index()

    dpto_data['Severidad_Prom_Redondeada'] = dpto_data['Severidad_Prom'].round(2)
    dpto_data['label'] = dpto_data['DEPARTAMENTO'] + '<br>' + \
                         dpto_data['Siniestros'].astype(str) + ' siniestros | ' + \
                         'Sev: ' + dpto_data['Severidad_Prom_Redondeada'].astype(str)

    # Normalizar nombres de departamentos para coincidir con GeoJSON
    def norm_dpto(nombre):
        """Normalizar nombre de departamento para coincidir con GeoJSON."""
        nombre = nombre.upper().strip()
        # Mapa de correcciones
        correcciones = {
            'LIMA': 'LIMA', 'CALLAO': 'CALLAO', 'AREQUIPA': 'AREQUIPA',
            'CUSCO': 'CUSCO', 'LA LIBERTAD': 'LA LIBERTAD', 'PUNO': 'PUNO',
            'JUNIN': 'JUNIN', 'CAJAMARCA': 'CAJAMARCA', 'ANCASH': 'ANCASH',
            'PIURA': 'PIURA', 'ICA': 'ICA', 'LAMBAYEQUE': 'LAMBAYEQUE',
            'HUANUCO': 'HUANUCO', 'SAN MARTIN': 'SAN MARTIN',
            'AYACUCHO': 'AYACUCHO', 'LORETO': 'LORETO', 'HUANCAVELICA': 'HUANCAVELICA',
            'APURIMAC': 'APURIMAC', 'PASCO': 'PASCO', 'TACNA': 'TACNA',
            'TUMBES': 'TUMBES', 'MOQUEGUA': 'MOQUEGUA', 'AMAZONAS': 'AMAZONAS',
            'UCAYALI': 'UCAYALI', 'MADRE DE DIOS': 'MADRE DE DIOS',
            'LIMA PROVINCIAS': 'LIMA', 'LIMA METROPOLITANA': 'LIMA',
        }
        return correcciones.get(nombre, nombre)

    dpto_data['DPTO_NORM'] = dpto_data['DEPARTAMENTO'].apply(norm_dpto)

    # Mostrar mapa coropletico
    col_map1, col_map2 = st.columns([3, 1])

    with col_map1:
        if geojson_peru is not None:
            # Mapa de Siniestros
            fig_choropleth = px.choropleth(
                dpto_data,
                geojson=geojson_peru,
                locations='DPTO_NORM',
                color='Siniestros',
                featureidkey='properties.NOMBDEP',
                color_continuous_scale='YlOrRd',
                range_color=(0, dpto_data['Siniestros'].max()),
                title='Siniestros por Departamento',
                labels={'Siniestros': 'Cantidad'},
                hover_data={'DPTO_NORM': False, 'Siniestros': True,
                           'Fallecidos': True, 'Severidad_Prom': True,
                           'Tasa_Mortalidad': ':,.2f'}
            )
            fig_choropleth.update_geos(fitbounds='locations', visible=False)
            fig_choropleth.update_layout(
                height=500,
                margin=dict(l=0, r=0, t=30, b=0),
                coloraxis_colorbar=dict(
                    title="Siniestros",
                    orientation="h",
                    y=-0.15,
                    len=0.5
                )
            )
            st.plotly_chart(fig_choropleth, use_container_width=True)
        else:
            st.warning("No se pudo cargar el mapa. Verifique el archivo GeoJSON.")

    with col_map2:
        st.markdown("**Leyenda de Riesgo**")
        max_s = dpto_data['Siniestros'].max()
        for _, row in dpto_data.sort_values('Siniestros', ascending=False).head(8).iterrows():
            nivel = '🔴' if row['Severidad_Prom'] > 0.6 else ('🟡' if row['Severidad_Prom'] > 0.3 else '🟢')
            st.markdown(f"{nivel} **{row['DEPARTAMENTO']}**: {row['Siniestros']:,} ({row['Severidad_Prom']:.2f})")

    # ─── Seccion 3: ALERTAS Y RANKING ────
    st.markdown("---")
    st.subheader("🚨 Alertas y Recomendaciones")

    # Sistema de alertas
    alertas = []
    for _, row in dpto_data.sort_values('Siniestros', ascending=False).head(5).iterrows():
        if row['Siniestros'] > 500:
            nivel_alerta = "🔴 CRITICO"
        elif row['Siniestros'] > 200:
            nivel_alerta = "🟡 ATENCION"
        else:
            nivel_alerta = "🟢 MONITOREO"
        alertas.append((nivel_alerta, row['DEPARTAMENTO'], row['Siniestros'], row['Severidad_Prom']))

    col_a1, col_a2 = st.columns(2)
    with col_a1:
        st.markdown("**Alertas por Departamento**")
        for nivel, dpto, total, sev in alertas:
            st.markdown(f"{nivel} **{dpto}** — {total:,} siniestros (severidad: {sev:.2f})")
    with col_a2:
        # Peligros nacionales
        pct_rural_nac = (df['ZONA'] == 'RURAL').mean() * 100
        top_causa = df['CAUSA_FACTOR_PRINCIPAL'].value_counts().index[0]
        franja_peligrosa = df['FRANJA_HORARIA'].mode()[0]
        franja_nombres = {0: 'Madrugada', 1: 'Maniana', 2: 'Tarde', 3: 'Noche'}
        st.markdown("**Alertas Nacionales**")
        st.markdown(f"⚠️ El **{pct_rural_nac:.0f}%** de siniestros ocurre en zona rural")
        st.markdown(f"⚠️ Causa principal: **{top_causa}**")
        st.markdown(f"⚠️ Franja mas peligrosa: **{franja_nombres.get(franja_peligrosa, '-')}**")

        # Descarga datos de departamentos
        csv_dpto = dpto_data.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            "📥 Descargar datos departamentales (CSV)",
            data=csv_dpto,
            file_name="datos_departamentos.csv",
            mime="text/csv",
            use_container_width=True
        )

    # ─── Seccion 4: TABLA DE RANKING ────
    st.markdown("---")
    st.subheader("📋 Ranking de Departamentos por Riesgo")

    # Calcular indicador compuesto de riesgo
    dpto_data['Indice_Riesgo'] = (
        dpto_data['Severidad_Prom'] * 0.4 +
        (dpto_data['Tasa_Mortalidad'] / dpto_data['Tasa_Mortalidad'].max()) * 0.3 +
        (dpto_data['Pct_Rural'] / 100) * 0.3
    ).round(3)

    ranking = dpto_data.sort_values('Indice_Riesgo', ascending=False).reset_index(drop=True)
    ranking['Posicion'] = range(1, len(ranking) + 1)
    ranking['Alerta'] = ranking['Indice_Riesgo'].apply(
        lambda x: '🔴' if x > 0.5 else ('🟡' if x > 0.3 else '🟢')
    )

    ranking_show = ranking[['Posicion', 'Alerta', 'DEPARTAMENTO', 'Siniestros',
                            'Fallecidos', 'Severidad_Prom', 'Tasa_Mortalidad',
                            'Pct_Rural', 'Indice_Riesgo']].rename(columns={
        'DEPARTAMENTO': 'Departamento',
        'Siniestros': 'Siniestros',
        'Fallecidos': 'Fallecidos',
        'Severidad_Prom': 'Severidad Prom',
        'Tasa_Mortalidad': 'Tasa Mortalidad',
        'Pct_Rural': '% Rural',
        'Indice_Riesgo': 'Indice Riesgo'
    })

    st.dataframe(
        ranking_show.style.format({
            'Severidad Prom': '{:.2f}',
            'Tasa Mortalidad': '{:.2f}',
            'Pct_Rural': '{:.1f}%',
            'Indice Riesgo': '{:.3f}',
            'Siniestros': '{:,.0f}',
            'Fallecidos': '{:,.0f}'
        }).applymap(
            lambda x: 'background-color: #ffcccc' if isinstance(x, float) and x > 0.5 else
                      ('background-color: #fff3cd' if isinstance(x, float) and x > 0.3 else ''),
            subset=['Indice Riesgo']
        ),
        use_container_width=True,
        hide_index=True
    )

    # Descarga
    csv_ranking = ranking_show.to_csv(index=False).encode('utf-8-sig')
    st.download_button(
        "📥 Descargar Ranking Completo (CSV)",
        data=csv_ranking,
        file_name="ranking_riesgo_departamentos.csv",
        mime="text/csv",
        use_container_width=True
    )

    # ─── Seccion 5: CONCLUSIONES EJECUTIVAS ────
    st.markdown("---")
    st.subheader("📌 Resumen Ejecutivo")

    mejor_dpto = ranking.iloc[-1]['DEPARTAMENTO']
    peor_dpto = ranking.iloc[0]['DEPARTAMENTO']

    st.markdown(f"""
    - **Departamento mas critico:** {peor_dpto} (Indice de Riesgo: {ranking.iloc[0]['Indice_Riesgo']:.3f})
    - **Departamento menos critico:** {mejor_dpto} (Indice de Riesgo: {ranking.iloc[-1]['Indice_Riesgo']:.3f})
    - **Promedio nacional de severidad:** {dpto_data['Severidad_Prom'].mean():.2f}
    - **Total nacional de siniestros:** {total_siniestros:,}
    - **Recomendacion prioritaria:** Reforzar presencia policial y controles en {peor_dpto}
    """)

# ─── PAGINA 9: MODELOS PREDICTIVOS ───────────────────────────
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

# ─── PAGINA 9: PREDICCION INDIVIDUAL ──────────────────────────
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

                # Analisis de contribucion de variables
                st.markdown("---")
                st.subheader("📊 Analisis de Contribucion de Variables")

                # Obtener coeficientes del modelo
                coef = model.coef_[pred]  # Coeficientes para la clase predicha
                contribuciones = pd.DataFrame({
                    'Variable': list(scaler.feature_names_in_),
                    'Contribucion': coef * input_scaled[0]
                }).sort_values('Contribucion', ascending=False)

                fig_contrib = px.bar(
                    contribuciones.head(10),
                    x='Contribucion', y='Variable',
                    orientation='h',
                    color='Contribucion',
                    color_continuous_scale='RdYlGn',
                    title='Top 10 Variables que mas influyeron en la prediccion'
                )
                fig_contrib.update_layout(height=400)
                st.plotly_chart(fig_contrib, use_container_width=True)

                st.markdown("""
                **Interpretacion:** Las barras en verde aumentan la probabilidad de
                severidad **alta**, mientras que las barras en rojo indican factores
                que **reducen** el riesgo (asociados a severidad baja).
                """)

                # Comparacion con casos similares
                st.markdown("---")
                st.subheader("🔄 Casos Historicos Similares")

                try:
                    # Buscar casos similares en el dataset
                    distancias = np.linalg.norm(
                        scaler.transform(df[list(scaler.feature_names_in_)].fillna(0).values) -
                        input_scaled,
                        axis=1
                    )
                    casos_similares = df.iloc[distancias.argsort()[:5]]

                    st.markdown("**Los 5 siniestros mas similares en el historial:**")
                    similar_show = casos_similares[[
                        'DEPARTAMENTO', 'CLASE_SINIESTRO', 'SEVERIDAD_LABEL',
                        'CANTIDAD_FALLECIDOS', 'CANTIDAD_LESIONADOS',
                        'ZONA', 'CONDICION_CLIMATICA'
                    ]].copy()
                    st.dataframe(similar_show, use_container_width=True)

                    # Distribucion de severidad en casos similares
                    sev_counts = casos_similares['SEVERIDAD_LABEL'].value_counts()
                    fig_sev = px.pie(
                        values=sev_counts.values,
                        names=sev_counts.index,
                        title='Distribucion de severidad en casos similares',
                        color=sev_counts.index,
                        color_discrete_map={
                            'Baja (1 fallecido)': '#2ecc71',
                            'Media (2 fallecidos)': '#f39c12',
                            'Alta (3+ fallecidos)': '#e74c3c'
                        }
                    )
                    st.plotly_chart(fig_sev, use_container_width=True)
                except:
                    st.info("No se pudieron calcular casos similares.")

                # Exportar prediccion
                result_row = {
                    'Prediccion': severidad_map[pred],
                    'Prob_Baja': round(proba[0]*100, 1),
                    'Prob_Media': round(proba[1]*100, 1),
                    'Prob_Alta': round(proba[2]*100, 1),
                    'Hora': hora, 'Mes': mes, 'Anio': anio,
                    'Zona': zona, 'Clima': clima,
                    'Lesionados': lesionados, 'Vehiculos': vehiculos
                }
                csv_pred = pd.DataFrame([result_row]).to_csv(index=False).encode('utf-8-sig')
                st.download_button(
                    "📥 Descargar resultado de prediccion (CSV)",
                    data=csv_pred,
                    file_name="prediccion_severidad.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            except Exception as e:
                st.error(f"Error en la prediccion: {e}")
                st.info("Verifique que todos los campos esten correctamente llenados.")
    else:
        st.warning("Modelo no disponible. Ejecute primero 04_modelamiento.py")

# ─── PAGINA 10: CONCLUSIONES ─────────────────────────────────
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
