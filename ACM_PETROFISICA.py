# -*- coding: utf-8 -*-
""""
Created on Thu Oct 31 17:35:14 2024

@author: juan.melendez
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import streamlit as st
from pyproj import Proj, Transformer
import numpy as np

# CONFIGURACIÓN DE LA PÁGINA STREAMLIT
def configure_page():
    st.set_page_config(page_title="PETROFISICA ACM", layout="wide")
    st.markdown("<h1 style='text-align: center; color: black;'>ANÁLISIS PETROFÍSICO DE ACM</h1>", 
                unsafe_allow_html=True)
    st.sidebar.markdown("<h2 style='color: gray;'>⚙️ CONTROLADORES</h2>", 
                        unsafe_allow_html=True)
    st.sidebar.markdown("<p style='text-align: center;'>---------------------------------</p>", 
                        unsafe_allow_html=True)

# CARGA DE ARCHIVO
def load_data():
    df_or = st.sidebar.file_uploader("📂", type=["csv", "CSV", "TXT", "txt"])
    if df_or:
        return pd.read_csv(df_or, sep=",")
    st.error("ARCHIVO NO CARGADO ❗❗")
    st.stop()

def process_data(df_loaded):

    if df_loaded is None:
        raise ValueError("df_loaded no cargó correctamente.")
      
    # Diccionario de tipos de datos para cada columna
    dtypes = {
        'POZO IP': 'str',              # Texto para identificador de pozo
        'YACIMIENTO': 'str',           # Texto para nombre de yacimiento
        'CIMA TVD': 'float64',         # Profundidad, valores decimales
        'BASE TVD': 'float64',
        'ESPESOR BRUTO MV': 'float64',
        'ESPESOR NETO MV': 'float64',
        'RELACION NB': 'float64',  
        'ESPESOR BRUTO MD': 'float64',
        'ESPESOR NETO MD': 'float64',
        'PHIE DEC': 'float64',         # Porosidad efectiva decimal
        'SW DEC': 'float64',           # Saturación de agua decimal
        'VCL DEC': 'float64',          # Volumen de arcilla decimal
        'KMTZ MD': 'float64',          # Permeabilidad matriz decimal
        'KFRAC MD': 'float64',         # Permeabilidad fracturas decimal
        'MAESTRA CAMPANA': 'str',
        'TIPO SUMARIO': 'str',
        'TIPO DE POZO': 'str',
        'MAESTRA CAMPANA': 'str',
        "XCOOR_OBJ": 'float64',
        "YCOOR_OBJ": 'float64',
        'EVALUACION': 'str'
    }

    df_loaded.replace(' ', np.nan, inplace=True) 
    df_loaded = df_loaded.astype(dtypes)    
     
    df_pozos_coor = df_loaded[["POZO IP", "XCOOR_OBJ","YCOOR_OBJ"]].drop_duplicates()
    df_tabla = df_loaded[['POZO IP','YACIMIENTO','CIMA TVD','BASE TVD',
                            'ESPESOR BRUTO MV','ESPESOR NETO MV','RELACION NB',	
                            'ESPESOR BRUTO MD','ESPESOR NETO MD',	
                            'PHIE DEC',	'SW DEC','VCL DEC','KMTZ MD','KFRAC MD',	
                            'TIPO SUMARIO','TIPO DE POZO','MAESTRA CAMPANA', "XCOOR_OBJ","YCOOR_OBJ", 'EVALUACION']]
    
    # Filtra las filas donde 'EVALUACION' es igual a 'CON EVALUACION'
    df_evaluacion = df_tabla[df_tabla['EVALUACION'] == 'CON EVALUACION']   
   
    # Definir el transformador para convertir de UTM a Lat/Long
    transformer = Transformer.from_crs(
        "epsg:32614",  # UTM Zone 14N WGS84 (ajustar según tu zona UTM)
        "epsg:4326",   # WGS84
        always_xy=True
    )
       
    # Función para convertir UTM a Lat/Long
    def utm_to_latlon(utm_easting, utm_northing):
        lon, lat = transformer.transform(utm_easting, utm_northing)
        return lat, lon
    
    # Aplicar la conversión de UTM a Lat/Long para los pozos
    df_pozos_coor[['Latitude', 'Longitude']] = df_pozos_coor.apply(
        lambda row: utm_to_latlon(row['XCOOR_OBJ'], row['YCOOR_OBJ']), 
        axis=1, result_type='expand'
    )
        
    # Aplicar la conversión de UTM a Lat/Long para los pozos
    df_evaluacion[['Latitude', 'Longitude']] = df_evaluacion.apply(
        lambda row: utm_to_latlon(row['XCOOR_OBJ'], row['YCOOR_OBJ']), 
        axis=1, result_type='expand'
    )
    
    # Definir y convertir las coordenadas UTM para el polígono
    polygon_utm_coords = [
        (629254, 2294990),
        (629205, 2305136),
        (643050, 2305249),
        (643137, 2295102),
        (629254, 2294990)
        
    ]
    polygon_latlon = [utm_to_latlon(easting, northing) for easting, northing in polygon_utm_coords]
    
    # Separar las coordenadas en latitudes y longitudes
    polygon_lats, polygon_lons = zip(*polygon_latlon)
    zoom = 1    
  
    return df_pozos_coor, df_evaluacion, polygon_lats, polygon_lons, zoom

def plot_density_map(df,df_p, variable, polygon_lats, polygon_lons,color_continuous_scale,zoom):
    # Ajusta el factor de escala según el nivel de detalle que necesites
    radius = max(2, 25 - zoom * 1.5)  # Ejemplo de fórmula que disminuye el radio a medida que el zoom aumenta

    # Convertir a numérico, ignorando errores
    df[variable] = pd.to_numeric(df[variable], errors='coerce')
    # Eliminar filas con NaN en la columna de la variable seleccionada
    df = df.dropna(subset=[variable])
    
    # Definir el rango de colores basado en los valores de la variable seleccionada
    min_val = 0
    max_val = df[variable].max()
    range_color = [min_val, max_val]
    
    # Crear el mapa de densidad
    fig = px.density_mapbox(df, lat='Latitude', lon='Longitude', z=variable, radius=radius,
                            center=dict(lat=df['Latitude'].mean(), lon=df['Longitude'].mean()), 
                            zoom=zoom,  # Ajustar el nivel de zoom según sea necesario
                            mapbox_style="carto-positron",
                            color_continuous_scale=color_continuous_scale,
                            opacity=0.9,
                            range_color=range_color,
                            title=f"Mapa de Densidad de {variable}")

    # Agregar los marcadores de todos los pozos
    well_coort = go.Scattermapbox(
        lat=df_p['Latitude'],
        lon=df_p['Longitude'],
        mode='markers',
        marker=go.scattermapbox.Marker(
            size=6,
            color='black',
            opacity=1
        ),
        text=df_p['POZO IP'],  # Texto emergente
        hoverinfo='text',
        name='Pozos'  # Nombre de la leyenda
    )    

    # Agregar los marcadores de los pozos filtrados
    well_coorf = go.Scattermapbox(
        lat=df['Latitude'],
        lon=df['Longitude'],
        mode='markers',
        marker=go.scattermapbox.Marker(
            size=6,
            color='red',
            opacity=1
        ),
        text=df['POZO IP'],  # Texto emergente
        hoverinfo='text',
        name='Seleccionados' # Nombre de la leyenda
    )
    
    # Agregar el polígono
    polygon_trace = go.Scattermapbox(
        fill="none",
        lat=polygon_lats,
        lon=polygon_lons,
        mode="lines",
        line=dict(width=2, color="black"),
        hoverinfo='none',
        opacity=1,
        name='ACE'  # Nombre de la leyenda
    )
    
    # Combinar el mapa de densidad con los marcadores y el polígono
    fig.add_trace(well_coort)
    fig.add_trace(well_coorf)
    fig.add_trace(polygon_trace)
    
    # Actualizar el diseño del gráfico
    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=20.795, lon=-97.691),
            zoom=11.4,
        ),
        margin={"r":1,"t":1,"l":1,"b":1},
        height=800,
        width=200,
        title_text=f"Mapa de Densidad de {variable}"
    )
    
    return fig

def main():
    # Configura la página y carga los datos
    configure_page()
    df_loaded = load_data()
    
    with st.spinner("Procesando datos..."):
        df_pozos_coor, df_evaluacion, polygon_lats, polygon_lons, zoom = process_data(df_loaded)
        
        # Crea las pestañas de la interfaz
        tabs = st.tabs(["YACIMIENTO", "POZO", "SUMARIO"])
        
        # Filtros desde la barra lateral

        
        # Pestaña "PRODUCCIÓN ACUMULADA"
        with tabs[0]:
            # Mostrar en cuatro columnas
            colu1, colu2, colu3 = st.columns(3)
            with colu1:
            
                well_info_expand = st.expander("INFORMACIÓN BÁSICA DE YACIMIENTOS")
                a1, a2 = well_info_expand.columns(2)
                df_eval = df_evaluacion.copy()
                
                ms_sumario_type = a1.multiselect("SELECCIONA EL TIPO DE SUMARIO ",  df_eval["TIPO SUMARIO"].unique(), default=[])
                df_sumario_type =  df_eval[ df_eval["TIPO SUMARIO"].isin(ms_sumario_type)]
                
                ms_yacimiento_type = a1.multiselect("SELECCIONA EL YACIMIENTO ", df_sumario_type["YACIMIENTO"].unique(), default=[])
                df_yacimiento_type = df_sumario_type[df_sumario_type["YACIMIENTO"].isin(ms_yacimiento_type)]
                
                ms_pozo_type = a2.multiselect("SELECCIONA EL POZO TIPO ", df_yacimiento_type["TIPO DE POZO"].unique(), default=[])
                df_pozo_type = df_yacimiento_type[df_yacimiento_type["TIPO DE POZO"].isin(ms_pozo_type)]
                
                #st.write(df_pozo_type)
                
                # selected_columns_promedio = ['ESPESOR BRUTO MV', 'ESPESOR NETO MV',
                #                              'RELACION NB','PHIE DEC','SW DEC', 
                #                              'VCL DEC', 'KMTZ MD', 'KFRAC MD']
                
                options_yac = []
                options_yac = a2.multiselect("SELECCIONA LA VARIABLE ",['ESPESOR BRUTO MV', 'ESPESOR NETO MV',
                                         'RELACION NB','PHIE DEC','SW DEC','VCL DEC', 'KMTZ MD', 'KFRAC MD'], default=[])
                
                if options_yac:
                     figNp = plot_density_map(df_pozo_type, df_pozos_coor, options_yac[0], polygon_lats, polygon_lons, 'turbo', zoom)
                     st.plotly_chart(figNp, use_container_width=True, key="figNp_key")
                else:
                     st.warning("Por favor, selecciona al menos una variable para el mapa de densidad.")
                                        
            with colu2:
                # Crear el histograma
                histograma_bruto = px.histogram(df_pozo_type, x="ESPESOR BRUTO MV", nbins=50, color_discrete_sequence=["#C89058"])
                
                # Configuraciones adicionales
                histograma_bruto.update_layout(
                    title="ESPESOR BRUTO",
                    xaxis_title="ESPESOR BRUTO (mv)",
                    yaxis_title="FRECUENCIA",
                    plot_bgcolor="white",
                    width=1000,  # Ancho en píxeles
                    height=250  # Alto en píxeles
                )
                
                # Configurar el rango del eje Y (opcional)
                #fig_bruto.update_yaxes(range=[0, 20])
                
                # Configurar título y etiquetas de los ejes
                histograma_bruto.update_layout(
                    title={
                        'text': "ESPESOR BRUTO",
                        'font': {'size': 20, 'color': '#000000'},  # Tamaño y color del encabezado
                        'x': 0.30  # Centra el título
                    },
                    xaxis_title={
                        'text': "BRUTO (mv)",
                        'font': {'size': 15, 'color': '#000000'}  # Tamaño y color del título del eje X
                    },
                    yaxis_title={
                        'text': "FRECUENCIA",
                        'font': {'size': 15, 'color': '#000000'}  # Tamaño y color del título del eje Y
                    },
                )
                
                # Configurar color y tamaño de etiquetas de los ejes
                histograma_bruto.update_xaxes(
                    tickfont={'size': 14, 'color': '#000000'}  # Tamaño y color de las etiquetas del eje X
                )
                histograma_bruto.update_yaxes(
                    tickfont={'size': 14, 'color': '#000000'}  # Tamaño y color de las etiquetas del eje Y
                )
                
                
                
                
                histograma_neto = px.histogram(df_pozo_type, x="ESPESOR NETO MV", nbins=50, color_discrete_sequence=["#99CC00"])
                
                # Configuraciones adicionales
                histograma_neto.update_layout(
                    title="ESPESOR NETO",
                    xaxis_title="ESPESOR NETO (mv)",
                    yaxis_title="FRECUENCIA",
                    plot_bgcolor="white",
                    width=1000,  # Ancho en píxeles
                    height=250  # Alto en píxeles
                )
                
                # Configurar el rango del eje Y (opcional)
                #fig_bruto.update_yaxes(range=[0, 20])
                
                # Configurar título y etiquetas de los ejes
                histograma_neto.update_layout(
                    title={
                        'text': "ESPESOR NETO",
                        'font': {'size': 20, 'color': '#000000'},  # Tamaño y color del encabezado
                        'x': 0.30  # Centra el título
                    },
                    xaxis_title={
                        'text': "NETO (mv)",
                        'font': {'size': 15, 'color': '#000000'}  # Tamaño y color del título del eje X
                    },
                    yaxis_title={
                        'text': "FRECUENCIA",
                        'font': {'size': 15, 'color': '#000000'}  # Tamaño y color del título del eje Y
                    },
                )
                
                # Configurar color y tamaño de etiquetas de los ejes
                histograma_neto.update_xaxes(
                    tickfont={'size': 14, 'color': '#000000'}  # Tamaño y color de las etiquetas del eje X
                )
                histograma_neto.update_yaxes(
                    tickfont={'size': 14, 'color': '#000000'}  # Tamaño y color de las etiquetas del eje Y
                )
                
                
                
                
                histograma_nb = px.histogram(df_pozo_type, x="RELACION NB", nbins=50, color_discrete_sequence=["#D7D200"])
                
                # Configuraciones adicionales
                histograma_nb.update_layout(
                    title="RELACION DE ESPESOR NETO-BRUTO",
                    xaxis_title="RELACION DE N/B (dec)",
                    yaxis_title="FRECUENCIA",
                    plot_bgcolor="white",
                    width=1000,  # Ancho en píxeles
                    height=250  # Alto en píxeles
                )
                
                # Configurar el rango del eje Y (opcional)
                #fig_bruto.update_yaxes(range=[0, 20])
                
                # Configurar título y etiquetas de los ejes
                histograma_nb.update_layout(
                    title={
                        'text': "RELACIÓN DE ESPESOR NETO-BRUTO",
                        'font': {'size': 20, 'color': '#000000'},  # Tamaño y color del encabezado
                        'x': 0.30  # Centra el título
                    },
                    xaxis_title={
                        'text': "RELACION DE N/B (dec)",
                        'font': {'size': 15, 'color': '#000000'}  # Tamaño y color del título del eje X
                    },
                    yaxis_title={
                        'text': "FRECUENCIA",
                        'font': {'size': 15, 'color': '#000000'}  # Tamaño y color del título del eje Y
                    },
                )
                
                # Configurar color y tamaño de etiquetas de los ejes
                histograma_nb.update_xaxes(
                    tickfont={'size': 14, 'color': '#000000'}  # Tamaño y color de las etiquetas del eje X
                )
                histograma_nb.update_yaxes(
                    tickfont={'size': 14, 'color': '#000000'}  # Tamaño y color de las etiquetas del eje Y
                )
                
                
                
                histograma_poro = px.histogram(df_pozo_type, x="PHIE DEC", nbins=50, color_discrete_sequence=["#30BD0F"])
                
                # Configuraciones adicionales
                histograma_poro.update_layout(
                    title="POROSIDAD",
                    xaxis_title="POROSIDAD (dec)",
                    yaxis_title="FRECUENCIA",
                    plot_bgcolor="white",
                    width=1000,  # Ancho en píxeles
                    height=250  # Alto en píxeles
                )
                
                # Configurar el rango del eje Y (opcional)
                #fig_bruto.update_yaxes(range=[0, 20])
                
                # Configurar título y etiquetas de los ejes
                histograma_poro.update_layout(
                    title={
                        'text': "POROSIDAD",
                        'font': {'size': 20, 'color': '#000000'},  # Tamaño y color del encabezado
                        'x': 0.30  # Centra el título
                    },
                    xaxis_title={
                        'text': "PHIE (dec)",
                        'font': {'size': 15, 'color': '#000000'}  # Tamaño y color del título del eje X
                    },
                    yaxis_title={
                        'text': "FRECUENCIA",
                        'font': {'size': 15, 'color': '#000000'}  # Tamaño y color del título del eje Y
                    },
                )
                
                # Configurar color y tamaño de etiquetas de los ejes
                histograma_poro.update_xaxes(
                    tickfont={'size': 14, 'color': '#000000'}  # Tamaño y color de las etiquetas del eje X
                )
                histograma_poro.update_yaxes(
                    tickfont={'size': 14, 'color': '#000000'}  # Tamaño y color de las etiquetas del eje Y
                )
                
                st.write(histograma_bruto)
                st.write(histograma_neto)
                st.write(histograma_nb)
                st.write(histograma_poro)
            
            with colu3:    
                                
                histograma_sw = px.histogram(df_pozo_type, x="SW DEC", nbins=50, color_discrete_sequence=["#66CCFF"])
                
                # Configuraciones adicionales
                histograma_sw.update_layout(
                    title="SATURACIÓN DE AGUA",
                    xaxis_title="SW (dec)",
                    yaxis_title="FRECUENCIA",
                    plot_bgcolor="white",
                    width=1000,  # Ancho en píxeles
                    height=250  # Alto en píxeles
                )
                
                # Configurar el rango del eje Y (opcional)
                #fig_bruto.update_yaxes(range=[0, 20])
                
                # Configurar título y etiquetas de los ejes
                histograma_sw.update_layout(
                    title={
                        'text': "SATURACIÓN DE AGUA",
                        'font': {'size': 20, 'color': '#000000'},  # Tamaño y color del encabezado
                        'x': 0.30  # Centra el título
                    },
                    xaxis_title={
                        'text': "SW (dec)",
                        'font': {'size': 15, 'color': '#000000'}  # Tamaño y color del título del eje X
                    },
                    yaxis_title={
                        'text': "FRECUENCIA",
                        'font': {'size': 15, 'color': '#000000'}  # Tamaño y color del título del eje Y
                    },
                )
                
                # Configurar color y tamaño de etiquetas de los ejes
                histograma_sw.update_xaxes(
                    tickfont={'size': 14, 'color': '#000000'}  # Tamaño y color de las etiquetas del eje X
                )
                histograma_sw.update_yaxes(
                    tickfont={'size': 14, 'color': '#000000'}  # Tamaño y color de las etiquetas del eje Y
                )
                
                
                histograma_vcl = px.histogram(df_pozo_type, x="VCL DEC", nbins=50, color_discrete_sequence=["#ACA800"])
                
                # Configuraciones adicionales
                histograma_vcl.update_layout(
                    title="ARCILLOSIDAD",
                    xaxis_title="VCL (dec)",
                    yaxis_title="FRECUENCIA",
                    plot_bgcolor="white",
                    width=1000,  # Ancho en píxeles
                    height=250  # Alto en píxeles
                )
                
                # Configurar el rango del eje Y (opcional)
                #fig_bruto.update_yaxes(range=[0, 20])
                
                # Configurar título y etiquetas de los ejes
                histograma_vcl.update_layout(
                    title={
                        'text': "ARCILLOSIDAD",
                        'font': {'size': 20, 'color': '#000000'},  # Tamaño y color del encabezado
                        'x': 0.30  # Centra el título
                    },
                    xaxis_title={
                        'text': "VCL (dec)",
                        'font': {'size': 15, 'color': '#000000'}  # Tamaño y color del título del eje X
                    },
                    yaxis_title={
                        'text': "FRECUENCIA",
                        'font': {'size': 15, 'color': '#000000'}  # Tamaño y color del título del eje Y
                    },
                )
                
                # Configurar color y tamaño de etiquetas de los ejes
                histograma_vcl.update_xaxes(
                    tickfont={'size': 14, 'color': '#000000'}  # Tamaño y color de las etiquetas del eje X
                )
                histograma_vcl.update_yaxes(
                    tickfont={'size': 14, 'color': '#000000'}  # Tamaño y color de las etiquetas del eje Y
                )
                
                
                histograma_kmtz = px.histogram(df_pozo_type, x="KMTZ MD", nbins=800, color_discrete_sequence=["#CC99FF"])
                
                # Configuraciones adicionales
                histograma_kmtz.update_layout(
                    title="PERMEABILIDAD DE MATRIZ",
                    xaxis_title="K MATRIZ (mD)",
                    yaxis_title="FRECUENCIA",
                    plot_bgcolor="white",
                    width=1000,  # Ancho en píxeles
                    height=250  # Alto en píxeles
                )
                
                histograma_kmtz.update_xaxes(type="linear", range=[0, 2])
                
                                
                # Configurar título y etiquetas de los ejes
                histograma_kmtz.update_layout(
                    title={
                        'text': "PERMEABILIDAD DE MATRIZ",
                        'font': {'size': 20, 'color': '#000000'},  # Tamaño y color del encabezado
                        'x': 0.30  # Centra el título
                    },
                    xaxis_title={
                        'text': "K MATRIZ (mD)",
                        'font': {'size': 15, 'color': '#000000'}  # Tamaño y color del título del eje X
                    },
                    yaxis_title={
                        'text': "FRECUENCIA",
                        'font': {'size': 15, 'color': '#000000'}  # Tamaño y color del título del eje Y
                    },
                )
                
                # Configurar color y tamaño de etiquetas de los ejes
                histograma_kmtz.update_xaxes(
                    tickfont={'size': 14, 'color': '#000000'}  # Tamaño y color de las etiquetas del eje X
                )
                histograma_kmtz.update_yaxes(
                    tickfont={'size': 14, 'color': '#000000'}  # Tamaño y color de las etiquetas del eje Y
                )
                
                
                histograma_kfrac = px.histogram(df_pozo_type, x="KFRAC MD", nbins=40, color_discrete_sequence=["#FD55F5"])
                
                # Configuraciones adicionales
                histograma_kfrac.update_layout(
                    title="PERMEABILIDAD DE FRACTURAS",
                    xaxis_title="K FRACTURAS (mD)",
                    yaxis_title="FRECUENCIA",
                    plot_bgcolor="white",
                    width=1000,  # Ancho en píxeles
                    height=250  # Alto en píxeles
                )
                
                histograma_kfrac.update_xaxes(type="linear", range=[0, 6000])
                
                                
                # Configurar título y etiquetas de los ejes
                histograma_kfrac.update_layout(
                    title={
                        'text': "PERMEABILIDAD DE FRACTURAS",
                        'font': {'size': 20, 'color': '#000000'},  # Tamaño y color del encabezado
                        'x': 0.30  # Centra el título
                    },
                    xaxis_title={
                        'text': "K FRACTURAS (mD)",
                        'font': {'size': 15, 'color': '#000000'}  # Tamaño y color del título del eje X
                    },
                    yaxis_title={
                        'text': "FRECUENCIA",
                        'font': {'size': 15, 'color': '#000000'}  # Tamaño y color del título del eje Y
                    },
                )
                
                # Configurar color y tamaño de etiquetas de los ejes
                histograma_kfrac.update_xaxes(
                    tickfont={'size': 14, 'color': '#000000'}  # Tamaño y color de las etiquetas del eje X
                )
                histograma_kfrac.update_yaxes(
                    tickfont={'size': 14, 'color': '#000000'}  # Tamaño y color de las etiquetas del eje Y
                )
                
                
                st.write(histograma_sw)
                st.write(histograma_vcl)    
                st.write(histograma_kmtz)
                st.write(histograma_kfrac)
                
                
                

                
                
                
                
                
            
        
        with tabs[1]: 

            # Mostrar en cuatro columnas
            col1, col2, col3 = st.columns(3)
            with col1:
            
                well_info_expander = st.expander("INFORMACIÓN BÁSICA DE POZO")
                c1, c2 = well_info_expander.columns(2)
                
                ms_pozo_tipo = c1.multiselect("SELECCIONA EL TIPO DE POZO", df_evaluacion["TIPO DE POZO"].unique(), default=[])
                df_pozo_tipo = df_evaluacion[df_evaluacion["TIPO DE POZO"].isin(ms_pozo_tipo)]
                
                ms_sumario_tipo = c1.multiselect("SELECCIONA EL TIPO DE SUMARIO", df_pozo_tipo["TIPO SUMARIO"].unique(), default=[])
                df_sumario_tipo = df_pozo_tipo[df_pozo_tipo["TIPO SUMARIO"].isin(ms_sumario_tipo)]
                
                ms_yacimiento = c1.multiselect("SELECCIONA EL YACIMIENTO", df_sumario_tipo["YACIMIENTO"].unique(), default=[])
                df_yacimiento = df_sumario_tipo[df_sumario_tipo["YACIMIENTO"].isin(ms_yacimiento)]
                
                options = []
                options = c2.multiselect("SELECCIONA LA VARIABLE",['ESPESOR BRUTO MV', 'ESPESOR NETO MV',
                                        'RELACION NB','PHIE DEC','SW DEC','VCL DEC', 'KMTZ MD', 'KFRAC MD'], default=[])
                
                ms_pozo = c2.multiselect("SELECCIONA EL POZO", df_yacimiento["POZO IP"].unique(), default=[])
                df_pozo = df_yacimiento[df_yacimiento["POZO IP"].isin(ms_pozo)]
                
                
                
                # Columnas seleccionadas
                selected_columns = [
                    'POZO IP', 'YACIMIENTO','MAESTRA CAMPANA',
                    'ESPESOR BRUTO MV', 'ESPESOR NETO MV', 'RELACION NB',
                    'PHIE DEC','SW DEC', 'VCL DEC', 'KMTZ MD', 'KFRAC MD']
                
                # Crear una copia del DataFrame para calcular el promedio
                df_pozo_avg = df_pozo[selected_columns].copy()
                
                # Calcular los promedios para las columnas numéricas
                numeric_columns = df_pozo_avg.select_dtypes(include=['float64', 'int64']).columns
                averages = df_pozo_avg[numeric_columns].mean()
                
                # Crear una fila de promedio y agregarla al DataFrame
                average_row = pd.Series(averages, name="PROMEDIO")
                df_pozo_avg = pd.concat([df_pozo_avg, average_row.to_frame().T], ignore_index=False)
                
                # Agregar etiquetas de texto a las columnas no numéricas en la fila de promedio
                df_pozo_avg.at["PROMEDIO", 'POZO IP'] = "-"
                df_pozo_avg.at["PROMEDIO", 'YACIMIENTO'] = "-"
                df_pozo_avg.at["PROMEDIO", 'MAESTRA CAMPANA'] = "-"
                
                # Mostrar el DataFrame en Streamlit
                #st.write(df_pozo_avg)    
                    
                # Mostrar el mapa (ya sea filtrado o no)
                col1.markdown("""<h3 style='text-align:center; font-family:Arial;
                              font-size:18px; color:#333333;'>ACEITE ACUMULADO TOTAL (Mbbl)</h3>""",
                            unsafe_allow_html=True
                        )
                
                if options:
                    figNp = plot_density_map(df_pozo, df_pozos_coor, options[0], polygon_lats, polygon_lons, 'turbo', zoom)
                    st.plotly_chart(figNp, use_container_width=True, key="figNp_key")
                else:
                    st.warning("Por favor, selecciona al menos una variable para el mapa de densidad.")
                                        
                #with col1:
            
                    #st.write(df_pozo_avg)
                
            with col2:
                
                fig_bruto = px.bar(df_pozo, x="POZO IP", y='ESPESOR BRUTO MV', color_discrete_sequence=["#C89058"])

                # Configuraciones adicionales
                fig_bruto.update_layout(
                    barmode='group',  # Alterna entre barras apiladas y agrupadas
                    title="ESPESOR BRUTO",
                    xaxis_title="POZO IP",
                    yaxis_title="ESPESOR BRUTO (mv)",
                    plot_bgcolor="white",
                    width=1000,  # Ancho en píxeles
                    height=250  # Alto en píxeles
                )
                
                fig_bruto.update_yaxes(range=[0, 200])
                
                fig_bruto.update_layout(
                    title={
                        'text': "ESPESOR BRUTO",
                        'font': {'size': 20, 'color': '#000000'},  # Tamaño y color del encabezado
                        'x': 0.30  # Centra el título
                    },
                    xaxis_title={
                        'text': "POZO IP",
                        'font': {'size': 15, 'color': '#000000'}  # Tamaño y color del título del eje X
                    },
                    yaxis_title={
                        'text': "BRUTO (mv)",
                        'font': {'size': 15, 'color': '#000000'}  # Tamaño y color del título del eje Y
                    },
                )
                
                # Configurar color y tamaño de etiquetas de los ejes
                fig_bruto.update_xaxes(
                    tickfont={'size': 14, 'color': '#000000'}  # Tamaño y color de las etiquetas del eje X
                )
                fig_bruto.update_yaxes(
                    tickfont={'size': 14, 'color': '#000000'}  # Tamaño y color de las etiquetas del eje Y
                )
                                                
                fig_neto = px.bar(df_pozo, x="POZO IP", y='ESPESOR NETO MV', color_discrete_sequence=["#99CC00"])

                # Configuraciones adicionales
                fig_neto.update_layout(
                    barmode='group',  # Alterna entre barras apiladas y agrupadas
                    title="ESPESOR NETO (mv)",
                    xaxis_title="POZO IP",
                    yaxis_title="ESPESOR NETO MV",
                    plot_bgcolor="white",
                    width=1000,  # Ancho en píxeles
                    height=250  # Alto en píxeles
                )
                
                fig_neto.update_yaxes(range=[0, 200])
                
                fig_neto.update_layout(
                    title={
                        'text': "ESPESOR NETO (mv)",
                        'font': {'size': 20, 'color': '#000000'},  # Tamaño y color del encabezado
                        'x': 0.35  # Centra el título
                    },
                    xaxis_title={
                        'text': "POZO IP",
                        'font': {'size': 15, 'color': '#000000'}  # Tamaño y color del título del eje X
                    },
                    yaxis_title={
                        'text': "NETO (mv)",
                        'font': {'size': 15, 'color': '#000000'}  # Tamaño y color del título del eje Y
                    },
                )
                
                # Configurar color y tamaño de etiquetas de los ejes
                fig_neto.update_xaxes(
                    tickfont={'size': 14, 'color': '#000000'}  # Tamaño y color de las etiquetas del eje X
                )
                fig_neto.update_yaxes(
                    tickfont={'size': 14, 'color': '#000000'}  # Tamaño y color de las etiquetas del eje Y
                )
                                
                fig_netobruto = px.bar(df_pozo, x="POZO IP", y='RELACION NB', color_discrete_sequence=["#D7D200"])

                # Configuraciones adicionales
                fig_netobruto.update_layout(
                    barmode='group',  # Alterna entre barras apiladas y agrupadas
                    title="RELACIÓN DE ESPESOR NETO-BRUTO",
                    xaxis_title="POZO IP",
                    yaxis_title="RELACION NB",
                    plot_bgcolor="white",
                    width=1000,  # Ancho en píxeles
                    height=250  # Alto en píxeles
                )
                
                fig_netobruto.update_yaxes(range=[0, 1])
                
                fig_netobruto.update_layout(
                    title={
                        'text': "RELACIÓN DE ESPESOR NETO-BRUTO",
                        'font': {'size': 20, 'color': '#000000'},  # Tamaño y color del encabezado
                        'x': 0.20  # Centra el título
                    },
                    xaxis_title={
                        'text': "POZO IP",
                        'font': {'size': 15, 'color': '#000000'}  # Tamaño y color del título del eje X
                    },
                    yaxis_title={
                        'text': "RELACIÓN DE N/B(dec)",
                        'font': {'size': 15, 'color': '#000000'}  # Tamaño y color del título del eje Y
                    },
                )
                
                # Configurar color y tamaño de etiquetas de los ejes
                fig_netobruto.update_xaxes(
                    tickfont={'size': 14, 'color': '#000000'}  # Tamaño y color de las etiquetas del eje X
                )
                fig_netobruto.update_yaxes(
                    tickfont={'size': 14, 'color': '#000000'}  # Tamaño y color de las etiquetas del eje Y
                )
                
                fig_porosidad = px.bar(df_pozo, x="POZO IP", y='PHIE DEC', color_discrete_sequence=["#30BD0F"])

                # Configuraciones adicionales
                fig_porosidad.update_layout(
                    barmode='group',  # Alterna entre barras apiladas y agrupadas
                    title="POROSIDAD",
                    xaxis_title="POZO IP",
                    yaxis_title="PHIE DEC",
                    plot_bgcolor="white",
                    width=1000,  # Ancho en píxeles
                    height=250  # Alto en píxeles
                )
                
                fig_porosidad.update_yaxes(range=[0, 0.16])
                
                fig_porosidad.update_layout(
                    title={
                        'text': "POROSIDAD",
                        'font': {'size': 20, 'color': '#000000'},  # Tamaño y color del encabezado
                        'x': 0.35  # Centra el título
                    },
                    xaxis_title={
                        'text': "POZO IP",
                        'font': {'size': 15, 'color': '#000000'}  # Tamaño y color del título del eje X
                    },
                    yaxis_title={
                        'text': "PHIE (dec)",
                        'font': {'size': 15, 'color': '#000000'}  # Tamaño y color del título del eje Y
                    },
                )
                
                # Configurar color y tamaño de etiquetas de los ejes
                fig_porosidad.update_xaxes(
                    tickfont={'size': 14, 'color': '#000000'}  # Tamaño y color de las etiquetas del eje X
                )
                fig_porosidad.update_yaxes(
                    tickfont={'size': 14, 'color': '#000000'}  # Tamaño y color de las etiquetas del eje Y
                )
                
                # Renderizar el gráfico en Streamlit
                st.plotly_chart(fig_bruto)
                st.plotly_chart(fig_neto)
                st.plotly_chart(fig_netobruto)
                st.plotly_chart(fig_porosidad)
                
                
            with col3:
               
                fig_sw = px.bar(df_pozo, x="POZO IP", y='SW DEC', color_discrete_sequence=["#66CCFF"])

                # Configuraciones adicionales
                fig_sw.update_layout(
                    barmode='group',  # Alterna entre barras apiladas y agrupadas
                    title="SATURACIÓN DE AGUA",
                    xaxis_title="POZO IP",
                    yaxis_title="SW DEC",
                    plot_bgcolor="white",
                    width=1000,  # Ancho en píxeles
                    height=250  # Alto en píxeles
                )
                
                fig_sw.update_yaxes(range=[0, 0.70])
                
                fig_sw.update_layout(
                    title={
                        'text': "SATURACIÓN DE AGUA",
                        'font': {'size': 20, 'color': '#000000'},  # Tamaño y color del encabezado
                        'x': 0.30  # Centra el título
                    },
                    xaxis_title={
                        'text': "POZO IP",
                        'font': {'size': 15, 'color': '#000000'}  # Tamaño y color del título del eje X
                    },
                    yaxis_title={
                        'text': "SW (dec)",
                        'font': {'size': 15, 'color': '#000000'}  # Tamaño y color del título del eje Y
                    },
                )
                
                # Configurar color y tamaño de etiquetas de los ejes
                fig_sw.update_xaxes(
                    tickfont={'size': 14, 'color': '#000000'}  # Tamaño y color de las etiquetas del eje X
                )
                fig_sw.update_yaxes(
                    tickfont={'size': 14, 'color': '#000000'}  # Tamaño y color de las etiquetas del eje Y
                )
                
                                
                fig_vcl = px.bar(df_pozo, x="POZO IP", y='VCL DEC', color_discrete_sequence=["#ACA800"])

                # Configuraciones adicionales
                fig_vcl.update_layout(
                    barmode='group',  # Alterna entre barras apiladas y agrupadas
                    title="ARCILLOSIDAD",
                    xaxis_title="POZO IP",
                    yaxis_title="VCL DEC",
                    plot_bgcolor="white",
                    width=1000,  # Ancho en píxeles
                    height=250  # Alto en píxeles
                )
                
                fig_vcl.update_yaxes(range=[0, 0.35])
                
                fig_vcl.update_layout(
                    title={
                        'text': "ARCILLOSIDAD",
                        'font': {'size': 20, 'color': '#000000'},  # Tamaño y color del encabezado
                        'x': 0.35  # Centra el título
                    },
                    xaxis_title={
                        'text': "POZO IP",
                        'font': {'size': 15, 'color': '#000000'}  # Tamaño y color del título del eje X
                    },
                    yaxis_title={
                        'text': "VCL (dec)",
                        'font': {'size': 15, 'color': '#000000'}  # Tamaño y color del título del eje Y
                    },
                )
                
                # Configurar color y tamaño de etiquetas de los ejes
                fig_vcl.update_xaxes(
                    tickfont={'size': 14, 'color': '#000000'}  # Tamaño y color de las etiquetas del eje X
                )
                fig_vcl.update_yaxes(
                    tickfont={'size': 14, 'color': '#000000'}  # Tamaño y color de las etiquetas del eje Y
                )
                               
                fig_perm_mtz = px.bar(df_pozo, x="POZO IP", y='KMTZ MD', color_discrete_sequence=["#CC99FF"])

                # Configuraciones adicionales
                fig_perm_mtz.update_layout(
                    barmode='group',  # Alterna entre barras apiladas y agrupadas
                    title="PERMEABILIDAD DE MATRIZ",
                    xaxis_title="POZO IP",
                    yaxis_title="KMTZ MD",
                    plot_bgcolor="white",
                    width=1000,  # Ancho en píxeles
                    height=250  # Alto en píxeles
                )
                
                fig_perm_mtz.update_yaxes(
                    type="log",  # Escala logarítmica en el eje Y
                    range=[-2, 2],  # Ajusta el rango según los valores logarítmicos de tu dataset
                    tickfont={'size': 14, 'color': '#000000'}
                    )
                
                fig_perm_mtz.update_layout(
                    title={
                        'text': "PERMEABILIDAD DE MATRIZ",
                        'font': {'size': 20, 'color': '#000000'},  # Tamaño y color del encabezado
                        'x': 0.30  # Centra el título
                    },
                    xaxis_title={
                        'text': "POZO IP",
                        'font': {'size': 15, 'color': '#000000'}  # Tamaño y color del título del eje X
                    },
                    yaxis_title={
                        'text': "K MATRIZ (mD)",
                        'font': {'size': 15, 'color': '#000000'}  # Tamaño y color del título del eje Y
                    },
                )
                
                # Configurar color y tamaño de etiquetas de los ejes
                fig_perm_mtz.update_xaxes(
                    tickfont={'size': 14, 'color': '#000000'}  # Tamaño y color de las etiquetas del eje X
                )
                fig_perm_mtz.update_yaxes(
                    tickfont={'size': 14, 'color': '#000000'}  # Tamaño y color de las etiquetas del eje Y
                )
                                
                fig_perm_frac = px.bar(df_pozo, x="POZO IP", y='KFRAC MD', color_discrete_sequence=["#FD55F5"])

                # Configuraciones adicionales
                fig_perm_frac.update_layout(
                    barmode='group',  # Alterna entre barras apiladas y agrupadas
                    title="PERMEABILIDAD DE FRACTURAS",
                    xaxis_title="POZO IP",
                    yaxis_title="KFRAC MD",
                    plot_bgcolor="white",
                    width=1000,  # Ancho en píxeles
                    height=250  # Alto en píxeles
                )
                
                fig_perm_frac.update_yaxes(
                    type="log",  # Escala logarítmica en el eje Y
                    range=[-2, 5],  # Ajusta el rango según los valores logarítmicos de tu dataset
                    tickfont={'size': 14, 'color': '#000000'}
                    )
                
                fig_perm_frac.update_layout(
                    title={
                        'text': "PERMEABILIDAD DE FRACTURAS",
                        'font': {'size': 20, 'color': '#000000'},  # Tamaño y color del encabezado
                        'x': 0.30  # Centra el título
                    },
                    xaxis_title={
                        'text': "POZO IP",
                        'font': {'size': 15, 'color': '#000000'}  # Tamaño y color del título del eje X
                    },
                    yaxis_title={
                        'text': "K FRACTURAS (mD)",
                        'font': {'size': 15, 'color': '#000000'}  # Tamaño y color del título del eje Y
                    },
                )
                
                # Configurar color y tamaño de etiquetas de los ejes
                fig_perm_frac.update_xaxes(
                    tickfont={'size': 14, 'color': '#000000'}  # Tamaño y color de las etiquetas del eje X
                )
                fig_perm_frac.update_yaxes(
                    tickfont={'size': 14, 'color': '#000000'}  # Tamaño y color de las etiquetas del eje Y
                )
                
                # Renderizar el gráfico en Streamlit
                st.plotly_chart(fig_sw)
                st.plotly_chart(fig_vcl)
                st.plotly_chart(fig_perm_mtz)
                st.plotly_chart(fig_perm_frac)
                
        with tabs[2]:
            
            st.write(df_pozo_avg)
            

if __name__ == "__main__":
    main()
    