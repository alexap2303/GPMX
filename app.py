import pandas as pd
import dash
from dash import dcc, html, Input, Output, callback
import plotly.graph_objects as go
import plotly.express as px
import warnings
# Suppress all warnings
warnings.filterwarnings('ignore')
# Crear el dataset original basado en la primera imagen (Seguimiento Renovaciones)
data_renovaciones = pd.read_excel("PRUEBA.xlsx",sheet_name='Sheet1')

# Crear el dataset de ventas Suzuki basado en la segunda imagen
data_ventas = pd.read_excel("PRUEBA.xlsx",sheet_name='Sheet2')

#agregamos MES_NOMBRE al dataset de ventas
data_ventas['MES_NOMBRE'] = data_ventas['MES'].apply(lambda x: ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre'][x-1])


df_renovaciones = data_renovaciones
# Combinar datos históricos y actuales de ventas
df_ventas_completo = data_ventas

# Calcular participación porcentual para cada mes
df_ventas_completo['PARTICIPACION'] = (df_ventas_completo['VENTA_GARANTIA_EXTENDIDA'] / 
                                      df_ventas_completo['VENTA_AUTOS_NUEVOS'] * 100)

# Inicializar la aplicación Dash
app = dash.Dash(__name__, suppress_callback_exceptions=True)

# Layout de la aplicación expandido
app.layout = html.Div([
    # Header con logos y título
    html.Div([
        html.Div([
            html.Span("Garanti", style={'color': 'white', 'font-weight': 'bold', 'font-size': '24px'}),
            html.Span("PLUS", style={'color': '#8BC34A', 'font-weight': 'bold', 'font-size': '24px'})
        ], style={'display': 'inline-block', 'margin-left': '20px'}),
    ], style={'background-color': '#2D3748', 'padding': '15px', 'margin': '0', 'overflow': 'hidden'}),
    
    # Tabs para navegar entre secciones
    dcc.Tabs(id="tabs", value='tab-1', children=[
        dcc.Tab(label='Cobertura de Red', value='tab-1', 
                style={'padding': '10px', 'background-color': '#8BC34A', 'color': 'white'},
                selected_style={'padding': '10px', 'background-color': '#5A9216', 'color': 'white'}),
        dcc.Tab(label='Participación de Mercado', value='tab-2',
                style={'padding': '10px', 'background-color': '#8BC34A', 'color': 'white'},
                selected_style={'padding': '10px', 'background-color': '#5A9216', 'color': 'white'})
    ]),
    
    # Contenido dinámico basado en la tab seleccionada
    html.Div(id='tab-content')
    
], style={'background-color': '#F7FAFC', 'min-height': '100vh', 'margin': '0', 'font-family': 'Arial, sans-serif'})

# Contenido para la primera tab (Seguimiento Renovaciones)
def render_tab1_content():
    return html.Div([
        # Título principal
        html.H1("COBERTURA DE RED", style={
            'text-align': 'center', 
            'color': '#4A5568', 
            'font-weight': 'bold',
            'margin-top': '30px',
            'margin-bottom': '30px',
            'font-size': '32px'
        }),
        
        # Filtros en tarjetas verdes para renovaciones
        html.Div([
            html.Div([
                html.Label("Filtro por Año:", style={'font-weight': 'bold', 'color': 'white', 'margin-bottom': '5px', 'display': 'block'}),
                dcc.Dropdown(
                    id='año-dropdown-renovaciones',
                    options=[{'label': año, 'value': año} for año in sorted(df_renovaciones['AÑO'].unique())],
                    value=[],
                    multi=True,
                    placeholder="Seleccionar año(s)",
                    style={'width': '100%', 'color': 'black'}
                )
            ], style={
                'background-color': '#8BC34A', 
                'color': 'white', 
                'padding': '15px', 
                'margin': '5px',
                'border-radius': '5px',
                'display': 'inline-block',
                'width': '22%',
                'vertical-align': 'top'
            }),
            
            html.Div([
                html.Label("Filtro por Mes:", style={'font-weight': 'bold', 'color': 'white', 'margin-bottom': '5px', 'display': 'block'}),
                dcc.Dropdown(
                    id='mes-dropdown-renovaciones',
                    options=[{'label': mes, 'value': mes} for mes in sorted(df_renovaciones['MES'].unique())],
                    value=[],
                    multi=True,
                    placeholder="Seleccionar mes(es)",
                    style={'width': '100%', 'color': 'black'}
                )
            ], style={
                'background-color': '#8BC34A', 
                'color': 'white', 
                'padding': '15px', 
                'margin': '5px',
                'border-radius': '5px',
                'display': 'inline-block',
                'width': '22%',
                'vertical-align': 'top'
            }),
            
            html.Div([
                html.Label("Filtro por DRM:", style={'font-weight': 'bold', 'color': 'white', 'margin-bottom': '5px', 'display': 'block'}),
                dcc.Dropdown(
                    id='drm-dropdown-renovaciones',
                    options=[{'label': drm, 'value': drm} for drm in sorted(df_renovaciones['DRM'].unique())],
                    value=[],
                    multi=True,
                    placeholder="Seleccionar DRM(s)",
                    style={'width': '100%', 'color': 'black'}
                )
            ], style={
                'background-color': '#8BC34A', 
                'color': 'white', 
                'padding': '15px', 
                'margin': '5px',
                'border-radius': '5px',
                'display': 'inline-block',
                'width': '22%',
                'vertical-align': 'top'
            }),
            
            html.Div([
                html.Label("Filtro por Grupo:", style={'font-weight': 'bold', 'color': 'white', 'margin-bottom': '5px', 'display': 'block'}),
                dcc.Dropdown(
                    id='grupo-dropdown-renovaciones',
                    options=[{'label': grupo, 'value': grupo} for grupo in sorted(df_renovaciones['GRUPO'].unique())],
                    value=[],
                    multi=True,
                    placeholder="Seleccionar grupo(s)",
                    style={'width': '100%', 'color': 'black'}
                )
            ], style={
                'background-color': '#8BC34A', 
                'color': 'white', 
                'padding': '15px', 
                'margin': '5px',
                'border-radius': '5px',
                'display': 'inline-block',
                'width': '22%',
                'vertical-align': 'top'
            })
        ], style={'text-align': 'center', 'margin-bottom': '30px', 'padding': '0 20px'}),
        
        # Gráfico Sankey en contenedor con fondo blanco y sombra
        html.Div([
            dcc.Graph(id='sankey-diagram', style={'height': '600px'})
        ], style={
            'background-color': 'white',
            'margin': '20px',
            'padding': '20px',
            'border-radius': '10px',
            'box-shadow': '0 4px 6px rgba(0, 0, 0, 0.1)'
        })
    ])

# Contenido para la segunda tab (Análisis de Ventas)
def render_tab2_content():
    return html.Div([
        # Título principal
        html.H1("PARTICIPACIÓN DE MERCADO", style={
            'text-align': 'center', 
            'color': '#4A5568', 
            'font-weight': 'bold',
            'margin-top': '30px',
            'margin-bottom': '30px',
            'font-size': '32px'
        }),
        
        # Filtros para la sección de ventas
        html.Div([
            html.Div([
                html.Label("Filtro por Año:", style={'font-weight': 'bold', 'color': 'white', 'margin-bottom': '5px', 'display': 'block'}),
                dcc.Dropdown(
                    id='año-dropdown-ventas',
                    options=[{'label': año, 'value': año} for año in sorted(df_ventas_completo['AÑO'].unique())],
                    value=[],
                    multi=True,
                    placeholder="Seleccionar año(s)",
                    style={'width': '100%', 'color': 'black'}
                )
            ], style={
                'background-color': '#8BC34A', 
                'color': 'white', 
                'padding': '15px', 
                'margin': '5px',
                'border-radius': '5px',
                'display': 'inline-block',
                'width': '22%',
                'vertical-align': 'top'
            }),
            
            html.Div([
                html.Label("Filtro por Mes:", style={'font-weight': 'bold', 'color': 'white', 'margin-bottom': '5px', 'display': 'block'}),
                dcc.Dropdown(
                    id='mes-dropdown-ventas',
                    options=[{'label': f"{row['MES_NOMBRE']} ({row['MES']})", 'value': row['MES']} 
                            for _, row in df_ventas_completo[['MES', 'MES_NOMBRE']].drop_duplicates().iterrows()],
                    value=[],  # Por defecto mostrar los meses con datos históricos
                    multi=True,
                    placeholder="Seleccionar mes(es)",
                    style={'width': '100%', 'color': 'black'}
                )
            ], style={
                'background-color': '#8BC34A', 
                'color': 'white', 
                'padding': '15px', 
                'margin': '5px',
                'border-radius': '5px',
                'display': 'inline-block',
                'width': '22%',
                'vertical-align': 'top'
            }),
            
            html.Div([
                html.Label("Filtro por DRM:", style={'font-weight': 'bold', 'color': 'white', 'margin-bottom': '5px', 'display': 'block'}),
                dcc.Dropdown(
                    id='drm-dropdown-ventas',
                    options=[{'label': drm, 'value': drm} for drm in sorted(df_ventas_completo['DRM'].unique())],
                    value=[],
                    multi=True,
                    placeholder="Seleccionar DRM(s)",
                    style={'width': '100%', 'color': 'black'}
                )
            ], style={
                'background-color': '#8BC34A', 
                'color': 'white', 
                'padding': '15px', 
                'margin': '5px',
                'border-radius': '5px',
                'display': 'inline-block',
                'width': '22%',
                'vertical-align': 'top'
            }),
            
            html.Div([
                html.Label("Filtro por Marca:", style={'font-weight': 'bold', 'color': 'white', 'margin-bottom': '5px', 'display': 'block'}),
                dcc.Dropdown(
                    id='marca-dropdown-ventas',
                    options=[{'label': marca, 'value': marca} for marca in sorted(df_ventas_completo['MARCA'].unique())],
                    value=[],
                    multi=True,
                    placeholder="Seleccionar marca(s)",
                    style={'width': '100%', 'color': 'black'}
                )
            ], style={
                'background-color': '#8BC34A', 
                'color': 'white', 
                'padding': '15px', 
                'margin': '5px',
                'border-radius': '5px',
                'display': 'inline-block',
                'width': '22%',
                'vertical-align': 'top'
            })
        ], style={'text-align': 'center', 'margin-bottom': '30px', 'padding': '0 20px'}),
        
        # Tabla de resumen estilo la imagen 2
        html.Div([
            html.H3("Resumen de Ventas por Mes", style={'text-align': 'center', 'color': '#4A5568', 'margin-bottom': '20px'}),
            html.Div(id='tabla-resumen')
        ], style={
            'background-color': 'white',
            'margin': '20px',
            'padding': '20px',
            'border-radius': '10px',
            'box-shadow': '0 4px 6px rgba(0, 0, 0, 0.1)'
        }),
        
        # Gráfico de barras estilo la imagen 2
        html.Div([
            dcc.Graph(id='grafico-barras-ventas', style={'height': '500px'})
        ], style={
            'background-color': 'white',
            'margin': '20px',
            'padding': '20px',
            'border-radius': '10px',
            'box-shadow': '0 4px 6px rgba(0, 0, 0, 0.1)'
        })
    ])

# Callback para manejar el contenido de las tabs
@callback(Output('tab-content', 'children'),
          Input('tabs', 'value'),
     suppress_callback_exceptions=True)
def render_content(tab):
    if tab == 'tab-1':
        return render_tab1_content()
    elif tab == 'tab-2':
        return render_tab2_content()

# Definir paletas de colores (mantener las originales)
color_for_nodes =['gray','green', 'firebrick','steelblue', 'goldenrod','darkviolet','black','orangered','firebrick']
color_for_links =['LightGreen','palevioletred', 'LightSkyBlue', 'gold', 
                   'blueviolet','gray','orange','palevioletred']

# Callback para actualizar el gráfico Sankey (mantener funcionalidad original)
@callback(
    Output('sankey-diagram', 'figure'),
    [Input('año-dropdown-renovaciones', 'value'),
     Input('mes-dropdown-renovaciones', 'value'),
     Input('drm-dropdown-renovaciones', 'value'),
     Input('grupo-dropdown-renovaciones', 'value')],
     suppress_callback_exceptions=True
)
def update_sankey(selected_años, selected_meses, selected_drms, selected_grupos):
    # Usar df_renovaciones en lugar de df para mayor claridad
    filtered_df = df_renovaciones.copy()
    
    # Aplicar filtros solo si hay valores seleccionados
    if selected_años is not None and len(selected_años) > 0:
        if not isinstance(selected_años, list):
            selected_años = [selected_años]
        filtered_df = filtered_df[filtered_df['AÑO'].isin(selected_años)]
    
    if selected_meses is not None and len(selected_meses) > 0:
        if not isinstance(selected_meses, list):
            selected_meses = [selected_meses]
        filtered_df = filtered_df[filtered_df['MES'].isin(selected_meses)]
    
    if selected_drms is not None and len(selected_drms) > 0:
        if not isinstance(selected_drms, list):
            selected_drms = [selected_drms]
        filtered_df = filtered_df[filtered_df['DRM'].isin(selected_drms)]
    
    if selected_grupos is not None and len(selected_grupos) > 0:
        if not isinstance(selected_grupos, list):
            selected_grupos = [selected_grupos]
        filtered_df = filtered_df[filtered_df['GRUPO'].isin(selected_grupos)]
    
    # Agrupar y sumar cantidades
    sankey_data = filtered_df.groupby(['ORIGEN', 'COBERTURA'])['CANTIDAD'].sum().reset_index()
    
    # Calcular totales por destino para las etiquetas
    destino_totals = sankey_data.groupby('COBERTURA')['CANTIDAD'].sum()
    origen_totals = sankey_data.groupby('ORIGEN')['CANTIDAD'].sum()
    
    # Crear listas de nodos únicos con etiquetas que incluyen valores
    origenes = sankey_data['ORIGEN'].unique()
    destinos = sankey_data['COBERTURA'].unique()
    
    # Crear etiquetas con valores entre paréntesis
    node_labels = []
    all_nodes = []
    
    # Agregar nodos de origen con sus totales
    for origen in origenes:
        total = origen_totals[origen]
        label = f"{origen} \n ({total})  <br> <b>{(total / origen_totals.sum()) * 100:.1f}%</b>"
        node_labels.append(label)
        all_nodes.append(origen)
    
    # Agregar nodos de destino con sus totales
    for destino in destinos:
        total = destino_totals[destino]
        porcentaje = (total / destino_totals.sum()) * 100 if destino_totals.sum() > 0 else 0
        label = f"{destino} \n ({total}) <br> <b>{porcentaje:.1f}%</b>"
        node_labels.append(label)
        all_nodes.append(destino)
    
    node_dict = {node: i for i, node in enumerate(all_nodes)}
    
    # Crear las conexiones para el diagrama Sankey
    source = [node_dict[origen] for origen in sankey_data['ORIGEN']]
    target = [node_dict[cobertura] for cobertura in sankey_data['COBERTURA']]
    value = sankey_data['CANTIDAD'].tolist()
    
    # Crear el gráfico Sankey
    fig = go.Figure(data=[go.Sankey(
        node = dict(
            pad = 15,
            thickness = 20,
            line = dict(color = "black", width = 0.5),
            label = node_labels,
            color = color_for_nodes[:len(all_nodes)]
        ),
        link = dict(
            source = source,
            target = target,
            value = value,
            color = color_for_links[:len(source)]
        )
    )])
    
    fig.update_layout(
        title_text="",
        font_size=12,
        height=600,
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    return fig

# Callback para actualizar la tabla de resumen de ventas
@callback(
    Output('tabla-resumen', 'children'),
    [Input('año-dropdown-ventas', 'value'),
     Input('mes-dropdown-ventas', 'value'),
     Input('drm-dropdown-ventas', 'value'),
     Input('marca-dropdown-ventas', 'value')],
     suppress_callback_exceptions=True
)
def update_tabla_resumen(selected_años, selected_meses, selected_drms, selected_marcas):
    # Filtrar datos de ventas
    filtered_df = df_ventas_completo.copy()
    
    if selected_años and len(selected_años) > 0:
        filtered_df = filtered_df[filtered_df['AÑO'].isin(selected_años)]
    if selected_meses and len(selected_meses) > 0:
        filtered_df = filtered_df[filtered_df['MES'].isin(selected_meses)]
    if selected_drms and len(selected_drms) > 0:
        filtered_df = filtered_df[filtered_df['DRM'].isin(selected_drms)]
    if selected_marcas and len(selected_marcas) > 0:
        filtered_df = filtered_df[filtered_df['MARCA'].isin(selected_marcas)]
    
    # Crear tabla HTML estilo la imagen 2
    table_header = [
        html.Thead([
            html.Tr([
                html.Th("Mes", style={'background-color': '#DC143C', 'color': 'white', 'padding': '10px', 'text-align': 'center'}),
                html.Th("VENTA AN", style={'background-color': '#DC143C', 'color': 'white', 'padding': '10px', 'text-align': 'center'}),
                html.Th("GE Nuevos", style={'background-color': '#DC143C', 'color': 'white', 'padding': '10px', 'text-align': 'center'}),
                html.Th("Participación", style={'background-color': '#DC143C', 'color': 'white', 'padding': '10px', 'text-align': 'center'})
            ])
        ])
    ]
    
    table_body = []
    total_vtas = 0
    total_ge = 0
    
    for _, row in filtered_df.iterrows():
        total_vtas += row['VENTA_AUTOS_NUEVOS']
        total_ge += row['VENTA_GARANTIA_EXTENDIDA']
        table_body.append(
            html.Tr([
                html.Td(row['MES_NOMBRE'], style={'padding': '8px', 'text-align': 'center'}),
                html.Td(f"{row['VENTA_AUTOS_NUEVOS']:,}", style={'padding': '8px', 'text-align': 'center'}),
                html.Td(row['VENTA_GARANTIA_EXTENDIDA'], style={'padding': '8px', 'text-align': 'center'}),
                html.Td(f"{row['PARTICIPACION']:.2f}%", style={'padding': '8px', 'text-align': 'center'})
            ])
        )
    
    # Agregar fila de total
    participacion_total = (total_ge / total_vtas * 100) if total_vtas > 0 else 0
    table_body.append(
        html.Tr([
            html.Td("Total", style={'padding': '8px', 'text-align': 'center', 'font-weight': 'bold'}),
            html.Td(f"{total_vtas:,}", style={'padding': '8px', 'text-align': 'center', 'font-weight': 'bold'}),
            html.Td(total_ge, style={'padding': '8px', 'text-align': 'center', 'font-weight': 'bold'}),
            html.Td(f"{participacion_total:.2f}%", style={'padding': '8px', 'text-align': 'center', 'font-weight': 'bold'})
        ], style={'background-color': '#F5F5F5'})
    )
    
    return html.Table(table_header + [html.Tbody(table_body)], 
                     style={'width': '100%', 'border-collapse': 'collapse', 'border': '1px solid #ddd'})

# Callback para actualizar el gráfico de barras de ventas
@callback(
    Output('grafico-barras-ventas', 'figure'),
    [Input('año-dropdown-ventas', 'value'),
     Input('mes-dropdown-ventas', 'value'),
     Input('drm-dropdown-ventas', 'value'),
     Input('marca-dropdown-ventas', 'value')],
     suppress_callback_exceptions=True
)
def update_grafico_barras(selected_años, selected_meses, selected_drms, selected_marcas):
    # Filtrar datos de ventas
    filtered_df = df_ventas_completo.copy()
    
    if selected_años and len(selected_años) > 0:
        filtered_df = filtered_df[filtered_df['AÑO'].isin(selected_años)]
    if selected_meses and len(selected_meses) > 0:
        filtered_df = filtered_df[filtered_df['MES'].isin(selected_meses)]
    if selected_drms and len(selected_drms) > 0:
        filtered_df = filtered_df[filtered_df['DRM'].isin(selected_drms)]
    if selected_marcas and len(selected_marcas) > 0:
        filtered_df = filtered_df[filtered_df['MARCA'].isin(selected_marcas)]
    
    # Crear gráfico de barras estilo la imagen 2
    fig = go.Figure()
    
    # Barra de ventas de autos nuevos (color gris claro como en la imagen)
    fig.add_trace(go.Bar(
        name='Venta Autos Nuevos',
        x=filtered_df['MES_NOMBRE'],
        y=filtered_df['VENTA_AUTOS_NUEVOS'],
        marker_color='#2E86AB',  # Gris claro similar al de la imagen
        yaxis='y'
    ))


    
   
    # Primera traza - Ventas GE (eje izquierdo)
    fig.add_trace(go.Bar(
        name='Venta GE Nuevos',
        x=filtered_df['MES_NOMBRE'],
        y=filtered_df['VENTA_GARANTIA_EXTENDIDA'],
        marker_color='#8BC34A',
        yaxis='y'  # Eje principal (izquierdo)
    ))
    #agregar etiquetas encima de las barras
    fig.update_traces(texttemplate='%{y:,.0f}', textposition='outside')
  

    # Segunda traza - Participación (eje derecho)
    fig.add_trace(go.Scatter(  # Cambié a Scatter para mejor visualización
        mode='lines+markers',
        name='Participación %',
        x=filtered_df['MES_NOMBRE'],
        y=filtered_df['PARTICIPACION'],
        line=dict(color='#E53E3E', width=3),
        marker=dict(color='#E53E3E', size=8),
        yaxis='y2'  # Eje secundario (derecho)
    ))
    #AGREGAR ETIQUETAS A LA LINEA DE PARTICIPACION
    fig.add_trace(go.Scatter(
        mode='text',
        x=filtered_df['MES_NOMBRE'],
        y=filtered_df['PARTICIPACION'],
        text=[f"{val:.1f}%" for val in filtered_df['PARTICIPACION']],
        #font color red
        textfont=dict(color="#A42323", size=12),
        textposition='top center',
        showlegend=False,
        yaxis='y2'
    ))


    # Agregar una segunda serie si fuera necesario (GE Nuevos como línea separada)
    # Por ahora solo mostramos las ventas de garantía extendida
    
    fig.update_layout(
    title="Ventas de Garantía Extendida y Participación por Mes",
    xaxis_title="Mes",
    
    # Configuración del eje Y principal (izquierdo)
    yaxis=dict(
        title="Cantidad de Ventas GE",
        side='left',
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray'
    ),
    
    # Configuración del eje Y secundario (derecho)
    yaxis2=dict(
        title="Participación %",
        overlaying='y',  # Superponer sobre el eje principal
        side='right',    # Posicionar a la derecha
        showgrid=False,  # Evitar conflicto de grillas
        ticksuffix='%' ,  # Agregar símbolo de porcentaje
        #set range from 0 to 100
        range=[0, 2]
    ),
    
    plot_bgcolor='white',
    paper_bgcolor='white',
    showlegend=True,
    height=500,
    margin=dict(l=50, r=50, t=80, b=50),
    font=dict(size=12)
)
    
    # Estilo de ejes similar a la imagen
    fig.update_xaxes(showgrid=False, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=False, gridwidth=1, gridcolor='lightgray')
    
    return fig

# Ejecutar la aplicación
if __name__ == '__main__':
    app.run(debug=True)
