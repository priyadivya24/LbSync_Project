from typing import List
from uuid import uuid4
import time
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from dash import MATCH, Input, Output, State, dcc, html, no_update
from dash_extensions.enrich import (
    DashProxy,
    Serverside,
    ServersideOutputTransform,
    Trigger,
    TriggerTransform,
)

from datetime import datetime
from plotly_resampler import FigureResampler
from plotly_resampler.aggregation import MinMaxLTTB

from data import get_doocs_properties, load_parquet_data
from pathlib import Path
import base64
import os
from sklearn.ensemble import IsolationForest


def ffill(arr):
    mask = np.isnan(arr)
    if not np.any(mask):
        return arr
    idx = np.where(~mask, np.arange(mask.shape[0]), 0)
    np.maximum.accumulate(idx, axis=0, out=idx)
    ffilled = arr[idx]  # Use the index array to forward fill the original array

    i = 0
    while i < len(ffilled) and np.isnan(ffilled[i]):
        i += 1
    
    if i > 0 and i < len(ffilled):
        filler = ffilled[i]
        ffilled[:i] = filler
        
    return ffilled


def process_data(np_array, outlier_removal=False, normalize=False):
    if outlier_removal:
        median = np.median(np_array)
        stdev = np.std(np_array)
        lower_threshold = median - 4 * stdev
        upper_threshold = median + 4 * stdev
        clean_idc = np.logical_and(upper_threshold >= np_array, lower_threshold <= np_array)
        np_array[~clean_idc] = np.nan
        np_array = ffill(np_array)

    if normalize:
        v_min = np.nanmin(np_array)
        v_max = np.nanmax(np_array)
        if v_max > v_min:
            np_array = (np_array - v_min) / (v_max - v_min)

    return np_array


# logo
try:
    with open("lbsync.png", "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
except FileNotFoundError:
    encoded_image = "" # Fallback if logo not found

# Define properties
# Updated to current project path
PROJECT_ROOT = Path(r"C:\Users\Priyadarshini Maha\OneDrive\Documents\Projects_git\Grafana_XFEL")
DATABASE_PATH = PROJECT_ROOT / "XFEL.SYNC"

doocs_properties = get_doocs_properties(DATABASE_PATH)
doocs_properties_inversed = {}
for k, v in doocs_properties.items():
    doocs_properties_inversed[v] = k

# Data storage
online_data = {}

# Create a Dash app with server-side output transformation
app = DashProxy(__name__, transforms=[ServersideOutputTransform(), TriggerTransform()])

header = html.Div([
    html.Img(src=f"data:image/png;base64,{encoded_image}",
             style={'height': '100px', 'width': 'auto', 'margin-right': '150px'}) if encoded_image else html.Div(),
    html.H1("LbSync Dashboard", style={'textAlign': 'center', 'line-height': '50px', 'flex-grow': '1'})
], style={'display': 'flex', 'align-items': 'center', 'padding': '20px', 'background': '#f8f9fa', 'border-bottom': '1px solid #dee2e6'})

body = html.Div([
    dcc.Tabs(
        children=[
            dcc.Tab(label="History Plotter", children=[
                html.Div([
                    html.Label('Laser Properties', style={'fontWeight': 'bold'}),
                    dcc.Dropdown([i for i in doocs_properties.values() if "LASER" in i], multi=True,
                                 id="dcc_dropdown-laser"),
                    html.Br(),
                    html.Label('Link Properties', style={'fontWeight': 'bold'}),
                    dcc.Dropdown([i for i in doocs_properties.values() if "LINK" in i], multi=True,
                                 id="dcc_dropdown-link"),
                    html.Br(),
                    html.Label('Climate Properties', style={'fontWeight': 'bold'}),
                    dcc.Dropdown([i for i in doocs_properties.values() if "CLIMATE" in i], multi=True,
                                 id="dcc_dropdown-climate"),
                    html.Br(),
                    dcc.Checklist(
                        id="dcc_checklist-config",
                        options=[
                            {'label': " Separate Plots", 'value': "separate plots"},
                            {'label': " Remove Outliers (4 sigma)", 'value': "remove outliers (1% - 99%)"},
                            {'label': " Normalize", 'value': "normalize"}
                        ],
                        value=[],
                        style={'margin': '10px 0'}
                    ),
                    html.Br(),
                    html.Button("Load Data and Plot", id="dcc_button-loadplot", 
                                style={'padding': '10px 20px', 'background': '#007bff', 'color': 'white', 'border': 'none', 'borderRadius': '4px', 'cursor': 'pointer'}),
                ], style={'padding': '20px'}),
                html.Div(id="container", children=[], style={'padding': '20px'}),
            ]),
            dcc.Tab(label="Data Analysis", children=[
                html.Div([
                    html.Label('Laser Properties', style={'fontWeight': 'bold'}),
                    dcc.Dropdown([i for i in doocs_properties.values() if "LASER" in i], multi=True,
                                 id="dcc_dropdown-laser2"),
                    html.Br(),
                    html.Label('Link Properties', style={'fontWeight': 'bold'}),
                    dcc.Dropdown([i for i in doocs_properties.values() if "LINK" in i], multi=True,
                                 id="dcc_dropdown-link2"),
                    html.Br(),
                    html.Label('Climate Properties', style={'fontWeight': 'bold'}),
                    dcc.Dropdown([i for i in doocs_properties.values() if "CLIMATE" in i], multi=True,
                                 id="dcc_dropdown-climate2"),
                    html.Br(),
                    html.Button("Correlations", id="dcc_button-correlation",
                                style={'padding': '10px 20px', 'background': '#28a745', 'color': 'white', 'border': 'none', 'borderRadius': '4px', 'cursor': 'pointer'}),
                ], style={'padding': '20px'}),
                html.Div(id="container2", children=[], style={'padding': '20px'}),
            ]),
            dcc.Tab(label="Anomaly Detection", children=[
                html.Div([
                    html.Label('Select Property', style={'fontWeight': 'bold'}),
                    dcc.Dropdown([i for i in doocs_properties.values()], id="dcc_dropdown-anomaly"),
                    html.Br(),
                    html.Label('Contamination level (Sensitivity)', style={'fontWeight': 'bold'}),
                    dcc.Slider(
                        id='anomaly-contamination',
                        min=0.001,
                        max=0.05,
                        step=0.001,
                        value=0.01,
                        marks={0.001: '0.1%', 0.01: '1%', 0.05: '5%'},
                    ),
                    html.Br(),
                    html.Button("Detect Anomalies", id="dcc_button-anomaly",
                                style={'padding': '10px 20px', 'background': '#dc3545', 'color': 'white', 'border': 'none', 'borderRadius': '4px', 'cursor': 'pointer'}),
                ], style={'padding': '20px'}),
                html.Div(id="container-anomaly", children=[], style={'padding': '20px'}),
            ])
        ], style={'padding': '10px'}
    )
])


## callbacks
@app.callback(
    Output("container2", "children"),
    Input("dcc_button-correlation", "n_clicks"),
    State("dcc_dropdown-laser2", "value"),
    State("dcc_dropdown-link2", "value"),
    State("dcc_dropdown-climate2", "value"),
    prevent_initial_call=True,
)
def correlation_analysis(n_clicks: int, laser_values: List[str], link_values: List[str], climate_values: List[str]):
    if not (laser_values or link_values or climate_values):
        return [html.Div("Please select at least one property.")]
        
    if laser_values is None: laser_values = []
    if link_values is None: link_values = []
    if climate_values is None: climate_values = []
    properties = laser_values + link_values + climate_values

    # Loading data for Oct/Nov 2023 range as seen in directory
    pq_data_set = load_parquet_data([Path(doocs_properties_inversed[p]) for p in properties],
                                    datetime(2023, 10, 1), datetime(2023, 11, 30))
    
    if not pq_data_set:
        return [html.Div("No data found for the selected properties and date range.")]

    merged_df = None
    
    for name_str, pq_table in pq_data_set.items():
        # Check available columns
        cols = pq_table.column_names
        target_col = "data" if "data" in cols else (cols[1] if len(cols) > 1 else cols[0])
        merge_col = "bunchID" if "bunchID" in cols else ("timestamp" if "timestamp" in cols else None)
        
        if not merge_col:
            continue
            
        tmp_df = pq_table.to_pandas()[[merge_col, target_col]].rename(columns={target_col: doocs_properties[str(name_str)]})
        
        if merged_df is None:
            merged_df = tmp_df
        else:
            merged_df = pd.merge(merged_df, tmp_df, on=merge_col, how="outer")
            
    if merged_df is None or merged_df.empty:
        return [html.Div("Could not merge data for correlations.")]

    merged_df.sort_values(by=merged_df.columns[0], inplace=True)
    merged_df.ffill(inplace=True)
    merged_df.dropna(inplace=True)
    
    # Drop IDs/Timestamps for correlation
    numeric_df = merged_df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 1:
        return [html.Div("No numeric data available for correlation.")]
        
    # Drop bunchID or timestamp if it made it into numeric_df
    numeric_df = numeric_df.drop(columns=['bunchID', 'timestamp'], errors='ignore')
    
    correlation_matrix = numeric_df.corr()

    fig_heatmap = go.Figure(
        data=go.Heatmap(
            z=correlation_matrix.values, 
            x=correlation_matrix.columns, 
            y=correlation_matrix.columns,
            colorscale='Viridis', 
            colorbar=dict(title='Correlation'), 
        )
    )

    annotations = []
    for i, row in enumerate(correlation_matrix.values):
        for j, value in enumerate(row):
            font_color = 'black' if abs(value) > 0.5 else 'white'
            annotations.append(
                dict(x=correlation_matrix.columns[j], y=correlation_matrix.columns[i], text=str(round(value, 2)),
                     xref='x', yref='y', font=dict(color=font_color), showarrow=False)
            )

    fig_heatmap.update_layout(title='Correlation Matrix', annotations=annotations, height=600)
    return [dcc.Graph(figure=fig_heatmap)]


@app.callback(
    Output("container-anomaly", "children"),
    Input("dcc_button-anomaly", "n_clicks"),
    State("dcc_dropdown-anomaly", "value"),
    State("anomaly-contamination", "value"),
    prevent_initial_call=True,
)
def anomaly_detection(n_clicks: int, property_name: str, contamination: float):
    if not property_name:
        return [html.Div("Please select a property.")]

    # Load data
    start_dt, end_dt = datetime(2023, 10, 1), datetime(2023, 11, 30)
    pq_data_set = load_parquet_data([Path(doocs_properties_inversed[property_name])], start_dt, end_dt)
    
    if not pq_data_set:
        return [html.Div("No data found for the selected property.")]

    # Extract table (there should be only one)
    path_key = list(pq_data_set.keys())[0]
    table = pq_data_set[path_key]
    df = table.to_pandas()
    
    if df.empty:
        return [html.Div("Dataframe is empty.")]

    # Prepare data for Isolation Forest
    target_col = "data" if "data" in df.columns else df.columns[1]
    df = df[['timestamp', target_col]].dropna()
    
    # ⚡ OPTIMIZATION 1: Sample data if it is too massive (e.g. > 50k points)
    # Fitting on a representative sample is much faster and usually just as accurate
    MAX_FIT_POINTS = 50000
    if len(df) > MAX_FIT_POINTS:
        fit_df = df.sample(n=MAX_FIT_POINTS, random_state=42).sort_values('timestamp')
    else:
        fit_df = df

    if len(df) < 10:
        return [html.Div("Insufficient data for anomaly detection.")]

    # Fit Isolation Forest
    X_fit = fit_df[target_col].values.reshape(-1, 1)
    X_all = df[target_col].values.reshape(-1, 1)
    
    # ⚡ OPTIMIZATION 2: Parallel execution (n_jobs=-1)
    model = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
    model.fit(X_fit)
    df['anomaly'] = model.predict(X_all)
    
    # -1 is outlier, 1 is inlier
    anomalies = df[df['anomaly'] == -1]
    
    # Create Visualization
    fig = go.Figure()
    
    # ⚡ OPTIMIZATION 3: Efficient Plotting
    # Display max 10k points for the main line to keep UI responsive
    DISPLAY_POINTS = 10000
    plot_df = df.iloc[::max(1, len(df)//DISPLAY_POINTS)]
    
    fig.add_trace(go.Scattergl( # Use Scattergl for better WebGL performance
        x=pd.to_datetime(plot_df['timestamp'], unit='s'),
        y=plot_df[target_col],
        mode='lines',
        name='Data',
        line=dict(color='rgba(0, 123, 255, 0.7)', width=1)
    ))
    
    # Limit number of anomaly markers to 1k
    MAX_MARKERS = 1000
    disp_anomalies = anomalies.iloc[::max(1, len(anomalies)//MAX_MARKERS)]
    
    fig.add_trace(go.Scattergl(
        x=pd.to_datetime(disp_anomalies['timestamp'], unit='s'),
        y=disp_anomalies[target_col],
        mode='markers',
        name='Anomalies',
        marker=dict(color='red', size=5, symbol='x')
    ))
    
    fig.update_layout(
        title=f"<b>Anomaly Detection: {property_name}</b> (Contamination: {contamination*100}%)",
        xaxis_title="Time",
        yaxis_title="Value",
        template='plotly_white',
        height=600
    )

    return [
        html.Div(f"Detected {len(anomalies)} potential anomalies out of {len(df)} points."),
        dcc.Graph(figure=fig)
    ]


@app.callback(
    Output("container", "children", allow_duplicate=True),
    Input("dcc_button-loadplot", "n_clicks"),
    State("dcc_dropdown-laser", "value"),
    State("dcc_dropdown-link", "value"),
    State("dcc_dropdown-climate", "value"),
    State("dcc_checklist-config", "value"),
    State("container", "children"),
    prevent_initial_call=True,
)
def add_graph_div(n_clicks: int, laser_values: List[str], link_values: List[str], climate_values: List[str],
                  plot_config: List, div_children: List):
    if not (laser_values or link_values or climate_values):
        return div_children

    if not ('separate plots' in plot_config):
        uid = str(uuid4())
        new_child = html.Div(
            children=[
                dcc.Graph(id={"type": "dynamic-graph", "index": uid, "name": "all"}, figure=go.Figure()),
                dcc.Loading(dcc.Store(id={"type": "store", "index": uid, "name": "all"})),
                dcc.Interval(id={"type": "interval", "index": uid, "name": "all"}, max_intervals=1, interval=1),
            ]
        )
        div_children = [new_child]
    else:
        div_children = []
        if laser_values is None: laser_values = []
        if link_values is None: link_values = []
        if climate_values is None: climate_values = []
        properties = laser_values + link_values + climate_values
        for p in properties:
            uid = str(uuid4())
            new_child = html.Div(
                children=[
                    dcc.Graph(id={"type": "dynamic-graph", "index": uid, "name": str(p)}, figure=go.Figure()),
                    dcc.Loading(dcc.Store(id={"type": "store", "index": uid, "name": str(p)})),
                    dcc.Interval(id={"type": "interval", "index": uid, "name": str(p)}, max_intervals=1, interval=1),
                ], style={'marginBottom': '20px'}
            )
            div_children.append(new_child)

    return div_children


@app.callback(
    State("dcc_dropdown-laser", "value"),
    State("dcc_dropdown-link", "value"),
    State("dcc_dropdown-climate", "value"),
    State("dcc_checklist-config", "value"),
    State({"type": "dynamic-graph", "index": MATCH, "name": MATCH}, "id"),
    Output({"type": "dynamic-graph", "index": MATCH, "name": MATCH}, "figure"),
    Output({"type": "store", "index": MATCH, "name": MATCH}, "data"),
    Trigger({"type": "interval", "index": MATCH, "name": MATCH}, "n_intervals"),
    prevent_initial_call=True,
)
def construct_display_graph(laser_values, link_values, climate_values, plot_config: List, graph_id) -> FigureResampler:
    fig = FigureResampler(
        go.Figure(),
        default_n_shown_samples=2_000,
        default_downsampler=MinMaxLTTB(parallel=True),
    )
    
    if laser_values is None: laser_values = []
    if link_values is None: link_values = []
    if climate_values is None: climate_values = []
    properties = laser_values + link_values + climate_values
    remove_outliers = 'remove outliers (1% - 99%)' in plot_config
    normalize_data = 'normalize' in plot_config

    # Use Oct/Nov 2023 range
    start_dt, end_dt = datetime(2023, 10, 1), datetime(2023, 11, 30)

    if graph_id["name"] == 'all':
        pq_data_set = load_parquet_data([Path(doocs_properties_inversed[p]) for p in properties], start_dt, end_dt)
        for pq_path, table in pq_data_set.items():
            x_values = table["timestamp"].to_numpy()
            x_datetime = pd.to_datetime(x_values, unit='s')
            y_values = table["data"].to_numpy().astype(float)
            hf_y = process_data(y_values, remove_outliers, normalize_data)
            fig.add_trace(dict(name=doocs_properties[str(pq_path)]), hf_x=x_datetime, hf_y=hf_y)
        
        fig.update_layout(title="<b>Combined Histories</b>", title_x=0.5)
    else:
        p_name = graph_id["name"]
        pq_data_set = load_parquet_data([Path(doocs_properties_inversed[p_name])], start_dt, end_dt)
        for pq_path, table in pq_data_set.items():
            x_values = table["timestamp"].to_numpy()
            x_datetime = pd.to_datetime(x_values, unit='s')
            y_values = table["data"].to_numpy().astype(float)
            hf_y = process_data(y_values, remove_outliers, normalize_data)
            fig.add_trace(dict(name=doocs_properties[str(pq_path)]), hf_x=x_datetime, hf_y=hf_y)
            fig.update_layout(title=f"<b>History: {doocs_properties[str(pq_path)]}</b>", title_x=0.5)

    fig.update_layout(yaxis_tickformat='.5f', template='plotly_white')
    return fig, Serverside(fig)


@app.callback(
    Output({"type": "dynamic-graph", "index": MATCH, "name": MATCH}, "figure", allow_duplicate=True),
    Input({"type": "dynamic-graph", "index": MATCH, "name": MATCH}, "relayoutData"),
    State({"type": "store", "index": MATCH, "name": MATCH}, "data"),
    prevent_initial_call=True,
    memoize=True,
)
def update_fig(relayoutdata: dict, fig: FigureResampler):
    if fig is not None:
        return fig.construct_update_data_patch(relayoutdata)
    return no_update


if __name__ == "__main__":
    print(f"Starting dashboard with {len(doocs_properties)} properties found.")
    app.layout = html.Div([header, body])
    app.run(debug=True, port=9025)
