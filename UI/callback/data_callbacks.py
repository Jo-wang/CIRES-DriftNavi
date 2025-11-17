import dash
from UI.app import app
from dash.dependencies import Input, Output, State, ALL  # ALL still needed for legacy validation callbacks
from dash.exceptions import PreventUpdate
import plotly.express as px
import base64
from agent import DatasetAgent
import datetime
from dash import callback_context
import io
from dash import dcc, html, dash_table
import pandas as pd
from UI.functions import *
import dash_bootstrap_components as dbc
from flask_login import current_user
from utils.dataset_eval import DatasetEval
from dash.dash_table.Format import Format, Scheme
from UI.functions import query_llm
import time
import random
import pandas as pd
import json
import hashlib
from utils.data_processor import preprocess_datasets, get_column_type_summary, identify_categorical_encoding_columns, create_encoding_notification_text
from UI.state.column_type_manager import ColumnTypeManager
from utils.snapshot_manager import (
    list_snapshots,
    create_original_if_absent,
    create_snapshot,
    restore_snapshot,
)

# ===================== Snapshot UI callbacks =====================
@app.callback(
    [Output("snapshot-selector", "options"),
     Output("snapshot-selector", "value")],
    [Input('dataset-upload-trigger', 'data'),
     Input('table-update-trigger', 'data')],
    [State("snapshot-selector", "value")],
    prevent_initial_call=True
)
def refresh_snapshot_options(_upload_trigger, _table_trigger, current_value):
    try:
        create_original_if_absent()
        items = list_snapshots()
        options = []
        for s in items:
            label_base = 'Original' if s.get('ver') == 'original' else s.get('ver')
            time_str = str(s.get('time', '')).replace('T', ' ')
            desc = s.get('desc')
            label = f"{label_base} ({time_str})" + (f" - {desc}" if desc else "")
            options.append({"label": label, "value": s.get('ver')})
        # keep current selection if possible to avoid unintended restore
        if current_value and any(o['value'] == current_value for o in options):
            return options, dash.no_update
        # otherwise default to original if available, but avoid forcing selection change when not needed
        default_val = 'original' if any(o['value'] == 'original' for o in options) else (options[0]['value'] if options else None)
        return options, default_val
    except Exception:
        # Fail silently to avoid impacting other features
        raise PreventUpdate


@app.callback(
    [Output("snapshot-selector", "options", allow_duplicate=True),
     Output("snapshot-selector", "value", allow_duplicate=True),
     Output('table-update-trigger', 'data', allow_duplicate=True),
     Output("context-version-modal", "is_open", allow_duplicate=True)],
    Input("snapshot-save-btn", "n_clicks"),
    [State('table-primary-overview', 'data'),
     State('table-primary-overview', 'columns'),
     State('table-secondary-overview', 'data'),
     State('table-secondary-overview', 'columns')],
    prevent_initial_call=True
)
def on_save_snapshot(n_clicks, primary_data, primary_cols, secondary_data, secondary_cols):
    if not n_clicks:
        raise PreventUpdate
    
    # Check if there are existing context items that would be cleared
    has_context = False
    try:
        from UI.functions.global_vars import global_vars
        # Check explain context data
        if hasattr(global_vars, 'explain_context_data') and global_vars.explain_context_data:
            has_context = True
        # Check if there are any analysis context items
        if hasattr(global_vars, 'analysis_context') and global_vars.analysis_context.get('analysis_path'):
            has_context = True
        # Check if there are any selected metrics
        if hasattr(global_vars, 'selected_metrics') and global_vars.selected_metrics:
            has_context = True
    except Exception:
        pass
    
    if has_context:
        # Show confirmation modal instead of directly creating snapshot
        return dash.no_update, dash.no_update, dash.no_update, True
    else:
        # No context to clear, proceed directly
        return _create_snapshot_and_update(primary_data, primary_cols, secondary_data, secondary_cols)


def _create_snapshot_and_update(primary_data, primary_cols, secondary_data, secondary_cols):
    """Helper function to create snapshot and return updated UI state"""
    create_original_if_absent()
    # 1) Persist current Preview tables back to global_vars before snapshot
    try:
        if primary_data is not None and primary_cols is not None:
            col_ids = [c.get('id') for c in primary_cols] if isinstance(primary_cols, list) else None
            if col_ids:
                df_primary = pd.DataFrame(primary_data)
                # Ensure column order per DataTable columns
                df_primary = df_primary[col_ids] if all(cid in df_primary.columns for cid in col_ids) else df_primary
                global_vars.df = df_primary
        if secondary_data is not None and secondary_cols is not None and len(secondary_data) > 0:
            col_ids2 = [c.get('id') for c in secondary_cols] if isinstance(secondary_cols, list) else None
            if col_ids2:
                df_secondary = pd.DataFrame(secondary_data)
                df_secondary = df_secondary[col_ids2] if all(cid in df_secondary.columns for cid in col_ids2) else df_secondary
                global_vars.secondary_df = df_secondary
    except Exception as _:
        # If preview is empty or any mismatch happens, fall back to current global vars
        pass
    # 2) Create new snapshot from current global state
    ver = create_snapshot()
    items = list_snapshots()
    options = []
    for s in items:
        label_base = 'Original' if s.get('ver') == 'original' else s.get('ver')
        time_str = str(s.get('time', '')).replace('T', ' ')
        desc = s.get('desc')
        label = f"{label_base} ({time_str})" + (f" - {desc}" if desc else "")
        options.append({"label": label, "value": s.get('ver')})
    # auto-restore the newly created version and trigger refresh
    ok = restore_snapshot(ver)
    if not ok:
        raise PreventUpdate
    # Mark metrics as outdated so chat Detect recomputes immediately
    try:
        from UI.functions.global_vars import global_vars
        if hasattr(global_vars, 'dataset_change_flags'):
            global_vars.dataset_change_flags['metrics_outdated'] = True
    except Exception:
        pass
    from datetime import datetime as _dt2
    trigger = {"reason": "snapshot_save_restore", "ver": ver, "ts": _dt2.now().isoformat()}
    return options, ver, trigger


from datetime import datetime as _dt

# Callback to handle context version confirmation modal
@app.callback(
    [Output("snapshot-selector", "options", allow_duplicate=True),
     Output("snapshot-selector", "value", allow_duplicate=True),
     Output('table-update-trigger', 'data', allow_duplicate=True),
     Output("context-version-modal", "is_open", allow_duplicate=True),
     Output("chat-context-data", "data", allow_duplicate=True),
     Output("explain-context-data", "data", allow_duplicate=True)],
    [Input("context-version-confirm", "n_clicks"),
     Input("context-version-cancel", "n_clicks")],
    [State('table-primary-overview', 'data'),
     State('table-primary-overview', 'columns'),
     State('table-secondary-overview', 'data'),
     State('table-secondary-overview', 'columns')],
    prevent_initial_call=True
)
def handle_context_version_confirmation(confirm_clicks, cancel_clicks, primary_data, primary_cols, secondary_data, secondary_cols):
    """Handle user confirmation or cancellation of context clearing"""
    from dash import callback_context
    
    if not callback_context.triggered:
        raise PreventUpdate
    
    triggered_id = callback_context.triggered[0]["prop_id"].split(".")[0]
    
    if triggered_id == "context-version-cancel":
        # User cancelled, just close modal
        return dash.no_update, dash.no_update, dash.no_update, False, dash.no_update, dash.no_update
    elif triggered_id == "context-version-confirm":
        # User confirmed, clear context and create snapshot
        print("[CONTEXT VERSION] User confirmed context clearing, proceeding with snapshot creation...")
        
        # Clear all context data
        _clear_all_context_data()
        
        # Create snapshot and update UI
        options, ver, trigger = _create_snapshot_and_update(primary_data, primary_cols, secondary_data, secondary_cols)
        
        # Add a trigger to force UI refresh
        trigger["context_cleared"] = True
        trigger["timestamp"] = time.time()
        
        return options, ver, trigger, False, [], []  # Clear both context stores
    
    raise PreventUpdate


def _clear_all_context_data():
    """Clear all context data from global_vars and prepare for UI updates"""
    try:
        from UI.functions.global_vars import global_vars
        
        # Clear explain context data
        if hasattr(global_vars, 'explain_context_data'):
            global_vars.explain_context_data = []
            print("[CONTEXT VERSION] Cleared global_vars.explain_context_data")
        
        # Clear any other context-related data
        if hasattr(global_vars, 'analysis_context'):
            global_vars.analysis_context = {
                "current_focus": None,
                "analysis_path": [],
            }
            print("[CONTEXT VERSION] Reset analysis_context")
        
        # Clear metrics cache to force recalculation
        if hasattr(global_vars, 'metrics_cache'):
            global_vars.metrics_cache = None
            print("[CONTEXT VERSION] Cleared metrics_cache")
        
        # Mark metrics as outdated
        if hasattr(global_vars, 'dataset_change_flags'):
            global_vars.dataset_change_flags['metrics_outdated'] = True
            print("[CONTEXT VERSION] Marked metrics as outdated")
            
    except Exception as e:
        print(f"[CONTEXT VERSION] Error clearing context data: {str(e)}")


@app.callback(
    Output('table-update-trigger', 'data', allow_duplicate=True),
    [Input('snapshot-selector', 'value'),
     Input('snapshot-restore-btn', 'n_clicks')],
    prevent_initial_call=True
)
def on_select_snapshot(ver, n_clicks_restore):
    # Only restore on explicit Restore button click to avoid accidental restores on list refresh
    ctx = callback_context
    if not ctx.triggered or ctx.triggered[0]['prop_id'].split('.')[0] != 'snapshot-restore-btn':
        raise PreventUpdate
    if not ver:
        raise PreventUpdate
    ok = restore_snapshot(ver)
    if not ok:
        raise PreventUpdate
    # emit global refresh trigger so preview/detect/explain refresh naturally
    return {"reason": "snapshot_restore", "ver": ver, "ts": _dt.now().isoformat()}

@app.callback(
    [Output('eval-info', 'children'),
     Output('eval-info', 'is_open'),
     Output('eval-res', 'children'),
     Output('fairness-scores', 'children'),
     Output('eval-explanation', 'children'),
     Output({'type': 'spinner-btn', 'index': 0, 'pattern_id': 'evaluate_dataset_button_0'}, 'children', allow_duplicate=True),
     Output("experiment-result-table", 'data', allow_duplicate=True),
     Output("experiment-result", 'data', allow_duplicate=True),
     # Output("recommended-op", "children", allow_duplicate=True), - Removed reference to non-existent component
     # Output("tooltip-expl", "children", allow_duplicate=True), - Removed reference to non-existent component
    ],
    [Input({'type': 'spinner-btn', 'index': 0, 'pattern_id': 'evaluate_dataset_button_0'}, 'children'),
     State('dataset-selection', 'value'),
     State('sensi-attr-selection', 'value'),
     State('label-selection', 'value'),
     State('task-selection', 'value'),
     State('model-selection', 'value'),
     State("experiment-result-table", 'data'),
     State("experiment-result", 'data')],
    prevent_initial_call=True
)
def evaluate_dataset(_, df_id, sens_attr, label, task, model, past_res_table, past_res):
    if global_vars.df is None or not global_vars.data_snapshots:
        return 'No dataset is loaded!', [True], [], [], [], " Run", dash.no_update, dash.no_update # removed references to non-existent components
    if df_id is None or sens_attr is None or label is None or task is None or model is None:
        return 'The experimental setting is incomplete!', [True], [], [], [], " Run", dash.no_update, dash.no_update # removed references to non-existent components
    data = global_vars.data_snapshots[int(df_id) - 1]
    if label in sens_attr:
        return 'The label cannot be in the sensitive attributes!', [True], [], [], [], " Run", dash.no_update, dash.no_update # removed references to non-existent components
    if data[label].dtype in ['float64', 'float32'] and task == 'Classification':
        return ('The target attribute is continuous (float) but the task is set to classification. Consider binning '
                'the target or setting the task to regression.'), [True], [], [], [], " Run", dash.no_update, dash.no_update # removed references to non-existent components
    if data[label].dtype == 'object' or data[label].dtype.name == 'bool' or data[label].dtype.name == 'category':
        if task == 'Regression':
            return 'The target attribute is categorical and cannot be used for regression task.', [True], [], [], [], " Run", dash.no_update, dash.no_update
    de = DatasetEval(data, label, ratio=0.2, task_type=task, sensitive_attribute=sens_attr, model_type=model)
    res, scores = de.train_and_test()
    tables = []
    for tid, frame in enumerate(scores):
        tables.append(dash_table.DataTable(
            id=f'table-{tid + 1}',
            columns=[
                {
                    'name': col, 'id': col, 'type': 'numeric',
                    'format': Format(precision=0, scheme=Scheme.fixed) if frame[col].dtype in ['int64',
                                                                                               'O'] else Format(
                        precision=4, scheme=Scheme.fixed)
                }
                for col in frame.columns
            ],
            data=frame.to_dict('records'),
            style_cell={'textAlign': 'center',
                        'fontFamily': 'Arial'},
            style_header={'backgroundColor': '#614385',
                          'color': 'white',
                          'fontWeight': 'bold'
                          },
            style_table={'overflowX': 'auto', 'marginTop': '20px', 'marginLeft': '0px'},  # Add margin here
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': '#f2f2f2'
                },
                {
                    'if': {'row_index': 'even'},
                    'backgroundColor': 'white'
                },
                # Highlight the last row
                {
                    'if': {'row_index': len(frame) - 1},
                    'backgroundColor': '#ffeb3b',  # Yellow background color for highlighting
                    'fontWeight': 'bold'
                },
            ]
        ))
    if task == 'Classification':
        tooltip = html.Div([
            html.Div([
                html.H5("Results", style={'paddingLeft': 0}),
                html.Span(
                    html.I(className="fas fa-question-circle"),
                    id="tooltip-eval",
                    style={
                        "fontSize": "20px",
                        "color": "#aaa",
                        "cursor": "pointer",
                        "marginLeft": "5px",
                        "alignSelf": "center"
                    }
                )
            ], style={"display": "flex", "alignItems": "center", "justifyContent": "space-between", "width": "100%"}),
            dbc.Tooltip(
                "The figures in the table represent the average predicted probability that the subgroup is classified "
                "into the corresponding category. The disparity score is calculated as the difference between "
                "the maximum and minimum values in each column. A larger score indicates a higher degree of potential drift.",
                target="tooltip-eval",
            ),
        ])
    else:
        tooltip = html.Div([
            html.Div([
                html.H5("Results", style={'paddingLeft': 0}),
                html.Span(
                    html.I(className="fas fa-question-circle"),
                    id="tooltip-eval",
                    style={
                        "fontSize": "20px",
                        "color": "#aaa",
                        "cursor": "pointer",
                        "marginLeft": "5px",
                        "alignSelf": "center"
                    }
                )
            ], style={"display": "flex", "alignItems": "center", "justifyContent": "spaceBetween", "width": "100%"}),
            dbc.Tooltip(
                "The figures in the table represent the predicted mean absolute error for each subgroup in the regression task. "
                "The disparity score is calculated as the difference between "
                "the maximum and minimum values in the column. A larger score indicates a higher degree of potential drift.",
                target="tooltip-eval",
            ),
        ])
    data_string = "\n".join(
        [f"Row {i + 1}: {row}" for i, row in enumerate(tables)]
    )
    query = f"Assess the drift level in the dataset using the following results: {data_string}. The model accuracy is {res}. These results were generated by executing the {task} task with the {model} model. The analysis centers on the sensitive attributes {sens_attr}, with {label} serving as the target attribute. The objective is to identify and minimize disparities among subgroups of the sensitive attributes without sacrificing accuracy."
    answer, media, suggestions, stage, op, expl = query_llm(query, global_vars.current_stage, current_user.id)
    answer = format_reply_to_markdown(answer)
    res_explanation = [dcc.Markdown(answer, className="llm-text")]
    now = datetime.datetime.now()
    formatted_date_time = now.strftime("%Y-%m-%d %H:%M:%S")
    str_sens_attr = ' '.join(sens_attr)
    settings = f"sensitive attributes: {str_sens_attr}, label:{label}, model:{model}, task:{task}"
    past_res_table.append({"Snapshot": df_id, "Timestamp": formatted_date_time, "Result": res,
                           "Setting": settings})
    past_res.append(tables)
    global_vars.agent.add_user_action_to_history(f"I have evaluated the dataset and got the result and disparity scores: {data_string}")
    return "", [False], [html.Hr(), tooltip, res], tables, res_explanation, " Run", past_res_table, past_res


@app.callback(
    [Output("experiment-result-table", 'data', allow_duplicate=True),
     Output("experiment-result", 'data', allow_duplicate=True)],
    [Input("remove-all-result-btn", 'n_clicks')],
    prevent_initial_call=True
)
def remove_all_result(n_clicks):
    if n_clicks:
        return [],{}
    return dash.no_update, dash.no_update


@app.callback(
    [Output('chosen-experiment-res', 'children')],
    [Input('experiment-result-table', 'active_cell')],
    [State("experiment-result","data")],
    prevent_initial_call=True
)
def show_past_experiment_result(active_cell, data):
    if active_cell is None:
        return []  # Return an empty figure if no cell is selected

    # Get the row id from the active cell
    row = active_cell['row']

    return data[row]


@app.callback(
    [Output('comparison-res', 'children', allow_duplicate=True),
     Output('comparison-alert', 'children', allow_duplicate=True),
     Output('comparison-alert', 'is_open', allow_duplicate=True),
     Output({'type': 'spinner-btn', 'index': 10, 'pattern_id': 'compare_datasets_button'}, 'children', allow_duplicate=True),
     # Output("recommended-op", "children", allow_duplicate=True), - Removed reference to non-existent component
     # Output("tooltip-expl", "children", allow_duplicate=True), - Removed reference to non-existent component
    ],
    [Input({'type': 'spinner-btn', 'index': 10, 'pattern_id': 'compare_datasets_button'}, 'children'),
     Input('experiment-result-table', 'selected_rows'),
     Input('experiment-result-table', 'data'),
     Input("experiment-result","data")],
    prevent_initial_call=True

)
def compare_experiment_results(_, selected_rows, table_data, res_data):
    if selected_rows is None:
        return [], "Choose two experiment results to compare.", [True], "Compare" # removed references to non-existent components
    if len(selected_rows)!=2:
        return [], "You can only choose two experiment results to compare.", [True], "Compare" # removed references to non-existent components
    # Get the row id from the active cell
    acc1 = table_data[selected_rows[0]]["Result"]
    acc2 = table_data[selected_rows[1]]["Result"]
    res1 = res_data[selected_rows[0]]
    res2 = res_data[selected_rows[1]]

    res_string1 = "\n".join(
        [f"Row {i + 1}: {row}" for i, row in enumerate(res1)]
    )
    res_string2 = "\n".join(
        [f"Row {i + 1}: {row}" for i, row in enumerate(res2)]
    )
    query = f"Please compare the results of two comparison. The first chosen result has the overall {acc1} and the accuracy across different subgroups and categories is {res_string1}. The second chosen result has the overall {acc2} and the accuracy across different subgroups and categories is {res_string2}. You should consider both the accuracy and disparity score to demonstrate which result is better."
    answer, media, suggestions, stage, op, expl = query_llm(query, global_vars.current_stage, current_user.id)
    answer = format_reply_to_markdown(answer)
    res_comparison = [dcc.Markdown(answer, className="llm-text")]
    return res_comparison, dash.no_update, dash.no_update, "Compare" # removed references to non-existent components


@app.callback(
    [Output("data-stat-modal", "is_open"),
     Output("data-stat-body", "children")],
    [Input("data-stat-button", "n_clicks"),
     Input("data-stat-close", "n_clicks")],
    [State("data-stat-modal", "is_open")],
    prevent_initial_call=True
)
def display_data_stat(n1, n2, is_open):
    if global_vars.df is not None:
        # Summarize the DataFrame and include column names as the first column
        summary = summarize_dataframe(global_vars.df)
        summary.reset_index(inplace=True)  # Turn column names into a column
        summary.rename(columns={"index": "Column Name"}, inplace=True)

        # Ensure serializable
        summary = summary.fillna("").astype(str)
        total_missing = global_vars.df.isnull().sum().sum()
        total_values = global_vars.df.size
        missing_rate = (total_missing / total_values) * 100
        desc = f"This dataset: {global_vars.file_name}, comprises {global_vars.df.shape[0]} rows and {global_vars.df.shape[1]} columns. with an overall missing rate {missing_rate:.2f}%. "
        # Define the DataTable
        table = dash_table.DataTable(
            columns=[
                {"name": col, "id": col} for col in summary.columns
            ],
            data=summary.to_dict("records"),  # Ensure serializable
            style_cell={"textAlign": "center", "fontFamily": "Arial"},
            style_header={"backgroundColor": "#614385", "color": "white", "fontWeight": "bold"},
            style_table={"overflowX": "auto", "marginTop": "20px", "marginLeft": "0px"},
            style_data_conditional=[
                {"if": {"row_index": "odd"}, "backgroundColor": "#f2f2f2"},
                {"if": {"row_index": "even"}, "backgroundColor": "white"},
            ]
        )

        # Toggle modal and return table
        if n1 or n2:
            return [not is_open], [desc,table]

    return [is_open], []


@app.callback(
    [Output("data-stat-summary", "children"),
     Output({'type': 'spinner-btn', 'index': 1, 'pattern_id': 'evaluate_dataset_button_1'}, "children",allow_duplicate=True),
     # Output("recommended-op", "children", allow_duplicate=True), - Removed reference to non-existent component
     # Output("tooltip-expl", "children", allow_duplicate=True), - Removed reference to non-existent component
    ],
    [Input({'type': 'spinner-btn', 'index': 1, 'pattern_id': 'evaluate_dataset_button_1'}, "children"),
     State("data-stat-body", "children")],
    prevent_initial_call=True
)
def display_data_summary(_, data):
    if global_vars.df is not None:
        # Summarize the DataFrame and include column names as the first column
        data_string = "\n".join(
            [f"Row {i + 1}: {row}" for i, row in enumerate(data)]
        )
        query = f"""
                The dataset is with the following summary statistics {data_string}. Please First provide a summary of this 
                dataset and then: 
                1. Identify any notable trends, patterns, or insights based on the provided 
                statistics. 
                2. Highlight potential issues, such as missing values, outliers, or unusual 
                distributions.                 
                3. Identify any signs of drift in the dataset, such as imbalances in distributions across key features.              
                4. Suggest strategies to mitigate drift, such as rebalancing, feature engineering, or fairness-aware 
                approaches. 
                """

        answer, media, suggestions, stage, op, expl = query_llm(query, global_vars.current_stage, current_user.id)
        answer = format_reply_to_markdown(answer)
        global_vars.agent.add_user_action_to_history(f"I have analyzed the dataset. ")
        return [dcc.Markdown(answer, className="llm-text")], "Analyze" # removed references to non-existent components

    return [], "Analyze" # removed references to non-existent components


# DISABLED: Phase 3 - This callback conflicts with unified modal system
# The unified modal system handles menu-dataset-info clicks
# @app.callback(
#     [Output('dataset-info-modal-body', 'children', allow_duplicate=True)],  # Keep only non-conflicting output
#     [Input('menu-dataset-info', 'n_clicks')],
#     [State('query-area', 'children')],
#     prevent_initial_call=True
# )



@app.callback(
    [Output('import-dataset-target', 'data')],
    [Input('import-primary-placeholder', 'data-dummy'),
     Input('import-secondary-placeholder', 'data-dummy')],
    prevent_initial_call=True
)
def update_import_target(primary_trigger, secondary_trigger):
    """
    Update the target dataset based on which menu item was clicked.
    """
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update
        
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if triggered_id == 'import-primary-placeholder':
        return ['primary']  # Return as a list with a single element
    elif triggered_id == 'import-secondary-placeholder':
        return ['secondary']  # Return as a list with a single element
        
    return dash.no_update


@app.callback(
    [Output('upload-modal-title', 'children'),
     Output('upload-modal-subtitle', 'children')],
    [Input('import-dataset-target', 'data'),
     Input('initial-upload', 'data')]
)
def update_modal_title(dataset_target, is_initial):
    """
    Update the modal title and subtitle based on which dataset is being imported
    and whether this is the initial upload or a replacement.
    """
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
    
    # Initial upload after login
    if triggered_id == 'initial-upload' and is_initial:
        return "Welcome to DriftNavi", "Please upload your primary and secondary datasets to get started"
    
    # Dataset replacement through Data Manager
    if dataset_target == ['primary']:
        return "Upload Primary Dataset", "The uploaded file will replace the current primary dataset"
    else:
        return "Upload Secondary Dataset", "The uploaded file will replace the current secondary dataset"


@app.callback(
    [Output('upload-modal', 'is_open', allow_duplicate=True)],  
    [Input('import-dataset-target', 'data')],
    prevent_initial_call=True
)
def open_upload_modal(dataset_target):
    """
    Open the upload modal when the import dataset target changes.
    This complements the JavaScript-based approach.
    """
    # If the dataset target has been updated, it means one of the import options was selected
    if dataset_target:
        return [True]  # Return as a list with a single element
    return dash.no_update


@app.callback(
    [Output('upload-single-dataset', 'style'),
     Output('upload-dual-datasets', 'style')],
    [Input('initial-upload', 'data'),
     Input('import-dataset-target', 'data')]
)
def toggle_upload_interface(is_initial, dataset_target):
    """
    Toggle between the single dataset upload (for replacements) and 
    dual dataset upload (for initial setup) interfaces.
    """
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
    
    # For menu-driven replacements (Data Manager options)
    if triggered_id == 'import-dataset-target' and dataset_target:
        # Show single upload, hide dual upload
        return {"display": "block"}, {"display": "none"}
    
    # For initial login
    if is_initial:
        # Hide single upload, show dual upload
        return {"display": "none"}, {"display": "block"}
    
    # Default case
    return {"display": "none"}, {"display": "block"}


def process_uploaded_file(contents, filename, chat_content, is_primary=True):
    """
    Process a single uploaded file and update the global state based on whether it's primary or secondary.
    
    Args:
        contents: The file contents
        filename: The file name
        chat_content: Current chat content
        is_primary: Whether this is the primary dataset (True) or secondary (False)
        
    Returns:
        dict: Result with error status and message
    """
    if 'csv' not in filename.lower():
        return {
            'error': True,
            'error_msg': "Only CSV files are supported!"
        }
    
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    
    try:
        # Read the csv file
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        
        # Update global variables based on whether this is primary or secondary
        if is_primary:
            # === OPTIMIZATION: Clear metrics cache when dataset changes ===
            global_vars.clear_metrics_cache("Primary dataset changed")
            
            global_vars.file_name = filename
            global_vars.df = df
            
            # Initialize snapshots if not already done
            if not hasattr(global_vars, 'data_snapshots'):
                global_vars.data_snapshots = []
            
            # Add to snapshots if not already present
            if len(global_vars.data_snapshots) == 0:
                global_vars.data_snapshots.append(df)
            
            # Initialize dataset agent
            if not hasattr(global_vars, 'conversation_session'):
                global_vars.conversation_session = f"{int(time.time() * 1000)}-{random.randint(1000, 9999)}"
            
            global_vars.agent = DatasetAgent(global_vars.df, file_name=global_vars.file_name, 
                                          conversation_session=global_vars.conversation_session)
            
            # Initialize fingerprint for primary dataset
            global_vars.initialize_dataset_fingerprints(force=True)
                                          
            # If we already have a secondary dataset, preprocess both together
            if hasattr(global_vars, 'secondary_df') and global_vars.secondary_df is not None:
                # Apply automatic data type detection and conversion
                processed_primary, processed_secondary, column_types = preprocess_datasets(df, global_vars.secondary_df)
                
                # Update with processed datasets
                global_vars.df = processed_primary
                global_vars.secondary_df = processed_secondary
                
                # Store column type information
                if not hasattr(global_vars, 'column_types'):
                    global_vars.column_types = {}
                global_vars.column_types = column_types
                
                # Update the data snapshots with processed data
                if len(global_vars.data_snapshots) > 0:
                    global_vars.data_snapshots[0] = processed_primary
                
                # Update agent with processed data
                global_vars.agent = DatasetAgent(global_vars.df, file_name=global_vars.file_name, 
                                              conversation_session=global_vars.conversation_session)
                global_vars.agent.add_secondary_dataset(global_vars.secondary_df, global_vars.secondary_file_name)
                
                # Initialize dataset fingerprints for change detection
                global_vars.initialize_dataset_fingerprints(force=True)
        else:
            # === OPTIMIZATION: Clear metrics cache when dataset changes ===
            global_vars.clear_metrics_cache("Secondary dataset changed")
            
            global_vars.secondary_file_name = filename
            global_vars.secondary_df = df
            
            # If we have both datasets, preprocess them together
            if hasattr(global_vars, 'df') and global_vars.df is not None:
                # Apply automatic data type detection and conversion
                processed_primary, processed_secondary, column_types = preprocess_datasets(global_vars.df, df)
                
                # Update with processed datasets
                global_vars.df = processed_primary
                global_vars.secondary_df = processed_secondary
                
                # Store column type information
                if not hasattr(global_vars, 'column_types'):
                    global_vars.column_types = {}
                global_vars.column_types = column_types
                
                # Update the data snapshots with processed data
                if len(global_vars.data_snapshots) > 0:
                    global_vars.data_snapshots[0] = processed_primary
            
            # Update the agent to handle the secondary dataset
            if hasattr(global_vars, 'agent') and global_vars.agent is not None:
                global_vars.agent.add_secondary_dataset(global_vars.secondary_df, filename)
            
            # Initialize dataset fingerprints for change detection
            global_vars.initialize_dataset_fingerprints(force=True)
        
        return {
            'error': False,
            'error_msg': "",
            'column_types': getattr(global_vars, 'column_types', {})
        }
    
    except Exception as e:
        return {
            'error': True,
            'error_msg': f"Error processing file: {str(e)}"
        }

@app.callback(
    [Output('table-update-trigger', 'data', allow_duplicate=True),  # Phase 3: Use trigger instead
     # REMOVED: Output('query-area', 'children') - Conflicts with chat_callbacks.py
     Output('upload-data-error-msg', 'children'),
     Output('upload-data-error-msg-primary', 'children'),
     Output('upload-data-error-msg-secondary', 'children'),
     Output('upload-modal', 'is_open', allow_duplicate=True),
     Output('initial-upload', 'data'),
     Output('column-types-store', 'data'),
     Output('target-attribute-modal', 'is_open', allow_duplicate=True)],  # target attribute modal
    [Input('upload-data-modal', 'contents'),
     Input('upload-data-modal-primary', 'contents'),
     Input('upload-data-modal-secondary', 'contents'),
     Input('close-upload-modal', 'n_clicks')],
    [State('upload-data-modal', 'filename'),
     State('upload-data-modal-primary', 'filename'),
     State('upload-data-modal-secondary', 'filename'),
     State('query-area', 'children'),
     State('upload-modal', 'is_open'),
     State('import-dataset-target', 'data'),
     State('initial-upload', 'data'),
     State('column-types-store', 'data')],
    prevent_initial_call=True
)
def import_data_and_update_table(
        single_contents, primary_contents, secondary_contents, close_clicks,
        single_filename, primary_filename, secondary_filename, 
        chat_content, is_open, dataset_target, is_initial, column_types_data
):
    """
    Process uploaded files and update the data table.
    
    This callback handles both initial uploads (primary and secondary datasets)
    and replacements through the Data Manager.
    """
    triggered_id = callback_context.triggered[0]['prop_id'].split('.')[0]
    
    if chat_content is None:
        chat_content = []
    
    # Initialize column types store if not already created
    if column_types_data is None:
        column_types_data = {}
    
    # Close modal button clicked
    if triggered_id == 'close-upload-modal' and close_clicks:
        return dash.no_update, "", "", "", False, is_initial, column_types_data, dash.no_update
    
    # SINGLE DATASET REPLACEMENT (from Data Manager)
    if triggered_id == 'upload-data-modal' and single_contents is not None:
        try:
            # Determine if this is primary or secondary based on the stored target
            is_primary = dataset_target == ['primary']
            
            # Process the dataset
            result = process_uploaded_file(single_contents[0], single_filename[0], chat_content, is_primary=is_primary)
            
            if result['error']:
                return dash.no_update, result['error_msg'], "", "", True, False, column_types_data, dash.no_update
            
            # Update column types data
            if 'column_types' in result:
                column_types_data = result['column_types']

                # Add encoding notification for categorical string columns
                # Get datasets for analysis
                primary_df = global_vars.df if hasattr(global_vars, 'df') else None
                secondary_df = global_vars.secondary_df if hasattr(global_vars, 'secondary_df') else None

                # Identify columns that will be encoded
                categorical_encoding_columns = identify_categorical_encoding_columns(
                    primary_df, secondary_df, column_types_data
                )

                # Add encoding notification if applicable (suppress auto-detected types message)
                encoding_notification = create_encoding_notification_text(
                    categorical_encoding_columns, is_comprehensive=True
                )

                if encoding_notification:
                    chat_content.append(dcc.Markdown(encoding_notification, className="llm-msg"))
            
            # Suppress verbose upload success chat message
            
            # Define modal visibility default for single replacement flow
            show_target_modal = False
            
            # Return updated table and close the modal
            # Create trigger for Phase 3 table update instead of direct table data
            trigger_data = {
                'timestamp': time.time(),
                'source': 'DataUpload',
                'reason': 'Single dataset uploaded successfully',
                'trigger_id': f"upload_{int(time.time())}"
            }
            
            return (
                trigger_data,  # Phase 3: Use trigger instead of direct table data
                "",  # Clear error message
                "",  # Clear primary error message
                "",  # Clear secondary error message
                False,  # Close modal after upload
                False,  # No longer initial upload
                column_types_data,  # Return updated column types
                False  # show target attribute modal
            )
        except Exception as e:
            return dash.no_update, f"Error processing file: {str(e)}", "", "", True, is_initial, column_types_data, dash.no_update
    
    # INITIAL UPLOAD - PRIMARY DATASET
    if triggered_id == 'upload-data-modal-primary' and primary_contents is not None:
        try:
            # Process the primary dataset
            result = process_uploaded_file(primary_contents[0], primary_filename[0], chat_content, is_primary=True)
            
            if result['error']:
                return dash.no_update, "", result['error_msg'], "", True, is_initial, column_types_data, dash.no_update
            
            # Update column types data
            if 'column_types' in result:
                column_types_data = result['column_types']
            
            # Suppress verbose upload success chat message
            
            # Return updated table and keep modal open for secondary dataset
            # Create trigger for Phase 3 table update instead of direct table data
            trigger_data = {
                'timestamp': time.time(),
                'source': 'DataUpload',
                'reason': 'Primary dataset uploaded successfully',
                'trigger_id': f"upload_{int(time.time())}"
            }
            
            return (
                trigger_data,  # Phase 3: Use trigger instead of direct table data
                "",  # Clear single error message
                "",  # Clear primary error message
                "",  # Keep secondary error message
                True,  # Keep modal open
                is_initial,  # Keep initial upload state
                column_types_data,  # Return column types data
                False  # do not show target attribute modal
            )
        except Exception as e:
            return dash.no_update, "", f"Error processing file: {str(e)}", "", True, is_initial, column_types_data, dash.no_update
    
    # INITIAL UPLOAD - SECONDARY DATASET
    if triggered_id == 'upload-data-modal-secondary' and secondary_contents is not None:
        try:
            # Check if primary dataset exists
            if not hasattr(global_vars, 'df') or global_vars.df is None:
                return dash.no_update, "", "", "Please upload a primary dataset first", True, is_initial, column_types_data, dash.no_update
            
            # Process the secondary dataset
            result = process_uploaded_file(secondary_contents[0], secondary_filename[0], chat_content, is_primary=False)
            
            if result['error']:
                return dash.no_update, "", "", result['error_msg'], True, is_initial, column_types_data, dash.no_update
            
            # Update column types data
            if 'column_types' in result:
                column_types_data = result['column_types']

                # Add encoding notification for categorical string columns
                # Get datasets for analysis
                primary_df = global_vars.df if hasattr(global_vars, 'df') else None
                secondary_df = global_vars.secondary_df if hasattr(global_vars, 'secondary_df') else None

                # Identify columns that will be encoded
                categorical_encoding_columns = identify_categorical_encoding_columns(
                    primary_df, secondary_df, column_types_data
                )

                # Add encoding notification if applicable (suppress auto-detected types message)
                encoding_notification = create_encoding_notification_text(
                    categorical_encoding_columns, is_comprehensive=True
                )

                if encoding_notification:
                    chat_content.append(dcc.Markdown(encoding_notification, className="llm-msg"))
            
            # Suppress verbose upload success chat message
            

            
            # Return updated table and close the modal
            # Create trigger for Phase 3 table update instead of direct table data
            trigger_data = {
                'timestamp': time.time(),
                'source': 'DataUpload',
                'reason': 'Secondary dataset uploaded successfully',
                'trigger_id': f"upload_{int(time.time())}"
            }
            
            return (
                trigger_data,  # Phase 3: Use trigger instead of direct table data
                "",  # Clear single error message
                "",  # Clear primary error message
                "",  # Clear secondary error message
                False,  # Close modal after upload
                False,  # No longer initial upload after both datasets are loaded
                column_types_data,  # Return updated column types
                True  # Show target attribute modal after both datasets loaded
            )
        except Exception as e:
            return dash.no_update, "", "", f"Error processing file: {str(e)}", True, is_initial, column_types_data, dash.no_update
    
    return dash.no_update, "", "", "", dash.no_update, is_initial, column_types_data, dash.no_update




# REMOVED: Legacy create_column_type_comparison() function
# This function used pattern-matching dropdowns that caused ID conflicts.
# Replaced with modern create_modern_column_comparison() in Phase 4 cleanup.
# The new component integrates with Phase 1-3 unified system and provides
# read-only display with link to unified modal for editing.


@app.callback(
    [Output('table-primary-overview', 'data', allow_duplicate=True),
     Output('table-primary-overview', 'columns', allow_duplicate=True),
     Output('table-secondary-overview', 'data', allow_duplicate=True),
     Output('table-secondary-overview', 'columns', allow_duplicate=True)],
    [Input('table-overview', 'data'),
     Input('table-overview', 'columns')],
    prevent_initial_call=True
)
def update_tables_from_main(main_data, main_columns):
    """
    Update the primary and secondary tables when the main table is modified.
    This ensures all tables stay in sync.
    """
    # Initialize return values
    primary_data = []
    primary_columns = []
    secondary_data = []
    secondary_columns = []
    
    if hasattr(global_vars, 'df') and global_vars.df is not None:
        primary_data = global_vars.df.to_dict('records')
        primary_columns = [{"name": col, "id": col, 'deletable': True} for col in global_vars.df.columns]
    
    if hasattr(global_vars, 'secondary_df') and global_vars.secondary_df is not None:
        secondary_data = global_vars.secondary_df.to_dict('records')
        secondary_columns = [{"name": col, "id": col, 'deletable': True} for col in global_vars.secondary_df.columns]
    
    return primary_data, primary_columns, secondary_data, secondary_columns


@app.callback(
    [Output('distribution-charts-container', 'style'),
     Output('selected-cell-info', 'children'),
     Output('combined-distribution-chart', 'children'),
     Output('distribution-comparison-summary', 'children')],
    [Input('table-primary-overview', 'active_cell'),
     Input('table-secondary-overview', 'active_cell'),
     Input('table-overview', 'active_cell')],
    [State('table-primary-overview', 'data'),
     State('table-primary-overview', 'columns'),
     State('table-secondary-overview', 'data'),
     State('table-secondary-overview', 'columns'),
     State('table-overview', 'data'),
     State('table-overview', 'columns')],
    prevent_initial_call=True
)
def update_distribution_charts(primary_active_cell, secondary_active_cell, main_active_cell,
                              primary_data, primary_columns,
                              secondary_data, secondary_columns,
                              main_data, main_columns):
    """
    Generate distribution charts when a cell is clicked in any table.
    This unified callback handles clicks from all tables without creating circular dependencies.
    """
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
    
    # Default return values
    no_chart_return = {'display': 'none'}, None, None, None
    
    # If no cell is active in any table, hide the charts
    if not triggered_id or triggered_id not in ['table-primary-overview', 'table-secondary-overview', 'table-overview']:
        return no_chart_return
    
    # Determine which table was clicked and get data
    if triggered_id == 'table-primary-overview' and primary_active_cell is not None:
        active_cell = primary_active_cell
        data = primary_data
        columns = primary_columns
        dataset_type = "Primary dataset"
    elif triggered_id == 'table-secondary-overview' and secondary_active_cell is not None:
        active_cell = secondary_active_cell
        data = secondary_data
        columns = secondary_columns
        dataset_type = "Secondary dataset"
    elif triggered_id == 'table-overview' and main_active_cell is not None:
        active_cell = main_active_cell
        data = main_data
        columns = main_columns
        dataset_type = "Main table"
    else:
        return no_chart_return
    
    # Extract cell info
    try:
        row_idx = active_cell['row']
        col_idx = active_cell['column']
        row = data[row_idx]
        column_name = columns[col_idx]['name']
        cell_value = row[column_name]
    except (TypeError, IndexError, KeyError):
        # Handle cases where data structure doesn't match expectations
        return no_chart_return
    
    # Check if we have datasets to work with
    has_primary = hasattr(global_vars, 'df') and global_vars.df is not None
    has_secondary = hasattr(global_vars, 'secondary_df') and global_vars.secondary_df is not None
    
    if not has_primary:
        return {'display': 'none'}, "No dataset available", None, None
    
    # Create info about the selected cell
    cell_info = html.Div([
        html.P(f"Selected from {dataset_type}:", style={"fontWeight": "bold"}),
        html.P(f"Column: {column_name}, Value: {cell_value}")
    ])
    
    # Generate combined distribution chart if both datasets exist
    if has_secondary:
        combined_fig = generate_combined_distribution_chart(
            primary_df=global_vars.df,
            secondary_df=global_vars.secondary_df,
            column_name=column_name,
            highlight_value=cell_value
        )
        
        # Generate comparison summary for backend use (add to chat/explain functionality)
        # Note: UI display is hidden, but data is still calculated for context functions
        comparison_summary = generate_comparison_summary(
            primary_df=global_vars.df,
            secondary_df=global_vars.secondary_df,
            column_name=column_name,
            value=cell_value
        )
        
        # Return the combined chart
        return {'display': 'block'}, cell_info, combined_fig, comparison_summary
    else:
        # If there's no secondary dataset, just show the primary
        primary_fig = generate_distribution_chart(
            df=global_vars.df,
            column_name=column_name,
            highlight_value=cell_value,
            is_primary=True
        )
        
        # Generate appropriate message for single dataset case
        comparison_summary = html.Div("Upload a secondary dataset to see comparison")
        
        # Return the chart
        return {'display': 'block'}, cell_info, primary_fig, comparison_summary


def generate_combined_distribution_chart(primary_df, secondary_df, column_name, highlight_value):
    """
    Generate a single distribution chart showing both primary and secondary datasets
    with different colors for easy comparison, using a professional style inspired by fairlens
    
    Parameters:
    -----------
    primary_df : pandas DataFrame
        The primary dataset to visualize
    secondary_df : pandas DataFrame
        The secondary dataset to visualize
    column_name : str
        Name of the column to create distribution for
    highlight_value : any
        Value to highlight in the distribution
    """
    import plotly.graph_objects as go
    import plotly.express as px
    import numpy as np
    
    # Check if column exists in both datasets
    if column_name not in primary_df.columns:
        return html.Div(f"Column '{column_name}' not found in primary dataset")
    
    if column_name not in secondary_df.columns:
        return html.Div(f"Column '{column_name}' not found in secondary dataset")
    
    # Get column data from both datasets
    primary_series = primary_df[column_name]
    secondary_series = secondary_df[column_name]
    
    # Use Plotly's Pastel color palette (more professional look)
    colors = px.colors.qualitative.Pastel
    primary_color = colors[0]  # Light blue
    secondary_color = colors[1]  # Light pink or orange
    
    # Handle different data types
    if pd.api.types.is_numeric_dtype(primary_series) and pd.api.types.is_numeric_dtype(secondary_series):
        # For numeric data, create histograms for both datasets
        
        # Calculate binning that works for both datasets
        min_val = min(primary_series.min(), secondary_series.min())
        max_val = max(primary_series.max(), secondary_series.max())
        
        # Create figure
        fig = go.Figure()
        
        # Check if the data is likely discrete (integer-like values)
        # Combine the series using pd.concat instead of append (which is deprecated)
        combined_series = pd.concat([primary_series.dropna(), secondary_series.dropna()])
        
        # Check if all numeric values are effectively integers
        # For int types: they are already integers
        # For float types: check if they can be represented as integers (e.g. 1.0, 2.0)
        is_discrete = all(
            (isinstance(val, int) or  # Integer types are discrete by definition
             (isinstance(val, float) and val.is_integer()))  # Check if float values are effectively integers
            for val in combined_series
        )
        
        # Count unique values to determine if this is a small set of discrete values
        unique_values = sorted(list(set(primary_series.dropna().tolist() + secondary_series.dropna().tolist())))
        few_unique_values = len(unique_values) <= 20  # Threshold for considering as "few" unique values
        
        # Modified binning strategy based on data characteristics
        if is_discrete and few_unique_values:
            # For discrete data with few unique values, create bins centered on each unique integer
            
            # Ensure bin edges include every integer between min and max
            # Add 0.5 offset to create bins centered on integers
            bin_edges = [min_val - 0.5]  # Start half a unit below min value
            
            # If 0 exists in the range but not in unique values, ensure it's included
            if min_val < 0 and max_val > 0 and 0 not in unique_values:
                unique_values = sorted(list(unique_values) + [0])
                
            # Create bin edges around each integer value
            for i in range(len(unique_values) - 1):
                current_val = unique_values[i]
                next_val = unique_values[i + 1]
                
                # If values are not consecutive, add appropriate bin edges
                if next_val - current_val > 1:
                    # Add current bin upper edge
                    bin_edges.append(current_val + 0.5)
                    # Add next bin lower edge
                    bin_edges.append(next_val - 0.5)
                else:
                    # For consecutive values, add the midpoint as a bin edge
                    bin_edges.append(current_val + 0.5)
            
            # Add final bin edge
            bin_edges.append(max_val + 0.5)
            
            # Add primary dataset histogram with custom bins
            fig.add_trace(go.Histogram(
                x=primary_series,
                name='Primary Dataset',
                marker=dict(color=primary_color, line=dict(color=primary_color, width=1)),
                opacity=0.7,
                autobinx=False,
                xbins=dict(start=bin_edges[0], end=bin_edges[-1], size=None)
            ))
            
            # Add secondary dataset histogram with same custom bins
            fig.add_trace(go.Histogram(
                x=secondary_series,
                name='Secondary Dataset',
                marker=dict(color=secondary_color, line=dict(color=secondary_color, width=1)),
                opacity=0.7,
                autobinx=False,
                xbins=dict(start=bin_edges[0], end=bin_edges[-1], size=None)
            ))
        else:
            # For continuous data or discrete data with many unique values, use enhanced Sturges' rule
            combined_count = len(primary_df) + len(secondary_df)
            bin_count = max(8, int(np.ceil(np.log2(combined_count) + 1)))  # Ensure minimum 8 bins
            
            # Calculate bin size - ensure we capture the full range
            bin_size = (max_val - min_val) / bin_count if max_val > min_val else 1
            
            # Adjust min_val and max_val slightly to ensure edges are included
            adjusted_min = min_val - (bin_size * 0.1)
            adjusted_max = max_val + (bin_size * 0.1)
            
            # Add primary dataset histogram
            fig.add_trace(go.Histogram(
                x=primary_series,
                name='Primary Dataset',
                marker=dict(color=primary_color, line=dict(color=primary_color, width=1)),
                opacity=0.7,
                autobinx=False,
                xbins=dict(start=adjusted_min, end=adjusted_max, size=bin_size),
                bingroup=1  # Ensure both histograms use same bins
            ))
            
            # Add secondary dataset histogram
            fig.add_trace(go.Histogram(
                x=secondary_series,
                name='Secondary Dataset',
                marker=dict(color=secondary_color, line=dict(color=secondary_color, width=1)),
                opacity=0.7,
                autobinx=False,
                xbins=dict(start=adjusted_min, end=adjusted_max, size=bin_size),
                bingroup=1  # Ensure both histograms use same bins
            ))
        
        # Overlay both histograms
        fig.update_layout(
            barmode='overlay',
            bargap=0.1,
        )
        
        # Add vertical line for the selected value and set initial zoom window
        try:
            value = float(highlight_value)
            # Add the vertical line marking selected value
            fig.add_vline(
                x=value,
                line=dict(dash="dash", color="red", width=2),
                annotation_text="Selected Value",
                annotation_position="top right",
                annotation=dict(font=dict(color="red"))
            )
            
            # Intelligent adaptive zoom strategy for better user experience
            data_range = max_val - min_val
            
            if data_range > 0:
                # Use adaptive zoom based on data characteristics and available space
                # Now that chart occupies full width, we can show more context
                
                # Determine zoom level based on data distribution and characteristics
                if data_range <= 10:
                    # For small ranges, show full data with some padding
                    zoom_factor = 1.2  # Show 120% of range for context
                elif data_range <= 100:
                    # For medium ranges, show 60% of data centered on selected value
                    zoom_factor = 0.6
                else:
                    # For large ranges, show 40% of data but ensure meaningful context
                    zoom_factor = 0.4
                
                # Calculate adaptive window size
                window_size = data_range * zoom_factor
                
                # Ensure window is not too small for discrete data
                if is_discrete and window_size < 10:
                    window_size = min(10, data_range * 0.8)  # Show at least 80% for discrete data
                
                # For very large datasets, ensure we don't zoom in too much
                if data_range > 1000:
                    window_size = max(window_size, data_range * 0.3)  # Never less than 30% for large ranges
                
                # Calculate window boundaries, ensuring they're within the data range
                window_min = max(min_val, value - window_size/2)
                window_max = min(max_val, value + window_size/2)
                
                # Ensure window has minimum size and doesn't exceed data bounds
                if window_max - window_min < window_size:
                    if window_min == min_val:
                        window_max = min(max_val, window_min + window_size)
                    else:
                        window_min = max(min_val, window_max - window_size)
                
                # Only apply zoom if it provides meaningful focus (not too close to full range)
                zoom_ratio = (window_max - window_min) / data_range
                if zoom_ratio > 0.9:
                    # If zoom window covers >90% of data, show full range instead
                    padding = data_range * 0.05  # 5% padding on each side
                    window_min = max(min_val - padding, min_val * 0.95)
                    window_max = min(max_val + padding, max_val * 1.05)
                
                # Set the initial x-axis range with improved bounds
                fig.update_xaxes(range=[window_min, window_max])
        except (ValueError, TypeError):
            # If value can't be converted to float, don't add the line or zoom
            pass
            
    else:
        # For categorical data, create bar charts of value counts side by side
        primary_counts = primary_series.value_counts().reset_index()
        primary_counts.columns = ['value', 'count']
        
        secondary_counts = secondary_series.value_counts().reset_index()
        secondary_counts.columns = ['value', 'count']
        
        # Get all unique categories
        all_categories = set(primary_counts['value']).union(set(secondary_counts['value']))
        
        # Add missing categories with zero count
        for category in all_categories:
            if category not in primary_counts['value'].values:
                primary_counts = pd.concat([primary_counts, 
                                         pd.DataFrame({'value': [category], 
                                                    'count': [0]})],
                                       ignore_index=True)
            
            if category not in secondary_counts['value'].values:
                secondary_counts = pd.concat([secondary_counts, 
                                          pd.DataFrame({'value': [category], 
                                                     'count': [0]})],
                                        ignore_index=True)
        
        # Sort both dataframes by value for consistent ordering
        primary_counts = primary_counts.sort_values('value').reset_index(drop=True)
        secondary_counts = secondary_counts.sort_values('value').reset_index(drop=True)
        
        # Normalize counts to percentages for better comparison
        primary_total = primary_counts['count'].sum()
        secondary_total = secondary_counts['count'].sum()
        
        primary_counts['percentage'] = primary_counts['count'] / primary_total * 100 if primary_total > 0 else 0
        secondary_counts['percentage'] = secondary_counts['count'] / secondary_total * 100 if secondary_total > 0 else 0
        
        # Create figure
        fig = go.Figure()
        
        # Add bar for primary dataset
        fig.add_trace(go.Bar(
            x=primary_counts['value'],
            y=primary_counts['percentage'],
            name='Primary Dataset',
            marker=dict(
                color=primary_color,
                line=dict(color=primary_color, width=1)
            ),
            opacity=0.7
        ))
        
        # Add bar for secondary dataset
        fig.add_trace(go.Bar(
            x=secondary_counts['value'],
            y=secondary_counts['percentage'],
            name='Secondary Dataset',
            marker=dict(
                color=secondary_color,
                line=dict(color=secondary_color, width=1)
            ),
            opacity=0.7
        ))
        
        # Set barmode to group for side-by-side bars
        fig.update_layout(
            barmode='group',
            bargap=0.2,
            xaxis_title=column_name,
            yaxis_title="Percentage (%)"
        )
        
        # Highlight the selected value
        if highlight_value in all_categories:
            fig.add_annotation(
                x=highlight_value,
                y=max(primary_counts['percentage'].max(), secondary_counts['percentage'].max()) * 1.1,
                text="Selected Value",
                showarrow=True,
                arrowhead=1,
                ax=0,
                ay=-40,
                font=dict(color="red", size=12),
                bordercolor="red",
                bgcolor="white"
            )
    
    # Enhanced layout with better zoom control and professional style
    fig.update_layout(
        title={
            'text': f"<b>Distribution Comparison of {column_name}</b>",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title=f"<b>{column_name}</b>",
        yaxis_title="<b>Frequency</b>",
        plot_bgcolor='rgba(240,240,240,0.8)',  # Light gray background
        paper_bgcolor='rgba(0,0,0,0)',
        height=420,  # Increased height for better zoom experience
        margin=dict(l=60, r=50, t=80, b=60),
        font=dict(family="Verdana, Arial, sans-serif"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='rgba(0,0,0,0.2)',
            borderwidth=1
        ),
        hovermode="closest",
        # Enhanced drag and zoom modes for better interaction
        dragmode="zoom",  # Default to zoom mode instead of pan
        # Add annotation to guide users on zoom controls
        annotations=[
            dict(
                text="💡 Tip: Scroll to zoom, drag to pan, double-click to auto-fit",
                showarrow=False,
                xref="paper",
                yref="paper",
                x=0.02,
                y=0.02,
                xanchor="left",
                yanchor="bottom",
                font=dict(size=10, color="gray"),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="gray",
                borderwidth=1
            )
        ]
    )
    
    # Add grid lines for better readability
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(211,211,211,0.6)',
        zeroline=True,
        zerolinewidth=1.5,
        zerolinecolor='rgba(0,0,0,0.2)'
    )
    
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(211,211,211,0.6)',
        zeroline=True,
        zerolinewidth=1.5,
        zerolinecolor='rgba(0,0,0,0.2)'
    )
    
    # Enhanced Plotly configuration for better user experience
    plotly_config = {
        'displayModeBar': True,
        'scrollZoom': True,
        'doubleClick': 'autosize',  # Double-click to auto-fit
        'showTips': True,
        'displaylogo': False,
        'modeBarButtonsToAdd': ['pan2d', 'select2d', 'lasso2d'],
        'modeBarButtonsToRemove': ['autoScale2d'],  # Remove auto-scale to prevent conflicts
        'toImageButtonOptions': {
            'format': 'png',
            'filename': f'distribution_comparison_{column_name}',
            'height': 500,
            'width': 1000,
            'scale': 1
        }
    }
    
    return dcc.Graph(
        figure=fig, 
        config=plotly_config,
        style={"height": "450px"}  # Slightly increased height for better visibility
    )


def generate_distribution_chart(df, column_name, highlight_value, is_primary, x_range=None):
    """
    Generate a distribution chart for a column with a highlighted value
    
    Parameters:
    -----------
    df : pandas DataFrame
        The dataset to visualize
    column_name : str
        Name of the column to create distribution for
    highlight_value : any
        Value to highlight in the distribution
    is_primary : bool
        Whether this is the primary (True) or secondary (False) dataset
    x_range : tuple, optional
        Range for x-axis (min, max) to ensure consistent scales between primary and secondary
    """
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import numpy as np
    
    if column_name not in df.columns:
        return html.Div(f"Column '{column_name}' not found in {'primary' if is_primary else 'secondary'} dataset")
    
    # Get column data
    series = df[column_name]
    
    # Handle different data types
    if pd.api.types.is_numeric_dtype(series):
        # For numeric data, create a histogram
        fig = px.histogram(
            df, 
            x=column_name,
            title=f"Distribution of {column_name}" + (" (Primary)" if is_primary else " (Secondary)"),
            color_discrete_sequence=['#614385' if is_primary else '#516395'],
            opacity=0.6
        )
        
        # Add vertical line for the selected value if it's numeric
        try:
            value = float(highlight_value)
            fig.add_vline(
                x=value,
                line_dash="dash",
                line_color="red",
                annotation_text="Selected Value",
                annotation_position="top right"
            )
        except (ValueError, TypeError):
            # If value can't be converted to float, don't add the line
            pass
            
        # Set x-axis range if provided
        if x_range is not None and len(x_range) == 2:
            fig.update_xaxes(range=x_range)
            
    else:
        # For categorical data, create a bar chart of value counts
        value_counts = series.value_counts().reset_index()
        value_counts.columns = ['value', 'count']
        
        fig = px.bar(
            value_counts, 
            x='value', 
            y='count',
            title=f"Distribution of {column_name}" + (" (Primary)" if is_primary else " (Secondary)"),
            color_discrete_sequence=['#614385' if is_primary else '#516395'],
            opacity=0.6
        )
        
        # Highlight the selected value
        if highlight_value in value_counts['value'].values:
            # Create a list of colors, with the selected value highlighted
            colors = ['#614385' if is_primary else '#516395'] * len(value_counts)
            selected_idx = value_counts[value_counts['value'] == highlight_value].index[0]
            colors[selected_idx] = 'red'
            
            fig.update_traces(marker_color=colors)
            
            # Add annotation for the selected value
            selected_count = value_counts.loc[selected_idx, 'count']
            fig.add_annotation(
                x=highlight_value,
                y=selected_count,
                text="Selected Value",
                showarrow=True,
                arrowhead=1,
                ax=0,
                ay=-40
            )
            
        # For categorical data, we'll ensure the same categories are shown
        # even if they don't exist in one of the datasets
        if x_range is not None and isinstance(x_range, list):
            fig.update_xaxes(categoryorder='array', categoryarray=x_range)
    
    # Improve layout
    fig.update_layout(
        height=400,
        margin=dict(l=50, r=50, t=50, b=50),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0.02)'
    )
    
    return dcc.Graph(figure=fig)


def generate_comparison_summary(primary_df, secondary_df, column_name, value):
    """
    Generate a summary comparing the distributions of a column in both datasets.
    Handles both numeric and categorical data (including strings) appropriately.
    
    Parameters:
    -----------
    primary_df : pandas DataFrame
        The primary dataset
    secondary_df : pandas DataFrame
        The secondary dataset for comparison
    column_name : str
        Name of the column to analyze
    value : any
        Selected value to highlight in the comparison
    
    Returns:
    --------
    dash_html_components.Div
        HTML component with the comparison summary
    """
    # Check if column exists in both datasets
    if column_name not in primary_df.columns:
        return html.Div(f"Column '{column_name}' not found in primary dataset")
    
    if column_name not in secondary_df.columns:
        return html.Div(f"Column '{column_name}' not found in secondary dataset")
    
    # Get column data from both datasets
    primary_series = primary_df[column_name]
    secondary_series = secondary_df[column_name]
    
    # Get data type information
    pandas_type = primary_series.dtype
    
    # Get intelligent type classification if available
    intelligent_type = "Unknown"
    if hasattr(global_vars, 'column_types') and column_name in global_vars.column_types:
        intelligent_type = global_vars.column_types[column_name]
    
    # Create a data type information component
    data_type_info = html.Div([
        html.P([
            "Column: ", 
            html.Strong(column_name),
            f" | Data Type: ",
            html.Strong(f"{pandas_type}"),
            f" | Classification: ",
            html.Strong(f"{intelligent_type}")
        ], style={"backgroundColor": "#f0f8ff", "padding": "8px", "borderRadius": "4px", "marginBottom": "10px"})
    ])
    
    # Generate summary statistics
    try:
        if pd.api.types.is_numeric_dtype(primary_series) and pd.api.types.is_numeric_dtype(secondary_series):
            # For numeric data, compare statistics
            primary_stats = primary_series.describe().to_dict()
            secondary_stats = secondary_series.describe().to_dict()
            
            # Calculate percentile of selected value in both distributions
            try:
                selected_value = float(value)
                primary_pct = (primary_series <= selected_value).mean() * 100
                secondary_pct = (secondary_series <= selected_value).mean() * 100
                
                percentile_comparison = html.Div([
                    html.P([
                        "Selected value ",
                        html.Strong(value),
                        f" is at the {primary_pct:.1f}th percentile in the primary dataset and the {secondary_pct:.1f}th percentile in the secondary dataset."
                    ])
                ])
            except (ValueError, TypeError):
                percentile_comparison = html.Div()
            
            # Create a comparison table
            comparison_table = html.Table([
                html.Thead(html.Tr([
                    html.Th("Statistic"), 
                    html.Th("Primary Dataset"), 
                    html.Th("Secondary Dataset"),
                    html.Th("Difference")
                ])),
                html.Tbody([
                    html.Tr([
                        html.Td("Count"),
                        html.Td(f"{primary_stats['count']:.0f}"),
                        html.Td(f"{secondary_stats['count']:.0f}"),
                        html.Td(f"{primary_stats['count'] - secondary_stats['count']:.0f}")
                    ]),
                    html.Tr([
                        html.Td("Mean"),
                        html.Td(f"{primary_stats['mean']:.2f}"),
                        html.Td(f"{secondary_stats['mean']:.2f}"),
                        html.Td(f"{primary_stats['mean'] - secondary_stats['mean']:.2f}")
                    ]),
                    html.Tr([
                        html.Td("Standard Deviation"),
                        html.Td(f"{primary_stats['std']:.2f}"),
                        html.Td(f"{secondary_stats['std']:.2f}"),
                        html.Td(f"{primary_stats['std'] - secondary_stats['std']:.2f}")
                    ]),
                    html.Tr([
                        html.Td("Min"),
                        html.Td(f"{primary_stats['min']:.2f}"),
                        html.Td(f"{secondary_stats['min']:.2f}"),
                        html.Td(f"{primary_stats['min'] - secondary_stats['min']:.2f}")
                    ]),
                    html.Tr([
                        html.Td("25%"),
                        html.Td(f"{primary_stats['25%']:.2f}"),
                        html.Td(f"{secondary_stats['25%']:.2f}"),
                        html.Td(f"{primary_stats['25%'] - secondary_stats['25%']:.2f}")
                    ]),
                    html.Tr([
                        html.Td("Median"),
                        html.Td(f"{primary_stats['50%']:.2f}"),
                        html.Td(f"{secondary_stats['50%']:.2f}"),
                        html.Td(f"{primary_stats['50%'] - secondary_stats['50%']:.2f}")
                    ]),
                    html.Tr([
                        html.Td("75%"),
                        html.Td(f"{primary_stats['75%']:.2f}"),
                        html.Td(f"{secondary_stats['75%']:.2f}"),
                        html.Td(f"{primary_stats['75%'] - secondary_stats['75%']:.2f}")
                    ]),
                    html.Tr([
                        html.Td("Max"),
                        html.Td(f"{primary_stats['max']:.2f}"),
                        html.Td(f"{secondary_stats['max']:.2f}"),
                        html.Td(f"{primary_stats['max'] - secondary_stats['max']:.2f}")
                    ])
                ])
            ], className="comparison-table", style={
                "width": "100%",
                "borderCollapse": "collapse",
                "marginTop": "10px"
            })
            
            return html.Div([
                html.H5("Comparison Summary", style={"textAlign": "center"}),
                data_type_info,
                percentile_comparison,
                html.P(f"Statistical comparison of '{column_name}' between datasets:"),
                comparison_table
            ])
            
        else:
            # For categorical data (including strings), compare frequencies
            primary_counts = primary_series.value_counts(normalize=True) * 100
            secondary_counts = secondary_series.value_counts(normalize=True) * 100
            
            # Get frequency of the selected value in both datasets
            primary_freq = primary_counts.get(value, 0)
            secondary_freq = secondary_counts.get(value, 0)
            
            # Count total unique values to provide info
            total_unique = len(set(primary_counts.index) | set(secondary_counts.index))
            
            # Get top values by frequency (combining both datasets)
            # Create a combined ranking to select the most representative values
            combined_ranking = {}
            for val in set(primary_counts.index) | set(secondary_counts.index):
                # Use the maximum frequency from either dataset to rank values
                combined_ranking[val] = max(primary_counts.get(val, 0), secondary_counts.get(val, 0))
            
            # Sort values by their combined ranking (highest frequency first)
            top_values = sorted(combined_ranking.keys(), key=lambda x: combined_ranking[x], reverse=True)
            
            # Always include the selected value in the top values if it exists
            if value in combined_ranking and value not in top_values[:10]:
                top_values = top_values[:9] + [value]
            else:
                top_values = top_values[:10]
            
            # Initialize rows list (fix for the original bug)
            rows = []
            
            # Add top categories with their counts
            for category in top_values:
                primary_pct = primary_counts.get(category, 0)
                secondary_pct = secondary_counts.get(category, 0)
                diff = primary_pct - secondary_pct
                
                # Highlight the selected value
                if category == value:
                    rows.append(html.Tr([
                        html.Td(str(category), style={"fontWeight": "bold", "backgroundColor": "rgba(255, 0, 0, 0.1)"}),
                        html.Td(f"{primary_pct:.2f}%", style={"fontWeight": "bold", "backgroundColor": "rgba(255, 0, 0, 0.1)"}),
                        html.Td(f"{secondary_pct:.2f}%", style={"fontWeight": "bold", "backgroundColor": "rgba(255, 0, 0, 0.1)"}),
                        html.Td(f"{diff:.2f}%", style={"fontWeight": "bold", "backgroundColor": "rgba(255, 0, 0, 0.1)"})
                    ]))
                else:
                    rows.append(html.Tr([
                        html.Td(str(category)),
                        html.Td(f"{primary_pct:.2f}%"),
                        html.Td(f"{secondary_pct:.2f}%"),
                        html.Td(f"{diff:.2f}%")
                    ]))
            
            # Create the comparison table
            comparison_table = html.Table([
                html.Thead(html.Tr([
                    html.Th("Value"), 
                    html.Th("Primary Dataset (%)"), 
                    html.Th("Secondary Dataset (%)"),
                    html.Th("Difference (%)")
                ])),
                html.Tbody(rows)
            ], className="comparison-table", style={
                "width": "100%",
                "borderCollapse": "collapse",
                "marginTop": "10px"
            })
            
            # Generate a summary for the selected value if it exists in either dataset
            selected_value_summary = html.Div([
                html.P([
                    "Selected value ",
                    html.Strong(str(value)),
                    f" appears in {primary_freq:.2f}% of the primary dataset and {secondary_freq:.2f}% of the secondary dataset, ",
                    f"a difference of {primary_freq - secondary_freq:.2f} percentage points."
                ])
            ]) if value in combined_ranking else html.Div()
            
            # Add unique value count information for context
            unique_value_info = html.P([
                f"This column has {total_unique} unique values. ",
                "Showing the top 10 most frequent values across both datasets."
            ]) if total_unique > 10 else html.Div()
            
            return html.Div([
                html.H5("Comparison Summary", style={"textAlign": "center"}),
                data_type_info,
                unique_value_info,
                selected_value_summary,
                html.P(f"Frequency comparison of '{column_name}' between datasets:"),
                comparison_table
            ])
            
    except Exception as e:
        return html.Div(f"Error generating comparison: {str(e)}")

def summarize_dataframe(df):
    """
    Summarizes a pandas DataFrame by providing:
    - Column names and data types
    - Number of missing values
    - Number of unique values
    - Basic descriptive statistics for numerical and categorical columns
    """
    # Create a summary DataFrame with basic information
    summary = pd.DataFrame({
        "Data Type": df.dtypes.astype(str),  # Data types of each column
        "Missing Values": df.isnull().sum(),  # Count of missing values in each column
        "Unique Values": df.nunique(),  # Number of unique values in each column
    })

    # Add statistics for numerical columns
    numerical_summary = df.describe().T  # Transpose the descriptive statistics for readability
    numerical_summary = numerical_summary[["mean", "std", "min", "25%", "50%", "75%", "max"]]
    summary = summary.join(numerical_summary, how="left")  # Join numerical stats to the summary

    # Handle categorical columns separately
    categorical_columns = df.select_dtypes(include=["object", "category"])  # Select only categorical columns

    # Calculate the most frequent value (mode) for each categorical column
    top_values = categorical_columns.apply(
        lambda col: col.mode().iloc[0] if not col.mode().empty else None  # Handle empty mode
    )

    # Calculate the frequency of the most frequent value
    top_frequencies = categorical_columns.apply(
        lambda col: col.value_counts().iloc[0] if not col.value_counts().empty else None  # Handle empty value_counts
    )

    # Create a summary DataFrame for categorical columns
    categorical_summary = pd.DataFrame({
        "Top Value": top_values,  # Most frequent value for each column
        "Top Frequency": top_frequencies,  # Frequency of the most frequent value
    })

    # Merge the categorical summary with the overall summary
    summary = summary.join(categorical_summary, how="left")

    return summary

def get_dataset_statistics(df, dataset_name="Dataset", include_chart=False):
    """
    Generate compact statistics for a dataset and return an HTML component with the information.
    
    Args:
        df: The DataFrame to analyze
        dataset_name: Name to display for the dataset
        include_chart: Whether to include the missing values chart
        
    Returns:
        dash_html_components.Div: HTML component with dataset statistics
    """
    try:
        # Basic dataset statistics
        row_count = df.shape[0]
        col_count = df.shape[1]
        
        # Check for missing values
        missing_values = df.isnull().sum()
        missing_count = missing_values.sum()
        has_missing = missing_count > 0
        
        # Get intelligent data type counts if available
        type_counts = {"Binary": 0, "Continuous": 0, "Datetime": 0, "Categorical": 0}
        
        if hasattr(global_vars, 'column_types') and len(global_vars.column_types) > 0:
            for col in df.columns:
                if col in global_vars.column_types:
                    col_type = global_vars.column_types[col]
                    if col_type in type_counts:
                        type_counts[col_type] += 1
        
        # Create a compact stats table
        stats_rows = [
            html.Tr([html.Td("Rows"), html.Td(f"{row_count:,}")]),
            html.Tr([html.Td("Columns"), html.Td(f"{col_count}")]),
            html.Tr([html.Td("Missing Values"), html.Td(f"{missing_count:,} ({missing_count/row_count*100:.1f}%)" if has_missing else "None")])
        ]
        
        # Add intelligent data type info
        type_rows = [
            html.Tr([html.Td("Binary"), html.Td(f"{type_counts['Binary']}"), 
                     html.Td(style={"width": "30px", "backgroundColor": "#e6f7ff"})]),
            html.Tr([html.Td("Continuous"), html.Td(f"{type_counts['Continuous']}"), 
                     html.Td(style={"width": "30px", "backgroundColor": "#f0fff0"})]),
            html.Tr([html.Td("Datetime"), html.Td(f"{type_counts['Datetime']}"), 
                     html.Td(style={"width": "30px", "backgroundColor": "#fff0f5"})]),
            html.Tr([html.Td("Categorical"), html.Td(f"{type_counts['Categorical']}"), 
                     html.Td(style={"width": "30px", "backgroundColor": "#fffacd"})]),
        ]
        
        # Create the stats table
        stats_table = html.Table([
            html.Thead(html.Tr([html.Th("Statistic"), html.Th("Value")])),
            html.Tbody(stats_rows)
        ], className="dataset-stats-table")
        
        # Create the data types table
        types_table = html.Table([
            html.Thead(html.Tr([html.Th("Data Type"), html.Th("Count"), html.Th("")])),
            html.Tbody(type_rows)
        ], className="dataset-stats-table", style={"marginTop": "10px"})
        
        # Return combined statistics
        if include_chart:
            missing_chart = create_combined_missing_values_chart(df, None)
            if missing_chart:
                missing_values_section = html.Div([
                    html.H5("Missing Values Analysis", className="dataset-info-title",
                           style={"backgroundColor": "#f8f9fa", "padding": "8px", "borderRadius": "4px", "marginTop": "20px"}),
                    missing_chart
                ], className="twelve columns", style={"marginTop": "20px"})
            else:
                missing_values_section = html.Div([
                    html.H5("Missing Values Analysis", className="dataset-info-title",
                           style={"backgroundColor": "#f8f9fa", "padding": "8px", "borderRadius": "4px", "marginTop": "20px"}),
                    html.P("No missing values in the dataset.", style={"textAlign": "center", "fontStyle": "italic"})
                ], className="twelve columns", style={"marginTop": "20px"})
            
            return html.Div([
                html.H4(dataset_name, className="dataset-info-title"),
                stats_table,
                html.H6("Data Type Classification", style={"marginTop": "15px", "fontSize": "0.95rem"}),
                types_table,
                missing_values_section
            ])
        else:
            return html.Div([
                html.H4(dataset_name, className="dataset-info-title"),
                stats_table,
                html.H6("Data Type Classification", style={"marginTop": "15px", "fontSize": "0.95rem"}),
                types_table
            ])
            
    except Exception as e:
        print(f"Error generating statistics: {str(e)}")
        return html.Div([
            html.P(f"Error generating statistics: {str(e)}")
        ])

def create_combined_missing_values_chart(df1, df2):
    """Create a combined chart showing missing values for both datasets."""
    # Get missing values for both datasets
    df1_missing = df1.isnull().sum()
    df2_missing = df2.isnull().sum()
    
    # Check if either dataset has missing values
    has_missing = any(df1_missing > 0) or any(df2_missing > 0)
    
    if not has_missing:
        return None
    
    # Create dataframe for plotting
    missing_data = []
    
    # Add primary dataset missing values
    for col in df1.columns:
        if df1_missing[col] > 0:
            missing_data.append({
                'Column': col,
                'Missing': df1_missing[col],
                'Dataset': 'Primary Dataset',
                'Percentage': df1_missing[col] / len(df1) * 100
            })
    
    # Add secondary dataset missing values
    for col in df2.columns:
        if df2_missing[col] > 0:
            missing_data.append({
                'Column': col,
                'Missing': df2_missing[col],
                'Dataset': 'Secondary Dataset',
                'Percentage': df2_missing[col] / len(df2) * 100
            })
    
    if not missing_data:
        return None
    
    # Create dataframe
    missing_df = pd.DataFrame(missing_data)
    
    # Create grouped bar chart
    fig = px.bar(
        missing_df, 
        x='Column', 
        y='Missing',
        color='Dataset',
        barmode='group',
        hover_data=['Percentage'],
        color_discrete_sequence=[px.colors.qualitative.Plotly[0], px.colors.qualitative.Plotly[1]]
    )
    
    fig.update_layout(
        xaxis_title="Attributes",
        yaxis_title="Number of Missing Values",
        legend_title="Dataset",
        height=350,
        margin=dict(l=30, r=30, t=30, b=50)
    )
    
    # Add percentage text labels
    for _, row in missing_df.iterrows():
        fig.add_annotation(
            x=row['Column'],
            y=row['Missing'],
            text=f"{row['Percentage']:.1f}%",
            showarrow=False,
            yshift=10,
            font=dict(size=9)
        )
    
    return dcc.Graph(figure=fig, className="missing-values-chart")



def update_column_type_ui(classification, data_type, id_dict, current_style):
    """
    Update column classification and data type UI, handling dropdown value changes.
    This callback only handles pattern-matching outputs without triggering global state changes.
    """
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update, dash.no_update
    
    # Get trigger component information
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    try:
        trigger_dict = json.loads(triggered_id)
        column = trigger_dict.get('column', '')
        trigger_type = trigger_dict.get('type', '')
    except:
        return dash.no_update, dash.no_update
    
    # Set background color based on classification
    new_style = current_style.copy() if current_style else {}
    bg_color = "#ffffff"  # Default white
    if classification == "Binary":
        bg_color = "#e6f7ff"  # Light blue
    elif classification == "Continuous":
        bg_color = "#f0fff0"  # Light green
    elif classification == "Datetime":
        bg_color = "#fff0f5"  # Light pink
    elif classification == "Categorical":
        bg_color = "#fffacd"  # Light yellow
    
    new_style["backgroundColor"] = bg_color
    
    # Check data validity
    valid_type = data_type
    valid = True
    
    # Ensure primary dataset exists
    if not hasattr(global_vars, 'df') or column not in global_vars.df.columns:
        # Store error message for global feedback
        global_vars.last_validation_result = {
            'column': column, 
            'valid': False, 
            'message': "Primary dataset unavailable or column does not exist",
            'time': str(datetime.datetime.now())
        }
        # Trigger global feedback callback
        if not hasattr(global_vars, 'validation_trigger'):
            global_vars.validation_trigger = 0
        global_vars.validation_trigger += 1
        return new_style, data_type
    
    # Check secondary dataset if it exists
    has_secondary = hasattr(global_vars, 'secondary_df') and column in global_vars.secondary_df.columns
    
    # Get data from both datasets (if available)
    primary_values = global_vars.df[column].copy()
    secondary_values = None
    if has_secondary:
        secondary_values = global_vars.secondary_df[column].copy()
    
    # Validate data type compatibility with classification
    original_dtype = str(global_vars.df[column].dtype)
    error_msg = ""
    
    try:
        # Validate based on classification
        if classification == "Binary":
            if data_type not in ['int64', 'float64', 'bool']:
                valid = False
                error_msg = f"Binary classification requires numeric or boolean data type, not {data_type}"
            else:
                # Check primary dataset
                try:
                    unique_values_primary = pd.Series(primary_values).astype(data_type).unique()
                    non_na_values_primary = [x for x in unique_values_primary if pd.notna(x)]
                    if len(non_na_values_primary) > 2:
                        valid = False
                        error_msg = f"Column '{column}' in primary dataset has more than 2 unique values after conversion to {data_type}"
                except Exception as e:
                    valid = False
                    error_msg = f"Cannot convert column '{column}' in primary dataset to {data_type}: {str(e)}"
                
                # Check secondary dataset
                if valid and has_secondary:
                    try:
                        unique_values_secondary = pd.Series(secondary_values).astype(data_type).unique()
                        non_na_values_secondary = [x for x in unique_values_secondary if pd.notna(x)]
                        if len(non_na_values_secondary) > 2:
                            valid = False
                            error_msg = f"Column '{column}' in secondary dataset has more than 2 unique values after conversion to {data_type}"
                    except Exception as e:
                        valid = False
                        error_msg = f"Cannot convert column '{column}' in secondary dataset to {data_type}: {str(e)}"
        
        elif classification == "Continuous":
            if data_type not in ['int64', 'float64']:
                valid = False
                error_msg = f"Continuous classification requires numeric data type, not {data_type}"
            else:
                # Check if trying to convert from float to int when decimal values exist
                if original_dtype == 'float64' and data_type == 'int64':
                    # Check primary dataset for true decimal values
                    if not all(x.is_integer() for x in primary_values.dropna()):
                        valid = False
                        error_msg = f"Cannot convert column '{column}' from float to int: column contains decimal values"
                    
                    # Also check secondary dataset if it exists
                    if valid and has_secondary:
                        if not all(x.is_integer() for x in secondary_values.dropna()):
                            valid = False
                            error_msg = f"Cannot convert column '{column}' from float to int: secondary dataset contains decimal values"
                
                # General conversion check for primary dataset
                if valid:
                    try:
                        pd.Series(primary_values).astype(data_type)
                    except Exception as e:
                        valid = False
                        error_msg = f"Cannot convert column '{column}' in primary dataset to {data_type}: {str(e)}"
                
                # General conversion check for secondary dataset
                if valid and has_secondary:
                    try:
                        pd.Series(secondary_values).astype(data_type)
                    except Exception as e:
                        valid = False
                        error_msg = f"Cannot convert column '{column}' in secondary dataset to {data_type}: {str(e)}"
        
        elif classification == "Datetime":
            if data_type != 'datetime64[ns]':
                valid = False
                error_msg = f"Datetime classification requires datetime64[ns] data type, not {data_type}"
            else:
                # Check primary dataset
                try:
                    pd.to_datetime(primary_values)
                except Exception as e:
                    valid = False
                    error_msg = f"Cannot convert column '{column}' in primary dataset to datetime: {str(e)}"
                
                # Check secondary dataset
                if valid and has_secondary:
                    try:
                        pd.to_datetime(secondary_values)
                    except Exception as e:
                        valid = False
                        error_msg = f"Cannot convert column '{column}' in secondary dataset to datetime: {str(e)}"
        
        elif classification == "Categorical":
            if data_type not in ['object', 'category']:
                # Can be more flexible but add warning
                if data_type in ['int64', 'float64']:
                    error_msg = f"Warning: Using numeric data as categorical may not be ideal. Consider using 'object' type."
                    # This is a warning, not an error
                else:
                    # Check primary dataset
                    try:
                        pd.Series(primary_values).astype(str)
                    except Exception as e:
                        valid = False
                        error_msg = f"Cannot use column '{column}' in primary dataset as categorical data: {str(e)}"
                    
                    # Check secondary dataset
                    if valid and has_secondary:
                        try:
                            pd.Series(secondary_values).astype(str)
                        except Exception as e:
                            valid = False
                            error_msg = f"Cannot use column '{column}' in secondary dataset as categorical data: {str(e)}"
    
    except Exception as e:
        valid = False
        error_msg = f"Error validating data type: {str(e)}"
    
    # If valid, apply changes
    if valid:
        try:
            # Update global column type dictionary
            if hasattr(global_vars, 'column_types'):
                global_vars.column_types[column] = classification
                
            # Trigger store update to sync with UI components
            from dash import callback_context
            if callback_context:
                # Force update of the column-types-store to propagate changes
                pass
            
            # Try to convert data type in primary dataset
            current_type = str(global_vars.df[column].dtype)
            if data_type != current_type:
                global_vars.df[column] = global_vars.df[column].astype(data_type)
            
            # If secondary dataset exists, update it too
            if has_secondary:
                secondary_type = str(global_vars.secondary_df[column].dtype)
                if data_type != secondary_type:
                    global_vars.secondary_df[column] = global_vars.secondary_df[column].astype(data_type)
            
            # Store success message
            global_vars.last_validation_result = {
                'column': column, 
                'valid': True, 
                'message': f"Successfully updated {column} to {classification} classification with {data_type} type",
                'time': str(datetime.datetime.now())
            }
            
            # If there's a warning, add it to the message
            if error_msg.startswith("Warning"):
                global_vars.last_validation_result['warning'] = error_msg
            
            # === OPTIMIZATION: Clear metrics cache only if type actually changed ===
            # Check if this is a real change vs just a UI update
            if hasattr(global_vars, 'column_types') and global_vars.column_types:
                old_type = global_vars.column_types.get(column)
                new_type = classification
                
                if old_type != new_type:
                    print(f"[METRICS CACHE] Data type actually changed for {column}: {old_type} -> {new_type}")
                    global_vars.clear_metrics_cache(f"Data type changed for column '{column}': {old_type} -> {new_type}")
                else:
                    print(f"[METRICS CACHE] Data type unchanged for {column}: {classification}")
            else:
                print(f"[METRICS CACHE] No previous column types, not clearing cache for {column}")
            
        except Exception as e:
            valid = False
            error_msg = f"Error applying changes: {str(e)}"
            global_vars.last_validation_result = {
                'column': column, 
                'valid': False, 
                'message': error_msg,
                'time': str(datetime.datetime.now())
            }
            
            # If triggered by data type dropdown, revert to original type
            if trigger_type == 'datatype-dropdown':
                valid_type = original_dtype
    else:
        # Store validation failure message
        global_vars.last_validation_result = {
            'column': column, 
            'valid': False, 
            'message': error_msg,
            'time': str(datetime.datetime.now())
        }
        
        # If triggered by data type dropdown, revert to original type
        if trigger_type == 'datatype-dropdown':
            valid_type = original_dtype
    
    # Trigger global feedback callback
    if not hasattr(global_vars, 'validation_trigger'):
        global_vars.validation_trigger = 0
    global_vars.validation_trigger += 1
    
    return new_style, valid_type


@app.callback(
    Output({"type": 'type-validation-feedback', "index": ALL}, 'children', allow_duplicate=True),
    Output({"type": 'hidden-validation-trigger', "index": ALL}, 'children', allow_duplicate=True),
    Output('column-types-store', 'data', allow_duplicate=True),
    Input('interval-component', 'n_intervals'),
    State({"type": 'type-validation-feedback', "index": ALL}, 'children'),
    State('column-types-store', 'data'),
    prevent_initial_call=True
)
def show_validation_feedback(n_intervals, existing_feedback_list, current_store_data):
    """
    Display validation feedback and trigger global store update.
    This callback is separate from the pattern-matching callback to avoid MATCH wildcard issues.
    """
    # Check if validation results exist
    count = len(existing_feedback_list) if isinstance(existing_feedback_list, list) else 0
    if count == 0:
        return [], [], dash.no_update
    if not hasattr(global_vars, 'last_validation_result'):
        return [dash.no_update] * count, [dash.no_update] * count, dash.no_update
    
    # Get validation results
    result = global_vars.last_validation_result
    column = result.get('column', '')
    valid = result.get('valid', False)
    message = result.get('message', '')
    warning = result.get('warning', '')
    
    # Generate feedback information
    if valid:
        feedback_elements = [html.P(message, style={"color": "green"})]
        if warning:
            feedback_elements.append(html.P(warning, style={"color": "orange"}))
        feedback = html.Div(feedback_elements)
    else:
        feedback = html.Div([
            html.P(f"Invalid selection: {message}", style={"color": "red", "fontWeight": "bold"})
        ])
    
    # Generate hidden trigger for updating global store
    hidden_trigger = f"Updated at {datetime.datetime.now()}"
    
    # Update column types store to propagate changes
    updated_store_data = getattr(global_vars, 'column_types', current_store_data or {})
    
    return [feedback] * count, [hidden_trigger] * count, updated_store_data


@app.callback(
    Output('column-types-store', 'data', allow_duplicate=True),
    Input({"type": 'hidden-validation-trigger', "index": ALL}, 'children'),
    State('column-types-store', 'data'),
    prevent_initial_call=True
)
def update_column_types_store(_triggers, current_data):
    """Update global column type store to ensure UI and backend data remain in sync"""
    if hasattr(global_vars, 'column_types'):
        return global_vars.column_types
    return current_data



# Add interval component for periodic validation check
if not any(component.id == 'interval-component' for component in app.layout.children if hasattr(component, 'id')):
    app.layout.children.append(
        dcc.Interval(
            id='interval-component',
            interval=500,  # milliseconds
            n_intervals=0
        )
    )


# ================================================================================
# UNIFIED DATA STORE - PHASE 1.2
# Single source of truth for all data state management
# ================================================================================

def generate_dataset_hash() -> str:
    """
    Generate a hash for dataset change detection.
    
    This function creates a unique hash based on the current state of both
    datasets to detect when data has changed and trigger appropriate updates.
    
    Returns:
        str: Unique hash representing current dataset state
    """
    try:
        hash_components = []
        
        # Add primary dataset hash
        if hasattr(global_vars, 'df') and global_vars.df is not None:
            # Use shape, columns, and a sample of data for hash
            primary_info = {
                'shape': global_vars.df.shape,
                'columns': list(global_vars.df.columns),
                'dtypes': {col: str(dtype) for col, dtype in global_vars.df.dtypes.items()},
                # Include a sample of the first few rows for content change detection
                'sample_hash': hashlib.md5(
                    str(global_vars.df.head(3).to_dict()).encode()
                ).hexdigest()
            }
            hash_components.append(('primary', str(primary_info)))
        
        # Add secondary dataset hash
        if hasattr(global_vars, 'secondary_df') and global_vars.secondary_df is not None:
            secondary_info = {
                'shape': global_vars.secondary_df.shape,
                'columns': list(global_vars.secondary_df.columns),
                'dtypes': {col: str(dtype) for col, dtype in global_vars.secondary_df.dtypes.items()},
                'sample_hash': hashlib.md5(
                    str(global_vars.secondary_df.head(3).to_dict()).encode()
                ).hexdigest()
            }
            hash_components.append(('secondary', str(secondary_info)))
        
        # Add column types hash
        if hasattr(global_vars, 'column_types') and global_vars.column_types:
            column_types_str = str(sorted(global_vars.column_types.items()))
            hash_components.append(('column_types', column_types_str))
        
        # Create final hash
        combined_string = '|'.join([f"{key}:{value}" for key, value in hash_components])
        final_hash = hashlib.md5(combined_string.encode()).hexdigest()
        
        return final_hash
        
    except Exception as e:
        print(f"[GLOBAL DATA STATE] Error generating dataset hash: {str(e)}")
        return f"error_{int(time.time())}"


@app.callback(
    [Output('global-data-state', 'data')],
    [Input('column-type-change-trigger', 'data'),
     Input('dataset-upload-trigger', 'data')],
    prevent_initial_call=True
)
def update_global_data_state(type_changes, upload_changes):
    """
    Master callback that updates all global state.
    
    This is the single source of truth for all data state changes in the application.
    It consolidates information from multiple sources and provides a unified view
    of the current data state.
    
    Args:
        type_changes: Trigger data from column type changes
        upload_changes: Trigger data from dataset uploads
        
    Returns:
        dict: Unified global data state
    """
    try:
        # Build the unified global state
        global_state = {
            'timestamp': time.time(),
            'last_update': datetime.datetime.now().isoformat(),
            'datasets_available': {
                'primary': hasattr(global_vars, 'df') and global_vars.df is not None,
                'secondary': hasattr(global_vars, 'secondary_df') and global_vars.secondary_df is not None
            }
        }
        
        # Add primary dataset information
        if hasattr(global_vars, 'df') and global_vars.df is not None:
            global_state['primary_dataset'] = {
                'shape': global_vars.df.shape,
                'columns': list(global_vars.df.columns),
                'dtypes': {col: str(dtype) for col, dtype in global_vars.df.dtypes.items()},
                'filename': getattr(global_vars, 'file_name', 'Unknown')
            }
        else:
            global_state['primary_dataset'] = None
        
        # Add secondary dataset information
        if hasattr(global_vars, 'secondary_df') and global_vars.secondary_df is not None:
            global_state['secondary_dataset'] = {
                'shape': global_vars.secondary_df.shape,
                'columns': list(global_vars.secondary_df.columns),
                'dtypes': {col: str(dtype) for col, dtype in global_vars.secondary_df.dtypes.items()},
                'filename': getattr(global_vars, 'secondary_file_name', 'Unknown')
            }
        else:
            global_state['secondary_dataset'] = None
        
        # Add column types (intelligent classification)
        if hasattr(global_vars, 'column_types') and global_vars.column_types:
            global_state['column_types'] = global_vars.column_types.copy()
        else:
            global_state['column_types'] = {}
        
        # Add dataset hash for change detection
        global_state['datasets_hash'] = generate_dataset_hash()
        
        # Add common columns information
        if global_state['primary_dataset'] and global_state['secondary_dataset']:
            primary_cols = set(global_state['primary_dataset']['columns'])
            secondary_cols = set(global_state['secondary_dataset']['columns'])
            global_state['common_columns'] = list(primary_cols.intersection(secondary_cols))
            global_state['primary_only_columns'] = list(primary_cols - secondary_cols)
            global_state['secondary_only_columns'] = list(secondary_cols - primary_cols)
        else:
            global_state['common_columns'] = []
            global_state['primary_only_columns'] = []
            global_state['secondary_only_columns'] = []
        
        # Add metrics cache status
        if hasattr(global_vars, 'is_cache_valid'):
            cache_valid, cache_reason = global_vars.is_cache_valid()
            global_state['metrics_cache'] = {
                'is_valid': cache_valid,
                'reason': cache_reason,
                'last_cleared': getattr(global_vars, 'last_cache_clear_time', None)
            }
        else:
            global_state['metrics_cache'] = {
                'is_valid': False,
                'reason': 'Cache validation not available',
                'last_cleared': None
            }
        
        # Add trigger information for debugging
        ctx = dash.callback_context
        if ctx.triggered:
            triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
            global_state['last_trigger'] = {
                'component': triggered_id,
                'timestamp': time.time(),
                'type_changes': type_changes,
                'upload_changes': upload_changes
            }
        
        print(f"[GLOBAL DATA STATE] Updated global state. Hash: {global_state['datasets_hash']}")
        
        # Phase 3: Global state no longer directly updates table-overview
        # The table-overview-controller handles all table updates centrally
        return [global_state]
        
    except Exception as e:
        error_msg = f"Error updating global data state: {str(e)}"
        print(f"[GLOBAL DATA STATE] {error_msg}")
        
        # Return minimal valid state on error
        error_state = {
            'timestamp': time.time(),
            'last_update': datetime.datetime.now().isoformat(),
            'error': error_msg,
            'datasets_available': {'primary': False, 'secondary': False},
            'datasets_hash': f"error_{int(time.time())}"
        }
        
        # Phase 3: Only return global state
        return error_state


# Helper function to trigger global data state updates
def trigger_global_data_update(reason: str = "Manual trigger"):
    """
    Helper function to trigger a global data state update.
    
    This function can be called from other parts of the application
    to signal that the global data state should be refreshed.
    
    Args:
        reason: Description of why the update was triggered
    """
    try:
        # We'll update this when we implement the specific triggers
        print(f"[GLOBAL DATA STATE] Trigger requested: {reason}")
        
        # For now, this is a placeholder. In the full implementation,
        # this would update the appropriate trigger store to cascade
        # the update through the callback system.
        
    except Exception as e:
        print(f"[GLOBAL DATA STATE] Error triggering update: {str(e)}")


# ================================================================================
# BACKWARDS COMPATIBILITY - Keep existing store callbacks working
# ================================================================================

@app.callback(
    Output('column-types-store', 'data', allow_duplicate=True),
    [Input('global-data-state', 'data')],
    prevent_initial_call=True
)
def sync_legacy_column_types_store(global_state):
    """
    Keep the legacy column-types-store in sync with the new global state.
    
    This ensures backwards compatibility with existing components that
    still rely on the old column-types-store.
    
    Args:
        global_state: The unified global data state
        
    Returns:
        dict: Column types data for legacy store
    """
    try:
        if not global_state or 'column_types' not in global_state:
            return {}
        
        return global_state['column_types']
        
    except Exception as e:
        print(f"[GLOBAL DATA STATE] Error syncing legacy column types store: {str(e)}")
        return {}


# ================================================================================
# DATASET CHANGE TRIGGERS - Update triggers when datasets change
# ================================================================================

@app.callback(
    Output('dataset-upload-trigger', 'data', allow_duplicate=True),
    [Input('table-overview', 'data'),
     Input('table-overview', 'columns')],
    prevent_initial_call=True
)
def trigger_dataset_upload_update(table_data, table_columns):
    """
    Trigger global state update when table data changes (indicating dataset upload).
    
    Args:
        table_data: Table data (triggers when datasets are uploaded)
        table_columns: Table columns (triggers when datasets are uploaded)
        
    Returns:
        dict: Trigger data with timestamp
    """
    try:
        trigger_data = {
            'timestamp': time.time(),
            'reason': 'Dataset upload detected',
            'has_data': table_data is not None and len(table_data) > 0,
            'column_count': len(table_columns) if table_columns else 0
        }
        
        print(f"[GLOBAL DATA STATE] Dataset upload trigger: {trigger_data}")
        return trigger_data
        
    except Exception as e:
        print(f"[GLOBAL DATA STATE] Error in dataset upload trigger: {str(e)}")
        return {'timestamp': time.time(), 'error': str(e)}


# ================================================================================
# COLUMN TYPE CHANGE BRIDGE - Connect Phase 1.1 to Phase 1.2
# ================================================================================

@app.callback(
    Output('column-type-change-trigger', 'data', allow_duplicate=True),
    [Input('interval-component', 'n_intervals')],
    prevent_initial_call=True
)
def bridge_column_type_changes_to_global_state(n_intervals):
    """
    Bridge callback to connect ColumnTypeManager changes to global data state.
    
    This callback runs periodically to check for column type changes from
    ColumnTypeManager and triggers the global data state update chain.
    
    This solves the critical integration gap between Phase 1.1 and Phase 1.2.
    
    Args:
        n_intervals: Interval trigger (used for periodic checking)
        
    Returns:
        dict: Trigger data for column type changes
    """
    try:
        # Check for latest column type changes from ColumnTypeManager
        latest_change = ColumnTypeManager.get_latest_change_for_trigger()
        
        if not latest_change or not latest_change.get('change_id'):
            # No new changes, don't trigger
            raise PreventUpdate
        
        # Create trigger data for global state update
        trigger_data = {
            'timestamp': time.time(),
            'source': 'ColumnTypeManager',
            'change_id': latest_change['change_id'],
            'column': latest_change['column'],
            'classification': latest_change['classification'],
            'data_type': latest_change['data_type'],
            'trigger_reason': latest_change['trigger_reason']
        }
        
        print(f"[BRIDGE] Column type change detected, triggering global state update: {trigger_data}")
        return trigger_data
        
    except PreventUpdate:
        raise
    except Exception as e:
        print(f"[BRIDGE] Error in column type change bridge: {str(e)}")
        # Return error trigger to still update global state
        return {
            'timestamp': time.time(),
            'source': 'ColumnTypeManager',
            'error': str(e),
            'trigger_reason': 'Bridge error occurred'
        }
