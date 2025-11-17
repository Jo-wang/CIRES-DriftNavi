"""
Retrain workflow components for Adapt stage.

This module provides interactive components for the retrain data preparation workflow,
which merges Primary and Secondary datasets completely, then resamples the merged dataset
for balanced representation across selected attributes.
"""

import dash
from dash import html, dcc, callback, Input, Output, State, MATCH, ALL
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import pandas as pd

from UI.app import app
from UI.functions.global_vars import global_vars


def create_retrain_config_bubble(unique_id: str) -> html.Div:
    """
    Create configuration bubble for retrain workflow.
    
    Args:
        unique_id: Unique identifier for this bubble instance
        
    Returns:
        html.Div containing the configuration interface
    """
    return html.Div([
        dbc.Alert([
            html.H5([
                html.I(className="fas fa-sync-alt me-2"),
                "Retrain Data Preparation"
            ], className="alert-heading"),
            html.P([
                "Complete model retraining combines Primary and Secondary datasets equally (100% each), ",
                "then balances them across selected attributes and target variable for fair representation. ",
                "This approach treats old and new data with equal importance."
            ], className="mb-3"),
        ], color="info"),
        
        # Step 1: Merge datasets (informational only)
        dbc.Card([
            dbc.CardHeader(html.Strong("Step 1: Merge Primary & Secondary Datasets")),
            dbc.CardBody([
                html.Div([
                    html.P([
                        html.I(className="fas fa-check-circle text-success me-2"),
                        "All primary data (100%)"
                    ], className="mb-2"),
                    html.P([
                        html.I(className="fas fa-check-circle text-success me-2"),
                        "All secondary data (100%)"
                    ], className="mb-2"),
                    html.P([
                        html.I(className="fas fa-arrow-right text-primary me-2"),
                        "Combined dataset will be resampled together"
                    ], className="mb-0 text-muted small")
                ])
            ])
        ], className="mb-3"),
        
        # Step 2: Resampling Configuration
        dbc.Card([
            dbc.CardHeader(html.Strong("Step 2: Select Attributes to Balance")),
            dbc.CardBody([
                html.Label("Select Attributes to Balance", className="fw-bold"),
                dcc.Dropdown(
                    id={"type": "retrain-attrs-dropdown", "index": unique_id},
                    options=[],  # Populated by callback
                    multi=True,
                    placeholder="Select one or more attributes (e.g., gender, race, age_group)"
                ),
                html.Small("These attributes will be balanced jointly with the target variable. Continuous variables are automatically binned.", 
                          className="text-muted d-block mb-3"),
                
                html.Label("Resampling Strategy", className="fw-bold mt-2"),
                dcc.RadioItems(
                    id={"type": "retrain-strategy-radio", "index": unique_id},
                    options=[
                        {"label": " Oversample (duplicate minority classes)", "value": "oversample"},
                        {"label": " Undersample (reduce majority classes)", "value": "undersample"},
                        {"label": " SMOTE (synthetic samples)", "value": "smote"}
                    ],
                    value="oversample",
                    className="mb-2"
                ),
                html.Small("Note: SMOTE requires categorical target and numerical features, otherwise falls back to oversampling.", 
                          className="text-muted")
            ])
        ], className="mb-3"),
        
        # Confirm Button
        dbc.Button(
            "Confirm & Generate Dataset",
            id={"type": "retrain-confirm-btn", "index": unique_id},
            color="danger",
            size="lg",
            className="w-100"
        )
    ])


def create_retrain_success_bubble(resampled_size: int, primary_size: int, secondary_size: int, merged_size: int, unique_id: str) -> html.Div:
    """
    Create success bubble after retrain processing completes.
    
    Args:
        resampled_size: Size of resampled merged dataset
        primary_size: Original size of primary dataset
        secondary_size: Original size of secondary dataset
        merged_size: Size of merged dataset (before resampling)
        unique_id: Unique identifier for this bubble instance
        
    Returns:
        html.Div containing success message and export button
    """
    return html.Div([
        dbc.Alert([
            html.H5([
                html.I(className="fas fa-check-circle me-2"),
                "Dataset for Retraining the Model is Ready!"
            ], className="alert-heading text-success"),
            html.Hr(),
            html.H6("Dataset Statistics:", className="mb-2"),
            html.Ul([
                html.Li(f"Original Primary: {primary_size:,} samples"),
                html.Li(f"Original Secondary: {secondary_size:,} samples"),
                html.Li(f"Merged Total: {merged_size:,} samples"),
                html.Li(html.Strong(f"After Resampling: {resampled_size:,} samples"))
            ], className="mb-3"),
            html.P("The resampled dataset is ready for model training. You can export it as a CSV file below.", 
                   className="mb-3"),
            dbc.Button([
                html.I(className="fas fa-download me-2"),
                "Export as CSV"
            ],
            id={"type": "retrain-export-btn", "index": unique_id},
            color="success",
            size="lg",
            className="w-100")
        ], color="light", className="border border-success")
    ])


# Callback 1: Populate attribute options for dropdown
@callback(
    Output({"type": "retrain-attrs-dropdown", "index": MATCH}, "options"),
    Input({"type": "retrain-attrs-dropdown", "index": MATCH}, "id"),
    prevent_initial_call=False
)
def populate_retrain_attrs_dropdown(_id):
    """
    Populate dropdown with merged dataset columns (excluding target).
    
    Since we're merging primary and secondary datasets, we get columns from primary
    (assuming both datasets have the same schema).
    """
    try:
        # Get primary dataset with proper None checking to avoid DataFrame ambiguity
        primary_df = getattr(global_vars, "df", None)
        if primary_df is None:
            print(f"[RETRAIN] No primary dataset available for dropdown population")
            return []
        
        # Get all columns from primary (schema should match secondary)
        all_cols = list(primary_df.columns)
        print(f"[RETRAIN] Populating dropdown with {len(all_cols)} columns from dataset")
        
        # Exclude target attribute if set
        target_col = getattr(global_vars, 'target_attribute', None)
        if target_col and target_col in all_cols:
            all_cols.remove(target_col)
        
        # Return as options
        return [{"label": col, "value": col} for col in all_cols]
        
    except Exception as e:
        print(f"[RETRAIN] Error populating attributes: {e}")
        return []


# Callback 2: Process retrain workflow - Single callback approach
# Uses ALL pattern to avoid pattern-matching conflicts
# Direct approach: ALL inputs → process → single output (query-area)
@callback(
    Output("query-area", "children", allow_duplicate=True),
    Input({"type": "retrain-confirm-btn", "index": ALL}, "n_clicks"),
    [State({"type": "retrain-attrs-dropdown", "index": ALL}, "value"),
     State({"type": "retrain-strategy-radio", "index": ALL}, "value"),
     State({"type": "retrain-confirm-btn", "index": ALL}, "id"),
     State("query-area", "children")],
    prevent_initial_call=True
)
def process_retrain_workflow(n_clicks_list, selected_attrs_list, strategies, button_ids, query_records):
    """
    Process retrain workflow using ALL pattern.
    Uses ctx.triggered to determine which button was clicked and processes accordingly.
    No button state update - only updates chat area.
    """
    import dash
    import time
    from drift.sampler import MultiAttributeSampler
    from UI.callback.chat_callbacks import create_timestamped_message, sort_chat_messages
    
    ctx = dash.callback_context
    if not ctx.triggered or not n_clicks_list:
        raise PreventUpdate
    
    # Check if any button was actually clicked
    if not any(n_clicks_list):
        raise PreventUpdate
    
    # Determine which button was clicked using ctx.triggered
    triggered_prop = ctx.triggered[0]["prop_id"]
    
    # Parse the triggered button's index
    try:
        import json
        # Extract the dict from the prop_id (format: '{"index":"xxx","type":"yyy"}.n_clicks')
        button_id_str = triggered_prop.split('.')[0]
        button_dict = json.loads(button_id_str)
        triggered_index_value = button_dict.get("index")
        
        # Find which position in the list this corresponds to
        clicked_idx = None
        for idx, btn_id in enumerate(button_ids):
            if btn_id.get("index") == triggered_index_value:
                clicked_idx = idx
                break
        
        if clicked_idx is None:
            print(f"[RETRAIN] Could not find clicked button index")
            raise PreventUpdate
        
        # Extract parameters for this specific button
        unique_id = triggered_index_value
        selected_attrs = selected_attrs_list[clicked_idx]
        strategy = strategies[clicked_idx]
        
        print(f"[RETRAIN_WORKFLOW] Button {unique_id} clicked (position {clicked_idx})")
        print(f"[RETRAIN_WORKFLOW] Parameters: attrs={selected_attrs}, strategy={strategy}")
        
    except Exception as e:
        print(f"[RETRAIN] Error parsing triggered button: {e}")
        raise PreventUpdate
    
    # Initialize query_records if None
    if query_records is None:
        query_records = []
    
    try:
        print(f"[RETRAIN_WORKFLOW] Starting retrain data generation...")
        print(f"[RETRAIN_WORKFLOW] Selected attributes: {selected_attrs}")
        print(f"[RETRAIN_WORKFLOW] Strategy: {strategy}")
        
        # Validate inputs with proper None checking to avoid DataFrame ambiguity
        if not hasattr(global_vars, 'df') or global_vars.df is None:
            raise ValueError("Primary dataset not loaded")
        
        # Get secondary dataset with proper None checking to avoid DataFrame ambiguity
        secondary_df = getattr(global_vars, "secondary_df", None)
        if secondary_df is None:
            secondary_df = getattr(global_vars, "df_secondary", None)
        
        if secondary_df is None:
            raise ValueError("Secondary dataset not loaded")
        
        if not hasattr(global_vars, 'target_attribute') or not global_vars.target_attribute:
            raise ValueError("Target attribute not set")
        
        # Store original sizes for reporting
        primary_size = len(global_vars.df)
        secondary_size = len(secondary_df)
        
        # Step 1: Merge Primary and Secondary Datasets
        print(f"[RETRAIN_WORKFLOW] Step 1: Merging primary and secondary datasets...")
        merged_df = pd.concat([global_vars.df, secondary_df], axis=0).reset_index(drop=True)
        global_vars.retrain_merged_df = merged_df
        print(f"[RETRAIN_WORKFLOW] Merged: primary={primary_size}, secondary={secondary_size}, total={len(merged_df)}")
        
        # Step 2: Resample Merged Dataset
        print(f"[RETRAIN_WORKFLOW] Step 2: Resampling merged dataset...")
        sampler = MultiAttributeSampler(
            df=merged_df,
            target_col=global_vars.target_attribute,
            protected_attrs=selected_attrs or []
        )
        
        # Execute selected strategy
        if strategy == "oversample":
            resampled_df = sampler.oversample_multiattr(verbose=True, random_state=42)
        elif strategy == "undersample":
            resampled_df = sampler.undersample_multiattr(verbose=True, random_state=42)
        else:  # smote
            resampled_df = sampler.smote_multiattr(verbose=True, random_state=42)
        
        global_vars.retrain_resampled_df = resampled_df
        print(f"[RETRAIN_WORKFLOW] Resampled: {len(resampled_df)} samples")
        
        # Step 3: Create success bubble
        success_bubble = create_retrain_success_bubble(
            resampled_size=len(resampled_df),
            primary_size=primary_size,
            secondary_size=secondary_size,
            merged_size=len(merged_df),
            unique_id=unique_id
        )
        
        success_message = create_timestamped_message(success_bubble, "llm-msg")
        
        # Add original_type for proper sorting
        if hasattr(success_message, 'id') and isinstance(success_message.id, dict):
            success_message.id.update({
                "original_type": "chat-adapt-bubble",
                "index": f"retrain-success-{int(time.time() * 1000)}"
            })
        
        query_records.append(success_message)
        sorted_records = sort_chat_messages(query_records)
        
        print(f"[RETRAIN_WORKFLOW] Workflow completed successfully!")
        
        # Return updated chat records (single output - no button state update)
        return sorted_records
        
    except Exception as e:
        print(f"[RETRAIN_WORKFLOW] Error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Create error bubble
        error_bubble = dbc.Alert([
            html.H5([html.I(className="fas fa-exclamation-triangle me-2"), "Error"], className="alert-heading"),
            html.P(f"Failed to generate retrain dataset: {str(e)}"),
            html.Hr(),
            html.P("Please check that both datasets are loaded and target attribute is set.", className="mb-0")
        ], color="danger")
        
        error_message = create_timestamped_message(error_bubble, "llm-msg")
        if hasattr(error_message, 'id') and isinstance(error_message.id, dict):
            error_message.id.update({
                "original_type": "chat-adapt-bubble",
                "index": f"retrain-error-{int(time.time() * 1000)}"
            })
        
        query_records.append(error_message)
        sorted_records = sort_chat_messages(query_records)
        
        # Return updated chat records with error message (single output - no button state update)
        return sorted_records


# Callback 3: Export CSV
# No button state updates, only triggers download
@callback(
    Output("retrain-download", "data"),
    Input({"type": "retrain-export-btn", "index": ALL}, "n_clicks"),
    prevent_initial_call=True
)
def export_retrain_csv(n_clicks_list):
    """Export the resampled retrain dataset as CSV."""
    # Check if any button was clicked
    if not n_clicks_list or not any(n_clicks_list):
        raise PreventUpdate
    
    try:
        # Verify resampled dataset exists
        if not hasattr(global_vars, 'retrain_resampled_df') or global_vars.retrain_resampled_df is None:
            print("[RETRAIN] No resampled dataset available for export")
            raise PreventUpdate
        
        # Convert to CSV
        csv_string = global_vars.retrain_resampled_df.to_csv(index=False)
        
        print(f"[RETRAIN] Exporting resampled dataset: {len(global_vars.retrain_resampled_df)} rows")
        
        # Return download dict
        return dict(
            content=csv_string,
            filename="retrain_resampled.csv",
            type="text/csv"
        )
        
    except Exception as e:
        print(f"[RETRAIN] Export error: {str(e)}")
        raise PreventUpdate

