"""
Finetune workflow components for Adapt stage.

This module provides interactive components for the finetune data preparation workflow,
including coreset selection from primary dataset and resampling of secondary dataset.
"""

import dash
from dash import html, dcc, callback, Input, Output, State, MATCH, ALL
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import pandas as pd

from UI.app import app
from UI.functions.global_vars import global_vars


def create_finetune_config_bubble(unique_id: str) -> html.Div:
    """
    Create configuration bubble for finetune workflow.
    
    Args:
        unique_id: Unique identifier for this bubble instance
        
    Returns:
        html.Div containing the configuration interface
    """
    return html.Div([
        dbc.Alert([
            html.H5([
                html.I(className="fas fa-tools me-2"),
                "Finetune Data Preparation"
            ], className="alert-heading"),
            html.P([
                "To prevent model forgetting while adapting to new data, we use a two-stage approach: ",
                html.Strong("(1) Coreset Selection"), " intelligently samples representative examples from Primary dataset, ",
                html.Strong("(2) Resampling"), " balances Secondary dataset across selected attributes and target variable."
            ], className="mb-3"),
        ], color="info"),
        
        # Coreset Configuration
        dbc.Card([
            dbc.CardHeader(html.Strong("Step 1: Primary Coreset Selection")),
            dbc.CardBody([
                html.Label("Coreset Retention Percentage (%)", className="fw-bold"),
                dcc.Slider(
                    id={"type": "finetune-coreset-slider", "index": unique_id},
                    min=1,
                    max=50,
                    step=1,
                    value=10,
                    marks={5: "5%", 10: "10%", 15: "15%", 20: "20%", 30: "30%", 40: "40%", 50: "50%"},
                    tooltip={"placement": "bottom", "always_visible": True}
                ),
                html.Small("Recommended: 5-20%. Higher percentages retain more old data but may dilute new patterns.", 
                          className="text-muted")
            ])
        ], className="mb-3"),
        
        # Secondary Resampling Configuration
        dbc.Card([
            dbc.CardHeader(html.Strong("Step 2: Secondary Dataset Resampling")),
            dbc.CardBody([
                html.Label("Select Attributes to Balance", className="fw-bold"),
                dcc.Dropdown(
                    id={"type": "finetune-attrs-dropdown", "index": unique_id},
                    options=[],  # Populated by callback
                    multi=True,
                    placeholder="Select one or more attributes (e.g., gender, race, age_group)"
                ),
                html.Small("These attributes will be balanced jointly with the target variable. Continuous variables are automatically binned.", 
                          className="text-muted d-block mb-3"),
                
                html.Label("Resampling Strategy", className="fw-bold mt-2"),
                dcc.RadioItems(
                    id={"type": "finetune-strategy-radio", "index": unique_id},
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
            id={"type": "finetune-confirm-btn", "index": unique_id},
            color="primary",
            size="lg",
            className="w-100"
        )
    ])


def create_finetune_success_bubble(combined_size: int, coreset_size: int, resampled_size: int, unique_id: str) -> html.Div:
    """
    Create success bubble after finetune processing completes.
    
    Args:
        combined_size: Total size of combined dataset
        coreset_size: Size of primary coreset
        resampled_size: Size of resampled secondary
        unique_id: Unique identifier for this bubble instance
        
    Returns:
        html.Div containing success message and export button
    """
    return html.Div([
        dbc.Alert([
            html.H5([
                html.I(className="fas fa-check-circle me-2"),
                "Dataset for Finetuning the Model is Ready!"
            ], className="alert-heading text-success"),
            html.Hr(),
            html.H6("Dataset Statistics:", className="mb-2"),
            html.Ul([
                html.Li(f"Primary Coreset: {coreset_size:,} samples"),
                html.Li(f"Resampled Secondary: {resampled_size:,} samples"),
                html.Li(html.Strong(f"Total Combined: {combined_size:,} samples"))
            ], className="mb-3"),
            html.P("The combined dataset is ready for model training. You can export it as a CSV file below.", 
                   className="mb-3"),
            dbc.Button([
                html.I(className="fas fa-download me-2"),
                "Export as CSV"
            ],
            id={"type": "finetune-export-btn", "index": unique_id},
            color="success",
            size="lg",
            className="w-100")
        ], color="light", className="border border-success")
    ])


# Callback 1: Populate attribute options for dropdown
@callback(
    Output({"type": "finetune-attrs-dropdown", "index": MATCH}, "options"),
    Input({"type": "finetune-attrs-dropdown", "index": MATCH}, "id"),
    prevent_initial_call=False
)
def populate_finetune_attrs_dropdown(_id):
    """Populate dropdown with secondary dataset columns (excluding target)."""
    try:
        # Try to get secondary dataset with proper None checking to avoid DataFrame ambiguity
        secondary_df = getattr(global_vars, "secondary_df", None)
        if secondary_df is None:
            secondary_df = getattr(global_vars, "df_secondary", None)
        
        if secondary_df is None:
            print(f"[FINETUNE] No secondary dataset available for dropdown population")
            return []
        
        # Get all columns
        all_cols = list(secondary_df.columns)
        print(f"[FINETUNE] Populating dropdown with {len(all_cols)} columns from secondary dataset")
        
        # Exclude target attribute if set
        target_col = getattr(global_vars, 'target_attribute', None)
        if target_col and target_col in all_cols:
            all_cols.remove(target_col)
        
        # Return as options
        return [{"label": col, "value": col} for col in all_cols]
        
    except Exception as e:
        print(f"[FINETUNE] Error populating attributes: {e}")
        return []


# Callback 2: Process finetune workflow - Single callback approach (Solution B)
# KEY LESSON: Don't use pattern-matching with intermediate stores
# Direct approach: ALL inputs → process → single output (query-area)
@callback(
    Output("query-area", "children", allow_duplicate=True),
    Input({"type": "finetune-confirm-btn", "index": ALL}, "n_clicks"),
    [State({"type": "finetune-coreset-slider", "index": ALL}, "value"),
     State({"type": "finetune-attrs-dropdown", "index": ALL}, "value"),
     State({"type": "finetune-strategy-radio", "index": ALL}, "value"),
     State({"type": "finetune-confirm-btn", "index": ALL}, "id"),
     State("query-area", "children")],
    prevent_initial_call=True
)
def process_finetune_workflow(n_clicks_list, coreset_percents, selected_attrs_list, strategies, button_ids, query_records):
    """
    Process finetune workflow using ALL pattern.
    Uses ctx.triggered to determine which button was clicked and processes accordingly.
    No button state update - only updates chat area.
    """
    import dash
    import time
    from utils.finetune_pipeline import select_coreset
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
            print(f"[FINETUNE] Could not find clicked button index")
            raise PreventUpdate
        
        # Extract parameters for this specific button
        unique_id = triggered_index_value
        coreset_percent = coreset_percents[clicked_idx]
        selected_attrs = selected_attrs_list[clicked_idx]
        strategy = strategies[clicked_idx]
        
        print(f"[FINETUNE_WORKFLOW] Button {unique_id} clicked (position {clicked_idx})")
        print(f"[FINETUNE_WORKFLOW] Parameters: coreset={coreset_percent}%, attrs={selected_attrs}, strategy={strategy}")
        
    except Exception as e:
        print(f"[FINETUNE] Error parsing triggered button: {e}")
        raise PreventUpdate
    
    # Initialize query_records if None
    if query_records is None:
        query_records = []
    
    try:
        print(f"[FINETUNE_WORKFLOW] Starting finetune data generation...")
        print(f"[FINETUNE_WORKFLOW] Coreset percent: {coreset_percent}%")
        print(f"[FINETUNE_WORKFLOW] Selected attributes: {selected_attrs}")
        print(f"[FINETUNE_WORKFLOW] Strategy: {strategy}")
        
        # Validate inputs with proper None checking
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
        
        # Step 1: Generate Primary Coreset
        print(f"[FINETUNE_WORKFLOW] Step 1: Generating primary coreset...")
        coreset_df = select_coreset(
            primary_df=global_vars.df,
            percent=coreset_percent,
            algo="dt",
            random_state=42
        )
        global_vars.primary_coreset_df = coreset_df
        print(f"[FINETUNE_WORKFLOW] Coreset generated: {len(coreset_df)} samples")
        
        # Step 2: Resample Secondary Dataset
        print(f"[FINETUNE_WORKFLOW] Step 2: Resampling secondary dataset...")
        sampler = MultiAttributeSampler(
            df=secondary_df,
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
        
        global_vars.secondary_resampled_df = resampled_df
        print(f"[FINETUNE_WORKFLOW] Secondary resampled: {len(resampled_df)} samples")
        
        # Step 3: Combine datasets
        print(f"[FINETUNE_WORKFLOW] Step 3: Combining datasets...")
        combined_df = pd.concat([coreset_df, resampled_df], axis=0).reset_index(drop=True)
        global_vars.finetune_combined_df = combined_df
        print(f"[FINETUNE_WORKFLOW] Combined dataset: {len(combined_df)} samples")
        
        # Step 4: Create success bubble
        success_bubble = create_finetune_success_bubble(
            combined_size=len(combined_df),
            coreset_size=len(coreset_df),
            resampled_size=len(resampled_df),
            unique_id=unique_id
        )
        
        success_message = create_timestamped_message(success_bubble, "llm-msg")
        
        # Add original_type for proper sorting
        if hasattr(success_message, 'id') and isinstance(success_message.id, dict):
            success_message.id.update({
                "original_type": "chat-adapt-bubble",
                "index": f"finetune-success-{int(time.time() * 1000)}"
            })
        
        query_records.append(success_message)
        sorted_records = sort_chat_messages(query_records)
        
        print(f"[FINETUNE_WORKFLOW] Workflow completed successfully!")
        
        # Return updated chat records (single output - no button state update)
        return sorted_records
        
    except Exception as e:
        print(f"[FINETUNE_WORKFLOW] Error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Create error bubble
        error_bubble = dbc.Alert([
            html.H5([html.I(className="fas fa-exclamation-triangle me-2"), "Error"], className="alert-heading"),
            html.P(f"Failed to generate finetune dataset: {str(e)}"),
            html.Hr(),
            html.P("Please check that both datasets are loaded and target attribute is set.", className="mb-0")
        ], color="danger")
        
        error_message = create_timestamped_message(error_bubble, "llm-msg")
        if hasattr(error_message, 'id') and isinstance(error_message.id, dict):
            error_message.id.update({
                "original_type": "chat-adapt-bubble",
                "index": f"finetune-error-{int(time.time() * 1000)}"
            })
        
        query_records.append(error_message)
        sorted_records = sort_chat_messages(query_records)
        
        # Return updated chat records with error message (single output - no button state update)
        return sorted_records


# Callback 3: Export CSV
# NOTE: Solution B - No button state updates, only chat area updates
@callback(
    Output("finetune-download", "data"),
    Input({"type": "finetune-export-btn", "index": ALL}, "n_clicks"),
    prevent_initial_call=True
)
def export_finetune_csv(n_clicks_list):
    """Export the combined finetune dataset as CSV."""
    # Check if any button was clicked
    if not n_clicks_list or not any(n_clicks_list):
        raise PreventUpdate
    
    try:
        # Verify combined dataset exists
        if not hasattr(global_vars, 'finetune_combined_df') or global_vars.finetune_combined_df is None:
            print("[FINETUNE] No combined dataset available for export")
            raise PreventUpdate
        
        # Convert to CSV
        csv_string = global_vars.finetune_combined_df.to_csv(index=False)
        
        print(f"[FINETUNE] Exporting combined dataset: {len(global_vars.finetune_combined_df)} rows")
        
        # Return download dict
        return dict(
            content=csv_string,
            filename="finetune_combined.csv",
            type="text/csv"
        )
        
    except Exception as e:
        print(f"[FINETUNE] Export error: {str(e)}")
        raise PreventUpdate

