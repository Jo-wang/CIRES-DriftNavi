"""
Unified Column Type Modal Callbacks - Phase 2

This module contains all callbacks for the unified column type modal,
providing centralized management and eliminating the dual entry point conflicts.

Key Features:
- Single modal management for both navigation and chatbox entry points
- DataTable-based editing without pattern-matching conflicts
- Integration with Phase 1 ColumnTypeManager
- Preparation for Phase 3 centralized change management

Following Dash best practices:
- Single responsibility per callback
- Clear dependency chains
- Proper error handling
- Integration with existing global state
"""

import dash
from dash import callback, Input, Output, State, ALL, callback_context
from dash.exceptions import PreventUpdate
import time

from UI.app import app
from UI.components.unified_column_type_modal import (
    prepare_column_type_data, 
    create_column_type_datatable,
    create_modal_status_component,
    create_modal_help_component
)
from UI.state.column_type_manager import ColumnTypeManager


@app.callback(
    [Output('unified-column-type-modal', 'is_open'),
     Output('unified-column-modal-content', 'children'),
     Output('unified-column-modal-status', 'children')],
    [Input('menu-column-types', 'n_clicks'),           # Navigation entry point for column types
     # REMOVED: Input('show-type-compare-btn', 'n_clicks') - Handled by chat_callbacks.py to avoid auto-trigger
     # REMOVED: Input('unified-column-modal-close', 'n_clicks') - Using close_button=True instead
     Input('unified-column-modal-cancel', 'n_clicks')], # Cancel button
    [State('unified-column-type-modal', 'is_open')],
    prevent_initial_call=True
)
def manage_unified_column_modal(nav_clicks, cancel_clicks, is_open):
    """
    Modal management for column type editing.
    
    This callback handles opening the column type modal from navigation menu only.
    Chatbox entry point is handled by chat_callbacks.py to avoid auto-trigger issues.
    
    Args:
        nav_clicks: Clicks from navigation menu "Column Type Comparison"
        cancel_clicks: Clicks from modal cancel button
        is_open: Current modal state
        
    Returns:
        tuple: (modal_open_state, modal_content, status_message)
    """
    try:
        ctx = callback_context
        if not ctx.triggered:
            return False, [], []
        
        triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        # Open modal from navigation entry point only
        if triggered_id in ['menu-column-types']:
            # Prepare column data for the DataTable
            column_data = prepare_column_type_data()
            
            if not column_data:
                # No data available
                status = create_modal_status_component(
                    "No column data available. Please ensure both datasets are loaded.", 
                    "warning"
                )
                content = [create_modal_help_component()]
                return True, content, status
            
            # Create DataTable with current data
            data_table = create_column_type_datatable(column_data)
            
            # Create content with help section
            content = [
                create_modal_help_component(),
                data_table
            ]
            
            # Success status
            status = create_modal_status_component(
                f"Column type comparison ready. {len(column_data)} columns loaded.", 
                "success"
            )
            
            print(f"[UNIFIED MODAL] Opened column type modal from navigation menu with {len(column_data)} columns")
            return True, content, status
        
        # Close modal via cancel button (close_button=True handles X button automatically)
        elif triggered_id in ['unified-column-modal-cancel']:
            print(f"[UNIFIED MODAL] Closed modal via {triggered_id}")
            return False, [], []
        
        # Default: don't change state
        return is_open, dash.no_update, dash.no_update
        
    except Exception as e:
        error_msg = f"Error managing unified column modal: {str(e)}"
        print(f"[UNIFIED MODAL] {error_msg}")
        
        status = create_modal_status_component(error_msg, "error")
        return is_open, [], status


@app.callback(
    Output('unified-column-modal-status', 'children', allow_duplicate=True),
    Input('unified-column-modal-cancel', 'n_clicks'),
    prevent_initial_call=True
)
def apply_column_type_changes(_):
    raise PreventUpdate


@app.callback(
    Output('unified-column-modal-status', 'children', allow_duplicate=True),
    Input('unified-column-modal-reset', 'n_clicks'),
    prevent_initial_call=True
)
def reset_column_types_to_detected(reset_clicks):
    return create_modal_status_component("Read-only mode: reset is disabled.", "info")


@app.callback(
    Output('unified-column-type-datatable', 'style_data_conditional', allow_duplicate=True),
    Input('unified-column-type-datatable', 'data'),
    prevent_initial_call=True
)
def update_datatable_conditional_styling(table_data):
    try:
        if not table_data:
            return []
        base_styles = [
            {
                'if': {'filter_query': '{classification} = Binary'},
                'backgroundColor': '#e6f7ff',
                'border': '1px solid #91d5ff'
            },
            {
                'if': {'filter_query': '{classification} = Continuous'},
                'backgroundColor': '#f0fff0',
                'border': '1px solid #95de64'
            },
            {
                'if': {'filter_query': '{classification} = Datetime'},
                'backgroundColor': '#fff0f5',
                'border': '1px solid #ffadd6'
            },
            {
                'if': {'filter_query': '{classification} = Categorical'},
                'backgroundColor': '#fffacd',
                'border': '1px solid #fadb14'
            },
            {
                'if': {'filter_query': '{secondary_exists} = ✗'},
                'color': '#ff4d4f',
                'fontWeight': 'bold'
            }
        ]
        return base_styles
    except Exception as e:
        print(f"[UNIFIED MODAL] Error updating conditional styling: {str(e)}")
        return []


# Disable legacy callbacks by overriding them with no-ops
# REMOVED: This callback conflicted with chat_callbacks.py query-area output
# The unified modal system handles show-type-compare-btn clicks through the main modal management
# callback above, so this legacy disable callback is not needed.


# Real-time DataTable editing feedback
@app.callback(
    Output('unified-column-modal-status', 'children', allow_duplicate=True),
    Input('unified-column-type-datatable', 'data'),
    prevent_initial_call=True
)
def update_datatable_status(_):
    raise PreventUpdate