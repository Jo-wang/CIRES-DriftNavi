"""
Enhanced Table Callbacks - Direct Column Type Editing
Provides inline column type editing in the main data tables

This module extends the table display to include editable column type information
directly in the data tables, eliminating the need for modal-based editing.

Key Features:
- Inline column type editing with dropdowns
- Direct save mechanism for global application
- Integration with existing Phase 1-3 system
- Improved user workflow

Following Dash best practices:
- Single responsibility principle
- Clear callback definitions
- Proper error handling
- Integration with existing architecture
"""

import dash
from dash import callback, Input, Output, State, ALL, callback_context, html, dcc, dash_table
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import pandas as pd
from typing import Dict, List, Any, Optional

from UI.app import app
from UI.functions import global_vars
from UI.state.column_type_manager import ColumnTypeManager


def create_enhanced_table_with_column_types(df: pd.DataFrame, table_id: str, title: str) -> html.Div:
    """
    Create an enhanced table that includes column type information and editing capabilities.
    
    Args:
        df: DataFrame to display
        table_id: ID for the table component
        title: Title for the table section
        
    Returns:
        html.Div: Enhanced table component with column type editing
    """
    try:
        if df is None or df.empty:
            return html.Div([
                html.H5(title, className="table-title"),
                html.P("No data available", className="text-muted text-center")
            ])
        
        # Get column type information
        all_columns_info = ColumnTypeManager.get_all_columns_info()
        
        # Create table for actual data (first few rows)
        data_rows = df.head(10).to_dict('records')
        data_columns = [{"name": col, "id": col} for col in df.columns]
        
        # Create column type summary table
        column_type_rows = []
        for col in df.columns:
            if col in all_columns_info:
                info = all_columns_info[col]
                column_type_rows.append({
                    'column': col,
                    'classification': info.get('classification', 'Unknown'),
                    'data_type': info.get('pandas_type', 'object'),
                    'unique_count': info.get('unique_count_primary' if 'primary' in table_id else 'unique_count_secondary', 0)
                })
            else:
                column_type_rows.append({
                    'column': col,
                    'classification': 'Unknown',
                    'data_type': str(df[col].dtype),
                    'unique_count': df[col].nunique()
                })
        
        # Column type table with editable cells
        column_type_columns = [
            {"name": "Column", "id": "column", "editable": False},
            {
                "name": "Classification", 
                "id": "classification", 
                "editable": True,
                "presentation": "dropdown"
            },
            {
                "name": "Data Type", 
                "id": "data_type", 
                "editable": True,
                "presentation": "dropdown"
            },
            {"name": "Unique Values", "id": "unique_count", "editable": False}
        ]
        
        # Define dropdown options
        classification_options = ['Binary', 'Continuous', 'Categorical', 'Datetime']
        data_type_options = ['object', 'int64', 'float64', 'datetime64[ns]', 'category', 'bool']
        
        component = html.Div([
            # Title and save button
            html.Div([
                html.H5(title, className="table-title mb-2"),
                dbc.Button([
                    html.I(className="fas fa-save me-1"),
                    "Save Column Types"
                ], 
                id=f"save-column-types-{table_id.split('-')[-1]}", 
                color="success", 
                size="sm",
                className="ms-auto")
            ], className="d-flex justify-content-between align-items-center mb-3"),
            
            # Status message area
            html.Div(id=f"column-type-status-{table_id.split('-')[-1]}", className="mb-2"),
            
            # Column types table (editable)
            html.Div([
                html.H6("Column Types (Editable)", className="mb-2"),
                dash_table.DataTable(
                    id=f"column-types-table-{table_id.split('-')[-1]}",
                    data=column_type_rows,
                    columns=column_type_columns,
                    editable=True,
                    dropdown={
                        'classification': {
                            'options': [{'label': opt, 'value': opt} for opt in classification_options]
                        },
                        'data_type': {
                            'options': [{'label': opt, 'value': opt} for opt in data_type_options]
                        }
                    },
                    style_cell={'textAlign': 'left', 'padding': '10px'},
                    style_header={'backgroundColor': '#f8f9fa', 'fontWeight': 'bold'},
                    style_data_conditional=[
                        {
                            'if': {'column_id': 'classification', 'filter_query': '{classification} = Binary'},
                            'backgroundColor': '#e6f7ff',
                            'border': '1px solid #91d5ff'
                        },
                        {
                            'if': {'column_id': 'classification', 'filter_query': '{classification} = Continuous'},
                            'backgroundColor': '#f0fff0',
                            'border': '1px solid #95de64'
                        },
                        {
                            'if': {'column_id': 'classification', 'filter_query': '{classification} = Categorical'},
                            'backgroundColor': '#fffacd',
                            'border': '1px solid #fadb14'
                        },
                        {
                            'if': {'column_id': 'classification', 'filter_query': '{classification} = Datetime'},
                            'backgroundColor': '#fff0f5',
                            'border': '1px solid #ffadd6'
                        }
                    ]
                )
            ], className="mb-4"),
            
            # Data preview table (read-only)
            html.Div([
                html.H6(f"Data Preview ({len(df)} total rows)", className="mb-2"),
                dash_table.DataTable(
                    id=table_id,
                    data=data_rows,
                    columns=data_columns,
                    page_size=10,
                    style_cell={'textAlign': 'left', 'padding': '8px'},
                    style_header={'backgroundColor': '#614385', 'color': 'white', 'fontWeight': 'bold'},
                    style_table={'overflowX': 'auto'}
                )
            ])
        ], className="enhanced-table-container mb-4")
        
        return component
        
    except Exception as e:
        print(f"[ENHANCED TABLE] Error creating enhanced table: {str(e)}")
        return html.Div([
            html.H5(title, className="table-title"),
            dbc.Alert(f"Error creating table: {str(e)}", color="warning")
        ])


def register_enhanced_table_callbacks(app):
    """Register callbacks for enhanced table functionality."""
    
    # Main callback to update enhanced table containers
    @app.callback(
        [Output('enhanced-table-container-primary', 'children'),
         Output('enhanced-table-container-secondary', 'children')],
        [Input('table-overview', 'data'),
         Input('table-overview', 'columns'),
         Input('global-data-state', 'data')],
        prevent_initial_call=True
    )
    def update_enhanced_tables(main_data, main_columns, global_state):
        """Update enhanced table containers with column type editing."""
        try:
            # Check if we have datasets
            has_primary = hasattr(global_vars, 'df') and global_vars.df is not None
            has_secondary = hasattr(global_vars, 'secondary_df') and global_vars.secondary_df is not None
            
            primary_content = html.Div()
            secondary_content = html.Div()
            
            if has_primary:
                primary_content = create_enhanced_table_with_column_types(
                    global_vars.df, 'table-primary-overview', 'Primary Dataset'
                )
            
            if has_secondary:
                secondary_content = create_enhanced_table_with_column_types(
                    global_vars.secondary_df, 'table-secondary-overview', 'Secondary Dataset'
                )
            
            return primary_content, secondary_content
            
        except Exception as e:
            print(f"[ENHANCED TABLE] Error updating enhanced tables: {str(e)}")
            return html.Div(), html.Div()
    
    # Save button callbacks for primary table
    @app.callback(
        Output('column-type-status-primary', 'children'),
        Input('save-column-types-primary', 'n_clicks'),
        State('column-types-table-primary', 'data'),
        prevent_initial_call=True
    )
    def save_primary_column_types(n_clicks, table_data):
        """Save column type changes from primary table."""
        return save_column_type_changes(n_clicks, table_data, 'primary')
    
    # Save button callbacks for secondary table  
    @app.callback(
        Output('column-type-status-secondary', 'children'),
        Input('save-column-types-secondary', 'n_clicks'),
        State('column-types-table-secondary', 'data'),
        prevent_initial_call=True
    )
    def save_secondary_column_types(n_clicks, table_data):
        """Save column type changes from secondary table."""
        return save_column_type_changes(n_clicks, table_data, 'secondary')


def save_column_type_changes(n_clicks, table_data, table_type):
    """
    Process and save column type changes.
    
    Args:
        n_clicks: Number of button clicks
        table_data: Data from the column types table
        table_type: Type of table ('primary' or 'secondary')
        
    Returns:
        Component for status display
    """
    try:
        if not n_clicks or not table_data:
            raise PreventUpdate
        
        print(f"[ENHANCED TABLE] Saving {table_type} column type changes...")
        
        # Track results
        success_count = 0
        error_count = 0
        results = []
        
        # Process each row
        for row in table_data:
            column = row['column']
            classification = row['classification']
            data_type = row['data_type']
            
            # Apply changes through ColumnTypeManager
            success, message, warning = ColumnTypeManager.update_column_type(
                column, classification, data_type
            )
            
            if success:
                success_count += 1
                if warning:
                    results.append(f"✓ {column}: {message} (Warning: {warning})")
                else:
                    results.append(f"✓ {column}: {message}")
            else:
                error_count += 1
                results.append(f"✗ {column}: {message}")
        
        # Create status message
        if error_count == 0:
            status_color = "success"
            summary = f"✅ Successfully saved {success_count} column type changes!"
        else:
            status_color = "warning"
            summary = f"⚠️ Saved {success_count} changes, {error_count} failed."
        
        # Trigger global update
        # This will be handled by the existing global state system
        
        return dbc.Alert([
            html.Strong(summary),
            html.Br(),
            html.Small("Changes applied globally to all analysis components.")
        ], color=status_color, dismissable=True)
        
    except PreventUpdate:
        raise
    except Exception as e:
        return dbc.Alert(f"Error saving changes: {str(e)}", color="danger", dismissable=True)