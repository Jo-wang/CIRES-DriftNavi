"""
Unified Column Type Modal Component - Phase 2

This module provides a single, unified modal component for column type management
that eliminates duplicate UI instances and pattern-matching ID conflicts.

Key Features:
- Single modal for both navigation and chatbox entry points
- DataTable-based editing (no pattern-matching conflicts)
- Integration with Phase 1 ColumnTypeManager
- Preparation for Phase 3 centralized change management

Following Dash best practices:
- Single responsibility principle
- Unique component IDs
- Reusable component design
- Clear separation of concerns
"""

import dash_bootstrap_components as dbc
from dash import html, dcc, dash_table
import pandas as pd
from typing import Dict, List, Any, Optional

from UI.state.column_type_manager import ColumnTypeManager


def create_unified_column_type_modal() -> dbc.Modal:
    """
    Create the unified column type management modal.
    
    This modal replaces the dual entry points (navigation + chatbox) with a single,
    centralized component that eliminates pattern-matching ID conflicts.
    
    Returns:
        dbc.Modal: Complete modal component ready for layout integration
    """
    modal = dbc.Modal([
        dbc.ModalHeader(
            dbc.ModalTitle("Column Type Comparison", id="unified-column-modal-title"),
            close_button=True
        ),
        
        dbc.ModalBody([
            # Status and feedback area
            html.Div(id="unified-column-modal-status", className="mb-3"),
            
            # Loading spinner for data preparation
            dcc.Loading(
                id="unified-column-modal-loading",
                children=[
                    html.Div(id="unified-column-modal-content")
                ],
                type="circle"
            )
        ]),
        
        dbc.ModalFooter([
            dbc.Button(
                "Close", 
                id="unified-column-modal-cancel", 
                color="light",
                n_clicks=0
            )
        ])
    ], 
    id="unified-column-type-modal", 
    size="xl", 
    scrollable=True,
    is_open=False)
    
    return modal


def prepare_column_type_data() -> List[Dict[str, Any]]:
    """
    Prepare column type data for the DataTable.
    
    This function consolidates column information from both datasets
    and prepares it in the format expected by the DataTable component.
    
    Returns:
        List[Dict]: Data rows for the DataTable
    """
    try:
        # Get all columns information from ColumnTypeManager
        all_columns_info = ColumnTypeManager.get_all_columns_info()
        
        if not all_columns_info:
            return []
        
        # Convert to DataTable format
        table_data = []
        for column, info in all_columns_info.items():
            # Only include columns that exist in at least one dataset
            if not info['exists_in_primary'] and not info['exists_in_secondary']:
                continue
                
            row = {
                'column': column,
                'classification': info['classification'],
                'data_type': info['pandas_type'],
                'primary_exists': '✓' if info['exists_in_primary'] else '✗',
                'secondary_exists': '✓' if info['exists_in_secondary'] else '✗',
                'primary_unique': info['unique_count_primary'] if info['exists_in_primary'] else 'N/A',
                'secondary_unique': info['unique_count_secondary'] if info['exists_in_secondary'] else 'N/A',
                'sample_values': ', '.join(info['sample_values'][:3]) if info['sample_values'] else 'N/A'
            }
            
            table_data.append(row)
        
        # Sort by column name for consistent display
        table_data.sort(key=lambda x: x['column'])
        
        # Debug: Print data format to console
        print(f"[UNIFIED MODAL] Prepared {len(table_data)} rows of data:")
        for i, row in enumerate(table_data[:3]):  # Print first 3 rows for debugging
            print(f"  Row {i}: {row}")
        
        return table_data
        
    except Exception as e:
        print(f"[UNIFIED MODAL] Error preparing column type data: {str(e)}")
        return []


def create_column_type_datatable(data: List[Dict[str, Any]]) -> dash_table.DataTable:
    """
    Create the DataTable component for column type editing.
    
    This replaces the pattern-matching dropdown system with a single DataTable
    that supports inline editing and eliminates ID conflicts.
    
    Args:
        data: Column type data prepared by prepare_column_type_data()
        
    Returns:
        dash_table.DataTable: Configured DataTable component
    """
    
    # Define column configuration
    columns = [
        {
            'name': 'Column', 
            'id': 'column', 
            'editable': False,
            'type': 'text'
        },
        {
            'name': 'Classification', 
            'id': 'classification', 
            'editable': False,
            'type': 'text'
        },
        {
            'name': 'Data Type', 
            'id': 'data_type', 
            'editable': False,
            'type': 'text'
        },
        {
            'name': 'Primary', 
            'id': 'primary_exists', 
            'editable': False,
            'type': 'text'
        },
        {
            'name': 'Secondary', 
            'id': 'secondary_exists', 
            'editable': False,
            'type': 'text'
        },
        {
            'name': 'Primary Unique', 
            'id': 'primary_unique', 
            'editable': False,
            'type': 'text'
        },
        {
            'name': 'Secondary Unique', 
            'id': 'secondary_unique', 
            'editable': False,
            'type': 'text'
        },
        {
            'name': 'Sample Values', 
            'id': 'sample_values', 
            'editable': False,
            'type': 'text'
        }
    ]
    
    # Create DataTable with proper dropdown configuration
    table = dash_table.DataTable(
        id='unified-column-type-datatable',
        columns=columns,
        data=data,
        row_deletable=False,
        editable=False,
        
        # Styling
        style_table={
            'overflowX': 'auto',
            'minWidth': '100%'
        },
        style_header={
            'backgroundColor': '#f8f9fa',
            'fontWeight': 'bold',
            'textAlign': 'center',
            'border': '1px solid #dee2e6'
        },
        style_cell={
            'textAlign': 'left',
            'padding': '8px',
            'border': '1px solid #dee2e6',
            'fontFamily': 'Arial, sans-serif',
            'fontSize': '14px'
        },
        style_data_conditional=[
            # Binary classification styling
            {
                'if': {'filter_query': '{classification} = Binary'},
                'backgroundColor': '#e6f7ff',
                'border': '1px solid #91d5ff'
            },
            # Continuous classification styling
            {
                'if': {'filter_query': '{classification} = Continuous'},
                'backgroundColor': '#f0fff0',
                'border': '1px solid #95de64'
            },
            # Datetime classification styling
            {
                'if': {'filter_query': '{classification} = Datetime'},
                'backgroundColor': '#fff0f5',
                'border': '1px solid #ffadd6'
            },
            # Categorical classification styling
            {
                'if': {'filter_query': '{classification} = Categorical'},
                'backgroundColor': '#fffacd',
                'border': '1px solid #fadb14'
            },
            # Highlight columns that don't exist in secondary dataset
            {
                'if': {'filter_query': '{secondary_exists} = ✗'},
                'color': '#ff4d4f',
                'fontWeight': 'bold'
            }
        ],
        
        # Tooltip configuration
        tooltip_data=[
            {
                'column': {
                    'value': f"Column: {row['column']}\nType: {row['classification']}\nData Type: {row['data_type']}",
                    'type': 'markdown'
                } if row else {}
                for row in data
            }
        ],
        tooltip_duration=None,
        
        # Pagination for large datasets
        page_size=20,
        page_action='native',
        
        # Sorting
        sort_action='native',
        sort_mode='multi'
    )
    
    return table


def create_modal_status_component(message: str = "", message_type: str = "info") -> html.Div:
    """
    Create a status message component for the modal.
    
    Args:
        message: Status message to display
        message_type: Type of message (info, success, warning, error)
        
    Returns:
        html.Div: Status component
    """
    if not message:
        return html.Div()
    
    color_map = {
        'info': 'info',
        'success': 'success', 
        'warning': 'warning',
        'error': 'danger'
    }
    
    alert = dbc.Alert(
        message,
        color=color_map.get(message_type, 'info'),
        dismissable=True,
        duration=5000 if message_type == 'success' else None
    )
    
    return html.Div([alert])


def create_modal_help_component() -> html.Div:
    """
    Create a help component explaining the column type management interface.
    
    Returns:
        html.Div: Help component
    """
    help_content = html.Div([
        dbc.Card([
            dbc.CardHeader("How to understand Column Types?"),
            dbc.CardBody([
                html.P([
                    html.Strong("Classification Types:"), html.Br(),
                    "• ", html.Strong("Binary"), ": Columns with exactly 2 unique values", html.Br(),
                    "• ", html.Strong("Continuous"), ": Numeric columns with many values", html.Br(), 
                    "• ", html.Strong("Datetime"), ": Date/time columns", html.Br(),
                    "• ", html.Strong("Categorical"), ": Text or limited-value columns"
                ]),
                html.P([
                    html.Strong("Data Types:"), html.Br(),
                    "• ", html.Strong("int64"), ": Integer numbers", html.Br(),
                    "• ", html.Strong("float64"), ": Decimal numbers", html.Br(),
                    "• ", html.Strong("object"), ": Text/string data", html.Br(),
                    "• ", html.Strong("datetime64[ns]"), ": Date/time data", html.Br(),
                    "• ", html.Strong("bool"), ": True/False values", html.Br(),
                    "• ", html.Strong("category"), ": Categorical data (memory efficient)"
                ])
                
            ])
        ], className="mb-3")
    ])
    
    return help_content