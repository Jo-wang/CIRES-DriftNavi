"""
Table Overview Controller - Phase 3
Single point of control for all table-overview updates

This controller eliminates the core problem of multiple callbacks writing to the same output
by providing a centralized master callback that manages all table-overview state changes.

Key Features:
- Single master callback with Output('table-overview', 'data')
- Cascading updates to all dependent tables
- Integration with Phase 1 ColumnTypeManager and Phase 2 Unified Modal
- Intelligent update propagation
- Conflict resolution and state consistency

Architecture:
┌─────────────────────────────────────────────────────────────┐
│                    TABLE OVERVIEW CONTROLLER                │
├─────────────────────────────────────────────────────────────┤
│  Master Callback: Output('table-overview', 'data')         │
│  ├─ Input: table-update-trigger (central trigger)          │
│  ├─ Input: global-data-state (from Phase 1.2)             │
│  ├─ Input: column-type-change-trigger (from Phase 1&2)     │
│  └─ State: Various component states                        │
├─────────────────────────────────────────────────────────────┤
│  Cascading Outputs:                                        │
│  ├─ Output: dependent-table-1.data                         │
│  ├─ Output: dependent-table-2.data                         │
│  └─ Output: ... (all tables that depend on main data)      │
└─────────────────────────────────────────────────────────────┘
"""

import time
import hashlib
import pandas as pd
from typing import Dict, List, Tuple
from dash import Input, Output, State, callback_context
from dash.exceptions import PreventUpdate

from UI.app import app
from UI.functions import global_vars
from UI.state.column_type_manager import ColumnTypeManager


class TableUpdateTrigger:
    """
    Centralized trigger system for table updates.
    
    This class manages all table update triggers and provides
    a single interface for triggering table-overview updates.
    """
    
    @staticmethod
    def create_trigger_data(source: str, reason: str, additional_data: Dict = None) -> Dict:
        """
        Create standardized trigger data for table updates.
        
        Args:
            source: Source component triggering the update
            reason: Reason for the update
            additional_data: Additional context data
            
        Returns:
            Dict: Standardized trigger data
        """
        trigger_data = {
            'timestamp': time.time(),
            'source': source,
            'reason': reason,
            'trigger_id': hashlib.md5(f"{source}_{reason}_{time.time()}".encode()).hexdigest()[:8]
        }
        
        if additional_data:
            trigger_data.update(additional_data)
            
        return trigger_data


class TableDataProcessor:
    """
    Processes and formats table data for different display contexts.
    
    This class handles the transformation of raw data into the formats
    required by different table components throughout the application.
    """
    
    @staticmethod
    def prepare_overview_table_data() -> Tuple[List[Dict], List[Dict]]:
        """
        Prepare data and columns for the main table-overview component.
        
        Returns:
            Tuple: (table_data, table_columns)
        """
        try:
            # Get current data state
            if not hasattr(global_vars, 'df') or global_vars.df is None:
                return [], []
            
            df = global_vars.df
            
            # Apply column type transformations if available
            if hasattr(global_vars, 'column_types') and global_vars.column_types:
                # Apply any column type conversions
                df_processed = df.copy()
                
                for column, type_info in global_vars.column_types.items():
                    if column in df_processed.columns:
                        try:
                            # Apply data type conversion based on ColumnTypeManager rules
                            if type_info.get('pandas_type') == 'datetime64[ns]':
                                df_processed[column] = pd.to_datetime(df_processed[column], errors='coerce')
                            elif type_info.get('pandas_type') == 'category':
                                df_processed[column] = df_processed[column].astype('category')
                            elif type_info.get('pandas_type') in ['int64', 'float64']:
                                df_processed[column] = pd.to_numeric(df_processed[column], errors='coerce')
                        except Exception as e:
                            print(f"[TABLE CONTROLLER] Warning: Could not convert {column} to {type_info.get('pandas_type')}: {e}")
                
                df = df_processed
            
            # Convert to table format
            table_data = df.to_dict('records')
            
            # Create enhanced column definitions
            table_columns = []
            for col in df.columns:
                column_def = {
                    "name": col, 
                    "id": col, 
                    'deletable': True,
                    'type': 'text'  # Default type
                }
                
                # Enhance with column type information
                if hasattr(global_vars, 'column_types') and col in global_vars.column_types:
                    type_info = global_vars.column_types[col]
                    classification = type_info.get('classification', 'Unknown')
                    pandas_type = type_info.get('pandas_type', 'object')
                    
                    # Add classification as tooltip
                    column_def['tooltip'] = f"Type: {classification} | Data Type: {pandas_type}"
                    
                    # Set appropriate column type for DataTable
                    if pandas_type in ['int64', 'float64']:
                        column_def['type'] = 'numeric'
                    elif pandas_type == 'datetime64[ns]':
                        column_def['type'] = 'datetime'
                    
                table_columns.append(column_def)
            
            print(f"[TABLE CONTROLLER] Prepared table data: {len(table_data)} rows, {len(table_columns)} columns")
            return table_data, table_columns
            
        except Exception as e:
            print(f"[TABLE CONTROLLER] Error preparing table data: {str(e)}")
            return [], []
    
    @staticmethod
    def prepare_dependent_table_data(table_id: str, base_data: List[Dict]) -> List[Dict]:
        """
        Prepare data for dependent tables based on the main table data.
        
        Args:
            table_id: ID of the dependent table
            base_data: Base data from main table
            
        Returns:
            List[Dict]: Processed data for the dependent table
        """
        try:
            # For now, most dependent tables use the same data
            # but this can be customized per table type
            
            if table_id.startswith('drift-'):
                # drift analysis tables might need specific formatting
                return base_data
            elif table_id.startswith('comparison-'):
                # Comparison tables might need different processing
                return base_data
            else:
                # Default: return base data
                return base_data
                
        except Exception as e:
            print(f"[TABLE CONTROLLER] Error preparing dependent table data for {table_id}: {str(e)}")
            return []


@app.callback(
    [Output('table-overview', 'data'),
     Output('table-overview', 'columns'),
     Output('table-overview-update-status', 'data')],  # Status for monitoring
    [Input('table-update-trigger', 'data'),
     Input('global-data-state', 'data'),
     Input('column-type-change-trigger', 'data')],
    [State('table-overview', 'data'),
     State('table-overview', 'columns')],
    prevent_initial_call=True
)
def master_table_overview_controller(update_trigger, global_state, type_changes, 
                                   current_data, current_columns):
    """
    Master callback for table-overview updates - Phase 3 Core
    
    This is the SINGLE point of control for all table-overview.data updates.
    All other callbacks that previously wrote to table-overview must use the
    table-update-trigger to request updates through this centralized system.
    
    Args:
        update_trigger: Central trigger for table updates
        global_state: Global data state from Phase 1.2
        type_changes: Column type changes from Phase 1&2
        current_data: Current table data
        current_columns: Current table columns
        
    Returns:
        tuple: (table_data, table_columns, update_status)
    """
    try:
        ctx = callback_context
        if not ctx.triggered:
            raise PreventUpdate
        
        triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
        trigger_value = ctx.triggered[0]['value']
        
        print(f"[TABLE CONTROLLER] Master callback triggered by: {triggered_id}")
        
        # Determine update reason and source
        update_source = "Unknown"
        update_reason = "Unknown trigger"
        
        if triggered_id == 'table-update-trigger' and trigger_value:
            update_source = trigger_value.get('source', 'Manual')
            update_reason = trigger_value.get('reason', 'Manual update')
        elif triggered_id == 'global-data-state':
            update_source = "GlobalDataState"
            update_reason = "Data state changed"
        elif triggered_id == 'column-type-change-trigger':
            update_source = "ColumnTypeManager"
            update_reason = "Column types changed"
        
        # Process table data
        table_data, table_columns = TableDataProcessor.prepare_overview_table_data()
        
        # Create update status for monitoring
        update_status = {
            'timestamp': time.time(),
            'source': update_source,
            'reason': update_reason,
            'triggered_by': triggered_id,
            'rows_updated': len(table_data),
            'columns_updated': len(table_columns),
            'data_hash': hashlib.md5(str(table_data).encode()).hexdigest()[:8] if table_data else "empty"
        }
        
        print(f"[TABLE CONTROLLER] Updated table: {len(table_data)} rows, {len(table_columns)} cols (Source: {update_source})")
        
        return table_data, table_columns, update_status
        
    except PreventUpdate:
        raise
    except Exception as e:
        error_msg = f"Error in master table controller: {str(e)}"
        print(f"[TABLE CONTROLLER] {error_msg}")
        
        # Return safe defaults on error
        error_status = {
            'timestamp': time.time(),
            'error': error_msg,
            'source': 'Error',
            'reason': 'Callback failed'
        }
        
        return current_data or [], current_columns or [], error_status

# Helper function for manual table updates
def trigger_table_update(source: str, reason: str, additional_data: Dict = None):
    """
    Helper function to programmatically trigger table updates.
    
    This function can be called from other parts of the application
    to request a table-overview update through the centralized system.
    
    Args:
        source: Source component requesting the update
        reason: Reason for the update
        additional_data: Additional context data
    """
    try:
        trigger_data = TableUpdateTrigger.create_trigger_data(source, reason, additional_data)
        
        # In a real implementation, this would update the table-update-trigger store
        # For now, this is a placeholder for the pattern
        print(f"[TABLE CONTROLLER] Manual update requested: {source} - {reason}")
        
    except Exception as e:
        print(f"[TABLE CONTROLLER] Error triggering manual update: {str(e)}")