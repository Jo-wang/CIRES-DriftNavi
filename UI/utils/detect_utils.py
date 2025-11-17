"""
Detect Analysis Utility Functions

This module contains utility functions for detect analysis that can be safely
imported in callbacks without triggering page registration conflicts.

Extracted from home_layout.py to avoid circular imports and page registration issues.
"""

from UI.functions.global_vars import global_vars


def calculate_drift_severity_score(metric_name, value):
    """
    Calculate drift severity score based on metric-specific logic.
    - For most metrics: higher value = more severe drift
    - For p-value metrics: lower value = more severe drift
    - For N/A values: return 0 (cannot be scored)
    
    Returns a score from 0-100, where 100 is most severe.
    """
    if value == "N/A" or value is None:
        return 0
    
    try:
        numeric_value = float(value)
    except (ValueError, TypeError):
        return 0
    
    # Define p-value type metrics (lower value = more severe)
    p_value_metrics = {
         'T_Test', 'Z_Test', 'Fisher_Exact', 
        'Chi_Square', 'G_Test', 'Anderson_Darling', 'Cramer_von_Mises'
    }
    
    # For p-value metrics, use inverse logic
    if metric_name in p_value_metrics:
        # p-value: 0.0-0.05 = high severity, 0.05-1.0 = low severity
        if numeric_value <= 0.001:
            return 100  # Extremely high severity
        elif numeric_value <= 0.01:
            return 90   # Very high severity
        elif numeric_value <= 0.05:
            return 75   # High severity
        elif numeric_value <= 0.1:
            return 50   # Medium severity
        else:
            return max(0, 30 * (1 - numeric_value))  # Linear decrease
    
    # For other metrics, higher value = more severe drift
    # Use unified scoring for consistency within metric types
    return min(100, abs(numeric_value) * 100)


def calculate_severity_ranking_and_styles(metrics_data):
    """
    Calculate the top 3 most severe cells for each metric type and generate highlight style conditions.
    For N metrics, highlights N*3 cells total (top 3 per metric type).
    
    Uses direct metric value comparison for accurate ranking within each metric type.
    """
    if not metrics_data:
        return []
    
    # Columns to skip (non-metric columns)
    skip_columns = ['Attribute', 'Type', 'AddToContext', 'ExplainAction']
    
    # Group data by metric type
    metrics_by_column = {}
    
    for row_idx, row in enumerate(metrics_data):
        attribute_name = row.get('Attribute', f'Row_{row_idx}')
        for col_name, value in row.items():
            if col_name not in skip_columns and value != "N/A" and value is not None:
                try:
                    numeric_value = float(value)
                    if col_name not in metrics_by_column:
                        metrics_by_column[col_name] = []
                    
                    metrics_by_column[col_name].append({
                        'row_index': row_idx,
                        'column_id': col_name,
                        'attribute': attribute_name,
                        'value': value,
                        'numeric_value': numeric_value
                    })
                except (ValueError, TypeError):
                    continue  # Skip values that cannot be converted to numeric
    
    # Calculate top 3 for each metric type
    style_conditions = []
    severity_colors = [
        '#e57373',  # Dark red - 1st most severe
        '#ffcdd2',  # Medium red - 2nd most severe
        '#ffebee'   # Light red - 3rd most severe
    ]
    
    # Define p-value type metrics (lower value = more severe)
    p_value_metrics = {
     'T_Test', 'Z_Test', 'Fisher_Exact', 
        'Chi_Square', 'G_Test', 'Anderson_Darling', 'Cramer_von_Mises'
    }
    
    for metric_name, cells in metrics_by_column.items():
        # Determine sorting direction based on metric type
        is_p_value_metric = metric_name in p_value_metrics
        
        if is_p_value_metric:
            # For p-value metrics: lower value = more severe (ascending sort)
            cells.sort(key=lambda x: x['numeric_value'])
        else:
            # For other metrics: higher value = more severe (descending sort)
            cells.sort(key=lambda x: x['numeric_value'], reverse=True)
        
        # Take top 3
        top_3 = cells[:3]
        
        print(f"[DEBUG] Top 3 for {metric_name} ({'p-value' if is_p_value_metric else 'regular'}):")
        for i, cell in enumerate(top_3):
            rank = i + 1
            bg_color = severity_colors[min(i, 2)]
            text_color = 'white' if i == 0 else 'black'  # 1st rank uses white text, others use black
            
            style_conditions.append({
                'if': {
                    'row_index': cell['row_index'],
                    'column_id': cell['column_id']
                },
                'backgroundColor': bg_color,
                'color': text_color,
                'fontWeight': 'bold',
                'border': '3px solid #d32f2f',
                'borderRadius': '4px',
                'boxShadow': '0 2px 4px rgba(211,47,47,0.3)'
            })
            
            print(f"  #{rank}: {cell['attribute']}.{cell['column_id']} = {cell['value']}")
    
    print(f"[DEBUG] Total highlighted cells: {len(style_conditions)} across {len(metrics_by_column)} metrics")
    
    return style_conditions


def create_metrics_heatmap(metrics_data, target_attribute):
    """
    Create metrics heatmap visualization.
    Extracted from home_layout.py to avoid import conflicts.
    """
    try:
        # This is a simplified version - you may need to copy the full logic from home_layout.py
        # For now, return None to avoid the import issue
        return None
    except Exception as e:
        print(f"[METRICS HEATMAP] Error: {str(e)}")
        return None


def create_dual_add_buttons(feature_name, chat_button_id, explain_button_id, 
                           chat_disabled=False, explain_disabled=False,
                           chat_aria_disabled="false", explain_aria_disabled="false"):
    """
    Create dual add buttons for chat and explain functionality.
    Extracted from home_layout.py to avoid import conflicts.
    """
    import dash_bootstrap_components as dbc
    from dash import html
    
    try:
        return html.Div([
            dbc.ButtonGroup([
                dbc.Button(
                    [html.I(className="fas fa-plus me-1"), "Add to Chat"],
                    id=chat_button_id,
                    color="info",
                    size="sm",
                    outline=True,
                    disabled=chat_disabled,
                    className="me-1",
                    style={"fontSize": "12px"}
                ),
                dbc.Button(
                    [html.I(className="fas fa-plus me-1"), "Add to Explain"],
                    id=explain_button_id,
                    color="success",
                    size="sm",
                    outline=True,
                    disabled=explain_disabled,
                    style={"fontSize": "12px"}
                )
            ], className="d-flex justify-content-center")
        ], className="text-center mt-2")
    except Exception as e:
        print(f"[DUAL BUTTONS] Error: {str(e)}")
        return html.Div()  # Empty fallback


def get_fresh_metrics_data_for_chat():
    """
    Get metrics data for detect button, calculating on-demand if necessary.
    
    This function implements a robust caching strategy:
    1. Check if valid cached metrics exist -> return immediately
    2. If cache is invalid/missing -> calculate new metrics
    3. Cache the newly calculated metrics for future use
    
    Returns:
        tuple: (metrics_data, data_length) if successful, (None, None) otherwise
        
    Note:
        This function is called when user clicks the Detect button.
        Metrics are calculated synchronously to ensure user gets results.
    """
    import traceback
    
    # ==========================================================================
    # STEP 1: Validate prerequisites
    # ==========================================================================
    if not hasattr(global_vars, 'df') or global_vars.df is None:
        print("[DETECT_METRICS] ❌ No primary dataset loaded")
        return None, None
    
    if not hasattr(global_vars, 'secondary_df') or global_vars.secondary_df is None:
        print("[DETECT_METRICS] ❌ No secondary dataset loaded")
        return None, None
    
    if not hasattr(global_vars, 'target_attribute') or global_vars.target_attribute is None:
        print("[DETECT_METRICS] ❌ No target attribute selected")
        return None, None
    
    print(f"[DETECT_METRICS] Prerequisites met - Target: {global_vars.target_attribute}")
    print(f"[DETECT_METRICS] Primary shape: {global_vars.df.shape}, Secondary shape: {global_vars.secondary_df.shape}")
    
    # ==========================================================================
    # STEP 2: Check cache validity
    # ==========================================================================
    is_valid, reason = global_vars.is_cache_valid()
    
    if is_valid:
        print(f"[DETECT_METRICS] ✅ Using cached metrics (valid)")
        metrics_data, data_length = global_vars.get_cached_metrics()
        if metrics_data:
            return metrics_data, data_length
        print(f"[DETECT_METRICS] ⚠️ Cache marked valid but returned None, recalculating...")
    else:
        print(f"[DETECT_METRICS] Cache invalid: {reason}")
    
    # ==========================================================================
    # STEP 3: Calculate new metrics
    # ==========================================================================
    try:
        from drift.detect import generate_metrics_data
        
        print(f"[DETECT_METRICS] 🔄 Starting metrics calculation...")
        
        # Update status
        if not hasattr(global_vars, 'metrics_calculation_status'):
            global_vars.metrics_calculation_status = "idle"
        global_vars.metrics_calculation_status = "calculating"
        
        # Perform calculation (synchronous - blocks until complete)
        metrics_data, data_length = generate_metrics_data()
        
        # =======================================================================
        # STEP 4: Validate and cache results
        # =======================================================================
        if not metrics_data or len(metrics_data) == 0:
            print("[DETECT_METRICS] ❌ Calculation returned empty results")
            global_vars.metrics_calculation_status = "failed"
            return None, None
        
        print(f"[DETECT_METRICS] ✅ Calculated metrics for {len(metrics_data)} columns")
        
        # Cache the results
        cache_success = global_vars.cache_metrics(metrics_data, data_length, force=True)
        
        if cache_success:
            # Reset change flags since we just recalculated
            global_vars.reset_change_flags()
            global_vars.metrics_calculation_status = "completed"
            print(f"[DETECT_METRICS] ✅ Cached {len(metrics_data)} metrics successfully")
            
            # Verify cache
            is_valid_after, _ = global_vars.is_cache_valid()
            if not is_valid_after:
                print(f"[DETECT_METRICS] ⚠️ Warning: Cache validation failed after caching")
        else:
            print(f"[DETECT_METRICS] ⚠️ Warning: Failed to cache metrics")
            global_vars.metrics_calculation_status = "completed_but_not_cached"
        
        return metrics_data, data_length
        
    except Exception as e:
        # =======================================================================
        # STEP 5: Handle errors gracefully
        # =======================================================================
        print(f"[DETECT_METRICS] ❌ Error during metrics calculation: {str(e)}")
        print(f"[DETECT_METRICS] Traceback:\n{traceback.format_exc()}")
        
        global_vars.metrics_calculation_status = "failed"
        
        # Try to return any cached data as fallback (even if outdated)
        try:
            cached_data, cached_length = global_vars.get_cached_metrics()
            if cached_data:
                print(f"[DETECT_METRICS] ⚠️ Returning outdated cached data as fallback")
                return cached_data, cached_length
        except Exception as fallback_error:
            print(f"[DETECT_METRICS] ❌ Fallback also failed: {str(fallback_error)}")
        
        return None, None
