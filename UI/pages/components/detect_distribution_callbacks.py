"""
Distribution Analysis Callbacks for Detect Stage.

This module contains callbacks to handle target distribution chart updates
and conditional distribution analysis in the Detect phase, leveraging shared
utility functions for consistent behavior across Detect and Explain phases.
"""

import dash
from dash import Input, Output, State, callback, html, dcc, ALL
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import Dict, List, Any

from UI.functions.global_vars import global_vars
from UI.state_connector import get_explain_context, record_analysis_path
from UI.shared.components.distribution_analysis_utils import (
    analyze_target_distribution,
    create_distribution_chart_component,
    get_distribution_data
)
from UI.pages.components.explain_utils import (
    rank_attributes,
    analyze_conditional_distribution,
    get_target_values_options
)
from UI.pages.components.explain_component import generate_distribution_chart


def get_target_distribution_data():
    """
    Helper function to retrieve target distribution data for reuse.
    Used both by the chart update callback and the add-to-chat functionality.
    
    Returns:
        tuple: (primary_dist, secondary_dist, column_type, attribute_to_display)
    """
    # Get context and determine which attribute to show
    attribute_to_display = None
    current_stage = global_vars.current_stage
    
    # Priority 1: Use focus attribute if available
    context = get_explain_context(stage=current_stage)
    focus_attribute = context.get('focus_attribute')
    target_attribute = context.get('target_attribute')
    
    if hasattr(global_vars, 'focus_attribute') and global_vars.focus_attribute:
        attribute_to_display = global_vars.focus_attribute
    elif focus_attribute:
        attribute_to_display = focus_attribute
    elif target_attribute:
        attribute_to_display = target_attribute
    elif hasattr(global_vars, 'target_attribute') and global_vars.target_attribute:
        attribute_to_display = global_vars.target_attribute
    
    # Last resort: first column
    if not attribute_to_display and hasattr(global_vars, 'df') and len(global_vars.df.columns) > 0:
        attribute_to_display = global_vars.df.columns[0]
    
    # If we still have no attribute to display, return None values
    if not attribute_to_display or not hasattr(global_vars, 'df') or not hasattr(global_vars, 'secondary_df'):
        return {}, {}, 'categorical', "unknown"
    
    # Analyze distribution of the target attribute
    primary_dist, column_type = analyze_target_distribution(global_vars.df, attribute_to_display)
    secondary_dist, _ = analyze_target_distribution(global_vars.secondary_df, attribute_to_display)
    
    return primary_dist, secondary_dist, column_type, attribute_to_display


@callback(
    Output("target-distribution-chart-container", "children"),
    [Input("metrics-table", "active_cell"),
     Input("metrics-table", "derived_virtual_selected_rows")],
    prevent_initial_call=True
)
def update_target_distribution_chart(active_cell, selected_rows):
    """
    Update the target distribution chart for the Detect stage.
    Always display the originally selected target attribute, regardless of
    clicks in the metrics table. This avoids inadvertently changing the
    definition of the target during analysis.
    
    Args:
        active_cell (dict): Information about the active cell in the metrics table
        selected_rows (list): List of selected row indices
        
    Returns:
        html.Div: The distribution chart component
    """
    # Only process for detect stage
    current_stage = str(global_vars.current_stage).lower()
    if current_stage != "detect":
        return dash.no_update
    
    # Check if we have datasets
    if not hasattr(global_vars, 'df') or not hasattr(global_vars, 'secondary_df'):
        return html.Div("No datasets available", style={"textAlign": "center", "marginTop": "100px"})
    
    # Always show the user-chosen target attribute in Detect stage
    attribute_to_display = getattr(global_vars, 'target_attribute', None)
    
    # Fallbacks: try context target, then first column, otherwise warn
    if not attribute_to_display:
        context = get_explain_context(stage=current_stage)
        attribute_to_display = context.get('target_attribute')
    if not attribute_to_display and hasattr(global_vars, 'df') and len(global_vars.df.columns) > 0:
        attribute_to_display = global_vars.df.columns[0]
    if not attribute_to_display:
        return html.Div("Please choose a target attribute first.", style={"textAlign": "center", "marginTop": "100px"})
    
    # Check if datasets exist
    has_primary = hasattr(global_vars, 'df') and global_vars.df is not None
    has_secondary = hasattr(global_vars, 'secondary_df') and global_vars.secondary_df is not None
    
    if not has_primary or not has_secondary:
        return html.Div(
            "Both primary and secondary datasets are required",
            style={"textAlign": "center", "marginTop": "100px", "color": "#666"}
        )
    
    # Analyze distributions
    primary_dist, column_type = analyze_target_distribution(global_vars.df, attribute_to_display)
    secondary_dist, _ = analyze_target_distribution(global_vars.secondary_df, attribute_to_display)
    
    # Generate chart component with button included
    chart_component = create_distribution_chart_component(
        primary_dist, secondary_dist, column_type, attribute_to_display,
        include_button=True, stage="detect"
    )
    
    # Store distributions in global state for other components to access
    global_vars.explain_primary_dist = primary_dist
    global_vars.explain_secondary_dist = secondary_dist
    
    return chart_component


@callback(
    [Output("detect-top-k-attrs", "data"),
     Output("detect-target-value-dropdown", "options")],
    [Input("metrics-table", "active_cell"),
     Input("target-distribution-chart-container", "children")],
    prevent_initial_call=False  # 允许初始触发
)
def initialize_detect_conditional_analysis(active_cell, target_chart):
    """
    Initialize conditional distribution analysis in the Detect stage.
    
    Args:
        active_cell (dict): Active cell in metrics table
        target_chart (component): Target distribution chart component
    
    Returns:
        tuple: (top_k_attributes, target_value_options)
    """
    # Check trigger conditions and get callback context
    ctx = dash.callback_context
    if not ctx.triggered:
        # No triggers on initial load
        triggered_id = None
    else:
        triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Only process for detect stage
    current_stage = str(global_vars.current_stage).lower()
    if current_stage != "detect":
        return dash.no_update, dash.no_update
    
    # Check if datasets are available
    if not hasattr(global_vars, 'df') or not hasattr(global_vars, 'secondary_df'):
        return [], []
    
    # Get focus attribute from context
    context = get_explain_context(stage=current_stage)
    focus_attribute = context.get('focus_attribute')
    
    # If metrics table has active cell, get attribute from it
    if active_cell and 'row' in active_cell and active_cell.get('column_id') == 'Attribute':
        if hasattr(global_vars, 'metrics_data') and global_vars.metrics_data:
            row_idx = active_cell['row']
            if row_idx < len(global_vars.metrics_data):
                focus_attribute = global_vars.metrics_data[row_idx]['Attribute']
    
    # If no focus attribute defined, use target attribute or first column
    if not focus_attribute:
        if hasattr(global_vars, 'target_attribute') and global_vars.target_attribute:
            focus_attribute = global_vars.target_attribute
        elif hasattr(global_vars, 'df') and len(global_vars.df.columns) > 0:
            focus_attribute = global_vars.df.columns[0]
        else:
            return [], []
    
    # Update focus attribute in global context
    global_vars.analysis_context['current_focus'] = focus_attribute
    
    # Get options for target value dropdown
    target_value_options = get_target_values_options(global_vars.df, focus_attribute)
    
    # Get metrics data containing shift indicators
    metrics_data = []
    if hasattr(global_vars, 'metrics_cache') and global_vars.metrics_cache is not None:
        if isinstance(global_vars.metrics_cache, dict) and 'data' in global_vars.metrics_cache:
            metrics_data = global_vars.metrics_cache['data']
    
    # Calculate top-k attributes (with highest shift scores) using metrics data
    top_k_data = rank_attributes(metrics_data, k=10)
    
    # Record analysis path
    record_analysis_path({
        "action": "detect_conditional_analysis_init",
        "focus_attribute": focus_attribute,
        "trigger": triggered_id
    })
    
    return top_k_data, target_value_options


@callback(
    Output("detect-conditional-chart-container", "children"),
    [Input("detect-target-value-dropdown", "value"),
     Input("detect-compare-attr-dropdown", "value")],
    prevent_initial_call=False
)
def update_detect_conditional_distribution_chart(target_value, compare_attribute):
    """
    Update the conditional distribution chart in the Detect stage.
    
    Args:
        target_value: Selected value of the target attribute
        compare_attribute: Attribute to compare against
        
    Returns:
        html.Div: Conditional distribution chart component
    """
    # Only process in the detect stage
    current_stage = global_vars.current_stage.lower()
    print(f"[DETECT_COND] current_stage={current_stage} | target_value={target_value} | compare_attribute={compare_attribute}")
    if current_stage != "detect":
        return dash.no_update
    
    # Check if both selections are made
    if not target_value or not compare_attribute:
        return html.Div(
            "Please select both a target value and a comparison attribute to view conditional distribution.",
            style={"textAlign": "center", "marginTop": "20px", "color": "#666"}
        )
    
    # Get focus attribute
    focus_attribute = None
    # analysis_context is a dict; use safe dict access instead of hasattr
    if isinstance(global_vars.analysis_context, dict) and global_vars.analysis_context.get('current_focus'):
        focus_attribute = global_vars.analysis_context['current_focus']
    elif hasattr(global_vars, 'target_attribute') and global_vars.target_attribute:
        focus_attribute = global_vars.target_attribute
    
    # If we still don't have a focus attribute but compare_attribute is set, use it
    if not focus_attribute and compare_attribute:
        focus_attribute = compare_attribute
    
    if not focus_attribute:
        return html.Div(
            "No focus attribute selected", 
            style={"textAlign": "center", "marginTop": "20px", "color": "#d9534f"}
        )
    
    # Record analysis operation
    record_analysis_path({
        "action": "detect_conditional_analysis",
        "focus_attribute": focus_attribute,
        "target_value": str(target_value) if target_value is not None else None,
        "compare_attribute": compare_attribute
    })
    
    # Check if datasets exist
    has_primary = hasattr(global_vars, 'df') and global_vars.df is not None
    has_secondary = hasattr(global_vars, 'secondary_df') and global_vars.secondary_df is not None
    
    if not has_primary or not has_secondary:
        return html.Div(
            "Both primary and secondary datasets are required.",
            style={"textAlign": "center", "marginTop": "20px", "color": "#666"}
        )
    
    try:
        # Analyze conditional distributions
        primary_dist, col_type = analyze_conditional_distribution(
            global_vars.df, focus_attribute, target_value, compare_attribute
        )
        
        secondary_dist, _ = analyze_conditional_distribution(
            global_vars.secondary_df, focus_attribute, target_value, compare_attribute
        )
    except Exception as e:
        # Catch and record any errors during analysis
        error_msg = str(e) if str(e) else "Unknown error"
        record_analysis_path({
            "action": "detect_conditional_analysis_error",
            "error": error_msg
        })
        return html.Div(
            f"Error analyzing conditional distribution: {error_msg}", 
            style={"textAlign": "center", "marginTop": "20px", "color": "#d9534f"}
        )
        
    # Generate distribution chart
    fig = generate_distribution_chart(primary_dist, secondary_dist, col_type)
    
    # Update chart title to include conditional information
    fig.update_layout(
        title=f"Distribution of {compare_attribute} when {focus_attribute} = {target_value}",
        title_x=0.5,  # Center the title
        margin=dict(t=60)  # Add more top margin for the title
    )
    
    # Create dcc.Graph component
    chart = dcc.Graph(
        figure=fig,
        config={'displayModeBar': True, 'scrollZoom': True},
        style={"height": "100%"}
    )
    
    return chart


@callback(
    Output("detect-compare-attr-dropdown", "options"),
    [Input("detect-top-k-attrs", "data")],
    prevent_initial_call=False
)
def update_detect_compare_attr_options(top_k_attrs):
    """
    Update the comparison attribute dropdown options based on top-k attributes.
    
    Args:
        top_k_attrs: Data containing top-k attributes ranked by shift magnitude
        
    Returns:
        list: List of dropdown options
    """
    # 只在检测阶段处理
    current_stage = str(global_vars.current_stage).lower()
    if current_stage != "detect":
        return dash.no_update
        
    if not top_k_attrs or len(top_k_attrs) == 0:
        # 如果没有top-k属性数据，则使用所有可用属性
        if hasattr(global_vars, 'df') and global_vars.df is not None:
            # 排除目标属性
            target = global_vars.target_attribute if hasattr(global_vars, 'target_attribute') else None
            attrs = [col for col in global_vars.df.columns if col != target]
            options = [{'label': attr, 'value': attr} for attr in attrs]
            return options
        return []
    
    # Handle different data formats from rank_attributes function
    attrs = []
    print(f"[DETECT ATTRS] Processing top_k_attrs: {type(top_k_attrs)}, length: {len(top_k_attrs)}")
    
    for i, attr in enumerate(top_k_attrs):
        print(f"[DETECT ATTRS] Item {i}: {type(attr)} = {attr}")
        if isinstance(attr, str):
            # If attr is a string (new format from rank_attributes)
            attrs.append(attr)
        elif isinstance(attr, dict) and attr.get('attribute'):
            # If attr is a dict with 'attribute' key (legacy format)
            attrs.append(attr.get('attribute'))
    
    # Filter out None and empty values
    attrs = [attr for attr in attrs if attr]
    print(f"[DETECT ATTRS] Final attrs list: {attrs}")
    
    # 跟踪分析路径
    record_analysis_path({
        "action": "detect_compare_options_update",
        "num_options": len(attrs)
    })
    
    # 创建下拉选项
    options = [{'label': attr, 'value': attr} for attr in attrs]
    
    return options


# Set default selections for detect dropdowns once options are available
@callback(
    [Output("detect-target-value-dropdown", "value", allow_duplicate=True),
     Output("detect-compare-attr-dropdown", "value", allow_duplicate=True)],
    [Input("detect-target-value-dropdown", "options"),
     Input("detect-compare-attr-dropdown", "options")],
    prevent_initial_call=True
)
def set_detect_default_values(target_options, compare_options):
    # Only set defaults when options are available; otherwise, keep current
    target_value = target_options[0]['value'] if target_options and len(target_options) > 0 else dash.no_update
    compare_value = compare_options[0]['value'] if compare_options and len(compare_options) > 0 else dash.no_update
    return target_value, compare_value
