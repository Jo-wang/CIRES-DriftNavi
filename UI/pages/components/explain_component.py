"""
Explain Component for driftNavi.

This module provides components and functionality for explaining distribution shifts
between datasets and providing GPT-powered analysis for training and adaptation strategies.
It also includes correlation analysis and root cause analysis for distribution shifts.
"""

import dash
from dash import html, dcc, callback, Input, Output, State, ALL, MATCH
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import time
import datetime
from datetime import datetime

# Import project modules
from UI.functions.global_vars import global_vars
from UI.state_connector import get_explain_context, record_analysis_path, update_target_relationship
from UI.shared.components.metrics_heatmap import create_metrics_heatmap
from UI.callback.correlation_analysis import generate_correlation_shift_ui

# Removed legacy prompt imports - now using unified prompt manager
from UI.pages.components.explain_utils import rank_attributes, analyze_conditional_distribution, get_target_values_options
from agent import generate_response_from_prompt

# Import new analysis components
from UI.pages.components.gpt_severity_analyzer import gpt_severity_analyzer
from UI.pages.components.context_item_boxes import create_context_item_boxes, create_context_item_modal
from UI.pages.components.context_item_boxes_static import create_static_context_item_boxes
from UI.pages.components.joint_analysis import create_joint_analysis_component
from UI.pages.components.strategy_selection import create_strategy_selection_component

# Add CSS styles for new explain interface
EXPLAIN_CSS_STYLES = """
<style>
/* Context item cards hover effects */
.context-item-card:hover {
    background-color: #f8f9fa !important;
    border-color: #614385 !important;
    box-shadow: 0 2px 8px rgba(97, 67, 133, 0.15) !important;
    transform: translateY(-1px);
}

.context-item-card.selected {
    background-color: #e7e3f0 !important;
    border-color: #614385 !important;
    border-width: 2px !important;
    box-shadow: 0 3px 12px rgba(97, 67, 133, 0.2) !important;
}

/* Expanded context item styling */
.context-item-expanded {
    border-color: #614385 !important;
    box-shadow: 0 3px 12px rgba(97, 67, 133, 0.15) !important;
}

/* Context type header clickable styling */
.context-type-header:hover {
    background-color: rgba(97, 67, 133, 0.1) !important;
}

/* Analysis panel styling */
.analysis-panel {
    border-left: 3px solid #614385;
}

/* Analysis button hover effects */
.analysis-type-btn:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}

/* Context group headers */
.context-group-header {
    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    border-bottom: 2px solid;
}

/* Loading animations */
.context-loading {
    text-align: center;
    padding: 40px;
}

/* Scrollbar styling for panels */
.explain-panel::-webkit-scrollbar {
    width: 6px;
}

.explain-panel::-webkit-scrollbar-track {
    background: #f1f1f1;
    border-radius: 3px;
}

.explain-panel::-webkit-scrollbar-thumb {
    background: #614385;
    border-radius: 3px;
}

.explain-panel::-webkit-scrollbar-thumb:hover {
    background: #4a2c6b;
}

/* Analysis results styling */
.analysis-results {
    background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
    border-radius: 8px;
    padding: 20px;
    box-shadow: inset 0 1px 3px rgba(0,0,0,0.1);
}

/* Context type indicators */
.context-type-indicator {
    display: inline-block;
    width: 12px;
    height: 12px;
    border-radius: 50%;
    margin-right: 8px;
}

/* Responsive adjustments */
@media (max-width: 768px) {
    .explain-panels-row {
        flex-direction: column;
    }
    
    .explain-panel {
        height: auto !important;
        max-height: 400px;
        margin-bottom: 20px;
    }
}
</style>
"""

# Create a global store for state management
class GlobalStore:
    """
    Global store for maintaining state in the Explain component.
    """
    def __init__(self):
        self.active_button = None
        self.primary_dist = None
        self.secondary_dist = None
        self.distribution_col_type = None

# Initialize global store
global_store = GlobalStore()


def generate_distribution_chart(primary_dist, secondary_dist, col_type):
    """
    Generate a plotly figure to visualize the distribution comparison between
    primary and secondary datasets for a given column type.
    
    Args:
        primary_dist (dict/array): Distribution data for primary dataset
        secondary_dist (dict/array): Distribution data for secondary dataset
        col_type (str): Column type, either 'categorical' or 'continuous'
        
    Returns:
        plotly.graph_objs.Figure: Distribution comparison chart
    """
    # Create figure with appropriate layout
    fig = go.Figure()
    
    # Define colors for the datasets
    primary_color = "#516395"  # Blue
    secondary_color = "#614385"  # Purple
    
    if col_type == "categorical":
        # For categorical data, create bar charts
        
        # Extract categories and frequencies
        categories = list(primary_dist.keys())
        primary_values = [primary_dist.get(cat, 0) for cat in categories]
        secondary_values = [secondary_dist.get(cat, 0) for cat in categories]
        
        # Add primary dataset bars
        fig.add_trace(go.Bar(
            x=categories,
            y=primary_values,
            name="Primary Dataset",
            marker_color=primary_color,
            opacity=0.7
        ))
        
        # Add secondary dataset bars
        fig.add_trace(go.Bar(
            x=categories,
            y=secondary_values,
            name="Secondary Dataset",
            marker_color=secondary_color,
            opacity=0.7
        ))
        
        # Update layout for categorical data
        fig.update_layout(
            barmode="group",
            xaxis_title="Categories",
            yaxis_title="Frequency",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=40, t=60, b=40),
            height=400
        )
        
    elif col_type == "continuous":
        # For continuous data, create histograms or KDE plots
        
        # Extract data for histograms
        primary_values = primary_dist
        secondary_values = secondary_dist
        
        # Add primary dataset histogram
        fig.add_trace(go.Histogram(
            x=primary_values,
            name="Primary Dataset",
            marker_color=primary_color,
            opacity=0.7,
            histnorm="probability density"
        ))
        
        # Add secondary dataset histogram
        fig.add_trace(go.Histogram(
            x=secondary_values,
            name="Secondary Dataset",
            marker_color=secondary_color,
            opacity=0.7,
            histnorm="probability density"
        ))
        
        # Update layout for continuous data
        fig.update_layout(
            barmode="overlay",
            xaxis_title="Value",
            yaxis_title="Density",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=40, t=60, b=40),
            height=400
        )
    
    # Apply common styling
    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Arial", size=12),
        hoverlabel=dict(font_size=14, font_family="Arial"),
        hovermode="closest"
    )
    
    return fig

# COMMENTED: Using chat-based explain workflow - main page explain component no longer needed
# def create_explain_component():
#     """
#     Create the NEW Explain component layout based on chat context analysis.
#     
#     Layout structure:
#     - Header: Title and context info
#     - Main content: Left panel (context items) + Right panel (analysis)
#     - Footer: Root Cause Analysis (preserved)
#     
#     Returns:
#         html.Div: The main container for the Explain component
#     """
#     # Get context from Detect phase
#     explain_context = get_explain_context()
#     focus_attribute = explain_context.get('focus_attribute')
#     target_attribute = explain_context.get('target_attribute')
#     metrics_data = explain_context.get('metrics_data', [])
    
#     # Record transition to explain phase
#     record_analysis_path('enter_explain_phase', {
#         'focus_attribute': focus_attribute,
#         'target_attribute': target_attribute
#     })
#     
#     # Log for debugging
#     print(f"Creating NEW explain component with context: focus={focus_attribute}, target={target_attribute}")
#     
#     # Updated Store components for new architecture
#     stores = [
#         # Original stores (kept for compatibility)
#         dcc.Store(id="explain-focus-attribute", data=focus_attribute),
#         dcc.Store(id="explain-primary-dist", data={}),
#         dcc.Store(id="explain-secondary-dist", data={}),
#         dcc.Store(id="explain-col-type", data=""),
#         dcc.Store(id="explain-top-k-attrs", data=[]),
#         dcc.Store(id="explain-conditional-primary-dist", data={}),
#         dcc.Store(id="explain-conditional-secondary-dist", data={}),
#         
#         # NEW stores for unified context-based analysis
#         dcc.Store(id="explain-selected-context", data=None),        # Legacy - kept for compatibility
#         dcc.Store(id="explain-analysis-type", data=None),           # Legacy - kept for compatibility
#         dcc.Store(id="explain-analysis-results", data={}),          # Legacy - kept for compatibility
#         dcc.Store(id="explain-context-groups", data={}),            # Legacy - kept for compatibility
#         dcc.Store(id="explain-context-expanded-states", data={}),   # Legacy - kept for compatibility
#         
#         # UNIFIED ANALYSIS stores
#         dcc.Store(id="unified-selected-strategy", data="monitor"),   # Selected strategy (retrain/finetune/monitor)
#         dcc.Store(id="unified-strategy-analysis", data={}),          # Comprehensive strategy analysis
#         dcc.Store(id="unified-perspective-state", data="technical"), # Current perspective (technical/business/executive)
#         dcc.Store(id="unified-analysis-cache", data={}),             # Cache for GPT analysis results
#     ]
#     
#     return html.Div([
#         # Store components for state management
#         *stores,
#         
#         # Toast notifications
#         dbc.Toast(
#             "Context item deleted successfully!",
#             id="context-delete-success-toast",
#             header="Success",
#             is_open=False,
#             dismissable=True,
#             icon="success",
#             duration=3000,
#             style={"position": "fixed", "top": 66, "right": 10, "width": 350, "zIndex": 9999}
#         ),
#         
#         # Page header with context info
#         html.Div([
#             html.H2(
#                 ["Explain Distribution Shifts", html.Small("Context-based Analysis", className="text-muted ms-3")],
#                 className="mb-1"
#             ),
#             
#             # Show context information
#             html.Div([
#                 html.Span(["Target Attribute: ", html.Strong(target_attribute or "Not set")], 
#                          className="me-4"),
#                 html.Span(["Analysis Stage: ", html.Strong("Explain")],
#                          className="badge", style={"backgroundColor": "#614385", "color": "white"}),
#             ], className="text-muted small mb-3"),
#         ], className="mb-4"),
#         
#         # Return to Detect button
#         html.Div([
#             dbc.Button(
#                 [html.I(className="fas fa-arrow-left me-2"), "Back to Detect"],
#                 id="back-to-detect-btn",
#                 color="light",
#                 size="sm",
#                 className="mb-3",
#                 style={"border": "1px solid #ddd"}
#             )
#         ], className="text-end mb-3"),
#         
#         # Main content area: Left panel + Right panel
#         dbc.Row([
#             # Left Panel: Context Items (40% width)
#             dbc.Col([
#                 html.Div([
#                     html.H4([
#                         html.I(className="fas fa-layer-group me-2", style={"color": "#614385"}),
#                         "Context Items"
#                     ], className="mb-3", style={"color": "#614385"}),
#                     
#                     # Context items will be loaded here
#                     dcc.Loading(
#                         id="explain-context-loading",
#                         type="circle",
#                         children=html.Div(id="explain-context-items-panel")
#                     )
#                 ], 
#                 style={
#                     "backgroundColor": "#f8f9fa",
#                     "padding": "15px",
#                     "borderRadius": "8px",
#                     "border": "1px solid #dee2e6",
#                     "minHeight": "400px",
#                     "maxHeight": "80vh",
#                     "overflowY": "auto"
#                 })
#             ], xs=12, sm=12, md=5, lg=4, xl=4),
#             
#             # Right Panel: Analysis Panel (60% width)
#             dbc.Col([
#                 html.Div([
#                     html.H4([
#                         html.I(className="fas fa-chart-line me-2", style={"color": "#516395"}),
#                         "Analysis Panel"
#                     ], className="mb-3", style={"color": "#516395"}),
#                     
#                     # Analysis content will be loaded here
#                     dcc.Loading(
#                         id="explain-analysis-panel-loading",
#                         type="circle",
#                         children=html.Div(id="explain-analysis-panel")
#                     )
#                 ],
#                 className="explain-analysis-panel",
#                 style={
#                     "backgroundColor": "#ffffff",
#                     "padding": "15px", 
#                     "borderRadius": "8px",
#                     "border": "1px solid #dee2e6",
#                     "minHeight": "400px",
#                     "maxHeight": "80vh",
#                     "overflowY": "auto"
#                 })
#             ], xs=12, sm=12, md=7, lg=8, xl=8)
#                  ], className="g-3 explain-main-row"),  # 添加gap类来控制列间距
#     ], id="explain-component-container")


def analyze_target_distribution(df, target_column):
    """
    Analyze the distribution of a target attribute in a dataset.
    
    Args:
        df (pd.DataFrame): Dataset to analyze
        target_column (str): Name of the target column
        
    Returns:
        tuple: (distribution_dict, column_type)
    """
    if df is None or target_column not in df.columns:
        return {}, "unknown"
    
    # Get the column data
    column_data = df[target_column]
    
    # Determine column type
    if pd.api.types.is_numeric_dtype(column_data):
        # For numeric data, consider it continuous if it has more than 10 unique values
        # and is not a boolean type
        if column_data.dtype == bool or len(column_data.unique()) <= 10:
            column_type = "categorical"
        else:
            column_type = "continuous"
    else:
        # Non-numeric data is always categorical
        column_type = "categorical"
    
    # Calculate distribution based on type
    if column_type == "categorical":
        # For categorical data, count occurrences of each value
        distribution = column_data.value_counts().to_dict()
    else:
        # For continuous data, create bins and count values in each bin
        hist, bin_edges = np.histogram(column_data.dropna(), bins=10)
        
        # Create distribution with bin ranges as keys
        distribution = {}
        for i in range(len(hist)):
            bin_label = f"{bin_edges[i]:.2f} to {bin_edges[i+1]:.2f}"
            distribution[bin_label] = int(hist[i])
    
    return distribution, column_type


def generate_distribution_chart(primary_dist, secondary_dist, column_type):
    """
    Generate a distribution chart comparing primary and secondary datasets.
    
    Args:
        primary_dist (dict): Distribution of values in primary dataset
        secondary_dist (dict): Distribution of values in secondary dataset
        column_type (str): Type of the column ('categorical' or 'continuous')
        
    Returns:
        go.Figure: Plotly figure object
    """
    if not primary_dist and not secondary_dist:
        # Return empty figure if no data
        return go.Figure().update_layout(
            title="No distribution data available",
            xaxis_title="Value",
            yaxis_title="Count",
            template="plotly_white"
        )
    
    # Combine keys from both distributions
    all_keys = sorted(set(primary_dist.keys()) | set(secondary_dist.keys()))
    
    # Prepare data for plotting
    x_values = list(all_keys)
    y_primary = [primary_dist.get(key, 0) for key in all_keys]
    y_secondary = [secondary_dist.get(key, 0) for key in all_keys]
    
    # Create figure based on column type
    if column_type == "categorical":
        # Bar chart for categorical data
        fig = go.Figure()
        
        # Add primary dataset bars
        fig.add_trace(go.Bar(
            x=x_values,
            y=y_primary,
            name="Primary Dataset",
            marker_color="#614385"
        ))
        
        # Add secondary dataset bars
        fig.add_trace(go.Bar(
            x=x_values,
            y=y_secondary,
            name="Secondary Dataset",
            marker_color="#516395"
        ))
        
        # Update layout
        fig.update_layout(
            barmode="group",
            xaxis_title="Value",
            yaxis_title="Count",
            legend_title="Dataset",
            template="plotly_white"
        )
    else:
        # Line chart for continuous data
        fig = go.Figure()
        
        # Add primary dataset line
        fig.add_trace(go.Scatter(
            x=x_values,
            y=y_primary,
            mode="lines+markers",
            name="Primary Dataset",
            line=dict(color="#614385", width=3),
            marker=dict(size=8)
        ))
        
        # Add secondary dataset line
        fig.add_trace(go.Scatter(
            x=x_values,
            y=y_secondary,
            mode="lines+markers",
            name="Secondary Dataset",
            line=dict(color="#516395", width=3),
            marker=dict(size=8)
        ))
        
        # Update layout
        fig.update_layout(
            xaxis_title="Value Range",
            yaxis_title="Count",
            legend_title="Dataset",
            template="plotly_white"
        )
    
    # Improve aesthetics
    fig.update_layout(
        margin=dict(l=40, r=40, t=40, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig


# ============================================================================
# Callbacks
# ============================================================================

# OLD CALLBACK - DISABLED FOR NEW CONTEXT-BASED ARCHITECTURE
# @callback(
#     [Output("explain-top-k-attrs", "data"),
#      Output("explain-target-value-dropdown", "options")],
#     [Input("current-stage", "children")],
#     prevent_initial_call=True
# )
def initialize_explain_conditional_analysis_OLD(current_stage):
    """
    Initialize the conditional analysis by calculating top-k shifted attributes
    and setting up the target value dropdown options.
    
    Args:
        current_stage (str): Current active stage
        
    Returns:
        tuple: (top_k_attributes, target_value_options)
    """
    # Process for both detect and explain modes
    if current_stage not in ["detect", "explain"]:
        return dash.no_update, dash.no_update
    
    # Check if target attribute is set
    if not hasattr(global_vars, 'target_attribute') or not global_vars.target_attribute:
        return [], []
    
    # Check if datasets exist
    has_primary = hasattr(global_vars, 'df') and global_vars.df is not None
    has_secondary = hasattr(global_vars, 'secondary_df') and global_vars.secondary_df is not None
    
    if not has_primary or not has_secondary:
        return [], []
    
    # Generate metrics data manually instead of calling generate_metrics_data
    # This avoids the circular import
    metrics_data = []
    
    # Get common columns between datasets
    if has_primary and has_secondary:
        common_cols = [col for col in global_vars.df.columns if col in global_vars.secondary_df.columns]
        
        # For each common column, calculate simple shift metrics
        for col in common_cols:
            # Skip the target attribute itself
            if col == global_vars.target_attribute:
                continue
                
            # Create a simple metric entry with JS divergence estimated from distribution differences
            try:
                # Calculate a simple shift metric (difference in mean/distribution)
                primary_vals = global_vars.df[col].value_counts(normalize=True).to_dict()
                secondary_vals = global_vars.secondary_df[col].value_counts(normalize=True).to_dict()
                
                # Calculate difference in distributions
                total_diff = 0
                for val in set(list(primary_vals.keys()) + list(secondary_vals.keys())):
                    p_val = primary_vals.get(val, 0)
                    s_val = secondary_vals.get(val, 0)
                    total_diff += abs(p_val - s_val)
                
                # Add to metrics data
                metrics_data.append({
                    "Attribute": col,
                    "JS_Divergence": total_diff / 2,  # Simple approximation
                    "PSI": total_diff,
                    "Wasserstein": total_diff * 10,  # Simple approximation
                    # "Test_Statistic": total_diff,
                    "p_value": max(0.001, 1 - total_diff)  # Lower p-value means more significant shift
                })
            except Exception:
                # Skip columns that cause errors in calculation
                continue
    
    # Get top-k attributes
    top_k = 5  # Number of top attributes to consider
    top_attrs = rank_attributes(metrics_data, k=top_k)
    
    # Get target attribute value options
    target_options = get_target_values_options(global_vars.df, global_vars.target_attribute)
    
    return top_attrs, target_options


# OLD CALLBACK - DISABLED FOR NEW CONTEXT-BASED ARCHITECTURE
# @callback(
#     Output("explain-shifted-attr-dropdown", "options"),
#     [Input("explain-top-k-attrs", "data")],
#     prevent_initial_call=True
# )
def update_shifted_attribute_options_OLD(top_k_attrs):
    """
    Update the shifted attribute dropdown options based on top-k attributes.
    
    Args:
        top_k_attrs (list): List of top-k attribute names
        
    Returns:
        list: Dropdown options for shifted attributes
    """
    if not top_k_attrs:
        return []
    
    # Create options from top-k attribute names
    return [{'label': attr, 'value': attr} for attr in top_k_attrs]


# OLD CALLBACK - DISABLED FOR NEW CONTEXT-BASED ARCHITECTURE
# @callback(
#     [Output("explain-conditional-chart", "children"),
#      Output("explain-conditional-primary-dist", "data"),
#      Output("explain-conditional-secondary-dist", "data")],
#     [Input("explain-target-value-dropdown", "value"),
#      Input("explain-shifted-attr-dropdown", "value")],
#     prevent_initial_call=True
# )
def update_conditional_distribution_chart_OLD(target_value, shifted_attr):
    """
    Update the conditional distribution chart based on selected target value and shifted attribute.
    Also store the distributions for later analysis.
    
    Args:
        target_value (str): Selected target attribute value
        shifted_attr (str): Selected shifted attribute
        
    Returns:
        tuple: (chart_component, primary_dist, secondary_dist)
    """
    # Check if both selections are made
    if not target_value or not shifted_attr:
        return html.Div(
            "Select both a target value and a shifted attribute to view conditional distribution.",
            style={"textAlign": "center", "marginTop": "100px", "color": "#666"}
        ), {}, {}
    
    # Check if target attribute is set
    if not hasattr(global_vars, 'target_attribute') or not global_vars.target_attribute:
        return html.Div(
            "Target attribute not set.",
            style={"textAlign": "center", "marginTop": "100px", "color": "#666"}
        ), {}, {}
    
    # Get target attribute
    target_attr = global_vars.target_attribute
    
    # Check if datasets exist
    has_primary = hasattr(global_vars, 'df') and global_vars.df is not None
    has_secondary = hasattr(global_vars, 'secondary_df') and global_vars.secondary_df is not None
    
    if not has_primary or not has_secondary:
        return html.Div(
            "Both primary and secondary datasets are required.",
            style={"textAlign": "center", "marginTop": "100px", "color": "#666"}
        ), {}, {}
    
    # Analyze conditional distributions
    primary_dist, col_type = analyze_conditional_distribution(
        global_vars.df, target_attr, target_value, shifted_attr
    )
    
    secondary_dist, _ = analyze_conditional_distribution(
        global_vars.secondary_df, target_attr, target_value, shifted_attr
    )
    
    # Generate distribution chart
    fig = generate_distribution_chart(primary_dist, secondary_dist, col_type)
    
    # Update chart title to include conditional information
    fig.update_layout(
        title=f"Distribution of {shifted_attr} when {target_attr} = {target_value}",
        title_x=0.5,  # Center the title
        margin=dict(t=60)  # Add more top margin for the title
    )
    
    # Create dcc.Graph component
    chart = dcc.Graph(
        figure=fig,
        config={'displayModeBar': True, 'scrollZoom': True},
        style={"height": "100%"}
    )
    
    # Store the shifted attribute type as metadata in the distributions
    primary_dist_with_meta = {"data": primary_dist, "type": col_type}
    secondary_dist_with_meta = {"data": secondary_dist, "type": col_type}
    
    return chart, primary_dist_with_meta, secondary_dist_with_meta


# OLD CALLBACK - DISABLED FOR NEW CONTEXT-BASED ARCHITECTURE
# @callback(
#     Output("explain-conditional-analysis-content", "children"),
#     [Input("explain-conditional-analyze-button", "n_clicks")],
#     [State("explain-target-value-dropdown", "value"),
#      State("explain-shifted-attr-dropdown", "value"),
#      State("explain-conditional-primary-dist", "data"),
#      State("explain-conditional-secondary-dist", "data")],
#     prevent_initial_call=True
# )
def update_conditional_analysis_OLD(analyze_clicks, target_value, shifted_attr, primary_dist_data, secondary_dist_data):
    """
    Generate GPT analysis for the conditional distribution shift using stored distribution data.
    
    Args:
        analyze_clicks (int): Number of clicks on the analyze button
        target_value (str): Selected target attribute value
        shifted_attr (str): Selected shifted attribute
        primary_dist_data (dict): Primary dataset conditional distribution data with metadata
        secondary_dist_data (dict): Secondary dataset conditional distribution data with metadata
        
    Returns:
        html.Div: Analysis content
    """
    # Check if button was clicked
    if not analyze_clicks:
        return dash.no_update
    
    # Check if selections are made
    if not target_value or not shifted_attr:
        return html.Div([
            html.I(className="fas fa-exclamation-triangle me-2", style={"color": "#ff9800"}),
            "Please select both a target value and a shifted attribute first."
        ])
    
    # Check if distributions are available
    if not primary_dist_data or not secondary_dist_data:
        return html.Div([
            html.I(className="fas fa-exclamation-triangle me-2", style={"color": "#ff9800"}),
            "Distribution data is not available. Please select a value and attribute first."
        ])
    
    # Check if target attribute is set
    if not hasattr(global_vars, 'target_attribute') or not global_vars.target_attribute:
        return html.Div([
            html.I(className="fas fa-exclamation-triangle me-2", style={"color": "#ff9800"}),
            "Target attribute is not set."
        ])
    
    # Get target attribute and distributions from stored data
    target_attr = global_vars.target_attribute
    primary_dist = primary_dist_data.get("data", {})
    secondary_dist = secondary_dist_data.get("data", {})
    shifted_attr_type = primary_dist_data.get("type", "unknown")
    
    # Generate prompt for GPT using unified prompt manager
    from .prompt_manager import prompt_manager
    
    prompt_context = {
        'target_name': target_attr,
        'target_value': target_value,
        'shifted_attr_name': shifted_attr,
        'shifted_attr_type': shifted_attr_type,
        'primary_dist': primary_dist,
        'secondary_dist': secondary_dist
    }
    
    prompt = prompt_manager.generate_prompt('conditional_analysis', prompt_context)
    
    # Get response from GPT
    try:
        response = generate_response_from_prompt(prompt)
        
        # Format the response
        return html.Div([
            html.H5([
                html.I(className="fas fa-chart-bar me-2", style={"color": "#516395"}),
                f"Conditional Analysis: {target_attr} = {target_value} → {shifted_attr}"
            ], className="mb-3"),
            html.Hr(),
            # Use Markdown to properly render the response
            dcc.Markdown(
                response,
                className="explain-markdown-content",
                style={
                    "lineHeight": "1.6",
                    "fontSize": "16px",
                    "padding": "10px",
                    "backgroundColor": "#f9f9f9",
                    "borderRadius": "5px",
                    "border": "1px solid #eee"
                }
            )
        ])
    except Exception as e:
        # Handle errors
        return html.Div([
            html.I(className="fas fa-exclamation-circle me-2", style={"color": "red"}),
            f"Error generating analysis: {str(e)}"
        ], className="text-danger")


# OLD CALLBACK - DISABLED FOR NEW CONTEXT-BASED ARCHITECTURE
# @callback(
#     [Output("explain-distribution-chart", "children"),
#      Output("explain-primary-dist", "data"),
#      Output("explain-secondary-dist", "data")],
#     [Input("current-stage", "children"),
#      Input("explain-focus-attribute", "data")],
#     prevent_initial_call=True
# )
def update_explain_chart_OLD(current_stage, focus_attribute):
    """
    Update the distribution chart when entering explain mode.
    
    Args:
        current_stage (str): Current pipeline stage
        focus_attribute (str): Attribute to focus on from the Detect phase
        
    Returns:
        tuple: Chart component, primary distribution data, secondary distribution data
    """
    # Check if we're in detect or explain stage
    if current_stage not in ["detect", "explain"]:
        return dash.no_update, dash.no_update, dash.no_update
    
    # Check if we have datasets
    if not hasattr(global_vars, 'df') or not hasattr(global_vars, 'secondary_df'):
        return "No datasets available", {}, {}
    
    # Get context from Detect phase
    explain_context = get_explain_context()
    context_focus_attr = explain_context.get('focus_attribute')
    
    # Determine which attribute to show (priority: passed focus attribute > context focus attr > target > first column)
    dropdown_value = None
    
    # First priority: explicitly passed focus attribute (from Detect table)
    if focus_attribute:
        dropdown_value = focus_attribute
    # Second priority: focus attribute from context
    elif context_focus_attr:
        dropdown_value = context_focus_attr
    # Third priority: target attribute
    elif hasattr(global_vars, 'target_attribute') and global_vars.target_attribute:
        dropdown_value = global_vars.target_attribute
    # Last resort: first attribute in dataset
    elif hasattr(global_vars, 'df') and global_vars.df is not None:
        dropdown_value = global_vars.df.columns[0] if len(global_vars.df.columns) > 0 else None
    
    # Check if datasets exist
    has_primary = hasattr(global_vars, 'df') and global_vars.df is not None
    has_secondary = hasattr(global_vars, 'secondary_df') and global_vars.secondary_df is not None
    
    if not has_primary or not has_secondary:
        return html.Div(
            "Both primary and secondary datasets are required",
            style={"textAlign": "center", "marginTop": "100px", "color": "#666"}
        ), {}, {}
    
    # Analyze distributions
    primary_dist, column_type = analyze_target_distribution(global_vars.df, dropdown_value)
    
    # Check for secondary dataset
    if has_secondary:
        secondary_dist, _ = analyze_target_distribution(global_vars.secondary_df, dropdown_value)
    else:
        secondary_dist = {}
    
    # Generate chart
    fig = generate_distribution_chart(primary_dist, secondary_dist, column_type)
    
    # Create dcc.Graph component
    chart = dcc.Graph(
        figure=fig,
        config={'displayModeBar': True, 'scrollZoom': True},
        style={"height": "100%"}
    )
    
    return chart, primary_dist, secondary_dist


# OLD CALLBACK - DISABLED FOR NEW CONTEXT-BASED ARCHITECTURE
# @callback(
#     [Output("explain-analysis-content", "children"),
#      Output("explain-active-button", "data")],
#     [Input("explain-train-button", "n_clicks"),
#      Input("explain-adapt-button", "n_clicks")],
#     [State("explain-primary-dist", "data"),
#      State("explain-secondary-dist", "data")],
#     prevent_initial_call=True
# )
def update_explain_analysis_OLD(train_clicks, adapt_clicks, primary_dist, secondary_dist):
    """
    Generate GPT analysis when train or adapt button is clicked.
    
    Args:
        train_clicks (int): Number of clicks on train button
        adapt_clicks (int): Number of clicks on adapt button
        primary_dist (dict): Primary dataset distribution
        secondary_dist (dict): Secondary dataset distribution
        
    Returns:
        html.Div: Analysis content
    """
    # Determine which button was clicked
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Check if distributions are available
    if not primary_dist or not secondary_dist:
        return html.Div([
            html.I(className="fas fa-exclamation-triangle me-2", style={"color": "#ff9800"}),
            "Distribution data is not available. Make sure both datasets are loaded and a target attribute is selected."
        ]), button_id  # 返回错误消息和当前按钮ID
    
    # Get target attribute info
    target_name = getattr(global_vars, 'target_attribute', 'Unknown')
    
    # Determine column type
    column_type = "categorical"
    if global_vars.df is not None and target_name in global_vars.df.columns:
        column = global_vars.df[target_name]
        if pd.api.types.is_numeric_dtype(column) and len(column.unique()) > 10:
            column_type = "continuous"
    
    # Generate prompt based on button clicked using unified prompt manager
    from .prompt_manager import prompt_manager
    
    prompt_context = {
        'target_name': target_name,
        'target_type': column_type,
        'primary_dist': primary_dist,
        'secondary_dist': secondary_dist
    }
    
    if button_id == "explain-train-button":
        prompt = prompt_manager.generate_prompt('train_analysis', prompt_context)
        title = "Training Impact Analysis"
        icon = "fas fa-graduation-cap"
        color = "#614385"
    else:  # adapt button
        prompt = prompt_manager.generate_prompt('adapt_analysis', prompt_context)
        title = "Adaptation Needs Analysis"
        icon = "fas fa-exchange-alt"
        color = "#516395"
    
    # Get response from GPT
    try:
        response = generate_response_from_prompt(prompt)
        
        # Format the response
        return html.Div([
            html.H5([
                html.I(className=f"{icon} me-2", style={"color": color}),
                title
            ], className="mb-3"),
            html.Hr(),
            # Use dcc.Markdown to render the response
            dcc.Markdown(
                response,
                className="explain-markdown-content",
                style={
                    "lineHeight": "1.6",
                    "fontSize": "16px",
                    "padding": "10px",
                    "backgroundColor": "#f9f9f9",
                    "borderRadius": "5px",
                    "border": "1px solid #eee"
                }
            )
        ]), button_id  # return analysis content and current button ID
    except Exception as e:
        # Handle errors
        return html.Div([
            html.I(className="fas fa-exclamation-circle me-2", style={"color": "red"}),
            f"Error generating analysis: {str(e)}"
        ], className="text-danger"), button_id  # return error message and current button ID


# OLD CALLBACK - DISABLED FOR NEW CONTEXT-BASED ARCHITECTURE  
# @callback(
#     [
#         Output("explain-train-button", "style"),
#         Output("explain-adapt-button", "style"),
#         Output("explain-train-button", "outline"),
#         Output("explain-adapt-button", "outline"),
#     ],
#     [Input("explain-active-button", "data")],
#     prevent_initial_call=True
# )
def update_button_highlight_OLD(active_button):
    """
    Update button styles to highlight the currently active button.
    
    Args:
        active_button (str): The ID of the currently active button
    
    Returns:
        tuple: The styles and outline properties of the two buttons
    """
    # Default style - no highlight
    train_style = {}
    adapt_style = {}
    
    # Default outline - default with outline (outline=True means no fill)
    train_outline = True
    adapt_outline = True
    
    # Update styles based on active button
    if active_button == "explain-train-button":
        # When the training button is selected
        train_style = {"boxShadow": "0 0 8px rgba(97, 67, 133, 0.7)", "transform": "scale(1.03)"}
        train_outline = False  # No outline, button filled
    elif active_button == "explain-adapt-button":
        # When the adaptation button is selected
        adapt_style = {"boxShadow": "0 0 8px rgba(81, 99, 149, 0.7)", "transform": "scale(1.03)"}
        adapt_outline = False  # No outline, button filled
    
    return train_style, adapt_style, train_outline, adapt_outline








# ===== NEW FUNCTIONS FOR CONTEXT-BASED ANALYSIS =====

def create_context_items_panel(context_data):
    """
    Create the left panel showing context items with simple expand/delete functionality.
    Uses CSS-based expansion and simplified delete callbacks.
    """
    if not context_data:
        return html.Div([
            html.Div([
                html.I(className="fas fa-info-circle me-2", style={"color": "#6c757d", "fontSize": "2rem"}),
                html.H5("No Context Items", style={"color": "#6c757d"}),
                html.P("Context items will appear here as you interact with the system", className="text-muted")
            ], className="text-center", style={"padding": "60px 20px"})
        ])
    
    # Group items by type
    grouped_items = {}
    for item in context_data:
        item_type = item.get("type", "unknown")
        if item_type not in grouped_items:
            grouped_items[item_type] = []
        grouped_items[item_type].append(item)
    
    # Type display names and colors
    type_config = {
        "target_distribution": {
            "name": "Target Distribution", 
            "icon": "fas fa-chart-bar", 
            "color": "#614385"
        },
        "conditional_distribution": {
            "name": "Conditional Distribution", 
            "icon": "fas fa-project-diagram", 
            "color": "#e67e22"
        },
        "distribution_comparison": {
            "name": "Distribution Comparison", 
            "icon": "fas fa-balance-scale", 
            "color": "#27ae60"
        },
        "metric": {
            "name": "Statistical Metrics", 
            "icon": "fas fa-calculator", 
            "color": "#3498db"
        },
        "drift_analysis": {
            "name": "Drift Analysis", 
            "icon": "fas fa-exclamation-triangle", 
            "color": "#e74c3c"
        }
    }
    
    panel_content = []
    
    for context_type, items in grouped_items.items():
        config = type_config.get(context_type, {
            "name": context_type.replace("_", " ").title(),
            "icon": "fas fa-question-circle",
            "color": "#6c757d"
        })
        
        # Create type header (simple display)
        type_header = html.Div([
            html.Div([
                html.I(className=f"{config['icon']} me-2", style={"color": config["color"]}),
                html.Span(f"{config['name']} ({len(items)})", style={"fontWeight": "500"})
            ], 
            className="d-flex align-items-center p-2",
            style={
                "backgroundColor": "#f8f9fa", 
                "border": f"1px solid {config['color']}", 
                "borderRadius": "4px",
                "marginBottom": "5px"
            })
        ])
        
        # Create items container 
        items_container = html.Div([
            create_simple_context_item(item, i) for i, item in enumerate(items)
        ], style={"marginBottom": "15px"})
        
        panel_content.extend([type_header, items_container])
    
    return html.Div(panel_content)


def extract_attribute_from_item(item):
    """
    Extract attribute name from context item using multiple possible field names.
    """
    # Try common field names in order of preference
    attr_name = (
        item.get('attribute_name') or 
        item.get('target_attribute') or 
        item.get('compare_attribute')
    )
    
    if attr_name:
        return attr_name
    
    # For distribution comparison, try to extract from cell_info
    cell_info = item.get('cell_info', '')
    if cell_info and isinstance(cell_info, str):
        import re
        # Look for patterns like "Column: attribute_name"
        match = re.search(r'(?:Column|column)\s*:?\s*([a-zA-Z_][a-zA-Z0-9_]*)', cell_info)
        if match:
            return match.group(1)
    
    return "Unknown"


def create_simple_context_item(item, index):
    """
    Create a simplified context item with expandable content and delete functionality.
    """
    # Get basic item info
    item_id = item.get("id", f"item_{index}")
    item_type = item.get("type", "unknown")
    timestamp = item.get("timestamp", "Unknown time")
    
    # Create title based on type with proper field name handling
    if item_type == "target_distribution":
        title = f"Target: {item.get('target_attribute', 'Unknown')}"
        subtitle = "Distribution analysis"
    elif item_type == "drift_analysis":
        # Drift analysis uses 'attribute_name' field
        attr_name = item.get('attribute_name') or item.get('target_attribute', 'Unknown')
        title = f"Drift: {attr_name}"
        subtitle = "Statistical drift detection"
    elif item_type == "conditional_distribution":
        target_attr = item.get('target_attribute', 'Unknown')
        target_value = item.get('target_value', 'Unknown')
        title = f"Conditional: {target_attr} = {target_value}"
        subtitle = f"vs {item.get('compare_attribute', 'Unknown')}"
    elif item_type == "distribution_comparison":
        # Extract attribute from cell_info or other fields
        attr_name = extract_attribute_from_item(item)
        title = f"Distribution: {attr_name}"
        subtitle = "Comparison analysis"
    elif item_type == "metric":
        metric_name = item.get('metric_name', 'Unknown')
        title = f"Metric: {metric_name}"
        subtitle = f"Value: {item.get('metric_value', 'N/A')}"
    else:
        title = item.get("title", item_type.replace("_", " ").title())
        subtitle = item.get("subtitle", "Analysis result")
    
    return dbc.Card([
        # Item header (clickable for expand/collapse)
        html.Div([
            html.Div([
                html.H6(title, className="mb-1", style={"color": "#2c3e50"}),
                html.Small(subtitle, className="text-muted")
            ], style={"flex": "1"}),
            html.Div([
                dbc.Button(
                    html.I(className="fas fa-trash-alt"),
                    id={"type": "delete-context", "item_id": item_id},
                    color="danger",
                    size="sm",
                    outline=True,
                    title="Delete this item",
                    className="me-2"
                ),
                dbc.Button(
                    html.I(className="fas fa-chevron-down", id={"type": "expand-icon", "item_id": item_id}),
                    id={"type": "expand-toggle", "item_id": item_id},
                    color="secondary",
                    size="sm",
                    outline=True,
                    title="Expand/Collapse details"
                )
            ], className="d-flex align-items-center")
        ], className="d-flex align-items-center p-3", style={"cursor": "pointer"}),
        
        # Item details (collapsible content - default collapsed)
        dbc.Collapse([
            html.Div([
                html.Hr(className="my-2"),
                *create_expanded_context_content(item)
            ], className="px-3 pb-3")
        ], 
        id={"type": "item-collapse", "item_id": item_id},
        is_open=False  # Default collapsed
        )
    ], className="mb-2", style={"border": "1px solid #dee2e6"})


# Simplified delete callback - only one simple callback needed
@callback(
    [Output("explain-context-data", "data", allow_duplicate=True),
     Output("context-delete-success-toast", "is_open", allow_duplicate=True)],
    [Input({"type": "delete-context", "item_id": ALL}, "n_clicks")],
    [State("explain-context-data", "data")],
    prevent_initial_call=True
)
def handle_simple_context_deletion(n_clicks_list, current_context_data):
    """
    Simplified context item deletion - only handles data removal.
    Frontend handles UI updates automatically.
    """
    if not n_clicks_list or not any(n_clicks_list):
        raise PreventUpdate
    
    # Find which button was clicked
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate
    
    triggered_id = ctx.triggered[0]['prop_id']
    print(f"[SIMPLE DELETE] Triggered: {triggered_id}")
    
    # Extract item ID from callback context
    import json
    try:
        button_info = json.loads(triggered_id.split('.')[0])
        item_id_to_delete = button_info['item_id']
        print(f"[SIMPLE DELETE] Deleting item: {item_id_to_delete}")
    except Exception as e:
        print(f"[SIMPLE DELETE] Parse error: {e}")
        raise PreventUpdate
    
    # Remove item from data
    if current_context_data:
        updated_data = [
            item for item in current_context_data 
            if item.get("id") != item_id_to_delete
        ]
        print(f"[SIMPLE DELETE] Removed item. Count: {len(current_context_data)} -> {len(updated_data)}")
        return updated_data, True
    
    return current_context_data, False


# Simplified analysis panel - loads on demand
def create_on_demand_analysis_panel():
    """
    Create analysis panel that loads content on-demand when user requests analysis.
    No automatic GPT calls. Enhanced with loading state for better UX.
    """
    return html.Div([
        # Main ready state
        html.Div([
            html.I(className="fas fa-chart-line me-2", style={"color": "#6c757d", "fontSize": "2rem"}),
            html.H5("Ready for Analysis", style={"color": "#6c757d"}),
            html.P("Click 'Analyze Context' below to start comprehensive GPT-powered analysis", className="text-muted"),
            html.Hr(),
            dbc.Button(
                [html.I(className="fas fa-play me-2"), "Analyze Context"],
                id="start-context-analysis-btn",
                color="primary",
                size="lg",
                className="mt-3"
            )
        ], className="text-center", style={"padding": "60px 20px"}, id="analysis-ready-state"),
        
        # Loading state (initially hidden)
        html.Div([
            dcc.Loading([
                html.Div([
                    html.I(className="fas fa-brain me-2", style={"color": "#614385", "fontSize": "2rem"}),
                    html.H5("GPT Analysis in Progress", style={"color": "#614385"}),
                    html.P([
                        "AI is analyzing your context items and assessing severity levels...",
                        html.Br(),
                        html.Small("This may take 10-30 seconds depending on the number of items", className="text-muted")
                    ]),
                    html.Div([
                        dbc.Progress(animated=True, color="primary", className="mb-3"),
                        html.Div([
                            html.I(className="fas fa-check-circle me-2 text-success"),
                            "Collecting dataset metadata",
                            html.Br(),
                            html.I(className="fas fa-check-circle me-2 text-success"),
                            "Preparing GPT prompt with user personalization",
                            html.Br(),
                            html.I(className="fas fa-spinner fa-spin me-2 text-primary"),
                            "Analyzing with GPT-4...",
                        ], style={"fontSize": "14px", "textAlign": "left"})
                    ])
                ])
            ], type="circle", color="#614385")
        ], className="text-center", style={"padding": "60px 20px", "display": "none"}, id="analysis-loading-state")
    ], id="on-demand-analysis-container")


def create_simple_analysis_panel(context_data):
    """
    Create a simple fallback analysis panel when unified analysis fails.
    """
    if not context_data:
        return html.Div([
            html.H5("No Context Data"),
            html.P("No context items available for analysis.", className="text-muted")
        ])
    
    # Count different types of context items
    type_counts = {}
    for item in context_data:
        item_type = item.get("type", "unknown")
        type_counts[item_type] = type_counts.get(item_type, 0) + 1
    
    # Create summary cards
    summary_cards = []
    for item_type, count in type_counts.items():
        config = {
            "drift_analysis": {"name": "Drift Analysis", "icon": "fas fa-exclamation-triangle", "color": "danger"},
            "distribution_comparison": {"name": "Distribution Comparison", "icon": "fas fa-balance-scale", "color": "success"},
            "conditional_distribution": {"name": "Conditional Distribution", "icon": "fas fa-project-diagram", "color": "warning"},
            "target_distribution": {"name": "Target Distribution", "icon": "fas fa-chart-bar", "color": "primary"},
            "metric": {"name": "Statistical Metrics", "icon": "fas fa-calculator", "color": "info"}
        }.get(item_type, {"name": item_type.title(), "icon": "fas fa-question", "color": "secondary"})
        
        summary_cards.append(
            dbc.Card([
                dbc.CardBody([
                    html.H6([
                        html.I(className=f"{config['icon']} me-2"),
                        config["name"]
                    ]),
                    html.H4(str(count), className="text-primary")
                ])
            ], color=config["color"], outline=True, className="mb-2")
        )
    
    return html.Div([
        html.H5([
            html.I(className="fas fa-chart-bar me-2", style={"color": "#0d6efd"}),
            "Context Analysis Summary"
        ]),
        html.Hr(),
        html.P(f"Analyzing {len(context_data)} context items:", className="text-muted"),
        html.Div(summary_cards),
        html.Hr(),
        dbc.Alert([
            html.I(className="fas fa-info-circle me-2"),
            "Basic analysis complete. For detailed insights, please ensure all dependencies are available."
        ], color="info"),
        dbc.Button(
            [html.I(className="fas fa-redo me-2"), "Retry Full Analysis"],
            id="retry-full-analysis-btn",
            color="primary",
            className="mt-3"
        )
    ])


@callback(
    Output("explain-analysis-panel", "children", allow_duplicate=True),
    [Input("start-context-analysis-btn", "n_clicks")],
    [State("explain-context-data", "data")],
    prevent_initial_call=True
)
def load_analysis_on_demand(n_clicks, context_data):
    """
    Load comprehensive four-layer analysis panel with GPT-powered intelligence.
    Implements all four layers: Severity Statistics, Context Item Boxes, Joint Analysis, Strategy Selection.
    """
    if not n_clicks or not context_data:
        raise PreventUpdate
    
    print(f"[COMPREHENSIVE ANALYSIS] Starting four-layer GPT analysis for {len(context_data)} context items")
    
    try:
        # Perform comprehensive GPT analysis using new severity analyzer
        print("[COMPREHENSIVE ANALYSIS] Calling GPT for comprehensive four-layer analysis...")
        strategy_analysis = gpt_severity_analyzer.analyze_context_severity(
            context_data 
            # Let GPT intelligently decide strategy based on analysis, not drift with defaults
        )
        
        # Log the comprehensive analysis results for debugging
        comprehensive_data = strategy_analysis.get("comprehensive_data", {})
        if comprehensive_data:
            layer1 = comprehensive_data.get("layer1_severity_statistics", {})
            layer2 = comprehensive_data.get("layer2_context_analysis", [])
            layer4 = comprehensive_data.get("layer4_strategy_selection", {})
            
            print(f"[COMPREHENSIVE ANALYSIS] Layer 1 - High: {layer1.get('high_count', 0)}, "
                  f"Medium: {layer1.get('medium_count', 0)}, Low: {layer1.get('low_count', 0)}")
            print(f"[COMPREHENSIVE ANALYSIS] Layer 2 - {len(layer2)} context items analyzed")
            print(f"[COMPREHENSIVE ANALYSIS] Layer 4 - Recommended strategy: {layer4.get('recommended_strategy', 'unknown')}")
        
        # Create the comprehensive four-layer analysis panel
        return create_comprehensive_analysis_panel(context_data, strategy_analysis)
        
    except Exception as e:
        print(f"[COMPREHENSIVE ANALYSIS] Error in GPT analysis: {str(e)}")
        
        # Enhanced fallback with error information
        fallback_analysis = {
            "recommended_strategy": "monitor",
            "confidence": 0.3,
            "analysis_summary": f"GPT analysis temporarily unavailable: {str(e)}. Please check your OpenAI API configuration.",
            "overall_assessment": {
                "high_count": 0,
                "medium_count": 0,
                "low_count": len(context_data),
                "recommendation": "monitor",
                "overall_risk_level": "Low",
                "confidence_score": 0.3,
                "summary": "Analysis fallback - unable to connect to GPT"
            },
            "comprehensive_data": {
                "layer1_severity_statistics": {
                    "high_count": 0,
                    "medium_count": 0,
                    "low_count": len(context_data),
                    "overall_risk_level": "Low",
                    "confidence_score": 0.3,
                    "summary": "Fallback analysis - GPT unavailable"
                },
                "layer2_context_analysis": [
                    {
                        "context_id": i,
                        "title": f"Analysis for {context_data[i].get('type', 'Unknown').replace('_', ' ').title()}",
                        "severity_score": 50,
                        "risk_level": "Medium",
                        "explanation": {
                            "beginner": f"Fallback analysis: This {context_data[i].get('type', 'unknown')} issue requires attention when GPT is unavailable.",
                            "intermediate": f"Fallback analysis: The {context_data[i].get('type', 'unknown')} issue shows moderate impact on model performance.",
                            "advanced": f"Fallback analysis: Technical assessment of {context_data[i].get('type', 'unknown')} suggests systematic review required."
                        },
                        "business_impact": f"GPT analysis temporarily unavailable. This {context_data[i].get('type', 'unknown')} issue should be reviewed for business impact.",
                        "technical_details": f"Technical details for {context_data[i].get('type', 'unknown')} analysis are being processed. Please retry when GPT is available.",
                        "action_required": f"Review and assess {context_data[i].get('type', 'unknown')} issue manually while waiting for GPT analysis."
                    }
                    for i in range(len(context_data))
                ],
                "layer3_joint_analysis": {
                    "overall_assessment": "Analysis pending due to GPT connectivity issues",
                    "interaction_effects": "Not analyzed",
                    "business_impact": "Impact assessment unavailable",
                    "technical_complexity": "Medium",
                    "urgency_level": "Medium",
                    "resource_requirements": "To be determined",
                    "success_probability": "Moderate"
                },
                "layer4_strategy_selection": {
                    "recommended_strategy": "monitor",
                    "strategy_overview": "Continue monitoring model performance with enhanced alerting and validation systems. Suitable when issues are manageable and model performance remains acceptable.",
                    "confidence": 0.3,
                    "reasoning": f"Default recommendation due to analysis error: {str(e)}",
                    "alternative_strategies": [],
                    "implementation_roadmap": {
                        "phase1_immediate": ["Check GPT API configuration"],
                        "phase2_short_term": ["Retry analysis"],
                        "phase3_long_term": ["Monitor system health"]
                    },
                    "success_metrics": ["System availability"],
                    "risk_factors": ["Analysis reliability"]
                }
            },
            "total_issues": len(context_data),
            "gpt_analysis": "Analysis temporarily unavailable. Please try again.",
            "error": str(e)
        }
        
        try:
            return create_comprehensive_analysis_panel(context_data, fallback_analysis)
        except Exception as fallback_error:
            print(f"[COMPREHENSIVE ANALYSIS] Fallback also failed: {str(fallback_error)}")
            # Final fallback to simple panel
            return create_simple_analysis_panel(context_data)


def create_expanded_context_content(context_item):
    """
    Create expanded content for a context item, matching the format used in chat box.
    
    Args:
        context_item (dict): Context item data
        
    Returns:
        list: List of HTML components for expanded content
    """
    context_type = context_item.get("type", "unknown")
    
    if context_type == "target_distribution":
        return create_target_distribution_expanded_content(context_item)
    elif context_type == "conditional_distribution":
        return create_conditional_distribution_expanded_content(context_item)
    elif context_type == "distribution_comparison":
        return create_distribution_comparison_expanded_content(context_item)
    elif context_type == "metric":
        return create_metric_expanded_content(context_item)
    elif context_type == "drift_analysis":
        return create_drift_analysis_expanded_content(context_item)
    else:
        return [html.P("No detailed information available.", className="text-muted")]


def create_target_distribution_expanded_content(item):
    """Create expanded content for target distribution context item - matches chat box format exactly."""
    content = []
    
    # Format and display summary text exactly like chat box
    summary_text = item.get("summary_text", "")
    if summary_text:
        # Split by lines and create paragraphs like in chat box
        summary_lines = [line for line in summary_text.split('\n') if line.strip()]
        if summary_lines:
            content.extend([
                html.P(line) for line in summary_lines
            ])
    else:
        content.append(html.P("No summary data available"))
    
    # Add chart container exactly like chat box
    chart_data = item.get("chart_data", "")
    if chart_data:
        content.append(html.Div(chart_data))
    else:
        content.append(html.Div("No chart data available"))
    
    return content


def create_conditional_distribution_expanded_content(item):
    """Create expanded content for conditional distribution context item - matches chat box format exactly."""
    content = []
    
    # Format and display summary text exactly like chat box
    summary_text = item.get("summary_text", "")
    if summary_text:
        # Split by lines and create paragraphs like in chat box
        summary_lines = [line for line in summary_text.split('\n') if line.strip()]
        if summary_lines:
            content.extend([
                html.P(line) for line in summary_lines
            ])
    else:
        content.append(html.P("No summary data available"))
    
    # Add chart container exactly like chat box
    chart_data = item.get("chart", "")
    if chart_data:
        content.append(html.Div(chart_data))
    else:
        content.append(html.Div("No chart data available"))
    
    return content


def create_distribution_comparison_expanded_content(item):
    """Create expanded content for distribution comparison context item - matches chat box format exactly."""
    content = []
    
    # Display summary content exactly like chat box
    if "stored_summary" in item:
        # Use stored_summary if available (like in chat box)
        content.append(item["stored_summary"])
    else:
        # Fall back to summary_text formatted as paragraphs
        summary_text = item.get("summary_text", "")
        if summary_text:
            content.append(html.Pre(summary_text, 
                         style={"whiteSpace": "pre-wrap", "fontSize": "12px", "backgroundColor": "#f8f9fa", 
                                "padding": "8px", "borderRadius": "4px"}))
        else:
            content.append(html.P("No summary data available"))
    
    # Add chart content exactly like chat box
    if "stored_chart" in item:
        content.append(html.Div([
            html.H6("Distribution Chart", 
                   style={"marginTop": "15px", "marginBottom": "10px", "fontWeight": "bold", "color": "#614385"}),
            item["stored_chart"]
        ]))
    elif item.get("chart_data"):
        content.append(html.Div(item["chart_data"]))
    else:
        content.append(html.Div("No chart data available"))
    
    return content


def create_metric_expanded_content(item):
    """Create expanded content for metric context item - matches chat box format exactly."""
    content = [
        html.H6("Statistical Metric Details", 
               style={"marginTop": "5px", "marginBottom": "10px", "fontWeight": "bold", "color": "#614385"}),
    ]
    
    # Add attribute type
    attribute_type = item.get("attribute_type", "Unknown")
    if attribute_type != "Unknown":
        content.append(html.P([html.Strong("Attribute Type: "), attribute_type]))
    
    # Add metric details exactly like chat box
    metric_details = item.get("metric_details", "")
    if metric_details:
        content.extend([
            html.P([html.Strong("Metric Details: ")]),
            html.Pre(metric_details, 
                   style={"whiteSpace": "pre-wrap", "fontSize": "12px", "backgroundColor": "#f8f9fa", 
                          "padding": "8px", "borderRadius": "4px"})
        ])
    
    # Add interpretation exactly like chat box
    interpretation = item.get("interpretation", "")
    if interpretation:
        content.extend([
            html.P([html.Strong("Interpretation: ")]),
            html.Pre(interpretation, 
                   style={"whiteSpace": "pre-wrap", "fontSize": "12px", "backgroundColor": "#f8f9fa", 
                          "padding": "8px", "borderRadius": "4px"})
        ])
    
    return content


def create_drift_analysis_expanded_content(item):
    """Create expanded content for drift analysis context item - matches chat box format exactly."""
    content = []
    
    # Format and display summary text exactly like chat box
    summary_text = item.get("summary_text", "")
    if summary_text:
        # Split by lines and create paragraphs like in chat box
        summary_lines = [line for line in summary_text.split('\n') if line.strip()]
        if summary_lines:
            content.extend([
                html.P(line) for line in summary_lines
            ])
    else:
        content.append(html.P("No summary data available"))
    
    # If there's additional detailed content, display it
    interpretation = item.get("interpretation", "")
    if interpretation and interpretation != summary_text:
        content.append(html.Div([
            html.H6("Statistical Interpretation", 
                   style={"marginTop": "15px", "marginBottom": "10px", "fontWeight": "bold", "color": "#dc3545"}),
            html.Pre(interpretation, 
                   style={"whiteSpace": "pre-wrap", "fontSize": "12px", "backgroundColor": "#f8f9fa", 
                          "padding": "8px", "borderRadius": "4px"})
        ]))
    
    return content


def create_analysis_panel(selected_context=None, analysis_type=None):
    """
    Create the right panel for analysis based on selected context.
    
    Args:
        selected_context (dict): Currently selected context item
        analysis_type (str): Type of analysis requested
        
    Returns:
        html.Div: Analysis panel component
    """
    if not selected_context:
        return html.Div([
            html.Div([
                html.I(className="fas fa-mouse-pointer me-2", style={"color": "#6c757d", "fontSize": "2rem"}),
                html.H5("Select a Context Item", style={"color": "#6c757d"}),
                html.P("Click on a context item from the left panel to start analysis", className="text-muted")
            ], className="text-center", style={"padding": "60px 20px"})
        ])
    
    context_type = selected_context.get("type", "unknown")
    
    # Define analysis options for each context type
    analysis_options = {
        "target_distribution": [
            {"id": "training-impact", "label": "Training Impact", "icon": "fas fa-graduation-cap", "color": "primary"},
            {"id": "distribution-shift", "label": "Distribution Shift", "icon": "fas fa-arrows-alt-h", "color": "info"},
            {"id": "fairness-impact", "label": "Fairness Impact", "icon": "fas fa-balance-scale", "color": "warning"}
        ],
        "conditional_distribution": [
            {"id": "relationship-analysis", "label": "Relationship Analysis", "icon": "fas fa-project-diagram", "color": "success"},
            {"id": "drift-detection", "label": "drift Detection", "icon": "fas fa-search", "color": "danger"}, 
            {"id": "interaction-effects", "label": "Interaction Effects", "icon": "fas fa-exchange-alt", "color": "info"}
        ],
        "distribution_comparison": [
            {"id": "statistical-significance", "label": "Statistical Significance", "icon": "fas fa-calculator", "color": "primary"},
            {"id": "practical-impact", "label": "Practical Impact", "icon": "fas fa-chart-line", "color": "success"},
            {"id": "recommendations", "label": "Recommendations", "icon": "fas fa-lightbulb", "color": "warning"}
        ],
        "metric": [
            {"id": "metric-interpretation", "label": "Metric Interpretation", "icon": "fas fa-info-circle", "color": "info"},
            {"id": "threshold-analysis", "label": "Threshold Analysis", "icon": "fas fa-sliders-h", "color": "primary"},
            {"id": "action-items", "label": "Action Items", "icon": "fas fa-tasks", "color": "success"}
        ],
        "drift_analysis": [
            {"id": "drift-severity", "label": "Drift Severity", "icon": "fas fa-thermometer-half", "color": "danger"},
            {"id": "root-causes", "label": "Root Causes", "icon": "fas fa-search-plus", "color": "primary"},
            {"id": "mitigation-strategies", "label": "Mitigation Strategies", "icon": "fas fa-shield-alt", "color": "success"}
        ]
    }
    
    # Get options for this context type
    options = analysis_options.get(context_type, [])
    
    # Create context summary with expand/collapse functionality
    context_summary_section = create_expandable_context_summary(selected_context)
    
    # Create analysis buttons
    analysis_buttons = []
    for option in options:
        button = dbc.Button(
            [html.I(className=f"{option['icon']} me-2"), option["label"]],
            id={"type": "analysis-type-btn", "analysis": option["id"]},
            color=option["color"],
            className="mb-2 w-100",
            size="sm"
        )
        analysis_buttons.append(button)
    
    # Create analysis results area
    results_area = html.Div(
        id="explain-analysis-results",
        children=[
            create_technical_perspective_content(selected_context)
        ],
        style={
            "border": "1px solid #dee2e6",
            "borderRadius": "4px",
            "padding": "20px",
            "marginTop": "20px",
            "minHeight": "200px",
            "backgroundColor": "#fafafa"
        }
    )
    
    return html.Div([
        # Expandable context summary
        context_summary_section,
        
        html.Hr(),
        
        # Analysis perspective buttons (replacing analysis type buttons)
        html.H6("Analysis Perspective:", className="mb-3"),
        html.Div([
            dbc.ButtonGroup([
                dbc.Button(
                    [html.I(className="fas fa-chart-line me-2"), "Technical"],
                    id="technical-perspective-btn",
                    color="primary",
                    size="sm",
                    active=True  # Default active
                ),
                dbc.Button(
                    [html.I(className="fas fa-briefcase me-2"), "Business"],
                    id="business-perspective-btn", 
                    color="success",
                    size="sm"
                ),
                dbc.Button(
                    [html.I(className="fas fa-user-tie me-2"), "Executive"],
                    id="executive-perspective-btn",
                    color="info", 
                    size="sm"
                )
            ], className="w-100")
        ]),
        
        # Results area
        results_area
    ])


def create_context_summary(context_item):
    """
    Create a summary display for the selected context item.
    
    Args:
        context_item (dict): Context item data
        
    Returns:
        html.Div: Context summary component
    """
    context_type = context_item.get("type", "unknown")
    
    # Type-specific summary creation
    if context_type == "target_distribution":
        target_attr = context_item.get("target_attribute", "Unknown")
        timestamp = context_item.get("timestamp", "Unknown")
        
        return dbc.Card([
            dbc.CardBody([
                html.H6([
                    html.I(className="fas fa-chart-bar me-2", style={"color": "#614385"}),
                    "Target Distribution Analysis"
                ]),
                html.P([
                    html.Strong("Target Attribute: "), target_attr, html.Br(),
                    html.Strong("Added: "), timestamp
                ], className="mb-0 small")
            ])
        ], className="border-primary")
        
    elif context_type == "conditional_distribution":
        target_attr = context_item.get("target_attribute", "Unknown")
        target_value = context_item.get("target_value", "Unknown") 
        compare_attr = context_item.get("compare_attribute", "Unknown")
        
        return dbc.Card([
            dbc.CardBody([
                html.H6([
                    html.I(className="fas fa-filter me-2", style={"color": "#516395"}),
                    "Conditional Distribution Analysis"
                ]),
                html.P([
                    html.Strong("Condition: "), f"{target_attr} = {target_value}", html.Br(),
                    html.Strong("Analyzed Attribute: "), compare_attr
                ], className="mb-0 small")
            ])
        ], className="border-info")
        
    elif context_type == "distribution_comparison":
        # Extract attribute from cell_info
        cell_info = context_item.get('cell_info', '')
        attr_name = "Unknown"
        if cell_info:
            lines = cell_info.split('\n')
            for line in lines:
                # Look for pattern "Column: column_name, Value: ..."
                if "Column:" in line:
                    # Extract column name between "Column: " and ", Value:"
                    parts = line.split("Column:")
                    if len(parts) > 1:
                        column_part = parts[1].strip()
                        if ", Value:" in column_part:
                            attr_name = column_part.split(", Value:")[0].strip()
                        else:
                            attr_name = column_part.strip()
                        break
                # Fallback: use first non-empty line with reasonable length
                elif len(line.strip()) > 0 and len(line.strip()) < 50:
                    attr_name = line.strip()
                    break
        
        return dbc.Card([
            dbc.CardBody([
                html.H6([
                    html.I(className="fas fa-balance-scale me-2", style={"color": "#28a745"}),
                    "Distribution Comparison"
                ]),
                html.P([
                    html.Strong("Attribute: "), attr_name
                ], className="mb-0 small")
            ])
        ], className="border-success")
        
    elif context_type == "metric":
        metric_name = context_item.get("metric_name", "Unknown")
        attr_name = context_item.get("attribute_name", "Unknown")
        
        return dbc.Card([
            dbc.CardBody([
                html.H6([
                    html.I(className="fas fa-chart-line me-2", style={"color": "#fd7e14"}),
                    "Individual Metric Analysis"
                ]),
                html.P([
                    html.Strong("Metric: "), metric_name, html.Br(),
                    html.Strong("Attribute: "), attr_name
                ], className="mb-0 small")
            ])
        ], className="border-warning")
        
    elif context_type == "drift_analysis":
        attr_name = context_item.get("attribute_name", "Unknown")
        
        return dbc.Card([
            dbc.CardBody([
                html.H6([
                    html.I(className="fas fa-exclamation-triangle me-2", style={"color": "#dc3545"}),
                    "Drift Analysis"
                ]),
                html.P([
                    html.Strong("Attribute: "), attr_name
                ], className="mb-0 small")
            ])
        ], className="border-danger")
        
    else:
        return dbc.Card([
            dbc.CardBody([
                html.H6("Unknown Context Type"),
                html.P(f"Type: {context_type}", className="mb-0 small text-muted")
            ])
        ], className="border-secondary")


def create_expandable_context_summary(context_item):
    """
    Create an expandable summary display for the selected context item in Analysis Panel.
    
    Args:
        context_item (dict): Context item data
        
    Returns:
        html.Div: Expandable context summary component
    """
    context_type = context_item.get("type", "unknown")
    context_id = context_item.get("id", "unknown")
    
    # Create basic summary card (always visible)
    basic_summary = create_context_summary(context_item)
    
    # Add expand/collapse button to the basic summary
    enhanced_summary = html.Div([
        # Header with expand button
        html.Div([
            html.Div([
                basic_summary.children[0]  # Extract the CardBody content correctly
            ], style={"flex": "1"}),
            html.Div([
                html.Button([
                    html.I(className="fas fa-chevron-down", id={"type": "context-expand-icon", "context_id": context_id})
                ], 
                id={"type": "context-expand-btn", "context_id": context_id},
                className="btn btn-sm btn-outline-primary",
                style={
                    "padding": "4px 8px",
                    "fontSize": "12px"
                },
                title="Show detailed content"
                )
            ], style={"display": "flex", "alignItems": "center"})
        ], style={
            "display": "flex",
            "justifyContent": "space-between",
            "alignItems": "center",
            "width": "100%"
        }),
        
        # Expanded content (initially hidden)
        html.Div(
            id={"type": "context-detailed-content", "context_id": context_id},
            children=create_expanded_context_content(context_item),
            style={
                "display": "none",
                "marginTop": "15px",
                "padding": "15px",
                "backgroundColor": "#f8f9fa",
                "borderRadius": "6px",
                "border": "1px solid #e9ecef"
            }
        )
    ], className=f"border-{get_context_border_color(context_type)}", style={"padding": "15px", "borderRadius": "8px", "border": "2px solid"})
    
    return enhanced_summary


def get_context_border_color(context_type):
    """Get the appropriate border color class for context type."""
    color_map = {
        "target_distribution": "primary",
        "conditional_distribution": "info", 
        "distribution_comparison": "success",
        "metric": "warning",
        "drift_analysis": "danger"
    }
    return color_map.get(context_type, "secondary")


# ===== NEW CALLBACKS FOR CONTEXT-BASED ANALYSIS =====

def convert_analysis_to_context_items(strategy_analysis_data):
    """
    Convert analysis data back to context items for display.
    This ensures Step 1 and Step 2 use the same reliable data source.
    
    Args:
        strategy_analysis_data (dict): Data from unified-strategy-analysis store
        
    Returns:
        list: Context items in the format expected by create_context_items_panel
    """
    context_items = []
    
    try:
        if not strategy_analysis_data or not isinstance(strategy_analysis_data, dict):
            print("[CONTEXT CONVERSION] No strategy analysis data available")
            return []
        
        comprehensive_data = strategy_analysis_data.get('comprehensive_data', {})
        if not comprehensive_data:
            print("[CONTEXT CONVERSION] No comprehensive_data in strategy analysis")
            return []
        
        layer2_analysis = comprehensive_data.get('layer2_context_analysis', [])
        if not layer2_analysis:
            print("[CONTEXT CONVERSION] No layer2_context_analysis data")
            return []
        
        print(f"[CONTEXT CONVERSION] Converting {len(layer2_analysis)} analysis items to context items")
        
        for analysis_item in layer2_analysis:
            # Extract core information from analysis item
            context_id = analysis_item.get('context_id', 0)
            context_type = analysis_item.get('context_type', 'unknown')
            title = analysis_item.get('title', 'Unknown Analysis')
            risk_level = analysis_item.get('risk_level', 'Medium')
            
            # Extract attribute name from title or other fields
            attribute_name = extract_attribute_name_from_analysis(analysis_item, title)
            
            # Create context item in the expected format
            context_item = {
                'id': f'converted-{context_id}-{int(time.time() * 1000)}',
                'type': context_type,
                'summary_text': analysis_item.get('explanation', {}).get('intermediate', title),
                'expanded': False,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                
                # Type-specific fields based on context_type
                'attribute_name': attribute_name,
                'risk_level': risk_level,
                
                # Store original analysis data for modal functionality
                '_analysis_data': analysis_item
            }
            
            # Add type-specific fields
            if context_type == 'target_distribution':
                context_item['target_attribute'] = attribute_name
            elif context_type == 'conditional_distribution':
                context_item['target_attribute'] = analysis_item.get('target_attribute', attribute_name)
                context_item['target_value'] = analysis_item.get('target_value', 'unknown')
                context_item['compare_attribute'] = analysis_item.get('compare_attribute', 'unknown')
            elif context_type == 'distribution_comparison':
                context_item['compare_attribute'] = attribute_name
            elif context_type == 'metric':
                context_item['metric_name'] = attribute_name
            
            context_items.append(context_item)
        
        print(f"[CONTEXT CONVERSION] Successfully converted {len(context_items)} context items")
        return context_items
        
    except Exception as e:
        print(f"[CONTEXT CONVERSION] Error converting analysis to context items: {str(e)}")
        import traceback
        traceback.print_exc()
        return []


def extract_attribute_name_from_analysis(analysis_item, title):
    """Extract attribute name from analysis item or title."""
    # Try direct fields first
    attribute_name = (
        analysis_item.get('attribute_name') or
        analysis_item.get('target_attribute') or
        analysis_item.get('compare_attribute') or
        analysis_item.get('metric_name')
    )
    
    if attribute_name:
        return attribute_name
    
    # Extract from title using pattern matching
    import re
    if title and isinstance(title, str):
        # Look for patterns like "Analysis: attribute_name" or "Type: attribute_name"
        match = re.search(r':\s*([a-zA-Z_][a-zA-Z0-9_]*)', title)
        if match:
            return match.group(1)
        
        # Fallback: split by space and take last word
        words = title.split()
        if len(words) > 1:
            return words[-1]
    
    return "unknown"


@callback(
    Output("explain-context-items-panel", "children"),
    [Input("explain-context-data", "data")],
    prevent_initial_call=True
)
def load_context_items_panel(explain_context_data):
    """
    Load and display clickable context items panel with type grouping.
    This callback creates the interactive left panel with expandable context types.
    """
    if global_vars.current_stage != "explain":
        return dash.no_update
    
    try:
        if not explain_context_data:
            explain_context_data = []
        
        print(f"[CONTEXT ITEMS] Loading clickable context panel for {len(explain_context_data)} context items")
        
        # Use our new clickable context items panel
        return create_context_items_panel(explain_context_data)
        
    except Exception as e:
        print(f"[CONTEXT ITEMS] Error loading context items panel: {str(e)}")
        return html.Div([
            html.Div([
                html.I(className="fas fa-exclamation-circle me-2", style={"color": "#dc3545"}),
                f"Error loading context items: {str(e)}"
            ], className="text-danger text-center", style={"padding": "40px 20px"})
        ])


def create_empty_context_panel():
    """Create empty context panel when no data is available."""
    return html.Div([
        html.Div([
            html.I(className="fas fa-info-circle me-2", style={"color": "#6c757d", "fontSize": "2rem"}),
            html.H5("No Context Items", style={"color": "#6c757d"}),
            html.P("Context items will appear here after running analysis", className="text-muted")
        ], className="text-center", style={"padding": "60px 20px"})
    ])


# COMMENTED: Using chat-based explain workflow - no need for global store updates
# Store Data Update Callback - Separated to handle store updates
# @callback(
#     Output("unified-strategy-analysis", "data"),
#     [Input("current-stage", "children"),
#      Input("explain-context-data", "data")],
#     prevent_initial_call=True
# )
# def update_strategy_analysis_store(current_stage, explain_context_data):
#     """Update unified-strategy-analysis store with GPT analysis data."""
#     if current_stage != "explain" or not explain_context_data:
#         return {}
#     
#     print(f"[STORE UPDATE] Updating unified-strategy-analysis store for {len(explain_context_data)} context items")
#     
#     try:
#         # Perform comprehensive GPT analysis and store the result
#         strategy_analysis = gpt_severity_analyzer.analyze_context_severity(
#             explain_context_data
#             # Let GPT intelligently choose strategy based on analysis  
#         )
#         
#         print(f"[STORE UPDATE] Successfully stored analysis data with comprehensive_data: {'comprehensive_data' in strategy_analysis}")
#         if 'comprehensive_data' in strategy_analysis:
#             layer2_count = len(strategy_analysis['comprehensive_data'].get('layer2_context_analysis', []))
#             print(f"[STORE UPDATE] Layer2 data count in store: {layer2_count}")
#         
#         return strategy_analysis
#         
#     except Exception as e:
#         print(f"[STORE UPDATE] Error updating strategy analysis store: {str(e)}")
#         return {}


# COMMENTED: Using chat-based explain workflow - no need for global analysis panel
# @callback(
#     Output("explain-analysis-panel", "children"),
#     [Input("unified-strategy-analysis", "data"),
#      Input("current-stage", "children")],
#     [State("explain-context-data", "data")],
#     prevent_initial_call=True
# )
# def load_unified_analysis_panel(strategy_analysis, current_stage, explain_context_data):
#     """Load unified analysis panel using data from store."""
#     print(f"[LOAD ANALYSIS PANEL] Stage: {current_stage}, Context items: {len(explain_context_data) if explain_context_data else 0}")
#     
#     if current_stage != "explain":
#         return html.Div([
#             html.Div([
#                 html.I(className="fas fa-lightbulb me-2", style={"color": "#6c757d", "fontSize": "2rem"}),
#                 html.H5("Analysis Ready", style={"color": "#6c757d"}),
#                 html.P("Switch to explain stage to begin analysis", className="text-muted")
#             ], className="text-center", style={"padding": "60px 20px"})
#         ])
#     
#     if not explain_context_data:
#         return create_on_demand_analysis_panel()
#     
#     # Use strategy analysis from store
#     if not strategy_analysis or not strategy_analysis.get('comprehensive_data'):
#         print(f"[LOAD PANEL] No strategy analysis data available, showing on-demand panel")
#         return create_on_demand_analysis_panel()
#     
#     print(f"[LOAD PANEL] Loading comprehensive analysis panel with store data")
#     print(f"[LOAD PANEL] Store data has comprehensive_data: {'comprehensive_data' in strategy_analysis}")
#     
#     try:
#         # Create the comprehensive four-layer analysis panel using store data
#         return create_comprehensive_analysis_panel(explain_context_data, strategy_analysis)
#         
#     except Exception as e:
#         print(f"[LOAD PANEL] Error creating analysis panel: {str(e)}")
#         
#         # Fallback to on-demand panel with error message
#         return html.Div([
#             dbc.Alert([
#                 html.I(className="fas fa-exclamation-triangle me-2"),
#                 html.Strong("Panel Creation Error: "),
#                 f"Failed to create analysis panel: {str(e)}. Please try manual analysis below."
#             ], color="warning", className="mb-3"),
#             create_on_demand_analysis_panel()
#         ])


# Removed old analysis type callback - replaced with perspective switching


@callback(
    [Output({"type": "context-detailed-content", "context_id": ALL}, "style", allow_duplicate=True),
     Output({"type": "context-expand-icon", "context_id": ALL}, "className", allow_duplicate=True)],
    [Input({"type": "context-expand-btn", "context_id": ALL}, "n_clicks")],
    [State({"type": "context-detailed-content", "context_id": ALL}, "style"),
     State("explain-selected-context", "data")],
    prevent_initial_call=True
)
def toggle_analysis_panel_context_expansion(n_clicks_list, current_styles, selected_context):
    """
    Toggle the expansion state of the context item in the Analysis Panel.
    
    Args:
        n_clicks_list (list): List of n_clicks for expand buttons
        current_styles (list): List of current styles for detailed content divs
        selected_context (dict): Currently selected context item
        
    Returns:
        tuple: (new_styles, new_icon_classes)
    """
    ctx = dash.callback_context
    
    if not ctx.triggered or not any(n_clicks_list) or not selected_context:
        return dash.no_update, dash.no_update
    
    try:
        context_id = selected_context.get("id", "unknown")
        
        # Find the triggered button index
        triggered_prop = ctx.triggered[0]["prop_id"]
        
        # For now, we only have one context item displayed, so we can handle the first (and only) item
        current_style = current_styles[0] if current_styles else {}
        current_display = current_style.get("display", "none")
        
        # Toggle display state
        new_display = "block" if current_display == "none" else "none"
        
        # Create new style
        new_style = {
            "display": new_display,
            "marginTop": "15px",
            "padding": "15px" if new_display == "block" else "0",
            "backgroundColor": "#f8f9fa",
            "borderRadius": "6px",
            "border": "1px solid #e9ecef" if new_display == "block" else "none"
        }
        
        # Update icon class
        new_icon_class = "fas fa-chevron-up" if new_display == "block" else "fas fa-chevron-down"
        
        print(f"[EXPLAIN ANALYSIS] Toggling context expansion: {current_display} -> {new_display}")
        
        return [new_style], [new_icon_class]
        
    except Exception as e:
        print(f"[EXPLAIN ANALYSIS] Error toggling context expansion: {str(e)}")
        return dash.no_update, dash.no_update


def generate_analysis_content(context_item, analysis_type):
    """
    Generate analysis content based on context item and analysis type.
    
    Args:
        context_item (dict): Selected context item
        analysis_type (str): Type of analysis requested
        
    Returns:
        html.Div: Analysis content
    """
    context_type = context_item.get("type", "unknown")
    
    # Create loading message with GPT call
    loading_msg = html.Div([
        dcc.Loading([
            html.Div([
                html.H6([
                    html.I(className="fas fa-brain me-2", style={"color": "#614385"}),
                    f"Generating {analysis_type.replace('-', ' ').title()} Analysis..."
                ]),
                html.P("AI is analyzing your context item and generating insights...", className="text-muted")
            ])
        ], type="circle")
    ])
    
    # For now, return simulated analysis content
    # In a real implementation, this would call the GPT API
    
    analysis_content = create_simulated_analysis(context_item, analysis_type)
    
    return analysis_content


def create_simulated_analysis(context_item, analysis_type):
    """
    Create simulated analysis content for demonstration.
    In production, this would be replaced with actual GPT API calls.
    """
    context_type = context_item.get("type", "unknown")
    
    # Get context-specific information
    if context_type == "target_distribution":
        target_attr = context_item.get("target_attribute", "target")
        analysis_content = f"""
        ### {analysis_type.replace('-', ' ').title()} Analysis
        
        **Target Attribute:** {target_attr}
        
        Based on the target distribution analysis, here are the key insights:
        
        - **Distribution Shift Impact**: The observed distribution differences in {target_attr} indicate potential model performance variations
        - **Training Implications**: Models trained on the primary dataset may not generalize well to the secondary dataset
        - **Recommendation**: Consider dataset balancing or domain adaptation techniques
        
        *This analysis is based on the statistical patterns identified in your target distribution comparison.*
        """
    
    elif context_type == "conditional_distribution":
        target_attr = context_item.get("target_attribute", "target")
        target_value = context_item.get("target_value", "unknown")
        compare_attr = context_item.get("compare_attribute", "attribute")
        
        analysis_content = f"""
        ### {analysis_type.replace('-', ' ').title()} Analysis
        
        **Conditional Analysis:** {target_attr} = {target_value} vs {compare_attr}
        
        Key findings from the conditional distribution analysis:
        
        - **Relationship Strength**: The distribution of {compare_attr} differs significantly when {target_attr} = {target_value}
        - **drift Indicators**: Potential drift detected in how {compare_attr} relates to the target outcome
        - **Fairness Implications**: This relationship may lead to disparate impact across different groups
        
        *Analysis based on your conditional distribution context.*
        """
    
    else:
        analysis_content = f"""
        ### {analysis_type.replace('-', ' ').title()} Analysis
        
        **Context Type:** {context_type.replace('_', ' ').title()}
        
        General analysis insights:
        
        - Statistical patterns have been identified in your data
        - Distribution differences may impact model performance
        - Consider reviewing the specific metrics and recommendations
        
        *This is a general analysis template. Specific insights would be generated based on your actual data.*
        """
    
    # Convert markdown-like content to HTML
    lines = analysis_content.strip().split('\n')
    components = []
    
    for line in lines:
        line = line.strip()
        if line.startswith('### '):
            components.append(html.H5(line[4:], className="text-primary mb-3"))
        elif line.startswith('**') and line.endswith('**'):
            components.append(html.H6(line[2:-2], className="mb-2"))
        elif line.startswith('- **') and '**:' in line:
            parts = line[3:].split('**:', 1)
            if len(parts) == 2:
                components.append(html.P([
                    html.Strong(parts[0] + ': '),
                    parts[1].strip()
                ], className="mb-2"))
        elif line.startswith('- '):
            components.append(html.Li(line[2:], className="mb-1"))
        elif line.startswith('*') and line.endswith('*'):
            components.append(html.P(line[1:-1], className="text-muted small font-italic mt-3"))
        elif line:
            components.append(html.P(line, className="mb-2"))
    
    return html.Div(components)


# REMOVED - Combined with main context loading callback above

def create_technical_perspective_content(context_item):
    """Create technical perspective content showing statistical analysis."""
    if not context_item:
        return html.P("Select a context item to view technical analysis.", className="text-muted text-center")
    
    context_type = context_item.get("type", "unknown")
    
    # Create technical analysis content based on context type
    content = [
        html.H5([
            html.I(className="fas fa-chart-line me-2", style={"color": "#0d6efd"}),
            "Technical Analysis"
        ]),
        html.Hr()
    ]
    
    if context_type == "drift_analysis":
        # Show drift analysis metrics
        metric_details = context_item.get("metric_details", "")
        interpretation = context_item.get("interpretation", "")
        
        content.extend([
            html.H6("Statistical Metrics", style={"color": "#dc3545"}),
            html.Pre(metric_details, style={
                "backgroundColor": "#f8f9fa", 
                "padding": "10px", 
                "borderRadius": "4px",
                "fontSize": "12px"
            }) if metric_details else html.P("No metric details available"),
            
            html.H6("Statistical Interpretation", className="mt-3", style={"color": "#dc3545"}),
            html.Pre(interpretation, style={
                "backgroundColor": "#f8f9fa",
                "padding": "10px", 
                "borderRadius": "4px",
                "fontSize": "12px"
            }) if interpretation else html.P("No interpretation available")
        ])
        
    elif context_type == "conditional_distribution":
        # Show conditional distribution technical details
        summary_text = context_item.get("summary_text", "")
        
        content.extend([
            html.H6("Statistical Summary", style={"color": "#516395"}),
            html.Div([
                html.P(line) for line in summary_text.split('\n') if line.strip()
            ]) if summary_text else html.P("No summary available")
        ])
        
    elif context_type == "metric":
        # Show metric technical details
        metric_details = context_item.get("metric_details", "")
        interpretation = context_item.get("interpretation", "")
        
        content.extend([
            html.H6("Metric Details", style={"color": "#fd7e14"}),
            html.Pre(metric_details, style={
                "backgroundColor": "#f8f9fa",
                "padding": "10px",
                "borderRadius": "4px", 
                "fontSize": "12px"
            }) if metric_details else html.P("No metric details available"),
            
            html.H6("Technical Interpretation", className="mt-3", style={"color": "#fd7e14"}),
            html.Pre(interpretation, style={
                "backgroundColor": "#f8f9fa",
                "padding": "10px",
                "borderRadius": "4px",
                "fontSize": "12px"
            }) if interpretation else html.P("No interpretation available")
        ])
        
    else:
        # Generic technical view for other types
        content.append(html.P(f"Technical analysis for {context_type} type context.", className="text-muted"))
    
    return html.Div(content)


def create_business_perspective_content(context_item, gpt_analysis):
    """Create business perspective content based on GPT analysis."""
    if not context_item or not gpt_analysis:
        return html.Div([
            html.Div([
                html.I(className="fas fa-spinner fa-spin me-2"),
                "Analyzing business impact..."
            ], className="text-center text-muted", style={"padding": "40px"})
        ])
    
    # Check if this is an error response
    if gpt_analysis.get("error", False):
        return gpt_analysis.get("content", html.Div("Analysis encountered a problem"))
    
    content = [
        html.H5([
            html.I(className="fas fa-briefcase me-2", style={"color": "#198754"}),
            "Business impact analysis"
        ]),
        html.Hr()
    ]
    
    # Business scenario identification
    if gpt_analysis.get("business_scenario"):
        content.extend([
            html.H6("🎯 Business scenario identification", style={"color": "#198754"}),
            html.Div([
                html.P(gpt_analysis["business_scenario"], className="alert alert-info")
            ])
        ])
    
    # Business impact translation
    if gpt_analysis.get("business_translation"):
        content.extend([
            html.H6("💼 Business impact interpretation", className="mt-3", style={"color": "#198754"}),
            html.Div(gpt_analysis["business_translation"])
        ])
    
    # Risk assessment
    if gpt_analysis.get("risk_assessment"):
        content.extend([
            html.H6("⚠️ Risk assessment", className="mt-3", style={"color": "#198754"}),
            html.Div(gpt_analysis["risk_assessment"])
        ])
    
    return html.Div(content)


def create_executive_perspective_content(context_item, gpt_analysis):
    """Create executive summary perspective content."""
    if not context_item or not gpt_analysis:
        return html.Div([
            html.Div([
                html.I(className="fas fa-spinner fa-spin me-2"),
                "Preparing executive summary..."
            ], className="text-center text-muted", style={"padding": "40px"})
        ])
    
    # Check if this is an error response
    if gpt_analysis.get("error", False):
        return gpt_analysis.get("content", html.Div("Analysis encountered a problem"))
    
    content = [
        html.H5([
            html.I(className="fas fa-user-tie me-2", style={"color": "#0dcaf0"}),
            "Executive summary"
        ]),
        html.Hr()
    ]
    
    # Key findings
    if gpt_analysis.get("key_findings"):
        content.extend([
            html.H6("🔍 Key findings", style={"color": "#0dcaf0"}),
            html.Div(gpt_analysis["key_findings"], className="mb-3")
        ])
    
    # Strategic recommendations
    if gpt_analysis.get("strategic_recommendations"):
        content.extend([
            html.H6("🚀 Strategic recommendations", style={"color": "#0dcaf0"}),
            html.Div(gpt_analysis["strategic_recommendations"], className="mb-3")
        ])
    
    # Resource requirements
    if gpt_analysis.get("resource_requirements"):
        content.extend([
            html.H6("💰 Resource requirements", style={"color": "#0dcaf0"}),
            html.Div(gpt_analysis["resource_requirements"])
        ])
    
    return html.Div(content)


def analyze_context_with_gpt(context_item, perspective_type="business"):
    """
    Analyze context item using GPT for business insights.
    
    Args:
        context_item (dict): The context item to analyze
        perspective_type (str): "business" or "executive"
        
    Returns:
        dict: GPT analysis results
    """
    try:
        # Use existing explain_api module
        from agent.explain_api import generate_response_from_prompt
        from flask_login import current_user
        from UI.functions.global_vars import global_vars
        import json
        
        # Extract target attribute from various possible sources
        target_attr = extract_target_attribute(context_item)
        
        # Get user domain information
        user_domain_info = get_user_domain_info(current_user)
        
        # Get dataset attribute information
        dataset_attributes_info = get_dataset_attributes_info()
        
        # Prepare enhanced context data for GPT
        context_data = {
            "type": context_item.get("type", "unknown"),
            "target_attribute": target_attr,
            "summary_text": context_item.get("summary_text", ""),
            "metric_details": context_item.get("metric_details", ""),
            "interpretation": context_item.get("interpretation", ""),
            "user_domain": user_domain_info,
            "dataset_attributes": dataset_attributes_info
        }
        
        # Use unified prompt manager for analysis
        from .prompt_manager import (
            prompt_manager,
            create_user_context,
            create_dataset_context
        )
        
        # Prepare context for prompt manager
        prompt_context = {
            'user_context': create_user_context(user_domain_info),
            'dataset_context': create_dataset_context(dataset_attributes_info),
            'context_items': [context_item],
            'strategy_focus': 'analyze'
        }
        
        # Generate prompt using unified template system
        template_name = f"{perspective_type}_analysis"
        prompt = prompt_manager.generate_prompt(template_name, prompt_context)
        
        # Make actual GPT API call using existing integration
        try:
            # Call GPT API - use user's selected model instead of hardcoded "gpt-4o"
            gpt_response = generate_response_from_prompt(prompt)
            
            # Try to parse JSON response, if it fails, return formatted text response
            try:
                result = json.loads(gpt_response)
                return format_gpt_response(result, perspective_type)
            except json.JSONDecodeError:
                # If response is not JSON, create structured response from text
                return format_text_response(gpt_response, context_item, perspective_type)
                
        except Exception as api_error:
            print(f"[GPT ANALYSIS] API Error: {str(api_error)}")
            # Return error information instead of mock data
            return create_api_error_response(api_error, perspective_type)
        
    except Exception as e:
        print(f"[GPT ANALYSIS] Error: {str(e)}")
        # Return error information instead of mock data
        return create_general_error_response(e, perspective_type)


# Legacy prompt functions removed - now using unified prompt manager
# All prompts are now handled by UI.pages.components.prompt_manager


def create_api_error_response(error, perspective_type):
    """Create user-friendly error response when GPT API fails."""
    error_msg = str(error).lower()
    
    # Determine specific error type
    if "timeout" in error_msg or "connection" in error_msg:
        error_type = "Network connection"
        error_description = "Network connection timeout or unstable"
        suggestions = [
            "Check if the network connection is normal",
            "Wait a few minutes and try again",
            "If the problem persists, please contact the administrator"
        ]
    elif "auth" in error_msg or "api" in error_msg or "key" in error_msg:
        error_type = "API authentication"
        error_description = "API key invalid or expired"
        suggestions = [
            "Please contact the system administrator to check the API configuration",
            "Confirm that the OpenAI service is normal",
            "Retry later or use the technical perspective to view the analysis results"
        ]
    elif "quota" in error_msg or "limit" in error_msg:
        error_type = "Service quota"
        error_description = "API call limit reached"
        suggestions = [
            "Please try again later",
            "Please contact the administrator to upgrade the service quota",
            "Use the technical perspective to view the statistical analysis temporarily"
        ]
    else:
        error_type = "服务异常"
        error_description = "GPT分析服务暂时不可用"
        suggestions = [
            "Please try again later",
            "Check the network connection",
            "You can first view the statistical analysis results of the technical perspective"
        ]
    
    perspective_name = "Business perspective" if perspective_type == "business" else "Executive summary"
    
    return {
        "error": True,
        "error_type": error_type,
        "perspective": perspective_name,
        "content": html.Div([
            html.Div([
                html.I(className="fas fa-exclamation-triangle me-3", 
                      style={"fontSize": "24px", "color": "#fd7e14"}),
                html.H5(f"{perspective_name} analysis is temporarily unavailable", style={"color": "#fd7e14", "display": "inline"})
            ], className="d-flex align-items-center mb-3"),
            
            html.Div([
                html.H6("Problem type", style={"color": "#6c757d"}),
                html.P(f"{error_type}: {error_description}", className="alert alert-warning")
            ], className="mb-3"),
            
            html.Div([
                html.H6("Suggested solutions", style={"color": "#6c757d"}),
                html.Ul([
                    html.Li(suggestion) for suggestion in suggestions
                ], className="mb-3")
            ]),
            
            html.Div([
                html.H6("Alternative solutions", style={"color": "#6c757d"}),
                html.P([
                    "You can click the ",
                    html.Strong("Technical perspective", style={"color": "#0d6efd"}),
                    " button to view the detailed statistical analysis results, or try again later to get business insights."
                ], className="alert alert-info")
            ]),
            
            html.Div([
                dbc.Button([
                    html.I(className="fas fa-redo me-2"),
                    "Try again"
                ], id="retry-gpt-analysis", color="primary", size="sm")
            ], className="text-center mt-3")
        ])
    }


def create_general_error_response(error, perspective_type):
    """Create user-friendly error response for general errors."""
    perspective_name = "Business perspective" if perspective_type == "business" else "Executive summary"
    
    return {
        "error": True,
        "error_type": "System error",
        "perspective": perspective_name,
        "content": html.Div([
            html.Div([
                html.I(className="fas fa-times-circle me-3", 
                      style={"fontSize": "24px", "color": "#dc3545"}),
                html.H5(f"{perspective_name} analysis encountered a problem", style={"color": "#dc3545", "display": "inline"})
            ], className="d-flex align-items-center mb-3"),
            
            html.Div([
                html.P("The system encountered an unexpected problem while processing your analysis request.", className="alert alert-danger")
            ], className="mb-3"),
            
            html.Div([
                html.H6("Suggested actions", style={"color": "#6c757d"}),
                html.Ul([
                    html.Li("Refresh the page and try again"),
                    html.Li("Check if the network connection is normal"),
                    html.Li("Use the technical perspective to view the statistical analysis results"),
                    html.Li("If the problem persists, please contact technical support")
                ], className="mb-3")
            ]),
            
            html.Div([
                html.P([
                    "You can still use the ",
                    html.Strong("Technical perspective", style={"color": "#0d6efd"}),
                    " to view the detailed statistical analysis and data comparison results."
                ], className="alert alert-info")
            ]),
            
            html.Div([
                dbc.Button([
                    html.I(className="fas fa-redo me-2"),
                    "Try again"
                ], id="retry-analysis", color="danger", size="sm")
            ], className="text-center mt-3")
        ])
    }


# Callback for perspective button switching
@callback(
    Output("explain-analysis-results", "children", allow_duplicate=True),
    [Input("technical-perspective-btn", "n_clicks"),
     Input("business-perspective-btn", "n_clicks"), 
     Input("executive-perspective-btn", "n_clicks")],
    [State("explain-selected-context", "data")],
    prevent_initial_call=True
)
def switch_analysis_perspective(technical_clicks, business_clicks, executive_clicks, selected_context):
    """Handle switching between different analysis perspectives."""
    ctx = dash.callback_context
    
    if not ctx.triggered or not selected_context:
        return dash.no_update
    
    triggered_button = ctx.triggered[0]["prop_id"].split(".")[0]
    
    if triggered_button == "technical-perspective-btn":
        return create_technical_perspective_content(selected_context)
    
    elif triggered_button == "business-perspective-btn":
        # Analyze with GPT for business perspective
        gpt_analysis = analyze_context_with_gpt(selected_context, "business")
        return create_business_perspective_content(selected_context, gpt_analysis)
    
    elif triggered_button == "executive-perspective-btn":
        # Analyze with GPT for executive perspective  
        gpt_analysis = analyze_context_with_gpt(selected_context, "executive")
        return create_executive_perspective_content(selected_context, gpt_analysis)
    
    return dash.no_update


# Callback to update button states
@callback(
    [Output("technical-perspective-btn", "active"),
     Output("business-perspective-btn", "active"),
     Output("executive-perspective-btn", "active")],
    [Input("technical-perspective-btn", "n_clicks"),
     Input("business-perspective-btn", "n_clicks"),
     Input("executive-perspective-btn", "n_clicks")],
    prevent_initial_call=True
)
def update_perspective_button_states(technical_clicks, business_clicks, executive_clicks):
    """Update active state of perspective buttons."""
    ctx = dash.callback_context
    
    if not ctx.triggered:
        return True, False, False  # Default to technical active
    
    triggered_button = ctx.triggered[0]["prop_id"].split(".")[0]
    
    if triggered_button == "technical-perspective-btn":
        return True, False, False
    elif triggered_button == "business-perspective-btn":
        return False, True, False
    elif triggered_button == "executive-perspective-btn":
        return False, False, True
    
    return True, False, False  # Default fallback


# Callback for retry buttons
@callback(
    Output("explain-analysis-results", "children", allow_duplicate=True),
    [Input("retry-gpt-analysis", "n_clicks"),
     Input("retry-analysis", "n_clicks")],
    [State("explain-selected-context", "data"),
     State("business-perspective-btn", "active"),
     State("executive-perspective-btn", "active")],
    prevent_initial_call=True
)
def handle_retry_analysis(retry_gpt_clicks, retry_general_clicks, selected_context, business_active, executive_active):
    """Handle retry button clicks for failed analyses."""
    ctx = dash.callback_context
    
    if not ctx.triggered or not selected_context:
        return dash.no_update
    
    # Determine which perspective to retry based on active button
    if business_active:
        perspective_type = "business"
    elif executive_active:
        perspective_type = "executive"
    else:
        # If neither business nor executive is active, don't retry
        return dash.no_update
    
    # Re-analyze with GPT
    gpt_analysis = analyze_context_with_gpt(selected_context, perspective_type)
    
    if perspective_type == "business":
        return create_business_perspective_content(selected_context, gpt_analysis)
    else:
        return create_executive_perspective_content(selected_context, gpt_analysis)


def format_gpt_response(result, perspective_type):
    """Format properly structured JSON response from GPT into HTML components."""
    if perspective_type == "business":
        return {
            "business_scenario": result.get("business_scenario", "Business scenario analysis"),
            "business_translation": format_business_translation(result.get("business_translation", [])),
            "risk_assessment": format_risk_assessment(result.get("risk_assessment", {}))
        }
    else:  # executive
        return {
            "key_findings": format_key_findings(result.get("key_findings", [])),
            "strategic_recommendations": format_strategic_recommendations(result.get("strategic_recommendations", [])),
            "resource_requirements": format_resource_requirements(result.get("resource_requirements", {}))
        }


def format_text_response(text_response, context_item, perspective_type):
    """Format unstructured text response from GPT into structured HTML components."""
    if perspective_type == "business":
        target_attr = extract_target_attribute(context_item)
        return {
            "business_scenario": f"Analysis scenario based on {target_attr}",
            "business_translation": html.Div([
                html.P("GPT analysis result:"),
                html.Pre(text_response, style={
                    "backgroundColor": "#f8f9fa",
                    "padding": "10px",
                    "borderRadius": "4px",
                    "whiteSpace": "pre-wrap"
                })
            ]),
            "risk_assessment": html.Div([
                html.P([html.Strong("Analysis status:"), "GPT has completed the analysis"]),
                html.P([html.Strong("Suggestion:"), "Please develop a corresponding strategy based on the above analysis"])
            ])
        }
    else:  # executive
        return {
            "key_findings": html.Div([
                html.P("GPT executive summary:"),
                html.Pre(text_response, style={
                    "backgroundColor": "#f8f9fa",
                    "padding": "10px",
                    "borderRadius": "4px",
                    "whiteSpace": "pre-wrap"
                })
            ]),
            "strategic_recommendations": html.P("Please develop a corresponding strategy based on the above analysis"),
            "resource_requirements": html.P("Resource requirements will be determined based on the specific implementation plan")
        }


def format_business_translation(translation_list):
    """Format business translation list into HTML components."""
    if not translation_list:
        return html.P("No business translation available")
    
    components = []
    for item in translation_list:
        if isinstance(item, dict):
            components.append(html.P([
                html.Strong(f"{item.get('metric', 'Metric')}: "),
                item.get('business_meaning', 'No meaning provided'),
                html.Br(),
                html.Small(f"Impact: {item.get('impact_estimate', 'Not estimated')}")
            ]))
        else:
            components.append(html.P(str(item)))
    
    return html.Div(components)


def format_risk_assessment(risk_data):
    """Format risk assessment data into HTML components."""
    if not risk_data:
        return html.P("No risk assessment available")
    
    return html.Div([
        html.P([html.Strong("Risk Level: "), risk_data.get("risk_level", "Unknown")]),
        html.P([html.Strong("Confidence: "), risk_data.get("confidence", "Unknown")]),
        html.P([html.Strong("3-Month Impact: "), risk_data.get("three_month_impact", "Unknown")]),
        html.P([html.Strong("Affected Customers: "), risk_data.get("affected_customers", "Unknown")])
    ])


def format_key_findings(findings_list):
    """Format key findings list into HTML components."""
    if not findings_list:
        return html.P("No key findings available")
    
    return html.Ul([
        html.Li(finding) for finding in findings_list
    ])


def format_strategic_recommendations(recommendations_list):
    """Format strategic recommendations list into HTML components."""
    if not recommendations_list:
        return html.P("No strategic recommendations available")
    
    components = []
    for rec in recommendations_list:
        if isinstance(rec, dict):
            components.append(html.Div([
                html.H6(f"Action: {rec.get('action', 'Unknown action')}", className="text-primary"),
                html.P([
                    html.Strong("Priority: "), rec.get('priority', 'Unknown'), html.Br(),
                    html.Strong("Timeline: "), rec.get('timeline', 'Unknown'), html.Br(),
                    html.Strong("Expected Outcome: "), rec.get('expected_outcome', 'Unknown')
                ])
            ], className="alert alert-light border-primary mb-2"))
        else:
            components.append(html.P(str(rec)))
    
    return html.Div(components)


def format_resource_requirements(resource_data):
    """Format resource requirements data into HTML components."""
    if not resource_data:
        return html.P("No resource requirements available")
    
    return html.Div([
        html.P([html.Strong("Effort Estimate: "), resource_data.get("effort_estimate", "Unknown")]),
        html.P([html.Strong("Team Involvement: "), resource_data.get("team_involvement", "Unknown")]),
        html.P([html.Strong("Budget Impact: "), resource_data.get("budget_impact", "Unknown")])
    ])


def extract_target_attribute(context_item):
    """
    Extract target attribute from context item, checking multiple possible sources.
    
    Args:
        context_item (dict): Context item data
        
    Returns:
        str: Target attribute name
    """
    # Try different possible field names
    target_attr = context_item.get("target_attribute")
    if target_attr:
        return target_attr
    
    # For conditional distribution, try compare_attribute
    target_attr = context_item.get("compare_attribute")
    if target_attr:
        return target_attr
    
    # For distribution comparison, try to extract from cell_info
    cell_info = context_item.get('cell_info', '')
    if cell_info:
        lines = cell_info.split('\n')
        for line in lines:
            # Look for pattern "Column: column_name, Value: ..."
            if "Column:" in line:
                # Extract column name between "Column: " and ", Value:"
                parts = line.split("Column:")
                if len(parts) > 1:
                    column_part = parts[1].strip()
                    if ", Value:" in column_part:
                        return column_part.split(", Value:")[0].strip()
                    else:
                        return column_part.strip()
            # Alternative pattern: "Column: column_name"
            elif "Column:" in line and "=" not in line:
                return line.split("Column:")[-1].strip()
    
    # For drift analysis, try to extract from summary_text
    summary_text = context_item.get("summary_text", "")
    if summary_text:
        # Look for patterns like "attribute_name drift analysis" or similar
        lines = summary_text.split('\n')
        for line in lines:
            if any(keyword in line.lower() for keyword in ['attribute', 'column', 'feature']):
                # Try to extract the attribute name
                words = line.split()
                for i, word in enumerate(words):
                    if word.lower() in ['attribute', 'column', 'feature'] and i > 0:
                        return words[i-1]
    
    # Try to extract from metric details
    metric_details = context_item.get("metric_details", "")
    if metric_details:
        lines = metric_details.split('\n')
        for line in lines:
            if ":" in line and any(keyword in line.lower() for keyword in ['attribute', 'column', 'feature']):
                parts = line.split(':')
                if len(parts) >= 2:
                    # Get the part after the colon, clean it up
                    value = parts[1].strip()
                    if value and not value.lower().startswith(('unknown', 'n/a', 'none')):
                        return value
    
    # Fallback: look for any field that might contain attribute name
    possible_fields = ['attribute_name', 'column_name', 'feature_name', 'variable_name']
    for field in possible_fields:
        value = context_item.get(field)
        if value:
            return value
    
    # Last resort: return the context type as identifier
    context_type = context_item.get("type", "unknown")
    return f"{context_type} attribute"


def get_user_domain_info(current_user):
    """
    Extract user domain and background information for context-aware analysis using centralized global system.
    
    Args:
        current_user: Flask-Login current user object
        
    Returns:
        dict: User domain information
    """
    try:
        # Use the centralized user context management system
        from UI.functions.global_vars import global_vars
        
        # Get comprehensive user context with caching and fallback handling
        user_context = global_vars.get_user_context(current_user)
        
        profile_complete = user_context.get('profile_completeness', 0)
        
        if user_context.get('has_profile', False):
            print(f"[EXPLAIN COMPONENT] Retrieved user context - Profile {profile_complete:.0f}% complete")
        else:
            print(f"[EXPLAIN COMPONENT] No user profile available - using general analysis")
        
        # Create domain-specific context only if we have industry information
        domain_context = ""
        industry_sector = user_context.get("industry_sector")
        
        if industry_sector:
            if industry_sector == "Healthcare":
                domain_context = "Healthcare: Focus on patient safety, diagnostic accuracy, treatment effectiveness, and medical equity."
            elif industry_sector == "Finance":
                domain_context = "Finance: Focus on risk management, credit decision-making, fraud detection, compliance, and customer value."
            elif industry_sector == "Technology":
                domain_context = "Technology: Focus on user experience, product performance, system stability, and data-driven decision-making."
            elif industry_sector == "Education":
                domain_context = "Education: Focus on learning outcomes, teaching quality, student achievements, and educational equity."
            elif industry_sector == "Media":
                domain_context = "Media: Focus on content quality, user engagement, recommendation accuracy, and content diversity."
            else:
                domain_context = f"{industry_sector} industry business characteristics and key metrics"
        else:
            domain_context = "General business scenario - no specific industry context available"
        
        # Add domain context to the user context
        user_context["domain_context"] = domain_context
        
        return user_context
        
    except Exception as e:
        print(f"[EXPLAIN COMPONENT] Error getting user domain info: {str(e)}")
        return {
            "has_profile": False,
            "professional_role": None,
            "industry_sector": None,
            "expertise_level": None,
            "technical_level": None,
            "drift_awareness": None,
            "areas_of_interest": None,
            "persona_prompt": None,
            "system_prompt": None,
            "prefix_prompt": None,
            "profile_completeness": 0,
            "domain_context": "General business scenario - user information unavailable",
            "error": str(e)
        }


def get_dataset_attributes_info():
    """
    Extract dataset attributes information for context-aware analysis.
    
    Returns:
        dict: Dataset attributes information
    """
    try:
        from UI.functions.global_vars import global_vars
        import pandas as pd
        
        attributes_info = {
            "primary_columns": [],
            "secondary_columns": [],
            "common_columns": [],
            "column_types": {},
            "column_descriptions": {},
            "dataset_summary": {}
        }
        
        # Get primary dataset info
        if hasattr(global_vars, 'df') and global_vars.df is not None:
            df = global_vars.df
            attributes_info["primary_columns"] = list(df.columns)
            
            # Get column types and basic stats
            for col in df.columns:
                col_type = str(df[col].dtype)
                unique_count = df[col].nunique()
                null_count = df[col].isnull().sum()
                
                # Determine column type category
                if pd.api.types.is_numeric_dtype(df[col]):
                    if unique_count == 2:
                        category = "Binary variable"
                    elif unique_count < 10:
                        category = "Categorical variable"
                    else:
                        category = "Continuous numerical variable"
                elif pd.api.types.is_datetime64_any_dtype(df[col]):
                    category = "Time variable"
                else:
                    if unique_count < 10:
                        category = "Categorical variable"
                    elif unique_count < 50:
                        category = "Multi-categorical variable"
                    else:
                        category = "Text/ID variable"
                
                attributes_info["column_types"][col] = {
                    "dtype": col_type,
                    "category": category,
                    "unique_count": unique_count,
                    "null_count": null_count,
                    "null_percentage": round(null_count / len(df) * 100, 2)
                }
                
                # Generate meaningful description
                if category == "二分类变量":
                    values = df[col].unique()[:2]
                    desc = f"Binary variable, values: {values[0]} and {values[1]}"
                elif category == "Categorical variable" or category == "Multi-categorical variable":
                    desc = f"{category}, {unique_count} categories"
                elif category == "Continuous numerical variable":
                    mean_val = df[col].mean()
                    std_val = df[col].std()
                    desc = f"Continuous numerical variable, mean: {mean_val:.2f}, standard deviation: {std_val:.2f}"
                elif category == "Time variable":
                    desc = "Time series variable"
                else:
                    desc = f"{category}, {unique_count} unique values"
                
                attributes_info["column_descriptions"][col] = desc
            
            # Dataset summary
            attributes_info["dataset_summary"]["primary"] = {
                "row_count": len(df),
                "column_count": len(df.columns),
                "numeric_columns": len([col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]),
                "categorical_columns": len([col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col])]),
                "missing_data_columns": len([col for col in df.columns if df[col].isnull().sum() > 0])
            }
        
        # Get secondary dataset info
        if hasattr(global_vars, 'secondary_df') and global_vars.secondary_df is not None:
            df_sec = global_vars.secondary_df
            attributes_info["secondary_columns"] = list(df_sec.columns)
            
            attributes_info["dataset_summary"]["secondary"] = {
                "row_count": len(df_sec),
                "column_count": len(df_sec.columns),
                "numeric_columns": len([col for col in df_sec.columns if pd.api.types.is_numeric_dtype(df_sec[col])]),
                "categorical_columns": len([col for col in df_sec.columns if not pd.api.types.is_numeric_dtype(df_sec[col])]),
                "missing_data_columns": len([col for col in df_sec.columns if df_sec[col].isnull().sum() > 0])
            }
        
        # Find common columns
        if attributes_info["primary_columns"] and attributes_info["secondary_columns"]:
            attributes_info["common_columns"] = list(set(attributes_info["primary_columns"]) & 
                                                   set(attributes_info["secondary_columns"]))
        
        # Get target attribute if available
        if hasattr(global_vars, 'target_attribute') and global_vars.target_attribute:
            attributes_info["target_attribute"] = global_vars.target_attribute
        
        return attributes_info
        
    except Exception as e:
        print(f"[DATASET ATTRIBUTES] Error getting dataset attributes info: {str(e)}")
        return {
            "primary_columns": [],
            "secondary_columns": [],
            "common_columns": [],
            "column_types": {},
            "column_descriptions": {},
            "dataset_summary": {}
        }


def create_unified_left_panel_summary(all_context_items, strategy_analysis):
    """
    Create a unified summary panel for the left side showing all context items overview.
    Enhanced for Layer 1: Severity Statistics with GPT-powered analysis.
    
    Args:
        all_context_items: List of all context items
        strategy_analysis: Strategy recommendation analysis from GPT
        
    Returns:
        html.Div: Left panel summary component with enhanced severity statistics
    """
    if not all_context_items:
        return html.Div([
            html.Div([
                html.I(className="fas fa-info-circle me-2", style={"color": "#6c757d", "fontSize": "1.5rem"}),
                html.H6("No Context Items", style={"color": "#6c757d"}),
                html.P("Add items from the detect stage to begin analysis.", className="text-muted small")
            ], className="text-center", style={"padding": "30px 15px"})
        ])
    
    # Check if analysis has error
    has_error = "error" in strategy_analysis
    
    # === LAYER 1: SEVERITY STATISTICS ===
    severity_breakdown = strategy_analysis.get("severity_breakdown", {"high": 0, "medium": 0, "low": 0})
    gpt_analysis_summary = strategy_analysis.get("gpt_analysis", "")
    
    # Enhanced Severity Statistics Section
    severity_section = html.Div([
        html.H6([
            html.I(className="fas fa-exclamation-triangle me-2", style={"color": "#dc3545"}),
            "Severity Analysis"
        ], className="mb-3", style={"fontWeight": "bold", "color": "#2c3e50"}),
        
        # Main severity display
        html.Div([
            # High severity
            html.Div([
                dbc.Badge([
                    html.I(className="fas fa-fire me-1"),
                    f"High severity: {severity_breakdown['high']}"
                ], color="danger", className="p-2 me-2 mb-2", style={"fontSize": "14px"}),
            ], className="d-inline-block"),
            
            # Medium severity  
            html.Div([
                dbc.Badge([
                    html.I(className="fas fa-exclamation-circle me-1"),
                    f"Medium severity: {severity_breakdown['medium']}"
                ], color="warning", className="p-2 me-2 mb-2", style={"fontSize": "14px"}),
            ], className="d-inline-block"),
            
            # Low severity
            html.Div([
                dbc.Badge([
                    html.I(className="fas fa-check-circle me-1"),
                    f"Low severity: {severity_breakdown['low']}"
                ], color="success", className="p-2 me-2 mb-2", style={"fontSize": "14px"}),
            ], className="d-inline-block"),
        ], className="mb-3"),
        
        # Analysis status indicator
        html.Div([
            dbc.Alert([
                html.I(className="fas fa-robot me-2"),
                html.Strong("GPT Analysis: "),
                "Complete" if not has_error else "Error - using fallback",
                html.Br() if has_error else "",
                html.Small(str(strategy_analysis.get("error", "")), className="text-muted") if has_error else ""
            ], color="success" if not has_error else "warning", className="py-2 px-3 mb-0", style={"fontSize": "13px"})
        ])
    ], className="mb-4", style={
        "backgroundColor": "#f8f9fa", 
        "padding": "20px", 
        "borderRadius": "8px",
        "border": "2px solid #e9ecef"
    })
    
    # === STRATEGY RECOMMENDATION ===
    recommended_strategy = strategy_analysis.get("recommended_strategy", "monitor")
    confidence = strategy_analysis.get("confidence", 0.0)
    
    strategy_colors = {
        "retrain": "danger",
        "finetune": "warning", 
        "monitor": "info"
    }
    
    strategy_icons = {
        "retrain": "fas fa-sync-alt",
        "finetune": "fas fa-sliders-h",
        "monitor": "fas fa-eye"
    }
    
    strategy_card = dbc.Alert([
        html.H6([
            html.I(className=f"{strategy_icons.get(recommended_strategy, 'fas fa-lightbulb')} me-2"),
            "Recommended Strategy"
        ], className="mb-2"),
        html.P([
            html.Strong(f"{recommended_strategy.upper()}"),
            html.Span(f" ({confidence:.0%} confidence)", className="ms-2 small")
        ], className="mb-2"),
        html.P(strategy_analysis.get("analysis_summary", "No analysis available"), className="small mb-0")
    ], color=strategy_colors.get(recommended_strategy, "info"), className="mb-3")
    
    # === CONTEXT ITEMS BREAKDOWN ===
    type_groups = {}
    for item in all_context_items:
        item_type = item.get("type", "unknown")
        if item_type not in type_groups:
            type_groups[item_type] = []
        type_groups[item_type].append(item)
    
    # Create summary cards for each type
    type_cards = []
    type_colors = {
        "drift_analysis": "#dc3545",
        "distribution_comparison": "#28a745",
        "conditional_distribution": "#516395", 
        "target_distribution": "#614385",
        "metric": "#fd7e14"
    }
    
    type_icons = {
        "drift_analysis": "fas fa-chart-line",
        "distribution_comparison": "fas fa-balance-scale",
        "conditional_distribution": "fas fa-filter",
        "target_distribution": "fas fa-bullseye",
        "metric": "fas fa-tachometer-alt"
    }
    
    for item_type, items in type_groups.items():
        color = type_colors.get(item_type, "#6c757d")
        icon = type_icons.get(item_type, "fas fa-question-circle")
        type_display = item_type.replace("_", " ").title()
        
        type_cards.append(
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.I(className=icon, style={"color": color, "fontSize": "1.1rem"}),
                        html.H6(type_display, className="mb-1 ms-2", style={"color": color})
                    ], className="d-flex align-items-center mb-2"),
                    html.P([
                        html.Strong(f"{len(items)} item{'s' if len(items) != 1 else ''}"),
                        html.Br(),
                        html.Small(f"Analyzed by GPT", className="text-muted")
                    ], className="mb-0 small")
                ], className="p-3")
            ], 
            className="mb-2 border-0 shadow-sm",
            style={"borderLeft": f"4px solid {color}"},
            id={"type": "unified-context-type-card", "context_type": item_type}
            )
        )
    
    # === GPT ANALYSIS SUMMARY (Expandable) ===
    gpt_summary_section = html.Div([
        html.H6([
            html.I(className="fas fa-brain me-2", style={"color": "#6f42c1"}),
            "GPT Analysis Summary"
        ], className="mb-2 small"),
        dbc.Collapse([
            html.Div([
                html.Pre(
                    gpt_analysis_summary[:500] + "..." if len(gpt_analysis_summary) > 500 else gpt_analysis_summary,
                    style={
                        "fontSize": "12px",
                        "backgroundColor": "#f8f9fa",
                        "padding": "12px",
                        "borderRadius": "4px",
                        "border": "1px solid #dee2e6",
                        "whiteSpace": "pre-wrap",
                        "maxHeight": "200px",
                        "overflowY": "auto"
                    }
                )
            ])
        ], id="gpt-analysis-collapse", is_open=False),
        dbc.Button([
            html.I(className="fas fa-chevron-down me-1", id="gpt-analysis-chevron"),
            "Show GPT Analysis"
        ], color="link", size="sm", id="gpt-analysis-toggle", className="p-0 text-decoration-none")
    ], className="mb-3") if gpt_analysis_summary else html.Div()
    
    return html.Div([
        html.H5([
            html.I(className="fas fa-chart-pie me-2", style={"color": "#614385"}),
            "Analysis Overview"
        ], className="mb-3"),
        
        # Layer 1: Enhanced Severity Statistics
        severity_section,
        
        # Strategy recommendation
        strategy_card,
        
        # Context items breakdown
        html.H6("Context Items by Type", className="mb-2 small"),
        html.Div(type_cards),
        
        # GPT analysis summary
        gpt_summary_section,
        
        html.Hr(className="my-3"),
        
        html.Small([
            html.I(className="fas fa-info-circle me-1"),
            f"Analysis powered by GPT-4. Total issues analyzed: {len(all_context_items)}. ",
            "Switch to the right panel to view comprehensive analysis."
        ], className="text-muted")
    ])


# =============================================================================
# UNIFIED ANALYSIS CALLBACKS
# =============================================================================

@callback(
    [Output("unified-selected-strategy", "data"),
     Output("unified-strategy-retrain-btn", "color"),
     Output("unified-strategy-retrain-btn", "outline"),
     Output("unified-strategy-finetune-btn", "color"),
     Output("unified-strategy-finetune-btn", "outline"),
     Output("unified-strategy-monitor-btn", "color"),
     Output("unified-strategy-monitor-btn", "outline")],
    [Input("unified-strategy-retrain-btn", "n_clicks"),
     Input("unified-strategy-finetune-btn", "n_clicks"),
     Input("unified-strategy-monitor-btn", "n_clicks")],
    [State("unified-selected-strategy", "data")],
    prevent_initial_call=True
)
def handle_unified_strategy_selection(retrain_clicks, finetune_clicks, monitor_clicks, current_strategy):
    """Handle unified strategy selection and update button states."""
    ctx = dash.callback_context
    
    if not ctx.triggered:
        # Set default button states
        if current_strategy == "retrain":
            return "retrain", "danger", False, "outline-warning", True, "outline-info", True
        elif current_strategy == "finetune":
            return "finetune", "outline-danger", True, "warning", False, "outline-info", True
        else:  # monitor
            return "monitor", "outline-danger", True, "outline-warning", True, "info", False
    
    triggered_button = ctx.triggered[0]["prop_id"].split(".")[0]
    
    if triggered_button == "unified-strategy-retrain-btn":
        selected_strategy = "retrain"
        return "retrain", "danger", False, "outline-warning", True, "outline-info", True
    elif triggered_button == "unified-strategy-finetune-btn":
        selected_strategy = "finetune"
        return "finetune", "outline-danger", True, "warning", False, "outline-info", True
    else:  # monitor
        selected_strategy = "monitor"
        return "monitor", "outline-danger", True, "outline-warning", True, "info", False


@callback(
    Output("unified-analysis-results", "children", allow_duplicate=True),
    [Input("unified-technical-perspective-btn", "n_clicks"),
     Input("unified-business-perspective-btn", "n_clicks"),
     Input("unified-executive-perspective-btn", "n_clicks"),
     Input("unified-selected-strategy", "data")],
    [State("explain-context-data", "data"),
     State("unified-strategy-analysis", "data")],
    prevent_initial_call=True
)
def switch_unified_analysis_perspective(technical_clicks, business_clicks, executive_clicks, 
                                      selected_strategy, all_context_items, strategy_analysis):
    """Handle switching between analysis perspectives for unified analysis."""
    ctx = dash.callback_context
    
    if not ctx.triggered or not all_context_items:
        return dash.no_update
    
    triggered_button = ctx.triggered[0]["prop_id"].split(".")[0]
    
    # Import unified analysis functions
    from .unified_analysis import create_unified_technical_perspective_content
    from .unified_gpt_analysis import analyze_all_contexts_with_gpt
    
    try:
        if triggered_button == "unified-technical-perspective-btn" or triggered_button == "unified-selected-strategy":
            # Technical perspective - no GPT needed
            return create_unified_technical_perspective_content(
                all_context_items, strategy_analysis, selected_strategy
            )
        
        elif triggered_button == "unified-business-perspective-btn":
            # Business perspective - use GPT
            gpt_analysis = analyze_all_contexts_with_gpt(all_context_items, selected_strategy, "business")
            return create_unified_business_perspective_content(all_context_items, gpt_analysis, selected_strategy)
        
        elif triggered_button == "unified-executive-perspective-btn":
            # Executive perspective - use GPT
            gpt_analysis = analyze_all_contexts_with_gpt(all_context_items, selected_strategy, "executive")
            return create_unified_executive_perspective_content(all_context_items, gpt_analysis, selected_strategy)
        
        else:
            return dash.no_update
            
    except Exception as e:
        print(f"[UNIFIED PERSPECTIVE] Error switching perspective: {str(e)}")
        return html.Div([
            dbc.Alert([
                html.I(className="fas fa-exclamation-triangle me-2"),
                html.Strong("Analysis Error"),
                html.Br(),
                f"Error generating {triggered_button.replace('unified-', '').replace('-perspective-btn', '')} perspective: {str(e)}"
            ], color="warning")
        ])


@callback(
    [Output("unified-technical-perspective-btn", "active"),
     Output("unified-business-perspective-btn", "active"),
     Output("unified-executive-perspective-btn", "active")],
    [Input("unified-technical-perspective-btn", "n_clicks"),
     Input("unified-business-perspective-btn", "n_clicks"),
     Input("unified-executive-perspective-btn", "n_clicks")],
    prevent_initial_call=True
)
def update_unified_perspective_button_states(technical_clicks, business_clicks, executive_clicks):
    """Update active state of unified perspective buttons."""
    ctx = dash.callback_context
    
    if not ctx.triggered:
        return True, False, False  # Default to technical active
    
    triggered_button = ctx.triggered[0]["prop_id"].split(".")[0]
    
    if triggered_button == "unified-technical-perspective-btn":
        return True, False, False
    elif triggered_button == "unified-business-perspective-btn":
        return False, True, False
    elif triggered_button == "unified-executive-perspective-btn":
        return False, False, True
    
    return True, False, False  # Default fallback


def create_unified_business_perspective_content(all_context_items, gpt_analysis, selected_strategy):
    """Create unified business perspective content."""
    if gpt_analysis.get("error", False):
        return gpt_analysis.get("content", html.Div("Analysis encountered a problem"))
    
    content = [
        html.H5([
            html.I(className="fas fa-briefcase me-2", style={"color": "#198754"}),
            f"Business Analysis - {selected_strategy.upper()} Strategy"
        ]),
        html.Hr()
    ]
    
    # Add the GPT analysis content
    if gpt_analysis.get("content"):
        content.append(gpt_analysis["content"])
    else:
        content.append(html.P("Business analysis content not available.", className="text-muted"))
    
    return html.Div(content)


def create_unified_executive_perspective_content(all_context_items, gpt_analysis, selected_strategy):
    """Create unified executive perspective content."""
    if gpt_analysis.get("error", False):
        return gpt_analysis.get("content", html.Div("Analysis encountered a problem"))
    
    content = [
        html.H5([
            html.I(className="fas fa-user-tie me-2", style={"color": "#0dcaf0"}),
            f"Executive Summary - {selected_strategy.upper()} Strategy"
        ]),
        html.Hr()
    ]
    
    # Add the GPT analysis content
    if gpt_analysis.get("content"):
        content.append(gpt_analysis["content"])
    else:
        content.append(html.P("Executive analysis content not available.", className="text-muted"))
    
    return html.Div(content)


# === NEW CALLBACKS FOR CONTEXT ITEMS INTERACTION ===

# === SIMPLE EXPAND/COLLAPSE CALLBACK ===

@callback(
    [Output({"type": "item-collapse", "item_id": MATCH}, "is_open"),
     Output({"type": "expand-icon", "item_id": MATCH}, "className")],
    [Input({"type": "expand-toggle", "item_id": MATCH}, "n_clicks")],
    [State({"type": "item-collapse", "item_id": MATCH}, "is_open")],
    prevent_initial_call=True
)
def toggle_context_item_expansion(n_clicks, is_open):
    """
    Simple toggle for individual context item expansion.
    Uses MATCH pattern for clean, isolated behavior.
    """
    if not n_clicks:
        raise PreventUpdate
    
    # Toggle the collapse state
    new_is_open = not is_open
    
    # Update icon based on new state
    new_icon_class = "fas fa-chevron-up" if new_is_open else "fas fa-chevron-down"
    
    return new_is_open, new_icon_class


# Note: Toast auto-hides due to duration=3000 setting, no additional callback needed

# =============================================================================
# GPT ANALYSIS SUMMARY TOGGLE CALLBACK
# =============================================================================

@callback(
    [Output("gpt-analysis-collapse", "is_open"),
     Output("gpt-analysis-chevron", "className"),
     Output("gpt-analysis-toggle", "children")],
    [Input("gpt-analysis-toggle", "n_clicks")],
    [State("gpt-analysis-collapse", "is_open")],
    prevent_initial_call=True
)
def toggle_gpt_analysis_summary(n_clicks, is_open):
    """
    Toggle the GPT analysis summary collapse/expand state.
    Part of Layer 1: Severity Statistics enhancement.
    """
    if n_clicks:
        new_state = not is_open
        
        if new_state:  # Opening
            chevron_class = "fas fa-chevron-up me-1"
            button_text = [
                html.I(className=chevron_class, id="gpt-analysis-chevron"),
                "Hide GPT Analysis"
            ]
        else:  # Closing
            chevron_class = "fas fa-chevron-down me-1" 
            button_text = [
                html.I(className=chevron_class, id="gpt-analysis-chevron"),
                "Show GPT Analysis"
            ]
        
        return new_state, chevron_class, button_text
    
    return dash.no_update, dash.no_update, dash.no_update


# =============================================================================
# ANALYSIS LOADING STATE CONTROL
# =============================================================================

@callback(
    [Output("analysis-ready-state", "style"),
     Output("analysis-loading-state", "style")], 
    [Input("start-context-analysis-btn", "n_clicks")],
    prevent_initial_call=True
)
def toggle_analysis_loading_state(n_clicks):
    """
    Control the loading state visibility during GPT analysis.
    Shows loading state when analysis starts.
    """
    if n_clicks:
        # Show loading state, hide ready state
        ready_style = {"padding": "60px 20px", "display": "none"}
        loading_style = {"padding": "60px 20px", "display": "block"}
        
        return ready_style, loading_style
    
    return dash.no_update, dash.no_update


# =============================================================================
# COMPREHENSIVE FOUR-LAYER ANALYSIS PANEL
# =============================================================================

def create_comprehensive_analysis_panel(context_data, strategy_analysis):
    """
    Create comprehensive four-layer analysis panel integrating all analysis components.
    
    Args:
        context_data: List of context items
        strategy_analysis: Analysis data from GPT with comprehensive_data
        
    Returns:
        html.Div: Complete four-layer analysis interface
    """
    if not context_data:
        return html.Div([
            dbc.Alert([
                html.I(className="fas fa-info-circle me-2"),
                "No context items available for analysis."
            ], color="info")
        ])
    
    # Get user expertise level (could be from user context)
    user_expertise_level = "intermediate"  # Default, could be dynamic
    
    return html.Div([
        # Store needed for context item callbacks
        dcc.Store(id="unified-strategy-analysis", data=strategy_analysis),
        
        # Add Context Item Modal
        create_context_item_modal(),
        
        # Layer 1: Severity Statistics (Enhanced)
        html.Div([
            html.H5([
                html.I(className="fas fa-layer-group me-2", style={"color": "#dc3545"}),
                "Layer 1: Severity Statistics"
            ], className="mb-3 layer-title", style={"color": "#dc3545"}),
            create_enhanced_severity_statistics(context_data, strategy_analysis)
        ], className="layer-section"),
        
        # Layer 2: Context Item Boxes
        html.Div([
            html.H5([
                html.I(className="fas fa-th-large me-2", style={"color": "#6f42c1"}),
                "Layer 2: Context Item Analysis"
            ], className="mb-3 layer-title", style={"color": "#6f42c1"}),
            html.Div(
                create_context_item_boxes(context_data, strategy_analysis, user_expertise_level),
                className="context-item-boxes"
            )
        ], className="layer-section"),
        
        # Layer 3: Joint Analysis
        html.Div([
            html.H5([
                html.I(className="fas fa-project-diagram me-2", style={"color": "#e74c3c"}),
                "Layer 3: Joint Analysis"
            ], className="mb-3 layer-title", style={"color": "#e74c3c"}),
            create_joint_analysis_component(context_data, {"comprehensive_data": strategy_analysis.get("comprehensive_data", {})})
        ], className="layer-section"),
        
        # Layer 4: Strategy Selection
        html.Div([
            html.H5([
                html.I(className="fas fa-route me-2", style={"color": "#8e44ad"}),
                "Layer 4: Strategy Selection"
            ], className="mb-3 layer-title", style={"color": "#8e44ad"}),
            create_strategy_selection_component(context_data, {"comprehensive_data": strategy_analysis.get("comprehensive_data", {})})
        ], className="layer-section"),
        
    ], style={
        "backgroundColor": "#ffffff",
        "padding": "20px",
        "borderRadius": "8px",
        "border": "1px solid #dee2e6"
    })


def create_enhanced_severity_statistics(context_data, strategy_analysis):
    """Create enhanced severity statistics display for Layer 1."""
    # Get severity data
    comprehensive_data = strategy_analysis.get("comprehensive_data", {})
    layer1_data = comprehensive_data.get("layer1_severity_statistics", {})
    
    high_count = layer1_data.get("high_count", 0)
    medium_count = layer1_data.get("medium_count", 0)
    low_count = layer1_data.get("low_count", len(context_data))
    overall_risk = layer1_data.get("overall_risk_level", "Medium")
    confidence = layer1_data.get("confidence_score", 0.5)
    summary = layer1_data.get("summary", "No summary available")
    
    # Risk level styling
    risk_styles = {
        "Low": {"color": "#28a745", "bg": "#d4edda"},
        "Medium": {"color": "#ffc107", "bg": "#fff3cd"},
        "High": {"color": "#fd7e14", "bg": "#ffe8d1"},
        "Critical": {"color": "#dc3545", "bg": "#f8d7da"}
    }
    risk_style = risk_styles.get(overall_risk, risk_styles["Medium"])
    
    return html.Div([
        # Metrics row - Responsive grid
        dbc.Row([
            # Critical Issues
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            html.I(className="fas fa-fire", 
                                  style={"fontSize": "1.5rem", "color": "#dc3545"}),
                            html.H5(str(high_count), className="mt-1 mb-1", 
                                   style={"color": "#dc3545", "fontWeight": "bold"}),
                            html.Small("High", className="text-muted")
                        ], className="text-center")
                    ], className="py-2 px-2")
                ], style={"border": "2px solid #dc3545"}, className="h-100 severity-card")
            ], xs=12, sm=6, md=6, lg=6),
            
            # Moderate Issues
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            html.I(className="fas fa-exclamation-circle", 
                                  style={"fontSize": "1.5rem", "color": "#fd7e14"}),
                            html.H5(str(medium_count), className="mt-1 mb-1", 
                                   style={"color": "#fd7e14", "fontWeight": "bold"}),
                            html.Small("Medium", className="text-muted")
                        ], className="text-center")
                    ], className="py-2 px-2")
                ], style={"border": "2px solid #fd7e14"}, className="h-100 severity-card")
            ], xs=12, sm=6, md=6, lg=6),
            
            # Minor Issues
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            html.I(className="fas fa-check-circle", 
                                  style={"fontSize": "1.5rem", "color": "#28a745"}),
                            html.H5(str(low_count), className="mt-1 mb-1", 
                                   style={"color": "#28a745", "fontWeight": "bold"}),
                            html.Small("Low", className="text-muted")
                        ], className="text-center")
                    ], className="py-2 px-2")
                ], style={"border": "2px solid #28a745"}, className="h-100 severity-card")
            ], xs=12, sm=6, md=6, lg=6),
            
            # Overall Risk Level
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            html.I(className="fas fa-shield-alt", 
                                  style={"fontSize": "1.5rem", "color": risk_style["color"]}),
                            html.H6(overall_risk, className="mt-1 mb-1", 
                                   style={"color": risk_style["color"], "fontWeight": "bold"}),
                            html.Small("Overall", className="text-muted")
                        ], className="text-center")
                    ], className="py-2 px-2")
                ], style={"border": f"2px solid {risk_style['color']}", 
                         "backgroundColor": risk_style["bg"]}, className="h-100 severity-card")
            ], xs=12, sm=6, md=6, lg=6)
        ], className="g-2 mb-3"),
        
        # Summary only (removed confidence per user request)
        html.Div([
            html.H6("GPT Analysis Summary", className="mb-2", style={"fontSize": "1rem"}),
            html.P(summary, className="text-justify", style={"lineHeight": "1.5", "fontSize": "0.9rem", "marginBottom": "0"})
        ], style={
            "backgroundColor": "#f8f9fa",
            "padding": "12px",
            "borderRadius": "6px",
            "border": "1px solid #dee2e6"
        })
    ])


