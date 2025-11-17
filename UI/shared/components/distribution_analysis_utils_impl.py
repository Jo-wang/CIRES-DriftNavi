"""
Canonical implementation of distribution analysis utilities (moved from UI/pages/components/distribution_analysis_utils.py)
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from dash import html, dcc

from UI.functions.global_vars import global_vars
from UI.state_connector import get_explain_context


def analyze_target_distribution(df, target_column):
    if df is None or target_column not in df.columns:
        return {}, "unknown"
    column_data = df[target_column]
    if pd.api.types.is_numeric_dtype(column_data):
        if column_data.dtype == bool or len(column_data.unique()) <= 10:
            column_type = "categorical"
        else:
            column_type = "continuous"
    else:
        column_type = "categorical"
    if column_type == "categorical":
        distribution = column_data.value_counts().to_dict()
    else:
        hist, bin_edges = np.histogram(column_data.dropna(), bins=10)
        distribution = {}
        for i in range(len(hist)):
            bin_label = f"{bin_edges[i]:.2f} to {bin_edges[i+1]:.2f}"
            distribution[bin_label] = int(hist[i])
    return distribution, column_type


def generate_distribution_chart(primary_dist, secondary_dist, column_type):
    if not primary_dist and not secondary_dist:
        return go.Figure().update_layout(
            title="No distribution data available",
            xaxis_title="Value",
            yaxis_title="Count",
            template="plotly_white"
        )
    all_keys = sorted(set(primary_dist.keys()) | set(secondary_dist.keys()))
    x_values = list(all_keys)
    y_primary = [primary_dist.get(key, 0) for key in all_keys]
    y_secondary = [secondary_dist.get(key, 0) for key in all_keys]
    if column_type == "categorical":
        fig = go.Figure()
        fig.add_trace(go.Bar(x=x_values, y=y_primary, name="Primary Dataset", marker_color="#614385"))
        fig.add_trace(go.Bar(x=x_values, y=y_secondary, name="Secondary Dataset", marker_color="#516395"))
        fig.update_layout(barmode="group", xaxis_title="Value", yaxis_title="Count", legend_title="Dataset", template="plotly_white")
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_values, y=y_primary, mode="lines+markers", name="Primary Dataset", line=dict(color="#614385", width=3), marker=dict(size=8)))
        fig.add_trace(go.Scatter(x=x_values, y=y_secondary, mode="lines+markers", name="Secondary Dataset", line=dict(color="#516395", width=3), marker=dict(size=8)))
        fig.update_layout(xaxis_title="Value Range", yaxis_title="Count", legend_title="Dataset", template="plotly_white")
    fig.update_layout(margin=dict(l=40, r=40, t=40, b=40), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig


def create_distribution_chart_component(primary_dist, secondary_dist, column_type, attribute_name=None, include_button=True, stage="detect", for_preview=False):
    fig = generate_distribution_chart(primary_dist, secondary_dist, column_type)
    title = html.H5(f"Distribution of {attribute_name}" if attribute_name else "Distribution Comparison", className="chart-title text-center mt-3 mb-4")
    chart = dcc.Graph(figure=fig, config={'displayModeBar': True, 'scrollZoom': True}, style={"height": "400px"})
    components = [title, chart]
    
    if include_button and not for_preview:
        from UI.utils.button_utils import create_dual_add_buttons
        dual_buttons = create_dual_add_buttons(
            feature_name="target distribution",
            chat_button_id="add-target-dist-to-chat",
            explain_button_id="add-target-dist-to-explain",
            chat_disabled=False,
            explain_disabled=False,
            chat_aria_disabled="false",
            explain_aria_disabled="false"
        )
        
        components.append(dual_buttons)
    
    return html.Div(components, className="distribution-chart-container")


def get_distribution_data(stage="detect"):
    try:
        context = get_explain_context(stage=stage)
        focus_attribute = context.get('focus_attribute')
        target_attribute = context.get('target_attribute')
        
        attribute_to_display = None
        if focus_attribute:
            attribute_to_display = focus_attribute
        elif target_attribute:
            attribute_to_display = target_attribute
        elif hasattr(global_vars, 'target_attribute') and global_vars.target_attribute:
            attribute_to_display = global_vars.target_attribute
        elif hasattr(global_vars, 'df') and global_vars.df is not None and len(global_vars.df.columns) > 0:
            attribute_to_display = global_vars.df.columns[0]
        
        if attribute_to_display is None:
            return {}, {}, "unknown", None
            
        has_primary = hasattr(global_vars, 'df') and global_vars.df is not None
        has_secondary = hasattr(global_vars, 'secondary_df') and global_vars.secondary_df is not None
        
        if not has_primary or not has_secondary:
            return {}, {}, "unknown", attribute_to_display
            
        primary_dist, column_type = analyze_target_distribution(global_vars.df, attribute_to_display)
        secondary_dist, _ = analyze_target_distribution(global_vars.secondary_df, attribute_to_display)
        
        return primary_dist, secondary_dist, column_type, attribute_to_display
        
    except Exception as e:
        print(f"[GET_DISTRIBUTION_DATA] Error: {str(e)}")
        return {}, {}, "unknown", None




