"""
Correlation Analysis Module for driftNavi

This module provides functionality for analyzing correlation shifts between datasets
and visualizing these differences to help users understand distribution changes.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from dash import html, dcc
import dash_bootstrap_components as dbc
from scipy.stats import fisher_exact
from scipy.stats import ttest_ind
import scipy.cluster.hierarchy as sch
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

def calculate_correlation_shift(primary_df, secondary_df, method='pearson', threshold=0.2):
    """
    Calculate correlation matrices for both datasets and their difference.
    
    Parameters:
    -----------
    primary_df : pandas DataFrame
        The primary dataset
    secondary_df : pandas DataFrame
        The secondary dataset for comparison
    method : str, default='pearson'
        Correlation method ('pearson', 'spearman', or 'kendall')
    threshold : float, default=0.2
        Threshold for significant correlation changes
    
    Returns:
    --------
    tuple
        (primary_corr, secondary_corr, diff_corr, significant_changes)
    """
    # Filter to include only numeric columns present in both dataframes
    numeric_cols = [col for col in primary_df.columns if 
                    col in secondary_df.columns and 
                    pd.api.types.is_numeric_dtype(primary_df[col]) and 
                    pd.api.types.is_numeric_dtype(secondary_df[col])]
    
    if not numeric_cols:
        return None, None, None, None
    
    # Calculate correlation matrices
    primary_corr = primary_df[numeric_cols].corr(method=method)
    secondary_corr = secondary_df[numeric_cols].corr(method=method)
    
    # Calculate differences and find significant changes
    diff_corr = secondary_corr - primary_corr
    
    # Find significant correlation changes
    significant_changes = []
    
    for i, col1 in enumerate(numeric_cols):
        for j, col2 in enumerate(numeric_cols):
            if i < j:  # Only consider upper triangle to avoid duplicates
                diff_value = abs(diff_corr.loc[col1, col2])
                if diff_value >= threshold:
                    # Determine if the correlation increased or decreased
                    change_type = "increased" if secondary_corr.loc[col1, col2] > primary_corr.loc[col1, col2] else "decreased"
                    significant_changes.append({
                        'feature_1': col1,
                        'feature_2': col2,
                        'primary_corr': primary_corr.loc[col1, col2],
                        'secondary_corr': secondary_corr.loc[col1, col2],
                        'diff': diff_value,
                        'change_type': change_type
                    })
    
    # Sort by absolute difference magnitude
    significant_changes = sorted(significant_changes, key=lambda x: x['diff'], reverse=True)
    
    return primary_corr, secondary_corr, diff_corr, significant_changes

def generate_correlation_heatmaps(primary_corr, secondary_corr, diff_corr):
    """
    Generate plotly figures comparing correlation matrices.
    
    Parameters:
    -----------
    primary_corr : pandas DataFrame
        Correlation matrix for primary dataset
    secondary_corr : pandas DataFrame
        Correlation matrix for secondary dataset  
    diff_corr : pandas DataFrame
        Difference between correlation matrices
    
    Returns:
    --------
    tuple
        (primary_fig, secondary_fig, diff_fig)
    """
    if primary_corr is None or secondary_corr is None or diff_corr is None:
        return None, None, None
    
    # Create heatmap for primary dataset
    primary_fig = go.Figure(data=go.Heatmap(
        z=primary_corr.values,
        x=primary_corr.columns,
        y=primary_corr.index,
        colorscale='Blues',
        zmin=-1, zmax=1
    ))
    primary_fig.update_layout(
        title="Primary Dataset Correlation",
        height=500,
        width=500,
        xaxis=dict(tickangle=45),
        margin=dict(l=50, r=50, t=50, b=100)
    )
    
    # Create heatmap for secondary dataset
    secondary_fig = go.Figure(data=go.Heatmap(
        z=secondary_corr.values,
        x=secondary_corr.columns,
        y=secondary_corr.index,
        colorscale='Reds',
        zmin=-1, zmax=1
    ))
    secondary_fig.update_layout(
        title="Secondary Dataset Correlation",
        height=500,
        width=500,
        xaxis=dict(tickangle=45),
        margin=dict(l=50, r=50, t=50, b=100)
    )
    
    # Create heatmap for difference
    diff_fig = go.Figure(data=go.Heatmap(
        z=diff_corr.values,
        x=diff_corr.columns,
        y=diff_corr.index,
        colorscale='RdBu',
        zmin=-1, zmax=1
    ))
    diff_fig.update_layout(
        title="Correlation Shift (Secondary - Primary)",
        height=500,
        width=500,
        xaxis=dict(tickangle=45),
        margin=dict(l=50, r=50, t=50, b=100)
    )
    
    return primary_fig, secondary_fig, diff_fig

def generate_correlation_network(significant_changes, threshold=0.5):
    """
    Generate a network graph showing significant correlation changes.
    
    Parameters:
    -----------
    significant_changes : list
        List of dictionaries containing significant correlation changes
    threshold : float, default=0.5
        Minimum correlation difference to include in the network
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Network visualization of correlation changes
    """
    if not significant_changes:
        return None
    
    # Filter changes by threshold
    filtered_changes = [change for change in significant_changes if change['diff'] >= threshold]
    
    if not filtered_changes:
        return None
        
    # Get unique features
    all_features = set()
    for change in filtered_changes:
        all_features.add(change['feature_1'])
        all_features.add(change['feature_2'])
    
    # Create node positions using a circular layout
    num_features = len(all_features)
    feature_to_idx = {feature: i for i, feature in enumerate(all_features)}
    
    # Generate circular layout coordinates
    import math
    radius = 1
    node_positions = {}
    for feature, idx in feature_to_idx.items():
        angle = 2 * math.pi * idx / num_features
        node_positions[feature] = (radius * math.cos(angle), radius * math.sin(angle))
    
    # Create edges data
    edge_x = []
    edge_y = []
    edge_text = []
    edge_colors = []
    
    for change in filtered_changes:
        x0, y0 = node_positions[change['feature_1']]
        x1, y1 = node_positions[change['feature_2']]
        
        # Add line segment
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        
        # Edge text and color
        edge_text.append(f"{change['feature_1']} - {change['feature_2']}<br>"
                        f"Primary: {change['primary_corr']:.3f}<br>"
                        f"Secondary: {change['secondary_corr']:.3f}<br>"
                        f"Difference: {change['diff']:.3f}")
        
        # Red for increased correlation, blue for decreased
        edge_colors.extend(['rgba(255,0,0,0.7)' if change['change_type'] == 'increased' 
                          else 'rgba(0,0,255,0.7)'] * 3)
    
    # Create edges trace
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=2, color=edge_colors),
        hoverinfo='text',
        mode='lines',
        text=edge_text
    )
    
    # Create nodes data
    node_x = [pos[0] for pos in node_positions.values()]
    node_y = [pos[1] for pos in node_positions.values()]
    
    # Create nodes trace
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        marker=dict(
            showscale=False,
            size=15,
            color='rgba(44, 160, 44, 0.8)',
            line=dict(width=2)
        ),
        text=list(node_positions.keys()),
        textposition="top center",
        hovertext=list(node_positions.keys())
    )
    
    # Create network figure
    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title="Correlation Shift Network",
        titlefont_size=16,
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20,l=5,r=5,t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=600,
        width=700
    )
    
    # Add legend for edge colors
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='lines',
        line=dict(color='rgba(255,0,0,0.7)', width=4),
        name='Correlation Increased'
    ))
    
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='lines',
        line=dict(color='rgba(0,0,255,0.7)', width=4),
        name='Correlation Decreased'
    ))
    
    fig.update_layout(showlegend=True)
    
    return fig

def generate_correlation_shift_ui(primary_df, secondary_df, method='pearson', threshold=0.2):
    """
    Generate complete UI components for correlation shift analysis.
    
    Parameters:
    -----------
    primary_df : pandas DataFrame
        The primary dataset
    secondary_df : pandas DataFrame
        The secondary dataset for comparison
    method : str, default='pearson'
        Correlation method ('pearson', 'spearman', or 'kendall')
    threshold : float, default=0.2
        Threshold for significant correlation changes
        
    Returns:
    --------
    dash_html_components.Div
        Complete UI component for correlation shift analysis
    """
    # Calculate correlations and their differences
    primary_corr, secondary_corr, diff_corr, significant_changes = calculate_correlation_shift(
        primary_df, secondary_df, method, threshold
    )
    
    if primary_corr is None:
        return html.Div([
            html.H3("Correlation Shift Analysis", className="mb-4"),
            html.Div("No numeric columns found in both datasets for correlation analysis.", 
                     className="alert alert-warning")
        ])
    
    # Generate heatmaps
    primary_fig, secondary_fig, diff_fig = generate_correlation_heatmaps(
        primary_corr, secondary_corr, diff_corr
    )
    
    # Generate network visualization
    network_fig = generate_correlation_network(significant_changes, threshold=threshold)
    
    # Create table of significant changes
    if significant_changes:
        changes_table = dbc.Table(
            [
                html.Thead(
                    html.Tr([
                        html.Th("Feature Pair"),
                        html.Th("Primary Corr."),
                        html.Th("Secondary Corr."),
                        html.Th("Difference"),
                        html.Th("Change")
                    ])
                ),
                html.Tbody([
                    html.Tr([
                        html.Td(f"{change['feature_1']} & {change['feature_2']}"),
                        html.Td(f"{change['primary_corr']:.3f}"),
                        html.Td(f"{change['secondary_corr']:.3f}"),
                        html.Td(f"{change['diff']:.3f}"),
                        html.Td(
                            html.Span(
                                "↑ Increased" if change['change_type'] == "increased" else "↓ Decreased",
                                style={
                                    'color': 'red' if change['change_type'] == "increased" else 'blue',
                                    'font-weight': 'bold'
                                }
                            )
                        )
                    ]) for change in significant_changes[:10]  # Limit to top 10 changes
                ])
            ],
            bordered=True,
            hover=True,
            responsive=True,
            striped=True,
            className="mt-3"
        )
    else:
        changes_table = html.Div("No significant correlation shifts found.", className="alert alert-info mt-3")
    
    return html.Div([
        html.H3("Correlation Shift Analysis", className="mb-4"),
        html.P([
            "This analysis shows how relationships between features have changed between datasets. ",
            "Large shifts in correlation may indicate fundamental changes in the data generation process."
        ]),
        
        # Summary of findings
        html.Div([
            html.H4("Key Findings", className="mt-4 mb-3"),
            html.Div([
                html.P([
                    f"Found ",
                    html.Strong(f"{len(significant_changes)}"),
                    f" significant correlation shifts (threshold: {threshold})."
                ]),
                changes_table
            ])
        ]),
        
        # Network visualization
        html.Div([
            html.H4("Correlation Shift Network", className="mt-4 mb-3"),
            html.P("This network shows features with significant correlation changes. "
                  "Red lines indicate increased correlation, blue lines indicate decreased correlation."),
            dcc.Graph(figure=network_fig) if network_fig else html.Div("No significant network changes to display.")
        ]),
        
        # Correlation matrices
        html.Div([
            html.H4("Correlation Matrices", className="mt-4 mb-3"),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=primary_fig), width=4),
                dbc.Col(dcc.Graph(figure=secondary_fig), width=4),
                dbc.Col(dcc.Graph(figure=diff_fig), width=4)
            ])
        ])
    ])
