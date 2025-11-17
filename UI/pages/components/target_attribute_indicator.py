"""
Target Attribute Indicator Component

This component provides a persistent, enhanced display of the selected target attribute
throughout the analysis pipeline (Detect and Explain stages). It shows key statistics
comparing the target attribute across both datasets and serves as a visual anchor for
the user during their analysis journey.

Features:
- Prominent display of target attribute name and type
- Key statistics comparison between primary and secondary datasets
- Expandable panel for more detailed information
- Visual emphasis as a central element of the analysis
"""

import dash
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
from UI.functions.global_vars import global_vars


def create_target_attribute_indicator():
    """
    Create an enhanced target attribute indicator component that serves as a visual
    anchor throughout the analysis pipeline.
    
    Returns:
        html.Div: The target attribute indicator component
    """
    return html.Div([
        # Main indicator container with hover effects
        html.Div([
            # Target attribute header with icon
            html.Div([
                html.I(className="fas fa-bullseye me-2", 
                       style={"color": "#614385"}),
                html.Span("Target: ", className="fw-bold"),
                html.Span(id="target-attribute-name", className="target-name"),
                html.Span(id="target-attribute-type", className="target-type ms-2")
            ], className="d-flex align-items-center"),
            
            # Quick stats preview (always visible)
            html.Div([
                html.Div([
                    html.Span("Distribution: ", className="stat-label"),
                    html.Span(id="target-distribution-summary", className="stat-value")
                ], className="quick-stat"),
                html.Div([
                    html.Span("Related Attributes: ", className="stat-label"),
                    html.Span(id="target-related-count", className="stat-value")
                ], className="quick-stat")
            ], className="d-flex justify-content-between mt-2"),
            
            # Expand/collapse button
            html.Button([
                html.I(className="fas fa-chevron-down")
            ], id="target-details-toggle", className="expand-button"),
            
        ], className="target-indicator-main"),
        
        # Expandable detailed section (hidden by default)
        html.Div([
            # Dataset comparison tabs
            dbc.Tabs([
                dbc.Tab([
                    html.Div(id="target-detailed-stats", className="p-3")
                ], label="Comparison"),
                dbc.Tab([
                    html.Div(id="target-distribution-chart-container", className="p-3")
                ], label="Distribution")
            ], id="target-detail-tabs"),
            
        ], id="target-details-panel", className="target-details")
        
    ], id="target-attribute-indicator", className="target-attribute-indicator")


@callback(
    Output("target-attribute-name", "children"),
    Output("target-attribute-type", "children"),
    Output("target-distribution-summary", "children"),
    Output("target-related-count", "children"),
    Input("target-attribute-indicator", "id")
)
def update_target_indicator(trigger):
    """
    Update the target attribute indicator with current data.
    
    Args:
        trigger: Dummy input to trigger the callback
        
    Returns:
        tuple: (target name, target type, distribution summary, related count)
    """
    # Get target attribute name
    target_attr = global_vars.target_attribute
    
    if not target_attr:
        return "Not selected", "(None)", "N/A", "0"
    
    # Get target attribute type
    target_type = "Unknown"
    if global_vars.primary_dataset is not None and target_attr in global_vars.primary_dataset.columns:
        col_type = global_vars.primary_dataset[target_attr].dtype
        if pd.api.types.is_numeric_dtype(col_type):
            if len(global_vars.primary_dataset[target_attr].unique()) <= 2:
                target_type = "(Binary)"
            else:
                target_type = "(Numeric)"
        elif pd.api.types.is_categorical_dtype(col_type) or len(global_vars.primary_dataset[target_attr].unique()) < 10:
            target_type = "(Categorical)"
        else:
            target_type = "(Text)"
    
    # Generate distribution summary
    distribution_summary = "No comparison data"
    if global_vars.primary_dataset is not None and global_vars.secondary_dataset is not None:
        if target_attr in global_vars.primary_dataset.columns and target_attr in global_vars.secondary_dataset.columns:
            try:
                # For categorical/binary
                if target_type in ["(Binary)", "(Categorical)"]:
                    p_counts = global_vars.primary_dataset[target_attr].value_counts(normalize=True)
                    s_counts = global_vars.secondary_dataset[target_attr].value_counts(normalize=True)
                    
                    # Find the most changed category
                    max_diff = 0
                    max_cat = ""
                    for cat in set(p_counts.index) & set(s_counts.index):
                        diff = abs(p_counts.get(cat, 0) - s_counts.get(cat, 0))
                        if diff > max_diff:
                            max_diff = diff
                            max_cat = cat
                    
                    if max_cat and max_diff > 0.05:  # Only report if significant change
                        p_pct = p_counts.get(max_cat, 0) * 100
                        s_pct = s_counts.get(max_cat, 0) * 100
                        change = s_pct - p_pct
                        distribution_summary = f"{max_cat}: {change:+.1f}% shift"
                    else:
                        distribution_summary = "Minor changes"
                
                # For numeric
                elif target_type == "(Numeric)":
                    p_mean = global_vars.primary_dataset[target_attr].mean()
                    s_mean = global_vars.secondary_dataset[target_attr].mean()
                    p_std = global_vars.primary_dataset[target_attr].std()
                    
                    if abs(p_mean) > 0.001:  # Avoid division by zero or very small numbers
                        pct_change = ((s_mean - p_mean) / abs(p_mean)) * 100
                        std_change = (s_mean - p_mean) / p_std if p_std > 0 else 0
                        
                        if abs(pct_change) > 5:  # Only report if significant change
                            distribution_summary = f"Mean: {pct_change:+.1f}% ({std_change:+.1f}σ)"
                        else:
                            distribution_summary = "Minor changes"
                    else:
                        distribution_summary = "Cannot calculate"
            except Exception as e:
                distribution_summary = "Error calculating"
    
    # Count related attributes
    related_count = len(global_vars.target_attribute_stats.get("related_attributes", []))
    related_text = f"{related_count} attributes"
    
    return target_attr, target_type, distribution_summary, related_text


@callback(
    Output("target-details-panel", "style"),
    Input("target-details-toggle", "n_clicks"),
    State("target-details-panel", "style"),
    prevent_initial_call=True
)
def toggle_details_panel(n_clicks, current_style):
    """
    Toggle the visibility of the detailed target information panel.
    
    Args:
        n_clicks: Number of clicks on the toggle button
        current_style: Current style dictionary
        
    Returns:
        dict: Updated style dictionary
    """
    if current_style and current_style.get("display") == "block":
        return {"display": "none"}
    else:
        return {"display": "block"}


@callback(
    Output("target-detailed-stats", "children"),
    Input("target-detail-tabs", "active_tab"),
    State("target-attribute-name", "children")
)
def update_target_details(active_tab, target_attr):
    """
    Update the detailed statistics panel for the target attribute.
    
    Args:
        active_tab: Currently active tab
        target_attr: Target attribute name
        
    Returns:
        html.Div: Detailed statistics content
    """
    if not target_attr or target_attr == "Not selected":
        return html.P("No target attribute selected")
    
    if active_tab and "comparison" in active_tab.lower():
        # Get datasets
        primary_df = global_vars.primary_dataset
        secondary_df = global_vars.secondary_dataset
        
        if primary_df is None or secondary_df is None:
            return html.P("Dataset comparison not available")
        
        if target_attr not in primary_df.columns or target_attr not in secondary_df.columns:
            return html.P(f"'{target_attr}' not found in both datasets")
        
        # Create comparison table
        try:
            stats = []
            
            # Basic stats for both datasets
            p_series = primary_df[target_attr]
            s_series = secondary_df[target_attr]
            
            if pd.api.types.is_numeric_dtype(p_series):
                # Numeric statistics
                p_stats = {
                    "Mean": p_series.mean(),
                    "Median": p_series.median(),
                    "Std Dev": p_series.std(),
                    "Min": p_series.min(),
                    "Max": p_series.max(),
                    "Missing": p_series.isna().sum() / len(p_series)
                }
                
                s_stats = {
                    "Mean": s_series.mean(),
                    "Median": s_series.median(),
                    "Std Dev": s_series.std(),
                    "Min": s_series.min(),
                    "Max": s_series.max(),
                    "Missing": s_series.isna().sum() / len(s_series)
                }
                
                for stat_name in p_stats:
                    p_val = p_stats[stat_name]
                    s_val = s_stats[stat_name]
                    
                    # Format as percentage for missing values
                    if stat_name == "Missing":
                        p_formatted = f"{p_val:.1%}"
                        s_formatted = f"{s_val:.1%}"
                        diff = f"{(s_val - p_val):.1%}"
                    else:
                        p_formatted = f"{p_val:.3g}"
                        s_formatted = f"{s_val:.3g}"
                        
                        # Calculate difference and percent change
                        abs_diff = s_val - p_val
                        if abs(p_val) > 0.001:  # Avoid division by zero
                            pct_change = (abs_diff / abs(p_val)) * 100
                            diff = f"{abs_diff:.3g} ({pct_change:+.1f}%)"
                        else:
                            diff = f"{abs_diff:.3g}"
                    
                    stats.append({
                        "Statistic": stat_name,
                        "Primary": p_formatted,
                        "Secondary": s_formatted,
                        "Difference": diff
                    })
            else:
                # Categorical statistics
                p_counts = p_series.value_counts(normalize=True)
                s_counts = s_series.value_counts(normalize=True)
                
                # Get all unique categories
                all_cats = sorted(set(list(p_counts.index) + list(s_counts.index)))
                
                # Top 5 categories by total frequency
                top_cats = []
                for cat in all_cats:
                    p_val = p_counts.get(cat, 0)
                    s_val = s_counts.get(cat, 0)
                    top_cats.append((cat, p_val + s_val))
                
                top_cats.sort(key=lambda x: x[1], reverse=True)
                top_cats = [cat for cat, _ in top_cats[:5]]
                
                # Calculate statistics for each category
                for cat in top_cats:
                    p_val = p_counts.get(cat, 0)
                    s_val = s_counts.get(cat, 0)
                    
                    # Format as percentages
                    p_formatted = f"{p_val:.1%}"
                    s_formatted = f"{s_val:.1%}"
                    
                    # Calculate absolute and relative differences
                    abs_diff = s_val - p_val
                    diff = f"{abs_diff:.1%}"
                    
                    stats.append({
                        "Statistic": str(cat),
                        "Primary": p_formatted,
                        "Secondary": s_formatted,
                        "Difference": diff
                    })
                
                # Add missing rate
                p_missing = p_series.isna().sum() / len(p_series)
                s_missing = s_series.isna().sum() / len(s_series)
                
                stats.append({
                    "Statistic": "Missing",
                    "Primary": f"{p_missing:.1%}",
                    "Secondary": f"{s_missing:.1%}",
                    "Difference": f"{(s_missing - p_missing):.1%}"
                })
            
            # Create the table
            table = dbc.Table([
                html.Thead([
                    html.Tr([
                        html.Th("Statistic"),
                        html.Th("Primary"),
                        html.Th("Secondary"),
                        html.Th("Difference")
                    ])
                ]),
                html.Tbody([
                    html.Tr([
                        html.Td(row["Statistic"]),
                        html.Td(row["Primary"]),
                        html.Td(row["Secondary"]),
                        html.Td(row["Difference"])
                    ]) for row in stats
                ])
            ], bordered=True, hover=True, size="sm", className="stats-table")
            
            return table
            
        except Exception as e:
            return html.P(f"Error calculating statistics: {str(e)}")
    
    elif active_tab and "distribution" in active_tab.lower():
        # This would be filled with distribution charts
        # For now just placeholder text
        return html.P("Distribution charts will be implemented here")
    
    return html.P("Select a tab to view details")


@callback(
    Output("target-distribution-chart-container", "children"),
    Input("target-detail-tabs", "active_tab"),
    State("target-attribute-name", "children")
)
def update_distribution_chart(active_tab, target_attr):
    """
    Update the distribution chart for the target attribute.
    
    Args:
        active_tab: Currently active tab
        target_attr: Target attribute name
        
    Returns:
        html.Div: Distribution chart content
    """
    if not active_tab or "distribution" not in active_tab.lower():
        return []
    
    if not target_attr or target_attr == "Not selected":
        return html.P("No target attribute selected")
    
    # Placeholder for actual chart implementation
    return html.Div([
        html.P("Distribution chart will be implemented here", className="text-center py-4"),
        html.P("This will show comparison of target attribute distribution across datasets", 
               className="text-muted text-center")
    ])
