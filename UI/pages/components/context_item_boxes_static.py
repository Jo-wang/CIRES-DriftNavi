"""
Static Context Item Boxes Component for Chat Explain

This is a simplified version of context_item_boxes that doesn't use callbacks
and is designed for use within chat bubbles where interactivity is not needed.
The original context_item_boxes.py with its global store dependencies causes
errors when used in chat context, so this static version provides the same
visual layout without the interactive modal functionality.
"""

from typing import Dict, Any, List
import dash_bootstrap_components as dbc
from dash import html


def create_static_context_item_boxes(
    context_items: List[Dict[str, Any]], 
    analysis_data: Dict[str, Any],
    user_expertise_level: str = "intermediate"
) -> html.Div:
    """
    Create static context item boxes without callbacks for chat use.
    
    Args:
        context_items: List of context items
        analysis_data: Analysis data from GPT containing layer2_context_analysis
        user_expertise_level: User's expertise level (not used in static version)
        
    Returns:
        html.Div: Static grid of context boxes (no modals/callbacks)
    """
    if not context_items:
        return html.Div([
            html.Div([
                html.I(className="fas fa-info-circle me-2", style={"color": "#6c757d"}),
                html.P("No context items detected", className="text-muted mb-0")
            ], className="text-center py-3")
        ])
    
    # Create static boxes
    boxes = []
    for i, context_item in enumerate(context_items):
        box = create_static_context_box(context_item, i, analysis_data)
        boxes.append(box)
    
    return html.Div([
        html.Div(
            boxes,
            className="row g-2"
        )
    ])


def create_static_context_box(
    context_item: Dict[str, Any],
    item_index: int,
    analysis_data: Dict[str, Any]
) -> html.Div:
    """Create a static context box without click handlers."""
    
    # Extract context information
    context_type = context_item.get('type', 'unknown')
    attribute_name = get_attribute_name(context_item)
    type_display = get_context_type_display(context_type)
    type_color = get_context_type_color(context_type)
    
    # Get analysis info for this item
    severity, analysis_summary = get_analysis_info_for_item(context_item, analysis_data, item_index)
    
    return html.Div([
        dbc.Card([
            dbc.CardBody([
                # Context type icon and name
                html.Div([
                    html.I(
                        className=get_context_type_icon(context_type), 
                        style={
                            "fontSize": "1.2rem", 
                            "color": type_color,
                            "minWidth": "20px",
                            "textAlign": "center"
                        }
                    ),
                    html.Div([
                        html.Strong(
                            type_display, 
                            style={
                                "fontSize": "0.85rem",
                                "fontWeight": "600",
                                "color": "#2c3e50",
                                "lineHeight": "1.2",
                                "display": "block",
                                "marginBottom": "2px"
                            }
                        ),
                        html.Small(
                            attribute_name,
                            style={
                                "fontSize": "0.75rem",
                                "color": "#6c757d",
                                "lineHeight": "1.1",
                                "display": "block"
                            }
                        )
                    ], style={"marginLeft": "10px", "flex": "1"})
                ], style={
                    "display": "flex",
                    "alignItems": "center",
                    "height": "100%",
                    "minHeight": "50px"
                }),
                
                # Add severity indicator if available
                html.Div([
                    dbc.Badge(
                        severity,
                        color=get_severity_color(severity),
                        className="mt-2"
                    )
                ] if severity else [])
                
            ], style={
                "padding": "8px 12px",
                "minHeight": "80px"
            })
        ], style={
            "border": f"2px solid {type_color}",
            "borderRadius": "8px",
            "transition": "all 0.2s ease"
        })
    ], className="col-6 col-md-4 col-lg-3")


def get_analysis_info_for_item(context_item, analysis_data, item_index):
    """Extract analysis info for a context item."""
    if not analysis_data or not isinstance(analysis_data, dict):
        return None, None
        
    comprehensive_data = analysis_data.get('comprehensive_data', {})
    layer2_data = comprehensive_data.get('layer2_context_analysis', [])
    
    # Try to find matching analysis
    analysis_item = None
    for analysis in layer2_data:
        if analysis.get('context_id') == item_index:
            analysis_item = analysis
            break
    
    if not analysis_item and 0 <= item_index < len(layer2_data):
        analysis_item = layer2_data[item_index]
        
    if analysis_item:
        severity = analysis_item.get('risk_level', 'Medium')
        return severity, analysis_item.get('business_impact', '')
        
    return None, None


def get_severity_color(severity):
    """Get color for severity level."""
    color_map = {
        'Critical': 'danger',
        'High': 'warning', 
        'Medium': 'info',
        'Low': 'success'
    }
    return color_map.get(severity, 'secondary')


# Helper functions (reuse from original context_item_boxes.py)
def get_attribute_name(context_item):
    """Extract attribute name from context item."""
    # Get the attribute name from different possible sources
    if context_item.get('type') == 'target_distribution':
        return context_item.get('target_attribute', 'target')
    elif context_item.get('type') == 'conditional_distribution':
        conditional_attr = context_item.get('conditional_attribute', 'unknown')
        target_value = context_item.get('target_value', 'unknown')
        return f"{conditional_attr} = {target_value}"
    elif context_item.get('type') == 'metrics_summary':
        return context_item.get('focus_attribute', 'metrics')
    else:
        # Fallback: try to extract from title or other fields
        title = context_item.get('title', '')
        if 'Analysis:' in title:
            return title.split('Analysis:')[-1].strip()
        return context_item.get('attribute_name', 'unknown')


def get_context_type_display(context_type):
    """Get display name for context type."""
    display_map = {
        'target_distribution': 'Target Distribution',
        'conditional_distribution': 'Conditional Analysis', 
        'metrics_summary': 'Metrics Summary',
        'drift_analysis': 'Drift Analysis',
        'drift_detection': 'drift Detection'
    }
    return display_map.get(context_type, context_type.replace('_', ' ').title())


def get_context_type_color(context_type):
    """Get color for context type."""
    color_map = {
        'target_distribution': '#e74c3c',
        'conditional_distribution': '#3498db',
        'metrics_summary': '#9b59b6',
        'drift_analysis': '#f39c12',
        'drift_detection': '#27ae60'
    }
    return color_map.get(context_type, '#6c757d')


def get_context_type_icon(context_type):
    """Get icon for context type."""
    icon_map = {
        'target_distribution': 'fas fa-bullseye',
        'conditional_distribution': 'fas fa-filter',
        'metrics_summary': 'fas fa-chart-bar',
        'drift_analysis': 'fas fa-trending-up',
        'drift_detection': 'fas fa-balance-scale'
    }
    return icon_map.get(context_type, 'fas fa-question-circle')

