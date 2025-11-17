"""
Strategy Selection Component for DriftNavi

This module provides UI components for displaying Layer 4: Strategy Selection
with simplified focus on the core recommended strategy based on Layer 3 analysis.

Author: DriftNavi Team
Created: 2025
"""

from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
from typing import Dict, Any, List, Optional


# =============================================================================
# STRATEGY SELECTION MAIN COMPONENT 
# =============================================================================

def create_strategy_selection_component(
    context_items: List[Dict[str, Any]], 
    analysis_data: Dict[str, Any]
) -> html.Div:
    """
    Create the strategy selection component showing Layer 4 analysis.
    Simplified to focus only on recommended strategy with clear rationale.
    
    Args:
        context_items: List of context items from explain-context-data
        analysis_data: Analysis data from GPT containing layer4_strategy_selection
        
    Returns:
        html.Div: Simplified strategy selection component
    """
    if not context_items:
        return html.Div([
            html.Div([
                html.I(className="fas fa-info-circle me-2", style={"color": "#6c757d"}),
                html.H6("No Data for Strategy Selection", className="text-muted"),
                html.P("Add context items to get strategic recommendations.", className="text-muted small")
            ], className="text-center py-4")
        ])
    
    # Extract layer4 data from comprehensive analysis
    layer4_data = analysis_data.get('comprehensive_data', {}).get('layer4_strategy_selection', {})
    
    if not layer4_data:
        return create_strategy_selection_fallback(len(context_items))
    
    return html.Div([
        # Header
        html.Div([
            # html.I(className="fas fa-route me-2", style={"color": "#8e44ad"}),
            # html.H6("Strategy Selection", style={"color": "#8e44ad"}),
            html.P("AI-driven recommendation based on comprehensive Layer 3 analysis", 
                  className="text-muted small mb-3")
        ]),
        
        # Main recommendation content
        create_main_recommendation_content(layer4_data),
        
    ], className="mb-4")


def create_main_recommendation_content(layer4_data: Dict[str, Any]) -> html.Div:
    """
    Create the main recommendation content with strategy, confidence, and rationale.
    
    Args:
        layer4_data: Layer 4 data from GPT analysis
        
    Returns:
        html.Div: Complete recommendation content
    """
    recommended_strategy = layer4_data.get('recommended_strategy', 'monitor')
    strategy_overview = layer4_data.get('strategy_overview', 'Strategy overview not available')
    confidence = layer4_data.get('confidence', 0.5)
    reasoning = layer4_data.get('reasoning', 'No reasoning provided')
    
    # Get strategy styling (icon and colors only, not description)
    strategy_info = get_strategy_info(recommended_strategy)
    confidence_color = get_confidence_color(confidence)
    
    return html.Div([
        # Main recommendation card
        dbc.Card([
            dbc.CardHeader([
                html.H5([
                    html.I(className=strategy_info["icon"], 
                          style={"color": strategy_info["color"], "marginRight": "10px"}),
                    "Recommended Strategy: ",
                    html.Span(recommended_strategy.upper(), 
                             style={"color": strategy_info["color"], "fontWeight": "bold"})
                ], className="mb-0")
            ], style={"backgroundColor": strategy_info["bg_color"]}),
            
            dbc.CardBody([
                # Strategy description (from GPT)
                html.Div([
                    html.H6([
                        html.I(className="fas fa-info-circle me-2"),
                        "Strategy Overview"
                    ]),
                    html.P(strategy_overview, className="text-muted mb-4"),
                ]),
                
                # AI reasoning section
                html.Div([
                    html.H6([
                        html.I(className="fas fa-brain me-2"),
                        "AI Analysis & Rationale"
                    ]),
                    format_reasoning_text(reasoning),
                ], className="mb-4"),
                
                # Confidence indicator
                html.Div([
                    html.H6([
                        html.I(className="fas fa-chart-line me-2"),
                        "Confidence Level"
                    ]),
                    dbc.Progress(
                        value=confidence * 100,
                        color=confidence_color,
                        striped=True,
                        animated=True,
                        style={"height": "25px"},
                        className="mb-2"
                    ),
                    html.Small(f"{confidence:.0%} confidence based on comprehensive context analysis", 
                             className="text-muted")
                ], className="mb-0")
            ])
        ], style={"border": f"3px solid {strategy_info['color']}"})
    ])


def format_reasoning_text(reasoning: str) -> html.Div:
    """
    Format the GPT reasoning text with better structure and readability.
    
    Args:
        reasoning: Raw reasoning text from GPT
        
    Returns:
        html.Div: Formatted reasoning content
    """
    if not reasoning or len(reasoning.strip()) < 20:
        return html.P("Detailed reasoning not available.", className="text-muted")
    
    # Split text into sentences for better readability
    sentences = [s.strip() for s in reasoning.split('.') if s.strip()]
    
    if len(sentences) <= 2:
        # Short reasoning - display as single paragraph
        return html.P(
            reasoning, 
            className="text-justify", 
            style={"lineHeight": "1.6", "fontSize": "0.95rem"}
        )
    
    # Longer reasoning - format with bullet points for key reasons
    formatted_content = []
    
    # First sentence as introduction
    if sentences:
        formatted_content.append(
            html.P(
                sentences[0] + ".", 
                className="text-justify mb-3", 
                style={"lineHeight": "1.6", "fontSize": "0.95rem", "fontWeight": "500"}
            )
        )
    
    # Remaining sentences as reasons
    if len(sentences) > 1:
        reason_items = []
        for sentence in sentences[1:]:
            if sentence.strip():
                reason_items.append(
                    html.Li(sentence.strip() + ".", className="mb-1")
                )
        
        if reason_items:
            formatted_content.append(
                html.Ul(
                    reason_items,
                    style={"lineHeight": "1.6", "fontSize": "0.95rem"}
                )
            )
    
    return html.Div(formatted_content)


def create_strategy_selection_fallback(num_items: int) -> html.Div:
    """
    Create fallback content when GPT analysis is not available.
    
    Args:
        num_items: Number of context items
        
    Returns:
        html.Div: Fallback content
    """
    return html.Div([
        dbc.Alert([
            html.I(className="fas fa-exclamation-triangle me-2"),
            html.Strong("Strategy Recommendation Pending"),
            html.Br(),
            f"Strategy selection based on {num_items} context items is being processed. "
            "The recommendation will be directly informed by the Layer 3 joint analysis."
        ], color="warning"),
        
        # Default strategy explanation while waiting
        html.Div([
            html.H6("Available Strategies", className="mb-3"),
            
            # Strategy options overview
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H6([
                                html.I(className="fas fa-eye me-2", style={"color": "#17a2b8"}),
                                "MONITOR"
                            ], style={"color": "#17a2b8"}),
                            html.P("Continue monitoring model performance with alerting systems", 
                                  className="text-muted small mb-0")
                        ])
                    ], style={"border": "1px solid #17a2b8"})
                ], width=4),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H6([
                                html.I(className="fas fa-tools me-2", style={"color": "#28a745"}),
                                "FINETUNE"
                            ], style={"color": "#28a745"}),
                            html.P("Adapt existing model to new data distribution patterns", 
                                  className="text-muted small mb-0")
                        ])
                    ], style={"border": "1px solid #28a745"})
                ], width=4),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H6([
                                html.I(className="fas fa-redo me-2", style={"color": "#dc3545"}),
                                "RETRAIN"
                            ], style={"color": "#dc3545"}),
                            html.P("Rebuild model from scratch with combined dataset", 
                                  className="text-muted small mb-0")
                        ])
                    ], style={"border": "1px solid #dc3545"})
                ], width=4)
            ], className="g-3 mb-3"),
            
            html.P([
                html.Strong("AI Recommendation: "),
                "Will be determined based on the severity and interaction patterns identified in Layer 3 analysis."
            ], className="text-muted")
        ], style={
            "backgroundColor": "#f8f9fa",
            "padding": "15px",
            "borderRadius": "6px",
            "border": "1px solid #dee2e6"
        })
    ])


def get_strategy_info(strategy: str) -> Dict[str, str]:
    """
    Get comprehensive information about a strategy.
    
    Args:
        strategy: Strategy name (monitor, finetune, retrain)
        
    Returns:
        Dict: Strategy information including icon, color, description
    """
    strategy = strategy.lower()
    
    if strategy == "monitor":
        return {
            "icon": "fas fa-eye",
            "color": "#17a2b8",
            "bg_color": "#e1f7fa",
            "description": "Continue monitoring model performance with enhanced alerting and validation systems. Suitable when issues are manageable and model performance remains acceptable."
        }
    elif strategy == "finetune":
        return {
            "icon": "fas fa-tools", 
            "color": "#28a745",
            "bg_color": "#e8f5e8",
            "description": "Adapt the existing model to new data patterns through fine-tuning techniques. Recommended when specific drift patterns are detected but core model structure remains valid."
        }
    elif strategy == "retrain":
        return {
            "icon": "fas fa-redo",
            "color": "#dc3545", 
            "bg_color": "#fdf2f2",
            "description": "Rebuild the model from scratch using combined primary and secondary datasets. Necessary when significant drift threatens model reliability and simpler approaches are insufficient."
        }
    else:
        return {
            "icon": "fas fa-question-circle",
            "color": "#6c757d",
            "bg_color": "#f8f9fa", 
            "description": "Strategy details not available for the specified approach."
        }


def get_confidence_color(confidence: float) -> str:
    """
    Get Bootstrap color for confidence level.
    
    Args:
        confidence: Confidence value (0.0-1.0)
        
    Returns:
        str: Bootstrap color name
    """
    if confidence >= 0.8:
        return "success"  # High confidence - green
    elif confidence >= 0.6:
        return "info"     # Medium-high confidence - blue
    elif confidence >= 0.4:
        return "warning"  # Medium confidence - yellow
    else:
        return "danger"   # Low confidence - red 