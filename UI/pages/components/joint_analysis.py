"""
Joint Analysis Component for driftNavi

This module provides UI components for displaying Layer 3: Joint Analysis
focusing on systematic assessment of primary-to-secondary dataset inference risks.

Author: driftNavi Team
Created: 2025
"""

from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
from typing import Dict, Any, List, Optional


# =============================================================================
# JOINT ANALYSIS MAIN COMPONENT (SIMPLIFIED)
# =============================================================================

def create_joint_analysis_component(
    context_items: List[Dict[str, Any]], 
    analysis_data: Dict[str, Any]
) -> html.Div:
    """
    Create the joint analysis component showing Layer 3 analysis.
    Simplified to focus only on overall assessment of primary-to-secondary inference risks.
    
    Args:
        context_items: List of context items from explain-context-data
        analysis_data: Analysis data from GPT containing layer3_joint_analysis
        
    Returns:
        html.Div: Simplified joint analysis component
    """
    if not context_items:
        return html.Div([
            html.Div([
                html.I(className="fas fa-info-circle me-2", style={"color": "#6c757d"}),
                html.H6("No Data for Joint Analysis", className="text-muted"),
                html.P("Add context items to see combined impact analysis.", className="text-muted small")
            ], className="text-center py-4")
        ])
    
    # Extract layer3 data from comprehensive analysis
    layer3_data = analysis_data.get('comprehensive_data', {}).get('layer3_joint_analysis', {})
    
    if not layer3_data:
        return create_joint_analysis_fallback(len(context_items))
    
    # Get overall assessment
    overall_assessment = layer3_data.get('overall_assessment', 'Assessment not available')
    
    return html.Div([
        # Header
        html.Div([
            # Removed duplicate icon - main title already has the icon
            # html.H6("Joint Analysis", style={"color": "#e74c3c"}),
            html.P(f"Systematic assessment of {len(context_items)} issues for primary-to-secondary inference", 
                  className="text-muted small mb-3")
        ]),
        
        # Main overall assessment content
        create_overall_assessment_content(overall_assessment, len(context_items)),
        
    ], className="mb-4")


def create_overall_assessment_content(assessment_text: str, num_items: int) -> html.Div:
    """
    Create the main overall assessment content focusing on inference risks.
    
    Args:
        assessment_text: GPT-generated overall assessment text
        num_items: Number of context items being analyzed
        
    Returns:
        html.Div: Formatted overall assessment content
    """
    return html.Div([
        # Assessment Card
        dbc.Card([
            dbc.CardHeader([
                html.H6([
                    html.I(className="fas fa-search-plus me-2", style={"color": "#e74c3c"}),
                    "Overall Risk Assessment"
                ], className="mb-0", style={"color": "#e74c3c"})
            ], style={"backgroundColor": "#fdf2f2"}),
            
            dbc.CardBody([
                # Main assessment text
                html.Div([
                    html.H6([
                        html.I(className="fas fa-brain me-2", style={"color": "#6c757d"}),
                        "AI Analysis"
                    ], className="mb-3"),
                    
                    # Format the assessment text with better structure
                    format_assessment_text(assessment_text)
                ], className="mb-4"),
                
                # Key focus areas summary
                html.Div([
                    html.H6([
                        html.I(className="fas fa-bullseye me-2", style={"color": "#fd7e14"}),
                        "Primary Focus Areas"
                    ], className="mb-3"),
                    
                    dbc.Row([
                        dbc.Col([
                            dbc.Alert([
                                html.I(className="fas fa-exclamation-triangle me-2"),
                                html.Strong("Model Deployment Risk"),
                                html.Br(),
                                "Direct inference on secondary dataset without adaptation"
                            ], color="warning", className="mb-2")
                        ], width=6),
                        
                        dbc.Col([
                            dbc.Alert([
                                html.I(className="fas fa-target me-2"),
                                html.Strong("Target Attribute Impact"),
                                html.Br(),
                                "Prediction reliability and accuracy concerns"
                            ], color="info", className="mb-2")
                        ], width=6)
                    ], className="g-2")
                ], className="mb-4"),
                
                # Connection to Layer 4
                html.Div([
                    dbc.Alert([
                        html.I(className="fas fa-arrow-right me-2"),
                        html.Strong("Strategic Implication: "),
                        "This analysis directly informs the recommended strategy in Layer 4 below."
                    ], color="secondary", className="mb-0")
                ])
            ])
        ], style={"border": "2px solid #e74c3c"})
    ])


def format_assessment_text(assessment_text: str) -> html.Div:
    """
    Format the GPT assessment text with better structure and readability.
    
    Args:
        assessment_text: Raw assessment text from GPT
        
    Returns:
        html.Div: Formatted text with improved structure
    """
    if not assessment_text or len(assessment_text.strip()) < 50:
        return html.P("Detailed assessment not available.", className="text-muted")
    
    # Split text into paragraphs for better readability
    paragraphs = [p.strip() for p in assessment_text.split('\n') if p.strip()]
    
    if len(paragraphs) <= 1:
        # Single paragraph - display as is
        return html.P(
            assessment_text, 
            className="text-justify", 
            style={"lineHeight": "1.6", "fontSize": "0.95rem"}
        )
    
    # Multiple paragraphs - format with spacing
    formatted_paragraphs = []
    for i, paragraph in enumerate(paragraphs):
        formatted_paragraphs.append(
            html.P(
                paragraph, 
                className="text-justify mb-3", 
                style={"lineHeight": "1.6", "fontSize": "0.95rem"}
            )
        )
    
    return html.Div(formatted_paragraphs)


def create_joint_analysis_fallback(num_items: int) -> html.Div:
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
            html.Strong("Analysis Pending"),
            html.Br(),
            f"Joint analysis of {num_items} context items is being processed. "
            "This will assess the combined risk of deploying a model trained on the primary dataset "
            "directly to the secondary dataset without adaptation."
        ], color="warning"),
        
        # Manual guidance while waiting
        html.Div([
            html.H6("Manual Assessment Guidance", className="mb-3"),
            html.Ul([
                html.Li("Consider the cumulative impact of all detected issues"),
                html.Li("Assess whether multiple drift patterns could compound inference errors"),
                html.Li("Evaluate target attribute prediction reliability across datasets"),
                html.Li("Determine if model recalibration or retraining is necessary")
            ], className="mb-3"),
            
            html.P([
                html.Strong("Next Step: "),
                "The assessment results will directly inform the strategy recommendation in Layer 4."
            ], className="text-muted")
        ], style={
            "backgroundColor": "#f8f9fa",
            "padding": "15px",
            "borderRadius": "6px",
            "border": "1px solid #dee2e6"
        })
    ]) 