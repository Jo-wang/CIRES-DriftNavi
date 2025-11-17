"""
Unified Analysis Module for Explain Component

This module provides comprehensive analysis functionality that processes all context items
together to provide strategy-based recommendations (retrain/finetune/monitor).
Now uses GPT-4 for intelligent severity assessment instead of rule-based thresholds.

Author: driftNavi Team
Created: 2025
"""

import dash
from dash import html, dcc, callback, Input, Output, State, ALL
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import json
import hashlib
from functools import lru_cache
from typing import List, Dict, Any, Optional, Tuple
from .gpt_severity_analyzer import gpt_severity_analyzer


# =============================================================================
# STRATEGY ANALYSIS FRAMEWORK
# =============================================================================

class StrategyRecommendationEngine:
    """Engine for analyzing context items using GPT-powered intelligent assessment."""
    
    # Strategy descriptions for reference
    STRATEGY_DESCRIPTIONS = {
        "retrain": "Complete model retraining from scratch",
        "finetune": "Targeted model adjustment and optimization", 
        "monitor": "Enhanced monitoring with potential future action"
    }
    
    @staticmethod
    def analyze_context_severity_with_gpt(
        all_context_items: List[Dict[str, Any]], 
        strategy_focus: str = None  # Let GPT decide strategy based on analysis 
    ) -> Dict[str, Any]:
        """
        Analyze all context items using GPT-powered intelligent assessment.
        
        Args:
            all_context_items: List of all context items to analyze
            strategy_focus: Strategic focus for analysis (retrain/finetune/monitor)
            
        Returns:
            Dictionary with GPT-based severity analysis results
        """
        try:
            # Use GPT analyzer for intelligent assessment
            gpt_result = gpt_severity_analyzer.analyze_context_severity(
                all_context_items, 
                strategy_focus
            )
            
            return {
                "gpt_analysis": gpt_result.get("gpt_analysis", ""),
                "individual_scores": gpt_result.get("individual_scores", {}),
                "overall_assessment": gpt_result.get("overall_assessment", {}),
                "dataset_metadata": gpt_result.get("dataset_metadata", {}),
                "analysis_timestamp": gpt_result.get("analysis_timestamp", 0),
                "strategy_focus": strategy_focus,
                "total_issues": len(all_context_items)
            }
            
        except Exception as e:
            print(f"[UNIFIED ANALYSIS] Error in GPT severity analysis: {str(e)}")
            # Fallback to basic analysis
            return {
                "gpt_analysis": f"Analysis error: {str(e)}. Please check your GPT configuration.",
                "individual_scores": {},
                "overall_assessment": {
                    "high_count": 0,
                    "medium_count": 0,
                    "low_count": len(all_context_items),
                    "recommendation": strategy_focus
                },
                "dataset_metadata": {},
                "analysis_timestamp": 0,
                "strategy_focus": strategy_focus,
                "total_issues": len(all_context_items),
                "error": str(e)
            }
    

    
    @classmethod
    def recommend_strategy_with_gpt(
        cls, 
        all_context_items: List[Dict[str, Any]], 
        strategy_focus: str = None  # Let GPT decide strategy based on analysis
    ) -> Dict[str, Any]:
        """
        Analyze all context items using GPT and provide strategy recommendations.
        
        Args:
            all_context_items: List of all context items
            strategy_focus: Strategy focus for analysis
            
        Returns:
            Dictionary with GPT-based strategy recommendation and analysis
        """
        if not all_context_items:
            return {
                "recommended_strategy": "monitor",
                "confidence": 0.0,
                "analysis_summary": "No context items to analyze",
                "severity_breakdown": {"high": 0, "medium": 0, "low": 0},
                "total_issues": 0,
                "gpt_analysis": "No issues detected",
                "strategy_focus": None  # No drift when no items exist
            }
        
        # Get GPT-powered analysis
        gpt_analysis_result = cls.analyze_context_severity_with_gpt(
            all_context_items, 
            strategy_focus
        )
        
        # Extract results from GPT analysis
        overall_assessment = gpt_analysis_result.get("overall_assessment", {})
        severity_breakdown = {
            "high": overall_assessment.get("high_count", 0),
            "medium": overall_assessment.get("medium_count", 0), 
            "low": overall_assessment.get("low_count", 0)
        }
        
        recommended_strategy = overall_assessment.get("recommendation", "monitor")  # Fallback to monitor if GPT doesn't provide recommendation
        
        # Calculate confidence based on GPT analysis quality
        if "error" in gpt_analysis_result:
            confidence = 0.3  # Low confidence if GPT failed
        else:
            total_analyzed = sum(severity_breakdown.values())
            if total_analyzed > 0:
                severity_ratio = severity_breakdown["high"] / total_analyzed
                confidence = min(0.95, 0.5 + severity_ratio * 0.45)
            else:
                confidence = 0.5
        
        return {
            "recommended_strategy": recommended_strategy,
            "confidence": confidence,
            "analysis_summary": gpt_analysis_result.get("gpt_analysis", "Analysis unavailable"),
            "severity_breakdown": severity_breakdown,
            "total_issues": len(all_context_items),
            "gpt_analysis": gpt_analysis_result.get("gpt_analysis", ""),
            "individual_scores": gpt_analysis_result.get("individual_scores", {}),
            "dataset_metadata": gpt_analysis_result.get("dataset_metadata", {}),
            "strategy_focus": strategy_focus,
            "analysis_timestamp": gpt_analysis_result.get("analysis_timestamp", 0)
        }
    



# =============================================================================
# UNIFIED ANALYSIS COMPONENTS
# =============================================================================

def create_unified_analysis_panel(all_context_items: Optional[List[Dict[str, Any]]] = None) -> html.Div:
    """
    Create the unified analysis panel that processes all context items together.
    
    Args:
        all_context_items: List of all context items from explain-context-data
        
    Returns:
        HTML div containing the unified analysis interface
    """
    if not all_context_items:
        return html.Div([
            html.Div([
                html.I(className="fas fa-info-circle me-2", style={"color": "#6c757d", "fontSize": "2rem"}),
                html.H5("No Context Items Available", style={"color": "#6c757d"}),
                html.P([
                    "Please add context items from the detect stage to begin unified analysis. ",
                    "You can add items by clicking the 'Add to Explain' buttons in the detect phase."
                ], className="text-muted")
            ], className="text-center", style={"padding": "60px 20px"})
        ])
    
    # Get GPT-powered strategy analysis - let GPT decide strategy based on analysis
    strategy_analysis = StrategyRecommendationEngine.recommend_strategy_with_gpt(all_context_items)
    
    # Create strategy selection section
    strategy_section = create_strategy_selection_section(strategy_analysis)
    
    # Create context summary section
    summary_section = create_unified_context_summary(all_context_items, strategy_analysis)
    
    # Create analysis perspective section
    perspective_section = create_perspective_selection_section()
    
    # Create results area with GPT analysis
    results_area = html.Div(
        id="unified-analysis-results",
        children=[
            create_gpt_analysis_summary_section(strategy_analysis),
            html.Hr(className="my-3"),
            create_unified_technical_perspective_content(all_context_items, strategy_analysis)
        ],
        style={
            "border": "1px solid #dee2e6",
            "borderRadius": "6px", 
            "padding": "20px",
            "marginTop": "20px",
            "minHeight": "400px",
            "backgroundColor": "#fafafa"
        }
    )
    
    return html.Div([
        # Store for selected strategy
        dcc.Store(id="unified-selected-strategy", data=strategy_analysis["recommended_strategy"]),
        dcc.Store(id="unified-strategy-analysis", data=strategy_analysis),
        
        strategy_section,
        html.Hr(style={"margin": "20px 0"}),
        summary_section,
        html.Hr(style={"margin": "20px 0"}),
        perspective_section,
        results_area
    ])


def create_strategy_selection_section(strategy_analysis: Dict[str, Any]) -> html.Div:
    """Create the strategy selection section without explicit recommendations."""
    current_strategy = strategy_analysis.get("strategy_focus", "retrain")
    
    # Create strategy buttons
    strategy_buttons = dbc.ButtonGroup([
        dbc.Button(
            [html.I(className="fas fa-sync-alt me-2"), "Retrain Model"],
            id="unified-strategy-retrain-btn",
            color="danger" if current_strategy == "retrain" else "outline-danger",
            size="sm",
            active=(current_strategy == "retrain")
        ),
        dbc.Button(
            [html.I(className="fas fa-tools me-2"), "Finetune Model"],
            id="unified-strategy-finetune-btn",
            color="warning" if current_strategy == "finetune" else "outline-warning",
            size="sm",
            active=(current_strategy == "finetune")
        ),
        dbc.Button(
            [html.I(className="fas fa-eye me-2"), "Monitor Only"],
            id="unified-strategy-monitor-btn",
            color="info" if current_strategy == "monitor" else "outline-info",
            size="sm",
            active=(current_strategy == "monitor")
        )
    ], className="w-100")
    
    return html.Div([
        html.H5([
            html.I(className="fas fa-route me-2", style={"color": "#0d6efd"}),
            "Strategy Selection"
        ]),
        html.P("Choose your approach to analyze the detected issues:", 
               className="text-muted small"),
        strategy_buttons
    ])


def create_unified_context_summary(all_context_items: List[Dict[str, Any]], 
                                 strategy_analysis: Dict[str, Any]) -> html.Div:
    """Create a comprehensive summary of all context items."""
    total_items = len(all_context_items)
    severity_breakdown = strategy_analysis["severity_breakdown"]
    
    # Group items by type
    type_counts = {}
    for item in all_context_items:
        item_type = item.get("type", "unknown")
        type_counts[item_type] = type_counts.get(item_type, 0) + 1
    
    # Create summary cards
    summary_cards = []
    
    # Total issues card
    summary_cards.append(
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4(str(total_items), className="text-primary mb-0"),
                    html.P("Total Issues", className="small text-muted mb-0")
                ], className="text-center py-2")
            ])
        ], width=3)
    )
    
    # Severity breakdown cards
    severity_colors = {"high": "danger", "medium": "warning", "low": "success"}
    for severity, count in severity_breakdown.items():
        summary_cards.append(
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(str(count), className=f"text-{severity_colors[severity]} mb-0"),
                        html.P(f"{severity.title()} Severity", className="small text-muted mb-0")
                    ], className="text-center py-2")
                ])
            ], width=3)
        )
    
    # Type breakdown
    type_breakdown_items = []
    for item_type, count in type_counts.items():
        type_display = item_type.replace("_", " ").title()
        type_breakdown_items.append(
            html.Li([
                html.Strong(f"{type_display}: "),
                f"{count} item{'s' if count != 1 else ''}"
            ])
        )
    
    return html.Div([
        html.H6([
            html.I(className="fas fa-chart-pie me-2", style={"color": "#198754"}),
            "Analysis Summary"
        ]),
        
        # Summary metrics
        dbc.Row(summary_cards, className="mb-3"),
        
        # Type breakdown
        html.Div([
            html.H6("Context Types:", className="small mb-2"),
            html.Ul(type_breakdown_items, className="small")
        ], style={"backgroundColor": "#f8f9fa", "padding": "15px", "borderRadius": "6px"})
    ])


def create_perspective_selection_section() -> html.Div:
    """Create the analysis perspective selection section."""
    return html.Div([
        html.H6("Analysis Perspective:", className="mb-3"),
        dbc.ButtonGroup([
            dbc.Button(
                [html.I(className="fas fa-chart-line me-2"), "Technical"],
                id="unified-technical-perspective-btn",
                color="primary",
                size="sm",
                active=True
            ),
            dbc.Button(
                [html.I(className="fas fa-briefcase me-2"), "Business"],
                id="unified-business-perspective-btn",
                color="success",
                size="sm"
            ),
            dbc.Button(
                [html.I(className="fas fa-user-tie me-2"), "Executive"],
                id="unified-executive-perspective-btn",
                color="info",
                size="sm"
            )
        ], className="w-100")
    ])


def create_gpt_analysis_summary_section(strategy_analysis: Dict[str, Any]) -> html.Div:
    """Create the GPT-powered analysis summary section."""
    gpt_analysis = strategy_analysis.get("gpt_analysis", "")
    error_message = strategy_analysis.get("error", None)
    
    if error_message:
        return dbc.Alert([
            html.H6([html.I(className="fas fa-exclamation-triangle me-2"), "Analysis Error"]),
            html.P(f"GPT analysis failed: {error_message}", className="mb-2"),
            html.Small("Please check your OpenAI API configuration and try again.", className="text-muted")
        ], color="warning")
    
    if not gpt_analysis or gpt_analysis.strip() == "":
        return dbc.Alert([
            html.H6([html.I(className="fas fa-info-circle me-2"), "Analysis Pending"]),
            html.P("Intelligent analysis is being generated...", className="mb-0")
        ], color="info")
    
    # Format GPT analysis for better display
    formatted_analysis = format_gpt_analysis_for_display(gpt_analysis)
    
    return html.Div([
        html.H6([
            html.I(className="fas fa-brain me-2", style={"color": "#6f42c1"}),
            "Intelligent Analysis Summary"
        ], className="mb-3"),
        html.Div(
            formatted_analysis,
            style={
                "backgroundColor": "#f8f9fa",
                "padding": "20px",
                "borderRadius": "6px",
                "border": "1px solid #e9ecef",
                "lineHeight": "1.6"
            }
        )
    ])


def format_gpt_analysis_for_display(gpt_text: str) -> html.Div:
    """Format GPT analysis text for better display in the UI."""
    if not gpt_text:
        return html.P("No analysis available", className="text-muted")
    
    # Split into sections and format
    sections = []
    current_section = []
    
    lines = gpt_text.split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check for section headers
        if line.startswith('##') or line.startswith('**') and line.endswith('**'):
            if current_section:
                sections.append('\n'.join(current_section))
                current_section = []
            current_section.append(line)
        else:
            current_section.append(line)
    
    if current_section:
        sections.append('\n'.join(current_section))
    
    # Create formatted output
    formatted_sections = []
    for section in sections:
        if section.strip():
            # Convert markdown-style formatting to HTML
            section_html = convert_analysis_text_to_html(section)
            formatted_sections.append(section_html)
    
    return html.Div(formatted_sections)


def convert_analysis_text_to_html(text: str) -> html.Div:
    """Convert analysis text with markdown-style formatting to HTML."""
    lines = text.split('\n')
    elements = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Headers
        if line.startswith('## '):
            elements.append(html.H5(line[3:], className="text-primary mt-3 mb-2"))
        elif line.startswith('**') and line.endswith('**'):
            elements.append(html.H6(line[2:-2], className="text-secondary mt-2 mb-1"))
        elif line.startswith('- '):
            elements.append(html.Li(line[2:], className="mb-1"))
        elif line.startswith('1. ') or line.startswith('2. ') or line.startswith('3. '):
            elements.append(html.Li(line[3:], className="mb-1"))
        else:
            elements.append(html.P(line, className="mb-2"))
    
    return html.Div(elements)


# =============================================================================
# PERSPECTIVE CONTENT CREATORS
# =============================================================================

def create_unified_technical_perspective_content(
    all_context_items: List[Dict[str, Any]], 
    strategy_analysis: Dict[str, Any],
    selected_strategy: str = None
) -> html.Div:
    """Create comprehensive technical analysis content for all context items."""
    if not all_context_items:
        return html.P("No context items available for technical analysis.", className="text-muted text-center")
    
    selected_strategy = selected_strategy or strategy_analysis["recommended_strategy"]
    
    content = [
        html.H5([
            html.I(className="fas fa-chart-line me-2", style={"color": "#0d6efd"}),
            f"Technical Analysis - {selected_strategy.upper()} Strategy"
        ]),
        html.Hr()
    ]
    
    # Strategy-specific technical guidance
    strategy_guidance = get_technical_strategy_guidance(selected_strategy)
    content.append(
        dbc.Alert([
            html.H6([html.I(className="fas fa-info-circle me-2"), "Technical Guidance"]),
            html.P(strategy_guidance, className="mb-0")
        ], color="primary", className="mb-4")
    )
    
    # Detailed analysis by context type
    context_by_type = group_contexts_by_type(all_context_items)
    
    for context_type, items in context_by_type.items():
        if items:
            type_section = create_technical_context_type_section(context_type, items, selected_strategy)
            content.append(type_section)
    
    # Strategy implementation recommendations
    implementation_section = create_technical_implementation_section(
        all_context_items, strategy_analysis, selected_strategy
    )
    content.append(implementation_section)
    
    return html.Div(content)


def get_technical_strategy_guidance(strategy: str) -> str:
    """Get technical guidance for the selected strategy."""
    guidance = {
        "retrain": (
            "Complete model retraining is recommended due to significant distribution shifts. "
            "This involves resampling data, adjusting feature engineering, and training from scratch. "
            "Focus on addressing the root causes of drift and ensuring robust data pipeline."
        ),
        "finetune": (
            "Targeted model adjustment is recommended to address specific distribution changes. "
            "Consider transfer learning, hyperparameter optimization, and selective layer retraining. "
            "Monitor performance metrics closely during the finetuning process."
        ),
        "monitor": (
            "Enhanced monitoring is sufficient for the current level of drift. "
            "Implement automated drift detection, performance tracking, and alert systems. "
            "Prepare escalation procedures if drift severity increases."
        )
    }
    return guidance.get(strategy, "Strategy-specific technical guidance not available.")


def group_contexts_by_type(all_context_items: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group context items by their type for organized analysis."""
    grouped = {}
    for item in all_context_items:
        item_type = item.get("type", "unknown")
        if item_type not in grouped:
            grouped[item_type] = []
        grouped[item_type].append(item)
    return grouped


def create_technical_context_type_section(context_type: str, items: List[Dict[str, Any]], 
                                        strategy: str) -> html.Div:
    """Create a technical analysis section for a specific context type."""
    type_display = context_type.replace("_", " ").title()
    type_colors = {
        "drift_analysis": "#dc3545",
        "distribution_comparison": "#28a745", 
        "conditional_distribution": "#516395",
        "target_distribution": "#614385",
        "metric": "#fd7e14"
    }
    
    color = type_colors.get(context_type, "#6c757d")
    
    # Create items list
    items_content = []
    for i, item in enumerate(items):
        item_content = create_technical_item_summary(item, strategy)
        items_content.append(
            html.Div([
                html.H6(f"Item {i+1}: {extract_item_title(item)}", className="mb-2"),
                item_content
            ], className="mb-3 p-3", style={"backgroundColor": "#f8f9fa", "borderRadius": "4px"})
        )
    
    return html.Div([
        html.H6([
            html.I(className="fas fa-layer-group me-2", style={"color": color}),
            f"{type_display} ({len(items)} item{'s' if len(items) != 1 else ''})"
        ], style={"color": color}),
        html.Div(items_content),
        html.Hr(style={"margin": "20px 0"})
    ])


def create_technical_item_summary(item: Dict[str, Any], strategy: str) -> html.Div:
    """Create a technical summary for a single context item without severity badges."""
    metric_details = item.get("metric_details", "")
    interpretation = item.get("interpretation", "")
    summary_text = item.get("summary_text", "")
    
    content = []
    
    # Context type indicator
    context_type = item.get("type", "unknown")
    type_colors = {
        "drift_analysis": "danger",
        "distribution_comparison": "warning", 
        "conditional_distribution": "info",
        "metric": "secondary",
        "target_distribution": "primary"
    }
    type_color = type_colors.get(context_type, "secondary")
    
    content.append(
        html.Span([
            html.I(className="fas fa-tag me-1"),
            context_type.replace("_", " ").title()
        ], className=f"badge bg-{type_color} mb-2")
    )
    
    # Details
    if metric_details:
        content.extend([
            html.P([html.Strong("Metrics: "), metric_details], className="small mb-1"),
        ])
    
    if interpretation:
        content.extend([
            html.P([html.Strong("Interpretation: "), interpretation], className="small mb-1"),
        ])
    
    if summary_text:
        content.extend([
            html.P([html.Strong("Summary: "), summary_text], className="small mb-1"),
        ])
    
    # Strategy-specific recommendations
    strategy_rec = get_item_strategy_recommendation(item, strategy)
    if strategy_rec:
        content.append(
            html.P([html.Strong("Action: "), strategy_rec], 
                  className="small text-primary mb-0")
        )
    
    return html.Div(content)


def get_item_strategy_recommendation(item: Dict[str, Any], strategy: str) -> str:
    """Get strategy-specific recommendation for a context item."""
    context_type = item.get("type", "")
    
    recommendations = {
        "retrain": {
            "drift_analysis": "Include this attribute in retraining data analysis and feature engineering review.",
            "distribution_comparison": "Address distribution mismatch through data resampling and collection strategy.",
            "conditional_distribution": "Ensure balanced representation in new training data.",
            "target_distribution": "Redefine target distribution requirements and sampling strategy.",
            "metric": "Establish new baseline metrics post-retraining."
        },
        "finetune": {
            "drift_analysis": "Apply targeted drift correction techniques and adaptive learning.",
            "distribution_comparison": "Use domain adaptation or transfer learning approaches.",
            "conditional_distribution": "Implement drift-aware finetuning with fairness constraints.",
            "target_distribution": "Adjust model outputs through calibration and threshold tuning.",
            "metric": "Monitor metric during finetuning and adjust accordingly."
        },
        "monitor": {
            "drift_analysis": "Set up automated drift detection alerts for this attribute.",
            "distribution_comparison": "Implement distribution monitoring with statistical tests.",
            "conditional_distribution": "Monitor drift metrics and fairness indicators regularly.",
            "target_distribution": "Track target distribution stability over time.",
            "metric": "Establish metric monitoring dashboard and alert thresholds."
        }
    }
    
    return recommendations.get(strategy, {}).get(context_type, "Monitor and reassess regularly.")


def create_technical_implementation_section(
    all_context_items: List[Dict[str, Any]], 
    strategy_analysis: Dict[str, Any],
    selected_strategy: str
) -> html.Div:
    """Create technical implementation recommendations section."""
    
    implementation_guides = {
        "retrain": {
            "title": "Model Retraining Implementation",
            "steps": [
                "1. Data Collection: Gather new representative training data",
                "2. Feature Engineering: Review and update feature pipelines", 
                "3. Model Architecture: Evaluate if current architecture is sufficient",
                "4. Training Process: Implement robust training with validation",
                "5. Testing: Comprehensive testing including drift and fairness metrics",
                "6. Deployment: Gradual rollout with monitoring"
            ],
            "considerations": [
                "Resource Requirements: High computational resources needed",
                "Timeline: 2-6 weeks depending on data availability",
                "Risk: Temporary service disruption during deployment",
                "Success Metrics: Improved accuracy and reduced drift"
            ]
        },
        "finetune": {
            "title": "Model Finetuning Implementation", 
            "steps": [
                "1. Data Preparation: Focus on problematic data segments",
                "2. Transfer Learning: Leverage existing model weights",
                "3. Hyperparameter Tuning: Optimize learning rates and regularization",
                "4. Targeted Training: Focus on specific layers or components",
                "5. Validation: Extensive testing on holdout data",
                "6. Deployment: A/B testing with gradual rollout"
            ],
            "considerations": [
                "Resource Requirements: Moderate computational resources",
                "Timeline: 1-3 weeks for implementation and testing",
                "Risk: Lower risk than full retraining",
                "Success Metrics: Targeted improvement in problem areas"
            ]
        },
        "monitor": {
            "title": "Enhanced Monitoring Implementation",
            "steps": [
                "1. Metric Definition: Define comprehensive monitoring metrics",
                "2. Alert Setup: Configure drift and performance alerts",
                "3. Dashboard Creation: Build monitoring dashboards",
                "4. Automated Testing: Implement automated model testing",
                "5. Response Procedures: Define escalation procedures",
                "6. Regular Review: Schedule periodic analysis reviews"
            ],
            "considerations": [
                "Resource Requirements: Low to moderate",
                "Timeline: 1-2 weeks for full implementation",
                "Risk: Minimal implementation risk",
                "Success Metrics: Early detection of issues and stable performance"
            ]
        }
    }
    
    guide = implementation_guides.get(selected_strategy, {})
    
    if not guide:
        return html.Div()
    
    return html.Div([
        html.H6([
            html.I(className="fas fa-cogs me-2", style={"color": "#28a745"}),
            guide["title"]
        ], style={"color": "#28a745"}),
        
        html.Div([
            html.H6("Implementation Steps:", className="small mb-2"),
            html.Ol([html.Li(step, className="small") for step in guide["steps"]])
        ], className="mb-3"),
        
        html.Div([
            html.H6("Key Considerations:", className="small mb-2"),
            html.Ul([html.Li(consideration, className="small") for consideration in guide["considerations"]])
        ], style={"backgroundColor": "#e8f4fd", "padding": "15px", "borderRadius": "6px"})
    ])


def extract_item_title(item: Dict[str, Any]) -> str:
    """Extract a descriptive title for a context item."""
    context_type = item.get("type", "unknown")
    
    if context_type == "drift_analysis":
        return f"Drift Analysis"
    elif context_type == "distribution_comparison":
        return f"Distribution Comparison"
    elif context_type == "conditional_distribution":
        target_attr = item.get("target_attribute", "unknown")
        target_value = item.get("target_value", "unknown")
        compare_attr = item.get("compare_attribute", "unknown")
        return f"Conditional: {target_attr}={target_value} vs {compare_attr}"
    elif context_type == "target_distribution":
        target_attr = item.get("target_attribute", "unknown")
        return f"Target Distribution: {target_attr}"
    elif context_type == "metric":
        return f"Metric Analysis"
    else:
        return f"{context_type.replace('_', ' ').title()}" 