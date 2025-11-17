"""
Unified GPT Analysis Module for Explain Component

This module handles GPT-based analysis for business and executive perspectives
of unified context analysis.

Author: driftNavi Team
Created: 2025
"""

import json
import hashlib
from functools import lru_cache
from typing import List, Dict, Any, Optional
from dash import html
import dash_bootstrap_components as dbc
from .unified_analysis import StrategyRecommendationEngine


# =============================================================================
# GPT ANALYSIS FUNCTIONS
# =============================================================================

def analyze_all_contexts_with_gpt(
    all_context_items: List[Dict[str, Any]], 
    selected_strategy: str, 
    perspective_type: str = "business"
) -> Dict[str, Any]:
    """
    Analyze all context items together with GPT for comprehensive insights.
    
    Args:
        all_context_items: List of all context items
        selected_strategy: "retrain", "finetune", or "monitor"
        perspective_type: "business" or "executive"
        
    Returns:
        Dictionary with comprehensive GPT analysis results
    """
    try:
        # Use existing explain_api module
        from agent.explain_api import generate_response_from_prompt
        from flask_login import current_user
        
        # Create context hash for caching
        context_hash = create_context_hash(all_context_items, selected_strategy, perspective_type)
        
        # Check cache first
        cached_result = get_cached_analysis(context_hash)
        if cached_result:
            print(f"[UNIFIED GPT] Using cached analysis for {perspective_type} perspective")
            return cached_result
        
        # 1. Aggregate all context information
        aggregated_context = aggregate_context_items_for_gpt(all_context_items)
        
        # 2. Get user domain information
        user_domain_info = get_user_domain_info_for_gpt(current_user)
        
        # 3. Get dataset attribute information
        dataset_attributes_info = get_dataset_attributes_info_for_gpt()
        
        # 4. Get strategy analysis - let GPT decide strategy based on analysis
        strategy_analysis = StrategyRecommendationEngine.recommend_strategy_with_gpt(all_context_items)
        
        # 5. Prepare comprehensive context data
        context_data = {
            "strategy": selected_strategy,
            "aggregated_contexts": aggregated_context,
            "user_domain": user_domain_info,
            "dataset_attributes": dataset_attributes_info,
            "strategy_analysis": strategy_analysis,
            "total_issues": len(all_context_items)
        }
        
        # 6. Generate strategy-specific prompt using unified prompt manager
        from .prompt_manager import (
            prompt_manager,
            create_user_context,
            create_dataset_context
        )
        
        # Prepare context for prompt manager
        prompt_context = {
            'user_context': create_user_context(user_domain_info),
            'dataset_context': create_dataset_context(dataset_attributes_info),
            'context_items': all_context_items,
            'strategy_focus': selected_strategy
        }
        
        # Generate prompt using unified template system
        template_name = f"{perspective_type}_analysis"
        prompt = prompt_manager.generate_prompt(template_name, prompt_context)
        
        print(f"[UNIFIED GPT] Generating {perspective_type} analysis for {selected_strategy} strategy")
        
        # 7. Call GPT API - use user's selected model instead of hardcoded "gpt-4o"
        gpt_response = generate_response_from_prompt(prompt)
        
        # 8. Format response (now expecting text format, not JSON)
        formatted_result = format_unified_text_response(gpt_response, perspective_type, selected_strategy)
        
        # Cache the result
        cache_analysis_result(context_hash, formatted_result)
        
        return formatted_result
            
    except Exception as e:
        print(f"[UNIFIED GPT ANALYSIS] Error: {str(e)}")
        return create_unified_error_response(e, perspective_type, selected_strategy)


def aggregate_context_items_for_gpt(all_context_items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate and categorize all context items for GPT analysis."""
    aggregated = {
        "drift_analysis": [],
        "distribution_comparisons": [],
        "conditional_distributions": [],
        "target_distributions": [],
        "metrics": [],
        "summary_statistics": {}
    }
    
    for item in all_context_items:
        context_type = item.get("type", "unknown")
        
        if context_type == "drift_analysis":
            aggregated["drift_analysis"].append({
                "attribute": extract_target_attribute_from_item(item),
                "severity": extract_severity_from_item(item),
                "details": item.get("metric_details", ""),
                "interpretation": item.get("interpretation", ""),
                "summary": item.get("summary_text", "")
            })
        elif context_type == "distribution_comparison":
            aggregated["distribution_comparisons"].append({
                "attribute": extract_target_attribute_from_item(item),
                "summary": item.get("summary_text", ""),
                "cell_info": item.get("cell_info", ""),
                "chart_data": item.get("chart_data", {})
            })
        elif context_type == "conditional_distribution":
            aggregated["conditional_distributions"].append({
                "target_attribute": item.get("target_attribute", ""),
                "target_value": item.get("target_value", ""),
                "compare_attribute": item.get("compare_attribute", ""),
                "summary": item.get("summary_text", "")
            })
        elif context_type == "target_distribution":
            aggregated["target_distributions"].append({
                "target_attribute": item.get("target_attribute", ""),
                "summary": item.get("summary_text", "")
            })
        elif context_type == "metric":
            aggregated["metrics"].append({
                "details": item.get("metric_details", ""),
                "interpretation": item.get("interpretation", ""),
                "summary": item.get("summary_text", "")
            })
    
    # Calculate summary statistics
    aggregated["summary_statistics"] = {
        "total_drifted_attributes": len(aggregated["drift_analysis"]),
        "high_severity_count": len([d for d in aggregated["drift_analysis"] if d["severity"] == "high"]),
        "affected_distributions": len(aggregated["distribution_comparisons"]),
        "conditional_driftes": len(aggregated["conditional_distributions"]),
        "target_shifts": len(aggregated["target_distributions"]),
        "metric_violations": len(aggregated["metrics"])
    }
    
    return aggregated


# Legacy prompt functions removed - now using unified prompt manager
# All prompts are now handled by UI.pages.components.prompt_manager


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def extract_target_attribute_from_item(item: Dict[str, Any]) -> str:
    """Extract target attribute from context item."""
    # Try different possible keys
    for key in ["target_attribute", "attribute", "column", "feature"]:
        if key in item and item[key]:
            return item[key]
    
    # Try to extract from summary text or cell info
    summary_text = item.get("summary_text", "")
    cell_info = item.get("cell_info", "")
    
    # Simple extraction from text
    if "attribute" in summary_text.lower():
        words = summary_text.split()
        for i, word in enumerate(words):
            if word.lower() == "attribute" and i + 1 < len(words):
                return words[i + 1]
    
    return "Unknown Attribute"


def extract_severity_from_item(item: Dict[str, Any]) -> str:
    """Extract severity level from context item."""
    text_to_check = f"{item.get('metric_details', '')} {item.get('interpretation', '')} {item.get('summary_text', '')}".lower()
    
    if any(keyword in text_to_check for keyword in ["critical", "severe", "high", "significant"]):
        return "high"
    elif any(keyword in text_to_check for keyword in ["moderate", "medium", "noticeable"]):
        return "medium"
    else:
        return "low"


def format_drift_analysis_for_prompt(drift_items: List[Dict[str, Any]]) -> str:
    """Format drift analysis items for GPT prompt."""
    if not drift_items:
        return "No drift analysis items detected."
    
    formatted_items = []
    for i, item in enumerate(drift_items, 1):
        formatted_items.append(
            f"   {i}. Attribute: {item['attribute']}\n"
            f"      Severity: {item['severity']}\n"
            f"      Details: {item['details'][:100]}...\n"
            f"      Impact: {item['interpretation'][:100]}..."
        )
    
    return "\n".join(formatted_items)


def format_distribution_problems_for_prompt(distribution_items: List[Dict[str, Any]]) -> str:
    """Format distribution comparison items for GPT prompt."""
    if not distribution_items:
        return "No distribution comparison issues detected."
    
    formatted_items = []
    for i, item in enumerate(distribution_items, 1):
        formatted_items.append(
            f"   {i}. Attribute: {item['attribute']}\n"
            f"      Summary: {item['summary'][:150]}...\n"
            f"      Details: {item['cell_info'][:100]}..."
        )
    
    return "\n".join(formatted_items)


def format_conditional_driftes_for_prompt(conditional_items: List[Dict[str, Any]]) -> str:
    """Format conditional distribution items for GPT prompt."""
    if not conditional_items:
        return "No conditional drift issues detected."
    
    formatted_items = []
    for i, item in enumerate(conditional_items, 1):
        formatted_items.append(
            f"   {i}. Condition: {item['target_attribute']} = {item['target_value']}\n"
            f"      Analyzed Attribute: {item['compare_attribute']}\n"
            f"      Summary: {item['summary'][:150]}..."
        )
    
    return "\n".join(formatted_items)


def get_user_domain_info_for_gpt(current_user) -> Dict[str, Any]:
    """Get user domain information for GPT analysis using centralized dynamic system."""
    try:
        # Use the centralized user context management system
        from UI.functions.global_vars import global_vars
        
        # Get actual user context without hardcoded defaults
        user_context = global_vars.get_user_context(current_user)
        
        profile_complete = user_context.get('profile_completeness', 0)
        
        if user_context.get('has_profile', False):
            print(f"[UNIFIED GPT ANALYSIS] Retrieved user context - Profile {profile_complete:.0f}% complete")
        else:
            print(f"[UNIFIED GPT ANALYSIS] No user profile available - using adaptive analysis")
        
        return user_context
        
    except Exception as e:
        print(f"[UNIFIED GPT ANALYSIS] Error getting user context: {str(e)}")
        
        # Return minimal structure indicating error
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
            "error": str(e)
        }


def get_dataset_attributes_info_for_gpt() -> Dict[str, Any]:
    """Get dataset attributes information for GPT analysis."""
    try:
        # This should be implemented based on your global_vars or dataset info
        from UI.functions.global_vars import global_vars
        
        primary_df = global_vars.get_primary_df()
        secondary_df = global_vars.get_secondary_df()
        
        if primary_df is not None:
            return {
                "primary_dataset": {
                    "shape": primary_df.shape,
                    "columns": list(primary_df.columns),
                    "numeric_columns": list(primary_df.select_dtypes(include=[np.number]).columns),
                    "categorical_columns": list(primary_df.select_dtypes(include=['object']).columns)
                },
                "secondary_dataset": {
                    "shape": secondary_df.shape if secondary_df is not None else None,
                    "columns": list(secondary_df.columns) if secondary_df is not None else None
                } if secondary_df is not None else None
            }
    except:
        pass
    
    # Default values
    return {
        "primary_dataset": {
            "shape": (1000, 10),
            "columns": ["feature_1", "feature_2", "target"],
            "numeric_columns": ["feature_1", "feature_2"],
            "categorical_columns": ["target"]
        }
    }


# =============================================================================
# CACHING AND RESPONSE FORMATTING
# =============================================================================

@lru_cache(maxsize=32)
def get_cached_analysis(context_hash: str) -> Optional[Dict[str, Any]]:
    """Get cached analysis result if available."""
    # In a production system, this would use Redis or another cache
    return None


def cache_analysis_result(context_hash: str, result: Dict[str, Any]) -> None:
    """Cache analysis result for future use."""
    # In a production system, this would save to Redis or another cache
    pass


def create_context_hash(all_context_items: List[Dict[str, Any]], strategy: str, perspective: str) -> str:
    """Create hash of context items, strategy, and perspective for caching."""
    context_str = json.dumps({
        "items": sorted([str(item) for item in all_context_items]),
        "strategy": strategy,
        "perspective": perspective
    }, sort_keys=True)
    return hashlib.md5(context_str.encode()).hexdigest()





def format_unified_text_response(text_response: str, perspective_type: str, strategy: str) -> Dict[str, Any]:
    """Format structured text response from GPT into properly formatted HTML components."""
    
    # Convert markdown-style formatting to HTML
    formatted_content = convert_gpt_text_to_html(text_response, perspective_type, strategy)
    
    content = html.Div([
        html.H5([
            html.I(className=f"fas fa-{'briefcase' if perspective_type == 'business' else 'user-tie'} me-2", 
                  style={"color": "#198754" if perspective_type == "business" else "#0dcaf0"}),
            f"{perspective_type.title()} Analysis - {strategy.upper()} Strategy"
        ]),
        html.Hr(),
        formatted_content
    ])
    
    return {"error": False, "content": content}


def convert_gpt_text_to_html(text_response: str, perspective_type: str, strategy: str) -> html.Div:
    """Convert GPT's structured text response to properly formatted HTML."""
    
    lines = text_response.strip().split('\n')
    content_elements = []
    current_section = []
    current_section_title = None
    
    for line in lines:
        line = line.strip()
        
        # Skip empty lines
        if not line:
            continue
            
        # Handle main section headers (## or **)
        if line.startswith('##') or (line.startswith('**') and line.endswith('**') and len(line) > 4):
            # Save previous section if exists
            if current_section and current_section_title:
                content_elements.append(create_section_html(current_section_title, current_section))
                current_section = []
            
            # Extract section title
            if line.startswith('##'):
                current_section_title = line.replace('##', '').strip()
            else:
                current_section_title = line.replace('**', '').strip()
            
        # Handle subsection headers (**text:**)
        elif line.startswith('**') and ':' in line and line.endswith(':'):
            subsection_title = line.replace('**', '').replace(':', '').strip()
            current_section.append(('subsection', subsection_title))
            
        # Handle bullet points
        elif line.startswith('- '):
            bullet_text = line[2:].strip()
            current_section.append(('bullet', bullet_text))
            
        # Handle regular paragraphs
        else:
            current_section.append(('paragraph', line))
    
    # Add the last section
    if current_section and current_section_title:
        content_elements.append(create_section_html(current_section_title, current_section))
    
    # If no structured content was found, fall back to simple formatting
    if not content_elements:
        content_elements.append(
            html.Div([
                html.P(text_response, style={
                    "whiteSpace": "pre-wrap",
                    "lineHeight": "1.6",
                    "fontSize": "14px"
                })
            ])
        )
    
    return html.Div(content_elements)


def create_section_html(title: str, content_items: list) -> html.Div:
    """Create HTML for a section with title and content items."""
    
    # Determine section color based on title
    section_colors = {
        "EXECUTIVE SUMMARY": "#0dcaf0",
        "BUSINESS IMPACT ASSESSMENT": "#198754", 
        "STRATEGY VALIDATION": "#fd7e14",
        "IMPLEMENTATION ROADMAP": "#6f42c1",
        "SUCCESS METRICS": "#20c997",
        "RISK ASSESSMENT": "#dc3545",
        "STRATEGIC IMPACT ANALYSIS": "#0dcaf0",
        "LEADERSHIP DECISIONS REQUIRED": "#6610f2",
        "ORGANIZATIONAL IMPLICATIONS": "#6f42c1",
        "EXECUTIVE ACTION ITEMS": "#fd7e14",
        "BOTTOM LINE RECOMMENDATION": "#198754",
        "RECOMMENDATIONS SUMMARY": "#198754"
    }
    
    color = section_colors.get(title.upper(), "#6c757d")
    
    # Create section header
    section_elements = [
        html.H6(title, style={"color": color, "marginTop": "20px", "marginBottom": "15px"})
    ]
    
    # Process content items
    current_list_items = []
    
    for item_type, item_text in content_items:
        if item_type == 'subsection':
            # Close any open bullet list
            if current_list_items:
                section_elements.append(html.Ul(current_list_items, className="mb-3"))
                current_list_items = []
            
            # Add subsection header
            section_elements.append(
                html.P([html.Strong(f"{item_text}:")], className="mb-2 mt-3", style={"color": color})
            )
            
        elif item_type == 'bullet':
            current_list_items.append(html.Li(item_text, style={"marginBottom": "5px"}))
            
        elif item_type == 'paragraph':
            # Close any open bullet list
            if current_list_items:
                section_elements.append(html.Ul(current_list_items, className="mb-3"))
                current_list_items = []
            
            # Add paragraph
            section_elements.append(
                html.P(item_text, className="mb-2", style={"lineHeight": "1.5"})
            )
    
    # Close any remaining bullet list
    if current_list_items:
        section_elements.append(html.Ul(current_list_items, className="mb-3"))
    
    return html.Div(
        section_elements,
        style={
            "backgroundColor": "#f8f9fa",
            "padding": "15px",
            "borderRadius": "6px",
            "marginBottom": "20px",
            "borderLeft": f"4px solid {color}"
        }
    )


def create_unified_error_response(error: Exception, perspective_type: str, strategy: str) -> Dict[str, Any]:
    """Create user-friendly error response for unified analysis."""
    content = html.Div([
        dbc.Alert([
            html.I(className="fas fa-exclamation-triangle me-2"),
            html.Strong("Analysis Temporarily Unavailable"),
            html.Br(),
            f"We're experiencing technical difficulties with {perspective_type} analysis. Please try again in a moment."
        ], color="warning", className="mb-3"),
        
        html.Div([
            html.H6("Alternative: Basic Analysis Summary"),
            html.P(f"While we resolve the issue, consider these general guidelines for {strategy} strategy:"),
            html.Ul([
                html.Li("Review all detected issues systematically"),
                html.Li("Consult with technical team on implementation approach"),
                html.Li("Establish monitoring and validation procedures"),
                html.Li("Plan for iterative improvements")
            ])
        ])
    ])
    
    return {"error": True, "content": content}


def create_formatting_error_response(result: Any, perspective_type: str) -> html.Div:
    """Create error response when GPT response formatting fails."""
    return html.Div([
        dbc.Alert([
            html.Strong("Response Processing Error"),
            html.Br(),
            f"Received response from GPT but couldn't format it properly for {perspective_type} perspective."
        ], color="warning", className="mb-3"),
        html.Details([
            html.Summary("Raw Response (for debugging)"),
            html.Pre(str(result)[:1000] + "..." if len(str(result)) > 1000 else str(result), 
                    style={"fontSize": "12px", "backgroundColor": "#f8f9fa", "padding": "10px"})
        ])
    ]) 