"""
Context Item Boxes Component for Explain Phase

This component displays context items from the explain phase analysis in a grid layout.
Each context item shows basic information and can be clicked to view detailed GPT analysis.
"""

from typing import Dict, Any, List, Optional
import re
import dash_bootstrap_components as dbc
from dash import html, dcc, callback, Input, Output, State, ALL, ctx
from dash.exceptions import PreventUpdate

# =============================================================================
# MAIN CONTEXT ITEM BOXES COMPONENT
# =============================================================================

def create_context_item_boxes(
    context_items: List[Dict[str, Any]], 
    analysis_data: Dict[str, Any],
    user_expertise_level: str = "intermediate"
) -> html.Div:
    """
    Create simple context item boxes for Layer 2.
    Each box shows only context type and attribute name.
    
    Args:
        context_items: List of context items from explain-context-data
        analysis_data: Analysis data from GPT containing layer2_context_analysis
        user_expertise_level: User's expertise level (not used in simple boxes)
        
    Returns:
        html.Div: Simple grid of context boxes
    """
    if not context_items:
        return html.Div([
            html.Div([
                html.I(className="fas fa-info-circle me-2", style={"color": "#6c757d"}),
                html.P("No context items detected", className="text-muted mb-0")
            ], className="text-center py-3")
        ])
    
    # Create simple boxes in a grid
    boxes = []
    for i, context_item in enumerate(context_items):
        box = create_simple_context_box(context_item, i)
        boxes.append(box)
    
    return html.Div([
        # Simple grid layout
        html.Div(
            boxes,
            className="row g-2"  # Small gaps between boxes
        )
    ])


def create_simple_context_box(
    context_item: Dict[str, Any],
    item_index: int
) -> html.Div:
    """Create a simple context box showing only type and attribute name with consistent sizing."""
    # Extract context information
    context_type = context_item.get('type', 'unknown')
    attribute_name = get_attribute_name(context_item)
    type_display = get_context_type_display(context_type)
    
    # Get type-specific color
    type_color = get_context_type_color(context_type)
    
    return html.Div([
        # Button wrapper for clickable functionality
        dbc.Button([
            dbc.Card([
                dbc.CardBody([
                    # Context type icon and name with consistent layout
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
                                className="text-muted",
                                style={
                                    "fontSize": "0.75rem",
                                    "lineHeight": "1.2",
                                    "display": "block",
                                    "wordBreak": "break-word",
                                    "overflow": "hidden",
                                    "textOverflow": "ellipsis",
                                    "whiteSpace": "nowrap",
                                    "maxWidth": "220px"
                                }
                            )
                        ], className="ms-2 flex-grow-1")
                    ], 
                    className="d-flex align-items-center justify-content-start",
                    style={"height": "100%", "minHeight": "50px"}
                    )
                ], 
                className="p-2",
                style={
                    "height": "70px",  # Fixed height for consistency
                    "display": "flex",
                    "alignItems": "center",
                    "justifyContent": "center"
                })
            ], 
            style={
                "border": f"1px solid {type_color}20",  # Light border with type color
                "borderRadius": "8px",
                "height": "70px",  # Fixed height
                "width": "100%",
                "backgroundColor": "#ffffff",
                "boxShadow": "0 1px 3px rgba(0,0,0,0.1)"
            })
        ],
        id={"type": "context-box-btn", "index": item_index},
        color="light",
        className="context-simple-box p-0",
        style={
            "border": "none",
            "background": "transparent",
            "width": "100%",
            "height": "70px",  # Consistent button height
            "cursor": "pointer",
            "transition": "all 0.2s ease"
        }
        )
    ], 
    className="col-xl-4 col-lg-4 col-md-6 col-sm-6 col-12 mb-2",  # Slightly wider on xl (3 per row)
    style={"height": "70px"}  # Ensure column height consistency
    )


def get_attribute_name(context_item: Dict[str, Any]) -> str:
    """Extract attribute name from context item based on type using comprehensive extraction logic."""
    context_type = context_item.get('type', 'unknown')
    
    if context_type == "drift_analysis":
        # For drift analysis, try attribute_name first, then target_attribute
        return context_item.get('attribute_name') or context_item.get('target_attribute', 'Unknown Attribute')
    
    elif context_type == "distribution_comparison":
        # For distribution comparison, use comprehensive extraction similar to explain_component
        attr_name = (
            context_item.get('attribute_name') or 
            context_item.get('target_attribute') or 
            context_item.get('compare_attribute')
        )
        
        if attr_name:
            return attr_name
        
        # Extract from cell_info using regex pattern matching
        cell_info = context_item.get('cell_info', '')
        if cell_info and isinstance(cell_info, str):
            import re
            # Look for patterns like "Column: attribute_name"
            match = re.search(r'(?:Column|column)\s*:?\s*([a-zA-Z_][a-zA-Z0-9_]*)', cell_info)
            if match:
                return match.group(1)
            
            # Alternative pattern: look for lines that contain attribute information
            lines = cell_info.split('\n')
            for line in lines:
                if "Column:" in line:
                    # Extract column name between "Column: " and ", Value:"
                    parts = line.split("Column:")
                    if len(parts) > 1:
                        column_part = parts[1].strip()
                        if ", Value:" in column_part:
                            return column_part.split(", Value:")[0].strip()
                        else:
                            return column_part.strip()
        
        return "Unknown Attribute"
    
    elif context_type == "conditional_distribution":
        # For conditional analysis, prefer target_attribute, fallback to compare_attribute
        return context_item.get('target_attribute') or context_item.get('compare_attribute', 'Unknown Target')
    
    elif context_type == "metric":
        # For metrics, try metric_name first, then attribute_name
        return context_item.get('metric_name') or context_item.get('attribute_name', 'Unknown Metric')
    
    elif context_type == "target_distribution":
        return context_item.get('target_attribute', 'Unknown Target')
    
    else:
        # For unknown types, try common field names
        return (context_item.get('attribute_name') or 
                context_item.get('target_attribute') or 
                context_item.get('compare_attribute') or 
                "Unknown")


def get_context_type_display(context_type: str) -> str:
    """Get display name for context type."""
    type_mapping = {
        "drift_analysis": "Drift Analysis",
        "distribution_comparison": "Distribution",
        "conditional_distribution": "Conditional",
        "metric": "Metric",
        "target_distribution": "Target",
        "unknown": "Unknown"
    }
    return type_mapping.get(context_type, "Unknown")


def get_context_type_color(context_type: str) -> str:
    """Get color for context type to ensure visual consistency."""
    color_mapping = {
        "drift_analysis": "#dc3545",      # Red for critical drift
        "distribution_comparison": "#28a745",  # Green for comparison
        "conditional_distribution": "#fd7e14", # Orange for conditional
        "metric": "#007bff",              # Blue for metrics
        "target_distribution": "#6f42c1", # Purple for target
        "unknown": "#6c757d"              # Gray for unknown
    }
    return color_mapping.get(context_type, "#6c757d")


# =============================================================================
# CONTEXT ITEM MODAL FOR GPT SCORING EXPLANATION
# =============================================================================

def create_specific_analysis_for_item(context_item: Dict[str, Any], item_index: int) -> Dict[str, Any]:
    """
    Create specific analysis for a context item when GPT analysis doesn't match.
    This ensures each item gets its own analysis instead of sharing fallback data.
    
    Args:
        context_item: The context item to analyze
        item_index: The index of the context item
        
    Returns:
        Dict: Analysis item tailored to the specific context item
    """
    context_type = context_item.get('type', 'unknown')
    attribute_name = get_attribute_name(context_item)
    
    # Create type-specific analysis
    if context_type == 'target_distribution':
        return {
            'context_id': item_index,
            'context_type': context_type,
            'risk_level': 'High',
            'title': f'Target Distribution Analysis: {attribute_name}',
            'explanation': {
                'beginner': f'The target attribute {attribute_name} shows distribution differences between datasets that could affect model predictions.',
                'intermediate': f'Distribution analysis of {attribute_name} reveals significant patterns that require careful consideration for model deployment.',
                'advanced': f'Statistical analysis of target variable {attribute_name} indicates distribution shift with potential impact on model calibration and predictive accuracy.'
            },
            'business_impact': f'Distribution differences in {attribute_name} may lead to drifted predictions and reduced model reliability in production.',
            'technical_details': f'Target variable {attribute_name} distribution analysis shows patterns requiring technical validation before deployment.',
            'action_required': f'Review {attribute_name} distribution characteristics and consider resampling or retraining strategies.'
        }
    
    elif context_type == 'drift_analysis':
        return {
            'context_id': item_index,
            'context_type': context_type,
            'risk_level': 'High',
            'title': f'Drift Analysis: {attribute_name}',
            'explanation': {
                'beginner': f'The attribute {attribute_name} shows significant drift between primary and secondary datasets.',
                'intermediate': f'Statistical drift in {attribute_name} indicates potential model performance degradation when applied to new data.',
                'advanced': f'Drift analysis of {attribute_name} reveals distributional changes that may compromise model assumptions and predictive validity.'
            },
            'business_impact': f'Drift in {attribute_name} poses risks to model accuracy and could result in incorrect business decisions.',
            'technical_details': f'Detected drift patterns in {attribute_name} suggest systematic differences requiring model adaptation.',
            'action_required': f'Investigate drift causes in {attribute_name} and implement monitoring or retraining procedures.'
        }
    
    elif context_type == 'conditional_distribution':
        target_attr = context_item.get('target_attribute', 'target')
        target_value = context_item.get('target_value', 'unknown')
        compare_attr = context_item.get('compare_attribute', 'unknown')
        
        return {
            'context_id': item_index,
            'context_type': context_type,
            'risk_level': 'Medium',
            'title': f'Conditional Analysis: {target_attr} = {target_value}',
            'explanation': {
                'beginner': f'When {target_attr} equals {target_value}, the relationship with {compare_attr} shows important patterns.',
                'intermediate': f'Conditional distribution analysis reveals how {compare_attr} behaves differently when {target_attr} = {target_value}.',
                'advanced': f'Conditional probability analysis of {compare_attr} given {target_attr} = {target_value} indicates potential interaction effects requiring consideration.'
            },
            'business_impact': f'Conditional relationships between {target_attr} and {compare_attr} may affect prediction accuracy for specific scenarios.',
            'technical_details': f'Statistical dependencies between {target_attr} and {compare_attr} show conditional patterns requiring analysis.',
            'action_required': f'Validate conditional relationships and consider feature engineering or stratified modeling approaches.'
        }
    
    elif context_type == 'distribution_comparison':
        return {
            'context_id': item_index,
            'context_type': context_type,
            'risk_level': 'Medium',
            'title': f'Distribution Comparison: {attribute_name}',
            'explanation': {
                'beginner': f'The distribution of {attribute_name} differs between the primary and secondary datasets.',
                'intermediate': f'Comparative analysis shows significant distributional differences in {attribute_name} that may impact model performance.',
                'advanced': f'Distribution comparison of {attribute_name} reveals statistical differences requiring evaluation of model transferability.'
            },
            'business_impact': f'Distribution differences in {attribute_name} may lead to performance degradation when model is applied to new data.',
            'technical_details': f'Distributional analysis of {attribute_name} shows patterns requiring technical assessment.',
            'action_required': f'Assess impact of distribution differences in {attribute_name} and consider adaptation strategies.'
        }
    
    elif context_type == 'metric':
        metric_name = context_item.get('metric_name', 'unknown')
        return {
            'context_id': item_index,
            'context_type': context_type,
            'risk_level': 'Medium',
            'title': f'Metric Analysis: {metric_name}',
            'explanation': {
                'beginner': f'The {metric_name} metric indicates potential data quality or distribution issues.',
                'intermediate': f'Statistical metric {metric_name} reveals patterns that require attention for reliable model deployment.',
                'advanced': f'Metric analysis using {metric_name} indicates statistical anomalies requiring technical evaluation.'
            },
            'business_impact': f'Issues indicated by {metric_name} may affect model reliability and business decision accuracy.',
            'technical_details': f'Statistical metric {metric_name} shows values requiring technical investigation.',
            'action_required': f'Investigate {metric_name} findings and implement appropriate remediation measures.'
        }
    
    else:
        # Generic fallback for unknown types
        return {
            'context_id': item_index,
            'context_type': context_type,
            'risk_level': 'Medium',
            'title': f'Analysis: {context_type.replace("_", " ").title()}',
            'explanation': {
                'beginner': f'This {context_type} analysis shows patterns that require attention.',
                'intermediate': f'The {context_type} analysis reveals issues that may impact model performance.',
                'advanced': f'Technical analysis of {context_type} indicates systematic patterns requiring evaluation.'
            },
            'business_impact': f'Issues identified in {context_type} analysis may affect model reliability.',
            'technical_details': f'{context_type.replace("_", " ").title()} analysis shows patterns requiring investigation.',
            'action_required': f'Review {context_type} findings and determine appropriate actions.'
        }


def create_context_item_modal() -> html.Div:
    """Create modal for GPT scoring explanation."""
    return html.Div([
        dbc.Modal([
            dbc.ModalHeader([
                html.H5("Context Analysis Details", className="modal-title")
                # 移除多余的关闭按钮，让dbc.Modal自带的关闭功能处理
            ]),
            dbc.ModalBody(
                html.Div(id="context-modal-content"),
                style={"maxHeight": "70vh", "overflowY": "auto"}
            ),
            dbc.ModalFooter([
                dbc.Button("Close", id="context-modal-close-btn", 
                          color="secondary", n_clicks=0)
            ])
        ], 
        id="context-item-modal", 
        is_open=False, 
        size="lg",
        scrollable=True),
        
        dcc.Store(id="current-modal-item-index", data=None)
    ])


def clean_section_markers(text: str) -> str:
    """
    Clean unwanted section markers and numbers from text content.
    
    Args:
        text: Raw text that may contain section markers
        
    Returns:
        str: Cleaned text without section markers
    """
    # Remove standalone numbers at the end of lines (like "2." "3." etc.)
    text = re.sub(r'\n\s*\d+\.\s*$', '', text, flags=re.MULTILINE)
    
    # Remove standalone numbers at the beginning of content
    text = re.sub(r'^\s*\d+\.\s*', '', text, flags=re.MULTILINE)
    
    # Remove patterns like "2. 3. 4." at the end
    text = re.sub(r'\s*(\d+\.\s*){2,}$', '', text)
    
    # Remove dangling numbers with periods
    text = re.sub(r'\s+\d+\.\s*$', '', text)
    
    # Clean up multiple spaces and newlines
    text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)  # Max 2 newlines
    text = re.sub(r' +', ' ', text)  # Multiple spaces to single space
    
    return text.strip()


def format_gpt_detailed_analysis(gpt_text: str) -> html.Div:
    """
    Format GPT detailed analysis text into structured sections.
    
    Args:
        gpt_text: Raw GPT analysis text
        
    Returns:
        html.Div: Formatted analysis content
    """
    if not gpt_text or len(gpt_text.strip()) < 50:
        return html.P("Detailed analysis not available.", className="text-muted")
    
    # Split text by common section markers
    sections = []
    current_section = {"title": "", "content": ""}
    
    # Look for section markers like **Section Name**:
    import re
    
    # First, split by ** markers to identify sections
    parts = re.split(r'\*\*([^*]+)\*\*:?', gpt_text)
    
    if len(parts) > 1:
        # We found section markers
        for i in range(1, len(parts), 2):
            if i < len(parts):
                section_title = parts[i].strip()
                section_content = parts[i + 1].strip() if i + 1 < len(parts) else ""
                
                if section_content:
                    sections.append({
                        "title": section_title,
                        "content": section_content
                    })
    
    # If no sections found, try alternative parsing
    if not sections:
        # Look for numbered sections (1., 2., etc.) but remove the numbers from display
        numbered_parts = re.split(r'\n\s*(\d+)\.\s*', gpt_text)
        if len(numbered_parts) > 1:
            for i in range(1, len(numbered_parts), 2):
                if i + 1 < len(numbered_parts):
                    section_num = numbered_parts[i]
                    section_content = numbered_parts[i + 1].strip()
                    
                    # Extract title from first line
                    lines = section_content.split('\n', 1)
                    title = lines[0].strip().rstrip(':')
                    content = lines[1].strip() if len(lines) > 1 else ""
                    
                    # Clean title - remove any trailing numbers or punctuation that look like section markers
                    title = re.sub(r'\s*\d+\.?\s*$', '', title)  # Remove trailing numbers
                    title = title.strip().rstrip(':').strip()   # Remove trailing colons and spaces
                    
                    # Only add section if title is meaningful
                    if title and len(title) > 2:
                        sections.append({
                            "title": title,  # Don't include section number in display
                            "content": content
                        })
    
    # If still no sections, use the whole text as one section
    if not sections:
        sections = [{"title": "Analysis", "content": gpt_text.strip()}]
    
    # Create formatted HTML
    formatted_sections = []
    
    for i, section in enumerate(sections):
        title = section["title"]
        content = section["content"]
        
        # Clean up content
        content = content.strip()
        if not content:
            continue
        
        # Remove unwanted section markers and numbers from content
        content = clean_section_markers(content)
            
        # Split content into paragraphs
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        if not paragraphs:
            paragraphs = [content]
        
        # Determine icon based on section title
        icon = get_section_icon(title)
        
        # Create section HTML
        section_html = html.Div([
            html.H6([
                html.I(className=icon, style={"color": "#6c757d", "marginRight": "8px"}),
                title
            ], className="mb-2", style={"color": "#495057", "fontWeight": "600"}),
            
            html.Div([
                html.P(paragraph, style={"marginBottom": "10px", "lineHeight": "1.5"}) 
                for paragraph in paragraphs
            ])
        ], className="mb-3", style={
            "padding": "12px 15px",
            "backgroundColor": "#f8f9fa" if i % 2 == 0 else "#ffffff",
            "borderLeft": "3px solid #007bff",
            "borderRadius": "4px"
        })
        
        formatted_sections.append(section_html)
    
    return html.Div(formatted_sections)


def get_section_icon(title: str) -> str:
    """Get appropriate icon for section based on title."""
    title_lower = title.lower()
    
    if "domain" in title_lower or "impact" in title_lower:
        return "fas fa-bullseye"
    elif "deployment" in title_lower or "risk" in title_lower:
        return "fas fa-exclamation-triangle"
    elif "reliability" in title_lower or "attribute" in title_lower:
        return "fas fa-shield-alt"
    elif "technical" in title_lower or "consequence" in title_lower:
        return "fas fa-cogs"
    elif "scenario" in title_lower or "world" in title_lower:
        return "fas fa-globe"
    elif "analysis" in title_lower:
        return "fas fa-chart-line"
    else:
        return "fas fa-info-circle"


def create_modal_content(
    context_item: Dict[str, Any],
    analysis_item: Optional[Dict[str, Any]],
    user_expertise_level: str
) -> html.Div:
    """
    Create modal content using pre-loaded detailed analysis from comprehensive GPT call.
    This avoids redundant GPT API calls and uses already-generated detailed content.
    
    Args:
        context_item: Original context item data
        analysis_item: GPT analysis data with detailed analysis already loaded
        user_expertise_level: User's expertise level
        
    Returns:
        html.Div: Modal content with detailed analysis
    """
    context_type = context_item.get('type', 'unknown')
    attribute_name = get_attribute_name(context_item)
    type_display = get_context_type_display(context_type)
    
    if analysis_item:
        # Extract GPT analysis data (only severity level, no scores)
        risk_level = analysis_item.get('risk_level', 'Medium')
        title = analysis_item.get('title', f"{type_display} - {attribute_name}")
        
        # Get explanations from GPT analysis
        explanations = analysis_item.get('explanation', {})
        base_explanation = (explanations.get(user_expertise_level) or 
                          explanations.get('intermediate') or 
                          explanations.get('beginner') or 
                          "")
        
        # Use pre-loaded detailed analysis instead of making new GPT calls
        detailed_analysis_text = analysis_item.get('detailed_analysis', '')
        
        # If no detailed analysis available, create from existing fields
        if not detailed_analysis_text:
            print(f"[MODAL DEBUG] No detailed_analysis field found, creating from existing fields")
            detailed_analysis_text = create_detailed_analysis_from_existing_fields(
                analysis_item, context_item, user_expertise_level
            )
        else:
            print(f"[MODAL DEBUG] ✅ Using pre-loaded detailed analysis ({len(detailed_analysis_text)} chars)")
            # Limit content size for better UX
            if len(detailed_analysis_text) > 2000:
                detailed_analysis_text = detailed_analysis_text[:1950] + "... [Content truncated for better readability]"
        
    else:
        # Fallback data
        risk_level = 'Medium'
        title = f"{type_display} - {attribute_name}"
        base_explanation = f"Analysis pending for {type_display} on attribute '{attribute_name}'"
        detailed_analysis_text = f"Detailed analysis not yet available for {type_display}. Please ensure the comprehensive analysis has been completed."
    
    # Get color scheme based on severity level (not score)
    severity_style = get_severity_styling(risk_level, 0)  # Pass 0 for score since we're not using it
    
    return html.Div([
        # Header with context info and severity level only
        html.Div([
            html.H4([
                html.I(className=get_context_type_icon(context_type), 
                      style={"color": severity_style["color"], "marginRight": "10px"}),
                f"{type_display}: {attribute_name}"
            ], style={"color": severity_style["color"]}),
            dbc.Badge(
                f"{risk_level} Risk",
                color=severity_style["badge_color"],
                className="ms-2",
                style={"fontSize": "1rem"}
            )
        ], className="d-flex justify-content-between align-items-center mb-4"),
        
        # Base explanation if available (brief summary)
        html.Div([
            html.Div([
                html.H6([
                    html.I(className="fas fa-info-circle me-2", style={"color": "#6c757d"}),
                    "Summary"
                ], className="mb-2"),
                html.P(base_explanation, style={"lineHeight": "1.6", "fontSize": "0.95rem"})
            ], style={
                "backgroundColor": "#e9ecef", 
                "padding": "12px", 
                "borderRadius": "6px",
                "marginBottom": "20px"
            })
        ]) if base_explanation else None,
        
        # Detailed Analysis Section - Using pre-loaded analysis (NO GPT CALLS)
        html.Div([
            html.H5([
                html.I(className="fas fa-brain me-2", style={"color": "#614385"}),
                "Domain-Specific Impact Assessment"
            ], className="mb-3", style={"color": "#495057"}),
            
            # Formatted pre-loaded detailed analysis content
            format_gpt_detailed_analysis(detailed_analysis_text)
            
        ], className="mb-4")
    ])


def generate_detailed_analysis_with_gpt(
    context_item: Dict[str, Any],
    analysis_item: Dict[str, Any],
    user_expertise_level: str
) -> str:
    """
    Generate detailed domain-specific analysis using GPT for a single context item.
    
    Args:
        context_item: The context item to analyze
        analysis_item: GPT analysis data with severity info
        user_expertise_level: User's expertise level
        
    Returns:
        str: Detailed analysis text
    """
    try:
        # Import required modules
        from agent.explain_api import generate_response_from_prompt
        from flask_login import current_user
        from .prompt_manager import (
            prompt_manager,
            create_user_context,
            create_dataset_context
        )
        
        # Get user context
        user_context = get_user_context_for_analysis(current_user)
        
        # Get dataset context
        dataset_context = get_dataset_context_for_analysis()
        
        # Prepare context for prompt manager
        prompt_context = {
            'user_context': create_user_context(user_context),
            'dataset_context': create_dataset_context(dataset_context),
            'context_item': context_item,
            'analysis_item': analysis_item
        }
        
        # Generate prompt using new template
        prompt = prompt_manager.generate_prompt('context_item_detailed_analysis', prompt_context)
        
        print(f"[CONTEXT ITEM ANALYSIS] Generating detailed analysis for {context_item.get('type', 'unknown')}")
        print(f"[CONTEXT ITEM ANALYSIS] Prompt length: {len(prompt)} characters")
        
        # Call GPT API - use user's selected model instead of hardcoded "gpt-4o"
        gpt_response = generate_response_from_prompt(prompt)
        
        print(f"[CONTEXT ITEM ANALYSIS] GPT response length: {len(gpt_response)} characters")
        
        # Return the analysis text directly
        return gpt_response.strip()
        
    except Exception as e:
        print(f"[CONTEXT ITEM ANALYSIS] Error generating detailed analysis: {str(e)}")
        print(f"[CONTEXT ITEM ANALYSIS] Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        
        # Return fallback analysis
        return generate_fallback_analysis(context_item, analysis_item, user_expertise_level)


def get_user_context_for_analysis(current_user) -> Dict[str, Any]:
    """Get user context information for analysis using centralized dynamic system."""
    try:
        # Use the centralized user context management system
        from UI.functions.global_vars import global_vars
        
        # Get actual user context without hardcoded defaults
        user_context = global_vars.get_user_context(current_user)
        
        profile_complete = user_context.get('profile_completeness', 0)
        
        if user_context.get('has_profile', False):
            print(f"[CONTEXT ITEM ANALYSIS] Retrieved user context - Profile {profile_complete:.0f}% complete")
        else:
            print(f"[CONTEXT ITEM ANALYSIS] No user profile available - using general analysis")
        
        return user_context
        
    except Exception as e:
        print(f"[CONTEXT ITEM ANALYSIS] Error getting user context: {str(e)}")
        
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


def get_dataset_context_for_analysis() -> Dict[str, Any]:
    """Get dataset context information for analysis."""
    try:
        from UI.functions.global_vars import global_vars
        
        # Get dataset information from global vars
        primary_dataset = global_vars.get_primary_dataset()
        secondary_dataset = global_vars.get_secondary_dataset()
        target_attribute = global_vars.get_target_attribute()
        
        return {
            'primary_dataset': {
                'dataset_name': getattr(primary_dataset, 'name', 'Primary Dataset') if primary_dataset else 'Primary Dataset',
                'record_count': len(primary_dataset) if primary_dataset is not None else 0,
                'columns': list(primary_dataset.columns) if primary_dataset is not None else []
            },
            'secondary_dataset': {
                'dataset_name': getattr(secondary_dataset, 'name', 'Secondary Dataset') if secondary_dataset else 'Secondary Dataset',
                'record_count': len(secondary_dataset) if secondary_dataset is not None else 0,
                'columns': list(secondary_dataset.columns) if secondary_dataset is not None else []
            },
            'comparison_context': {
                'target_attribute': target_attribute or 'Unknown',
                'analysis_type': 'drift Detection and Drift Analysis'
            }
        }
    except Exception as e:
        print(f"[DATASET CONTEXT] Error getting dataset context: {str(e)}")
        return {
            'primary_dataset': {
                'dataset_name': 'Primary Dataset',
                'record_count': 0,
                'columns': []
            },
            'secondary_dataset': {
                'dataset_name': 'Secondary Dataset',
                'record_count': 0,
                'columns': []
            },
            'comparison_context': {
                'target_attribute': 'Unknown',
                'analysis_type': 'drift Detection and Drift Analysis'
            }
        }


def generate_fallback_analysis(
    context_item: Dict[str, Any],
    analysis_item: Dict[str, Any],
    user_expertise_level: str
) -> str:
    """Generate fallback analysis when GPT is not available."""
    context_type = context_item.get('type', 'unknown')
    attribute_name = get_attribute_name(context_item)
    risk_level = analysis_item.get('risk_level', 'Medium') if analysis_item else 'Medium'
    
    if context_type == "drift_analysis":
        return f"""This drift analysis for '{attribute_name}' indicates {risk_level.lower()} risk level. 
        Distribution shifts in this attribute could affect model performance when applying the trained model to new data. 
        Consider monitoring this attribute closely and evaluating whether model retraining or adaptation strategies are needed 
        to maintain prediction accuracy."""
    
    elif context_type == "distribution_comparison":
        return f"""The distribution comparison for '{attribute_name}' shows {risk_level.lower()} risk level. 
        Significant distributional differences between datasets could lead to model degradation when deployed. 
        This may require careful evaluation of model transferability and potential drift in predictions."""
    
    elif context_type == "conditional_distribution":
        return f"""The conditional distribution analysis reveals {risk_level.lower()} risk level. 
        Changes in conditional relationships could indicate underlying data generation process shifts that 
        may affect model reliability and fairness in production environments."""
    
    else:
        return f"""This {context_type.replace('_', ' ')} analysis indicates {risk_level.lower()} risk level. 
        The detected patterns suggest potential issues that could impact model performance and reliability 
        when deployed in production environments."""


# =============================================================================
# UTILITY FUNCTIONS (Updated to not rely on severity_score)
# =============================================================================

def get_severity_styling(risk_level: str, severity_score: int = 0) -> Dict[str, str]:
    """Get styling based on severity level only (ignoring score)."""
    risk_level = risk_level.lower()
    
    if risk_level == "high" or risk_level == "critical":
        return {
            "color": "#dc3545",
            "border_color": "#dc3545",
            "badge_color": "danger",
            "bg_color": "#fff5f5"
        }
    elif risk_level == "medium" or risk_level == "moderate":
        return {
            "color": "#fd7e14",
            "border_color": "#fd7e14", 
            "badge_color": "warning",
            "bg_color": "#fffbf0"
        }
    else:  # low, minor, or any other value
        return {
            "color": "#198754",
            "border_color": "#198754",
            "badge_color": "success",
            "bg_color": "#f0fff4"
        }


def get_context_type_icon(context_type: str) -> str:
    """Get icon class for context type."""
    icon_mapping = {
        "drift_analysis": "fas fa-exchange-alt",
        "distribution_comparison": "fas fa-chart-bar",
        "conditional_distribution": "fas fa-filter",
        "metric": "fas fa-ruler",
        "target_distribution": "fas fa-bullseye",
        "unknown": "fas fa-question-circle"
    }
    return icon_mapping.get(context_type, "fas fa-question-circle") 


# =============================================================================
# CALLBACKS
# =============================================================================

@callback(
    [Output("context-item-modal", "is_open"),
     Output("context-modal-content", "children"),
     Output("current-modal-item-index", "data")],
    [Input({"type": "context-box-btn", "index": ALL}, "n_clicks"),
     Input("context-modal-close-btn", "n_clicks")],  # 移除了多余的close input
    [State("explain-context-data", "data"),
     State("unified-strategy-analysis", "data"),
     State("context-item-modal", "is_open"),
     State("current-modal-item-index", "data")],
    prevent_initial_call=True
)
def toggle_context_modal(box_clicks_list, close_btn_clicks, 
                        context_data, analysis_data, is_open, current_item_index):
    """Handle context item modal for GPT analysis explanation."""
    from dash import ctx
    
    if not ctx.triggered:
        return False, html.Div(), None
    
    trigger_id = ctx.triggered[0]["prop_id"]
    
    # Debug: Print trigger information
    print(f"[MODAL DEBUG] Trigger ID: {trigger_id}")
    print(f"[MODAL DEBUG] Context data available: {context_data is not None}")
    print(f"[MODAL DEBUG] Analysis data available: {analysis_data is not None}")
    
    # Close modal
    if "close" in trigger_id:
        print(f"[MODAL DEBUG] Closing modal")
        return False, html.Div(), None
    
    # Open modal for specific context item
    if "context-box-btn" in trigger_id and box_clicks_list and any(box_clicks_list):
        # Extract the exact item index from the triggered button using ctx.triggered
        import json
        try:
            trigger_prop = ctx.triggered[0]["prop_id"]
            # Parse the JSON-like button ID to get the index
            button_info = json.loads(trigger_prop.split('.')[0])
            item_index = button_info['index']
            print(f"[MODAL DEBUG] FIXED: Extracted exact clicked item index: {item_index}")
        except Exception as e:
            print(f"[MODAL DEBUG] Error parsing trigger: {e}, falling back to old method")
            # Fallback to old method if parsing fails
            item_index = None
            for i, clicks in enumerate(box_clicks_list):
                if clicks and clicks > 0:
                    item_index = i
                    break
            print(f"[MODAL DEBUG] Fallback clicked item index: {item_index}")
        
        print(f"[MODAL DEBUG] Final clicked item index: {item_index}")
        
        if item_index is None or not context_data or item_index >= len(context_data):
            print(f"[MODAL DEBUG] Error: Invalid item index or no context data")
            return False, html.Div([
                html.H5("Error"),
                html.P("Context item not found.")
            ]), None
        
        context_item = context_data[item_index]
        print(f"[MODAL DEBUG] Context item type: {context_item.get('type', 'unknown')}")
        
        # Get analysis data with improved mapping logic
        analysis_item = None
        if analysis_data and isinstance(analysis_data, dict):
            print(f"[MODAL DEBUG] Analysis data keys: {list(analysis_data.keys())}")
            
            if 'comprehensive_data' in analysis_data:
                comprehensive_data = analysis_data['comprehensive_data']
                print(f"[MODAL DEBUG] Comprehensive data keys: {list(comprehensive_data.keys())}")
                
                layer2_data = comprehensive_data.get('layer2_context_analysis', [])
                print(f"[MODAL DEBUG] Layer2 data count: {len(layer2_data)}")
                
                # Enhanced debugging: Print all context_ids in layer2_data
                for i, analysis in enumerate(layer2_data):
                    analysis_context_id = analysis.get('context_id')
                    analysis_type = analysis.get('context_type', 'unknown')
                    print(f"[MODAL DEBUG] Layer2 item {i}: context_id={analysis_context_id}, type={analysis_type}")
                
                print(f"[MODAL DEBUG] Looking for analysis for UI item_index {item_index}")
                print(f"[MODAL DEBUG] Context item type: {context_item.get('type', 'unknown')}")
                print(f"[MODAL DEBUG] Available GPT context_ids: {[a.get('context_id') for a in layer2_data]}")
                
                # Enhanced Strategy 1: Direct context_id match with type verification
                for analysis in layer2_data:
                    if analysis.get('context_id') == item_index:
                        # Additional verification: check if types match
                        if analysis.get('context_type') == context_item.get('type'):
                            analysis_item = analysis
                            print(f"[MODAL DEBUG] ✅ Found exact match with type verification: {item_index}")
                            break
                        else:
                            print(f"[MODAL DEBUG] ⚠️ Context_id match but type mismatch: {analysis.get('context_type')} vs {context_item.get('type')}")
                
                # Strategy 2: Fallback - array position match with bounds checking
                if analysis_item is None and 0 <= item_index < len(layer2_data):
                    analysis_item = layer2_data[item_index]
                    actual_context_id = analysis_item.get('context_id', 'unknown')
                    actual_type = analysis_item.get('context_type', 'unknown')
                    print(f"[MODAL DEBUG] ⚠️ Using array position fallback: UI index {item_index} → GPT context_id {actual_context_id}, type {actual_type}")
                
                # Strategy 3: Create specific analysis if no match found
                if analysis_item is None:
                    print(f"[MODAL DEBUG] ❌ No analysis found, creating specific analysis for item {item_index}")
                    analysis_item = create_specific_analysis_for_item(context_item, item_index)
                    print(f"[MODAL DEBUG] ✅ Created specific analysis for item {item_index}")
                
                # Final verification
                final_title = analysis_item.get('title', 'Unknown')
                expected_attribute = get_attribute_name(context_item)
                print(f"[MODAL DEBUG] 🎯 Final match - UI: {expected_attribute} | Analysis: {final_title}")
                
                # DEBUG: Print the actual analysis content to verify uniqueness
                explanation = analysis_item.get('explanation', {})
                business_impact = analysis_item.get('business_impact', '')
                print(f"[MODAL DEBUG] 📝 Analysis content preview:")
                print(f"[MODAL DEBUG]   - Title: {final_title}")
                print(f"[MODAL DEBUG]   - Explanation keys: {list(explanation.keys()) if isinstance(explanation, dict) else 'Not dict'}")
                print(f"[MODAL DEBUG]   - Business impact: {business_impact[:100]}{'...' if len(business_impact) > 100 else ''}")
                print(f"[MODAL DEBUG]   - Severity: {analysis_item.get('risk_level', 'unknown')}")
                
                # Sanity check: warn if there's a possible mismatch
                if expected_attribute.lower() not in final_title.lower() and 'specific analysis' not in final_title.lower():
                    print(f"[MODAL DEBUG] ⚠️ POSSIBLE MISMATCH: Expected {expected_attribute} but got {final_title}")
                else:
                    print(f"[MODAL DEBUG] ✅ Match verification passed")
            else:
                print(f"[MODAL DEBUG] No comprehensive_data in analysis_data")
        else:
            print(f"[MODAL DEBUG] Analysis data is not a dict or is None")
        
        # Create modal content
        print(f"[MODAL DEBUG] Creating modal content with analysis_item: {analysis_item is not None}")
        modal_content = create_modal_content(context_item, analysis_item, "intermediate")
        return True, modal_content, item_index
    
    return is_open, html.Div(), current_item_index 


def create_detailed_analysis_from_existing_fields(
    analysis_item: Dict[str, Any], 
    context_item: Dict[str, Any], 
    user_expertise_level: str
) -> str:
    """
    Create detailed analysis from existing analysis fields when detailed_analysis field is not available.
    This provides backward compatibility and fallback content.
    
    Args:
        analysis_item: GPT analysis data with existing fields
        context_item: Original context item data
        user_expertise_level: User's expertise level
        
    Returns:
        str: Detailed analysis text created from existing fields
    """
    try:
        # Extract available information
        business_impact = analysis_item.get('business_impact', '')
        technical_details = analysis_item.get('technical_details', '')
        action_required = analysis_item.get('action_required', '')
        explanations = analysis_item.get('explanation', {})
        
        # Get the most appropriate explanation level
        explanation = (explanations.get(user_expertise_level) or 
                      explanations.get('intermediate') or 
                      explanations.get('advanced') or 
                      explanations.get('beginner') or 
                      "Analysis details are being processed.")
        
        # Create comprehensive analysis text
        analysis_parts = []
        
        if explanation:
            analysis_parts.append(f"Technical Overview: {explanation}")
        
        if business_impact:
            analysis_parts.append(f"Business Impact: {business_impact}")
        
        if technical_details:
            analysis_parts.append(f"Technical Analysis: {technical_details}")
        
        if action_required:
            analysis_parts.append(f"Recommended Actions: {action_required}")
        
        # Combine all parts
        detailed_analysis = " ".join(analysis_parts)
        
        # Add fallback content if we don't have enough information
        if len(detailed_analysis) < 100:
            context_type = context_item.get('type', 'unknown')
            attribute_name = get_attribute_name(context_item)
            detailed_analysis = f"This {context_type.replace('_', ' ')} analysis for {attribute_name} requires comprehensive assessment. The detected patterns indicate potential model reliability concerns that should be evaluated in the context of your specific use case and deployment environment. Consider the implications for target attribute prediction accuracy and overall model performance when deploying to new data distributions."
        
        # Limit content size
        if len(detailed_analysis) > 2000:
            detailed_analysis = detailed_analysis[:1950] + "... [Content truncated for better readability]"
        
        return detailed_analysis
        
    except Exception as e:
        print(f"[MODAL DEBUG] Error creating detailed analysis from existing fields: {str(e)}")
        return "Detailed analysis is currently unavailable. Please try refreshing the analysis or contact support if the issue persists." 