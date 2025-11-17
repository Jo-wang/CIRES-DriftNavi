"""
Context management utilities for Chat and Explain features.

This module provides shared functions for managing context items across
both Chat and Explain functionalities.
"""

import copy
import datetime
import random
import time
from typing import Dict, List, Any, Optional


def generate_context_id(context_type: str) -> str:
    """
    Generate a unique context item ID.
    
    Args:
        context_type: Type of context (e.g., 'target-dist', 'cond-dist', etc.)
        
    Returns:
        str: Unique context ID
    """
    timestamp = int(time.time() * 1000)
    random_suffix = random.randint(1000, 9999)
    return f"{context_type}-ctx-{timestamp}-{random_suffix}"


def validate_context_list(context_data: Any) -> List[Dict]:
    """
    Validate and normalize context data to ensure it's a proper list.
    
    Args:
        context_data: Raw context data from store
        
    Returns:
        List[Dict]: Validated and normalized context list
    """
    if context_data is None:
        return []
    
    if not isinstance(context_data, list):
        try:
            if isinstance(context_data, dict):
                return [context_data]
            else:
                return []
        except Exception:
            return []
    
    # Filter out invalid items
    valid_items = []
    for item in context_data:
        if isinstance(item, dict) and 'id' in item and 'type' in item:
            valid_items.append(item)
    
    return valid_items


def create_context_item(
    context_id: str,
    context_type: str,
    summary_text: str,
    **kwargs
) -> Dict[str, Any]:
    """
    Create a standardized context item dictionary.
    
    Args:
        context_id: Unique identifier for the context item
        context_type: Type of context item
        summary_text: Detailed text summary of the analysis
        **kwargs: Additional type-specific fields
        
    Returns:
        Dict: Standardized context item
    """
    base_item = {
        "id": context_id,
        "type": context_type,
        "summary_text": summary_text,
        "expanded": False,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Add any additional fields
    base_item.update(kwargs)
    
    return base_item


def add_context_item_to_list(
    current_context: List[Dict],
    new_item: Dict[str, Any]
) -> List[Dict]:
    """
    Safely add a new context item to the existing context list with deduplication.
    
    Args:
        current_context: Current context items list
        new_item: New context item to add
        
    Returns:
        List[Dict]: Updated context list with new item (if not duplicate)
    """
    # Validate current context
    validated_context = validate_context_list(current_context)
    
    # Deduplication for conditional distribution items
    if new_item.get('type') == 'conditional_distribution':
        for existing_item in validated_context:
            if (existing_item.get('type') == 'conditional_distribution' and
                existing_item.get('target_attribute') == new_item.get('target_attribute') and
                existing_item.get('target_value') == new_item.get('target_value') and
                existing_item.get('compare_attribute') == new_item.get('compare_attribute')):
                # Skip duplicate conditional distribution
                return validated_context
    
    # Deduplication for target distribution items
    if new_item.get('type') == 'target_distribution':
        for existing_item in validated_context:
            if (existing_item.get('type') == 'target_distribution' and
                existing_item.get('target_attribute') == new_item.get('target_attribute')):
                # Skip duplicate target distribution
                print(f"[DEDUPLICATION] Skipping duplicate target_distribution for '{new_item.get('target_attribute')}'")
                return validated_context
    
    # Create a deep copy to avoid reference issues
    updated_context = copy.deepcopy(validated_context)
    updated_context.append(new_item)
    
    return updated_context


def filter_buttons_from_component(component):
    """
    Recursively remove button components from a Dash component tree.
    
    This is used to create button-free versions of charts for context storage.
    
    Args:
        component: Dash component tree
        
    Returns:
        Cleaned component tree without buttons
    """
    if component is None:
        return None
        
    # If it's a dictionary (representing a component)
    if isinstance(component, dict):
        # Skip buttons we want to remove
        button_ids = [
            'add-target-dist-to-chat',
            'add-target-dist-to-explain', 
            'add-cond-dist-to-chat',
            'add-cond-dist-to-explain',
            'add-distribution-to-chat',
            'add-distribution-to-explain'
        ]
        
        if component.get('props', {}).get('id') in button_ids:
            return None
            
        # Process children recursively if they exist
        if 'props' in component and 'children' in component['props']:
            children = component['props']['children']
            
            if isinstance(children, list):
                # Filter the list of children
                filtered_children = []
                for child in children:
                    filtered_child = filter_buttons_from_component(child)
                    if filtered_child is not None:
                        filtered_children.append(filtered_child)
                component['props']['children'] = filtered_children
            else:
                # Single child case
                filtered_child = filter_buttons_from_component(children)
                component['props']['children'] = filtered_child
                
        return component
    
    # If it's a list (of components)
    elif isinstance(component, list):
        filtered_components = []
        for item in component:
            filtered_item = filter_buttons_from_component(item)
            if filtered_item is not None:
                filtered_components.append(filtered_item)
        return filtered_components
    
    # For simple values or strings, just return them
    return component


def create_button_feedback_content(
    action_type: str,
    target_name: str,
    success: bool = True
) -> tuple:
    """
    Create consistent button feedback content and styling.
    
    Args:
        action_type: Type of action (e.g., 'chat', 'explain')
        target_name: Name of the target/attribute
        success: Whether the action was successful
        
    Returns:
        tuple: (content_elements, style_dict)
    """
    if success:
        from dash import html
        
        content = [
            html.I(className="fas fa-check me-1"),
            "Added!",
            html.Span(
                f"Added to {action_type} context",
                className="visually-hidden",
                **{"aria-live": "polite"}
            )
        ]
        
        style = {
            "backgroundColor": "#28a745",
            "color": "white",
            "border": "none",
            "borderRadius": "3px",
            "padding": "0 8px",
            "height": "28px",
            "transition": "all 0.5s"
        }
    else:
        content = [
            html.I(className="fas fa-exclamation-triangle me-1"),
            "Error"
        ]
        
        style = {
            "backgroundColor": "#dc3545",
            "color": "white",
            "border": "none",
            "borderRadius": "3px",
            "padding": "0 8px",
            "height": "28px"
        }
    
    return content, style


def create_original_button_content(action_type: str) -> tuple:
    """
    Create original button content and styling for reset.
    
    Args:
        action_type: Type of action ('chat' or 'explain')
        
    Returns:
        tuple: (content_elements, style_dict)
    """
    from dash import html
    
    if action_type == "chat":
        icon = "fas fa-comments"
        text = "Add to Chat"
        color = "#0d6efd"
    else:  # explain
        icon = "fas fa-chart-line"
        text = "Add to Explain"
        color = "#198754"
    
    content = [
        html.I(className=f"{icon} me-1"),
        text
    ]
    
    style = {
        "backgroundColor": "transparent",
        "color": color,
        "border": f"1px solid {color}",
        "borderRadius": "3px",
        "padding": "0 8px",
        "height": "28px"
    }
    
    return content, style 