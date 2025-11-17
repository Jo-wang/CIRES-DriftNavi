"""
Dash Component Text Converter

This module provides utility functions to convert Dash components to plain text,
ensuring that no formatting placeholders or special characters cause issues when
the text is used in string templates or LLM prompts.
"""

from dash import html, dcc
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import json
import re

def dash_to_text(component, max_depth=10):
    """
    Convert a Dash component or structure to plain text.
    
    This function recursively extracts text content from Dash components,
    properly handles formatting placeholders, and ensures the result is
    suitable for use in string templates or LLM prompts.
    
    Args:
        component: The Dash component or data structure to convert
        max_depth (int): Maximum recursion depth to prevent infinite loops
        
    Returns:
        str: Plain text representation of the component
    """
    if max_depth <= 0:
        return "[Recursion limit reached]"
    
    # Handle None case
    if component is None:
        return ""
    
    # Handle basic types directly
    if isinstance(component, (int, float, bool)):
        return str(component)
    
    # Handle string with escape for format placeholders
    if isinstance(component, str):
        # Double all curly braces to escape them for formatting functions
        return component.replace("{", "{{").replace("}", "}}")
    
    # Handle lists and tuples
    if isinstance(component, (list, tuple)):
        results = [dash_to_text(item, max_depth - 1) for item in component]
        return "\n".join(item for item in results if item)
    
    # Handle dictionaries
    if isinstance(component, dict):
        # Check if this is a Dash component in dict form
        if "props" in component and "type" in component:
            return _process_component_dict(component, max_depth)
        
        # Regular dictionary
        results = []
        for key, value in component.items():
            text_value = dash_to_text(value, max_depth - 1)
            if text_value:
                results.append(f"{key}: {text_value}")
        return "\n".join(results)
    
    # Handle DataFrames
    if isinstance(component, pd.DataFrame):
        return component.to_string()
    
    # Handle numpy arrays
    if isinstance(component, np.ndarray):
        return np.array2string(component)
    
    # Handle specific Dash components
    return _process_dash_component(component, max_depth)

def _process_component_dict(component_dict, max_depth):
    """
    Process a Dash component in dictionary form.
    
    Args:
        component_dict (dict): Dictionary representation of a Dash component
        max_depth (int): Current recursion depth
        
    Returns:
        str: Text content of the component
    """
    # Extract type and props
    component_type = component_dict.get("type", "")
    props = component_dict.get("props", {})
    
    # Handle children specially
    if "children" in props:
        return dash_to_text(props["children"], max_depth - 1)
    
    # Extract any text content from props
    if "value" in props:
        return dash_to_text(props["value"], max_depth - 1)
    
    if "data" in props:
        return dash_to_text(props["data"], max_depth - 1)
    
    # If no content found, return empty string
    return ""

def _process_dash_component(component, max_depth):
    """
    Process a Dash component object to extract its text content.
    
    Args:
        component: A Dash component object
        max_depth (int): Current recursion depth
        
    Returns:
        str: Text content of the component
    """
    # Handle HTML components
    if isinstance(component, html.Div) or isinstance(component, html.Span) or \
       isinstance(component, dbc.Container) or isinstance(component, dbc.Row) or \
       isinstance(component, dbc.Col):
        return dash_to_text(component.children, max_depth - 1)
    
    # Handle text components
    if isinstance(component, html.P) or isinstance(component, html.H1) or \
       isinstance(component, html.H2) or isinstance(component, html.H3) or \
       isinstance(component, html.H4) or isinstance(component, html.H5) or \
       isinstance(component, html.H6) or isinstance(component, html.Label):
        return dash_to_text(component.children, max_depth - 1)
    
    # Handle special text components
    if isinstance(component, html.Strong) or isinstance(component, html.Em) or \
       isinstance(component, html.Pre) or isinstance(component, html.Code):
        return dash_to_text(component.children, max_depth - 1)
    
    # Handle lists
    if isinstance(component, html.Ul) or isinstance(component, html.Ol):
        items = dash_to_text(component.children, max_depth - 1)
        return items
    
    if isinstance(component, html.Li):
        content = dash_to_text(component.children, max_depth - 1)
        return f"• {content}"
    
    # Handle tables
    if isinstance(component, html.Table):
        return _process_table_component(component, max_depth)
    
    # Handle dash core components
    if isinstance(component, dcc.Graph):
        # For graphs, extract the figure title or create a description
        figure = getattr(component, 'figure', {})
        if figure and isinstance(figure, dict):
            title = figure.get('layout', {}).get('title', {})
            if isinstance(title, dict):
                return title.get('text', 'Graph visualization')
            return str(title) if title else 'Graph visualization'
        return 'Graph visualization'
    
    if isinstance(component, dcc.Markdown):
        return component.children if component.children else ""
    
    # Handle Bootstrap components
    if isinstance(component, dbc.Card):
        card_content = []
        for part in ['header', 'body', 'footer']:
            part_content = getattr(component, part, None)
            if part_content:
                card_content.append(dash_to_text(part_content, max_depth - 1))
        return "\n".join(card_content)
    
    if isinstance(component, dbc.Table):
        return _process_table_component(component, max_depth)
    
    # For any other components, try to extract children or value
    try:
        if hasattr(component, 'children'):
            return dash_to_text(component.children, max_depth - 1)
        
        if hasattr(component, 'value'):
            return str(component.value)
        
        # Last resort: convert to string and escape format placeholders
        component_str = str(component)
        return component_str.replace("{", "{{").replace("}", "}}")
    except:
        return "[Component text extraction failed]"

def _process_table_component(table_component, max_depth):
    """
    Process a table component to extract structured text.
    
    Args:
        table_component: A Dash table component
        max_depth (int): Current recursion depth
        
    Returns:
        str: Text representation of the table
    """
    result = []
    
    # Try to process the table headers
    headers = None
    if hasattr(table_component, 'children'):
        for child in table_component.children:
            if isinstance(child, html.Thead):
                headers = dash_to_text(child, max_depth - 1)
                result.append(headers)
            
            if isinstance(child, html.Tbody):
                rows = dash_to_text(child, max_depth - 1)
                result.append(rows)
    
    return "\n".join(result)

def escape_format_placeholders(text):
    """
    Escapes format string placeholders in text.
    
    Args:
        text (str): Original text with potential format placeholders
        
    Returns:
        str: Text with escaped format placeholders
    """
    if not isinstance(text, str):
        return str(text)
    
    # Escape curly braces by doubling them
    return text.replace("{", "{{").replace("}", "}}")
