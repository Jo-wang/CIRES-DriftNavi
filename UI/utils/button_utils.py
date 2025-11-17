"""
Button utility functions for creating consistent dual add-to-context buttons.

This module provides functions to create standardized button groups for
adding content to both Chat and Explain contexts.
"""

from dash import html
import dash_bootstrap_components as dbc


def create_dual_add_buttons(
    feature_name: str,
    chat_button_id: str,
    explain_button_id: str,
    chat_disabled: bool = False,
    explain_disabled: bool = False,
    chat_aria_disabled: str = "false",
    explain_aria_disabled: str = "false"
) -> html.Div:
    """
    Create a dual button group for adding content to Chat and Explain contexts.
    
    Args:
        feature_name: Name of the feature (e.g., "target distribution", "conditional distribution")
        chat_button_id: HTML ID for the chat button
        explain_button_id: HTML ID for the explain button  
        chat_disabled: Whether chat button is disabled
        explain_disabled: Whether explain button is disabled
        chat_aria_disabled: Aria-disabled value for chat button
        explain_aria_disabled: Aria-disabled value for explain button
        
    Returns:
        html.Div: Button group container with both buttons
    """
    
    chat_button = html.Button(
        [
            html.I(className="fas fa-comments me-1"),
            "Add to Chat",
            html.Span(
                f"Add {feature_name} data to chat context",
                className="visually-hidden",
                **{"aria-live": "polite"}
            )
        ],
        id=chat_button_id,
        className="btn btn-sm btn-primary me-1",
        disabled=chat_disabled,
        **{
            "aria-label": f"Add {feature_name} to chat",
            "aria-disabled": chat_aria_disabled
        },
        style={
            "fontSize": "0.75rem",
            "padding": "0.25rem 0.5rem",
            "borderRadius": "0.25rem"
        }
    )
    
    explain_button = html.Button(
        [
            html.I(className="fas fa-chart-line me-1"),
            "Add to Explain",
            html.Span(
                f"Add {feature_name} data to explain context",
                className="visually-hidden",
                **{"aria-live": "polite"}
            )
        ],
        id=explain_button_id,
        className="btn btn-sm btn-success",
        disabled=explain_disabled,
        **{
            "aria-label": f"Add {feature_name} to explain",
            "aria-disabled": explain_aria_disabled
        },
        style={
            "fontSize": "0.75rem",
            "padding": "0.25rem 0.5rem",
            "borderRadius": "0.25rem"
        }
    )
    
    return html.Div(
        [chat_button, explain_button],
        className="d-flex align-items-center mt-2 justify-content-center",
        style={"gap": "0.5rem"}
    )


def create_dual_modal_buttons(
    feature_name: str,
    chat_button_id: str,
    explain_button_id: str
) -> html.Div:
    """
    Create dual buttons for modal footers with consistent styling.
    
    Args:
        feature_name: Name of the feature
        chat_button_id: HTML ID for the chat button
        explain_button_id: HTML ID for the explain button
        
    Returns:
        html.Div: Button group for modal footer
    """
    
    chat_button = dbc.Button(
        [
            html.I(className="fas fa-comments me-1"),
            "Add to Chat"
        ],
        id=chat_button_id,
        color="primary",
        outline=True,
        size="sm",
        title=f"Add {feature_name} to chat context"
    )
    
    explain_button = dbc.Button(
        [
            html.I(className="fas fa-chart-line me-1"),
            "Add to Explain"
        ],
        id=explain_button_id,
        color="success",
        outline=True,
        size="sm",
        title=f"Add {feature_name} to explain context"
    )
    
    return html.Div(
        [chat_button, explain_button],
        className="d-flex gap-2"
    )


def create_single_context_button(
    action_type: str,
    feature_name: str,
    button_id: str,
    disabled: bool = False,
    aria_disabled: str = "false"
) -> html.Button:
    """
    Create a single context button (for backward compatibility or special cases).
    
    Args:
        action_type: 'chat' or 'explain'
        feature_name: Name of the feature
        button_id: HTML ID for the button
        disabled: Whether button is disabled
        aria_disabled: Aria-disabled value
        
    Returns:
        html.Button: Single context button
    """
    
    if action_type == "chat":
        icon = "fas fa-comments"
        text = "Add to Chat"
        color_class = "btn-primary"
        color_value = "#0d6efd"
    else:  # explain
        icon = "fas fa-chart-line"
        text = "Add to Explain"
        color_class = "btn-success"
        color_value = "#198754"
    
    return html.Button(
        [
            html.I(className=f"{icon} me-1"),
            text,
            html.Span(
                f"Add {feature_name} to {action_type} context",
                className="visually-hidden",
                **{"aria-live": "polite"}
            )
        ],
        id=button_id,
        className=f"btn btn-sm {color_class}",
        disabled=disabled,
        **{
            "aria-label": f"Add {feature_name} to {action_type}",
            "aria-disabled": aria_disabled
        },
        style={
            "fontSize": "0.75rem",
            "padding": "0.25rem 0.5rem",
            "borderRadius": "0.25rem"
        }
    ) 