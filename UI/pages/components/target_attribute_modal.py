"""
Target Attribute Selection Modal Component

This component is displayed after datasets are uploaded, allowing users to select a target attribute
for subsequent distribution shift analysis and model training. The selection is stored in a global
variable for use throughout the application.

This version uses a card-based selection UI instead of a dropdown for a more intuitive experience.
"""
import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from UI.functions import global_vars
from UI.pages.components.attribute_card_selector import attribute_card_selector

def target_attribute_modal():
    """
    Target Attribute Selection Modal
    
    Displayed after both datasets (primary and secondary) are uploaded, requiring users to select a target attribute
    (typically the column to be predicted) for subsequent analysis.
    
    Returns:
        dbc.Modal: Target attribute selection modal component with card-based attribute selector
    """
    return dbc.Modal(
        [
            dbc.ModalHeader(
                html.H4("Select Target Attribute", className="modal-title"),
                close_button=False  # No close button as this is a required step
            ),
            dbc.ModalBody([
                html.P(
                    "Please select the target attribute you want to predict or analyze. This column will be used for distribution shift analysis and model training.",
                    className="mb-3"
                ),
                
                # Card selector component (dynamically populated)
                html.Div(
                    id="attribute-card-container",
                    children=[]  # Will be populated by callback
                ),
                
                # Warning alert
                html.Div(
                    dbc.Alert(
                        "Please select a target attribute to continue",
                        color="warning",
                        id="target-attribute-warning",
                        is_open=False
                    ),
                    className="mt-3 mb-3"
                ),
                
                # Selected attribute display (hidden, for compatibility)
                html.Div(id="target-attribute-type-display", style={"display": "none"}),
                
                # CSS will be added in the app.py/index.py file
            ]),
            dbc.ModalFooter([
                dbc.Button(
                    "Confirm and Continue", 
                    id="confirm-target-attribute", 
                    color="primary",
                    className="ml-auto"
                )
            ])
        ],
        id="target-attribute-modal",
        backdrop="static",  # User cannot click outside to dismiss
        keyboard=False,     # User cannot press Escape key to dismiss
        is_open=False,      # Initially closed
        size="xl",          # Extra large size for better card display
        centered=True       # Centered on screen
    )
