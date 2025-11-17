"""
Context Version Confirmation Modal Component

This modal appears when users try to save a new dataset version,
warning them that all existing context (chat and explain) will be cleared.
"""

from dash import html, dcc
import dash_bootstrap_components as dbc


def context_version_modal():
    """
    Create a confirmation modal for dataset version changes that will clear context.
    
    Returns:
        dbc.Modal: The confirmation modal component
    """
    return dbc.Modal([
        dbc.ModalHeader([
            dbc.ModalTitle([
                html.I(className="fas fa-exclamation-triangle me-2", style={"color": "#f39c12"}),
                "Confirm Dataset Version Change"
            ])
        ]),
        dbc.ModalBody([
            html.Div([
                html.H5("⚠️ This action will clear all existing context data", 
                       className="text-warning mb-3"),
                html.P([
                    "Creating a new dataset version will automatically clear all existing context items from both Chat and Explain stages. This includes:"
                ], className="mb-3"),
                html.Ul([
                    html.Li("All items added via 'Add to Chat' buttons"),
                    html.Li("All items added via 'Add to Explain' buttons"),
                    html.Li("Any accumulated analysis context from previous sessions")
                ], className="mb-3"),
                html.P([
                    "This ensures that all analysis is based on the current dataset version and prevents confusion from outdated context."
                ], className="mb-3"),
                html.Div([
                    html.Strong("Are you sure you want to proceed?"),
                    html.Br(),
                    html.Small("You can always add new context items after creating the new version.", 
                             className="text-muted")
                ], className="alert alert-info")
            ])
        ]),
        dbc.ModalFooter([
            dbc.Button("Cancel", id="context-version-cancel", color="secondary", className="me-2"),
            dbc.Button("Yes, Clear Context & Create Version", 
                      id="context-version-confirm", 
                      color="warning")
        ])
    ], 
    id="context-version-modal",
    is_open=False,
    backdrop="static",
    keyboard=False,
    size="lg"
    )
