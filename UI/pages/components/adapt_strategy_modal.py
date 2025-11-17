# """
# Adaptation Strategy Selection Modal Component

# This module provides the UI component for selecting adaptation strategies
# (Retrain vs Finetune) when the user clicks the Adapt button in the chat interface.

# Author: DriftNavi Team
# Created: 2025
# """

# from dash import html, dcc
# import dash_bootstrap_components as dbc
# from typing import Dict, Any


# def create_adapt_strategy_modal() -> html.Div:
#     """
#     Create the adaptation strategy selection modal.
    
#     This modal appears when the user clicks the Adapt button, allowing them to choose
#     between Retrain and Finetune strategies for data adaptation.
    
#     Returns:
#         html.Div: Modal component for strategy selection
#     """
#     return html.Div([
#         dbc.Modal([
#             dbc.ModalHeader([
#                 html.H4([
#                     html.I(className="fas fa-wrench me-2", style={"color": "#28a745"}),
#                     "Select Adaptation Strategy"
#                 ], className="modal-title")
#             ]),
#             dbc.ModalBody([
#                 html.P(
#                     "Choose your approach for adapting the model to handle distribution shifts between your datasets:",
#                     className="mb-4 text-muted"
#                 ),
                
#                 # Strategy selection cards
#                 html.Div([
#                     # Retrain Strategy Card
#                     html.Div(
#                         id="retrain-strategy-card",
#                         n_clicks=0
#                     ),
                    
#                     # Finetune Strategy Card
#                     html.Div(
#                         id="finetune-strategy-card",
#                         n_clicks=0
#                     ),
#                 ]),
                
#                 # Selection indicator
#                 html.Div([
#                     dbc.Alert(
#                         "Please select a strategy to continue",
#                         color="info",
#                         id="strategy-selection-alert",
#                         is_open=False,
#                         className="mt-3"
#                     )
#                 ]),
                
#                 # Selected strategy display
#                 html.Div([
#                     html.H6("Selected Strategy:", className="mt-3 mb-2"),
#                     html.Div(id="selected-strategy-display", className="p-3 bg-light rounded")
#                 ], id="selected-strategy-container", style={"display": "none"}),
                
#             ]),
#             dbc.ModalFooter([
#                 dbc.Button("Cancel", id="adapt-strategy-cancel", 
#                           color="secondary", className="me-2", n_clicks=0),
#                 dbc.Button("Confirm Strategy", id="adapt-strategy-confirm", 
#                           color="primary", disabled=True, n_clicks=0)
#             ])
#         ],
#         id="adapt-strategy-modal",
#         is_open=False,
#         size="lg",
#         backdrop=True,
#         keyboard=True,
#         centered=True
#         ),
        
#         # Store for selected strategy
#         dcc.Store(id="selected-adaptation-strategy", data=None)
#     ])


# def get_strategy_info(strategy: str) -> Dict[str, Any]:
#     """
#     Get detailed information about a specific adaptation strategy.
    
#     Args:
#         strategy (str): Strategy name ('retrain' or 'finetune')
        
#     Returns:
#         Dict[str, Any]: Strategy information including description, icon, color
#     """
#     strategies = {
#         "retrain": {
#             "name": "Retrain Model",
#             "icon": "fas fa-sync-alt",
#             "color": "#dc3545",
#             "description": "Completely rebuild the model using both datasets",
#             "details": [
#                 "Merges primary and secondary datasets",
#                 "Reapplies data balancing and preprocessing", 
#                 "Rebuilds model from scratch",
#                 "Suitable for major distribution changes"
#             ],
#             "use_cases": [
#                 "Significant distribution shifts (>30% statistical difference)",
#                 "New data contains completely new feature patterns",
#                 "Original model performance has significantly degraded",
#                 "Sufficient computational resources available"
#             ]
#         },
#         "finetune": {
#             "name": "Finetune Model", 
#             "icon": "fas fa-tools",
#             "color": "#ffc107",
#             "description": "Adapt the existing model using incremental data modifications",
#             "details": [
#                 "Preserves original dataset structure",
#                 "Applies domain adaptation techniques",
#                 "Incremental model updates", 
#                 "Suitable for minor distribution changes"
#             ],
#             "use_cases": [
#                 "Moderate distribution shifts (<30% statistical difference)",
#                 "New data has similarities with original data",
#                 "Need to maintain model stability",
#                 "Limited computational resources"
#             ]
#         }
#     }
    
#     return strategies.get(strategy, {
#         "name": "Unknown Strategy",
#         "icon": "fas fa-question-circle",
#         "color": "#6c757d",
#         "description": "Strategy information not available",
#         "details": [],
#         "use_cases": []
#     })
