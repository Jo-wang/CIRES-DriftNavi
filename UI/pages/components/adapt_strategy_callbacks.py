# """
# Adaptation Strategy Selection Callbacks

# This module contains all callbacks related to the adaptation strategy selection modal,
# including card selection, strategy confirmation, and integration with the chat system.

# Author: DriftNavi Team
# Created: 2025
# """

# from dash import callback, Input, Output, State, html, dash
# import dash_bootstrap_components as dbc
# from UI.functions.global_vars import global_vars
# from UI.pages.components.adapt_strategy_modal import get_strategy_info


# @callback(
#     [Output("retrain-strategy-card", "children"),
#      Output("finetune-strategy-card", "children")],
#     [Input("adapt-strategy-modal", "is_open")],
#     prevent_initial_call=True
# )
# def initialize_strategy_cards(is_open):
#     """
#     Initialize strategy cards when modal opens.
    
#     Args:
#         is_open: Whether the modal is open
        
#     Returns:
#         tuple: Initial card children for both strategies
#     """
#     if is_open:
#         return (
#             create_strategy_card("retrain", False),
#             create_strategy_card("finetune", False)
#         )
#     else:
#         return (
#             html.Div(),  # Empty div when modal is closed
#             html.Div()   # Empty div when modal is closed
#         )


# @callback(
#     [Output("retrain-strategy-card", "children"),
#      Output("finetune-strategy-card", "children"),
#      Output("selected-strategy-container", "style"),
#      Output("selected-strategy-display", "children"),
#      Output("adapt-strategy-confirm", "disabled"),
#      Output("selected-adaptation-strategy", "data")],
#     [Input("retrain-strategy-card", "n_clicks"),
#      Input("finetune-strategy-card", "n_clicks")],
#     [State("selected-adaptation-strategy", "data")],
#     prevent_initial_call=True
# )
# def handle_strategy_card_selection(retrain_clicks, finetune_clicks, current_strategy):
#     """
#     Handle strategy card selection and update UI accordingly.
    
#     Args:
#         retrain_clicks: Number of clicks on retrain card
#         finetune_clicks: Number of clicks on finetune card
#         current_strategy: Currently selected strategy
        
#     Returns:
#         tuple: Updated card children, display visibility, confirm button state, and selected strategy
#     """
#     ctx = dash.callback_context
    
#     if not ctx.triggered:
#         # Default state - no selection
#         return (
#             create_strategy_card("retrain", False),  # retrain card children
#             create_strategy_card("finetune", False),  # finetune card children
#             {"display": "none"},  # selected strategy container
#             "",  # selected strategy display
#             True,  # confirm button disabled
#             None  # selected strategy
#         )
    
#     triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
#     if triggered_id == "retrain-strategy-card":
#         selected_strategy = "retrain"
#         retrain_selected = True
#         finetune_selected = False
#     elif triggered_id == "finetune-strategy-card":
#         selected_strategy = "finetune"
#         retrain_selected = False
#         finetune_selected = True
#     else:
#         # Fallback to current strategy or no selection
#         selected_strategy = current_strategy
#         retrain_selected = (current_strategy == "retrain")
#         finetune_selected = (current_strategy == "finetune")
    
#     # Create selected strategy display
#     if selected_strategy:
#         strategy_info = get_strategy_info(selected_strategy)
#         display_content = html.Div([
#             html.H6([
#                 html.I(className=f"{strategy_info['icon']} me-2", 
#                       style={"color": strategy_info['color']}),
#                 strategy_info['name']
#             ], style={"color": strategy_info['color'], "marginBottom": "10px"}),
#             html.P(strategy_info['description'], className="mb-2"),
#             html.Small([
#                 html.Strong("Key features: "),
#                 ", ".join(strategy_info['details'][:2])  # Show first 2 details
#             ], className="text-muted")
#         ])
        
#         return (
#             create_strategy_card("retrain", retrain_selected),
#             create_strategy_card("finetune", finetune_selected),
#             {"display": "block"},
#             display_content,
#             False,  # Enable confirm button
#             selected_strategy
#         )
#     else:
#         return (
#             create_strategy_card("retrain", retrain_selected),
#             create_strategy_card("finetune", finetune_selected),
#             {"display": "none"},
#             "",
#             True,  # Disable confirm button
#             selected_strategy
#         )


# def create_strategy_card(strategy, is_selected):
#     """
#     Create a strategy card with appropriate styling based on selection state.
    
#     Args:
#         strategy: Strategy name ('retrain' or 'finetune')
#         is_selected: Whether this card is currently selected
        
#     Returns:
#         html.Div: Strategy card component
#     """
#     if strategy == "retrain":
#         icon_class = "fas fa-sync-alt fa-2x mb-3"
#         icon_color = "#dc3545"
#         title = "Retrain Model"
#         title_color = "#dc3545"
#         description = (
#             "Completely rebuild the model using both datasets. "
#             "Recommended when distribution shifts are significant and "
#             "the original model structure may not be suitable."
#         )
#         features = [
#             "Merges primary and secondary datasets",
#             "Reapplies data balancing and preprocessing",
#             "Rebuilds model from scratch",
#             "Suitable for major distribution changes"
#         ]
#     else:  # finetune
#         icon_class = "fas fa-tools fa-2x mb-3"
#         icon_color = "#ffc107"
#         title = "Finetune Model"
#         title_color = "#ffc107"
#         description = (
#             "Adapt the existing model using incremental data modifications. "
#             "Recommended when distribution shifts are moderate and "
#             "the original model structure remains valid."
#         )
#         features = [
#             "Preserves original dataset structure",
#             "Applies domain adaptation techniques",
#             "Incremental model updates",
#             "Suitable for minor distribution changes"
#         ]
    
#     # Card styling based on selection state
#     if is_selected:
#         card_style = {
#             "cursor": "pointer", 
#             "border": "3px solid #007bff", 
#             "boxShadow": "0 4px 8px rgba(0,123,255,0.3)", 
#             "transition": "all 0.3s ease"
#         }
#     else:
#         card_style = {
#             "cursor": "pointer", 
#             "border": "2px solid #e9ecef", 
#             "transition": "all 0.3s ease"
#         }
    
#     return html.Div([
#         dbc.Card([
#             dbc.CardBody([
#                 html.Div([
#                     html.I(className=icon_class, style={"color": icon_color}),
#                     html.H5(title, className="card-title", style={"color": title_color}),
#                     html.P(description, className="card-text"),
#                     html.Ul([
#                         html.Li(feature) for feature in features
#                     ], className="small text-muted"),
#                 ], className="text-center")
#             ])
#         ], 
#         className="strategy-card mb-3",
#         style=card_style)
#     ])


# @callback(
#     [Output("adapt-strategy-modal", "is_open"),
#      Output("strategy-selection-alert", "is_open"),
#      Output("strategy-selection-alert", "children"),
#      Output("strategy-selection-alert", "color")],
#     [Input("adapt-strategy-confirm", "n_clicks"),
#      Input("adapt-strategy-cancel", "n_clicks")],
#     [State("selected-adaptation-strategy", "data"),
#      State("adapt-strategy-modal", "is_open")],
#     prevent_initial_call=True
# )
# def handle_strategy_confirmation(confirm_clicks, cancel_clicks, selected_strategy, is_open):
#     """
#     Handle strategy confirmation or cancellation.
    
#     Args:
#         confirm_clicks: Number of clicks on confirm button
#         cancel_clicks: Number of clicks on cancel button
#         selected_strategy: Currently selected strategy
#         is_open: Current modal state
        
#     Returns:
#         tuple: Modal state, alert visibility, alert content, alert color
#     """
#     ctx = dash.callback_context
    
#     if not ctx.triggered:
#         return False, False, "", "info"
    
#     triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
#     if triggered_id == "adapt-strategy-cancel":
#         # Cancel - close modal
#         return False, False, "", "info"
    
#     elif triggered_id == "adapt-strategy-confirm":
#         if not selected_strategy:
#             # No strategy selected - show error
#             return True, True, "Please select a strategy before confirming.", "warning"
        
#         # Strategy selected - save to global variables and close modal
#         try:
#             global_vars.set_adaptation_strategy(selected_strategy)
#             print(f"[ADAPT STRATEGY] Strategy confirmed: {selected_strategy}")
            
#             # Show success message briefly
#             return False, True, f"Strategy '{selected_strategy}' selected successfully!", "success"
            
#         except Exception as e:
#             print(f"[ADAPT STRATEGY] Error saving strategy: {str(e)}")
#             return True, True, f"Error saving strategy: {str(e)}", "danger"
    
#     return is_open, False, "", "info"


# @callback(
#     [Output("adapt-strategy-modal", "is_open", allow_duplicate=True),
#      Output("retrain-strategy-card", "n_clicks"),
#      Output("finetune-strategy-card", "n_clicks"),
#      Output("selected-adaptation-strategy", "data", allow_duplicate=True)],
#     [Input("chat-adapt-btn", "n_clicks")],
#     [State("adapt-strategy-modal", "is_open")],
#     prevent_initial_call=True
# )
# def handle_adapt_button_click(adapt_clicks, is_open):
#     """
#     Handle Adapt button click to open strategy selection modal.
    
#     Args:
#         adapt_clicks: Number of clicks on adapt button
#         is_open: Current modal state
        
#     Returns:
#         tuple: Modal state, reset card clicks, reset selected strategy
#     """
#     if not adapt_clicks:
#         return is_open, 0, 0, None
    
#     print(f"[ADAPT BUTTON] Adapt button clicked (click #{adapt_clicks})")
    
#     # Open modal and reset state
#     return True, 0, 0, None


# @callback(
#     Output("strategy-selection-alert", "is_open", allow_duplicate=True),
#     [Input("strategy-selection-alert", "is_open")],
#     prevent_initial_call=True
# )
# def auto_hide_alert(is_open):
#     """
#     Auto-hide alert after a few seconds if it's a success message.
    
#     Args:
#         is_open: Current alert state
        
#     Returns:
#         bool: Alert visibility state
#     """
#     if is_open:
#         # Auto-hide after 3 seconds for success messages
#         import time
#         time.sleep(3)
#         return False
    
#     return is_open
