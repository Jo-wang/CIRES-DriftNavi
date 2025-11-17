"""
Target Attribute Selection Callbacks

This module contains the interaction logic for the target attribute selection modal dialog,
including displaying attribute types, validating selections, storing selection results, and other functionalities.

This version is designed for the card-based attribute selector UI, replacing the dropdown version.
"""
import dash
from dash import callback, Input, Output, State, ALL
from dash.exceptions import PreventUpdate
from UI.functions import global_vars
import dash_bootstrap_components as dbc
from dash import html, dcc
import pandas as pd

# Import the timestamped message creation and sorting functions for proper chat ordering
from UI.callback.chat_callbacks import create_timestamped_message, sort_chat_messages
from UI.pages.components.attribute_card_selector import attribute_card_selector
import threading
import time

def check_button_exists_in_chat(chat_content, button_id):
    """
    Check if a button with specific ID already exists in chat content.
    
    Args:
        chat_content: List of chat messages
        button_id: ID of button to search for
        
    Returns:
        bool: True if button exists, False otherwise
    """
    if not chat_content:
        return False
    
    for message in chat_content:
        if hasattr(message, 'children'):
            # Handle different message structures
            children = message.children
            if isinstance(children, list):
                for child in children:
                    if hasattr(child, 'id') and child.id == button_id:
                        return True
                    # Also check nested children (for complex message structures)
                    if hasattr(child, 'children') and isinstance(child.children, list):
                        for nested_child in child.children:
                            if hasattr(nested_child, 'id') and nested_child.id == button_id:
                                return True
    return False

def calculate_metrics_background(selected_attribute):
    """
    Calculate metrics in the background without blocking the UI.
    
    Args:
        selected_attribute (str): The selected target attribute
    """
    try:
        print(f"[BACKGROUND METRICS] Starting background calculation for attribute: {selected_attribute}")
        
        # Check if we have both datasets to calculate metrics
        if (hasattr(global_vars, 'df') and global_vars.df is not None and 
            hasattr(global_vars, 'secondary_df') and global_vars.secondary_df is not None):
            
            print(f"[BACKGROUND METRICS] Prerequisites met - starting calculation")
            print(f"[BACKGROUND METRICS] Primary dataset shape: {global_vars.df.shape}")
            print(f"[BACKGROUND METRICS] Secondary dataset shape: {global_vars.secondary_df.shape}")
            
            # Import and calculate metrics
            from drift.detect import generate_metrics_data
            
            # Set the calculation status in global vars
            global_vars.metrics_calculation_status = "calculating"
            
            metrics_data, data_length = generate_metrics_data()
            
            print(f"[BACKGROUND METRICS] Calculation completed. Result: {len(metrics_data) if metrics_data else 0} columns")
            
            if metrics_data:
                # Cache the calculated metrics
                cache_success = global_vars.cache_metrics(metrics_data, data_length, force=True)
                print(f"[BACKGROUND METRICS] Cache operation success: {cache_success}")
                
                # Reset dataset change flags since metrics are now current
                if cache_success:
                    global_vars.reset_change_flags()
                    print("[BACKGROUND METRICS] Dataset change flags reset after successful caching")
                    global_vars.metrics_calculation_status = "completed"
                else:
                    global_vars.metrics_calculation_status = "failed"
                
                # Verify cache was stored
                is_valid, reason = global_vars.is_cache_valid()
                print(f"[BACKGROUND METRICS] Cache validation: {is_valid}, reason: {reason}")
                
                print(f"[BACKGROUND METRICS] Successfully cached metrics for {len(metrics_data)} columns")
            else:
                print("[BACKGROUND METRICS] Metrics calculation returned empty data")
                global_vars.metrics_calculation_status = "failed"
        else:
            print("[BACKGROUND METRICS] Prerequisites not met - missing datasets")
            global_vars.metrics_calculation_status = "failed"
            
    except Exception as e:
        import traceback
        print(f"[BACKGROUND METRICS] Error calculating metrics: {str(e)}")
        print(f"[BACKGROUND METRICS] Traceback: {traceback.format_exc()}")
        global_vars.metrics_calculation_status = "failed"

@callback(
    Output("attribute-card-container", "children"),
    Input("target-attribute-modal", "is_open")
)
def update_attribute_cards(is_open):
    """
    Update the attribute cards when the modal is opened
    
    This ensures the cards always show the latest columns from the primary dataset
    even if the datasets were loaded after the application started.
    """
    if not is_open:
        raise PreventUpdate
    
    # Return the card selector component with current dataset information
    return [attribute_card_selector()]

# New callback to handle card clicks
@callback(
    [Output({"type": "attribute-card", "index": ALL}, "className"),
     Output("selected-attribute-store", "data")],
    [Input({"type": "attribute-card", "index": ALL}, "n_clicks")],
    [State({"type": "attribute-card", "index": ALL}, "id"),
     State("selected-attribute-store", "data")]
)
def handle_card_selection(n_clicks, ids, current_selection):
    """
    Handle selection of attribute cards
    
    When a card is clicked, update its appearance and store the selected attribute
    """
    if not n_clicks or not any(n_clicks):
        raise PreventUpdate
        
    # Find which card was clicked (the one with the highest n_clicks)
    triggered_idx = None
    max_clicks = -1
    for i, clicks in enumerate(n_clicks):
        if clicks and clicks > max_clicks:
            max_clicks = clicks
            triggered_idx = i
    
    if triggered_idx is None:
        raise PreventUpdate
        
    # Get the column name from the clicked card's id
    selected_column = ids[triggered_idx]["index"]
    
    # Update all card classes
    card_classes = []
    for i, card_id in enumerate(ids):
        column = card_id["index"]
        # Check if column exists in secondary dataset
        in_secondary = True
        if hasattr(global_vars, 'df') and hasattr(global_vars, 'secondary_df'):
            in_secondary = column in global_vars.secondary_df.columns
            
        base_class = "attribute-card"
        if not in_secondary:
            base_class += " attribute-card-disabled"
            
        # Add selected class if this is the selected card
        if column == selected_column:
            base_class += " attribute-card-selected"
            
        card_classes.append(base_class)
    
    return card_classes, selected_column

# Update the existing callback to use the selected-attribute-store instead of dropdown
@callback(
    [Output("target-attribute-modal", "is_open"),
     Output("query-area", "children", allow_duplicate=True)],
    [Input("confirm-target-attribute", "n_clicks")],
    [State("selected-attribute-store", "data"),
     State("query-area", "children")],
    prevent_initial_call=True
)
def handle_target_attribute_selection(confirm_clicks, selected_attribute, chat_content):
    """
    Handle the target attribute selection process.
    
    Features:
    - Validate user selection
    - Store the selected attribute in global variables
    - Add confirmation message to the chat area
    - Immediately calculate and cache metrics for detect stage
    
    Args:
        confirm_clicks: Number of times the confirm button has been clicked
        selected_attribute: The attribute selected by the user
        chat_content: Current chat content
        
    Returns:
        tuple: (whether the modal is open, updated chat content)
    """
    # Add proper trigger detection to prevent ghost re-execution
    ctx = dash.callback_context
    if not ctx.triggered or not ctx.triggered[0]['prop_id'].split('.')[0] == "confirm-target-attribute":
        raise PreventUpdate
        
    if not confirm_clicks:
        raise PreventUpdate
        
    if selected_attribute is None:
        # No attribute selected, keep modal open and show warning
        sorted_chat_content = sort_chat_messages(chat_content)
        return True, sorted_chat_content
    
    # Store the selected attribute in global variables
    # Reset button state if selecting a different attribute
    if hasattr(global_vars, 'target_attribute') and global_vars.target_attribute != selected_attribute:
        global_vars.target_attribute_button_added = False
        
    global_vars.target_attribute = selected_attribute
    
    # Initialize chat content (if None)
    if chat_content is None:
        chat_content = []
        
    # Get column type (if available)
    column_type = "Unknown"
    if hasattr(global_vars, 'column_types') and global_vars.column_types and selected_attribute in global_vars.column_types:
        column_type = global_vars.column_types[selected_attribute]
        
    # Add confirmation message to chat area with timestamp for proper ordering
    # TODO: need to check if this message works
    confirmation_message = create_timestamped_message(
        dcc.Markdown(
                f"✅ **Target attribute set to:** `{selected_attribute}`\n\n"
                f"📊 Distribution shift metrics will be calculated when you click the **Detect** button.",
                className="llm-msg"
            )
    )
    chat_content.append(confirmation_message)

    # Add prompt to check attribute types with a button - using timestamped message for proper ordering
    # Only add button if it doesn't already exist (prevent duplicates from ghost triggers)
    if (not hasattr(global_vars, 'target_attribute_button_added') or not global_vars.target_attribute_button_added) and \
       not check_button_exists_in_chat(chat_content, "show-type-compare-btn"):
        
        button_message = create_timestamped_message([
            dcc.Markdown(
                "You can check the column types across both datasets. Tap the button:",
            ),
            html.Button(
                "Check attribute types",
                id="show-type-compare-btn",
                n_clicks=0,
                className="gradient-primary-button",
                style={"marginTop": "6px"}
            )
        ], "llm-msg")
        chat_content.append(button_message)
        
        # Mark button as added to prevent future duplicates
        global_vars.target_attribute_button_added = True
    
    # =============================================================================
    # ON-DEMAND METRICS CALCULATION
    # =============================================================================
    # Rationale: Metrics are now calculated only when user explicitly clicks the
    # Detect button, not automatically when target attribute is selected.
    # This prevents resource contention and improves UI responsiveness.
    # =============================================================================
    
    # Initialize metrics calculation status
    if not hasattr(global_vars, 'metrics_calculation_status'):
        global_vars.metrics_calculation_status = "idle"
    
    # Check prerequisites and set status
    has_primary = hasattr(global_vars, 'df') and global_vars.df is not None
    has_secondary = hasattr(global_vars, 'secondary_df') and global_vars.secondary_df is not None
    
    print(f"[TARGET_ATTR] Target attribute selected: {selected_attribute}")
    print(f"[TARGET_ATTR] Has datasets - Primary: {has_primary}, Secondary: {has_secondary}")
    
    if has_primary and has_secondary:
        # Mark that metrics need calculation (will be done when Detect button is clicked)
        global_vars.metrics_calculation_status = "pending"
        print(f"[TARGET_ATTR] Metrics status set to 'pending' - will calculate on Detect button click")
        
    
    else:
        # Missing datasets - inform user
        missing = []
        if not has_primary:
            missing.append("primary dataset")
        if not has_secondary:
            missing.append("secondary dataset")
        
        chat_content.append(
            dcc.Markdown(
                f"⚠️ **Target attribute set to:** `{selected_attribute}`\n\n"
                f"📊 Please upload the {' and '.join(missing)} to enable metrics calculation.",
                className="llm-msg"
            )
        )
    
    # Close the modal and return updated chat content with proper ordering
    sorted_chat_content = sort_chat_messages(chat_content)
    return False, sorted_chat_content

# Update the warning callback
@callback(
    Output("target-attribute-warning", "is_open"),
    [Input("confirm-target-attribute", "n_clicks")],
    [State("selected-attribute-store", "data")],
    prevent_initial_call=True
)
def show_target_attribute_warning(confirm_clicks, selected_attribute):
    """
    Show warning when user tries to confirm without selecting an attribute.
    
    Args:
        confirm_clicks: Number of times the confirm button has been clicked
        selected_attribute: The attribute selected by the user
        
    Returns:
        bool: Whether to show the warning
    """
    if confirm_clicks and selected_attribute is None:
        return True
    
    return False

# Keep this callback for compatibility with existing code that might use it
@callback(
    Output("target-attribute-type-display", "children"),
    Input("selected-attribute-store", "data")
)
def display_target_attribute_type(selected_attribute):
    """
    Display the data type and sample values of the selected attribute.
    This is kept for compatibility but is now hidden in the UI.
    
    Args:
        selected_attribute: The attribute selected by the user
        
    Returns:
        html.Div: Component displaying attribute type and sample values
    """
    if selected_attribute is None:
        return ""
    
    # Get type information (if available)
    column_type = "Unknown"
    if hasattr(global_vars, 'column_types') and global_vars.column_types and selected_attribute in global_vars.column_types:
        column_type = global_vars.column_types[selected_attribute]
    
    # Get a few sample values
    sample_values = []
    if hasattr(global_vars, 'df') and global_vars.df is not None and selected_attribute in global_vars.df.columns:
        sample_values = global_vars.df[selected_attribute].dropna().head(3).tolist()
        sample_text = ", ".join([str(val) for val in sample_values])
        if len(sample_values) > 0:
            sample_display = html.Div([
                html.Strong("Sample values: "),
                html.Span(sample_text)
            ])
        else:
            sample_display = ""
    else:
        sample_display = ""
    
    return html.Div([
        html.Div([
            html.Strong("Type: "),
            html.Span(column_type)
        ]),
        sample_display
    ])

# New callback to provide status updates on background metrics calculation
@callback(
    Output("query-area", "children", allow_duplicate=True),
    Input("metrics-status-interval", "n_intervals"),
    State("query-area", "children"),
    prevent_initial_call=True
)
def update_metrics_status(n_intervals, chat_content):
    """
    Update the chat area with metrics calculation status.
    
    This callback runs periodically to check if background metrics calculation
    has completed and update the user interface accordingly.
    
    Args:
        n_intervals: Number of intervals passed (used for triggering)
        chat_content: Current chat content
    """
    # n_intervals triggers this callback but we don't need its value
    if not hasattr(global_vars, 'metrics_calculation_status'):
        raise PreventUpdate
    
    status = global_vars.metrics_calculation_status
    
    # Only update if status has changed to completed or failed
    if status == "completed":
        # Reset status to avoid repeated notifications
        global_vars.metrics_calculation_status = "idle"
        
        if chat_content is None:
            chat_content = []
        
        return chat_content
    
    elif status == "failed":
        # Reset status to avoid repeated notifications
        global_vars.metrics_calculation_status = "idle"
        
        if chat_content is None:
            chat_content = []
        
        chat_content.append(
            dcc.Markdown(
                f"⚠️ **Background metrics calculation failed.** Metrics will be calculated when you enter the detect stage.",
                className="llm-msg"
            )
        )
        
        return chat_content
    
    # No status change, don't update
    raise PreventUpdate
