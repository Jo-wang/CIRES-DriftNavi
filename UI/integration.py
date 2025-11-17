"""
driftNavi UI Integration Module

This module is responsible for integrating various UI components into the application's main layout,
ensuring all components are properly loaded and can interact with each other.
"""
from dash import html
from UI.app import app
from UI.pages.components.target_attribute_modal import target_attribute_modal

def integrate_components():
    """
    Integrate all UI components into the application's main layout
    
    This function checks the application's current layout and adds each component to the appropriate position.
    It ensures components are only added once to avoid duplication.
    """
    # Ensure the target attribute selection modal is added to the layout
    integrate_target_attribute_modal()
    
    # Additional component integration code can be added here

def integrate_target_attribute_modal():
    """
    Integrate the target attribute selection modal into the application's main layout
    
    This function checks the current layout, and if the target attribute selection modal
    has not been added yet, adds it to the top level of the layout.
    """
    if not hasattr(app, '_target_attribute_modal_added'):
        # Mark the modal as added to prevent duplicate additions
        app._target_attribute_modal_added = True
        
        # Get the current layout
        current_layout = app.layout
        
        # If current layout is a function, we need special handling
        if callable(current_layout):
            original_layout = current_layout
            
            # Create a new layout function that wraps the original layout and adds the target attribute selection modal
            def new_layout(*args, **kwargs):
                layout_content = original_layout(*args, **kwargs)
                
                # Ensure layout_content is a Div
                if not isinstance(layout_content, html.Div):
                    layout_content = html.Div([layout_content])
                
                # Check if modal is already in the layout
                if not any(getattr(child, 'id', None) == 'target-attribute-modal' for child in layout_content.children):
                    # Add target attribute selection modal to the layout
                    layout_content.children.append(target_attribute_modal())
                
                return layout_content
            
            # Replace the application layout
            app.layout = new_layout
        else:
            # If current layout is a component, add directly
            if isinstance(current_layout, html.Div):
                if not any(getattr(child, 'id', None) == 'target-attribute-modal' for child in current_layout.children):
                    current_layout.children.append(target_attribute_modal())
            else:
                # If not a Div, create a new wrapper Div
                app.layout = html.Div([current_layout, target_attribute_modal()])
