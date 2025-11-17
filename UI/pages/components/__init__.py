"""
UI component initialization file
Ensure all components are correctly imported and registered
"""
# import all components
from UI.shared.components.survey_modal import survey_modal
try:
    from UI.pages.components.target_attribute_modal import target_attribute_modal
except ImportError:
    # if target_attribute_modal is not available, provide a fallback implementation
    def target_attribute_modal():
        """Fallback implementation if the real component is not available"""
        from dash import html
        return html.Div(id="target-attribute-modal")
