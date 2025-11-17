"""
Attribute Card Selector Component

This component provides a card-based interface for selecting target attributes,
replacing the traditional dropdown approach with a more intuitive visual selection.
Each attribute is displayed as a card showing key information about the field.
"""
import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from UI.functions import global_vars
import pandas as pd

def attribute_card_selector(primary_df=None, secondary_df=None, column_types=None):
    """
    Create a card-based attribute selector component.
    
    Args:
        primary_df: The primary dataset DataFrame
        secondary_df: The secondary dataset DataFrame
        column_types: Dictionary mapping column names to their detected types
        
    Returns:
        html.Div: Card selector component with all columns from the primary dataset
    """
    cards = []
    selected_class = "attribute-card-selected"
    
    # If no dataframes provided, try to get from global vars
    if primary_df is None and hasattr(global_vars, 'df'):
        primary_df = global_vars.df
    
    if secondary_df is None and hasattr(global_vars, 'secondary_df'):
        secondary_df = global_vars.secondary_df
        
    if column_types is None and hasattr(global_vars, 'column_types'):
        column_types = global_vars.column_types
    
    # Check if we have valid data to work with
    if primary_df is None or not isinstance(primary_df, pd.DataFrame):
        return html.Div("No data available")
    
    # Get columns from primary dataset
    primary_columns = list(primary_df.columns)
    
    # Sort columns by type (Binary/Categorical first, then Continuous, then others)
    def get_type_priority(col):
        if column_types and col in column_types:
            col_type = column_types[col]
            if col_type == "Binary":
                return 0
            elif col_type == "Categorical":
                return 1
            elif col_type == "Continuous":
                return 2
            else:
                return 3
        return 4  # Unknown type gets lowest priority
    
    # Sort columns by type priority
    primary_columns.sort(key=get_type_priority)
    
    # Create a card for each column
    for col in primary_columns:
        # Get column type
        col_type = "Unknown"
        if column_types and col in column_types:
            col_type = column_types[col]
        
        # Calculate non-null percentages
        primary_non_null = 100 * (1 - primary_df[col].isna().mean())
        primary_non_null_text = f"{primary_non_null:.0f}% non-null"
        
        # Check if column exists in secondary dataset
        in_secondary = secondary_df is not None and col in secondary_df.columns
        
        # Secondary dataset non-null percentage (if applicable)
        secondary_non_null_text = ""
        if in_secondary:
            secondary_non_null = 100 * (1 - secondary_df[col].isna().mean())
            secondary_non_null_text = f"{secondary_non_null:.0f}% non-null"
        
        # Determine values display
        values_display = ""
        if col_type == "Binary":
            # For binary fields, check if they're actually 0/1
            if primary_df[col].dropna().isin([0, 1]).all():
                values_display = html.Div([
                    html.Span("✓ ", className="attribute-card-check"),
                    "Values 0,1"
                ])
            else:
                # Get unique values for binary that aren't 0/1
                unique_vals = primary_df[col].dropna().unique()
                if len(unique_vals) <= 3:  # Show only if few unique values
                    values_display = html.Div([
                        html.Span("✓ ", className="attribute-card-check"),
                        f"Values: {', '.join(map(str, unique_vals[:3]))}"
                    ])
        
        # Check if present in both datasets
        presence_display = ""
        if in_secondary:
            presence_display = html.Div([
                html.Span("✓ ", className="attribute-card-check"),
                "Present in both datasets"
            ])
        else:
            presence_display = html.Div([
                html.Span("✕ ", className="attribute-card-x"),
                "Present only in primary dataset"
            ], className="attribute-card-primary-only")
        
        # Create the card
        card = html.Div([
            # Card header with column name and type
            html.Div([
                html.Div(col, className="attribute-card-name"),
                html.Div(col_type, className="attribute-card-type")
            ], className="attribute-card-header"),
            
            # Card body with stats
            html.Div([
                # Non-null percentages
                html.Div([
                    html.Div(primary_non_null_text, className="attribute-card-primary-stats"),
                    html.Div(secondary_non_null_text, className="attribute-card-secondary-stats")
                ], className="attribute-card-stats-row"),
                
                # Values display (if applicable)
                values_display,
                
                # Presence indicator
                presence_display
            ], className="attribute-card-body")
        ], 
        id={"type": "attribute-card", "index": col},
        className="attribute-card" + (" attribute-card-disabled" if not in_secondary else ""),
        n_clicks=0)
        
        cards.append(card)
    
    # Create the card grid container
    card_container = html.Div(cards, className="attribute-cards-container")
    
    # Add the store for keeping track of selected attribute
    selector_with_store = html.Div([
        card_container,
        dcc.Store(id="selected-attribute-store", data=None)
    ])
    
    return selector_with_store


# CSS styles are in /assets/attribute_cards.css
