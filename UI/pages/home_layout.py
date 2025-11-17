import dash
from dash import callback, dcc, html, Input, Output, State, ClientsideFunction, no_update, register_page, clientside_callback, MATCH, ALL, dash_table, callback_context
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from dash_bootstrap_components import Container, Row, Col, Card, CardBody, CardHeader
import dash_daq as daq
import dash_editor_components
import dash_editor_components.PythonEditor
from flask_login import logout_user, current_user

# Data handling imports
import pandas as pd
import numpy as np
import time
import datetime
import json
import re
import os
import ast
import plotly.graph_objs as go
import plotly.express as px
import math

# Application specific imports
from db_models.conversation import Conversation
from utils.data_processor import identify_categorical_encoding_columns, create_encoding_notification_text
from agent import ConversationFormat
from constant_prompt import DEFAULT_NEXT_QUESTION_PROMPT, DEFAULT_SYSTEM_PROMPT, DEFAULT_PREFIX_PROMPT, \
    DEFAULT_PERSONA_PROMPT
from UI.shared.components.survey_modal import survey_modal
from UI.functions.global_vars import global_vars
from UI.constants import PIPELINE_STAGES
from UI.pages.home_ids import HomeIds as IDS
# Import functions from UI.functions package
from UI.functions import format_reply_to_markdown, parse_code_blocks

# Import state connector for Detect-Explain integration
from UI.state_connector import save_detect_results, get_explain_context

# Import chat layout component
from UI.pages.chat_layout import layout as chat_layout

        # Import target attribute modal component
from UI.pages.components.target_attribute_modal import target_attribute_modal

# Import context version confirmation modal
from UI.pages.components.context_version_modal import context_version_modal

# Import target attribute indicator component - temporarily commented out
# from UI.pages.components.target_attribute_indicator import create_target_attribute_indicator

# Import metrics heatmap component
from UI.shared.components.metrics_heatmap import create_metrics_heatmap
from UI.components.unified_column_type_modal import create_unified_column_type_modal

# Import explain component - DEPRECATED: Now using chat-based explain workflow
# from UI.pages.components.explain_component import create_explain_component

# Import shared distribution analysis utilities
from UI.shared.components.distribution_analysis_utils import (
    analyze_target_distribution, 
    generate_distribution_chart, 
    create_distribution_chart_component,
    get_distribution_data
)

# Import detect stage distribution callbacks
from UI.pages.components.detect_distribution_callbacks import update_target_distribution_chart

# Import functions for conditional analysis - UPDATED FOR NEW CONTEXT-BASED ARCHITECTURE
# OLD imports commented out as functions were renamed/removed in new architecture:
# from UI.pages.components.explain_component import update_conditional_distribution_chart, update_shifted_attribute_options
from UI.pages.components.explain_utils import rank_attributes, analyze_conditional_distribution, get_target_values_options

# Third-party libraries
import markdown
from drift.detect import generate_metrics_data
# Add Store for global chat context and notification system
register_page(__name__, path='/home/', title='Home')

# Import new utility modules
from UI.utils.context_utils import (
    generate_context_id, validate_context_list, create_context_item,
    add_context_item_to_list, filter_buttons_from_component,
    create_button_feedback_content, create_original_button_content
)
from UI.utils.button_utils import create_dual_add_buttons, create_dual_modal_buttons

# ===== Recursive data extraction function =====
def extract_all_text_recursively(component, depth=0):
    """Recursive data extraction function"""
    if depth > 15:  # prevent infinite recursion
        return ""
    
    result = ""
    try:
        if isinstance(component, str):
            return component + " "
        elif isinstance(component, (int, float)):
            return str(component) + " "
        elif isinstance(component, list):
            for item in component:
                result += extract_all_text_recursively(item, depth + 1)
        elif isinstance(component, dict):
            # process Dash component structure
            if 'props' in component:
                props = component['props']
                
                # extract all possible text attributes
                text_props = ['children', 'value', 'title', 'label', 'placeholder', 'content']
                for prop in text_props:
                    if prop in props:
                        result += extract_all_text_recursively(props[prop], depth + 1)
                
                # special processing of table data
                if 'data' in props and isinstance(props['data'], list):
                    result += "TABLE_DATA: "
                    for row in props['data']:
                        if isinstance(row, dict):
                            for key, value in row.items():
                                result += f"{key}={value} | "
                            result += "\n"
                
                # process table column information
                if 'columns' in props and isinstance(props['columns'], list):
                    for col in props['columns']:
                        if isinstance(col, dict) and 'name' in col:
                            result += f"COLUMN={col['name']} "
            
            # process normal dictionary
            else:
                for key, value in component.items():
                    if isinstance(value, (str, int, float)):
                        result += f"{key}={value} "
                    else:
                        result += extract_all_text_recursively(value, depth + 1)
    except Exception as e:
        print(f"[EXTRACTION ERROR] Depth {depth}: {str(e)}")
    
    return result

# Use shared pipeline stages constant
# pipeline_stages = PIPELINE_STAGES  # Commented out - using constant directly

# Add toast notification to confirm when content has been added to chat
toast_notification = dbc.Toast(
    id="toast-notification",
    icon="primary",
    header="Content added to chat",
    dismissable=True,
    duration=2000,
    style={"position": "fixed", "top": 10, "right": 10, "width": 350},
)

def layout(**kwargs):
    if not current_user.is_authenticated:
        return html.Div([
            dcc.Location(id="redirect-to-login",
                         refresh=True, pathname="/"),
        ])
    return html.Div([
        # Home Layout
        dbc.Container(fluid=True, children=[
            # For user wizard
            dbc.Modal(
                [
                    dbc.ModalHeader(dbc.ModalTitle(id="wizard-title")),
                    dbc.ModalBody(id="wizard-body", style={"fontSize": "0.7vw"}),
                    dbc.ModalFooter(
                        dbc.Button("Next", id="next-step", className="ms-auto", n_clicks=0)
                    ),
                ],
                id="wizard-modal",
                is_open=False,
                backdrop=False,  # Allow interaction with the underlying page
                style={"position": "fixed !important", "z-index": "1500", "color": "black"},
                # Float above other elements
            ),
            dcc.Store(id=IDS.BASE_STYLES if hasattr(IDS, 'BASE_STYLES') else "base-styles", data={}),
            html.Div(id=IDS.OVERLAY if hasattr(IDS, 'OVERLAY') else "overlay",
                     style={"position": "fixed", "top": "0", "left": "0", "width": "100%", "height": "100%",
                            "backgroundColor": "rgba(0, 0, 0, 0.7)", "z-index": "100", "display": "none"}),
            dcc.Store(id=IDS.IMPORT_DATASET_TARGET if hasattr(IDS, 'IMPORT_DATASET_TARGET') else "import-dataset-target", data="primary"),  # Store to track which dataset to replace
            dcc.Store(id=IDS.INITIAL_UPLOAD if hasattr(IDS, 'INITIAL_UPLOAD') else "initial-upload", data=True),  # Store to track if this is initial upload or replacement
            dcc.Store(id=IDS.COLUMN_TYPES_STORE if hasattr(IDS, 'COLUMN_TYPES_STORE') else "column-types-store", data={}),  # Store to track column data types
            
            # Unified Global Data State Store - Single source of truth for all data changes
            dcc.Store(id="global-data-state", data={}, storage_type='memory'),
            
            # Trigger stores for the unified data state
            dcc.Store(id="column-type-change-trigger", data={}, storage_type='memory'),
            dcc.Store(id="dataset-upload-trigger", data={}, storage_type='memory'),
            
            # Phase 3: Master Table Controller stores
            dcc.Store(id="table-update-trigger", data={}, storage_type='memory'),
            dcc.Store(id="table-overview-update-status", data={}, storage_type='memory'),
            dbc.Modal(
                [
                    dbc.ModalHeader(dbc.ModalTitle(id="upload-modal-title", style={"color": "#614385"})),
                    dbc.ModalBody(id="upload-modal-body", children=[
                        html.Div(id="upload-single-dataset", children=[
                            # Single dataset upload UI (for replacement through Data Manager)
                            html.Div([
                                html.H5(id="upload-modal-subtitle", style={"color": "#614385", "marginBottom": "10px"}),
                                dcc.Loading(
                                    html.Div(
                                        dcc.Upload(
                                            id='upload-data-modal',
                                            children=html.Div(['Drag and Drop or ', html.A('Select File')]),
                                            style={
                                                'width': '100%',
                                                'height': '120px',
                                                'lineHeight': '120px',
                                                'borderWidth': '1px',
                                                'borderStyle': 'dashed',
                                                'borderRadius': '5px',
                                                'textAlign': 'center',
                                                'margin': '10px',
                                                'backgroundColor': '#f5f5f5',
                                                'display': 'flex',
                                                'justifyContent': 'center',
                                                'alignItems': 'center'
                                            },
                                            multiple=True
                                        ),
                                        style={
                                            'display': 'flex',
                                            'justifyContent': 'center',
                                            'alignItems': 'center'
                                        }
                                    )
                                ),
                                html.Div(id="upload-data-error-msg", style={"color": "red", "marginBottom": "20px"}),
                            ]),
                        ], style={"display": "none"}),
                        
                        html.Div(id="upload-dual-datasets", children=[
                            # Initial dual dataset upload UI
                            html.Div([
                                html.H5("Primary Dataset", style={"color": "#614385", "marginBottom": "10px"}),
                                dcc.Loading(
                                    html.Div(
                                        dcc.Upload(
                                            id='upload-data-modal-primary',
                                            children=html.Div(['Drag and Drop or ', html.A('Select Primary File')]),
                                            style={
                                                'width': '100%',
                                                'height': '120px',
                                                'lineHeight': '120px',
                                                'borderWidth': '1px',
                                                'borderStyle': 'dashed',
                                                'borderRadius': '5px',
                                                'textAlign': 'center',
                                                'margin': '10px',
                                                'backgroundColor': '#f5f5f5',
                                                'display': 'flex',
                                                'justifyContent': 'center',
                                                'alignItems': 'center'
                                            },
                                            multiple=True
                                        ),
                                        style={
                                            'display': 'flex',
                                            'justifyContent': 'center',
                                            'alignItems': 'center'
                                        }
                                    )
                                ),
                                html.Div(id="upload-data-error-msg-primary", style={"color": "red", "marginBottom": "20px"}),
                            ]),
                            html.Hr(style={"margin": "20px 0"}),
                            html.Div([
                                html.H5("Secondary Dataset", style={"color": "#516395", "marginBottom": "10px"}),
                                dcc.Loading(
                                    html.Div(
                                        dcc.Upload(
                                            id='upload-data-modal-secondary',
                                            children=html.Div(['Drag and Drop or ', html.A('Select Secondary File')]),
                                            style={
                                                'width': '100%',
                                                'height': '120px',
                                                'lineHeight': '120px',
                                                'borderWidth': '1px',
                                                'borderStyle': 'dashed',
                                                'borderRadius': '5px',
                                                'textAlign': 'center',
                                                'margin': '10px',
                                                'backgroundColor': '#f5f5f5',
                                                'display': 'flex',
                                                'justifyContent': 'center',
                                                'alignItems': 'center'
                                            },
                                            multiple=True
                                        ),
                                        style={
                                            'display': 'flex',
                                            'justifyContent': 'center',
                                            'alignItems': 'center'
                                        }
                                    )
                                ),
                                html.Div(id="upload-data-error-msg-secondary", style={"color": "red"}),
                            ]),
                        ]),
                    ]),
                    dbc.ModalFooter([
                        dbc.Button("Close", id="close-upload-modal", className="ml-auto"),
                    ],
                        style={
                            'display': 'flex',
                            'justifyContent': 'end',
                            'alignItems': 'center'
                        }
                    ),
                ],
                id="upload-modal",
                is_open=True,
                centered=True,
                style={
                    "boxShadow": "0 2px 4px 0 rgba(0, 0, 0, 0.2);",
                }
            ),
            dbc.Modal(
                [
                    dbc.ModalBody(
                        children=html.Div(id="survey-modal-body"),
                    )
                ],
                id="survey-modal",
                is_open=False,
                centered=True,
                style={
                    "boxShadow": "0 2px 4px 0 rgba(0, 0, 0, 0.2);"
                },
                backdrop_class_name="backdrop-survey-modal",
                content_class_name="content-survey-modal"
            ),
            dbc.Modal(
                [
                    dbc.ModalHeader(dbc.ModalTitle("Dataset Statistics"), close_button=True),
                    dbc.ModalBody(children=[html.Div(id="data-stat-body"),
                                            html.Div(id="data-stat-summary", style={"marginTop": "20px"})
                                            ]),
                    dbc.ModalFooter(children=[
                        dbc.Button("Analyze", id={'type': 'spinner-btn', 'index': 1, 'pattern_id': 'evaluate_dataset_button_1'}, className="ml-auto"),
                        dbc.Button("Close", id="data-stat-close", className="ml-auto")]
                    ),
                ],
                id="data-stat-modal",
                is_open=False,
                centered=True,
                size="xl",
                style={
                    "boxShadow": "0 2px 4px 0 rgba(0, 0, 0, 0.2);",
                }
            ),
            dbc.Modal(
                [
                    dbc.ModalHeader("Exporting Options"),
                    dbc.ModalBody(
                        html.Div([
                            dcc.Dropdown(id='export-format-dropdown', options=[
                                v.value for v in ConversationFormat],
                                         value=ConversationFormat.SIMPLIFIED_JSON.value),
                        ], className="query-header"),
                    ),
                    dbc.ModalFooter([dbc.Button(
                        "Export", id="download-button", className="ml-auto"),
                        dcc.Download(id="export-conversation"), dbc.Button("Close", id="close", className="ml-auto")]),
                ],
                id="export-history-modal",
                centered=True,
                is_open=False,
            ),

            dbc.Modal(
                [
                    dbc.ModalHeader("Commonly Asked Questions"),
                    dbc.ModalBody(
                        dcc.Dropdown(
                            id="question-modal-list",
                            placeholder="Choose a question",
                        )
                    ),
                    dbc.ModalFooter(
                        [
                            dbc.Button("Choose", id="question-modal-choose-btn", className="me-2", n_clicks=0),
                            dbc.Button("Close", id="question-modal-close-btn", n_clicks=0),
                        ],
                        className="d-flex justify-content-end"
                    ),
                ],
                id="question-modal",
                is_open=False,  # Initially not open
                centered=True,
            ),

            # Metrics Explanation Modal
            dbc.Modal(
                [
                    dbc.ModalHeader(dbc.ModalTitle("Metrics Explanation"), close_button=True),
                    dbc.ModalBody(
                        html.Div(
                            id="metrics-modal-table-container",
                            style={"padding": "10px"}
                        )
                    ),
                    dbc.ModalFooter(
                        dbc.Button("Close", id="close-metrics-modal", className="ms-auto")
                    ),
                ],
                id="metrics-explanation-modal",
                size="lg",
                is_open=False,
                centered=True,
                backdrop="static",
            ),
            
            # Attribute Metrics Explanation Modal (for clicked attributes)
            dbc.Modal(
                [
                    dbc.ModalHeader(dbc.ModalTitle("Attribute Distribution Analysis"), close_button=True),
                    dbc.ModalBody(id="attribute-metric-explanation-modal-body"),
                    dbc.ModalFooter(
                        dbc.Button("Close", id="close-attribute-explanation-modal", className="ms-auto")
                    ),
                ],
                id="attribute-metric-explanation-modal",
                size="lg",
                is_open=False,
                centered=True,
                backdrop="static",
                style={"boxShadow": "0 2px 4px 0 rgba(0, 0, 0, 0.2);"}
            ),

            # Chat History Modal
            dbc.Modal(
                [
                    dbc.ModalHeader(dbc.ModalTitle("Chat History"), close_button=True),
                    dbc.ModalBody(children=[
                        html.Div(id="chat-history-modal-content")
                    ]),
                    dbc.ModalFooter(
                        dbc.Button("Close", id="chat-history-close-btn", className="ml-auto")
                    ),
                ],
                id="chat-history-modal",
                is_open=False,
                centered=True,
                size="lg",
                style={
                    "boxShadow": "0 2px 4px 0 rgba(0, 0, 0, 0.2);",
                }
            ),

            # Dataset Snapshots Modal
            dbc.Modal(
                [
                    dbc.ModalHeader(dbc.ModalTitle("Dataset Snapshots"), close_button=True),
                    dbc.ModalBody(children=[
                        html.Div([
                            html.Div([
                                html.Span(
                                    html.I(className="fas fa-question-circle"),
                                    id="tooltip-snapshot-modal",
                                    style={
                                        "fontSize": "20px",
                                        "color": "#aaa",
                                        "cursor": "pointer",
                                        "marginLeft": "5px",
                                        "alignSelf": "center"
                                    }
                                )
                            ], style={"display": "flex", "alignItems": "center", "justifyContent": "flex-end"}),
                            dbc.Tooltip(
                                "Click the Restore button to use the selected snapshot in the middle data view.",
                                target="tooltip-snapshot-modal",
                            ),
                            html.Div([
                                dash_table.DataTable(
                                    id="snapshot-table",
                                    row_selectable='single',
                                    columns=[
                                        {"name": "ID", "id": "ver"},
                                        {"name": "Description", "id": "desc"},
                                        {"name": "Timestamp", "id": "time"}
                                    ],
                                    data=[],
                                    style_table={'overflowX': 'auto'},
                                    style_cell={'textAlign': 'left'},
                                )
                            ]),
                            html.Div([
                                dbc.Button("Restore", id="restore-snapshot", n_clicks=0, className='primary-button'),
                            ], className='right-align-div', style={"marginTop": "15px"}),
                        ])
                    ]),
                    dbc.ModalFooter(
                        dbc.Button("Close", id="snapshots-close-btn", className="ml-auto")
                    ),
                ],
                id="snapshots-modal",
                is_open=False,
                centered=True,
                size="lg",
                style={
                    "boxShadow": "0 2px 4px 0 rgba(0, 0, 0, 0.2);",
                }
            ),

            # Dataset Evaluation Modal
            dbc.Modal(
                [
                    dbc.ModalHeader(dbc.ModalTitle("Dataset Evaluation"), close_button=True),
                    dbc.ModalBody(children=[
                        dbc.Tabs(
                            [
                                dbc.Tab(children=[
                                    html.Div([
                                        html.Div([
                                            'Snapshot:',
                                            dcc.Dropdown(
                                                id='dataset-selection',
                                                style={'width': '70%'},
                                                clearable=False
                                            ),
                                        ], style={'display': 'flex', 'alignItems': 'center',
                                                'justifyContent': 'flex-start', 'gap': '10px',
                                                'marginBottom': '20px'}),
                                        html.Div([
                                            'Target:',
                                            dcc.Dropdown(
                                                id='target-selection',
                                                style={'width': '70%'},
                                                clearable=True
                                            ),
                                        ], style={'display': 'flex', 'alignItems': 'center',
                                                'justifyContent': 'flex-start', 'gap': '10px',
                                                'marginBottom': '20px'}),
                                        html.Div(id='evaluation-result', style={'marginTop': '20px'}),
                                        html.Div([
                                            dbc.Button("Calculate", id="calculate-btn",
                                                    n_clicks=0, className='primary-button')
                                        ], className='right-align-div', style={'marginTop': '20px'})
                                    ], style={'padding': '10px'})
                                ], label="Basic"), 
                                dbc.Tab(id='fairness-metrics-tab', children=[
                                    html.Div(id='fairness-evaluation-result')
                                ], label="Fairness Metrics"),
                                dbc.Tab(id='custom-metrics-tab', children=[
                                    html.Div(id='custom-evaluation-result')
                                ], label="Custom Metrics")
                            ],
                            id="dataset-eval-tabs"
                        )
                    ]),
                    dbc.ModalFooter(
                        dbc.Button("Close", id="evaluation-close-btn", className="ml-auto")
                    ),
                ],
                id="evaluation-modal",
                is_open=False,
                centered=True,
                size="xl",
                style={
                    "boxShadow": "0 2px 4px 0 rgba(0, 0, 0, 0.2);",
                }
            ),

            # Python Sandbox Modal
            dbc.Modal(
                [
                    dbc.ModalHeader(dbc.ModalTitle("Python Sandbox"), close_button=True),
                    dbc.ModalBody(children=[
                        html.Div([
                            html.Div([
                                html.Span(
                                    html.I(className="fas fa-question-circle"),
                                    id="tooltip-code-modal",
                                    style={
                                        "fontSize": "20px",
                                        "color": "#aaa",
                                        "cursor": "pointer",
                                        "marginLeft": "5px",
                                        "alignSelf": "center"
                                    }
                                )
                            ], style={"display": "flex", "alignItems": "center", "justifyContent": "flex-end"}),
                            dbc.Tooltip(
                                "The variable df is a reference of the Pandas dataframe of the current dataset. "
                                "Any Modification on it will be reflected in the data view",
                                target="tooltip-code-modal",
                            ),
                            html.Div([dash_editor_components.PythonEditor(id='commands-input',
                                                                        style={'height': '500px'},
                                                                        value="",
                                                                        tabSize=4)],
                                    className='commands_editor'),
                            html.Div([dbc.Button("Run", id="run-commands", n_clicks=0, className='primary-button')],
                                    className='right-align-div', style={"marginTop": "15px"}),
                            html.Div(id='commands-output', style={'marginTop': '20px', 'whiteSpace': 'pre-wrap'})
                        ])
                    ]),
                    dbc.ModalFooter(
                        dbc.Button("Close", id="sandbox-close-btn", className="ml-auto")
                    ),
                ],
                id="sandbox-modal",
                is_open=False,
                centered=True,
                size="xl",
                style={
                    "boxShadow": "0 2px 4px 0 rgba(0, 0, 0, 0.2);",
                }
            ),

            # =======================================================
            # banner and menu bar layout
            dbc.Row(justify="center", align="center", children=[
                html.Div(children=[
                    html.Img(src='../assets/logo.svg', className="logo"),
                    html.P('DriftNavi', className="title"),
                    dbc.Nav(
                        className='navbar d-flex flex-wrap',
                        children=[
                            dbc.DropdownMenu(
                                [
                                    dbc.DropdownMenuItem("Import Primary Dataset", id="menu-import-primary"),
                                    dbc.DropdownMenuItem("Import Secondary Dataset", id="menu-import-secondary"),
                                    dbc.DropdownMenuItem("Column Type Comparison", id="menu-column-types"),
                                    dbc.DropdownMenuItem(divider=True),
                                    dbc.DropdownMenuItem("Export Dataset Analysis Report")
                                ],
                                label="Data Manager",
                                nav=True,
                                toggleClassName="dropdown-toggle",
                                className='menu-item',
                                id="menu-data-manager",
                            ),
                            # New User dropdown menu with consolidated user-related functions
                            dbc.DropdownMenu(
                                [
                                    dbc.DropdownMenuItem("User Profile", id="menu-profile"),
                                    dbc.DropdownMenuItem("Prompts", id="menu-prompt"),
                                    dbc.DropdownMenuItem("Export Chat History", id="menu-export-chat"),
                                    dbc.DropdownMenuItem("View Chat History", id="menu-view-chat-history"),
                                    dbc.DropdownMenuItem(divider=True),
                                    dbc.DropdownMenuItem("API Settings", id="menu-api-settings", header=True),
                                    dbc.DropdownMenuItem("GPT-4o-mini  ✔", id="menu-model-gpt4omini"),
                                    dbc.DropdownMenuItem("GPT-4o", id="menu-model-gpt4o")
                                ],
                                label="User",
                                nav=True,
                                toggleClassName="dropdown-toggle",
                                className='menu-item'
                            ),
                            # Model & Tools dropdown menu
                            dbc.DropdownMenu(
                                [
                                    dbc.DropdownMenuItem("Model Settings", id="menu-model-settings"),
                                    dbc.DropdownMenuItem(divider=True),
                                    dbc.DropdownMenuItem("Dataset Evaluation", id="menu-evaluation"),
                                    dbc.DropdownMenuItem("Python Sandbox", id="menu-sandbox")
                                ],
                                label="Model & Tools",
                                nav=True,
                                toggleClassName="dropdown-toggle",
                                className='menu-item',
                            ),
                            # drift Pipeline removed
                            dbc.DropdownMenu(
                                [
                                    dbc.DropdownMenuItem(
                                        "About CIRES", href="https://cires.org.au/"),
                                    dbc.DropdownMenuItem(
                                        "Logout", id="logout-button", href="/")
                                ],
                                label="More",
                                nav=True,
                                toggleClassName="dropdown-toggle",
                                className='menu-item'
                            )
                        ],
                    ),
                    dcc.Location(id=IDS.URL, refresh=False)
                ], className='banner'),
            ]),

            # Dataset Versions box under navbar
            # Removed global snapshot box; now anchored in preview area
            
            # Toast notification for context added feedback
            dbc.Toast(
                id="context-notification",
                header="Context Added",
                is_open=False,
                dismissable=True,
                duration=3000,
                icon="success",
                style={"position": "fixed", "top": 20, "right": 20, "zIndex": 1999}
            ),
            dcc.Store(id="notification-trigger", data={}),
            
            # Tooltip for operation recommendations removed
            
            dbc.Card(id="setting-container",
                     children=[
                         html.H4("Prompt for Eliciting Model's ability"),
                         dcc.Textarea(rows=7, id="system-prompt-input", className="mb-4 prompt-input p-2",
                                      value=current_user.system_prompt),
                         html.H4("Prompt for Handling Dataset"),
                         dcc.Textarea(rows=7, id="prefix-prompt-input", className="mb-4 prompt-input p-2",
                                      value=current_user.prefix_prompt),
                         html.H4("Prompt for Enhancing Personalization"),
                         dcc.Textarea(rows=8, id="persona-prompt-input", className="mb-4 prompt-input p-2",
                                      value=current_user.persona_prompt),
                         html.H4("Prompt for Generating Follow-up Questions"),
                         dcc.Textarea(rows=2, id="next-question-input-1", className="mb-4 prompt-input p-2",
                                      value=current_user.follow_up_questions_prompt_1),
                         # html.H4("Prompt for Generating Follow-up Questions 2"),
                         # dcc.Textarea(rows=2, id="next-question-input-2", className="mb-4 prompt-input p-2",
                         #              value=current_user.follow_up_questions_prompt_2),
                         html.Div(children=[
                             dbc.Button("Reset Default", id="reset-prompt-button", className="prompt-button",
                                        n_clicks=0),
                             dbc.Button("Save", id={'type': 'spinner-btn', 'index': 2, 'pattern_id': 'evaluate_dataset_button_2'}, className="prompt-button",
                                        n_clicks=0),
                             dbc.Button("Home", id="return-home-button", className="prompt-button", n_clicks=0),
                         ], className="save-button"),
                     ],
                     className="prompt-card p-4", style={"display": "none"}),

            # =======================================================
            # chatbox layout
            dbc.Row([
                # Middle column now with responsive width
                dbc.Col(xs=12, sm=12, md=7, lg=7, xl=7, id=IDS.MIDDLE_COLUMN, children=[
                    dbc.Card(body=True, id=IDS.DATA_VIEW, className='card', children=[
                        dcc.Loading(id=IDS.TABLE_LOADING, children=[
                            html.Div(children=[
                                # Table header - simplified, removed dataset info button
                                html.Div([
                                    # Empty div for layout consistency
                                    html.Div()
                                ], style={"display": "flex", "justifyContent": "flex-end", "alignItems": "center", "marginBottom": "15px", "paddingRight": "10px"}),
                                
                                # Create a sliding container to hold both dataset preview and metrics table
                                html.Div([
                                    # First slide: Dataset Preview
                                    html.Div([
                                        # Version box anchored above primary dataset
                                        html.Div([
                                            dbc.Card([
                                                dbc.CardHeader("Dataset Versions", className="py-2"),
                                                dbc.CardBody([
                                                    html.Div([
                                                        dcc.RadioItems(
                                                            id="snapshot-selector",
                                                            options=[],
                                                            value="original",
                                                            labelStyle={"marginRight": "12px"}
                                                        ),
                                                        dbc.Button(
                                                            "Save as new version",
                                                            id="snapshot-save-btn",
                                                            size="sm",
                                                            color="secondary",
                                                            className="ms-2"
                                                        ),
                                                        dbc.Button(
                                                            "Restore",
                                                            id="snapshot-restore-btn",
                                                            size="sm",
                                                            color="primary",
                                                            className="ms-2"
                                                        )
                                                    ], style={"display": "flex", "alignItems": "center", "flexWrap": "wrap"})
                                                ], className="py-2")
                                            ], style={"marginBottom": "10px", "width": "100%"})
                                        ]),
                                        html.Div([
                                            html.H5("Primary Dataset", style={"textAlign": "center", "color": "#614385", "margin": "10px 0"}),
                                             dash_table.DataTable(
                                                id=IDS.TABLE_PRIMARY, 
                                                page_size=8, 
                                                page_action='native',
                                                editable=False, 
                                                row_deletable=False, 
                                                column_selectable=False,
                                                # hide default delete column icon if any residual
                                                css=[{"selector": ".dash-table-container .row-deletion", "rule": "display: none;"}],
                                                style_cell={'textAlign': 'center', 'fontFamiliy': 'Arial',"padding":"0px 10px"},
                                                style_header={'backgroundColor': '#614385', 'color': 'white',
                                                            'fontWeight': 'bold'
                                                            },
                                                style_table={'overflowX': 'auto'},
                                                style_data_conditional=[
                                                    {
                                                        'if': {'row_index': 'odd'},
                                                        'backgroundColor': '#f2f2f2'
                                                    },
                                                    {
                                                        'if': {'row_index': 'even'},
                                                        'backgroundColor': 'white'
                                                    },
                                                ]
                                            )
                                        ], style={"marginBottom": "20px"}),
                                        
                                        html.Div([
                                            html.H5("Secondary Dataset", style={"textAlign": "center", "color": "#516395", "margin": "10px 0"}),
                                             dash_table.DataTable(
                                                id=IDS.TABLE_SECONDARY, 
                                                page_size=8, 
                                                page_action='native',
                                                editable=False, 
                                                row_deletable=False, 
                                                column_selectable=False,
                                                css=[{"selector": ".dash-table-container .row-deletion", "rule": "display: none;"}],
                                                style_cell={'textAlign': 'center', 'fontFamiliy': 'Arial',"padding":"0px 10px"},
                                                style_header={'backgroundColor': '#516395', 'color': 'white',
                                                            'fontWeight': 'bold'
                                                            },
                                                style_table={'overflowX': 'auto'},
                                                style_data_conditional=[
                                                    {
                                                        'if': {'row_index': 'odd'},
                                                        'backgroundColor': '#f2f2f2'
                                                    },
                                                    {
                                                        'if': {'row_index': 'even'},
                                                        'backgroundColor': 'white'
                                                    },
                                                ]
                                            )
                                        ]),
                                    ], id=IDS.DATASET_PREVIEW_CONTAINER, style={"width": "100%"}),
                                    
                                    # Second slide: Metrics Table
                                    html.Div([
                                        # html.P("The table below shows statistical metrics comparing the primary and secondary datasets for each attribute.", 
                                        #        className="mb-3", style={"textAlign": "center"}),
                                        # The metrics table will be rendered directly here with proper styling
                                         html.Div(id=IDS.METRICS_TABLE_CONTAINER, style={"width": "100%", "overflowX": "auto", "padding": "10px"})
                                    ], id=IDS.METRICS_TABLE_SLIDE, style={"width": "100%", "display": "none"}),
                                    
                                    # Third slide: Explain Component
                                    html.Div([
                                        # The explain component will be rendered here
                                         html.Div(id=IDS.EXPLAIN_COMPONENT_CONTAINER, style={"width": "100%", "padding": "5px"})
                                    ], id=IDS.EXPLAIN_COMPONENT_SLIDE, style={"width": "100%", "display": "none"}),
                                    
                                ], id="sliding-container", style={"width": "100%", "transition": "transform 0.5s ease-in-out"}),
                                
                                # Keep the original table-overview as a hidden element for backward compatibility
                                html.Div(
                                     dash_table.DataTable(id=IDS.TABLE_OVERVIEW, page_size=10, page_action='native',
                                                      editable=False, row_deletable=False, column_selectable=False,
                                                      style_cell={'textAlign': 'center', 'fontFamiliy': 'Arial',"padding":"0px 10px"},
                                                      style_header={'backgroundColor': '#614385', 'color': 'white',
                                                                    'fontWeight': 'bold'
                                                                    },
                                                      style_table={'overflowX': 'auto'},
                                                      style_data_conditional=[
                                                          {
                                                              'if': {'row_index': 'odd'},
                                                              'backgroundColor': '#f2f2f2'
                                                          },
                                                          {
                                                              'if': {'row_index': 'even'},
                                                              'backgroundColor': 'white'
                                                          },
                                                      ]
                                                      ),
                                    style={"display": "none"}  # Hide the original table
                                )
                            ],
                            style={"margin": "15px", "marginLeft": "0px"})
                        ],
                        overlay_style={
                            "visibility": "hidden", "opacity": .8, "backgroundColor": "white"},
                        target_components={"table-overview": ["data", "columns"]}
                        ),
                        dcc.Store("data-view-table-style",data=[
                                                         {
                                                             'if': {'row_index': 'odd'},
                                                             'backgroundColor': '#f2f2f2'
                                                         },
                                                         {
                                                             'if': {'row_index': 'even'},
                                                             'backgroundColor': 'white'
                                                         },
                                                     ]),
                        html.Div(id='datatable-interactivity-container', style={"display": "none"}, children=[
                            html.Div(id="distribution-charts-container", style={"display": "none"}, children=[
                                html.H4("Data Distribution Comparison", className="chart-title", style={"textAlign": "center", "marginTop": "20px"}),
                                html.Div(id="selected-cell-info", style={"textAlign": "center", "marginBottom": "10px"}),
                                dbc.Row([
                                    # Display distribution chart in full width since statistical comparison is commented out
                                    dbc.Col([
                                        dcc.Loading(
                                            id="combined-chart-loading",
                                            type="circle",
                                            children=html.Div(id="combined-distribution-chart")
                                        )
                                    ], width=12),  # Changed from width=6 to width=12 for full width
                                    # COMMENTED OUT: Statistical Comparison section
                                    # dbc.Col([
                                    #     html.H5("Statistical Comparison", className="chart-subtitle", style={"textAlign": "center"}),
                                    #     html.Div(id="distribution-comparison-summary", style={"padding": "10px", "backgroundColor": "#f8f9fa", "borderRadius": "5px", "height": "100%"})
                                    # ], width=6),
                                    # Keep the distribution-comparison-summary div hidden for callback compatibility
                                    html.Div(id="distribution-comparison-summary", style={"display": "none"})
                                ]),
                            ])
                        ]),
                        dbc.Alert(
                            "",
                            id="data-alert",
                            is_open=False,
                            dismissable=True,
                            color="primary",
                            duration=5000,
                        ),
                        
                        # Pipeline functionality moved to chat box - left side shows data preview only
                        html.Div([
                            
                            # Pipeline Stage Transition Alert - Smart workflow guidance system
                            # This component provides intelligent stage transition notifications when AI
                            # determines the user should move between Detect/Explain/Adapt phases
                            dbc.Alert(
                                [
                                    html.I(id="pipeline-alert-icon", className="fas fa-route me-2"),
                                    html.Span(id="pipeline-alert-text", children="Workflow stage updated")
                                ],
                                id="pipeline-alert",
                                color="primary",
                                dismissable=True,
                                is_open=False,  # Hidden by default, shown when AI suggests stage transitions
                                fade=True,      # Smooth fade-in/out animation
                                duration=8000,  # Auto-hide after 8 seconds
                                style={
                                    "width": "90%", 
                                    "margin": "0 auto 15px auto", 
                                    "text-align": "center",
                                    "border": "1px solid #0d6efd",
                                    "borderRadius": "8px",
                                    "boxShadow": "0 2px 8px rgba(13, 110, 253, 0.15)"
                                }
                            ),
                            
                            # Hidden ID for backwards compatibility with removed analyze metrics button
                            # html.Div(id="explain-metrics-btn", style={"display": "none"}),  # COMMENTED: Using chat-based explain workflow

                            
                            # Hidden triggering elements for backward compatibility
                            html.Div(id="tab-change-indicator", style={"display": "none"}),
                            html.Div(id="window-resize-indicator", style={"display": "none"}),
                            
                            # Stores for detect/explain mode state
                            dcc.Store(id="detect-mode", storage_type="memory", data=False),
                            # dcc.Store(id="explain-mode", storage_type="memory", data=False),  # COMMENTED: Using chat-based explain workflow
                            # Store for tracking detect button toggle state
                            dcc.Store(id="detect-button-active", storage_type="memory", data=False),
                        ], className="mt-3 mb-3"),
                    ]),

                    # Add modal for distribution visualization popup
                    dbc.Modal([
                        dbc.ModalHeader(dbc.ModalTitle("Data Distribution Comparison")),
                        dbc.ModalBody([
                            html.Div(id="modal-selected-cell-info", style={"textAlign": "center", "marginBottom": "15px"}),
                            dbc.Row([
                                # Distribution chart in full width since statistical comparison is commented out
                                dbc.Col([
                                    dcc.Loading(
                                        id="modal-combined-chart-loading",
                                        type="circle",
                                        children=html.Div(id="modal-combined-distribution-chart")
                                    )
                                ], width=12),  # Changed from width=6 to width=12 for full width
                                
                                # COMMENTED OUT: Statistical comparison on the right
                                # dbc.Col([
                                #     html.H5("Statistical Comparison", className="chart-subtitle", style={"textAlign": "center", "marginBottom": "10px"}),
                                #     html.Div(id="modal-distribution-comparison-summary", style={"padding": "15px", "backgroundColor": "#f8f9fa", "borderRadius": "5px", "height": "calc(100% - 40px)"})
                                # ], width=6),
                                
                                # Keep the modal-distribution-comparison-summary div hidden for callback compatibility
                                html.Div(id="modal-distribution-comparison-summary", style={"display": "none"})
                            ]),
                        ]),
                        dbc.ModalFooter([
                            create_dual_modal_buttons(
                                feature_name="distribution comparison",
                                chat_button_id="add-distribution-to-chat",
                                explain_button_id="add-distribution-to-explain"
                            ),
                            dbc.Button("Close", id="close-distribution-modal", className="ms-auto")
                        ])
                    ], id="distribution-modal", is_open=False, size="xl"),
                    
                    # Add modal for metrics explanation
                    dbc.Modal([
                        dbc.ModalHeader(dbc.ModalTitle("Statistical Metrics Explanation")),
                        dbc.ModalBody([
                            html.Div([
                                # html.P("The table below shows statistical metrics comparing the primary and secondary datasets for each attribute.", className="mb-3"),
                                html.Div(id="metrics-modal-table-container"),  # avoid ID conflict
                                html.Div(id="metric-explanation-container", style={"display": "none", "marginTop": "20px"}),
                                html.Div(id="attribute-metric-explanation-container", style={"display": "none", "marginTop": "20px"}),
                            ])
                        ]),
                        dbc.ModalFooter(
                            dbc.Button("Close", id="close-metrics-modal", className="ml-auto")
                        )
                    ], id="metrics-explanation-modal", is_open=False, size="xl"),
                ]),

                dbc.Col(xs=12, sm=12, md=5, lg=5, xl=5, id="right-column", children=[
                    # Load chat components from chat_layout.py
                    chat_layout()
                ]),
            ], id=IDS.HOME_CONTAINER, className="g-2"),  # Add gap and ensure responsive
        ], className="body fade-in"),
        dcc.Store(id='current-dataset-info', storage_type='memory'),
        dcc.Store(id='current-secondary-dataset-info', storage_type='memory'),
        dcc.Store(id='graph2-selection', storage_type='memory'),
        dcc.Store(id='highlight-store', storage_type='memory', data={}),
        
        # Auto-update system components
        dcc.Interval(
            id='metrics-monitor-interval',
            interval=2000,  # Check every 2 seconds
            n_intervals=0,
            disabled=False
        ),
        dcc.Interval(
            id='metrics-status-interval',
            interval=3000,  # Check every 3 seconds for background metrics status
            n_intervals=0,
            disabled=False
        ),
        dcc.Store(id='metrics-auto-update-trigger', storage_type='memory'),  
        
        # Register client-side callback for table highlight synchronization
        dash.clientside_callback(
            """
            window.dash_clientside.table_sync.syncTableSelections
            """,
            [Output('table-primary-overview', 'active_cell'),
             Output('table-secondary-overview', 'active_cell'),
             Output('table-overview', 'active_cell')],
            [Input('table-primary-overview', 'active_cell'),
             Input('table-secondary-overview', 'active_cell'),
             Input('table-overview', 'active_cell')]
        ),
        
        # Add a clientside callback to prevent JavaScript errors in the sandbox
        clientside_callback(
            """
            function preventSandboxErrors(n_clicks) {
                // This callback solely exists to prevent JavaScript reference errors
                // by ensuring dependent components are properly initialized
                return window.dash_clientside.no_update;
            }
            """,
            Output("sandbox-code-editor", "value", allow_duplicate=True),
            Input("sandbox-modal", "is_open"),
            prevent_initial_call=True
        ),
        

        
        dcc.Store(id=IDS.CURRENT_STAGE, storage_type='memory', data="detect"),
        dcc.Store(id='stage-change-trigger', storage_type='memory', data=0),
        dcc.Store(id='stage-sync-store', storage_type='memory'),
        dcc.Store(id='tab-change-indicator', storage_type='memory'),
        dcc.Store(id='window-resize-indicator', storage_type='memory'),
        dcc.Store(id='graph1-selection', storage_type='memory'),
        html.Div(id="import-primary-placeholder", **{"data-dummy": ""}, style={"display": "none"}),
        html.Div(id="import-secondary-placeholder", **{"data-dummy": ""}, style={"display": "none"}),
        dcc.Store(id='chat-update-trigger', data=0),
        dcc.Store(id="distribution-chat-data", data=None),
        dcc.Store(id="type-compare-rendered", data=False),
        
        # Import unified column type modal component (Phase 2)
        create_unified_column_type_modal(),
        
        # Context version confirmation modal
        context_version_modal(),
        # Store for drift detection table collapse state (True = expanded, False = collapsed)
        dcc.Store(id="drift-table-expanded", data=True),
        # Enhanced store for distribution comparison context data
        # Using session storage to ensure data is cleared when the user closes the browser
        # This is critical for maintaining multiple context items
        dcc.Store(
            id="distribution-chat-context",
            data=[],  # Initialize with empty list
            storage_type='session',  # Use session storage to ensure data is cleared when the user closes the browser
            clear_data=False  # Never clear this data automatically
        ),
        
        # NEW: Separate stores for Chat and Explain contexts
        dcc.Store(
            id="chat-context-data",
            data=[],  # Initialize with empty list
            storage_type='session',  # Use session storage for persistence
            clear_data=False
        ),
        dcc.Store(
            id="explain-context-data", 
            data=[],  # Initialize with empty list
            storage_type='session',  # Use session storage for persistence
            clear_data=False
        ),
        
        # Target attribute selection modal
        target_attribute_modal(),
        
        # Target attribute indicator component - temporarily commented out
        # html.Div(
        #     id="target-attribute-indicator-container",
        #     children=[
        #         # Will be populated with target attribute indicator when target is selected
        #         create_target_attribute_indicator()
        #     ],
        #     style={"display": "none"}
        # ),
        
        # Pipeline stage indicator
        html.Div(
            id=IDS.PIPELINE_STAGE_INDICATOR,
            className="pipeline-stage mt-3 mb-3",
            children=[
                html.Div([
                    html.Span("Preview", className="stage-label"),
                    html.Div(className="stage-dot")
                ]),
                html.Div(className="stage-connector"),
                html.Div([
                    html.Span("Detect", className="stage-label"),
                    html.Div(className="stage-dot")
                ]),
                html.Div(className="stage-connector"),
                html.Div([
                    html.Span("Explain", className="stage-label"),
                    html.Div(className="stage-dot")
                ])
            ],
            style={"display": "none"}
        ),
        
        # Add a hidden div for the scroll trigger
        html.Div(id="metrics-scroll-trigger", style={"display": "none"}),
        
        # Update clientside callback to handle scrolling when metrics table is displayed
        clientside_callback(
            """
            function(table_content, display_style) {
                // Log state for debugging (visible in browser console)
                console.log("Scroll trigger: table content exists=", table_content !== null, 
                           ", display_style=", display_style ? display_style.display : "none");
                
                // Only scroll if we have table content AND the display style is set to block
                if (table_content && display_style && 
                    (display_style.display === 'block' || display_style.opacity === '1')) {
                    
                    // Use requestAnimationFrame for smoother rendering
                    window.requestAnimationFrame(function() {
                        // Add a small delay to ensure DOM is updated
                        setTimeout(function() {
                            // Get the metrics table element
                            const metricsTable = document.getElementById('metrics-table-slide');
                            
                            // If we found it, scroll to it with smooth animation
                            if (metricsTable) {
                                console.log("Scrolling to metrics table");
                                metricsTable.scrollIntoView({ behavior: 'smooth', block: 'start' });
                            }
                        }, 100);
                    });
                }
                
                // Return null as we don't need to update any output
                return null;
            }
            """,
            Output("metrics-scroll-trigger", "children"),  # Dummy output that doesn't affect the UI
            [Input("metrics-table-container", "children"),  # Only scroll when table content exists
             Input("metrics-table-slide", "style")],       # And when the container is visible
            prevent_initial_call=True
        )
    ])


@callback(
    Output('url', 'pathname', allow_duplicate=True),
    Input('logout-button', 'n_clicks'),
    Input('menu-prompt', 'n_clicks'),
    prevent_initial_call=True
)
def logout_and_redirect(logout_clicks, setting_clicks):
    ctx = callback_context
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if (logout_clicks is not None and logout_clicks > 0) or (setting_clicks is not None and setting_clicks > 0):
        if button_id == "logout-button":
            logout_user()
            return "/"
        if button_id == "menu-prompt":
            return "/settings/prompts"


@callback(
    Output("setting-container", "style"),
    Output("home-container", "style"),
    Input("url", "pathname")
)
def show_page_content(pathname):
    if (pathname == "/home"):
        return {'display': 'none'}, {'display': 'flex'}

    if (pathname == "/settings/prompts"):
        return {'display': 'block'}, {'display': 'none'}
    return dash.no_update, dash.no_update


# ================================================================
# =survey modal ===================================================

@callback(
    Output("survey-modal-body", "children"),
    [Input("survey-modal", "is_open")],  # only use modal toggle as trigger
    prevent_initial_call=False  # allow initial call to ensure button exists on page load
)
def update_survey_content(modal_is_open):
    """Update survey modal content, including user profile form"""
    try:
        print(f"[SURVEY CONTENT] Updating survey content - modal_is_open: {modal_is_open}")
        content = survey_modal()
        print(f"[SURVEY CONTENT] Survey modal content created successfully")
        return content
    except Exception as e:
        print(f"[SURVEY CONTENT] Error creating survey modal: {e}")
        # return a simple error message
        return html.Div([
            html.H4("Error loading user profile"),
            html.P(f"Error: {str(e)}")
        ])


# ================================================================
# =Chat history===================================================


def format_message(msg):
    role_class = "user-message" if msg['role'] == 'user' else "assistant-message"
    content = msg.get("content")
    try:
        parsed_content = ast.literal_eval(content)
        if isinstance(parsed_content, dict) and "answer" in parsed_content:
            text = parsed_content["answer"]
    except (ValueError, SyntaxError):
        # If it isn't a dictionary-like string, return the string as is
        text = content
    return html.Div([
        html.Div([
            html.Span(msg['role'].capitalize(), className="message-role"),
        ], className="message-header"),
        dcc.Markdown(text, className="message-content")
    ], className=f"chat-message {role_class}")


@callback(
    Output("chat-history-modal-content", "children", allow_duplicate=True),
    Input("chat-history-modal", "is_open"),
    Input("chat-update-trigger", "data"),
    prevent_initial_call=True
)
def update_chat_history_modal(is_open, trigger):
    """Update the chat history modal content when the modal is opened or when the chat history is updated."""
    if not is_open:
        return dash.no_update
    
    # Create chat history content directly here
    try:
        # Get chat history from conversation
        conversation = Conversation()
        messages = conversation.get_history()
        
        if not messages:
            return html.Div([
                html.P("No chat history available.", className="text-muted text-center")
            ])
        
        # Format messages
        history_content = []
        for msg in messages:
            formatted_msg = format_message(msg)
            history_content.append(formatted_msg)
        
        return html.Div(history_content, style={"maxHeight": "400px", "overflowY": "auto"})
        
    except Exception as e:
        print(f"[CHAT HISTORY] Error loading chat history: {str(e)}")
        return html.Div([
            html.P("Error loading chat history.", className="text-danger text-center")
        ])


# ================================================================
# =Tool Modals===================================================

@callback(
    Output("evaluation-modal", "is_open"),
    [Input("menu-evaluation", "n_clicks"), 
     Input("evaluation-close-btn", "n_clicks")],
    [State("evaluation-modal", "is_open")],
    prevent_initial_call=True
)
def toggle_evaluation_modal(view_clicks, close_clicks, is_open):
    """Toggle the evaluation modal."""
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update
    
    prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if prop_id == "menu-evaluation":
        return True
    elif prop_id == "evaluation-close-btn":
        return False
    return dash.no_update

@callback(
    Output("sandbox-modal", "is_open"),
    [Input("menu-sandbox", "n_clicks"), 
     Input("sandbox-close-btn", "n_clicks")],
    [State("sandbox-modal", "is_open")],
    prevent_initial_call=True
)
def toggle_sandbox_modal(view_clicks, close_clicks, is_open):
    """Toggle the Python sandbox modal."""
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update
    
    prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if prop_id == "menu-sandbox":
        return True
    elif prop_id == "sandbox-close-btn":
        return False
    return dash.no_update

@callback(
    Output("url", "pathname", allow_duplicate=True),
    [Input("menu-model-settings", "n_clicks")],
    prevent_initial_call=True
)
def redirect_to_model_settings(n_clicks):
    """Redirect to model settings when menu-model-settings is clicked."""
    if n_clicks:
        return "/model"
    return dash.no_update





# Left side now permanently shows data preview - pipeline functionality moved to chat




# Simple callback to ensure left side always shows data preview
@callback(
    [Output("dataset-preview-container", "style"),
     Output("metrics-table-slide", "style")],
    [Input("url", "pathname")],
    prevent_initial_call=False
)
def ensure_data_preview_mode(pathname):
    """Ensure left side always shows data preview, never switches to detect mode"""
    dataset_preview_style = {"display": "block", "width": "100%"}
    metrics_table_style = {"display": "none"}
    return dataset_preview_style, metrics_table_style

                

@callback(
    Output("chat-history-modal", "is_open"),
    [Input("menu-view-chat-history", "n_clicks"), 
     Input("chat-history-close-btn", "n_clicks")],
    [State("chat-history-modal", "is_open")],
    prevent_initial_call=True
)
def toggle_chat_history_modal(view_clicks, close_clicks, is_open):
    """Toggle the chat history modal when the user clicks on the view chat history menu item or close button."""
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update
    
    prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if prop_id == "menu-view-chat-history":
        return True
    elif prop_id == "chat-history-close-btn":
        return False
    return dash.no_update

@callback(
    Output("snapshots-modal", "is_open"),
    [Input("menu-snapshots", "n_clicks"), 
     Input("snapshots-close-btn", "n_clicks")],
    [State("snapshots-modal", "is_open")],
    prevent_initial_call=True
)
def toggle_snapshots_modal(view_clicks, close_clicks, is_open):
    """Toggle the snapshots modal."""
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update
    
    prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if prop_id == "menu-snapshots":
        return True
    elif prop_id == "snapshots-close-btn":
        return False
    return dash.no_update

@callback(
    Output("commands-input", "disabled"),
    Output("run-commands", "disabled"),
    Input("run-commands", "n_clicks"),
    prevent_initial_call=True
)
def toggle_disable(n_clicks):
    return True, True

@callback(
    Output('next-question-input-1', "value"),
    Output('system-prompt-input', "value"),
    Output('persona-prompt-input', "value"),
    Output('prefix-prompt-input', "value"),
    Input("reset-prompt-button", "n_clicks"),
    prevent_initial_call=True
)
def reset_default_prompts(n_clicks):
    return [
        DEFAULT_NEXT_QUESTION_PROMPT,
        DEFAULT_SYSTEM_PROMPT,
        DEFAULT_PERSONA_PROMPT,
        DEFAULT_PREFIX_PROMPT
    ]

# Survey modal handling is done in UI/callback/user_callbacks.py
# This avoids output conflicts with the existing survey callbacks

@callback(
    Output("detect-distribution-btn", "color"),
    Output("detect-distribution-btn", "children"),
    Output("detect-distribution-btn", "style"),
    Input("detect-distribution-btn", "n_clicks"),
    State("detect-distribution-btn", "color"),
    prevent_initial_call=True
)
def toggle_detect_button(n_clicks, current_color):
    """Toggle the detect button on/off state with custom styling."""
    if current_color == "light":
        # Active state: white text on purple background
        return "primary", "Detect ", {"color": "white", "background-color": "#614385", "font-weight": "500"}
    else:
        # Inactive state: dark gray text on white background with visible border
        return "light", "Detect", {"color": "#333", "background-color": "white", "border": "1px solid #777", "font-weight": "400"}

# COMMENTED: Using chat-based explain workflow - no need for main page explain button
# @callback(
#     Output("explain-metrics-btn", "color"),
#     Output("explain-metrics-btn", "children"),
#     Output("explain-metrics-btn", "style"),
#     Input("explain-metrics-btn", "n_clicks"),
#     State("explain-metrics-btn", "color"),
#     prevent_initial_call=True
# )
# def toggle_explain_button(n_clicks, current_color):
#     """Toggle the explain button on/off state with custom styling."""
#     if current_color == "light":
#         # Active state: white text on purple background
#         return "primary", "Explain ", {"color": "white", "background-color": "#614385", "font-weight": "500"}
#     else:
#         # Inactive state: dark gray text on white background with visible border
#         return "light", "Explain", {"color": "#333", "background-color": "white", "border": "1px solid #777", "font-weight": "400"}

@callback(
    Output("distribution-modal", "is_open", allow_duplicate=True),
    [Input("close-distribution-modal", "n_clicks")],
    [State("distribution-modal", "is_open")],
    prevent_initial_call=True
)
def toggle_distribution_modal(close_clicks, is_open):
    """Close the distribution modal when the close button is clicked."""
    if not close_clicks:
        return dash.no_update
    
    # Get the property ID that triggered the callback
    ctx = dash.callback_context
    prop_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else ''
    
    if prop_id == "close-distribution-modal":
        return False
    
    return is_open

@callback(
    Output("metrics-explanation-modal", "is_open", allow_duplicate=True),
    [Input("close-metrics-modal", "n_clicks")],
    [State("metrics-explanation-modal", "is_open")],
    prevent_initial_call=True,
    id="simple-close-metrics-modal-callback"  # add unique ID
)
def toggle_metrics_modal_simple(close_clicks, is_open):  # modify function name to avoid name conflict
    """Close the metrics explanation modal when the close button is clicked."""
    if close_clicks:
        return False
    return dash.no_update

# Deprecated callback removed - detect-stage button no longer exists

def format_metric_value(value):
    """Format metric values for display."""
    if value == "N/A":
        return value
    elif isinstance(value, float):
        return f"{value:.4f}"
    return str(value)




@callback(
    [Output("distribution-modal", "is_open", allow_duplicate=True),
     Output("modal-selected-cell-info", "children", allow_duplicate=True),
     Output("modal-combined-distribution-chart", "children", allow_duplicate=True),
     Output("modal-distribution-comparison-summary", "children", allow_duplicate=True)],
    [Input('table-primary-overview', 'active_cell'),
     Input('table-secondary-overview', 'active_cell'),
     Input('table-overview', 'active_cell')],
    [State('table-primary-overview', 'data'),
     State('table-primary-overview', 'columns'),
     State('table-secondary-overview', 'data'),
     State('table-secondary-overview', 'columns'),
     State('table-overview', 'data'),
     State('table-overview', 'columns')],
    prevent_initial_call=True
)
def show_distribution_in_modal(primary_active_cell, secondary_active_cell, main_active_cell,
                          primary_data, primary_columns,
                          secondary_data, secondary_columns,
                          main_data, main_columns):
    """
    Open distribution modal and populate with charts when a cell is clicked.
    """
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
    
    # Default return values
    no_modal_return = False, None, None, None
    
    # If no cell is active in any table, don't show modal
    if not triggered_id or triggered_id not in ['table-primary-overview', 'table-secondary-overview', 'table-overview']:
        return no_modal_return
    
    # Determine which table was clicked and get data
    if triggered_id == 'table-primary-overview' and primary_active_cell is not None:
        active_cell = primary_active_cell
        data = primary_data
        columns = primary_columns
        dataset_type = "Primary dataset"
    elif triggered_id == 'table-secondary-overview' and secondary_active_cell is not None:
        active_cell = secondary_active_cell
        data = secondary_data
        columns = secondary_columns
        dataset_type = "Secondary dataset"
    elif triggered_id == 'table-overview' and main_active_cell is not None:
        active_cell = main_active_cell
        data = main_data
        columns = main_columns
        dataset_type = "Main table"
    else:
        return no_modal_return
    
    # Extract cell info
    try:
        row_idx = active_cell['row']
        col_idx = active_cell['column']
        row = data[row_idx]
        column_name = columns[col_idx]['name']
        cell_value = row[column_name]
    except (TypeError, IndexError, KeyError):
        # Handle cases where data structure doesn't match expectations
        return no_modal_return
    
    # Check if we have datasets to work with
    has_primary = hasattr(global_vars, 'df') and global_vars.df is not None
    has_secondary = hasattr(global_vars, 'secondary_df') and global_vars.secondary_df is not None
    
    if not has_primary:
        return True, "No dataset available", None, None
    
    # Create info about the selected cell
    cell_info = html.Div([
        html.P(f"Selected from {dataset_type}:", style={"fontWeight": "bold"}),
        html.P(f"Column: {column_name}, Value: {cell_value}")
    ])
    
    # Import the generate_distribution_chart and generate_comparison_summary functions from data_callbacks
    from UI.callback.data_callbacks import generate_distribution_chart, generate_comparison_summary, generate_combined_distribution_chart
    
    # Generate combined distribution chart if both datasets exist
    if has_secondary:
        combined_fig = generate_combined_distribution_chart(
            primary_df=global_vars.df,
            secondary_df=global_vars.secondary_df,
            column_name=column_name,
            highlight_value=cell_value
        )
        
        # Generate comparison summary for backend use (add to chat/explain functionality)
        # Note: UI display is hidden, but data is still calculated for context functions
        comparison_summary = generate_comparison_summary(
            primary_df=global_vars.df,
            secondary_df=global_vars.secondary_df,
            column_name=column_name,
            value=cell_value
        )
        
        # Return the combined chart in the primary chart area and empty the secondary
        return True, cell_info, combined_fig, comparison_summary
    else:
        # If only primary dataset exists, generate single chart
        primary_fig = generate_distribution_chart(
            df=global_vars.df,
            column_name=column_name,
            highlight_value=cell_value,
            is_primary=True
        )
        
        secondary_fig = html.Div("Upload a secondary dataset to see comparison")
        # Generate appropriate message for single dataset case
        comparison_summary = html.Div("Upload a secondary dataset to see comparison")
        
        # Return the chart
        return True, cell_info, primary_fig, comparison_summary

# Pipeline stage functionality moved to chat box

# Callback for drift detection table collapse/expand toggle
@callback(
    Output("drift-table-collapse", "is_open"),  # Controls the collapse component state
    Output("drift-table-toggle", "className"),  # Controls the icon appearance
    Output("drift-table-expanded", "data"),  # Stores the state
    Input("drift-table-toggle", "n_clicks"),  # Toggle button clicks
    State("drift-table-expanded", "data"),  # Current state
)
def toggle_drift_table_collapse(n_clicks, is_expanded):
    """Toggle the collapse/expand state of the drift detection table.
    
    Args:
        n_clicks: Number of clicks on the toggle button
        is_expanded: Current state (True = expanded, False = collapsed)
        
    Returns:
        tuple: (collapse_is_open, icon_class, new_state)
    """
    if n_clicks is None:
        # Initialize with default state
        return True, "fa fa-chevron-up", True
    
    # Toggle the state
    new_state = not is_expanded
    
    # Return appropriate values based on the new state
    if new_state:  # Expanded
        return True, "fa fa-chevron-up", True
    else:  # Collapsed
        return False, "fa fa-chevron-down", False
        
# Callback for target distribution analysis collapse/expand toggle
@callback(
    Output("target-dist-collapse", "is_open"),  # Controls the collapse component state
    Output("target-dist-toggle", "className"),  # Controls the icon appearance
    Output("target-dist-expanded", "data"),  # Stores the state
    Input("target-dist-toggle", "n_clicks"),  # Toggle button clicks
    State("target-dist-expanded", "data"),  # Current state
    prevent_initial_call=True
)
def toggle_target_dist_collapse(n_clicks, is_expanded):
    """Toggle the collapse/expand state of the target distribution analysis section.
    
    Args:
        n_clicks: Number of clicks on the toggle button
        is_expanded: Current state (True = expanded, False = collapsed)
        
    Returns:
        tuple: (collapse_is_open, icon_class, new_state)
    """
    if n_clicks is None:
        # Initialize with default state
        return True, "fa fa-chevron-up", True
    
    # Toggle the state
    new_state = not is_expanded
    
    # Return appropriate values based on the new state
    if new_state:  # Expanded
        return True, "fa fa-chevron-up", True
    else:  # Collapsed
        return False, "fa fa-chevron-down", False

# Conditional distribution analysis collapse/expand toggle has been removed as the section is now integrated in the target distribution section

# Callback to enable or disable the conditional distribution dual buttons based on dropdown selections
@callback(
    [Output("add-cond-dist-to-chat", "disabled"),
     Output("add-cond-dist-to-chat", "aria-disabled"),
     Output("add-cond-dist-to-explain", "disabled"), 
     Output("add-cond-dist-to-explain", "aria-disabled")],
    [Input("detect-target-value-dropdown", "value"),
     Input("detect-compare-attr-dropdown", "value")],
    prevent_initial_call=False
)
def update_cond_dist_button_state(target_value, compare_attribute):
    """
    Enable or disable the conditional distribution dual buttons based on whether 
    both the target value and comparison attribute have been selected.
    Also updates the aria-disabled attribute for accessibility.
    
    Args:
        target_value: The selected target value from the dropdown
        compare_attribute: The selected comparison attribute from the dropdown
        
    Returns:
        tuple: (chat_disabled, chat_aria_disabled, explain_disabled, explain_aria_disabled)
    """
    # Enable buttons only if both dropdowns have values
    if target_value and compare_attribute:
        return False, "false", False, "false"
    return True, "true", True, "true"


# Callback to enable or disable the target distribution dual buttons based on chart content  
@callback(
    [Output("add-target-dist-to-chat", "disabled"),
     Output("add-target-dist-to-chat", "aria-disabled"),
     Output("add-target-dist-to-explain", "disabled"),
     Output("add-target-dist-to-explain", "aria-disabled")],
    [Input("target-distribution-chart-container", "children")],
    prevent_initial_call=False
)
def update_target_dist_button_state(chart_children):
    """
    Enable or disable the target distribution dual buttons based on whether 
    the chart has been generated with valid data.
    Also updates the aria-disabled attribute for accessibility.
    
    Args:
        chart_children: The children of the target distribution chart container
        
    Returns:
        tuple: (chat_disabled, chat_aria_disabled, explain_disabled, explain_aria_disabled)
    """
    # Enable buttons if chart_children is not None and not an empty list
    if chart_children:
        return False, "false", False, "false" 
    return True, "true", True, "true"  # Buttons disabled and aria-disabled set to true

@callback(
    Output('url', 'pathname', allow_duplicate=True),
    [Input('content-nav', 'active_tab')],
    prevent_initial_call=True
)
def navigate_to_tab(active_tab):
    return f"/{active_tab}"

@callback(
    [Output("metric-explanation-container", "children"),
     Output("metric-explanation-container", "style"),
     Output("attribute-metric-explanation-modal", "is_open"),
     Output("attribute-metric-explanation-modal-body", "children")],
    Input("metrics-table", "active_cell"),
    [State("metrics-table", "data")],
    prevent_initial_call=True,
    id="explain-metric-cell-callback"
)
def explain_metric_cell(active_cell, data):
    """
    Generate structured explanation using GPT for the clicked attribute's metrics.
    Only responds to clicks on the Attribute column, showing metrics explanation
    in a modal with comprehensive analysis based on distribution shift metrics.
    """
    # Import at the top level to avoid issues
    from UI.functions import global_vars
    
    if not active_cell:
        return dash.no_update, {"display": "none"}, False, ""
    
    # Extract information about the clicked cell
    row_idx = active_cell['row']
    col_id = active_cell['column_id']
    
    # Only respond to clicks on the Attribute column
    if col_id != "Attribute":
        return dash.no_update, {"display": "none"}, False, ""
    
    # Get the row data for all metrics
    row_data = data[row_idx]
    attribute_name = row_data.get("Attribute", "Unknown")
    attribute_type = row_data.get("Type", "Unknown")
    js_divergence = row_data.get("JS_Divergence", "N/A")
    psi_value = row_data.get("PSI", "N/A")
    wasserstein = row_data.get("Wasserstein", "N/A")
    # test_statistic = row_data.get("Test_Statistic", "N/A")
    p_value = row_data.get("p_value", "N/A")
    
    # First, show loading state while GPT generates explanation
    loading_content = html.Div([
        html.H4(f"Distribution Shift Analysis: {attribute_name}", className="modal-title text-center mb-4"),
        dcc.Loading(
            id="loading-metrics-explanation",
            type="circle",
            children=html.Div(id="loading-metrics-explanation-output", style={"height": "300px"})
        )
    ])
    
    # Immediately show modal with loading state
    return dash.no_update, {"display": "none"}, True, loading_content


@callback(
    Output("attribute-metric-explanation-modal-body", "children", allow_duplicate=True),
    [Input("attribute-metric-explanation-modal", "is_open")],
    [State("metrics-table", "active_cell"),
     State("metrics-table", "data")],
    prevent_initial_call=True,
    id="generate-gpt-metrics-explanation-callback"
)
def generate_gpt_metrics_explanation(is_open, active_cell, data):
    """
    Generate a GPT-based explanation of metrics when the modal is opened.
    This happens asynchronously after showing the loading state.
    """
    # Import at the top level to avoid issues
    from UI.functions import global_vars
    
    if not is_open or not active_cell:
        return dash.no_update
    
    # Extract information from the active cell
    row_idx = active_cell['row']
    col_id = active_cell['column_id']
    
    # Only process for Attribute column
    if col_id != "Attribute":
        return dash.no_update
    
    # Get metric data for the selected attribute with safe access
    row_data = data[row_idx]
    attribute_name = row_data.get("Attribute", "Unknown")
    attribute_type = row_data.get("Type", "Unknown")
    js_divergence = row_data.get("JS_Divergence", "N/A")
    psi_value = row_data.get("PSI", "N/A")
    wasserstein = row_data.get("Wasserstein", "N/A")
    # test_statistic = row_data.get("Test_Statistic", "N/A")
    p_value = row_data.get("p_value", "N/A")
    
    # Prepare prompt for GPT
    metrics_info = {
        "Attribute": attribute_name,
        "Type": attribute_type,
        "JS_Divergence": js_divergence,
        "PSI": psi_value,
        "Wasserstein": wasserstein,
        # "Test_Statistic": test_statistic,
        "p_value": p_value
    }
    
    # Domain context to provide background
    domain_context = "The user is analyzing distribution shifts between two datasets using statistical metrics. "
    
    # Build the prompt using the structured requirements
    prompt = f"""
    {domain_context}
    
    Analyze and explain the distribution shift metrics for attribute '{attribute_name}' (Type: {attribute_type}) with these values:
    
    - JS Divergence: {js_divergence}
    - PSI: {psi_value}
    - Wasserstein Distance: {wasserstein}

    
    Please provide a structured explanation following these requirements:
    
    1. Explain each metric:
       - What is this metric?
       - How is it calculated?
       - How to intuitively understand this calculation method?
       - Why do we need to calculate this metric?
       - If the value is N/A, explain why it cannot be calculated for this attribute type.
    
    2. Comprehensive Analysis:
       - What distribution characteristics or changes are reflected by the values of these metrics for this attribute?
       - Explain the meaning in an easy-to-understand way.
    
    3. Format requirements:
       - Clear, organized format suitable for UI display in a modal window
       - Use bullet points, headings, or line breaks to improve readability
       - Keep explanations concise and focused on interpretability
    """
    
    try:
        # Use agent to generate explanation with GPT
        if hasattr(global_vars, 'agent') and global_vars.agent is not None:
            # Get GPT response using the run method
            response = global_vars.agent.run(prompt, stage="detect")
            
            # Handle different response formats
            if isinstance(response, dict) and 'answer' in response:
                # Dictionary format with answer key
                answer_text = response['answer']
            elif isinstance(response, tuple):
                # Tuple format - take the first element as answer
                try:
                    answer_text = response[0]
                except (IndexError, TypeError):
                    answer_text = str(response)
            else:
                # Assume response is the answer directly
                answer_text = str(response)
                
            # Print for debugging
            print(f"GPT response type: {type(response)}")
            print(f"Extracted answer text type: {type(answer_text)}")
            
            # Ensure we have string content
            if not isinstance(answer_text, str):
                answer_text = str(answer_text)
                
            # Format the response to ensure proper markdown
            formatted_text = format_reply_to_markdown(answer_text)
            
            # Print for debugging
            print(f"Formatted markdown text: {formatted_text[:100]}...")
            
            # Use dcc.Markdown directly instead of converting to HTML
            # This matches how the chat box displays messages
            formatted_response = html.Div([
                dcc.Markdown(
                    formatted_text,
                    className="gpt-explanation-content"
                )
            ], className="gpt-explanation-container")
            
            # Create the final content with proper structure and styling
            # We keep the same structure but use direct Markdown components instead of HTML conversion
            explanation_content = html.Div([
                html.H4(f"Distribution Shift Analysis: {attribute_name}", className="modal-title text-center mb-4"),
                html.Div([
                    html.H5("Attribute Information", className="mb-3 text-primary"),
                    html.P([html.Strong("Name: "), attribute_name]),
                    html.P([html.Strong("Type: "), attribute_type])
                ], className="mb-4"),
                # Display the formatted GPT explanation directly
                formatted_response
            ], className="metrics-explanation-container")
            
            return explanation_content
        else:
            # Fallback if agent is not available
            return html.Div([
                html.H4(f"Distribution Shift Analysis: {attribute_name}", className="modal-title text-center mb-4"),
                html.Div("Unable to generate explanation: Agent not available.", className="alert alert-warning")
            ])
    except Exception as e:
        # Handle errors gracefully
        return html.Div([
            html.H4(f"Distribution Shift Analysis: {attribute_name}", className="modal-title text-center mb-4"),
            html.Div(f"Error generating explanation: {str(e)}", className="alert alert-danger")
        ])


# Helper functions for generating structured metric explanations
def generate_attribute_description(attribute_name, attribute_type):
    """
    Generate a generic description for any attribute based on its name and type.
    This function creates descriptive text that works for any dataset without hardcoding.
    
    Args:
        attribute_name (str): The name of the attribute
        attribute_type (str): The type of the attribute (Binary, Categorical, Continuous, etc.)
        
    Returns:
        str: A generic description of the attribute
    """
    # Create a descriptive sentence based on attribute type
    type_descriptions = {
        "Binary": f"'{attribute_name}' is a binary attribute that represents a yes/no or true/false condition in the dataset.",
        "Categorical": f"'{attribute_name}' is a categorical attribute representing discrete categories or classes in the dataset.",
        "Continuous": f"'{attribute_name}' is a continuous numerical attribute representing quantitative measurements in the dataset.",
        "Datetime": f"'{attribute_name}' is a datetime attribute representing temporal information in the dataset."
    }
    
    # Return the description based on type, or a generic one if type is not recognized
    return type_descriptions.get(attribute_type, f"'{attribute_name}' is an attribute in the dataset that may contain important information for analysis.")



def generate_comprehensive_analysis(attribute_name, attribute_type, js_divergence, psi_value, wasserstein, test_statistic, p_value):
    """Generate a comprehensive analysis based on all metrics for this attribute."""
    
    # Default response if we can't make a meaningful analysis
    default_analysis = html.P("Insufficient data to provide a comprehensive analysis for this attribute.")
    
    # Significant shift detected flag
    significant_shift = False
    
    analysis_content = []
    
    # For Binary and Categorical attributes
    if attribute_type in ["Binary", "Categorical"]:
        try:
            # Check if JS divergence indicates a shift
            if "N/A" not in str(js_divergence) and float(js_divergence) > 0.1:
                significant_shift = True
                analysis_content.append(html.P([
                    f"The Jensen-Shannon divergence of {js_divergence} indicates a ",
                    float(js_divergence) > 0.2 and "substantial" or "moderate",
                    f" shift in the distribution of {attribute_name} between the two datasets."
                ]))
            elif "N/A" not in str(js_divergence):
                analysis_content.append(html.P([
                    f"The Jensen-Shannon divergence of {js_divergence} suggests minimal change in the distribution of {attribute_name}."
                ]))
            
            # Check p-value for statistical significance
            if "N/A" not in str(p_value) and float(p_value) < 0.05:
                significant_shift = True
                analysis_content.append(html.P([
                    f"The p-value of {p_value} from the Chi-squared test indicates that the observed differences are ",
                    "statistically significant. This means the distribution change is unlikely to be due to random chance."
                ]))
            elif "N/A" not in str(p_value):
                analysis_content.append(html.P([
                    f"The p-value of {p_value} suggests that any observed differences could be attributed to random variation ",
                    "rather than a systematic shift in distribution."
                ]))
                
            # For Binary attributes, also check PSI
            if attribute_type == "Binary" and "N/A" not in str(psi_value):
                psi_float = float(psi_value)
                if psi_float > 0.25:
                    significant_shift = True
                    analysis_content.append(html.P([
                        f"The Population Stability Index (PSI) of {psi_value} indicates a significant shift in the ",
                        f"proportion of different values for {attribute_name}. This level of change often requires attention."
                    ]))
                elif psi_float > 0.1:
                    analysis_content.append(html.P([
                        f"The PSI value of {psi_value} shows a moderate shift that may warrant monitoring for {attribute_name}."
                    ]))
                else:
                    analysis_content.append(html.P([
                        f"The PSI value of {psi_value} suggests stability in the distribution of {attribute_name}."
                    ]))
        except (ValueError, TypeError):
            pass
    
    # For Continuous attributes
    elif attribute_type == "Continuous":
        try:
            # Check p-value for statistical significance
            if "N/A" not in str(p_value) and float(p_value) < 0.05:
                significant_shift = True
                analysis_content.append(html.P([
                    f"The Kolmogorov-Smirnov test p-value of {p_value} indicates a statistically significant difference ",
                    f"in the distribution of {attribute_name} between the two datasets."
                ]))
            elif "N/A" not in str(p_value):
                analysis_content.append(html.P([
                    f"The p-value of {p_value} from the KS test suggests that the distributions of {attribute_name} ",
                    "are not significantly different between the two datasets."
                ]))
            
            # Context for Wasserstein distance
            if "N/A" not in str(wasserstein):
                analysis_content.append(html.P([
                    f"The Wasserstein distance of {wasserstein} represents the average change in {attribute_name} values ",
                    "between the two datasets. Note that this metric is scale-dependent, so it should be interpreted ",
                    f"relative to the overall range of {attribute_name}."
                ]))
        except (ValueError, TypeError):
            pass
    
    # Check if we managed to generate any analysis
    if not analysis_content:
        return default_analysis
    
    # Add a summary statement
    if significant_shift:
        analysis_content.append(html.P([
            html.Strong("Summary: "),
            f"The metrics indicate a notable distribution shift for {attribute_name}. ",
            "This suggests that the underlying patterns or characteristics of this attribute ",
            "have changed between the two datasets."
        ], className="analysis-summary"))
    else:
        analysis_content.append(html.P([
            html.Strong("Summary: "),
            f"The metrics suggest relatively stable distribution for {attribute_name} between the two datasets. ",
            "Any observed differences are likely within expected variation."
        ], className="analysis-summary"))
    
    return html.Div(analysis_content, className="comprehensive-analysis")


def generate_recommendations(attribute_name, attribute_type, js_divergence, psi_value, wasserstein, test_statistic, p_value):
    """Generate recommendations based on the metrics analysis."""
    
    # Determine if there's a significant shift
    significant_shift = False
    try:
        if (attribute_type in ["Binary", "Categorical"] and 
            (("N/A" not in str(js_divergence) and float(js_divergence) > 0.2) or 
             ("N/A" not in str(p_value) and float(p_value) < 0.01) or
             (attribute_type == "Binary" and "N/A" not in str(psi_value) and float(psi_value) > 0.25))):
            significant_shift = True
        elif (attribute_type == "Continuous" and 
              "N/A" not in str(p_value) and float(p_value) < 0.01):
            significant_shift = True
    except (ValueError, TypeError):
        pass
    
    # Generate recommendations
    recommendations = []
    
    if significant_shift:
        recommendations.extend([
            html.Li([
                "Investigate the root causes of this distribution shift in ", 
                html.Strong(attribute_name), 
                ". Potential causes could include changes in data collection methods, population demographics, or natural trends."
            ]),
            html.Li(["Consider creating a subset analysis to identify exactly which values or ranges have shifted most significantly."]),
            html.Li(["Evaluate the impact of this shift on any models or analyses that rely on this attribute."]),
            html.Li(["Document this shift for future reference and consider monitoring this attribute more closely going forward."])
        ])
        
        if "target" in attribute_name.lower() or "label" in attribute_name.lower():
            recommendations.append(html.Li(["As this appears to be a target variable, this shift could significantly impact model performance and may require model retraining or recalibration."]))
    else:
        recommendations.extend([
            html.Li(["Continue to monitor ", html.Strong(attribute_name), " for potential future shifts."]),
            html.Li(["Include this attribute in your regular data quality checks to ensure stability over time."]),
            html.Li(["Consider this attribute relatively stable for current analysis and modeling purposes."])
        ])
    
    # For sensitive attributes, add specific recommendations
    sensitive_keywords = ["gender", "sex", "race", "ethnicity", "age", "income", "religion", "disability", "nationality", "marital"]
    if any(keyword in attribute_name.lower() for keyword in sensitive_keywords):
        recommendations.append(html.Li(["As this appears to be a sensitive attribute, carefully consider the ethical implications and potential drift that may result from any distribution shifts."]))
    
    return html.Ul(recommendations, className="recommendations-list")


# Add callback to close the attribute-metric-explanation-modal
@callback(
    Output("attribute-metric-explanation-modal", "is_open", allow_duplicate=True),
    [Input("close-attribute-explanation-modal", "n_clicks")],
    [State("attribute-metric-explanation-modal", "is_open")],
    prevent_initial_call=True
)
def toggle_attribute_explanation_modal(n_clicks, is_open):
    if n_clicks:
        return False
    return is_open


    # Default stage if not set
    if not hasattr(global_vars, 'current_stage'):
        global_vars.current_stage = "explain"
    
    # Handle attribute column click
    if col_id == "Attribute":
        # Get all the metrics for this attribute
        psi_value = row_data.get("PSI", "N/A")
        wasserstein_value = row_data.get("Wasserstein", "N/A")
        p_value = row_data.get("p_value", "N/A")
        alert_level = row_data.get("Alert", "N/A")
        
        # Prepare the prompt for attribute explanation
        prompt = f"""
        Analyze the distribution shift for attribute '{attribute_name}' based on these metrics:
        
        PSI: {psi_value}
        Wasserstein Distance: {wasserstein_value}
        p-value (KS test): {p_value}
        Alert Level: {alert_level}
        
        Please provide:
        1. A comprehensive analysis of what these metrics collectively indicate about distribution shift for this attribute
        2. How to interpret these specific values
        3. What they indicate about distribution shift for this attribute
        4. Any recommendations for addressing this finding
        5. Whether further investigation is warranted and what form it should take
        """
        
        try:
            # Call the agent to generate the explanation
            answer, _, _, _, _, _ = global_vars.agent.run(prompt, global_vars.current_stage)
            
            # Format the explanation for display
            explanation_content = dcc.Markdown(answer, className="llm-text")
            
            # Create a header for the explanation
            header = html.H5(f"Comprehensive Analysis for '{attribute_name}'", 
                            style={"marginBottom": "15px", "color": "#614385"})
            
            # Metrics summary section
            metrics_summary = html.Div([
                html.H6("Metrics Summary:", style={"marginTop": "10px", "color": "#516395"}),
                html.Ul([
                    html.Li(f"PSI: {psi_value}"),
                    html.Li(f"Wasserstein Distance: {wasserstein_value}"),
                    html.Li(f"p-value: {p_value}"),
                    html.Li(f"Alert Level: {alert_level}")
                ])
            ])
            
            # Assemble the complete explanation component
            explanation_div = html.Div([
                header,
                metrics_summary,
                html.Hr(style={"marginBottom": "15px"}),
                explanation_content
            ])
            
            return dash.no_update, {"display": "none"}, explanation_div, {"display": "block", "marginTop": "20px", "padding": "15px", "border": "1px solid #ddd", "borderRadius": "5px", "backgroundColor": "#f9f9f9"}
            
        except Exception as e:
            # Handle any errors in generating the explanation
            error_message = html.Div([
                html.H5("Error Generating Attribute Analysis", style={"color": "red"}),
                html.P(f"An error occurred: {str(e)}")
            ])
            return dash.no_update, {"display": "none"}, error_message, {"display": "block", "marginTop": "20px", "color": "red"}
    
    # We only want to explain metric columns
    if col_id not in ["PSI", "Wasserstein", "p_value"]:
        return dash.no_update, {"display": "none"}, dash.no_update, {"display": "none"}
    
    metric_value = row_data[col_id]
    
    # Format the column name for display
    if col_id == "p_value":
        metric_name = "p-value (Kolmogorov-Smirnov test)"
    elif col_id == "Wasserstein":
        metric_name = "Wasserstein distance"
    else:
        metric_name = col_id
    
    # Prepare the prompt for GPT
    prompt = f"""
    Explain the following statistical metric in the context of distribution shift detection:
    
    Metric: {metric_name}
    Value: {metric_value}
    Attribute: {attribute_name}
    
    Please explain:
    1. What this metric measures
    2. How to interpret this specific value
    3. What it indicates about distribution shift for this attribute
    4. Any recommendations for addressing this finding
    5. Whether further investigation is warranted and what form it should take
    """
    
    try:
        # Call the agent to generate the explanation
        answer, _, _, _, _, _ = global_vars.agent.run(prompt, global_vars.current_stage)
        
        # Format the explanation for display
        explanation_content = dcc.Markdown(answer, className="llm-text")
        
        # Create a header for the explanation
        header = html.H5(f"Explanation of {metric_name} for {attribute_name}", 
                         style={"marginBottom": "15px", "color": "#614385"})
        
        # Assemble the complete explanation component
        explanation_div = html.Div([
            header,
            html.Hr(style={"marginBottom": "15px"}),
            explanation_content
        ])
        
        return explanation_div, {"display": "block", "marginTop": "20px", "padding": "15px", "border": "1px solid #ddd", "borderRadius": "5px"}, dash.no_update, {"display": "none"}
    
    except Exception as e:
        # Handle any errors in generating the explanation
        error_message = html.Div([
            html.H5("Error Generating Explanation", style={"color": "red"}),
            html.P(f"An error occurred: {str(e)}")
        ])
        return error_message, {"display": "block", "marginTop": "20px", "color": "red"}, dash.no_update, {"display": "none"}

# Reset to simple approach for file upload UI
clientside_callback(
    """
    function(contents, filename) {
        if (contents && filename && filename.length > 0) {
            return "Selected: " + filename[0];
        }
        return "Drag and Drop or Select File";
    }
    """,
    Output('upload-data-modal', 'children'),
    [Input('upload-data-modal', 'contents'),
     Input('upload-data-modal', 'filename')]
)

clientside_callback(
    """
    function(contents, filename) {
        if (contents && filename && filename.length > 0) {
            return "Selected: " + filename[0];
        }
        return "Drag and Drop or Select Primary File";
    }
    """,
    Output('upload-data-modal-primary', 'children'),
    [Input('upload-data-modal-primary', 'contents'),
     Input('upload-data-modal-primary', 'filename')]
)

clientside_callback(
    """
    function(contents, filename) {
        if (contents && filename && filename.length > 0) {
            return "Selected: " + filename[0];
        }
        return "Drag and Drop or Select Secondary File";
    }
    """,
    Output('upload-data-modal-secondary', 'children'),
    [Input('upload-data-modal-secondary', 'contents'),
     Input('upload-data-modal-secondary', 'filename')]
)

# Add callback for adding target distribution data to chat context
@callback(
    [Output("chat-context-data", "data", allow_duplicate=True),
     Output("add-target-dist-to-chat", "children"),
     Output("add-target-dist-to-chat", "style"),
     Output("notification-trigger", "data", allow_duplicate=True)],
    [Input("add-target-dist-to-chat", "n_clicks")],
    [State("target-distribution-chart-container", "children"),
     State("chat-context-data", "data")],
    prevent_initial_call=True
)
def add_target_distribution_to_chat(n_clicks, chart_children, current_context):
    """
    Add the target distribution data to the chat context when the + button is clicked.
    This adds a comparison of datasets across all target values to the chat context.
    
    Args:
        n_clicks: Number of button clicks
        chart_children: The children of the target distribution chart container
        current_context: The current chat context data
        
    Returns:
        tuple: Updated chat context, button appearance, and notification data
    """
    # Get target attribute from global variables
    target_attribute = None
    if hasattr(global_vars, 'target_attribute') and global_vars.target_attribute:
        target_attribute = global_vars.target_attribute
    else:
        # Fallback for when the target attribute hasn't been set
        target_attribute = "target attribute"
    if not n_clicks:
        return dash.no_update
    
    print(f"[TARGET DISTRIBUTION CONTEXT] Adding target distribution to chat context")
    
    # Generate a unique ID for this context item using utility function
    context_id = generate_context_id("target-dist")
    
    # Get target attribute name from global variables
    target_attribute = global_vars.target_attribute if hasattr(global_vars, 'target_attribute') else "target attribute"
    
    # Build the summary text for the target distribution comparison
    try:
        from UI.pages.components.detect_distribution_callbacks import get_target_distribution_data
        primary_dist, secondary_dist, column_type, attribute_name = get_target_distribution_data()
        
        print(f"[TARGET DISTRIBUTION CONTEXT] Distribution data retrieved:")
        print(f"  - Attribute: {attribute_name}")
        print(f"  - Column type: {column_type}")
        print(f"  - Primary distribution: {primary_dist}")
        print(f"  - Secondary distribution: {secondary_dist}")
        
        # generate summary text based on the column type
        if column_type == "categorical":
            # classification task: give the count of each class
            summary_text = f"Target Distribution Analysis for '{attribute_name}' (Classification):\n"
            summary_text += f"Attribute Type: Categorical with {len(set(primary_dist.keys()) | set(secondary_dist.keys()))} unique values\n\n"
            
            # get all unique values
            all_values = sorted(set(primary_dist.keys()) | set(secondary_dist.keys()))
            
            summary_text += "=== Distribution Counts by Class ===\n"
            summary_text += f"{'Value':<15} {'Primary':<10} {'Secondary':<10} {'Difference':<10}\n"
            summary_text += "-" * 50 + "\n"
            
            total_primary = sum(primary_dist.values()) if primary_dist else 0
            total_secondary = sum(secondary_dist.values()) if secondary_dist else 0
            
            for value in all_values:
                p_count = primary_dist.get(value, 0)
                s_count = secondary_dist.get(value, 0)
                diff = s_count - p_count
                
                summary_text += f"{str(value):<15} {p_count:<10} {s_count:<10} {diff:+<10}\n"
            
            summary_text += "-" * 50 + "\n"
            summary_text += f"{'Total':<15} {total_primary:<10} {total_secondary:<10}\n\n"
            
            # calculate percentage distribution
            summary_text += "=== Percentage Distribution ===\n"
            summary_text += f"{'Value':<15} {'Primary %':<12} {'Secondary %':<12} {'Shift':<10}\n"            
            summary_text += "-" * 50 + "\n"
            
            for value in all_values:
                p_count = primary_dist.get(value, 0)
                s_count = secondary_dist.get(value, 0)
                
                p_pct = (p_count / total_primary * 100) if total_primary > 0 else 0
                s_pct = (s_count / total_secondary * 100) if total_secondary > 0 else 0
                shift = s_pct - p_pct
                
                summary_text += f"{str(value):<15} {p_pct:<12.1f} {s_pct:<12.1f} {shift:+.1f}\n"

        elif column_type == "continuous":
            # regression task: give the binning information
            summary_text = f"Target Distribution Analysis for '{attribute_name}' (Regression):\n"
            summary_text += f"Attribute Type: Continuous (binned into {len(primary_dist)} ranges)\n\n"
            
            # get all bin ranges
            all_bins = sorted(set(primary_dist.keys()) | set(secondary_dist.keys()))
            
            summary_text += "=== Distribution by Bins (Ranges) ===\n"
            summary_text += f"{'Range':<20} {'Primary':<10} {'Secondary':<10} {'Difference':<10}\n"
            summary_text += "-" * 55 + "\n"
            
            total_primary = sum(primary_dist.values()) if primary_dist else 0
            total_secondary = sum(secondary_dist.values()) if secondary_dist else 0
            
            for bin_range in all_bins:
                p_count = primary_dist.get(bin_range, 0)
                s_count = secondary_dist.get(bin_range, 0)
                diff = s_count - p_count
                
                summary_text += f"{bin_range:<20} {p_count:<10} {s_count:<10} {diff:+<10}\n"
            
            summary_text += "-" * 55 + "\n"
            summary_text += f"{'Total':<20} {total_primary:<10} {total_secondary:<10}\n\n"
            
            # calculate density distribution
            summary_text += "=== Density Distribution ===\n"
            summary_text += f"{'Range':<20} {'Primary %':<12} {'Secondary %':<12} {'Shift':<10}\n"
            summary_text += "-" * 55 + "\n"
            
            for bin_range in all_bins:
                p_count = primary_dist.get(bin_range, 0)
                s_count = secondary_dist.get(bin_range, 0)
                
                p_pct = (p_count / total_primary * 100) if total_primary > 0 else 0
                s_pct = (s_count / total_secondary * 100) if total_secondary > 0 else 0
                shift = s_pct - p_pct
                
                summary_text += f"{bin_range:<20} {p_pct:<12.1f} {s_pct:<12.1f} {shift:+.1f}\n"
        
        else:
            # unknown column type: give a fallback summary
            summary_text = f"Target Distribution Analysis for '{attribute_name}':\n"
            summary_text += "Comparison of dataset distributions across all target values.\n"
            summary_text += f"Column type: {column_type}\n"
            
            if primary_dist or secondary_dist:
                summary_text += f"\nPrimary dataset distribution: {primary_dist}\n"
                summary_text += f"Secondary dataset distribution: {secondary_dist}\n"
        
        # add dataset basic information
        try:
            if hasattr(global_vars, 'df') and hasattr(global_vars, 'secondary_df'):
                if global_vars.df is not None and global_vars.secondary_df is not None:
                    primary_size = len(global_vars.df)
                    secondary_size = len(global_vars.secondary_df)
                    summary_text += f"\n=== Dataset Information ===\n"
                    summary_text += f"Primary dataset: {primary_size} samples\n"
                    summary_text += f"Secondary dataset: {secondary_size} samples\n"
                    summary_text += f"Size ratio: {secondary_size/primary_size:.2f}x\n"
        except Exception as e:
            print(f"[TARGET DISTRIBUTION CONTEXT] Error adding dataset info: {str(e)}")
        
    except Exception as e:
        print(f"[TARGET DISTRIBUTION CONTEXT] Error extracting enhanced statistics: {str(e)}")
        # fallback to original simple summary
        summary_text = f"Target Distribution Analysis for '{target_attribute}':\n"
        summary_text += "Comparison of dataset distributions across all target values.\n"
        
        try:
            # Add dataset statistics if available from global vars
            if hasattr(global_vars, 'df') and hasattr(global_vars, 'secondary_df'):
                dataset1_count = len(global_vars.df) if global_vars.df is not None else "unknown"
                dataset2_count = len(global_vars.secondary_df) if global_vars.secondary_df is not None else "unknown"
                summary_text += f"Primary dataset: {dataset1_count} samples\n"
                summary_text += f"Secondary dataset: {dataset2_count} samples\n"
        except Exception as e2:
            print(f"[TARGET DISTRIBUTION CONTEXT] Error extracting dataset info: {str(e2)}")
    
    # Make a deep copy of chart children and remove any buttons using utility function
    import copy
    preview_chart = copy.deepcopy(chart_children)
    preview_chart = filter_buttons_from_component(preview_chart)
    
    # Create the context item using utility function
    new_context = create_context_item(
        context_id=context_id,
        context_type="target_distribution",
        summary_text=summary_text,
        target_attribute=target_attribute,  # Additional field specific to target distribution
        chart_data=preview_chart  # Use the button-free preview chart
    )
    
    # Add to context list using utility function
    updated_context = add_context_item_to_list(current_context, new_context)
    
    # Store context info (will be processed when user asks questions)
    if hasattr(global_vars, 'agent') and global_vars.agent:
        global_vars.agent.add_user_action_to_history(
            f"User added target distribution analysis for '{target_attribute}' to chat context"
        )
    
    print(f"[TARGET DISTRIBUTION CONTEXT] Added target distribution with ID: {context_id}")
    

    
    # Provide visual feedback using utility function
    success_content, success_style = create_button_feedback_content(
        action_type="chat",
        target_name=target_attribute,
        success=True
    )
    
    # Also trigger the notification toast
    notification_trigger = {
        "timestamp": int(time.time() * 1000),
        "message": f"Target Distribution for '{target_attribute}' added to chat context",
        "type": "target_distribution"
    }
    
    # Add debug information to help track callback execution
    print(f"[TARGET DISTRIBUTION CONTEXT] Returning context with {len(updated_context)} items, notification_trigger: {notification_trigger}")
    
    # Return all 4 required outputs: context data, button content, button style, and notification trigger
    # ✅ CRITICAL FIX: Sync with global_vars for chat compatibility
    if hasattr(global_vars, 'explain_context_data'):
        global_vars.explain_context_data = updated_context.copy()
    else:
        global_vars.explain_context_data = updated_context.copy()
    print(f"[TABLE METRIC CONTEXT] ✅ Synced global_vars.explain_context_data: {len(updated_context)} items")
    
    return updated_context, success_content, success_style, notification_trigger


# Add callback for adding target distribution data to explain context
@callback(
    [Output("explain-context-data", "data", allow_duplicate=True),
     Output("add-target-dist-to-explain", "children"),
     Output("add-target-dist-to-explain", "style"),
     Output("notification-trigger", "data", allow_duplicate=True)],
    [Input("add-target-dist-to-explain", "n_clicks")],
    [State("target-distribution-chart-container", "children"),
     State("explain-context-data", "data")],
    prevent_initial_call=True
)
def add_target_distribution_to_explain(n_clicks, chart_children, current_context):
    """
    Add the target distribution data to the explain context when the + button is clicked.
    This adds a comparison of datasets across all target values to the explain context.
    
    Args:
        n_clicks: Number of button clicks
        chart_children: The children of the target distribution chart container
        current_context: The current explain context data
        
    Returns:
        tuple: Updated explain context, button appearance, and notification data
    """
    if not n_clicks:
        return dash.no_update
    
    print(f"[TARGET DISTRIBUTION CONTEXT] Adding target distribution to explain context")
    
    # Generate a unique ID for this context item using utility function
    context_id = generate_context_id("target-dist")
    
    # Get target attribute name from global variables
    target_attribute = global_vars.target_attribute if hasattr(global_vars, 'target_attribute') else "target attribute"
    
    # Generate the same comprehensive summary text as chat version
    summary_text = "Target Distribution Analysis:\n"
    summary_text += f"Target attribute: {target_attribute}\n\n"
    
    try:
        # Extract distribution data from global variables if available
        if hasattr(global_vars, 'df') and hasattr(global_vars, 'secondary_df'):
            dataset1_count = len(global_vars.df) if global_vars.df is not None else "unknown"
            dataset2_count = len(global_vars.secondary_df) if global_vars.secondary_df is not None else "unknown"
            summary_text += f"=== Dataset Overview ===\n"
            summary_text += f"Primary dataset: {dataset1_count} samples\n"
            summary_text += f"Secondary dataset: {dataset2_count} samples\n\n"
            
            # Add detailed distribution analysis
            try:
                primary_dist = global_vars.df[target_attribute].value_counts().to_dict() if global_vars.df is not None else {}
                secondary_dist = global_vars.secondary_df[target_attribute].value_counts().to_dict() if global_vars.secondary_df is not None else {}
                
                # Calculate totals
                total_primary = sum(primary_dist.values()) if primary_dist else 0
                total_secondary = sum(secondary_dist.values()) if secondary_dist else 0
                
                # Get all unique values
                all_values = set(primary_dist.keys()) | set(secondary_dist.keys())
                
                # Raw counts
                summary_text += "=== Raw Counts ===\n"
                summary_text += f"{'Value':<15} {'Primary':<10} {'Secondary':<10}\n"
                summary_text += "-" * 40 + "\n"
                
                for value in sorted(all_values):
                    p_count = primary_dist.get(value, 0)
                    s_count = secondary_dist.get(value, 0)
                    summary_text += f"{str(value):<15} {p_count:<10} {s_count:<10}\n"
                
                summary_text += f"{'Total':<15} {total_primary:<10} {total_secondary:<10}\n\n"
                
                # Percentage distribution  
                summary_text += "=== Percentage Distribution ===\n"
                summary_text += f"{'Value':<15} {'Primary %':<12} {'Secondary %':<12} {'Shift':<10}\n"            
                summary_text += "-" * 50 + "\n"
                
                for value in sorted(all_values):
                    p_count = primary_dist.get(value, 0)
                    s_count = secondary_dist.get(value, 0)
                    
                    p_pct = (p_count / total_primary * 100) if total_primary > 0 else 0
                    s_pct = (s_count / total_secondary * 100) if total_secondary > 0 else 0
                    shift = s_pct - p_pct
                    
                    summary_text += f"{str(value):<15} {p_pct:<12.1f} {s_pct:<12.1f} {shift:+.1f}\n"
                    
            except Exception as e1:
                print(f"[TARGET DISTRIBUTION CONTEXT] Error extracting distribution data: {str(e1)}")
                summary_text += f"Note: Detailed distribution data unavailable\n"
    except Exception as e2:
        print(f"[TARGET DISTRIBUTION CONTEXT] Error extracting dataset info: {str(e2)}")
    
    # Make a deep copy of chart children and remove any buttons using utility function
    import copy
    preview_chart = copy.deepcopy(chart_children)
    preview_chart = filter_buttons_from_component(preview_chart)
    
    # Create the context item using utility function
    new_context = create_context_item(
        context_id=context_id,
        context_type="target_distribution",
        summary_text=summary_text,
        target_attribute=target_attribute,  # Additional field specific to target distribution
        chart_data=preview_chart  # Use the button-free preview chart
    )
    
    # Add to context list using utility function
    updated_context = add_context_item_to_list(current_context, new_context)
    
    # Store context info (will be processed when user asks questions)
    if hasattr(global_vars, 'agent') and global_vars.agent:
        global_vars.agent.add_user_action_to_history(
            f"User added target distribution analysis for '{target_attribute}' to explain context"
        )
    
    print(f"[TARGET DISTRIBUTION CONTEXT] Added target distribution with ID: {context_id}")
    
    # Provide visual feedback using utility function
    success_content, success_style = create_button_feedback_content(
        action_type="explain",
        target_name=target_attribute,
        success=True
    )
    
    # Also trigger the notification toast
    import time
    notification_trigger = {
        "timestamp": int(time.time() * 1000),
        "message": f"Target Distribution for '{target_attribute}' added to explain context",
        "type": "target_distribution"
    }
    
    # ✅ CRITICAL FIX: Sync with global_vars for chat compatibility
    if hasattr(global_vars, 'explain_context_data'):
        global_vars.explain_context_data = updated_context.copy()
    else:
        global_vars.explain_context_data = updated_context.copy()
    print(f"[TARGET DISTRIBUTION CONTEXT] ✅ Synced global_vars.explain_context_data: {len(updated_context)} items")
    
    # Return all 4 required outputs: context data, button content, button style, and notification trigger
    # ✅ CRITICAL FIX: Sync with global_vars for chat compatibility
    if hasattr(global_vars, 'explain_context_data'):
        global_vars.explain_context_data = updated_context.copy()
    else:
        global_vars.explain_context_data = updated_context.copy()
    print(f"[TABLE METRIC CONTEXT] ✅ Synced global_vars.explain_context_data: {len(updated_context)} items")
    
    return updated_context, success_content, success_style, notification_trigger


# Add callback for adding distribution data to chat context
@callback(
    Output("chat-context-data", "data", allow_duplicate=True),
    [Input("add-distribution-to-chat", "n_clicks")],
    [State("modal-selected-cell-info", "children"),
     State("modal-distribution-comparison-summary", "children"),
     State("modal-combined-distribution-chart", "children"),
     State("chat-context-data", "data"),
     State("distribution-modal", "is_open")],
    prevent_initial_call=True
)
def add_distribution_to_chat(n_clicks, cell_info, comparison_summary, chart, current_context, _):
    """
    Add the current distribution comparison data to the chat context when the + button is clicked.
    This will be displayed in the chat interface as a context block that can be used for asking questions.
    
    The function ensures data integrity by properly handling the current_context,
    maintaining all existing context items while adding the new one. Each comparison item
    will be accumulated in the chatbox rather than replacing previous items.
    """
    if not n_clicks:
        return dash.no_update
    
    # ==== 详细输入数据调试 ====
    print(f"\n[🔍 INPUT DEBUG] ===============================")
    print(f"📋 cell_info:")
    print(f"   Type: {type(cell_info)}")
    print(f"   Length: {len(cell_info) if isinstance(cell_info, (list, dict, str)) else 'N/A'}")
    print(f"   Content: {str(cell_info)[:1000]}...")
    print(f"\n📊 comparison_summary:")
    print(f"   Type: {type(comparison_summary)}")
    print(f"   Content: {str(comparison_summary)[:1000]}...")
    print(f"\n📈 chart:")
    print(f"   Type: {type(chart)}")
    print(f"   Content: {str(chart)[:500]}...")
    print(f"[🔍 INPUT DEBUG] ===============================\n")
    
    # ===== 使用增强提取函数替换原有逻辑 =====
    print(f"[DISTRIBUTION CONTEXT] Starting ENHANCED data extraction...")
    
    # Enhanced cell info extraction
    cell_info_text = ""
    if cell_info:
        cell_info_text = extract_all_text_recursively(cell_info).strip()
        print(f"[DISTRIBUTION CONTEXT] ✅ Enhanced cell info: {len(cell_info_text)} chars")
        print(f"[DISTRIBUTION CONTEXT] 📋 Cell preview: {cell_info_text[:500]}...")
    else:
        print(f"[DISTRIBUTION CONTEXT] ❌ No cell_info provided")

    # Create a copy of the comparison summary for storage
    # Store the entire component structure so we can render it again later
    stored_summary = comparison_summary
    
    # Enhanced summary extraction using utility function
    summary_text = "Distribution Comparison Summary:\n"
    if comparison_summary:
        summary_extracted = extract_all_text_recursively(comparison_summary).strip()
        print(f"[DISTRIBUTION CONTEXT] ✅ Enhanced summary: {len(summary_extracted)} chars")
        print(f"[DISTRIBUTION CONTEXT] 📊 Summary preview: {summary_extracted[:500]}...")
        
        if summary_extracted:
            summary_text += summary_extracted + "\n"
        else:
            summary_text += "Statistical comparison data was processed\n"
    else:
        summary_text += "No statistical comparison data available\n"
        print(f"[DISTRIBUTION CONTEXT] ❌ No comparison_summary provided")
    
    
    # Generate unique context ID using utility function
    context_id = generate_context_id("dist")
    print(f"[DISTRIBUTION CONTEXT] Generated new item ID: {context_id}")
    
    # Create context item using utility function
    new_context = create_context_item(
        context_id=context_id,
        context_type="distribution_comparison",
        summary_text=summary_text,
        cell_info=cell_info_text,
        stored_summary=stored_summary,
        stored_chart=chart
    )
    
    # Add to context list using utility function
    updated_context = add_context_item_to_list(current_context, new_context)
    
    print(f"[DISTRIBUTION CONTEXT] Creating new context based on existing {len(current_context)} items")
    
    # Enhanced debug output to trace context data state after modification
    print(f"[DISTRIBUTION CONTEXT] After adding: {len(updated_context)} items with IDs: {[item.get('id') for item in updated_context if isinstance(item, dict)]}")
    
    # Verify the new item is present in the updated context
    item_ids = [item.get('id') for item in updated_context if isinstance(item, dict)]
    if context_id in item_ids:
        print(f"[DISTRIBUTION CONTEXT] Successfully added new item with ID: {context_id}")
    else:
        print(f"[DISTRIBUTION CONTEXT] ERROR: Failed to add new item with ID: {context_id}")
    
    # Safety check - ensure no duplicates (should never happen with timestamp-based IDs)
    id_counts = {}
    for item in updated_context:
        if isinstance(item, dict) and 'id' in item:
            id_counts[item['id']] = id_counts.get(item['id'], 0) + 1
    
    duplicate_ids = [id for id, count in id_counts.items() if count > 1]
    if duplicate_ids:
        print(f"[DISTRIBUTION CONTEXT] Warning: Found duplicate IDs: {duplicate_ids}")
        # Keep only one instance of each duplicate
        seen_ids = set()
        deduplicated_context = []
        for item in updated_context:
            if isinstance(item, dict) and 'id' in item:
                if item['id'] not in seen_ids:
                    seen_ids.add(item['id'])
                    deduplicated_context.append(item)
            else:
                # Keep non-dict items or items without IDs
                deduplicated_context.append(item)
        updated_context = deduplicated_context
        print(f"[DISTRIBUTION CONTEXT] Removed duplicates. New context size: {len(updated_context)}")
    
    # Final verification of the complete context data
    print(f"[DISTRIBUTION CONTEXT] Final context data contains {len(updated_context)} items")
    
    # Extract column name from cell info or context data
    attribute_name = "Unknown"
    
    # Extract column name from cell info text if available
    if isinstance(cell_info_text, str) and cell_info_text.strip():
        # Parse cell info lines looking for column information
        cell_info_lines = cell_info_text.strip().split('\n')
        for line in cell_info_lines:
            # Try to extract column name from common formats like "Column: value"
            if ':' in line and len(line.split(':')) > 1:
                possible_name = line.split(':', 1)[1].strip()
                if possible_name and len(possible_name) < 50:  # Reasonable length check
                    attribute_name = possible_name
                    break
        
        # Fallback: if extraction failed, try using the first line as column name
        if attribute_name == "Unknown" and len(cell_info_lines) > 0:
            first_line = cell_info_lines[0].strip()
            if len(first_line) < 50:  # Reasonable length check
                attribute_name = first_line
    
    # Log the extracted attribute name    
    print(f"[DISTRIBUTION CONTEXT] Extracted attribute name: '{attribute_name}'")
    
    # Store context info for later use when user asks questions (not immediately adding to agent)
    if hasattr(global_vars, 'agent') and global_vars.agent:
        # Only add user action to history, not system messages (those will be added when user asks questions)
        global_vars.agent.add_user_action_to_history(f"User added distribution comparison context for attribute '{attribute_name}' to chat context")
        print(f"[DISTRIBUTION CONTEXT] Context stored for attribute: {attribute_name} (will be activated when user asks questions)")
    
    # 调试打印 - 显示所有context items的属性名到单行
    all_attributes = []
    for idx, ctx_item in enumerate(updated_context):
        if isinstance(ctx_item, dict):
            # 从每个项目中提取列名
            item_attr = "Unknown"
            
            # 尝试从cell_info中提取
            if "cell_info" in ctx_item:
                cell_info_lines = ctx_item["cell_info"].split('\n')
                for line in cell_info_lines:
                    if "Column:" in line or "column:" in line.lower():
                        parts = re.split(r"column:", line, flags=re.IGNORECASE)
                        if len(parts) > 1:
                            item_attr = parts[1].strip()
                            break
                
                # 如果未找到，使用第一行
                if item_attr == "Unknown" and len(cell_info_lines) > 0:
                    first_line = cell_info_lines[0].strip()
                    if len(first_line) < 50:
                        item_attr = first_line
            
            all_attributes.append(f"Item_{idx}:{item_attr}")
    
    print(f"[DISTRIBUTION CONTEXT DEBUG] ALL ITEMS: {' | '.join(all_attributes)}")
    
    # CRITICAL FIX: Make sure we return the updated context with the new item appended
    # This ensures multiple items accumulate rather than being replaced
    print(f"[DISTRIBUTION CONTEXT] Returning updated context with {len(updated_context)} items")
    
    # ==== DETAILED DEBUG INFO FOR CHAT CONTEXT ====
    print(f"\n{'='*60}")
    print(f"[CHAT CONTEXT DEBUG] DISTRIBUTION COMPARISON ADDED TO CHAT BOX")
    print(f"{'='*60}")
    print(f"Context ID: {context_id}")
    print(f"Context Type: distribution_comparison")
    print(f"Cell Info: {cell_info_text[:100]}..." if cell_info_text else "No cell info")
    print(f"Timestamp: {new_context['timestamp']}")
    print(f"Summary Text Preview: {summary_text[:200]}...")
    print(f"Total Context Items After Addition: {len(updated_context)}")
    print(f"All Context IDs: {[item.get('id') for item in updated_context if isinstance(item, dict)]}")
    print(f"All Context Types: {[item.get('type') for item in updated_context if isinstance(item, dict)]}")
    print(f"What will be shared with chat:")
    print(f"  - Distribution comparison data: Yes")
    print(f"  - Statistical summary: {'Yes' if summary_text else 'No'}")
    print(f"  - Chart data: {'Yes' if new_context.get('stored_chart') else 'No'}")
    print(f"  - Cell information: {'Yes' if cell_info_text else 'No'}")
    print(f"  - Expanded by default: {new_context['expanded']}")
    print(f"{'='*60}\n")
    
    # ✅ CRITICAL FIX: Sync with global_vars for chat compatibility
    if hasattr(global_vars, 'explain_context_data'):
        global_vars.explain_context_data = updated_context.copy()
    else:
        global_vars.explain_context_data = updated_context.copy()
    print(f"[TABLE METRIC CONTEXT] ✅ Synced global_vars.explain_context_data: {len(updated_context)} items")
    
    return updated_context  # This return statement is critical for accumulating items


# Add callback for adding distribution data to explain context
@callback(
    Output("explain-context-data", "data", allow_duplicate=True),
    [Input("add-distribution-to-explain", "n_clicks")],
    [State("modal-selected-cell-info", "children"),
     State("modal-distribution-comparison-summary", "children"),
     State("modal-combined-distribution-chart", "children"),
     State("explain-context-data", "data"),
     State("distribution-modal", "is_open")],
    prevent_initial_call=True
)
def add_distribution_to_explain(n_clicks, cell_info, comparison_summary, chart, current_context, _):
    """
    Add the current distribution comparison data to the explain context when the + button is clicked.
    This will be displayed in the explain interface as a context block.
    
    The function ensures data integrity by properly handling the current_context,
    maintaining all existing context items while adding the new one.
    """
    if not n_clicks:
        return dash.no_update
    
    print(f"[DISTRIBUTION CONTEXT] Adding distribution comparison to explain context")
    
    # Enhanced cell info extraction using utility function
    cell_info_text = ""
    if cell_info:
        cell_info_text = extract_all_text_recursively(cell_info).strip()
        print(f"[DISTRIBUTION CONTEXT] ✅ Enhanced cell info: {len(cell_info_text)} chars")
    else:
        print(f"[DISTRIBUTION CONTEXT] ❌ No cell_info provided")

    # Create a copy of the comparison summary for storage
    stored_summary = comparison_summary
    
    # Enhanced summary extraction using utility function
    summary_text = "Distribution Comparison Summary:\n"
    if comparison_summary:
        summary_extracted = extract_all_text_recursively(comparison_summary).strip()
        print(f"[DISTRIBUTION CONTEXT] ✅ Enhanced summary: {len(summary_extracted)} chars")
        
        if summary_extracted:
            summary_text += summary_extracted + "\n"
        else:
            summary_text += "Statistical comparison data was processed\n"
    else:
        summary_text += "No statistical comparison data available\n"
        print(f"[DISTRIBUTION CONTEXT] ❌ No comparison_summary provided")
    
    # Generate unique context ID using utility function
    context_id = generate_context_id("dist")
    print(f"[DISTRIBUTION CONTEXT] Generated new item ID: {context_id}")
    
    # Create context item using utility function
    new_context = create_context_item(
        context_id=context_id,
        context_type="distribution_comparison",
        summary_text=summary_text,
        cell_info=cell_info_text,
        stored_summary=stored_summary,
        stored_chart=chart
    )
    
    # Add to context list using utility function
    updated_context = add_context_item_to_list(current_context, new_context)
    
    # Extract column name from cell info for logging
    attribute_name = "Unknown"
    if isinstance(cell_info_text, str) and cell_info_text.strip():
        cell_info_lines = cell_info_text.strip().split('\n')
        for line in cell_info_lines:
            if ':' in line and len(line.split(':')) > 1:
                possible_name = line.split(':', 1)[1].strip()
                if possible_name and len(possible_name) < 50:
                    attribute_name = possible_name
                    break
        
        if attribute_name == "Unknown" and len(cell_info_lines) > 0:
            first_line = cell_info_lines[0].strip()
            if len(first_line) < 50:
                attribute_name = first_line
    
    # Store context info for later use
    if hasattr(global_vars, 'agent') and global_vars.agent:
        global_vars.agent.add_user_action_to_history(
            f"User added distribution comparison context for attribute '{attribute_name}' to explain context"
        )
        print(f"[DISTRIBUTION CONTEXT] Context stored for attribute: {attribute_name}")
    
    print(f"[DISTRIBUTION CONTEXT] Successfully added distribution comparison to explain context with ID: {context_id}")
    print(f"[DISTRIBUTION CONTEXT] Total context items after addition: {len(updated_context)}")
    
    # ✅ CRITICAL FIX: Sync with global_vars for chat compatibility
    if hasattr(global_vars, 'explain_context_data'):
        global_vars.explain_context_data = updated_context.copy()
    else:
        global_vars.explain_context_data = updated_context.copy()
    print(f"[DISTRIBUTION CONTEXT] ✅ Synced global_vars.explain_context_data: {len(updated_context)} items")
    
    # ✅ CRITICAL FIX: Sync with global_vars for chat compatibility
    if hasattr(global_vars, 'explain_context_data'):
        global_vars.explain_context_data = updated_context.copy()
    else:
        global_vars.explain_context_data = updated_context.copy()
    print(f"[TABLE METRIC CONTEXT] ✅ Synced global_vars.explain_context_data: {len(updated_context)} items")
    
    return updated_context


@callback(
    Output("chat-context-data", "data", allow_duplicate=True),
    [Input({"type": "add-metric-to-chat", "metric": ALL}, "n_clicks")],
    [State("metrics-modal-attribute", "children"),
     State("modal-selected-cell-info", "children"),
     State("chat-context-data", "data")],
    prevent_initial_call=True
)
# Add callback for adding conditional distribution data to chat context
@callback(
    [Output("chat-context-data", "data", allow_duplicate=True),
     Output("add-cond-dist-to-chat", "children"),
     Output("add-cond-dist-to-chat", "style"),
     Output("notification-trigger", "data", allow_duplicate=True)],
    [Input("add-cond-dist-to-chat", "n_clicks")],
    [State("detect-target-value-dropdown", "value"),
     State("detect-compare-attr-dropdown", "value"),
     State("detect-conditional-chart-container", "children"),
     State("chat-context-data", "data")],
    prevent_initial_call=True
)
def add_conditional_distribution_to_chat(n_clicks, target_value, compare_attribute, chart, current_context):
    """
    Add the conditional distribution analysis to the chat context when the + button is clicked.
    This function captures the currently selected target value and comparison attribute data.
    
    Args:
        n_clicks: Number of button clicks
        target_value: The selected target value from the dropdown
        compare_attribute: The selected comparison attribute from the dropdown
        chart: The chart component from the conditional distribution chart container
        current_context: The current chat context data
        
    Returns:
        tuple: (updated_context, button_content, button_style, notification_trigger)
    """
    if not n_clicks or target_value is None or compare_attribute is None:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update
    
    print(f"[CONDITIONAL DISTRIBUTION CONTEXT] Adding conditional distribution analysis to chat context")
    
    # Generate a unique ID for this context item using utility function
    context_id = generate_context_id("cond-dist")
    
    # Get target attribute name from global variables
    target_attribute = global_vars.target_attribute if hasattr(global_vars, 'target_attribute') else "target attribute"
    
    # Prepare enhanced summary text for the context
    summary_text = "Conditional Distribution Analysis:\n"

    summary_text += f"Target attribute: {target_attribute}\n"
    summary_text += f"Target value: {target_value}\n"
    summary_text += f"Compared with attribute: {compare_attribute}\n"
    
    try:
        # Add comprehensive dataset information if available
        if hasattr(global_vars, 'df') and hasattr(global_vars, 'secondary_df'):
            dataset1_count = len(global_vars.df) if global_vars.df is not None else "unknown"
            dataset2_count = len(global_vars.secondary_df) if global_vars.secondary_df is not None else "unknown"
            summary_text += f"\n=== Dataset Overview ===\n"
            summary_text += f"Primary dataset: {dataset1_count} samples\n"
            summary_text += f"Secondary dataset: {dataset2_count} samples\n"
            
            # Enhanced target value frequency analysis
            try:
                if hasattr(global_vars, 'df') and hasattr(global_vars, 'secondary_df'):
                    # === correct the data type of target_value ===
                    # debug the original data type
                    print(f"[CONDITIONAL DEBUG] target_value: {target_value} (type: {type(target_value)})")
                    print(f"[CONDITIONAL DEBUG] target_attribute column dtype: {global_vars.df[target_attribute].dtype}")
                    print(f"[CONDITIONAL DEBUG] target_attribute unique values: {global_vars.df[target_attribute].unique()}")
                    
                    # smart type conversion
                    converted_target_value = target_value
                    
                    # if the target column is numeric, try to convert target_value
                    if pd.api.types.is_numeric_dtype(global_vars.df[target_attribute]):
                        try:
                            if isinstance(target_value, str):
                                # try to convert to numeric
                                if '.' in target_value:
                                    converted_target_value = float(target_value)
                                else:
                                    converted_target_value = int(target_value)
                            print(f"[CONDITIONAL DEBUG] Converted target_value: {converted_target_value} (type: {type(converted_target_value)})")
                        except (ValueError, TypeError) as e:
                            print(f"[CONDITIONAL DEBUG] Could not convert target_value to numeric: {e}")
                    
                    # use the converted value to filter the data
                    df1_target_subset = global_vars.df[global_vars.df[target_attribute] == converted_target_value]
                    df2_target_subset = global_vars.secondary_df[global_vars.secondary_df[target_attribute] == converted_target_value]
                    
                    print(f"[CONDITIONAL DEBUG] Primary subset size: {len(df1_target_subset)}")
                    print(f"[CONDITIONAL DEBUG] Secondary subset size: {len(df2_target_subset)}")
                    
                    df1_target_count = len(df1_target_subset)
                    df2_target_count = len(df2_target_subset)
                    
                    summary_text += f"\n=== Target Value Analysis ===\n"
                    summary_text += f"'{target_attribute}={converted_target_value}' frequency:\n"
                    summary_text += f"- Primary dataset: {df1_target_count} samples ({df1_target_count/len(global_vars.df)*100:.1f}%)\n"
                    summary_text += f"- Secondary dataset: {df2_target_count} samples ({df2_target_count/len(global_vars.secondary_df)*100:.1f}%)\n"
                    
                    # Analyze the comparison attribute distribution within target value subsets
                    if len(df1_target_subset) > 0 and len(df2_target_subset) > 0:
                        summary_text += f"\n=== Conditional Distribution of '{compare_attribute}' ===\n"
                        
                        # Primary dataset distribution
                        df1_compare_dist = df1_target_subset[compare_attribute].value_counts().sort_index()
                        summary_text += f"Primary dataset (when {target_attribute}={converted_target_value}):\n"
                        for value, count in df1_compare_dist.items():
                            percentage = count / len(df1_target_subset) * 100
                            summary_text += f"  {compare_attribute}={value}: {count} ({percentage:.1f}%)\n"
                        
                        # Secondary dataset distribution  
                        df2_compare_dist = df2_target_subset[compare_attribute].value_counts().sort_index()
                        summary_text += f"Secondary dataset (when {target_attribute}={converted_target_value}):\n"
                        for value, count in df2_compare_dist.items():
                            percentage = count / len(df2_target_subset) * 100
                            summary_text += f"  {compare_attribute}={value}: {count} ({percentage:.1f}%)\n"
                        
                        # Calculate distribution shift
                        summary_text += f"\n=== Distribution Shift Analysis ===\n"
                        all_compare_values = set(df1_compare_dist.index) | set(df2_compare_dist.index)
                        for value in sorted(all_compare_values):
                            df1_pct = (df1_compare_dist.get(value, 0) / len(df1_target_subset) * 100) if len(df1_target_subset) > 0 else 0
                            df2_pct = (df2_compare_dist.get(value, 0) / len(df2_target_subset) * 100) if len(df2_target_subset) > 0 else 0
                            shift = df2_pct - df1_pct
                            summary_text += f"  {compare_attribute}={value}: {df1_pct:.1f}% → {df2_pct:.1f}% (shift: {shift:+.1f}%)\n"
                    
            except Exception as e:
                print(f"[CONDITIONAL DISTRIBUTION CONTEXT] Error extracting detailed analysis: {str(e)}")
                summary_text += f"\nNote: Detailed analysis unavailable due to data access issues\n"
    except Exception as e:
        print(f"[CONDITIONAL DISTRIBUTION CONTEXT] Error extracting dataset info: {str(e)}")
        summary_text += f"\nNote: Dataset information unavailable\n" 
    
    # Create the context item using utility function
    new_context = create_context_item(
        context_id=context_id,
        context_type="conditional_distribution",
        summary_text=summary_text,
        target_attribute=target_attribute,
        target_value=target_value,
        compare_attribute=compare_attribute,
        chart=chart
    )
    
    # Add to context list using utility function
    updated_context = add_context_item_to_list(current_context, new_context)
    
    # Store context info (will be processed when user asks questions)
    if hasattr(global_vars, 'agent') and global_vars.agent:
        global_vars.agent.add_user_action_to_history(
            f"User added conditional distribution analysis for '{target_attribute}={target_value}' by '{compare_attribute}' to chat context"
        )
    
    print(f"[CONDITIONAL DISTRIBUTION CONTEXT] Added conditional distribution with ID: {context_id}")
    
    # ==== DETAILED DEBUG INFO FOR CHAT CONTEXT ====
    print(f"\n{'='*60}")
    print(f"[CHAT CONTEXT DEBUG] CONDITIONAL DISTRIBUTION ADDED TO CHAT BOX")
    print(f"{'='*60}")
    print(f"Context ID: {context_id}")
    print(f"Context Type: conditional_distribution")
    print(f"Target Attribute: {target_attribute}")
    print(f"Target Value: {target_value}")
    print(f"Compare Attribute: {compare_attribute}")
    print(f"Timestamp: {new_context['timestamp']}")
    print(f"Summary Text Preview: {summary_text[:200]}...")
    print(f"Summary Text Length: {len(summary_text)} characters")
    print(f"Summary Text Full:\n{summary_text}")
    print(f"Total Context Items After Addition: {len(updated_context)}")
    print(f"All Context IDs: {[item.get('id') for item in updated_context if isinstance(item, dict)]}")
    print(f"All Context Types: {[item.get('type') for item in updated_context if isinstance(item, dict)]}")
    print(f"What will be shared with chat:")
    print(f"  - Conditional distribution analysis: {target_attribute}={target_value}")
    print(f"  - Comparison attribute: {compare_attribute}")
    print(f"  - Chart data: {'Yes' if new_context.get('chart') else 'No'}")
    print(f"  - Statistical summary: {'Yes' if summary_text else 'No'}")
    print(f"  - Expanded by default: {new_context['expanded']}")
    print(f"{'='*60}\n")
    
    # Provide visual feedback using utility function
    success_content, success_style = create_button_feedback_content(
        action_type="chat",
        target_name=f"{target_attribute}={target_value}",
        success=True
    )
    
    # Create notification trigger data
    import time
    notification_data = {
        "timestamp": int(time.time() * 1000),
        "type": "conditional_distribution",
        "message": f"Conditional Distribution for '{target_attribute}={target_value}' by '{compare_attribute}' added to chat context"
    }
    
    # Return updated context, visual feedback, and notification trigger
    # ✅ CRITICAL FIX: Sync with global_vars for chat compatibility
    if hasattr(global_vars, 'explain_context_data'):
        global_vars.explain_context_data = updated_context.copy()
    else:
        global_vars.explain_context_data = updated_context.copy()
    print(f"[TABLE METRIC CONTEXT] ✅ Synced global_vars.explain_context_data: {len(updated_context)} items")
    
    return updated_context, success_content, success_style, notification_data


# Add callback for adding conditional distribution data to explain context
@callback(
    [Output("explain-context-data", "data", allow_duplicate=True),
     Output("add-cond-dist-to-explain", "children"),
     Output("add-cond-dist-to-explain", "style"),
     Output("notification-trigger", "data", allow_duplicate=True)],
    [Input("add-cond-dist-to-explain", "n_clicks")],
    [State("detect-target-value-dropdown", "value"),
     State("detect-compare-attr-dropdown", "value"),
     State("detect-conditional-chart-container", "children"),
     State("explain-context-data", "data")],
    prevent_initial_call=True
)
def add_conditional_distribution_to_explain(n_clicks, target_value, compare_attribute, chart, current_context):
    """
    Add the conditional distribution analysis to the explain context when the + button is clicked.
    This function captures the currently selected target value and comparison attribute data.
    
    Args:
        n_clicks: Number of button clicks
        target_value: The selected target value from the dropdown
        compare_attribute: The selected comparison attribute from the dropdown
        chart: The chart component from the conditional distribution chart container
        current_context: The current explain context data
        
    Returns:
        tuple: (updated_context, button_content, button_style, notification_trigger)
    """
    if not n_clicks or target_value is None or compare_attribute is None:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update
    
    print(f"[CONDITIONAL DISTRIBUTION CONTEXT] Adding conditional distribution analysis to explain context")
    
    # Generate a unique ID for this context item using utility function
    context_id = generate_context_id("cond-dist")
    
    # Get target attribute name from global variables
    target_attribute = global_vars.target_attribute if hasattr(global_vars, 'target_attribute') else "target attribute"
    
    # Prepare enhanced summary text for the context
    summary_text = "Conditional Distribution Analysis:\n"
    summary_text += f"Target attribute: {target_attribute}\n"
    summary_text += f"Target value: {target_value}\n"
    summary_text += f"Compared with attribute: {compare_attribute}\n"
    
    try:
        # Add comprehensive dataset information if available
        if hasattr(global_vars, 'df') and hasattr(global_vars, 'secondary_df'):
            dataset1_count = len(global_vars.df) if global_vars.df is not None else "unknown"
            dataset2_count = len(global_vars.secondary_df) if global_vars.secondary_df is not None else "unknown"
            summary_text += f"\n=== Dataset Overview ===\n"
            summary_text += f"Primary dataset: {dataset1_count} samples\n"
            summary_text += f"Secondary dataset: {dataset2_count} samples\n"
            
            # Enhanced target value frequency analysis
            try:
                if hasattr(global_vars, 'df') and hasattr(global_vars, 'secondary_df'):
                    # Convert target_value to correct data type
                    converted_target_value = target_value
                    
                    # Smart type conversion for numeric columns
                    if pd.api.types.is_numeric_dtype(global_vars.df[target_attribute]):
                        try:
                            if isinstance(target_value, str):
                                if '.' in target_value:
                                    converted_target_value = float(target_value)
                                else:
                                    converted_target_value = int(target_value)
                        except (ValueError, TypeError) as e:
                            print(f"[CONDITIONAL DEBUG] Could not convert target_value to numeric: {e}")
                    
                    # Filter data by target value
                    df1_target_subset = global_vars.df[global_vars.df[target_attribute] == converted_target_value]
                    df2_target_subset = global_vars.secondary_df[global_vars.secondary_df[target_attribute] == converted_target_value]
                    
                    df1_target_count = len(df1_target_subset)
                    df2_target_count = len(df2_target_subset)
                    
                    summary_text += f"\n=== Target Value Analysis ===\n"
                    summary_text += f"'{target_attribute}={converted_target_value}' frequency:\n"
                    summary_text += f"- Primary dataset: {df1_target_count} samples ({df1_target_count/len(global_vars.df)*100:.1f}%)\n"
                    summary_text += f"- Secondary dataset: {df2_target_count} samples ({df2_target_count/len(global_vars.secondary_df)*100:.1f}%)\n"
                    
                    # Analyze conditional distribution
                    if len(df1_target_subset) > 0 and len(df2_target_subset) > 0:
                        summary_text += f"\n=== Conditional Distribution of '{compare_attribute}' ===\n"
                        
                        # Primary dataset distribution
                        df1_compare_dist = df1_target_subset[compare_attribute].value_counts().sort_index()
                        summary_text += f"Primary dataset (when {target_attribute}={converted_target_value}):\n"
                        for value, count in df1_compare_dist.items():
                            percentage = count / len(df1_target_subset) * 100
                            summary_text += f"  {compare_attribute}={value}: {count} ({percentage:.1f}%)\n"
                        
                        # Secondary dataset distribution  
                        df2_compare_dist = df2_target_subset[compare_attribute].value_counts().sort_index()
                        summary_text += f"Secondary dataset (when {target_attribute}={converted_target_value}):\n"
                        for value, count in df2_compare_dist.items():
                            percentage = count / len(df2_target_subset) * 100
                            summary_text += f"  {compare_attribute}={value}: {count} ({percentage:.1f}%)\n"
                        
                        # Calculate distribution shift
                        summary_text += f"\n=== Distribution Shift Analysis ===\n"
                        all_compare_values = set(df1_compare_dist.index) | set(df2_compare_dist.index)
                        for value in sorted(all_compare_values):
                            df1_pct = (df1_compare_dist.get(value, 0) / len(df1_target_subset) * 100) if len(df1_target_subset) > 0 else 0
                            df2_pct = (df2_compare_dist.get(value, 0) / len(df2_target_subset) * 100) if len(df2_target_subset) > 0 else 0
                            shift = df2_pct - df1_pct
                            summary_text += f"  {compare_attribute}={value}: {df1_pct:.1f}% → {df2_pct:.1f}% (shift: {shift:+.1f}%)\n"
                    
            except Exception as e:
                print(f"[CONDITIONAL DISTRIBUTION CONTEXT] Error extracting detailed analysis: {str(e)}")
                summary_text += f"\nNote: Detailed analysis unavailable due to data access issues\n"
    except Exception as e:
        print(f"[CONDITIONAL DISTRIBUTION CONTEXT] Error extracting dataset info: {str(e)}")
        summary_text += f"\nNote: Dataset information unavailable\n" 
    
    # Create the context item using utility function
    new_context = create_context_item(
        context_id=context_id,
        context_type="conditional_distribution",
        summary_text=summary_text,
        target_attribute=target_attribute,
        target_value=target_value,
        compare_attribute=compare_attribute,
        chart=chart
    )
    
    # Add to context list using utility function
    updated_context = add_context_item_to_list(current_context, new_context)
    
    # Store context info (will be processed when user asks questions)
    if hasattr(global_vars, 'agent') and global_vars.agent:
        global_vars.agent.add_user_action_to_history(
            f"User added conditional distribution analysis for '{target_attribute}={target_value}' by '{compare_attribute}' to explain context"
        )
    
    print(f"[CONDITIONAL DISTRIBUTION CONTEXT] Added conditional distribution with ID: {context_id}")
    
    # Provide visual feedback using utility function
    success_content, success_style = create_button_feedback_content(
        action_type="explain",
        target_name=f"{target_attribute}={target_value}",
        success=True
    )
    
    # Create notification trigger data
    import time
    notification_data = {
        "timestamp": int(time.time() * 1000),
        "type": "conditional_distribution",
        "message": f"Conditional Distribution for '{target_attribute}={target_value}' by '{compare_attribute}' added to explain context"
    }
    
    # ✅ CRITICAL FIX: Sync with global_vars for chat compatibility
    if hasattr(global_vars, 'explain_context_data'):
        global_vars.explain_context_data = updated_context.copy()
    else:
        global_vars.explain_context_data = updated_context.copy()
    print(f"[CONDITIONAL DISTRIBUTION CONTEXT] ✅ Synced global_vars.explain_context_data: {len(updated_context)} items")
    
    # Return updated context, visual feedback, and notification trigger
    # ✅ CRITICAL FIX: Sync with global_vars for chat compatibility
    if hasattr(global_vars, 'explain_context_data'):
        global_vars.explain_context_data = updated_context.copy()
    else:
        global_vars.explain_context_data = updated_context.copy()
    print(f"[TABLE METRIC CONTEXT] ✅ Synced global_vars.explain_context_data: {len(updated_context)} items")
    
    return updated_context, success_content, success_style, notification_data


def add_metric_to_chat(n_clicks, attribute_name, cell_info, current_context):
    """
    Add the selected metric to the chat context when the + button is clicked.
    This will be displayed in the chat interface as a context block similar to distribution comparisons.
    
    The function ensures data integrity by properly handling the current_context,
    maintaining all existing context items while adding the new metric item.
    
    This implementation extracts metric information directly from the modal content.
    """
    # Return no update if not triggered by a click
    if not n_clicks or n_clicks is None or not any(n_clicks):
        return dash.no_update
    
    # Generate a diagnostic log of incoming data
    ctx = callback_context
    if not ctx.triggered:
        return dash.no_update
    
    # Get the id of the button that was clicked
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Extract the metric type from the button id (which is a JSON string)
    import json
    metric_type = json.loads(button_id)["metric"]
    
    print(f"[METRIC CONTEXT] Adding metric of type {metric_type} to chat context")
    print(f"[METRIC CONTEXT] Current context type: {type(current_context)}")
    print(f"[METRIC CONTEXT] Pre-existing items: {len(current_context) if isinstance(current_context, list) else 'N/A - not a list'}")
    
    # Generate unique context ID using utility function
    context_id = generate_context_id("metric")
    print(f"[METRIC CONTEXT] Generated new item ID: {context_id}")
    
    # Extract metric name and details from the modal content
    metric_name = ""
    metric_details = ""
    interpretation = ""
    attribute_type = ""
    metric_value = ""
    metric_statistic = ""
    metric_pvalue = ""
    
    # Find the metric block in the modal that matches the clicked metric type
    # Since we don't have direct access to the metric block, we need to parse the modal content
    
    # First, extract the attribute type from the modal
    try:
        # Look for a block title or header in the modal content that contains the attribute type
        if isinstance(cell_info, list):
            for item in cell_info:
                if isinstance(item, dict) and 'props' in item and 'children' in item['props']:
                    children = item['props']['children']
                    if isinstance(children, str) and 'Type:' in children:
                        parts = children.split('Type:')
                        if len(parts) > 1:
                            attribute_type = parts[1].strip()
                            break
    except Exception as e:
        print(f"[METRIC CONTEXT] Error extracting attribute type: {str(e)}")
        attribute_type = "Unknown"
    
    # Set metric name and interpretation based on the metric type
    if metric_type == "js_divergence":
        metric_name = "Jensen-Shannon Divergence"
        try:
            # Try to find the JS divergence value in the modal
            if isinstance(cell_info, list):
                for item in cell_info:
                    if isinstance(item, dict) and 'props' in item and 'children' in item['props']:
                        children = item['props']['children']
                        if isinstance(children, str) and 'Jensen-Shannon Divergence' in children and ':' in children:
                            parts = children.split(':')
                            if len(parts) > 1:
                                metric_value = parts[1].strip()
                                break
        except Exception as e:
            print(f"[METRIC CONTEXT] Error extracting JS divergence value: {str(e)}")
            metric_value = "N/A"
        
        metric_details = f"Value: {metric_value}"
        interpretation = "Ranges from 0 (identical distributions) to 1 (completely different distributions)."
    
    elif metric_type == "chi_squared":
        metric_name = "Chi-squared Test"
        try:
            # Try to find the Chi-squared values in the modal
            if isinstance(cell_info, list):
                for item in cell_info:
                    if isinstance(item, dict) and 'props' in item and 'children' in item['props']:
                        children = item['props']['children']
                        if isinstance(children, str) and 'Test Statistic:' in children:
                            parts = children.split('Test Statistic:')
                            if len(parts) > 1 and 'p-value:' in parts[1]:
                                stat_pval = parts[1].split('p-value:')
                                metric_statistic = stat_pval[0].strip()
                                metric_pvalue = stat_pval[1].strip() if len(stat_pval) > 1 else "N/A"
                                break
        except Exception as e:
            print(f"[METRIC CONTEXT] Error extracting Chi-squared values: {str(e)}")
            metric_statistic = "N/A"
            metric_pvalue = "N/A"
        
        metric_details = f"Test Statistic: {metric_statistic}\nP-value: {metric_pvalue}"
        interpretation = "A small p-value (<0.05) indicates significant differences in distribution."
    
    elif metric_type == "psi":
        metric_name = "Population Stability Index (PSI)"
        try:
            # Try to find the PSI value in the modal
            if isinstance(cell_info, list):
                for item in cell_info:
                    if isinstance(item, dict) and 'props' in item and 'children' in item['props']:
                        children = item['props']['children']
                        if isinstance(children, str) and 'PSI' in children and ':' in children:
                            parts = children.split(':')
                            if len(parts) > 1:
                                metric_value = parts[1].strip()
                                break
        except Exception as e:
            print(f"[METRIC CONTEXT] Error extracting PSI value: {str(e)}")
            metric_value = "N/A"
            
        metric_details = f"Value: {metric_value}"
        interpretation = "Guidelines:\n<0.1: Minimal shift\n0.1 - 0.25: Moderate shift\n>0.25: Significant shift"
    
    elif metric_type == "wasserstein" or metric_type == "wasserstein_date":
        metric_name = "Wasserstein Distance"
        try:
            # Try to find the Wasserstein value in the modal
            if isinstance(cell_info, list):
                for item in cell_info:
                    if isinstance(item, dict) and 'props' in item and 'children' in item['props']:
                        children = item['props']['children']
                        if isinstance(children, str) and 'Wasserstein' in children and ':' in children:
                            parts = children.split(':')
                            if len(parts) > 1:
                                metric_value = parts[1].strip()
                                break
        except Exception as e:
            print(f"[METRIC CONTEXT] Error extracting Wasserstein value: {str(e)}")
            metric_value = "N/A"
            
        metric_details = f"Value: {metric_value}"
        interpretation = "Scale-dependent measure. Larger values indicate greater distribution differences."
    
    # elif metric_type == "ks_test":
    #     metric_name = "Kolmogorov-Smirnov Test"
    #     try:
    #         # Try to find the K-S test values in the modal
    #         if isinstance(cell_info, list):
    #             for item in cell_info:
    #                 if isinstance(item, dict) and 'props' in item and 'children' in item['props']:
    #                     children = item['props']['children']
    #                     if isinstance(children, str) and 'Test Statistic:' in children and 'K-S' in children:
    #                         parts = children.split('Test Statistic:')
    #                         if len(parts) > 1 and 'p-value:' in parts[1]:
    #                             stat_pval = parts[1].split('p-value:')
    #                             metric_statistic = stat_pval[0].strip()
    #                             metric_pvalue = stat_pval[1].strip() if len(stat_pval) > 1 else "N/A"
    #                             break
    #     except Exception as e:
    #         print(f"[METRIC CONTEXT] Error extracting K-S test values: {str(e)}")
    #         metric_statistic = "N/A"
    #         metric_pvalue = "N/A"
            
    # Extract attribute name from modal
    attr_name = ""
    if attribute_name and isinstance(attribute_name, str):
        attr_name = attribute_name
    elif attribute_name and isinstance(attribute_name, list):
        for item in attribute_name:
            if isinstance(item, str):
                attr_name += item
            elif isinstance(item, dict) and 'props' in item and 'children' in item['props']:
                if isinstance(item['props']['children'], str):
                    attr_name += item['props']['children']
    
    # Extract column name from cell info as fallback
    if not attr_name and cell_info:
        if isinstance(cell_info, list):
            cell_info_text = ""
            for item in cell_info:
                if isinstance(item, dict) and 'props' in item and 'children' in item['props']:
                    if isinstance(item['props']['children'], str):
                        cell_info_text += item['props']['children'] + "\n"
                    elif isinstance(item['props']['children'], list):
                        for child in item['props']['children']:
                            if isinstance(child, str):
                                cell_info_text += child + " "
                        cell_info_text += "\n"
            
            # Try to extract column name from cell info text
            cell_info_lines = cell_info_text.strip().split('\n')
            for line in cell_info_lines:
                if ':' in line and len(line.split(':')) > 1:
                    possible_name = line.split(':', 1)[1].strip()
                    if possible_name and len(possible_name) < 50:
                        attr_name = possible_name
                        break
    
    # If still no attribute name, use a default
    if not attr_name:
        attr_name = "Unknown attribute"
    
    print(f"[METRIC CONTEXT] Using attribute name: '{attr_name}'")
    
    # Create a summary text for GPT context
    summary_text = f"Statistical Metric: {metric_name}\n"
    summary_text += f"Attribute: {attr_name}\n"
    summary_text += f"Attribute Type: {attribute_type}\n"
    summary_text += f"{metric_details}\n"
    summary_text += f"Interpretation: {interpretation}\n"
    
    # Create the context item using utility function
    new_context = create_context_item(
        context_id=context_id,
        context_type="metric",
        summary_text=summary_text,
        metric_type=metric_type,
        attribute_name=attr_name,
        attribute_type=attribute_type,
        metric_name=metric_name,
        metric_details=metric_details,
        interpretation=interpretation,
        expanded=True  # Default expanded state for ease of viewing
    )
    
    # Add to context list using utility function
    updated_context = add_context_item_to_list(current_context, new_context)
    
    # Enhanced debug output
    print(f"[METRIC CONTEXT] After adding: {len(updated_context)} items with IDs: {[item.get('id') for item in updated_context if isinstance(item, dict)]}")
    
    # Store metric context info (will be processed when user asks questions)
    if hasattr(global_vars, 'agent') and global_vars.agent:
        global_vars.agent.add_user_action_to_history(f"User added metric context for {metric_name} on attribute '{attr_name}' to chat context")
        print(f"[METRIC CONTEXT] Context stored for later activation")
    
    # ==== DETAILED DEBUG INFO FOR CHAT CONTEXT ====
    print(f"\n{'='*60}")
    print(f"[CHAT CONTEXT DEBUG] METRIC ADDED TO CHAT BOX (FROM MODAL)")
    print(f"{'='*60}")
    print(f"Context ID: {context_id}")
    print(f"Context Type: metric")
    print(f"Metric Type: {metric_type}")
    print(f"Metric Name: {metric_name}")
    print(f"Attribute Name: {attr_name}")
    print(f"Attribute Type: {attribute_type}")
    print(f"Timestamp: {new_context['timestamp']}")
    print(f"Metric Details: {metric_details[:100]}...")
    print(f"Interpretation: {interpretation[:100]}...")
    print(f"Total Context Items After Addition: {len(updated_context)}")
    print(f"All Context IDs: {[item.get('id') for item in updated_context if isinstance(item, dict)]}")
    print(f"All Context Types: {[item.get('type') for item in updated_context if isinstance(item, dict)]}")
    print(f"What will be shared with chat:")
    print(f"  - Metric analysis for: {attr_name}")
    print(f"  - Metric type: {metric_type}")
    print(f"  - Statistical details: {'Yes' if metric_details else 'No'}")
    print(f"  - Interpretation: {'Yes' if interpretation else 'No'}")
    print(f"  - Expanded by default: {new_context['expanded']}")
    print(f"{'='*60}\n")
    
    # ✅ CRITICAL FIX: Sync with global_vars for chat compatibility
    if hasattr(global_vars, 'explain_context_data'):
        global_vars.explain_context_data = updated_context.copy()
    else:
        global_vars.explain_context_data = updated_context.copy()
    print(f"[TABLE METRIC CONTEXT] ✅ Synced global_vars.explain_context_data: {len(updated_context)} items")
    
    return updated_context


# Add callback for adding metrics to explain context
@callback(
    Output("explain-context-data", "data", allow_duplicate=True),
    [Input({"type": "add-metric-to-explain", "metric": ALL}, "n_clicks")],
    [State("metrics-modal-attribute", "children"),
     State("modal-selected-cell-info", "children"),
     State("explain-context-data", "data")],
    prevent_initial_call=True
)
def add_metric_to_explain(n_clicks, attribute_name, cell_info, current_context):
    """
    Add the selected metric to the explain context when the + button is clicked.
    This will be displayed in the explain interface as a context block.
    """
    # Return no update if not triggered by a click
    if not n_clicks or n_clicks is None or not any(n_clicks):
        return dash.no_update
    
    # Generate a diagnostic log of incoming data
    ctx = callback_context
    if not ctx.triggered:
        return dash.no_update
    
    # Get the id of the button that was clicked
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Extract the metric type from the button id (which is a JSON string)
    import json
    metric_type = json.loads(button_id)["metric"]
    
    print(f"[METRIC CONTEXT] Adding metric of type {metric_type} to explain context")
    
    # Generate unique context ID using utility function
    context_id = generate_context_id("metric")
    print(f"[METRIC CONTEXT] Generated new item ID: {context_id}")
    
    # Extract metric information (using same logic as chat version)
    metric_name = ""
    metric_details = ""
    interpretation = ""
    attribute_type = ""
    metric_value = ""
    metric_statistic = ""
    metric_pvalue = ""
    
    # Extract attribute type from modal
    try:
        if isinstance(cell_info, list):
            for item in cell_info:
                if isinstance(item, dict) and 'props' in item and 'children' in item['props']:
                    children = item['props']['children']
                    if isinstance(children, str) and 'Type:' in children:
                        parts = children.split('Type:')
                        if len(parts) > 1:
                            attribute_type = parts[1].strip()
                            break
    except Exception as e:
        print(f"[METRIC CONTEXT] Error extracting attribute type: {str(e)}")
        attribute_type = "Unknown"
    
    # Set metric information based on type (reusing logic from chat version)
    if metric_type == "js_divergence":
        metric_name = "Jensen-Shannon Divergence"
        # Extract value logic here (simplified for brevity)
        metric_details = f"Value: {metric_value if 'metric_value' in locals() else 'N/A'}"
        interpretation = "Ranges from 0 (identical distributions) to 1 (completely different distributions)."
    elif metric_type == "chi_squared":
        metric_name = "Chi-squared Test"
        metric_details = f"Test Statistic: N/A\\nP-value: N/A"
        interpretation = "A small p-value (<0.05) indicates significant differences in distribution."
    elif metric_type == "psi":
        metric_name = "Population Stability Index (PSI)"
        metric_details = f"Value: N/A"
        interpretation = "Guidelines:\\n<0.1: Minimal shift\\n0.1 - 0.25: Moderate shift\\n>0.25: Significant shift"
    elif metric_type == "wasserstein" or metric_type == "wasserstein_date":
        metric_name = "Wasserstein Distance"
        metric_details = f"Value: N/A"
        interpretation = "Scale-dependent measure. Larger values indicate greater distribution differences."
    # elif metric_type == "ks_test":
    #     metric_name = "Kolmogorov-Smirnov Test"
    #     metric_details = f"Test Statistic: N/A\\nP-value: N/A"
    #     interpretation = "Test statistic ranges from 0 to 1, with higher values indicating greater differences."
    
    # Extract attribute name from modal
    attr_name = "Unknown attribute"
    if attribute_name and isinstance(attribute_name, str):
        attr_name = attribute_name
    elif attribute_name and isinstance(attribute_name, list):
        for item in attribute_name:
            if isinstance(item, str):
                attr_name += item
            elif isinstance(item, dict) and 'props' in item and 'children' in item['props']:
                if isinstance(item['props']['children'], str):
                    attr_name += item['props']['children']
    
    print(f"[METRIC CONTEXT] Using attribute name: '{attr_name}'")
    
    # Create summary text for explain context
    summary_text = f"Statistical Metric: {metric_name}\\n"
    summary_text += f"Attribute: {attr_name}\\n"
    summary_text += f"Attribute Type: {attribute_type}\\n"
    summary_text += f"{metric_details}\\n"
    summary_text += f"Interpretation: {interpretation}\\n"
    
    # Create the context item using utility function
    new_context = create_context_item(
        context_id=context_id,
        context_type="metric",
        summary_text=summary_text,
        metric_type=metric_type,
        attribute_name=attr_name,
        attribute_type=attribute_type,
        metric_name=metric_name,
        metric_details=metric_details,
        interpretation=interpretation,
        expanded=True  # Default expanded state for ease of viewing
    )
    
    # Add to context list using utility function
    updated_context = add_context_item_to_list(current_context, new_context)
    
    # Store metric context info for explain usage
    if hasattr(global_vars, 'agent') and global_vars.agent:
        global_vars.agent.add_user_action_to_history(f"User added metric context for {metric_name} on attribute '{attr_name}' to explain context")
        print(f"[METRIC CONTEXT] Context stored for explain analysis")
    
    print(f"[METRIC CONTEXT] Successfully added metric {metric_name} to explain context with ID: {context_id}")
    print(f"[METRIC CONTEXT] Total explain context items after addition: {len(updated_context)}")
    
    # ✅ CRITICAL FIX: Sync with global_vars for chat compatibility
    if hasattr(global_vars, 'explain_context_data'):
        global_vars.explain_context_data = updated_context.copy()
    else:
        global_vars.explain_context_data = updated_context.copy()
    print(f"[METRIC CONTEXT] ✅ Synced global_vars.explain_context_data: {len(updated_context)} items")
    
    # ✅ CRITICAL FIX: Sync with global_vars for chat compatibility
    if hasattr(global_vars, 'explain_context_data'):
        global_vars.explain_context_data = updated_context.copy()
    else:
        global_vars.explain_context_data = updated_context.copy()
    print(f"[TABLE METRIC CONTEXT] ✅ Synced global_vars.explain_context_data: {len(updated_context)} items")
    
    return updated_context


# Add a callback to handle clicking the + button in the metrics table
@callback(
    Output("chat-context-data", "data", allow_duplicate=True),
    [Input("metrics-table", "active_cell")],
    [State("metrics-table", "data"),
     State("chat-context-data", "data")],
    prevent_initial_call=True
)
def add_table_metric_to_chat(active_cell, table_data, current_context):
    """
    Add a metric from the metrics table to the chat context when the + button is clicked.
    This allows users to quickly add metrics from the main table view without opening the explanation modal.
    """
    if not active_cell:
        return dash.no_update
    
    # Check if the clicked cell is in the AddToChat column
    if active_cell['column_id'] != 'AddToChat':
        return dash.no_update
    
    # Get the row data that was clicked
    row_idx = active_cell['row']
    row_data = table_data[row_idx]
    
    print(f"[TABLE METRIC CONTEXT] Adding metric from table row {row_idx} to context")
    print(f"[TABLE METRIC CONTEXT] Row data: {row_data}")
    
    # Extract attribute, type and target relevance information
    attribute_name = row_data.get('Attribute', 'Unknown')
    attribute_type = row_data.get('Type', 'Unknown')
    primary_target_relevance = row_data.get('PrimaryTargetRelevance', 'Unknown')
    secondary_target_relevance = row_data.get('SecondaryTargetRelevance', 'Unknown')
    relevance_delta = row_data.get('RelevanceDelta', 'Unknown')
    
    # Generate unique context ID using utility function
    context_id = generate_context_id("metric")
    
    # Collect dataset metadata
    def collect_dataset_metadata():
        """collect dataset metadata"""
        metadata = {}
        
        try:
            if hasattr(global_vars, 'df') and global_vars.df is not None:
                primary_df = global_vars.df
                metadata['primary_shape'] = primary_df.shape
                metadata['primary_rows'] = len(primary_df)
                metadata['primary_columns'] = len(primary_df.columns)
                metadata['primary_missing_values'] = primary_df.isnull().sum().sum()
                metadata['primary_missing_percentage'] = round((metadata['primary_missing_values'] / (metadata['primary_rows'] * metadata['primary_columns'])) * 100, 2)
            
            if hasattr(global_vars, 'secondary_df') and global_vars.secondary_df is not None:
                secondary_df = global_vars.secondary_df
                metadata['secondary_shape'] = secondary_df.shape
                metadata['secondary_rows'] = len(secondary_df)
                metadata['secondary_columns'] = len(secondary_df.columns)
                metadata['secondary_missing_values'] = secondary_df.isnull().sum().sum()
                metadata['secondary_missing_percentage'] = round((metadata['secondary_missing_values'] / (metadata['secondary_rows'] * metadata['secondary_columns'])) * 100, 2)
            
            # calculate common columns between two datasets
            if hasattr(global_vars, 'df') and hasattr(global_vars, 'secondary_df'):
                if global_vars.df is not None and global_vars.secondary_df is not None:
                    common_columns = set(global_vars.df.columns) & set(global_vars.secondary_df.columns)
                    metadata['common_columns_count'] = len(common_columns)
                    metadata['common_columns'] = list(common_columns)
            
            # target variable information
            if hasattr(global_vars, 'target_attribute') and global_vars.target_attribute:
                metadata['target_attribute'] = global_vars.target_attribute
        
        except Exception as e:
            print(f"[METADATA] Error collecting dataset metadata: {str(e)}")
        
        return metadata
    
    # collect all metrics dynamically
    def collect_all_metrics_dynamically(row_data):
        """collect all available drift detection metrics"""
        metric_details = ""
        
        # skip non-metric fields
        skip_fields = ['Attribute', 'Type', 'AddToChat', 'AddToExplain', 'ExplainAction']
        
        # metric name mapping and grouping
        metric_groups = {
            # 'Target Relevance': {
            #     'PrimaryTargetRelevance': 'Target Relevance (Primary)',
            #     'SecondaryTargetRelevance': 'Target Relevance (Secondary)', 
            #     'RelevanceDelta': 'Target Relevance Change',
            #     'TargetRelevance': 'Target Relevance Category',
            #     'TargetRelevanceScore': 'Target Relevance Score'
            # },
            'Distance-based Metrics': {
                'JS_Divergence': 'Jensen-Shannon Divergence',
                'KL_Divergence': 'Kullback-Leibler Divergence',
                'Wasserstein': 'Wasserstein Distance',
                'Hellinger': 'Hellinger Distance',
                'TVD': 'Total Variation Distance',
                'Energy_Distance': 'Energy Distance'
            },
            'Statistical Tests': {
                # 'KS_Test': 'Kolmogorov-Smirnov Test',
                'Anderson_Darling': 'Anderson-Darling Test',
                'Cramer_von_Mises': 'Cramer-von-Mises Test',
                'T_Test': 'T-Test',
                'Z_Test': 'Z-Test',
                'Fisher_Exact': 'Fisher Exact Test',
                'Chi_Square': 'Chi-Square Test',
                'G_Test': 'G-Test',
                # 'Test_Statistic': 'Test Statistic',
                'p_value': 'p-value'
            },
            'Stability Metrics': {
                'PSI': 'Population Stability Index (PSI)',
                'E_Squared': 'E-Squared Test',
                'Empirical_MMD': 'Empirical Maximum Mean Discrepancy'
            }
        }
        
        # collect metrics by group
        for group_name, metrics in metric_groups.items():
            group_metrics = []
            for key, display_name in metrics.items():
                value = row_data.get(key, 'N/A')
                if value != 'N/A' and value is not None:
                    group_metrics.append(f"  {display_name}: {value}")
            
            if group_metrics:
                metric_details += f"\n{group_name}:\n"
                metric_details += "\n".join(group_metrics) + "\n"
        
        # collect other metrics
        other_metrics = []
        all_grouped_keys = set()
        for metrics in metric_groups.values():
            all_grouped_keys.update(metrics.keys())
        
        for key, value in row_data.items():
            if key not in skip_fields and key not in all_grouped_keys and value != 'N/A' and value is not None:
                display_name = key.replace('_', ' ').title()
                other_metrics.append(f"  {display_name}: {value}")
        
        if other_metrics:
            metric_details += f"\nOther Metrics:\n"
            metric_details += "\n".join(other_metrics) + "\n"
        
        return metric_details
    
    # collect dataset metadata
    metadata = collect_dataset_metadata()
    
    # collect all metrics
    metric_details = collect_all_metrics_dynamically(row_data)
    
    # format dataset metadata to readable text
    def format_metadata_summary(metadata):
        # summary = "\n=== Dataset Metadata ===\n"
        
        # if 'primary_shape' in metadata:
        #     summary += f"Primary Dataset: {metadata['primary_rows']} rows × {metadata['primary_columns']} columns\n"
        #     summary += f"  - Missing values: {metadata['primary_missing_values']} ({metadata['primary_missing_percentage']}%)\n"
        
        # if 'secondary_shape' in metadata:
        #     summary += f"Secondary Dataset: {metadata['secondary_rows']} rows × {metadata['secondary_columns']} columns\n"
        #     summary += f"  - Missing values: {metadata['secondary_missing_values']} ({metadata['secondary_missing_percentage']}%)\n"
        
        # Add resampling information dynamically
        try:
            from drift.detect import resample_datasets, encode_categorical_features
            
            # Get original datasets
            primary_df = global_vars.df
            secondary_df = global_vars.secondary_df
            
            if primary_df is not None and secondary_df is not None:
                # Calculate what the processing would look like
                primary_encoded, secondary_encoded = encode_categorical_features(primary_df, secondary_df)
                primary_resampled, secondary_resampled = resample_datasets(primary_encoded, secondary_encoded)
                
                # Get common columns
                common_columns = list(set(primary_df.columns) & set(secondary_df.columns))
                
                summary = f"\n=== Processing Information ===\n"
                summary += f"Original Data Sizes:\n"
                summary += f"  - Primary: {len(primary_df)} rows\n"
                summary += f"  - Secondary: {len(secondary_df)} rows\n"
                summary += f"Resampled for Analysis:\n"
                summary += f"  - Primary (used): {len(primary_resampled)} rows\n"
                summary += f"  - Secondary (used): {len(secondary_resampled)} rows\n"
                summary += f"Column Analysis:\n"
                summary += f"  - Common columns analyzed: {len(common_columns)}\n"
                summary += f"  - Total primary columns: {len(primary_df.columns)}\n"
                summary += f"  - Total secondary columns: {len(secondary_df.columns)}\n"
        except Exception as e:
            print(f"[DEBUG] Error calculating resampling info: {str(e)}")
            # Fallback to basic info if available
            if 'common_columns_count' in metadata:
                summary += f"Common Attributes: {metadata['common_columns_count']} shared columns\n"
        
        if 'target_attribute' in metadata:
            target_attr = metadata['target_attribute']
            if isinstance(target_attr, dict):
                target_name = target_attr.get('name', str(target_attr))
            else:
                target_name = str(target_attr)
            summary += f"Target Variable: {target_name}\n"
        
        return summary
    
    # Create interpretation based on the attribute type
    interpretation = f"Statistical drift detection analysis for attribute '{attribute_name}' of type {attribute_type}.\n\n"
    
    # add dataset metadata
    if metadata:
        interpretation += format_metadata_summary(metadata)
    
    
    # Create the comprehensive summary text for GPT context
    summary_text = f"=== Statistical Drift Detection Report ===\n"
    summary_text += f"Attribute: {attribute_name} (Type: {attribute_type})\n"
    
    if metadata:
        summary_text += format_metadata_summary(metadata)
    
    summary_text += f"\n=== Drift Detection Metrics ===\n{metric_details}"
    
    # Create the context item using utility function
    new_context = create_context_item(
        context_id=context_id,
        context_type="drift_analysis",
        summary_text=summary_text,
        attribute_name=attribute_name,
        attribute_type=attribute_type,
        metric_name=f"Drift Analysis: {attribute_name}",
        metric_details=metric_details,
        dataset_metadata=metadata,
        interpretation=interpretation,
        expanded=True,  # Default expanded state for ease of viewing
        # Additional fields for better organization
        analysis_scope="attribute_level",
        metrics_count=len([k for k, v in row_data.items() if k not in ['Attribute', 'Type', 'AddToContext', 'ExplainAction'] and v != 'N/A']),
        has_target_relevance=any(key.startswith(('Target', 'Primary', 'Secondary')) for key in row_data.keys())
    )
    
    # Add to context list using utility function
    updated_context = add_context_item_to_list(current_context, new_context)
    
    # Add context to global vars so the agent can use it
    if hasattr(global_vars, 'agent') and global_vars.agent:
        # create more detailed agent history
        agent_summary = f"User added comprehensive drift analysis for attribute '{attribute_name}' ({attribute_type}):\n"
        agent_summary += f"- Dataset context: Primary ({metadata.get('primary_rows', 'N/A')} rows) vs Secondary ({metadata.get('secondary_rows', 'N/A')} rows)\n"
        agent_summary += f"- Available metrics: {new_context['metrics_count']} drift detection measures\n"
        # agent_summary += f"- Target relevance analysis: {'Included' if new_context['has_target_relevance'] else 'Not available'}\n"
        agent_summary += f"- Full analysis: {summary_text[:200]}..."  # truncate the text to 200 characters to avoid too long
        
        global_vars.agent.add_user_action_to_history(f"User added comprehensive drift analysis for attribute '{attribute_name}' to chat context")
        print(f"[TABLE METRIC CONTEXT] Drift analysis context stored for {attribute_name}")
    
    # ==== DETAILED DEBUG INFO FOR CHAT CONTEXT ====
    print(f"\n{'='*60}")
    print(f"[CHAT CONTEXT DEBUG] DRIFT ANALYSIS ADDED TO CHAT BOX (FROM TABLE)")
    print(f"{'='*60}")
    print(f"Context ID: {context_id}")
    print(f"Context Type: drift_analysis")
    print(f"Attribute Name: {attribute_name}")
    print(f"Attribute Type: {attribute_type}")
    print(f"Timestamp: {new_context['timestamp']}")
    print(f"Metrics Count: {new_context['metrics_count']}")
    # print(f"Has Target Relevance: {new_context['has_target_relevance']}")
    print(f"Analysis Scope: {new_context['analysis_scope']}")
    print(f"Dataset Metadata:")
    if metadata:
        print(f"  - Primary Dataset: {metadata.get('primary_rows', 'N/A')} rows × {metadata.get('primary_columns', 'N/A')} cols")
        print(f"  - Secondary Dataset: {metadata.get('secondary_rows', 'N/A')} rows × {metadata.get('secondary_columns', 'N/A')} cols")
        print(f"  - Common Columns: {metadata.get('common_columns_count', 'N/A')}")
        print(f"  - Target Attribute: {metadata.get('target_attribute', 'N/A')}")
    print(f"Metric Details Preview: {metric_details[:200]}...")
    print(f"Total Context Items After Addition: {len(updated_context)}")
    print(f"All Context IDs: {[item.get('id') for item in updated_context if isinstance(item, dict)]}")
    print(f"All Context Types: {[item.get('type') for item in updated_context if isinstance(item, dict)]}")
    print(f"What will be shared with chat:")
    print(f"  - Comprehensive drift analysis for: {attribute_name}")
    print(f"  - Dataset comparison metadata: {'Yes' if metadata else 'No'}")
    print(f"  - Multiple drift metrics: {new_context['metrics_count']} metrics")
    # print(f"  - Target relevance analysis: {'Yes' if new_context['has_target_relevance'] else 'No'}")
    print(f"  - Statistical interpretation: {'Yes' if interpretation else 'No'}")
    print(f"  - Expanded by default: {new_context['expanded']}")
    print(f"{'='*60}\n")
    
    # ✅ CRITICAL FIX: Sync with global_vars for chat compatibility
    if hasattr(global_vars, 'explain_context_data'):
        global_vars.explain_context_data = updated_context.copy()
    else:
        global_vars.explain_context_data = updated_context.copy()
    print(f"[TABLE METRIC CONTEXT] ✅ Synced global_vars.explain_context_data: {len(updated_context)} items")
    
    return updated_context


# Add a callback to handle clicking the explain button in the metrics table
@callback(
    Output("explain-context-data", "data", allow_duplicate=True),
    [Input("metrics-table", "active_cell")],
    [State("metrics-table", "data"),
     State("explain-context-data", "data")],
    prevent_initial_call=True
)
def add_table_metric_to_explain(active_cell, table_data, current_context):
    """
    Add a metric from the metrics table to the explain context when the 📊 button is clicked.
    This allows users to quickly add metrics from the main table view to the explain context.
    """
    if not active_cell:
        return dash.no_update
    
    # Check if the clicked cell is in the AddToExplain column
    if active_cell['column_id'] != 'AddToExplain':
        return dash.no_update
    
    # Get the row data that was clicked
    row_idx = active_cell['row']
    row_data = table_data[row_idx]
    
    print(f"[TABLE METRIC EXPLAIN] Adding metric from table row {row_idx} to explain context")
    print(f"[TABLE METRIC EXPLAIN] Row data: {row_data}")
    
    # Extract attribute information
    attribute_name = row_data.get('Attribute', 'Unknown')
    attribute_type = row_data.get('Type', 'Unknown')
    
    # Generate unique context ID using utility function
    context_id = generate_context_id("drift_analysis")
    
    # ==============================================
    # COLLECT DATASET METADATA (same as chat version)
    # ==============================================
    def collect_dataset_metadata():
        """Collect metadata about the datasets being compared."""
        metadata = {}
        
        try:
            # Get primary and secondary dataset info from global vars
            if hasattr(global_vars, 'primary_data') and global_vars.primary_data is not None:
                primary_df = global_vars.primary_data
                metadata['primary_rows'] = len(primary_df)
                metadata['primary_columns'] = len(primary_df.columns)
                metadata['primary_missing_values'] = primary_df.isnull().sum().sum()
                metadata['primary_missing_percentage'] = round((metadata['primary_missing_values'] / (metadata['primary_rows'] * metadata['primary_columns'])) * 100, 2)
                
            if hasattr(global_vars, 'secondary_data') and global_vars.secondary_data is not None:
                secondary_df = global_vars.secondary_data
                metadata['secondary_rows'] = len(secondary_df)
                metadata['secondary_columns'] = len(secondary_df.columns)
                metadata['secondary_missing_values'] = secondary_df.isnull().sum().sum()
                metadata['secondary_missing_percentage'] = round((metadata['secondary_missing_values'] / (metadata['secondary_rows'] * metadata['secondary_columns'])) * 100, 2)
            
            # Get common columns
            if hasattr(global_vars, 'primary_data') and hasattr(global_vars, 'secondary_data') and global_vars.primary_data is not None and global_vars.secondary_data is not None:
                common_cols = set(global_vars.primary_data.columns) & set(global_vars.secondary_data.columns)
                metadata['common_columns_count'] = len(common_cols)
            
            # Get target attribute info
            if hasattr(global_vars, 'current_target_attribute') and global_vars.current_target_attribute:
                metadata['target_attribute'] = global_vars.current_target_attribute
            
        except Exception as e:
            print(f"[TABLE METRIC EXPLAIN] Warning: Could not collect metadata: {str(e)}")
        
        return metadata
    
    # ==============================================
    # COLLECT ALL METRICS DYNAMICALLY (same as chat version)
    # ==============================================
    def collect_all_metrics_dynamically(row_data):
        """Dynamically collect all available metrics from the row data."""
        metrics_collected = {}
        
        # skip non-metric fields
        skip_fields = ['Attribute', 'Type', 'AddToChat', 'AddToExplain', 'ExplainAction']
        
        for key, value in row_data.items():
            if key not in skip_fields and value != 'N/A' and value is not None:
                try:
                    # Try to convert to float for numerical metrics
                    if isinstance(value, str) and value.replace('.', '').replace('-', '').replace('e', '').replace('+', '').isdigit():
                        metrics_collected[key] = float(value)
                    else:
                        metrics_collected[key] = value
                except (ValueError, TypeError):
                    metrics_collected[key] = value
        
        return metrics_collected
    
    # Collect data
    metadata = collect_dataset_metadata()
    metric_details = collect_all_metrics_dynamically(row_data)
    
    # ==============================================
    # CREATE COMPREHENSIVE SUMMARY (same as chat version)
    # ==============================================
    
    summary_text = f"Statistical drift detection analysis for attribute '{attribute_name}' of type {attribute_type}.\n\n"
    
    # Add processing information
    summary_text += "=== Processing Information ===\n"
    summary_text += "Original Data Sizes:\n"
    if metadata.get('primary_rows'):
        summary_text += f"  - Primary: {metadata['primary_rows']} rows\n"
    if metadata.get('secondary_rows'):
        summary_text += f"  - Secondary: {metadata['secondary_rows']} rows\n"
    
    # Only add resampling info if we have small secondary data (< 30 rows) - common scenario
    if metadata.get('secondary_rows') and metadata.get('secondary_rows') < 30:
        summary_text += "Resampled for Analysis:\n"
        summary_text += f"  - Primary (used): {metadata.get('secondary_rows', 'N/A')} rows\n"
        summary_text += f"  - Secondary (used): {metadata.get('secondary_rows', 'N/A')} rows\n"
    
    summary_text += "Column Analysis:\n"
    if metadata.get('common_columns_count'):
        summary_text += f"  - Common columns analyzed: {metadata['common_columns_count']}\n"
    if metadata.get('primary_columns'):
        summary_text += f"  - Total primary columns: {metadata['primary_columns']}\n"
    if metadata.get('secondary_columns'):
        summary_text += f"  - Total secondary columns: {metadata['secondary_columns']}\n"
    if metadata.get('target_attribute'):
        summary_text += f"Target Variable: {metadata['target_attribute']}\n"
    
    summary_text += "\n"
    
    
    target_relevance_metrics = []
    
    # Add distance-based metrics
    distance_metrics = ['JS_Divergence', 'KL_Divergence', 'Hellinger_Distance', 'Total_Variation_Distance', 'Wasserstein']
    distance_info = {k: v for k, v in metric_details.items() if k in distance_metrics and v != 'N/A'}
    
    if distance_info:
        summary_text += "Distance-based Metrics:\n"
        for metric, value in distance_info.items():
            display_name = metric.replace('_', '-').replace('JS', 'Jensen-Shannon').replace('KL', 'Kullback-Leibler')
            summary_text += f"  {display_name}: {value}\n"
        summary_text += "\n"
    
    # Add statistical tests
    test_metrics = ['Z_Test', 'Fisher_Exact', 'Chi_Square']
    test_info = {k: v for k, v in metric_details.items() if k in test_metrics and v != 'N/A'}
    
    if test_info:
        summary_text += "Statistical Tests:\n"
        for metric, value in test_info.items():
            display_name = metric.replace('_', '-').replace('KS', 'Kolmogorov-Smirnov').replace('Z', 'Z-').replace('Chi', 'Chi-squared')
            summary_text += f"  {display_name}: {value}\n"
        summary_text += "\n"
    
    # Add stability metrics
    stability_metrics = ['PSI']
    stability_info = {k: v for k, v in metric_details.items() if k in stability_metrics and v != 'N/A'}
    
    if stability_info:
        summary_text += "Stability Metrics:\n"
        for metric, value in stability_info.items():
            display_name = "Population Stability Index (PSI)" if metric == 'PSI' else metric.replace('_', ' ')
            summary_text += f"  {display_name}: {value}\n"
        summary_text += "\n"
    
    # Add any other metrics
    other_metrics = {k: v for k, v in metric_details.items() 
                   if k not in target_relevance_metrics + distance_metrics + test_metrics + stability_metrics 
                   and v != 'N/A'}
    
    if other_metrics:
        summary_text += "Other Metrics:\n"
        for metric, value in other_metrics.items():
            display_name = metric.replace('_', ' ')
            summary_text += f"  {display_name}: {value}\n"
    
    # Create interpretation
    interpretation = f"Statistical drift detection analysis for attribute '{attribute_name}' of type {attribute_type}."
    
    # Create context item using utility function (matching chat version structure)
    new_context = create_context_item(
        context_id=context_id,
        context_type="drift_analysis",
        summary_text=summary_text,
        attribute_name=attribute_name,
        attribute_type=attribute_type,
        metric_name=f"Drift Analysis: {attribute_name}",
        metric_details=str(metric_details),
        dataset_metadata=metadata,
        interpretation=interpretation,
        expanded=False,  # Default collapsed for explain context
        # Additional fields for better organization
        analysis_scope="attribute_level",
        metrics_count=len([k for k, v in row_data.items() if k not in ['Attribute', 'Type', 'AddToChat', 'AddToExplain', 'ExplainAction'] and v != 'N/A']),
        has_target_relevance=any(key.startswith(('Target', 'Primary', 'Secondary')) for key in row_data.keys())
    )
    
    # Add to context list using utility function
    updated_context = add_context_item_to_list(current_context, new_context)
    
    # Store metric context info for explain usage
    if hasattr(global_vars, 'agent') and global_vars.agent:
        # create more detailed agent history
        agent_summary = f"User added comprehensive drift analysis for attribute '{attribute_name}' ({attribute_type}) to explain context:\n"
        agent_summary += f"- Dataset context: Primary ({metadata.get('primary_rows', 'N/A')} rows) vs Secondary ({metadata.get('secondary_rows', 'N/A')} rows)\n"
        agent_summary += f"- Available metrics: {new_context['metrics_count']} drift detection measures\n"
        agent_summary += f"- Target relevance analysis: {'Included' if new_context['has_target_relevance'] else 'Not available'}\n"
        agent_summary += f"- Full analysis: {summary_text[:200]}..."  # truncate the text to 200 characters to avoid too long
        
        global_vars.agent.add_user_action_to_history(f"User added comprehensive drift analysis for attribute '{attribute_name}' to explain context from table")
        print(f"[METRIC EXPLAIN] Context stored for explain analysis")
    
    # ==== DETAILED DEBUG INFO FOR EXPLAIN CONTEXT ====
    print(f"\n{'='*60}")
    print(f"[EXPLAIN CONTEXT DEBUG] DRIFT ANALYSIS ADDED TO EXPLAIN (FROM TABLE)")
    print(f"{'='*60}")
    print(f"Context ID: {context_id}")
    print(f"Context Type: drift_analysis")
    print(f"Attribute Name: {attribute_name}")
    print(f"Attribute Type: {attribute_type}")
    print(f"Timestamp: {new_context['timestamp']}")
    print(f"Metrics Count: {new_context['metrics_count']}")
    print(f"Has Target Relevance: {new_context['has_target_relevance']}")
    print(f"Analysis Scope: {new_context['analysis_scope']}")
    print(f"Dataset Metadata:")
    if metadata:
        print(f"  - Primary Dataset: {metadata.get('primary_rows', 'N/A')} rows × {metadata.get('primary_columns', 'N/A')} cols")
        print(f"  - Secondary Dataset: {metadata.get('secondary_rows', 'N/A')} rows × {metadata.get('secondary_columns', 'N/A')} cols")
        print(f"  - Common Columns: {metadata.get('common_columns_count', 'N/A')}")
        print(f"  - Target Attribute: {metadata.get('target_attribute', 'N/A')}")
    print(f"Metric Details Preview: {str(metric_details)[:200]}...")
    print(f"Total Context Items After Addition: {len(updated_context)}")
    print(f"All Context IDs: {[item.get('id') for item in updated_context if isinstance(item, dict)]}")
    print(f"All Context Types: {[item.get('type') for item in updated_context if isinstance(item, dict)]}")
    print(f"What will be shared with explain analysis:")
    print(f"  - Comprehensive drift analysis for: {attribute_name}")
    print(f"  - Dataset comparison metadata: {'Yes' if metadata else 'No'}")
    print(f"  - Multiple drift metrics: {new_context['metrics_count']} metrics")
    print(f"  - Target relevance analysis: {'Yes' if new_context['has_target_relevance'] else 'No'}")
    print(f"  - Statistical interpretation: {'Yes' if interpretation else 'No'}")
    print(f"  - Expanded by default: {new_context['expanded']}")
    print(f"{'='*60}\n")
    
    # ✅ CRITICAL FIX: Sync with global_vars for chat compatibility
    if hasattr(global_vars, 'explain_context_data'):
        global_vars.explain_context_data = updated_context.copy()
    else:
        global_vars.explain_context_data = updated_context.copy()
    print(f"[TABLE METRIC CONTEXT] ✅ Synced global_vars.explain_context_data: {len(updated_context)} items")
    
    return updated_context


# Notification callback for content added to chat context
@callback(
    [Output("context-notification", "is_open"),
     Output("context-notification", "header"),
     Output("context-notification", "children")],
    [Input("chat-context-data", "data"),
     Input("notification-trigger", "data")],
    prevent_initial_call=True
)
def show_context_added_notification(context_data, notification_data):
    """
    Notification callback for context added to chat, now disabled per user request.
    Always returns False for is_open to prevent notifications from showing.
    
    Args:
        context_data: The current chat context data
        notification_data: Notification trigger data
        
    Returns:
        tuple: (is_open=False, header, message) to never show the notification
    """
    # Always return False for is_open to prevent any notifications
    return False, "", ""

# Add timer callbacks to reset button appearance after success feedback
@callback(
    [Output("add-target-dist-to-chat", "children", allow_duplicate=True),
     Output("add-target-dist-to-chat", "style", allow_duplicate=True)],
    [Input("add-target-dist-to-chat", "n_clicks")],
    prevent_initial_call=True
)
def reset_target_dist_button_after_delay(n_clicks):
    """
    Reset the target distribution add button appearance after a short delay.
    Provides a better user experience by showing temporary success feedback.
    """
    if not n_clicks:
        return dash.no_update, dash.no_update
    
    # Add a delay to simulate the button staying green for a moment
    import time
    time.sleep(1.5)
    
    # Reset the button to its original appearance
    original_content = [
        html.I(className="fas fa-plus me-1"),
        "Add to chat"
    ]
    
    original_style = {
        "backgroundColor": "transparent",
        "color": "#516395",
        "border": "1px solid #516395",
        "borderRadius": "3px",
        "padding": "0 8px",
        "height": "28px"
    }
    
    return original_content, original_style

@callback(
    [Output("add-cond-dist-to-chat", "children", allow_duplicate=True),
     Output("add-cond-dist-to-chat", "style", allow_duplicate=True)],
    [Input("add-cond-dist-to-chat", "n_clicks")],
    prevent_initial_call=True
)
def reset_cond_dist_button_after_delay(n_clicks):
    """
    Reset the conditional distribution add button appearance after a short delay.
    Provides a better user experience by showing temporary success feedback.
    """
    if not n_clicks:
        return dash.no_update, dash.no_update
    
    # Add a delay to simulate the button staying green for a moment
    import time
    time.sleep(1.5)
    
    # Reset the button to its original appearance
    original_content = [
        html.I(className="fas fa-plus me-1"),
        "Add to chat"
    ]
    
    original_style = {
        "backgroundColor": "transparent",
        "color": "#516395",
        "border": "1px solid #516395",
        "borderRadius": "3px",
        "padding": "0 8px",
        "height": "28px",
        "opacity": 1
    }
    
    return original_content, original_style

@callback(
    [Output("add-dataset-dist-to-chat", "children", allow_duplicate=True),
     Output("add-dataset-dist-to-chat", "style", allow_duplicate=True)],
    [Input("add-dataset-dist-to-chat", "n_clicks")],
    prevent_initial_call=True
)
def reset_dataset_dist_button_after_delay(n_clicks):
    """
    Reset the dataset distribution add button appearance after a short delay.
    Provides a better user experience by showing temporary success feedback.
    """
    if not n_clicks:
        return dash.no_update, dash.no_update
    
    # Add a delay to simulate the button staying green for a moment
    import time
    time.sleep(1.5)
    
    # Reset the button to its original appearance
    original_content = [
        html.I(className="fas fa-plus me-1"),
        "Add dataset dist"
    ]
    
    original_style = {
        "backgroundColor": "transparent",
        "color": "#516395",
        "border": "1px solid #516395",
        "borderRadius": "3px",
        "padding": "0 8px",
        "height": "28px",
        "opacity": 1
    }
    
    return original_content, original_style


# Add timer callbacks to reset explain button appearance after success feedback
@callback(
    [Output("add-target-dist-to-explain", "children", allow_duplicate=True),
     Output("add-target-dist-to-explain", "style", allow_duplicate=True)],
    [Input("add-target-dist-to-explain", "n_clicks")],
    prevent_initial_call=True
)
def reset_target_dist_explain_button_after_delay(n_clicks):
    """
    Reset the target distribution explain button appearance after a short delay.
    Provides a better user experience by showing temporary success feedback.
    """
    if not n_clicks:
        return dash.no_update, dash.no_update
    
    # Add a delay to simulate the button staying green for a moment
    import time
    time.sleep(1.5)
    
    # Reset the button to its original appearance using utility function
    original_content, original_style = create_original_button_content("explain")
    
    return original_content, original_style


# Removed reset callback to prevent duplicate context additions
# The button reset callback was causing the add_conditional_distribution_to_explain 
# callback to be triggered multiple times, resulting in duplicate context items.
# Button styling can be handled via CSS transitions or clientside callbacks if needed.


# Add a separate callback to close the modal after adding to context
@callback(
    Output("distribution-modal", "is_open", allow_duplicate=True),
    [Input("add-distribution-to-chat", "n_clicks")],
    prevent_initial_call=True
)
# def close_modal_after_adding_context(n_clicks):
#     """
#     Close the distribution modal after adding content to chat context.
#     """
#     if n_clicks:
#         return False
#     return is_open

# Add callback to update the chat context display
@callback(
    [Output("chat-context-area", "children"),
     Output("chat-context-area", "style")],
    [Input("chat-context-data", "data")],
    prevent_initial_call=True
)
def update_chat_context_display(context_data):
    """
    Update the chat context display with the current context items.
    Shows a formatted display of distribution comparison data that has been added to the chat.
    
    This function carefully processes each context item and ensures all are properly rendered
    in the chat interface, with appropriate error handling and data validation.
    """
    try:
        # Validate input data with detailed diagnostics
        if context_data is None:
            print("[DISTRIBUTION VIEW] Warning: context_data is None")
            return None, {'display': 'none'}
            
        if not isinstance(context_data, list):
            print(f"[DISTRIBUTION VIEW] Warning: context_data is not a list but {type(context_data)}")
            # Try to recover by converting to list if possible
            if context_data:
                context_data = [context_data]
                print("[DISTRIBUTION VIEW] Attempted recovery by converting to single-item list")
            else:
                return None, {'display': 'none'}
            
        if len(context_data) == 0:
            print("[DISTRIBUTION VIEW] No context items to display")
            return None, {'display': 'none'}
        
        # Enhanced debug output with detailed item information
        print(f"[DISTRIBUTION VIEW] Preparing to display {len(context_data)} context items:")
        for i, item in enumerate(context_data):
            if isinstance(item, dict):
                print(f"[DISTRIBUTION VIEW]   Item {i}: id={item.get('id', 'unknown')}, type={item.get('type', 'unknown')}")
            else:
                print(f"[DISTRIBUTION VIEW]   Item {i}: Invalid type: {type(item)}")
                
        # Critical validation and error recovery
        valid_items = []
        for item in context_data:
            if isinstance(item, dict) and 'id' in item and 'type' in item:
                valid_items.append(item)
            elif isinstance(item, dict):
                print(f"[DISTRIBUTION VIEW] Found incomplete item: {item.keys()}")
                
        if len(valid_items) < len(context_data):
            print(f"[DISTRIBUTION VIEW] Warning: Found {len(context_data) - len(valid_items)} invalid items in context_data")
            context_data = valid_items
        
        context_items = []
        
        for item in context_data:
            # Create toggle and remove buttons for all context types
            toggle_button = html.Button(
                html.I(className="fas fa-eye" if not item.get("expanded", False) else "fas fa-eye-slash"),
                id={"type": "toggle-context", "index": item["id"]},
                className="toggle-context-btn",
                style={
                    "background": "none",
                    "border": "none",
                    "color": "#516395",
                    "cursor": "pointer",
                    "padding": "0 5px",
                    "fontSize": "14px"
                }
            )
            
            remove_button = html.Button(
                html.I(className="fas fa-times"),
                id={"type": "remove-context", "index": item["id"]},
                className="remove-context-btn",
                style={
                    "background": "none",
                    "border": "none",
                    "color": "#dc3545",
                    "cursor": "pointer",
                    "padding": "0 5px",
                    "fontSize": "14px"
                }
            )
            
            # Process different context types
            if item["type"] == "target_distribution":
                # Create header with target attribute information
                target_attribute = item.get("target_attribute", "Unknown attribute")
                
                # Item content structure
                item_content = html.Div([
                    # Header - always visible
                    html.Div([
                        # Left information
                        html.Div([
                            html.I(className="fas fa-chart-bar", style={"color": "#614385", "marginRight": "5px"}),
                            html.Strong("Target Distribution: "),
                            html.Span(target_attribute, style={"fontWeight": "bold", "color": "#516395"})
                        ], style={"display": "flex", "alignItems": "center", "flexGrow": 1, "overflow": "hidden"}),
                        
                        # Right buttons
                        html.Div([
                            toggle_button,
                            remove_button
                        ], style={"display": "flex", "alignItems": "center"})
                    ],
                    style={
                        "display": "flex", 
                        "justifyContent": "space-between", 
                        "alignItems": "center", 
                        "width": "100%",
                        "backgroundColor": "#f0f5ff",
                        "padding": "8px 10px",
                        "borderRadius": "4px 4px 0 0",
                        "border": "1px solid #d0e0ff",
                        "borderBottom": "none"
                    }),
                    
                    # Timestamp
                    html.Div(f"Time Added: {item['timestamp']}", 
                            style={
                                "fontSize": "11px", 
                                "color": "#888",
                                "padding": "0px 10px 5px",
                                "backgroundColor": "#f0f5ff",
                                "borderLeft": "1px solid #d0e0ff",
                                "borderRight": "1px solid #d0e0ff"
                            }),
                    
                    # Expandable content
                    html.Div([
                        # Format summary text
                        html.Div([
                            html.P(line) for line in item.get("summary_text", "").split('\n') if line.strip()
                        ], className="mb-3"),
                        
                        # Chart container
                        html.Div(item.get("chart_data", "No chart data available"))
                    ],
                    id={"type": "expanded-content", "index": item["id"]},
                    style={
                        "display": "block" if item.get("expanded", False) else "none",
                        "padding": "10px", 
                        "backgroundColor": "#fbfcff",
                        "borderRadius": "0 0 4px 4px",
                        "border": "1px solid #d0e0ff",
                        "borderTop": "none"
                    })
                ], className="mb-3")
                
                context_items.append(item_content)
                
            elif item["type"] == "conditional_distribution":
                # Create header with conditional analysis information
                target_attribute = item.get("target_attribute", "Unknown attribute")
                target_value = item.get("target_value", "Unknown value")
                compare_attribute = item.get("compare_attribute", "Unknown attribute")
                
                # Item content structure
                item_content = html.Div([
                    # Header - always visible
                    html.Div([
                        # Left information
                        html.Div([
                            html.I(className="fas fa-chart-bar", style={"color": "#614385", "marginRight": "5px"}),
                            html.Strong("Conditional Analysis: "),
                            html.Span(f"{target_attribute}={target_value} by {compare_attribute}", 
                                      style={"fontWeight": "bold", "color": "#516395"})
                        ], style={"display": "flex", "alignItems": "center", "flexGrow": 1, "overflow": "hidden"}),
                        
                        # Right buttons
                        html.Div([
                            toggle_button,
                            remove_button
                        ], style={"display": "flex", "alignItems": "center"})
                    ],
                    style={
                        "display": "flex", 
                        "justifyContent": "space-between", 
                        "alignItems": "center", 
                        "width": "100%",
                        "backgroundColor": "#f0f5ff",
                        "padding": "8px 10px",
                        "borderRadius": "4px 4px 0 0",
                        "border": "1px solid #d0e0ff",
                        "borderBottom": "none"
                    }),
                    
                    # Timestamp
                    html.Div(f"Time Added: {item['timestamp']}", 
                            style={
                                "fontSize": "11px", 
                                "color": "#888",
                                "padding": "0px 10px 5px",
                                "backgroundColor": "#f0f5ff",
                                "borderLeft": "1px solid #d0e0ff",
                                "borderRight": "1px solid #d0e0ff"
                            }),
                    
                    # Expandable content
                    html.Div([
                        # Format summary text
                        html.Div([
                            html.P(line) for line in item.get("summary_text", "").split('\n') if line.strip()
                        ], className="mb-3"),
                        
                        # Chart container
                        html.Div(item.get("chart", "No chart data available"))
                    ],
                    id={"type": "expanded-content", "index": item["id"]},
                    style={
                        "display": "block" if item.get("expanded", False) else "none",
                        "padding": "10px", 
                        "backgroundColor": "#fbfcff",
                        "borderRadius": "0 0 4px 4px",
                        "border": "1px solid #d0e0ff",
                        "borderTop": "none"
                    })
                ], className="mb-3")
                
                context_items.append(item_content)
                
            elif item["type"] == "distribution_comparison":
                # create toggle button
                toggle_button = html.Button(
                    html.I(className="fas fa-eye" if not item.get("expanded", False) else "fas fa-eye-slash"),
                    id={"type": "toggle-context", "index": item["id"]},
                    className="toggle-context-btn",
                    style={
                        "background": "none",
                        "border": "none",
                        "color": "#516395",
                        "cursor": "pointer",
                        "padding": "0 5px",
                        "fontSize": "14px"
                    }
                )
                
                # simplify button definition, remove attributes that may cause JavaScript errors
                remove_button = html.Button(
                    html.I(className="fas fa-times"),
                    id={"type": "remove-context", "index": item["id"]},
                    className="remove-context-btn",
                    style={
                        "background": "none",
                        "border": "none",
                        "color": "#dc3545",
                        "cursor": "pointer",
                        "padding": "0 5px",
                        "fontSize": "14px"
                    }
                )
                
                # extract column name information
                column_name = ""
                
                # process cell_info text
                cell_info_lines = item["cell_info"].split('\n') if "cell_info" in item else []
                
                # try to match directly from cell_info
                for line in cell_info_lines:
                    if "Column:" in line or "column:" in line.lower():
                        # use case-insensitive matching to extract column name
                        parts = re.split(r"column:", line, flags=re.IGNORECASE)
                        if len(parts) > 1:
                            column_name = parts[1].strip()
                            break
                
                # if not found, try to extract from summary_text
                if not column_name and "summary_text" in item:
                    summary_text = item["summary_text"]
                    match = re.search(r"column:?\s*([^\n,]+)", summary_text, re.IGNORECASE)
                    if match:
                        column_name = match.group(1).strip()
                
                # if still not found, look for other common patterns
                if not column_name:
                    for pattern in [r"attribute:?\s*([^\n,]+)", r"feature:?\s*([^\n,]+)"]:
                        for text in [item.get("cell_info", ""), item.get("summary_text", "")]:
                            match = re.search(pattern, text, re.IGNORECASE)
                            if match:
                                column_name = match.group(1).strip()
                                break
                        if column_name:
                            break
                
                # if still not found, check for patterns in quotes or brackets
                if not column_name:
                    for text in [item.get("cell_info", ""), item.get("summary_text", "")]:
                        for pattern in [r"['\"]([\w\s-]+)['\"]" , r"\[([\w\s-]+)\]"]:
                            match = re.search(pattern, text)
                            if match:
                                potential_name = match.group(1).strip()
                                if len(potential_name) > 0 and len(potential_name) < 30:
                                    column_name = potential_name
                                    break
                        if column_name:
                            break
                
                # last resort: if reasonable and short, use the first line of cell_info
                if not column_name and len(cell_info_lines) > 0:
                    first_line = cell_info_lines[0].strip()
                    if len(first_line) < 50:
                        column_name = first_line
                
                print(f"Extracted column name: '{column_name}' from item: {item['id']}")
                
                # if all attempts fail, use default value
                if not column_name or column_name.isspace():
                    column_name = "Unknown column"
                
                # create context item component - use fully separated structure to ensure header is always visible
                # outer container - always visible
                item_content = html.Div([
                    # first part: header - always visible
                    html.Div([
                        # left information
                        html.Div([
                            html.I(className="fas fa-chart-bar", style={"color": "#614385", "marginRight": "5px"}),
                            html.Strong("Distribution Comparison: "),
                            html.Span(column_name, style={"fontWeight": "bold", "color": "#516395"})
                        ], style={"display": "flex", "alignItems": "center", "flexGrow": 1, "overflow": "hidden"}),
                        
                        # right buttons
                        html.Div([
                            toggle_button,
                            remove_button
                        ], style={"display": "flex", "alignItems": "center"})
                    ], 
                    style={
                        "display": "flex", 
                        "justifyContent": "space-between", 
                        "alignItems": "center", 
                        "width": "100%",
                        "backgroundColor": "#f0f5ff",
                        "padding": "8px 10px",
                        "borderRadius": "4px 4px 0 0",
                        "border": "1px solid #d0e0ff",
                        "borderBottom": "none"
                    }),

                    # second part: timestamp - always visible
                    html.Div(f"Time Added: {item['timestamp']}", 
                            style={
                                "fontSize": "11px", 
                                "color": "#666", 
                                "padding": "2px 10px",
                                "backgroundColor": "#f8f9f9",
                                "borderLeft": "1px solid #d0e0ff",
                                "borderRight": "1px solid #d0e0ff",
                                # if content is collapsed, add bottom border and rounded corners
                                "borderBottom": "1px solid #d0e0ff" if not item.get("expanded", False) else "none",
                                "borderRadius": "0 0 4px 4px" if not item.get("expanded", False) else "0",
                                # ensure timestamp always visible
                                "display": "block"
                            }),
                    
                    # third part: detailed content - collapsible
                    html.Div(
                        children=[
                            html.Div([
                                html.H6("Distribution Comparison Details", 
                                       style={"marginTop": "5px", "marginBottom": "10px", "fontWeight": "bold", "color": "#614385"}),
                                
                                # summary content
                                item["stored_summary"] if "stored_summary" in item else 
                                html.Pre(item.get("summary_text", ""), 
                                        style={"whiteSpace": "pre-wrap", "fontSize": "12px", "backgroundColor": "#f8f9fa", 
                                               "padding": "8px", "borderRadius": "4px"}),
                                
                                # chart content
                                html.Div([
                                    html.H6("Distribution Chart", 
                                           style={"marginTop": "15px", "marginBottom": "10px", "fontWeight": "bold", "color": "#614385"}),
                                    item["stored_chart"] if "stored_chart" in item else html.Div()
                                ]) if "stored_chart" in item else html.Div()
                            ], 
                            style={
                                "width": "100%",
                                "padding": "10px"
                            })
                        ],
                        id={"type": "expanded-content", "index": item["id"]},
                        style={
                            # use display property to control visibility
                            "display": "block" if item.get("expanded", False) else "none",
                            "backgroundColor": "#f8f9fa",
                            "borderRadius": "0 0 4px 4px",
                            "border": "1px solid #d0e0ff",
                            "borderTop": "none",
                            "width": "100%",
                            # ensure detailed content does not affect other parts
                            "position": "relative",
                            "zIndex": "1"
                        }
                    )
                ], 
                # container style - ensure always visible
                style={
                    "marginBottom": "15px", 
                    "width": "100%", 
                    "borderRadius": "5px",
                    "boxShadow": "0 1px 3px rgba(0,0,0,0.1)",
                    # key styles to ensure container is always visible
                    "display": "block",
                    "position": "relative"
                })
                
                context_items.append(item_content)
                
            elif item["type"] == "drift_analysis":
                # Create header for drift analysis
                attribute_name = item.get("attribute_name", "Unknown attribute")
                attribute_type = item.get("attribute_type", "Unknown type")
                
                # Item content structure
                item_content = html.Div([
                    # Header - always visible
                    html.Div([
                        # Left information
                        html.Div([
                            html.I(className="fas fa-chart-line", style={"color": "#614385", "marginRight": "5px"}),
                            html.Strong("Drift Analysis: "),
                            html.Span(f"{attribute_name} ({attribute_type})", 
                                      style={"fontWeight": "bold", "color": "#516395"})
                        ], style={"display": "flex", "alignItems": "center", "flexGrow": 1, "overflow": "hidden"}),
                        
                        # Right buttons
                        html.Div([
                            toggle_button,
                            remove_button
                        ], style={"display": "flex", "alignItems": "center"})
                    ],
                    style={
                        "display": "flex", 
                        "justifyContent": "space-between", 
                        "alignItems": "center", 
                        "width": "100%",
                        "backgroundColor": "#f0f8ff",  # Light blue for drift analysis
                        "padding": "8px 10px",
                        "borderRadius": "4px 4px 0 0",
                        "border": "1px solid #b3d9ff",
                        "borderBottom": "none"
                    }),
                    
                    # Timestamp
                    html.Div(f"Time Added: {item['timestamp']}", 
                            style={
                                "fontSize": "11px", 
                                "color": "#888",
                                "padding": "0px 10px 5px",
                                "backgroundColor": "#f0f8ff",
                                "borderLeft": "1px solid #b3d9ff",
                                "borderRight": "1px solid #b3d9ff"
                            }),
                    
                    # Expandable content
                    html.Div([
                        # Dataset metadata section
                        html.Div([
                            html.H6("Dataset Information", 
                                   style={"marginTop": "5px", "marginBottom": "10px", "fontWeight": "bold", "color": "#614385"}),
                            html.Pre(item.get("dataset_metadata", {}) and 
                                   f"Primary: {item['dataset_metadata'].get('primary_rows', 'N/A')} rows × {item['dataset_metadata'].get('primary_columns', 'N/A')} cols\n" +
                                   f"Secondary: {item['dataset_metadata'].get('secondary_rows', 'N/A')} rows × {item['dataset_metadata'].get('secondary_columns', 'N/A')} cols\n" +
                                   f"Common attributes: {item['dataset_metadata'].get('common_columns_count', 'N/A')}" or "No metadata available",
                                   style={"whiteSpace": "pre-wrap", "fontSize": "11px", "backgroundColor": "#f8f9fa", 
                                          "padding": "8px", "borderRadius": "4px", "marginBottom": "10px"})
                        ]) if item.get("dataset_metadata") else html.Div(),
                        
                        # Metrics details section
                        html.Div([
                            html.H6("Statistical Metrics", 
                                   style={"marginTop": "5px", "marginBottom": "10px", "fontWeight": "bold", "color": "#614385"}),
                            html.Pre(item.get("metric_details", "No metrics available"), 
                                   style={"whiteSpace": "pre-wrap", "fontSize": "12px", "backgroundColor": "#f8f9fa", 
                                          "padding": "8px", "borderRadius": "4px", "marginBottom": "10px"})
                        ]),
                        
                        # Interpretation section
                        html.Div([
                            html.H6("Interpretation Guide", 
                                   style={"marginTop": "5px", "marginBottom": "10px", "fontWeight": "bold", "color": "#614385"}),
                            html.Pre(item.get("interpretation", "No interpretation available"), 
                                   style={"whiteSpace": "pre-wrap", "fontSize": "12px", "backgroundColor": "#f0f8ff", 
                                          "padding": "8px", "borderRadius": "4px"})
                        ])
                    ],
                    id={"type": "expanded-content", "index": item["id"]},
                    style={
                        "display": "block" if item.get("expanded", False) else "none",
                        "padding": "10px", 
                        "backgroundColor": "#fbfcff",
                        "borderRadius": "0 0 4px 4px",
                        "border": "1px solid #b3d9ff",
                        "borderTop": "none"
                    })
                ], className="mb-3")
                
                context_items.append(item_content)
                
            elif item["type"] == "metric":
                # Create buttons for metric item
                toggle_button = html.Button(
                    html.I(className="fas fa-eye" if not item.get("expanded", False) else "fas fa-eye-slash"),
                    id={"type": "toggle-context", "index": item["id"]},
                    className="toggle-context-btn",
                    style={
                        "background": "none",
                        "border": "none",
                        "color": "#516395",
                        "cursor": "pointer",
                        "padding": "0 5px",
                        "fontSize": "14px"
                    }
                )
                
                remove_button = html.Button(
                    html.I(className="fas fa-times"),
                    id={"type": "remove-context", "index": item["id"]},
                    className="remove-context-btn",
                    style={
                        "background": "none",
                        "border": "none",
                        "color": "#dc3545",
                        "cursor": "pointer",
                        "padding": "0 5px",
                        "fontSize": "14px"
                    }
                )
                
                # Get attribute and metric information
                attr_name = item.get("attribute_name", "Unknown attribute")
                metric_name = item.get("metric_name", "Unknown metric")
                
                # Create metric context item component
                item_content = html.Div([
                    # Header bar - always visible
                    html.Div([
                        # Left side information
                        html.Div([
                            html.I(className="fas fa-chart-line", style={"color": "#614385", "marginRight": "5px"}),
                            html.Strong(f"{metric_name}: "),
                            html.Span(attr_name, style={"fontWeight": "bold", "color": "#516395"})
                        ], style={"display": "flex", "alignItems": "center", "flexGrow": 1, "overflow": "hidden"}),
                        
                        # Right side buttons
                        html.Div([
                            toggle_button,
                            remove_button
                        ], style={"display": "flex", "alignItems": "center"})
                    ],
                    style={
                        "display": "flex", 
                        "justifyContent": "space-between", 
                        "alignItems": "center", 
                        "width": "100%",
                        "backgroundColor": "#f5f0ff",  # Slightly different color than distribution items
                        "padding": "8px 10px",
                        "borderRadius": "4px 4px 0 0",
                        "border": "1px solid #e0d0ff",
                        "borderBottom": "none"
                    }),
                    
                    # Timestamp - always visible
                    html.Div(f"Time Added: {item['timestamp']}", 
                            style={
                                "fontSize": "11px", 
                                "color": "#666", 
                                "padding": "2px 10px",
                                "backgroundColor": "#f8f9f9",
                                "borderLeft": "1px solid #e0d0ff",
                                "borderRight": "1px solid #e0d0ff",
                                "borderBottom": "1px solid #e0d0ff" if not item.get("expanded", False) else "none",
                                "borderRadius": "0 0 4px 4px" if not item.get("expanded", False) else "0",
                                "display": "block"
                            }),
                    
                    # Detailed content - collapsible
                    html.Div(
                        children=[
                            html.Div([
                                html.H6("Statistical Metric Details", 
                                       style={"marginTop": "5px", "marginBottom": "10px", "fontWeight": "bold", "color": "#614385"}),
                                
                                # Metric details
                                html.Div([
                                    html.P([html.Strong("Attribute Type: "), item.get("attribute_type", "Unknown")]),
                                    html.P([html.Strong("Metric Details: ")]),
                                    html.Pre(item.get("metric_details", ""), 
                                           style={"whiteSpace": "pre-wrap", "fontSize": "12px", "backgroundColor": "#f8f9fa", 
                                                  "padding": "8px", "borderRadius": "4px"}),
                                    html.P([html.Strong("Interpretation: ")]),
                                    html.Pre(item.get("interpretation", ""), 
                                           style={"whiteSpace": "pre-wrap", "fontSize": "12px", "backgroundColor": "#f8f9fa", 
                                                  "padding": "8px", "borderRadius": "4px"})
                                ])
                            ], 
                            style={"width": "100%", "padding": "10px"})
                        ],
                        id={"type": "expanded-content", "index": item["id"]},
                        style={
                            "display": "block" if item.get("expanded", False) else "none",
                            "backgroundColor": "#f8f9fa",
                            "borderRadius": "0 0 4px 4px",
                            "border": "1px solid #e0d0ff",
                            "borderTop": "none",
                            "width": "100%",
                            "position": "relative",
                            "zIndex": "1"
                        }
                    )
                ],
                style={
                    "marginBottom": "15px", 
                    "width": "100%", 
                    "borderRadius": "5px",
                    "boxShadow": "0 1px 3px rgba(0,0,0,0.1)",
                    "display": "block",
                    "position": "relative"
                })
                
                context_items.append(item_content)
        
        # Render the context items container with all valid items
        if context_items:
            # Count different types of items
            distribution_items = [item for item in context_data if item.get("type") == "distribution_comparison"]
            metric_items = [item for item in context_data if item.get("type") == "metric"]
            drift_analysis_items = [item for item in context_data if item.get("type") == "drift_analysis"]
            target_dist_items = [item for item in context_data if item.get("type") == "target_distribution"]
            conditional_dist_items = [item for item in context_data if item.get("type") == "conditional_distribution"]
            
            # Create the header text with comprehensive type counting
            item_type_counts = []
            if drift_analysis_items:
                item_type_counts.append(f"{len(drift_analysis_items)} drift analysis")
            if distribution_items:
                item_type_counts.append(f"{len(distribution_items)} distributions")
            if metric_items:
                item_type_counts.append(f"{len(metric_items)} metrics")
            if target_dist_items:
                item_type_counts.append(f"{len(target_dist_items)} target distributions")
            if conditional_dist_items:
                item_type_counts.append(f"{len(conditional_dist_items)} conditional distributions")
            
            if item_type_counts:
                header_text = f"Context Items ({', '.join(item_type_counts)})"
            else:
                header_text = f"Context Items ({len(context_items)} items)"
            
            context_display = html.Div(
                [
                    # Header to show how many items are present
                    html.Div(header_text, 
                            style={
                                "fontWeight": "bold", 
                                "marginBottom": "10px", 
                                "color": "#516395",
                                "borderBottom": "1px solid #d0e0ff",
                                "paddingBottom": "5px"
                            }),
                    # Container for all context items
                    html.Div(context_items)
                ],
                style={
                    "marginTop": "15px",
                    "padding": "10px",
                    "backgroundColor": "#f9f9f9",
                    "borderRadius": "5px",
                    "border": "1px dashed #aaa",
                    "display": "block"
                }
            )
            print(f"Successfully rendered {len(context_items)} context items")
            return context_display, {'display': 'block'}
        else:
            print("No valid context items to render")
            return None, {'display': 'none'}
            
    except Exception as e:
        print(f"Error in update_chat_context_display: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, {'display': 'none'}


@callback(
    [
        Output({"type": "expanded-content", "index": MATCH}, "style"),
        Output({"type": "toggle-context", "index": MATCH}, "children")
    ],
    Input({"type": "toggle-context", "index": MATCH}, "n_clicks"),
    State({"type": "expanded-content", "index": MATCH}, "style"),
    prevent_initial_call=True
)
def toggle_context_ui(n_clicks, current_style):
    """
    Pure frontend way to handle the display/hiding of context items
    Does not change data state, only changes CSS styles
    """
    if not n_clicks:
        return dash.no_update, dash.no_update
    
    try:
        # get current display state
        current_display = current_style.get("display", "block") if current_style else "block"
        
        # flip display state
        new_display = "none" if current_display == "block" else "block"
        print(f"Toggle UI: Changing display from {current_display} to {new_display}")
        
        # create new style object
        new_style = {
            # only update display property, keep other styles unchanged
            "display": new_display,
            "backgroundColor": "#f8f9fa",
            "borderRadius": "0 0 4px 4px",
            "border": "1px solid #d0e0ff",
            "borderTop": "none",
            "width": "100%",
            "padding": "10px" if new_display == "block" else "0"
        }
        
        # update icon
        new_icon = html.I(className="fas fa-eye-slash" if new_display == "block" else "fas fa-eye")
        
        return new_style, new_icon
        
    except Exception as e:
        print(f"Error in toggle_context_ui: {str(e)}")
        import traceback
        traceback.print_exc()
        return dash.no_update, dash.no_update
        
    except Exception as e:
        print(f"Error in toggle_context_ui: {str(e)}")
        # whether to return default icon, avoid returning empty
        import traceback
        traceback.print_exc()
        return dash.no_update, dash.no_update

# Add callback for removing context items
@callback(
    Output("chat-context-data", "data", allow_duplicate=True),
    [Input({"type": "remove-context", "index": ALL}, "n_clicks")],
    [State("chat-context-data", "data")],
    prevent_initial_call=True
)
def remove_context_item(n_clicks, context_data):
    """
    Remove a context item when the remove button is clicked.
    
    This function ensures proper data handling when removing items,
    with validation and safety checks to maintain data integrity.
    """
    # Check if triggered by initialization to prevent automatic deletion
    if not callback_context.triggered or not any(n_clicks):
        return dash.no_update
        
    # Check if all n_clicks are None or 0 (initialization state)
    if all(n is None or n == 0 for n in n_clicks):
        print("[REMOVE CONTEXT] Ignoring automatic trigger with no real clicks")
        return dash.no_update
    
    # Validate context_data
    if not isinstance(context_data, list) or not context_data:
        print("Warning: Invalid or empty context_data in remove_context_item")
        return [] if context_data is None else context_data
    
    # Get the ID of the context item to remove
    try:
        triggered_id = callback_context.triggered[0]["prop_id"]
        context_id = json.loads(triggered_id.split(".")[0])["index"]
        
        print(f"Removing context item with ID: {context_id}")
        print(f"Before removal: {len(context_data)} items")
        
        # Filter out the item with the matching ID
        updated_context = [item for item in context_data if isinstance(item, dict) and item.get("id") != context_id]
        
        print(f"After removal: {len(updated_context)} items remaining")
        if len(updated_context) == len(context_data):
            print(f"Warning: Item with ID {context_id} not found in context data")
        
        # ✅ CRITICAL FIX: Sync with global_vars for chat compatibility
        if hasattr(global_vars, 'explain_context_data'):
            global_vars.explain_context_data = updated_context.copy()
        else:
            global_vars.explain_context_data = updated_context.copy()
        print(f"[REMOVE CONTEXT] ✅ Synced global_vars.explain_context_data: {len(updated_context)} items")
        
        return updated_context
        
    except Exception as e:
        print(f"Error in remove_context_item: {str(e)}")
        import traceback
        traceback.print_exc()
        return context_data  # Return original data in case of error
def return_to_dataset_preview_style(n_clicks):
    """
    Return to dataset preview view when the back button is clicked (style changes).
    """
    if n_clicks is None:
        return dash.no_update
    
    print("Back button clicked - returning to dataset preview")  # Debug log
    
    # Show dataset preview, hide metrics table
    dataset_preview_style = {
        "width": "100%", 
        "opacity": "1", 
        "height": "auto", 
        "overflow": "visible", 
        "position": "relative",
        "transition": "opacity 0.4s ease-in-out"
    }
    metrics_table_style = {
        "width": "100%", 
        "opacity": "0", 
        "height": "0", 
        "overflow": "hidden", 
        "display": "none",
        "position": "absolute",
        "transition": "opacity 0.4s ease-in-out, height 0.4s ease-in-out"
    }
    
    return dataset_preview_style, metrics_table_style, "explain"




def calculate_drift_severity_score(metric_name, value):
    """
    Calculate drift severity score based on metric-specific logic.
    - For most metrics: higher value = more severe drift
    - For p-value metrics: lower value = more severe drift
    - For N/A values: return 0 (cannot be scored)
    
    Returns a score from 0-100, where 100 is most severe.
    """
    if value == "N/A" or value is None:
        return 0
    
    try:
        numeric_value = float(value)
    except (ValueError, TypeError):
        return 0
    
    # Define p-value type metrics (lower value = more severe)
    p_value_metrics = {
         'T_Test', 'Z_Test', 'Fisher_Exact', 
        'Chi_Square', 'G_Test', 'Anderson_Darling', 'Cramer_von_Mises'
    }
    
    # For p-value metrics, use inverse logic
    if metric_name in p_value_metrics:
        # p-value: 0.0-0.05 = high severity, 0.05-1.0 = low severity
        if numeric_value <= 0.001:
            return 100  # Extremely high severity
        elif numeric_value <= 0.01:
            return 90   # Very high severity
        elif numeric_value <= 0.05:
            return 75   # High severity
        elif numeric_value <= 0.1:
            return 50   # Medium severity
        else:
            return max(0, 30 * (1 - numeric_value))  # Linear decrease
    
    # For other metrics, higher value = more severe drift
    else:
        # Use unified scoring for consistency within metric types
        return min(100, abs(numeric_value) * 100)

def calculate_severity_ranking_and_styles(metrics_data):
    """
    Calculate the top 3 most severe cells for each metric type and generate highlight style conditions.
    For N metrics, highlights N*3 cells total (top 3 per metric type).
    
    Uses direct metric value comparison for accurate ranking within each metric type.
    """
    if not metrics_data:
        return []
    
    # Columns to skip (non-metric columns)
    skip_columns = ['Attribute', 'Type', 'AddToContext', 'ExplainAction']
    
    # Group data by metric type
    metrics_by_column = {}
    
    for row_idx, row in enumerate(metrics_data):
        attribute_name = row.get('Attribute', f'Row_{row_idx}')
        for col_name, value in row.items():
            if col_name not in skip_columns and value != "N/A" and value is not None:
                try:
                    numeric_value = float(value)
                    if col_name not in metrics_by_column:
                        metrics_by_column[col_name] = []
                    
                    metrics_by_column[col_name].append({
                        'row_index': row_idx,
                        'column_id': col_name,
                        'attribute': attribute_name,
                        'value': value,
                        'numeric_value': numeric_value
                    })
                except (ValueError, TypeError):
                    continue  # Skip values that cannot be converted to numeric
    
    # Calculate top 3 for each metric type
    style_conditions = []
    severity_colors = [
        '#e57373',  # Dark red - 1st most severe
        '#ffcdd2',  # Medium red - 2nd most severe
        '#ffebee'   # Light red - 3rd most severe
    ]
    
    # Define p-value type metrics (lower value = more severe)
    p_value_metrics = {
     'T_Test', 'Z_Test', 'Fisher_Exact', 
        'Chi_Square', 'G_Test', 'Anderson_Darling', 'Cramer_von_Mises'
    }
    
    for metric_name, cells in metrics_by_column.items():
        # Determine sorting direction based on metric type
        is_p_value_metric = metric_name in p_value_metrics
        
        if is_p_value_metric:
            # For p-value metrics: lower value = more severe (ascending sort)
            cells.sort(key=lambda x: x['numeric_value'])
        else:
            # For other metrics: higher value = more severe (descending sort)
            cells.sort(key=lambda x: x['numeric_value'], reverse=True)
        
        # Take top 3
        top_3 = cells[:3]
        
        print(f"[DEBUG] Top 3 for {metric_name} ({'p-value' if is_p_value_metric else 'regular'}):")
        for i, cell in enumerate(top_3):
            rank = i + 1
            bg_color = severity_colors[min(i, 2)]
            text_color = 'white' if i == 0 else 'black'  # 1st rank uses white text, others use black
            
            style_conditions.append({
                'if': {
                    'row_index': cell['row_index'],
                    'column_id': cell['column_id']
                },
                'backgroundColor': bg_color,
                'color': text_color,
                'fontWeight': 'bold',
                'border': '3px solid #d32f2f',
                'borderRadius': '4px',
                'boxShadow': '0 2px 4px rgba(211,47,47,0.3)'
            })
            
            print(f"  #{rank}: {cell['attribute']}.{cell['column_id']} = {cell['value']}")
    
    print(f"[DEBUG] Total highlighted cells: {len(style_conditions)} across {len(metrics_by_column)} metrics")
    
    return style_conditions

   

# =============================================================================
# SEAMLESS AUTO-UPDATE SYSTEM
# =============================================================================

class MetricsAutoUpdater:
    """
    A clean, well-structured class to handle seamless metrics auto-updating.
    
    Responsibilities:
    1. Continuously monitor for dataset changes
    2. Silently recalculate metrics in background
    3. Automatically refresh any visible metrics tables
    4. Maintain cache consistency and performance
    """
    
    @staticmethod
    def should_monitor() -> bool:
        """
        Check if auto-update monitoring should be active.
        
        Returns:
            bool: True if monitoring conditions are met
        """
        # Only monitor when we have all prerequisites
        return (
            hasattr(global_vars, 'target_attribute') and 
            global_vars.target_attribute is not None and
            hasattr(global_vars, 'df') and 
            global_vars.df is not None and
            hasattr(global_vars, 'secondary_df') and 
            global_vars.secondary_df is not None
        )
    
    @staticmethod
    def recalculate_metrics_silently() -> bool:
        """
        Silently recalculate metrics in the background without user disruption.
        
        Returns:
            bool: True if recalculation was successful
        """
        try:
            from drift.detect import generate_metrics_data
            
            print(f"[AUTO-UPDATE] Silently recalculating metrics for target: {global_vars.target_attribute}")
            
            # Calculate new metrics
            metrics_data, data_length = generate_metrics_data()
            
            if not metrics_data:
                print("[AUTO-UPDATE] Failed to calculate metrics")
                return False
            
            # Update cache and reset change flags
            cache_success = global_vars.cache_metrics(metrics_data, data_length, force=True)
            if cache_success:
                global_vars.reset_change_flags()
                global_vars.metrics_data = metrics_data
                print(f"[AUTO-UPDATE] ✅ Successfully updated {len(metrics_data)} metrics silently")
                return True
            
            return False
            
        except Exception as e:
            print(f"[AUTO-UPDATE] ❌ Error during silent recalculation: {str(e)}")
            return False
    
    @staticmethod
    def get_fresh_metrics_data():
        """
        Get the most up-to-date metrics data, recalculating if necessary.
        
        Returns:
            tuple: (metrics_data, data_length) or (None, None)
        """
        # If data is current, return cached version
        if not global_vars.are_metrics_outdated():
            return global_vars.get_cached_metrics()
        
        # Otherwise, recalculate silently and return fresh data
        if MetricsAutoUpdater.recalculate_metrics_silently():
            return global_vars.get_cached_metrics()
        
        return None, None



#     




def generate_metrics_table_component(metrics_data, data_length, auto_updated=False):
    """
    Generate the complete metrics table component with IDENTICAL styling to original.
    Uses the EXACT same dynamic column generation and styling system as toggle_metrics_table_visibility
    to ensure perfect consistency between manual detect and auto-updated tables.
    
    Args:
        metrics_data: The calculated metrics data
        data_length: Data length information
        auto_updated: Whether this was triggered by auto-update
        
    Returns:
        html.Div: The complete metrics table component with identical styling
    """
    try:
        # Use the EXACT same table generation logic from toggle_metrics_table_visibility
        # This ensures 100% identical styling and behavior
        
        # Add action buttons to each row (same as original)
        for row in metrics_data:
            row["AddToChat"] = "💬"  # chat emoji 
            row["AddToExplain"] = "📊"  # chart emoji
            
            # Add an "Explain" button for direct navigation to Explain stage
            row["ExplainAction"] = "Explain"
        
        # EXACT COPY of dynamic column generation from toggle_metrics_table_visibility
        def create_dynamic_columns(sample_data):
            """Create dynamic column definitions based on actual data"""
            # base columns (always exist)
            base_columns = [
                {"name": "Attribute", "id": "Attribute", "selectable": True},
                {"name": "Type", "id": "Type", "selectable": False}
            ]
            
            # metric columns (added dynamically based on data)
            metric_columns = []
            
            # define metric display name mapping
            metric_display_names = {
                'JS_Divergence': 'JS Divergence',
                'PSI': 'PSI',
                'Wasserstein': 'Wasserstein',
                'KL_Divergence': 'KL Divergence',
                'Hellinger': 'Hellinger',
                'TVD': 'TVD',
                # 'KS_Test': 'KS Test',
                'Anderson_Darling': 'Anderson-Darling',
                'Cramer_von_Mises': 'Cramer-von-Mises',
                'Energy_Distance': 'Energy Distance',
                'E_Squared': 'E-Squared',
                'T_Test': 'T-Test',
                'Empirical_MMD': 'Empirical MMD',
                'Z_Test': 'Z-Test',
                'Fisher_Exact': 'Fisher Exact',
                'Chi_Square': 'Chi-Square',
                'G_Test': 'G-Test',
                # 'TargetRelevance': 'Target Relevance',
                'TargetRelevanceScore': 'Relevance Score (Primary)',
                # 'TargetRelevanceSecondary': 'Relevance (Secondary)',
                'TargetRelevanceScoreSecondary': 'Relevance Score (Secondary)',
                'RelevanceDelta': 'Relevance Change',
            }
            
            # get available metric columns from the first data row
            if sample_data:
                first_row = sample_data[0]
                for key in first_row.keys():
                    if key not in ['Attribute', 'Type', 'AddToChat', 'AddToExplain', 'ExplainAction']:
                        display_name = metric_display_names.get(key, key.replace('_', ' ').title())
                        metric_columns.append({
                            "name": display_name, 
                            "id": key, 
                            "selectable": False
                        })
            
            # action columns
            action_columns = [
                {"name": "Chat", "id": "AddToChat", "selectable": False},
                {"name": "Explain", "id": "AddToExplain", "selectable": False}
            ]
            
            return base_columns + metric_columns + action_columns
        
        # create dynamic columns (same as original)
        columns = create_dynamic_columns(metrics_data)
        
        # EXACT COPY of dynamic style conditions from toggle_metrics_table_visibility
        def create_dynamic_style_conditions(columns, metrics_data):
            style_conditions = []
            
            # first add N/A value gray style (low priority)
            for col in columns:
                if col['id'] not in ['Attribute', 'Type', 'AddToChat', 'AddToExplain', 'ExplainAction']:
                    style_conditions.append({
                        'if': {'filter_query': f'{{{col["id"]}}} contains "N/A"', 'column_id': col['id']},
                        'backgroundColor': '#e0e0e0',
                        'color': '#757575',
                        'fontStyle': 'italic'
                    })
                    # add default style for non-Attribute columns
                    if col['id'] not in ['AddToChat', 'AddToExplain']:
                        style_conditions.append({
                            'if': {'column_id': col['id']},
                            'cursor': 'default'
                        })
            
            # calculate severity highlighting style (high priority, will override other styles)
            severity_styles = calculate_severity_ranking_and_styles(metrics_data)
            
            # put severity styles at the end to ensure highest priority
            style_conditions.extend(severity_styles)
            
            return style_conditions
        
        # Create table with EXACT original styling from toggle_metrics_table_visibility
        table = dash_table.DataTable(
            id='metrics-table',
            # Track active cells to know which + button was clicked
            active_cell=None,
            columns=columns,
            data=metrics_data,
            style_table={'overflowX': 'auto', 'width': '100%'},
            style_cell={
                'padding': '10px',
                'textAlign': 'center',
                'cursor': 'pointer'
            },
            style_header={
                'backgroundColor': '#614385',
                'color': 'white',
                'fontWeight': 'bold',
                'padding': '12px'
            },
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': '#f5f5f5'
                },
                # Make Attribute column look clickable
                {
                    'if': {'column_id': 'Attribute'},
                    'cursor': 'pointer',
                    'color': '#614385',
                    'fontWeight': '600',
                    'textDecoration': 'underline'
                },
                # Style for the Add to Chat button column
                {
                    'if': {'column_id': 'AddToChat'},
                    'cursor': 'pointer',
                    'color': '#007bff',  # blue
                    'fontWeight': 'bold',
                    'fontSize': '18px',  # emoji size
                    'textAlign': 'center',
                    'verticalAlign': 'middle',
                    'backgroundColor': '#f8f9fa',  # light background
                    'border': '1px solid #007bff',  # blue border
                    'borderRadius': '4px',  # slightly rounded corners
                    'padding': '4px',
                    'margin': '0 auto'   # center
                },
                # Style for the Add to Explain button column
                {
                    'if': {'column_id': 'AddToExplain'},
                    'cursor': 'pointer',
                    'color': '#28a745',  # green
                    'fontWeight': 'bold',
                    'fontSize': '18px',  # emoji size
                    'textAlign': 'center',
                    'verticalAlign': 'middle',
                    'backgroundColor': '#f8f9fa',  # light background
                    'border': '1px solid #28a745',  # green border
                    'borderRadius': '4px',  # slightly rounded corners
                    'padding': '4px',
                    'margin': '0 auto'   # center
                },
                # Style for the Target Relevance columns
                {
                    'if': {'column_id': ['PrimaryTargetRelevance', 'SecondaryTargetRelevance', 'RelevanceDelta']},
                    'backgroundColor': '#f0f8ff',  # light blue background to distinguish the group
                    'borderLeft': '1px solid #d9edf7',
                    'borderRight': '1px solid #d9edf7',
                    'textAlign': 'center'
                },
                # Style for the Target Relevance column
                {
                    'if': {'column_id': 'TargetRelevance'},
                    'textAlign': 'center',
                    'fontWeight': '500'
                },
                # Style for Target relevance = High
                {
                    'if': {'filter_query': '{TargetRelevance} = "High"', 'column_id': 'TargetRelevance'},
                    'color': '#d32f2f',  # red for high relevance
                    'fontWeight': 'bold'
                },
                # Style for Target relevance = Target
                {
                    'if': {'filter_query': '{TargetRelevance} = "Target"', 'column_id': 'TargetRelevance'},
                    'color': '#614385',  # purple for target
                    'fontWeight': 'bold'
                }
            ] + create_dynamic_style_conditions(columns, metrics_data),
            css=[{
                'selector': '.dash-table-tooltip',
                'rule': 'background-color: white; font-size: 12px; text-align: left;'
            }],
            tooltip_data=[
                {
                    column: {'value': f"Click to view detailed distribution analysis for {row['Attribute']}", 'type': 'markdown'}
                    for column in row.keys() if column == 'Attribute'
                }
                for row in metrics_data
            ],
            row_selectable="single",
            selected_rows=[],
            page_action="native",
            page_current=0,
            page_size=20,
            sort_action="native",
            sort_mode="multi",
            cell_selectable=True,
            selected_cells=[]
        )
        
        # Build complete component with all original elements
        components_list = []
        
        # Add auto-update success message if this was auto-updated
        if auto_updated:
            auto_update_alert = dbc.Alert([
                html.Div([
                    html.I(className="fas fa-sync-alt", style={"marginRight": "8px"}),
                    html.Strong("📊 Metrics Auto-Updated: "),
                    f"Table refreshed with latest data for target attribute '{global_vars.target_attribute}'."
                ])
            ], color="info", className="mb-3", dismissable=True, duration=4000)
            components_list.append(auto_update_alert)
        
        # Add encoding alert if needed (copy from original)
        encoding_alert = None
        if hasattr(global_vars, 'df') and hasattr(global_vars, 'secondary_df'):
            primary_has_categorical = any(dtype == 'object' or str(dtype).startswith('category') 
                                        for dtype in global_vars.df.dtypes)
            secondary_has_categorical = any(dtype == 'object' or str(dtype).startswith('category') 
                                          for dtype in global_vars.secondary_df.dtypes)
            
            if primary_has_categorical or secondary_has_categorical:
                encoding_alert = dbc.Alert([
                    html.Div([
                        html.I(className="fas fa-info-circle", style={"marginRight": "8px"}),
                        html.Strong("Categorical Data Processing: "),
                        "String columns have been automatically encoded for numerical analysis. ",
                        "This ensures all metrics can be calculated consistently across different data types."
                    ])
                ], color="info", className="mb-3")
        
        # Severity explanation (copy from original)
        severity_explanation = html.Div([
            html.Div([
                html.P([
                    "🚨 ",
                    html.Strong("Drift Severity Indicators: "),
                    "Red highlighted cells show the ",
                    html.Strong("top 3 most severe values for each metric type", style={"color": "#d32f2f"}),
                    ". Deeper red = higher rank within each metric."
                ], className="mb-2 small text-muted", style={"fontSize": "0.9rem"})
            ], className="px-2"),
            # Table section - scrollable
            html.Div([
                table
            ], style={"maxHeight": "600px", "overflowY": "auto", "border": "1px solid #dee2e6", "borderRadius": "5px", "width": "100%"})
        ], style={"width": "100%"})
        
        # Build components list exactly as original
        if encoding_alert:
            components_list.append(encoding_alert)
        components_list.append(severity_explanation)
        
        # Create target distribution component exactly as original
        if global_vars.target_attribute:
            # Target distribution content (copy from original)
            target_dist_content = html.Div([
                dbc.Row([
                    # Left column - Distribution visualization
                    dbc.Col([
                        html.H4("Target Attribute Distribution", className="mb-3", style={"textAlign": "center"}),
                        dcc.Loading(
                            id="target-chart-loading",
                            type="circle",
                            children=html.Div(id="target-distribution-chart-container", style={"height": "400px"})
                        ),
                    ], width=6),
                    
                    # Right column - Conditional Distribution Analysis
                    dbc.Col([
                        html.H4("Conditional Distribution Analysis", className="mb-3", style={"textAlign": "center"}),
                        
                        # Hidden storage for top-k attributes data
                        dcc.Store(id="detect-top-k-attrs", data=[]),
                        
                        html.Div([
                            html.Label("Select Target Value:", className="form-label mb-2"),
                            dcc.Dropdown(
                                id="detect-target-value-dropdown",
                                placeholder="Choose a value...",
                                className="mb-3"
                            )
                        ]),
                        
                        html.Div([
                            html.Label("Compare with Attribute:", className="form-label mb-2"),
                            dcc.Dropdown(
                                id="detect-compare-attr-dropdown",
                                placeholder="Choose an attribute...",
                                className="mb-3"
                            )
                        ]),
                        
                        dcc.Loading(
                            id="detect-conditional-chart-loading",
                            type="circle",
                            children=html.Div(id="detect-conditional-chart-container", style={"height": "350px"})
                        )
                    ], width=6)
                ])
            ], style={"padding": "15px", "width": "100%"})
            
            # Create collapsible card for Target Distribution Analysis
            target_dist_component = dbc.Card([
                dbc.CardHeader(
                    html.Div([
                        html.Div(
                            html.H4("Overall Target Distribution Analysis", className="mb-0"),
                            style={"display": "inline-block", "width": "80%"}
                        ),
                        html.Div(
                            html.I(id="target-dist-toggle", className="fa fa-chevron-up",
                                  style={"cursor": "pointer", "fontSize": "20px"}),
                            style={"display": "inline-block", "width": "20%", "textAlign": "right"}
                        )
                    ], style={"display": "flex", "alignItems": "center"}, className="d-flex justify-content-between")
                ),
                dbc.Collapse(
                    dbc.CardBody(target_dist_content),
                    id="target-dist-collapse",
                    is_open=True
                )
            ], className="mb-3", style={"margin": "10px 0", "width": "100%"})
            
            target_dist_store = dcc.Store(id="target-dist-expanded", data=True)
            
            components_list.extend([target_dist_component, target_dist_store])
        
        return html.Div(components_list)
        
    except Exception as e:
        print(f"[AUTO-UPDATE] Error generating metrics table component: {str(e)}")
        return html.Div([
            html.Div("❌ Error displaying metrics table", 
                    style={"textAlign": "center", "color": "#d9534f", "padding": "20px"})
        ])


