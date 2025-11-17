import dash_bootstrap_components as dbc
import dash_editor_components.PythonEditor
from db_models.conversation import Conversation
from db_models.users import User
from dash import dcc, html, dash_table, callback, Input, Output, State, MATCH, ALL
import dash_daq as daq
import dash
from flask_login import logout_user, current_user


dash.register_page(__name__, path='/chat_component/', title='Chat Component')


def layout():
    if not current_user.is_authenticated:
        return html.Div([
            dcc.Location(id="redirect-to-login",
                         refresh=True, pathname="/login"),
        ])

    return html.Div([

#
        # Chat Box Component
        dbc.Card(id="chat-box", children=[
            html.Div([
                html.Div([
                    html.H4("Chat with DriftNavi", className="secondary-title"),
                    html.Button(id="common-question-btn", children="Common Questions",
                               style={"backgroundColor": "white", "color": "grey", "border": "none"}, )
                ], style={"display": "flex", "alignItems": "center", "justifyContent": "space-between",
                          "width": "100%"}),
                html.Div(id='query-area', className='query-area'),
                html.Div(id='chat-context-area', className='chat-context-area', style={
                    'marginTop': '10px',
                    'marginBottom': '10px',
                    'padding': '10px',
                    'backgroundColor': '#f8f9fa',
                    'borderRadius': '5px',
                    'border': '1px dashed #aaa',
                    'display': 'none'
                }),
                dbc.Alert(
                    "Forget to import a dataset or enter a query?",
                    id="error-alert",
                    is_open=False,
                    dismissable=True,
                    color="danger",
                    duration=5000,
                ),
                # Pipeline Buttons Section - New Feature for Detect/Explain/Adapt
                html.Div([
                    html.H6("Analysis Pipeline", 
                           style={"margin": "0 0 10px 0", "color": "#666", "fontSize": "14px", "fontWeight": "600"}),
                    html.Div([
                        html.Button([
                            html.I(className="fas fa-search-plus", style={"marginRight": "8px"}),
                            "Detect"
                        ], 
                        id="chat-detect-btn", 
                        className="pipeline-chat-btn active",
                        title="Analyze and quantify distribution shifts between datasets",
                        n_clicks=0),
                        
                        html.Button([
                            html.I(className="fas fa-chart-line", style={"marginRight": "8px"}),
                            "Explain"
                        ], 
                        id="chat-explain-btn", 
                        className="pipeline-chat-btn",
                        title="Understand and interpret dataset differences",
                        disabled=False,
                        n_clicks=0),
                        
                        html.Button([
                            html.I(className="fas fa-wrench", style={"marginRight": "8px"}),
                            "Adapt"
                        ], 
                        id="chat-adapt-btn", 
                        className="pipeline-chat-btn",
                        title="Get actionable strategies for distribution shift adaptation",
                        disabled=False,
                        n_clicks=0)
                    ], className="pipeline-chat-buttons-container"),
                    
                    # Status indicator
                    html.Div([
                        html.I(className="fas fa-info-circle", style={"marginRight": "5px", "color": "#17a2b8"}),
                        html.Span("Click Detect to analyze distribution shifts", id="pipeline-status-text")
                    ], className="pipeline-status-indicator")
                ], className="pipeline-section", style={
                    "marginBottom": "15px", 
                    "padding": "12px", 
                    "backgroundColor": "#f8f9fa", 
                    "borderRadius": "8px",
                    "border": "1px solid #e9ecef"
                }),

                # Message input row
                html.Div([
                    dcc.Loading(
                        id="loading-1",
                        children=[
                            html.Div(id='next-suggested-questions', style={"marginBottom":"20px"}),
                            html.Div(
                                style={'display': 'flex', 'flexDirection': 'row', 'alignItems': 'center', "gap":"10px"},
                                children=[
                                    dcc.Input(id='query-input', type='text', className='query-input',
                                              placeholder='Type your message here'),
                                    html.Button(html.Span(className="fas fa-paper-plane"), id='send-button',
                                                title="Send your message.", n_clicks=0,
                                                className='send-button'),

                                    dcc.Upload(id="upload-rag",
                                               children=html.Button(html.Span(className="fas fa-file"),
                                                                    id='RAG-button',
                                                                    title="Upload your document for RAG.",
                                                                    n_clicks=0,
                                                                    className='send-button'),
                                               multiple=True),

                                    html.Div(id='rag-output'),

                                    daq.ToggleSwitch(id='rag-switch', value=False),

                                    html.Div(id='rag-switch-output'),
                                ])
                        ],
                        type="default",  # Choose from "graph", "cube", "circle", "dot", or "default"
                    ),

                ], style={"marginTop":"10px", "marginBottom":"10px"}),
            ], className='query')
        ], className='card'),

        # RAG Card - Additional component for RAG functionality
        dbc.Card(id="rag-card", style={'display': 'block'}, children=[
            html.Div([
                # RAG display area
                html.H4("RAG Documents", className="secondary-title"),
                dcc.Loading(
                    id="loading-2",
                    children=[
                        html.Div(id='RAG-area', className='RAG-area')],
                    type="dot",  # Choose from "graph", "cube", "circle", "dot", or "default"
                ),
            ], className='query'),
        ]),

        # Modal for Analysis Panel
        dbc.Modal([
            dbc.ModalHeader([
                html.H4([
                    html.I(className="fas fa-chart-line me-2", style={"color": "#516395"}),
                    "Analysis Results"
                ], className="modal-title")
            ]),
            dbc.ModalBody(
                id="analysis-modal-body",
                style={"maxHeight": "70vh", "overflowY": "auto"}
            ),
            dbc.ModalFooter([
                dbc.Button("Close", id="analysis-modal-close", className="ms-auto", n_clicks=0)
            ])
        ],
        id="analysis-modal",
        is_open=False,
        size="xl",
        backdrop=True,
        scrollable=True
        ),

        # Store components for managing state
        dcc.Store(id="filtered-context-data", storage_type="memory"),
        dcc.Store(id="chat-explain-trigger", storage_type="memory", data=0),
        dcc.Store(id="finetune-trigger-store", storage_type="memory"),
        dcc.Store(id="finetune-result-store", storage_type="memory"),
        
        # Download component for finetune workflow
        dcc.Download(id="finetune-download"),
        
        # Download component for retrain workflow
        dcc.Download(id="retrain-download"),
    ])
