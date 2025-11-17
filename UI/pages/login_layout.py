from flask_login import login_user
import dash_bootstrap_components as dbc
from dash import html, dcc, callback, Input, Output, State, callback_context
import dash
from dash.exceptions import PreventUpdate
from db_models.users import User
from db_models.databases import db
from UI.app import server
import time

dash.register_page(__name__, path='/', title='Login')

layout = dbc.Container(fluid=True, children=[
    dbc.Row(className="justify-content-center align-items-center vh-100", children=[
        dbc.Col([
            dbc.Row(className="justify-content-center", children=[
                dbc.Col([
                    html.H1("DriftNavi", className='fade-in logo-text text-center', 
                           style={'color': 'white', "fontSize": "4rem", "marginBottom": "0px", "fontWeight": "700"}),
                    html.Div([
                        html.I(className="fas fa-chart-line me-2", style={"color": "#f8f9fa"}),
                        html.Span("Navigate through distribution shifts with AI", style={"whiteSpace": "nowrap"})
                    ], className='fade-in tagline text-center', 
                       style={'color': '#f8f9fa', "fontSize": "1.2rem", "marginBottom": "40px", "opacity": "0.9"})
                ], width={"size": 12}, className="text-center")
            ]),
            
            dbc.Row(className="justify-content-center", children=[
                dbc.Col([
                    dbc.Card(
                        dbc.CardBody([
                            html.H4("Welcome Back", className="card-title mb-4 text-center", style={"color": "#614385"}),
                            dbc.Input(id="email-input", placeholder="Email",
                                     type="text", className="mb-3", style={"borderRadius": "8px", "padding": "12px 15px"}),
                            dbc.Input(id="password-input", placeholder="Password",
                                     type="password", className="mb-4", style={"borderRadius": "8px", "padding": "12px 15px"}),
                            html.Form([  # Wrap buttons in a standard HTML form for direct browser submission
                                dbc.Row([
                                    dbc.Col(
                                        dbc.Button(
                                            [html.I(className="fas fa-sign-in-alt me-2"), "Log in"], 
                                            id="login-button", color="primary", n_clicks=0, 
                                            class_name="w-100 py-2", type="button",
                                            style={"background": "linear-gradient(to right, #614385, #516395)", 
                                                 "borderColor": "transparent", "borderRadius": "8px", "fontWeight": "600"}),
                                        width=6
                                    ),
                                    dbc.Col(
                                        html.A(  # Use direct HTML link for signup
                                            dbc.Button(
                                                [html.I(className="fas fa-user-plus me-2"), "Sign up"], 
                                                color="light", n_clicks=0, 
                                                class_name="w-100 py-2", type="button",
                                                style={"borderRadius": "8px", "color": "#614385", "fontWeight": "600"}),
                                            href="/signup", id="signup-link"
                                        ),
                                        width=6, className="ms-auto"
                                    )
                                ], className="d-flex justify-content-between"),
                            ], id="login-form"),
                            html.Div(id="auth-result", className="mt-3", style={'color': '#dc3545', 'textAlign': 'center', 'fontWeight': '500'}),
                            dcc.Location(id='url', refresh=True)
                        ]),
                        style={"width": "100%", "maxWidth": "450px", "borderRadius": "15px", "boxShadow": "0 15px 30px rgba(0,0,0,0.2)"},
                        className="fade-in mx-auto"
                    )
                ], width={"size": 12, "md": 8, "lg": 6, "xl": 4}, className="d-flex justify-content-center")
            ])
        ], width={"size": 10})
    ])
], className="p-0", style={"background": "linear-gradient(135deg, #614385, #516395)", "minHeight": "100vh"})

# Use dcc.Location with refresh=True to force a full page reload
# dcc.Location(id='url', refresh=True)

# Update the login callback to use a different approach
@callback(
    Output("url", "pathname"),
    Input("login-button", "n_clicks"),
    State("email-input", "value"),
    State("password-input", "value"),
    prevent_initial_call=True
)
def handle_login(login_clicks, email, password):
    """Handle only login button clicks"""
    if login_clicks is None or login_clicks == 0:
        raise PreventUpdate
        
    if not email or not password:
        return dash.no_update

    user = User.query.filter_by(email=email).first()
    if user and user.check_password(password):
        login_user(user, remember=True)
        return "/home" 
    
    return dash.no_update

@callback(
    Output("auth-result", "children"),
    Input("login-button", "n_clicks"),
    State("email-input", "value"),
    State("password-input", "value"),
    prevent_initial_call=True
)
def handle_login_messages(login_clicks, email, password):
    """Handle authentication error messages"""
    if login_clicks is None or login_clicks == 0:
        raise PreventUpdate
        
    if not email or not password:
        return "Please fill in all fields."

    user = User.query.filter_by(email=email).first()
    if user and user.check_password(password):
        return ""
    else:
        return "Invalid username or password."
