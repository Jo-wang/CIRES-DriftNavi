import time
import dash
from UI.app import app
from dash.dependencies import Input, Output, State
from UI.functions import *

"""
Removed legacy 3-column view toggles (hide_chatbox, hide_dataviews, hide_chartview).
These relied on menu items not present in current UI and produced duplicate width outputs.
"""

@app.callback(
    [Output('menu-model-gpt4omini', 'children', allow_duplicate=True),
     Output('menu-model-gpt4o', 'children', allow_duplicate=True)],
    [Input('menu-model-gpt4omini', 'n_clicks'),
     Input('menu-model-gpt4o', 'n_clicks')],
    prevent_initial_call=True
)
def change_llm_model(n_clicks_gpt4omini, n_clicks_gpt4o):
    ctx = dash.callback_context

    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate

    clicked_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if clicked_id == 'menu-model-gpt4omini':
        global_vars.agent.set_llm_model('gpt-4o-mini')
        return "GPT-4o-mini ✔", "GPT-4o"
    elif clicked_id == 'menu-model-gpt4o':
        global_vars.agent.set_llm_model('gpt-4o')
        return "GPT-4o-mini", "GPT-4o ✔"

    raise dash.exceptions.PreventUpdate

@app.callback(
    Output("export-history-modal", "is_open"),
    [Input("menu-export-chat", "n_clicks"), Input("close", "n_clicks")],
    [State("export-history-modal", "is_open")],
)
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open