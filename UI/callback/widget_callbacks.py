import dash

from UI.app import app
from dash.dependencies import Input, Output, State
from UI.variable import global_vars



drift_management_questions = {
    "Detect": [
        "Is there any distribution shift in my data?",
        "How do I know if my datasets have drifted?",
        "What steps can I take to detect drift?",
        "Are there any signals or patterns that indicate distribution changes?",
        "Can you help me identify if drift exists between my datasets?"
    ],
    "Explain": [
        "Why did the distribution shift occur?",
        "What attributes are most affected by the drift?",
        "Can you explain the patterns behind the detected shifts?",
        "What does the drift analysis tell me about my data?",
        "How can I understand the root causes of these distribution changes?"
    ],
    "Adapt": [
        "How can I address the drift in my data or system?",
        "What are the options for mitigating distribution shifts?",
        "What changes should I make to handle drift effectively?",
        "Can you suggest ways to adapt my model to the new distribution?",
        "How can I test if the drift adaptation strategies are working?"
    ]
}
@app.callback(
    Output("question-modal", "is_open"),
    Output("question-modal-list", "options"),
    [Input("common-question-btn", "n_clicks"),
     Input("question-modal-close-btn", "n_clicks")],
    [State("question-modal", "is_open")],
    prevent_initial_call=True,
)
def display_common_questions(open_clicks, close_clicks, is_open):
    questions = drift_management_questions.get(global_vars.current_stage, [])
    options = [{"label": question, "value": question} for question in questions]
    if open_clicks or close_clicks:
        return not is_open, options
    return is_open, options


@app.callback(
    Output("query-input", "value", allow_duplicate=True),
    Output("question-modal", "is_open", allow_duplicate=True),
    [Input("question-modal-choose-btn", "n_clicks"),
     State("question-modal-list", "value")],
    prevent_initial_call=True,
)
def choose_question(n_clicks, question):
    return question, False

@app.callback(
    Output("upload-modal", "is_open", allow_duplicate=True),
    Input("close-upload-modal", "n_clicks"),
    prevent_initial_call=True,
)
def close_upload_modal(n_clicks):
    return False
