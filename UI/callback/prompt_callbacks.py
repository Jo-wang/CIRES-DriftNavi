from UI.app import app
from db_models.users import User
from db_models.databases import db
from dash.dependencies import Input, Output, State
from flask_login import current_user
from UI.variable import global_vars

@app.callback(
    Output({'type': 'spinner-btn', 'index': 2}, 'children', allow_duplicate=True),
    Input({'type': 'spinner-btn', 'index': 2}, 'children'),
    [State('next-question-input-1', "value"),
     # State('next-question-input-2', "value"),
     State('system-prompt-input', "value"),
     State('persona-prompt-input', "value"),
     State('prefix-prompt-input', "value")],
    prevent_initial_call=True,
)
def update_prompt(update_prompt_click, new_next_question_1, new_system_prompt, new_persona_prompt,
                  new_prefix_prompt):
    try:
        # Fetch
        user = User.query.get(current_user.id)

        # Update and commit
        user.follow_up_questions_prompt_1 = new_next_question_1
        user.prefix_prompt = new_prefix_prompt
        user.persona_prompt = new_persona_prompt
        user.system_prompt = new_system_prompt
        db.session.commit()
        global_vars.agent.update_agent_prompt()
    except Exception as e:
        db.session.rollback()
        print("Error when update prompt", e)
    return "Save"


@app.callback(
    Output('url', 'pathname', allow_duplicate=True),
    Input('return-home-button', 'n_clicks'),
    prevent_initial_call=True
)
def logout_and_redirect(n_clicks):
    if n_clicks > 0:
        return "/home"
