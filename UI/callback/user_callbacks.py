import dash
from UI.app import app
from db_models.users import User
from db_models.databases import db
from flask_login import current_user
from dash import Input, Output, State, callback_context
from dash.exceptions import PreventUpdate
from constant_prompt import DEFAULT_PERSONA_PROMPT
from UI.variable import global_vars



# Survey modal functionality
@app.callback(
    Output("survey-modal", "is_open", allow_duplicate=True),
    Input("menu-profile", "n_clicks"),
    prevent_initial_call=True
)
def open_survey_modal(n_clicks):
    """Open the survey modal when the profile menu is clicked."""
    if n_clicks:
        return True
    return False


@app.callback(
    [Output("url", "pathname", allow_duplicate=True),
     Output("survey-result", "children"),
     Output("survey-modal", "is_open", allow_duplicate=True)],
    [Input("submit-button", "n_clicks"),
     Input("skip-button", "n_clicks")],
    [State("username-input", "value"),
     State("professional-role-input", "value"),
     State("industry-sector-dropdown", "value"),
     State("expertise-level-dropdown", "value"),
     State("technical-level-dropdown", "value"),
     State("drift-awareness-dropdown", "value"),
     State("areas-of-interest-checklist", "value")],
    prevent_initial_call=True
)
def handle_survey_submission(submit_clicks, skip_clicks, user_name, professional_role, industry_sector, expertise_level, technical_level, drift_awareness, areas_of_interest):
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if submit_clicks is not None and button_id == "submit-button":
        if not all([professional_role, industry_sector, expertise_level, areas_of_interest, technical_level, drift_awareness]):
            return dash.no_update, "Please fill in all fields.", dash.no_update

        try:
            # Fetch current user
            user = User.query.get(current_user.id)

            # Update user information
            if user_name:
                user.username = user_name
            user.professional_role = professional_role
            user.industry_sector = industry_sector
            user.expertise_level = expertise_level
            user.technical_level = technical_level
            user.drift_awareness = drift_awareness
            user.areas_of_interest = areas_of_interest
            user.persona_prompt = DEFAULT_PERSONA_PROMPT.format(
                professional_role=user.professional_role,
                industry_sector=user.industry_sector,
                expertise_level=user.expertise_level,
                technical_level=user.technical_level,
                drift_level=user.drift_awareness
            )

            db.session.commit()
            
            # Only update agent prompt if agent is initialized (i.e., dataset is loaded)
            if global_vars.agent is not None:
                global_vars.agent.update_agent_prompt()
            
            return '/home', 'Profile updated successfully!', False
        except Exception as e:
            db.session.rollback()
            return dash.no_update, f"An error occurred: {str(e)}", dash.no_update

    elif button_id == "skip-button":
        return '/home', 'Profile update skipped.', False

    raise PreventUpdate
