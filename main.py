import sys
import os

# Ensure consistent import paths by setting working directory
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
os.chdir(project_root)

from UI.app import app, server
from db_models.users import db
from UI.callback.code_callbacks import *
from UI.callback.menu_callbacks import *
from UI.callback.user_callbacks import *
from UI.callback.data_callbacks import *
from UI.callback.chat_callbacks import *
from UI.callback.client_callbacks import *
from UI.callback.user_callbacks import *
# from UI.callback.report_callbacks import *
from UI.callback.wizard_callbacks import *
from UI.callback.chat_mode_callbacks import *
from UI.callback.prompt_callbacks import *
from UI.callback.widget_callbacks import *

# Import target attribute callbacks to register them with Dash
from UI.callback import target_attribute_callbacks

# Import unified column modal callbacks (Phase 2)
from UI.callback import unified_column_modal_callbacks

# Import table overview controller (Phase 3)
from UI.controller import table_overview_controller

from UI.pages.components import adapt_finetune
from UI.pages.components import adapt_retrain
# Phase 4 column comparison callbacks moved to unified modal system

if __name__ == '__main__':
    # Init tables
    try:
        with server.app_context():
            db.create_all()
            db.session.commit()
    except Exception as e:
        print(str(e))
        sys.stdout.flush()
        with server.app_context():
            db.session.rollback()

    # Check if there are any users, if not, create admin user
    try:
        with server.app_context():
            user = User.query.filter_by(email='admin').first()
            if not user:
                user = User(email='admin', username='admin')
                user.set_password('admin')
                db.session.add(user)
                db.session.commit()
    except Exception as e:
        print(f"Error creating tables: {e}")
        with server.app_context():
            db.session.rollback()
            
    # All components are now directly integrated in their respective layouts

    # Run the server
    # Disable reloader due to errors while writing temp data for sandboxes
    app.run(debug=True, use_reloader=False, dev_tools_hot_reload=False)
