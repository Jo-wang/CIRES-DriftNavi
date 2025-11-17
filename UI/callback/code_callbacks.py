import dash
import docker.errors
import logging
from UI.app import app
from dash.dependencies import Input, Output, State
from agent import DatasetAgent
from dash import html
import pandas as pd
from UI.functions import *
from flask_login import current_user
import docker
import os
import shutil
import time
import random
import hashlib

logger = logging.getLogger(__name__)

def create_code_execution_trigger(command: str) -> dict:
    """Create trigger data for code execution updates."""
    return {
        'timestamp': time.time(),
        'source': 'CodeExecution',
        'reason': f'Command executed: {command[:50]}...' if len(command) > 50 else f'Command executed: {command}',
        'command': command,
        'trigger_id': hashlib.md5(f"code_{command}_{time.time()}".encode()).hexdigest()[:8]
    }


@app.callback(
    [Output("commands-output", "children"),
     Output("commands-input", "disabled", allow_duplicate=True),
     Output("run-commands", "disabled", allow_duplicate=True),
     Output('table-update-trigger', 'data', allow_duplicate=True),  # Phase 3: Use trigger instead
     Output('data-alert', 'children', allow_duplicate=True),
     Output('data-alert', 'is_open', allow_duplicate=True),
     ],
    Input("run-commands", "n_clicks"),
    State("commands-input", "value"),
    prevent_initial_call=True
)
def execute_commands(n_click, commands):
    if n_click is None or n_click == 0:
        # Initial load, don't do anything
        return ["", False, False, dash.no_update, dash.no_update, dash.no_update]
    if global_vars.df is None and n_click > 0:
        return ["Have you imported a dataset and entered a query?", False, False, dash.no_update, dash.no_update, dash.no_update]
    if n_click > 0 and commands is not None:
        try:
            print("Running sandbox...")
            user_id = str(current_user.id)
            current_path = os.path.dirname(os.path.realpath(__file__))
            parent_path = os.path.dirname(current_path)
            user_data_dir = os.path.join(os.path.dirname(parent_path), 'tmp', user_id)

            if not os.path.exists(user_data_dir):
                print("Creating user's directory...")
                os.makedirs(user_data_dir, exist_ok=True)
                shutil.copyfile(os.path.join(parent_path, 'assets', 'sandbox_main.py'),
                                os.path.join(user_data_dir, 'sandbox_main.py'))
                print("Create user's directory successfully")
            else:
                if not os.path.exists(os.path.join(user_data_dir, 'sandbox_main.py')):
                    shutil.copyfile(os.path.join(parent_path, 'assets', 'sandbox_main.py'),
                                    os.path.join(user_data_dir, 'sandbox_main.py'))

            user_output_file = os.path.join(user_data_dir, f"_output_{user_id}.out")
            user_error_file = os.path.join(user_data_dir, f"_error_{user_id}.err")
            container_name = 'driftnavi-' + user_id

            # Create the preamble code first
            preamble = []
            preamble.append("# Auto-imported libraries")
            preamble.append("import pandas as pd")
            preamble.append("import numpy as np")
            preamble.append("import matplotlib.pyplot as plt")
            preamble.append("import seaborn as sns")
            preamble.append("from scipy import stats")
            preamble.append("")
            preamble.append("# Dataset definitions")
            preamble.append("# Primary dataset")
            preamble.append("df1 = pd.read_csv('df.csv')")
            preamble.append("df = df1  # For backward compatibility")
            
            # Add secondary dataset if available
            if hasattr(global_vars, 'secondary_df') and global_vars.secondary_df is not None:
                preamble.append("# Secondary dataset")
                preamble.append("df2 = pd.read_csv('df_secondary.csv')")
            preamble.append("\n# User code begins here:")
            preamble.append("# " + "-" * 50)
            
            # Combine preamble with user's code for display
            full_code = "\n".join(preamble) + "\n" + commands
            
            # Create a numbered version of the full code
            numbered_commands = []
            for i, line in enumerate(full_code.split('\n'), 1):
                numbered_commands.append(f"{i}: {line}")
            
            # Create a display version of the code with line numbers
            code_with_line_numbers = '\n'.join(numbered_commands)
            
            # Save the full code (without line numbers) for execution
            with open(os.path.join(user_data_dir, 'sandbox_commands.py'), "w") as f:
                f.write(full_code)

            # Save primary dataset (df1)
            global_vars.df.to_csv(os.path.join(user_data_dir, "df.csv"), index=False)
            
            # Save secondary dataset (df2) if available
            if hasattr(global_vars, 'secondary_df') and global_vars.secondary_df is not None:
                global_vars.secondary_df.to_csv(os.path.join(user_data_dir, "df_secondary.csv"), index=False)
            
            if os.path.exists(user_output_file):
                os.remove(user_output_file)
            if os.path.exists(user_error_file):
                os.remove(user_error_file)

            client = docker.from_env()
            print("Running commands inside container")
            user_container = client.containers.get(container_name)
            user_container.exec_run(cmd="python sandbox_main.py " + user_id, workdir='/home/sandbox/' + user_id)
            print("Run commands inside container successfully")

            output = []
            # First, add the code with line numbers and proper formatting
            output.append(html.Div([
                html.H5("Your Code:"),
                html.Div([
                    html.Pre(code_with_line_numbers, 
                             className="code-section",
                             style={"backgroundColor": "#f8f9fa", 
                                    "padding": "10px", 
                                    "border": "1px solid #ddd",
                                    "borderRadius": "5px",
                                    "marginBottom": "15px",
                                    "fontFamily": "monospace",
                                    "maxHeight": "400px",
                                    "overflowY": "auto"})
                ])
            ]))
            
            # Add dataset info
            dataset_info = []
            dataset_info.append("Available variables:")
            dataset_info.append("- df1: Primary dataset" + (f" ({global_vars.file_name})" if hasattr(global_vars, 'file_name') else ""))
            dataset_info.append("- df: Alias for df1 (for backward compatibility)")
            
            if hasattr(global_vars, 'secondary_df') and global_vars.secondary_df is not None:
                dataset_info.append("- df2: Secondary dataset" + (f" ({global_vars.secondary_file_name})" if hasattr(global_vars, 'secondary_file_name') else ""))
            else:
                dataset_info.append("- df2: Not available (no secondary dataset loaded)")
            
            dataset_info.append("\nPre-loaded libraries:")
            dataset_info.append("- pandas (pd): Data manipulation and analysis")
            dataset_info.append("- numpy (np): Numerical computing")
            dataset_info.append("- matplotlib.pyplot (plt): Plotting and visualization")
            dataset_info.append("- seaborn (sns): Statistical data visualization")
            dataset_info.append("- scipy.stats (stats): Statistical functions")
            
            output.append(html.Div([
                html.H5("Dataset Information:"),
                html.Pre("\n".join(dataset_info), 
                         style={"backgroundColor": "#e8f4f8", 
                                "padding": "10px", 
                                "border": "1px solid #d1e0e5",
                                "borderRadius": "5px",
                                "marginBottom": "15px",
                                "fontFamily": "monospace"})
            ]))
            
            # Add output or error message
            if os.path.isfile(user_output_file):
                with open(user_output_file, "r") as f:
                    output_content = f.read()
                    if output_content.strip():
                        output.append(html.Div([
                            html.H5("Output:"),
                            html.Pre(output_content, 
                                    style={"backgroundColor": "#f8f9fa", 
                                           "padding": "10px", 
                                           "border": "1px solid #ddd",
                                           "borderRadius": "5px",
                                           "color": "black",
                                           "fontFamily": "monospace"})
                        ]))
                    else:
                        output.append(html.P("Code executed successfully with no output.", 
                                           style={"color": "green", "marginTop": "10px"}))

            if os.path.isfile(user_error_file):
                with open(user_error_file, "r") as f:
                    error_content = f.read()
                    if error_content.strip():
                        output.append(html.Div([
                            html.H5("Error:", style={"color": "#d9534f"}),
                            html.Div([
                                html.Pre(error_content, 
                                        className="error-section",
                                        style={"backgroundColor": "#fff0f0", 
                                               "padding": "10px", 
                                               "border": "1px solid #ffcccc",
                                               "borderLeft": "4px solid #d9534f",
                                               "borderRadius": "5px",
                                               "marginBottom": "15px",
                                               "fontFamily": "monospace",
                                               "maxHeight": "300px",
                                               "overflowY": "auto",
                                               "color": "#cc0000"})
                            ])
                        ]))

            # Read updated datasets back
            new_df = pd.read_csv(os.path.join(user_data_dir, "df.csv"))
            global_vars.df = new_df
            
            # Update secondary dataset if it exists
            secondary_df_path = os.path.join(user_data_dir, "df_secondary.csv")
            if os.path.exists(secondary_df_path):
                global_vars.secondary_df = pd.read_csv(secondary_df_path)
            
            # Detect dataset changes and invalidate metrics cache if needed
            changes = global_vars.detect_dataset_changes()
            if changes['any_changed']:
                print(f"[SANDBOX] Dataset changes detected after sandbox execution")
            
            global_vars.conversation_session = f"{int(time.time() * 1000)}-{random.randint(1000, 9999)}"
            global_vars.agent = DatasetAgent(global_vars.df, file_name=global_vars.file_name, conversation_session=global_vars.conversation_session)
            
            # Update secondary dataset in the agent if it exists
            if hasattr(global_vars, 'secondary_df') and global_vars.secondary_df is not None:
                global_vars.agent.add_secondary_dataset(global_vars.secondary_df, global_vars.secondary_file_name if hasattr(global_vars, 'secondary_file_name') else "Secondary Dataset")
            
            # Create trigger for Phase 3 table update
            trigger_data = create_code_execution_trigger(commands)
            
            return [output, 
                    False, 
                    False,
                    trigger_data,  # Phase 3: Use trigger instead of direct table update
                    "The data might have been changed.", True]

        except Exception as e:
            if isinstance(e, docker.errors.NotFound):
                try:
                    print("Recreate container")
                    client = docker.from_env()
                    container_name = 'driftnavi-' + user_id
                    client.containers.run('daisyy512/hello-docker',
                                          name=container_name,
                                          volumes=[user_data_dir + ':/home/sandbox/' + user_id],
                                          detach=True,
                                          tty=True)
                    print("Create container successfully")
                    user_container = client.containers.get(container_name)
                    user_container.exec_run(cmd="python sandbox_main.py " + user_id, workdir='/home/sandbox/' + user_id)
                    output = []
                    # First, add the code with line numbers and proper formatting
                    output.append(html.Div([
                        html.H5("Your Code:"),
                        html.Div([
                            html.Pre(code_with_line_numbers, 
                                     className="code-section",
                                     style={"backgroundColor": "#f8f9fa", 
                                            "padding": "10px", 
                                            "border": "1px solid #ddd",
                                            "borderRadius": "5px",
                                            "marginBottom": "15px",
                                            "fontFamily": "monospace",
                                            "maxHeight": "400px",
                                            "overflowY": "auto"})
                        ])
                    ]))
                    
                    # Add dataset info
                    dataset_info = []
                    dataset_info.append("Available variables:")
                    dataset_info.append("- df1: Primary dataset" + (f" ({global_vars.file_name})" if hasattr(global_vars, 'file_name') else ""))
                    dataset_info.append("- df: Alias for df1 (for backward compatibility)")
                    
                    if hasattr(global_vars, 'secondary_df') and global_vars.secondary_df is not None:
                        dataset_info.append("- df2: Secondary dataset" + (f" ({global_vars.secondary_file_name})" if hasattr(global_vars, 'secondary_file_name') else ""))
                    else:
                        dataset_info.append("- df2: Not available (no secondary dataset loaded)")
                    
                    dataset_info.append("\nPre-loaded libraries:")
                    dataset_info.append("- pandas (pd): Data manipulation and analysis")
                    dataset_info.append("- numpy (np): Numerical computing")
                    dataset_info.append("- matplotlib.pyplot (plt): Plotting and visualization")
                    dataset_info.append("- seaborn (sns): Statistical data visualization")
                    dataset_info.append("- scipy.stats (stats): Statistical functions")
                    
                    output.append(html.Div([
                        html.H5("Dataset Information:"),
                        html.Pre("\n".join(dataset_info), 
                                 style={"backgroundColor": "#e8f4f8", 
                                        "padding": "10px", 
                                        "border": "1px solid #d1e0e5",
                                        "borderRadius": "5px",
                                        "marginBottom": "15px",
                                        "fontFamily": "monospace"})
                    ]))
                    
                    # Add output or error message
                    if os.path.isfile(user_output_file):
                        with open(user_output_file, "r") as f:
                            output_content = f.read()
                            if output_content.strip():
                                output.append(html.Div([
                                    html.H5("Output:"),
                                    html.Pre(output_content, 
                                            style={"backgroundColor": "#f8f9fa", 
                                                   "padding": "10px", 
                                                   "border": "1px solid #ddd",
                                                   "borderRadius": "5px",
                                                   "color": "black",
                                                   "fontFamily": "monospace"})
                                ]))
                            else:
                                output.append(html.P("Code executed successfully with no output.", 
                                                   style={"color": "green", "marginTop": "10px"}))

                    if os.path.isfile(user_error_file):
                        with open(user_error_file, "r") as f:
                            error_content = f.read()
                            if error_content.strip():
                                output.append(html.Div([
                                    html.H5("Error:", style={"color": "#d9534f"}),
                                    html.Div([
                                        html.Pre(error_content, 
                                                className="error-section",
                                                style={"backgroundColor": "#fff0f0", 
                                                       "padding": "10px", 
                                                       "border": "1px solid #ffcccc",
                                                       "borderLeft": "4px solid #d9534f",
                                                       "borderRadius": "5px",
                                                       "marginBottom": "15px",
                                                       "fontFamily": "monospace",
                                                       "maxHeight": "300px",
                                                       "overflowY": "auto",
                                                       "color": "#cc0000"})
                                    ])
                                ]))

                    # Read updated datasets back
                    new_df = pd.read_csv(os.path.join(user_data_dir, "df.csv"))
                    global_vars.df = new_df
                    
                    # Update secondary dataset if it exists
                    secondary_df_path = os.path.join(user_data_dir, "df_secondary.csv")
                    if os.path.exists(secondary_df_path):
                        global_vars.secondary_df = pd.read_csv(secondary_df_path)
                    
                    # Detect dataset changes and invalidate metrics cache if needed
                    changes = global_vars.detect_dataset_changes()
                    if changes['any_changed']:
                        print(f"[SANDBOX] Dataset changes detected after sandbox execution")
                    
                    global_vars.conversation_session = f"{int(time.time() * 1000)}-{random.randint(1000, 9999)}"
                    global_vars.agent = DatasetAgent(global_vars.df, file_name=global_vars.file_name, conversation_session=global_vars.conversation_session)
                    
                    # Update secondary dataset in the agent if it exists
                    if hasattr(global_vars, 'secondary_df') and global_vars.secondary_df is not None:
                        global_vars.agent.add_secondary_dataset(global_vars.secondary_df, global_vars.secondary_file_name if hasattr(global_vars, 'secondary_file_name') else "Secondary Dataset")
                    
                    # Create trigger for Phase 3 table update
                    trigger_data = create_code_execution_trigger(commands)
                    
                    return [output, 
                            False, 
                            False, 
                            trigger_data,  # Phase 3: Use trigger instead of direct table update
                            "The data might have been changed.", True]

                except Exception as e:
                    print("Create container failed: ", e)
                    return ["Sandbox is not available", False, False, dash.no_update, dash.no_update, dash.no_update]
            elif isinstance(e, docker.errors.APIError):
                if e.status_code is not None and e.status_code == 409:
                    user_container = client.containers.get(container_name)
                    print("Restart container")
                    user_container.start()
                    user_container.exec_run(cmd="python sandbox_main.py " + user_id, workdir='/home/sandbox/' + user_id)

                    output = []
                    # First, add the code with line numbers and proper formatting
                    output.append(html.Div([
                        html.H5("Your Code:"),
                        html.Div([
                            html.Pre(code_with_line_numbers, 
                                     className="code-section",
                                     style={"backgroundColor": "#f8f9fa", 
                                            "padding": "10px", 
                                            "border": "1px solid #ddd",
                                            "borderRadius": "5px",
                                            "marginBottom": "15px",
                                            "fontFamily": "monospace",
                                            "maxHeight": "400px",
                                            "overflowY": "auto"})
                        ])
                    ]))
                    
                    # Add dataset info
                    dataset_info = []
                    dataset_info.append("Available variables:")
                    dataset_info.append("- df1: Primary dataset" + (f" ({global_vars.file_name})" if hasattr(global_vars, 'file_name') else ""))
                    dataset_info.append("- df: Alias for df1 (for backward compatibility)")
                    
                    if hasattr(global_vars, 'secondary_df') and global_vars.secondary_df is not None:
                        dataset_info.append("- df2: Secondary dataset" + (f" ({global_vars.secondary_file_name})" if hasattr(global_vars, 'secondary_file_name') else ""))
                    else:
                        dataset_info.append("- df2: Not available (no secondary dataset loaded)")
                    
                    dataset_info.append("\nPre-loaded libraries:")
                    dataset_info.append("- pandas (pd): Data manipulation and analysis")
                    dataset_info.append("- numpy (np): Numerical computing")
                    dataset_info.append("- matplotlib.pyplot (plt): Plotting and visualization")
                    dataset_info.append("- seaborn (sns): Statistical data visualization")
                    dataset_info.append("- scipy.stats (stats): Statistical functions")
                    
                    output.append(html.Div([
                        html.H5("Dataset Information:"),
                        html.Pre("\n".join(dataset_info), 
                                 style={"backgroundColor": "#e8f4f8", 
                                        "padding": "10px", 
                                        "border": "1px solid #d1e0e5",
                                        "borderRadius": "5px",
                                        "marginBottom": "15px",
                                        "fontFamily": "monospace"})
                    ]))
                    
                    # Add output or error message
                    if os.path.isfile(user_output_file):
                        with open(user_output_file, "r") as f:
                            output_content = f.read()
                            if output_content.strip():
                                output.append(html.Div([
                                    html.H5("Output:"),
                                    html.Pre(output_content, 
                                            style={"backgroundColor": "#f8f9fa", 
                                                   "padding": "10px", 
                                                   "border": "1px solid #ddd",
                                                   "borderRadius": "5px",
                                                   "color": "black",
                                                   "fontFamily": "monospace"})
                                ]))
                            else:
                                output.append(html.P("Code executed successfully with no output.", 
                                                   style={"color": "green", "marginTop": "10px"}))

                    if os.path.isfile(user_error_file):
                        with open(user_error_file, "r") as f:
                            error_content = f.read()
                            if error_content.strip():
                                output.append(html.Div([
                                    html.H5("Error:", style={"color": "#d9534f"}),
                                    html.Div([
                                        html.Pre(error_content, 
                                                className="error-section",
                                                style={"backgroundColor": "#fff0f0", 
                                                       "padding": "10px", 
                                                       "border": "1px solid #ffcccc",
                                                       "borderLeft": "4px solid #d9534f",
                                                       "borderRadius": "5px",
                                                       "marginBottom": "15px",
                                                       "fontFamily": "monospace",
                                                       "maxHeight": "300px",
                                                       "overflowY": "auto",
                                                       "color": "#cc0000"})
                                    ])
                                ]))
                    # Create trigger for Phase 3 table update
                    trigger_data = create_code_execution_trigger(commands)
                    
                    return [output, 
                            False, 
                            False,
                            trigger_data,  # Phase 3: Use trigger instead of direct table update
                            "The data might have been changed.", True]

                return [str(e), False, False, dash.no_update, dash.no_update, dash.no_update]
            else:
                logger.error(type(e))
                logger.error(e)
                return [str(e), False, False, dash.no_update, dash.no_update, dash.no_update]
    return [dash.no_update, False, False, dash.no_update, dash.no_update, dash.no_update]
