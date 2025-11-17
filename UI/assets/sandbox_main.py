import ast
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
from os.path import basename
import sys
import traceback

if __name__ == '__main__':
    try:
        user_id = sys.argv[1]
        
        # Load primary dataset as df1
        df1 = pd.read_csv("df.csv")
        
        # Create a global df reference for backward compatibility
        df = df1
        
        # Load secondary dataset as df2 if it exists
        df2 = None
        if os.path.exists("df_secondary.csv"):
            df2 = pd.read_csv("df_secondary.csv")
        
        with open('sandbox_commands.py', 'r') as file:
            commands = file.read()
            
        tree = ast.parse(commands)
        io_buffer = StringIO()
        
        # Create a globals dictionary with the dataframes and commonly used libraries
        globals_dict = {
            # Dataframes
            'df1': df1,
            'df': df,
            'df2': df2,
            
            # Data manipulation libraries
            'pd': pd,
            'np': np,
            
            # Visualization libraries
            'plt': plt,
            'sns': sns,
            
            # Statistical analysis
            'stats': stats,
            
            # System libraries
            'os': os,
        }
        
        with open(f"_output_{user_id}.out", "w") as f:
            with redirect_stdout(f):
                module = ast.Module(tree.body, type_ignores=[])
                # Pass the variables to the execution context
                exec(ast.unparse(module), globals_dict) 
        
        # Get the potentially modified dataframes back from globals
        df1 = globals_dict.get('df1', df1)
        df2 = globals_dict.get('df2', df2)
        
        # Save only the primary dataset back (df1)
        df1.to_csv("df.csv", mode="w", index=False)
        
        # Save the secondary dataset if it exists and was modified
        if df2 is not None:
            df2.to_csv("df_secondary.csv", mode="w", index=False)
            
    except Exception as e:
        with open(f"_error_{user_id}.err", "w") as f:
            # Enhanced error message with line numbers
            error_type = type(e).__name__
            error_msg = str(e)
            
            # Get traceback information with line numbers
            tb_lines = traceback.format_exc().splitlines()
            
            # Find which line in user code caused the error
            user_code_line = None
            for line in tb_lines:
                if 'sandbox_commands.py' in line:
                    parts = line.split('line ')
                    if len(parts) > 1:
                        try:
                            user_code_line = int(parts[1].split(',')[0])
                        except ValueError:
                            pass
            
            # Format the error message with line information
            if user_code_line is not None:
                # Add line numbers to the original code for reference
                with open('sandbox_commands.py', 'r') as code_file:
                    numbered_code = []
                    for i, line in enumerate(code_file.readlines(), 1):
                        prefix = f"{i}: "
                        if i == user_code_line:
                            prefix = f"{i}: --> "
                        numbered_code.append(prefix + line.rstrip())
                
                f.write(f"Error on line {user_code_line}: {error_type}: {error_msg}\n\n")
                f.write("Code with line numbers:\n")
                f.write("\n".join(numbered_code))
            else:
                f.write(f"{error_type}: {error_msg}\n")
                f.write(traceback.format_exc())