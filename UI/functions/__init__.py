"""
Package initialization for UI functions.
"""

import re
# Import the global vars correctly
from UI.functions.global_vars import global_vars

def parse_code_blocks(reply):
    """
    Parses a reply and extracts code blocks, returning the text parts and code blocks separately.
    
    Args:
        reply (str): The raw reply from the LLM.
        
    Returns:
        list: A list of components, each either a text string or a tuple (code, is_python)
    """
    components = []
    # Updated pattern to better match code blocks with or without language indicator
    pattern = r"```(python|)\s*(.*?)```"
    last_end = 0
    
    for match in re.finditer(pattern, reply, re.DOTALL):
        start, end = match.span()
        lang = match.group(1) or ""
        code = match.group(2).strip()
        
        # Add text before the code block
        if start > last_end:
            text_part = reply[last_end:start]
            if text_part.strip():
                components.append(text_part)
        
        # Add the code block - ensure code is not empty
        if code:
            components.append((code, lang.lower() == "python"))
        
        last_end = end
    
    # Add any remaining text after the last code block
    if last_end < len(reply):
        text_part = reply[last_end:]
        if text_part.strip():
            components.append(text_part)
    
    # If no code blocks were found, just return the full text
    if not components:
        components.append(reply)
        
    return components


def format_reply_to_markdown(reply):
    """
    Converts an LLM reply into proper Markdown format.

    Args:
        reply (str): The raw reply from the LLM.

    Returns:
        str: A Markdown-friendly formatted reply.
    """
    # Remove wrapping curly braces if present
    if reply.startswith("{") and reply.endswith("}"):
        reply = reply[1:-1]

    reply = reply.replace("\\n\\n", "\n\n")
    reply = reply.replace("\\n", "\n")
    
    # Just return the formatted markdown - we'll handle the code blocks separately
    return reply


def query_llm(query, stage, user_id):
    """
    Query the LLM with the given query at the specified stage.
    
    Args:
        query (str): The query to ask the LLM
        stage (str): Current application stage
        user_id (str): User identifier
        
    Returns:
        tuple: (response, media, suggestions, stage, recommended_op, explanation)
    """
    print(query, stage)
    response, media, suggestions, stage, op, explanation = global_vars.agent.run(query, stage)
    global_vars.agent.persist_history(user_id=str(user_id))
    global_vars.suggested_questions = suggestions
    
    # Handle case when op is None
    recommended_op = "Recommended Operation: " + op if op is not None else "No operation recommended"
    
    return response, media, suggestions, stage, recommended_op, explanation
