"""
GPT API handler for Explain component.

This module provides functionality for sending prompts to GPT and processing
responses specifically for the Explain component's distribution shift analysis.
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from flask_login import current_user
import traceback
import logging

# Configure logger
logger = logging.getLogger(__name__)

def generate_response_from_prompt(prompt, model_name=None):
    """
    Generate a response from GPT based on the provided prompt.
    
    Args:
        prompt (str): The prompt to send to GPT
        model_name (str, optional): Specific model to use. Defaults to user's current model.
        
    Returns:
        str: The generated response text
    """
    try:
        # Get the current model preference from user settings if not specified
        if not model_name:
            # First, try to get the model from global_vars.agent (navigation bar selection)
            try:
                from UI.functions.global_vars import global_vars
                if hasattr(global_vars, 'agent') and global_vars.agent and hasattr(global_vars.agent, 'model_name'):
                    model_name = global_vars.agent.model_name
                    print(f"[EXPLAIN API] Using model from global agent: {model_name}")
                else:
                    print(f"[EXPLAIN API] No global agent found, falling back to user preference")
                    raise AttributeError("No global agent available")
            except (ImportError, AttributeError):
                # Fall back to user preference if global_vars not available
                print(f"[EXPLAIN API] Using user preference fallback")
                model_preference = getattr(current_user, 'model_preference', 'gpt-4o-mini')
                
                # Map preferences to actual model names
                model_map = {
                    'gpt4o': 'gpt-4o',
                    'gpt4omini': 'gpt-4o-mini'
                }
                
                model_name = model_map.get(model_preference, 'gpt-4o-mini')
        
        print(f"[EXPLAIN API] Using model: {model_name}")
        
        # Initialize the ChatOpenAI instance
        llm = ChatOpenAI(
            temperature=0.3,  # Lower temperature for more factual responses
            model=model_name
        )
        
        # Create system and human messages
        system_message = SystemMessage(
            content=(
                "You are an expert data scientist analyzing distribution shifts between datasets. "
                "Provide clear, concise analysis focusing on practical implications and actionable advice. "
                "Keep your response under 500 words and organized in paragraphs. "
                "Base your analysis solely on the distribution data provided."
            )
        )
        
        human_message = HumanMessage(content=prompt)
        
        # Generate response
        response = llm.invoke([system_message, human_message])
        
        # Return just the content
        return response.content
        
    except Exception as e:
        logger.error(f"Error generating GPT response: {str(e)}")
        logger.error(traceback.format_exc())
        return f"Error generating analysis. Please try again later. (Error: {str(e)})"
