"""
State connector module for driftNavi application.

This module provides interface functions that facilitate communication between
different components of the application, particularly bridging the Detect and 
Explain functionalities for a seamless analysis experience.
"""

from UI.functions.global_vars import global_vars
import json
import pandas as pd
from datetime import datetime


def save_detect_results(metrics_data, focus_attribute=None):
    """
    Save results from the Detect phase for use in the Explain phase.
    
    This function acts as a bridge between Detect and Explain components,
    storing metrics results and setting a focus attribute if specified.
    
    Args:
        metrics_data (dict): The metrics data calculated in Detect phase
        focus_attribute (str, optional): Attribute to set as current focus
        
    Returns:
        bool: True if successfully saved, False otherwise
    """
    try:
        # Store metrics data for reuse
        global_vars.store_metrics_results(metrics_data)
        
        # Record timestamp of this analysis
        timestamp = datetime.now().isoformat()
        
        # If a specific attribute is focused on, update the context
        if focus_attribute:
            global_vars.set_focus_attribute(focus_attribute, source="detect")
            
            # Add to analysis path
            global_vars.analysis_context["analysis_path"].append({
                "action": "detect_save",
                "timestamp": timestamp,
                "metrics_saved": True,
                "focus_attribute": focus_attribute
            })
        else:
            global_vars.analysis_context["analysis_path"].append({
                "action": "detect_save",
                "timestamp": timestamp,
                "metrics_saved": True
            })
            
        return True
        
    except Exception as e:
        print(f"Error saving detect results: {str(e)}")
        return False


def get_explain_context(stage=None):
    """
    Retrieve context data for the Explain or Detect phase based on results.
    
    Args:
        stage (str, optional): The current stage requesting context ("detect" or "explain").
                               If None, defaults to "explain".
    
    Returns:
        dict: Context data containing focus attribute, metrics data, and other
              information needed for Explain/Detect phase
    """
    # Get metrics data from cache properly
    metrics_data = None
    if hasattr(global_vars, 'metrics_cache') and global_vars.metrics_cache is not None:
        if isinstance(global_vars.metrics_cache, dict) and 'data' in global_vars.metrics_cache:
            metrics_data = global_vars.metrics_cache['data']
    
    # Get context data from global_vars if available
    context_data = []
    if hasattr(global_vars, 'explain_context_data') and global_vars.explain_context_data:
        context_data = global_vars.explain_context_data
        print(f"[GET_EXPLAIN_CONTEXT] Found {len(context_data)} context items in global_vars")
    else:
        print(f"[GET_EXPLAIN_CONTEXT] No context data found in global_vars")
    
    context = {
        "focus_attribute": global_vars.analysis_context.get("current_focus"),
        "metrics_data": metrics_data,  # Use the actual data, not the whole cache
        "selected_metrics": global_vars.selected_metrics,
        "target_attribute": global_vars.target_attribute,
        "target_stats": global_vars.target_attribute_stats,
        "previous_stage": global_vars.analysis_context.get("previous_stage"),
        "analysis_path": global_vars.analysis_context.get("analysis_path", []),
        "context_data": context_data  # ✅ ADD: Include context items data
    }
    
    # Determine actual stage for logging
    actual_stage = stage if stage else "explain"
    
    # Add timestamp for this context retrieval
    global_vars.analysis_context["analysis_path"].append({
        "action": f"{actual_stage}_context_get",
        "timestamp": datetime.now().isoformat()
    })
    
    return context


def record_analysis_path(action, details=None):
    """
    Record an action in the analysis path for context continuity.
    
    Args:
        action (str): Type of action being recorded
        details (dict, optional): Additional details about the action
        
    Returns:
        bool: True if successfully recorded, False otherwise
    """
    try:
        path_entry = {
            "action": action,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add any additional details
        if details:
            path_entry.update(details)
            
        global_vars.analysis_context["analysis_path"].append(path_entry)
        return True
        
    except Exception as e:
        print(f"Error recording analysis path: {str(e)}")
        return False


def update_target_relationship(attribute_name, relationship_type, value):
    """
    Update the relationship between an attribute and the target attribute.
    
    Args:
        attribute_name (str): Name of the attribute
        relationship_type (str): Type of relationship (correlation, impact, etc.)
        value (float): Strength/value of the relationship
        
    Returns:
        bool: True if successfully updated, False otherwise
    """
    try:
        if relationship_type == "impact":
            global_vars.target_attribute_stats["impact_scores"][attribute_name] = value
            
            # If impact is high, add to related attributes if not already there
            if abs(value) > 0.3:  # Threshold for "high" impact
                if attribute_name not in global_vars.target_attribute_stats["related_attributes"]:
                    global_vars.target_attribute_stats["related_attributes"].append(attribute_name)
        
        # Record this update in analysis path
        global_vars.analysis_context["analysis_path"].append({
            "action": "target_relationship_update",
            "attribute": attribute_name,
            "relationship_type": relationship_type,
            "value": value,
            "timestamp": datetime.now().isoformat()
        })
        
        return True
        
    except Exception as e:
        print(f"Error updating target relationship: {str(e)}")
        return False


def get_recommended_attributes():
    """
    Get attributes recommended for analysis based on target relationships.
    
    Returns:
        list: List of attribute names recommended for further analysis
    """
    # Start with explicitly related attributes
    recommended = global_vars.target_attribute_stats["related_attributes"].copy()
    
    # Add high impact attributes not already in the list
    for attr, score in global_vars.target_attribute_stats["impact_scores"].items():
        if abs(score) > 0.2 and attr not in recommended:
            recommended.append(attr)
    
    # Sort by impact if possible
    if global_vars.target_attribute_stats["impact_scores"]:
        recommended.sort(
            key=lambda x: abs(global_vars.target_attribute_stats["impact_scores"].get(x, 0)), 
            reverse=True
        )
    
    return recommended
