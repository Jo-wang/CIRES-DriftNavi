"""
Utility functions for the Explain component in driftNavi.

This module provides utility functions for attribute ranking and conditional
distribution analysis used in the Explain component for analyzing distribution
shifts between datasets.
"""

import pandas as pd
import numpy as np


def rank_attributes(metrics_data, k=5):
    """
    Rank attributes based on multiple shift metrics, handling N/A values appropriately.
    
    This function ranks attributes by their combined shift metrics, considering only
    valid (non-N/A) metrics for each attribute. It normalizes ranks for different metrics
    and calculates an average rank for each attribute.
    
    Args:
        metrics_data (list): List of dictionaries containing metric data for each attribute
        k (int): Number of top attributes to return
        
    Returns:
        list: List of top-k attribute names sorted by combined shift metric ranking
    """
    if not metrics_data:
        return []
    
    # Initialize dictionaries to track metrics and rankings
    attribute_scores = {}
    
    # Dynamically detect available metric columns (exclude non-metric columns)
    if not metrics_data:
        return []
    
    # Get all available columns from first row
    all_columns = set(metrics_data[0].keys())
    
    # Define columns that are NOT metrics (exclude these)
    non_metric_columns = {
        "Attribute", "Type", "AddToChat", "AddToExplain", "ExplainAction",
        "PrimaryTargetRelevance", "SecondaryTargetRelevance", "RelevanceDelta",
        "TargetRelevance", "TargetRelevanceScore"
    }
    
    # Get metric columns by excluding non-metric columns
    metric_columns = list(all_columns - non_metric_columns)
    
    print(f"[RANK ATTRIBUTES] Available columns: {sorted(list(all_columns))}")
    print(f"[RANK ATTRIBUTES] Detected metric columns: {sorted(metric_columns)}")
    
    # For each detected metric, extract valid values and calculate normalized ranks
    for metric in metric_columns:
        # Extract valid values (skip N/A)
        valid_attrs = {}
        for row in metrics_data:
            attr_name = row["Attribute"]
            
            # Safely get the metric value, skip if not present
            if metric not in row:
                continue
                
            value = row[metric]
            if value != "N/A" and isinstance(value, (int, float)) or (isinstance(value, str) and value.replace('.', '', 1).isdigit()):
                # Convert string numeric values to float
                if isinstance(value, str):
                    value = float(value)
                valid_attrs[attr_name] = value
        
        # Skip empty metrics
        if not valid_attrs:
            continue
            
        # Determine if smaller values are better based on metric name patterns
        # Metrics where smaller values indicate more significant differences
        smaller_is_better_patterns = ["p_value", "pvalue", "p-value", "significance"]
        smaller_is_better = any(pattern in metric.lower() for pattern in smaller_is_better_patterns)
        
        # Rank attributes based on metric values
        sorted_attrs = sorted(valid_attrs.items(), key=lambda x: x[1], reverse=not smaller_is_better)
        
        # Calculate normalized ranks (0 to 1, higher is better shift indicator)
        n_attrs = len(sorted_attrs)
        for i, (attr, _) in enumerate(sorted_attrs):
            # Calculate normalized rank (0 to 1)
            normalized_rank = (n_attrs - i) / n_attrs if not smaller_is_better else (i + 1) / n_attrs
            
            # Initialize attribute in scores dict if not present
            if attr not in attribute_scores:
                attribute_scores[attr] = {"total_rank": 0, "count": 0}
            
            # Add the normalized rank to the attribute's total
            attribute_scores[attr]["total_rank"] += normalized_rank
            attribute_scores[attr]["count"] += 1
    
    # Calculate average rank for each attribute
    for attr in attribute_scores:
        if attribute_scores[attr]["count"] > 0:
            attribute_scores[attr]["avg_rank"] = attribute_scores[attr]["total_rank"] / attribute_scores[attr]["count"]
        else:
            attribute_scores[attr]["avg_rank"] = 0
    
    # Sort attributes by average rank and take top k
    sorted_attrs = sorted(
        [(attr, scores["avg_rank"]) for attr, scores in attribute_scores.items()],
        key=lambda x: x[1],
        reverse=True
    )
    
    # Return just the attribute names
    return [attr for attr, _ in sorted_attrs[:k]]


def analyze_conditional_distribution(df, target_column, target_value, shifted_column):
    """
    Analyze the distribution of a shifted attribute conditional on a target attribute value.
    
    Args:
        df (pd.DataFrame): Dataset to analyze
        target_column (str): Name of the target attribute to condition on
        target_value (str): Value of the target attribute to filter by
        shifted_column (str): Name of the attribute to analyze distribution for
        
    Returns:
        tuple: (distribution_dict, column_type)
    """
    if df is None or target_column not in df.columns or shifted_column not in df.columns:
        return {}, "unknown"
    
    # Filter data by target value
    if pd.api.types.is_numeric_dtype(df[target_column]):
        # For continuous target attributes with range values like "10.0 to 20.0"
        if " to " in str(target_value):
            low, high = map(float, target_value.split(" to "))
            filtered_df = df[(df[target_column] >= low) & (df[target_column] <= high)]
        else:
            # Try to convert to appropriate numeric type
            try:
                numeric_value = float(target_value)
                filtered_df = df[df[target_column] == numeric_value]
            except (ValueError, TypeError):
                return {}, "unknown"
    else:
        # For categorical target attributes
        filtered_df = df[df[target_column] == target_value]
    
    # If no data matches the filter, return empty
    if len(filtered_df) == 0:
        return {}, "unknown"
    
    # Get the shifted column data
    column_data = filtered_df[shifted_column]
    
    # Determine column type
    if pd.api.types.is_numeric_dtype(column_data):
        # For numeric data, consider it continuous if it has more than 10 unique values
        # and is not a boolean type
        if column_data.dtype == bool or len(column_data.unique()) <= 10:
            column_type = "categorical"
        else:
            column_type = "continuous"
    else:
        # Non-numeric data is always categorical
        column_type = "categorical"
    
    # Calculate distribution based on type
    if column_type == "categorical":
        # For categorical data, count occurrences of each value
        distribution = column_data.value_counts().to_dict()
    else:
        # For continuous data, create bins and count values in each bin
        hist, bin_edges = np.histogram(column_data.dropna(), bins=10)
        
        # Create distribution with bin ranges as keys
        distribution = {}
        for i in range(len(hist)):
            bin_label = f"{bin_edges[i]:.2f} to {bin_edges[i+1]:.2f}"
            distribution[bin_label] = int(hist[i])
    
    return distribution, column_type


def get_target_values_options(df, target_column):
    """
    Get options for target attribute values dropdown.
    
    Args:
        df (pd.DataFrame): Dataset to analyze
        target_column (str): Name of the target attribute
        
    Returns:
        list: List of dictionaries with label and value for dropdown options
    """
    if df is None or target_column not in df.columns:
        return []
    
    column_data = df[target_column]
    
    # Determine column type
    if pd.api.types.is_numeric_dtype(column_data):
        if column_data.dtype == bool or len(column_data.unique()) <= 10:
            # For categorical numeric data, use unique values
            unique_values = sorted(column_data.unique())
            return [{"label": str(val), "value": str(val)} for val in unique_values]
        else:
            # For continuous data, create bins
            hist, bin_edges = np.histogram(column_data.dropna(), bins=10)
            options = []
            for i in range(len(hist)):
                bin_label = f"{bin_edges[i]:.2f} to {bin_edges[i+1]:.2f}"
                options.append({"label": bin_label, "value": bin_label})
            return options
    else:
        # For categorical data, use unique values
        unique_values = sorted(column_data.unique())
        return [{"label": str(val), "value": str(val)} for val in unique_values]
