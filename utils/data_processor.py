"""
Data preprocessing and type detection utilities for driftNavi datasets.
"""
import pandas as pd
import numpy as np
from datetime import datetime
import warnings

def preprocess_datasets(primary_df, secondary_df=None):
    """
    Preprocess both datasets by attempting to convert string data to numeric or datetime
    while maintaining consistency between datasets.
    
    Args:
        primary_df (DataFrame): The primary dataset
        secondary_df (DataFrame, optional): The secondary dataset if available
    
    Returns:
        tuple: A tuple containing (processed_primary_df, processed_secondary_df, column_types)
    """
    # Make copies to avoid modifying the original dataframes
    primary_df = primary_df.copy()
    
    # Prepare result for column types classification
    column_types = {}
    
    # Process datasets differently based on whether secondary dataset exists
    if secondary_df is not None:
        secondary_df = secondary_df.copy()
        
        # Find common columns between the two datasets
        common_columns = set(primary_df.columns).intersection(set(secondary_df.columns))
        
        # Process each column in the primary dataset
        for col in primary_df.columns:
            if col in common_columns:
                # For common columns, ensure consistent types between datasets
                primary_df[col], secondary_df[col], column_type = process_column_pair(
                    primary_df[col], secondary_df[col]
                )
            else:
                # For primary-only columns
                primary_df[col], _, column_type = process_column_solo(primary_df[col])
            
            column_types[col] = column_type
        
        # Process secondary-only columns
        for col in secondary_df.columns:
            if col not in common_columns:
                secondary_df[col], _, column_type = process_column_solo(secondary_df[col])
                column_types[col] = column_type
    else:
        # Process primary dataset only
        for col in primary_df.columns:
            primary_df[col], _, column_type = process_column_solo(primary_df[col])
            column_types[col] = column_type
    
    return primary_df, secondary_df, column_types

def process_column_pair(primary_series, secondary_series):
    """
    Process a pair of columns from two datasets to ensure consistent data types.
    
    Args:
        primary_series (Series): Column from primary dataset
        secondary_series (Series): Column from secondary dataset
    
    Returns:
        tuple: Processed primary series, processed secondary series, and detected column type
    """
    # Try numeric conversion first
    primary_numeric, secondary_numeric, numeric_success = try_numeric_conversion(primary_series, secondary_series)
    if numeric_success:
        # Further classify numeric columns
        if len(set(primary_numeric.dropna().unique()) | set(secondary_numeric.dropna().unique())) == 2:
            return primary_numeric, secondary_numeric, "Binary"
        else:
            return primary_numeric, secondary_numeric, "Continuous"
    
    # Try datetime conversion if numeric failed
    primary_datetime, secondary_datetime, datetime_success = try_datetime_conversion(primary_series, secondary_series)
    if datetime_success:
        return primary_datetime, secondary_datetime, "Datetime"
    
    # Default to categorical if other conversions fail
    return primary_series, secondary_series, "Categorical"

def process_column_solo(series):
    """
    Process a single column when there's no matching column in the other dataset.
    
    Args:
        series (Series): The column to process
        
    Returns:
        tuple: Processed series, None (placeholder for second series), and detected column type
    """
    # Try numeric conversion
    numeric_series, nan_increased = try_numeric_conversion_solo(series)
    if not nan_increased:
        # Classify as binary or continuous
        if len(set(numeric_series.dropna().unique())) == 2:
            return numeric_series, None, "Binary"
        else:
            return numeric_series, None, "Continuous"
    
    # Try datetime conversion
    datetime_series, nan_increased = try_datetime_conversion_solo(series)
    if not nan_increased:
        return datetime_series, None, "Datetime"
    
    # Default to categorical
    return series, None, "Categorical"

def try_numeric_conversion(primary_series, secondary_series):
    """
    Try to convert both series to numeric type if it doesn't increase NaN count.
    
    Returns:
        tuple: Converted primary series, converted secondary series, and success flag
    """
    # Count original NaNs
    primary_nan_count = primary_series.isna().sum()
    secondary_nan_count = secondary_series.isna().sum()
    
    # Try conversion with pd.to_numeric
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            primary_numeric = pd.to_numeric(primary_series, errors='coerce')
            secondary_numeric = pd.to_numeric(secondary_series, errors='coerce')
            
            # Check if NaN count increased
            primary_nan_increased = primary_numeric.isna().sum() > primary_nan_count
            secondary_nan_increased = secondary_numeric.isna().sum() > secondary_nan_count
            
            if not primary_nan_increased and not secondary_nan_increased:
                return primary_numeric, secondary_numeric, True
        except:
            pass
    
    return primary_series, secondary_series, False

def try_datetime_conversion(primary_series, secondary_series):
    """
    Try to convert both series to datetime type if it doesn't increase NaN count.
    
    Returns:
        tuple: Converted primary series, converted secondary series, and success flag
    """
    # Count original NaNs
    primary_nan_count = primary_series.isna().sum()
    secondary_nan_count = secondary_series.isna().sum()
    
    # Try conversion with pd.to_datetime
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            primary_datetime = pd.to_datetime(primary_series, errors='coerce')
            secondary_datetime = pd.to_datetime(secondary_series, errors='coerce')
            
            # Check if NaN count increased
            primary_nan_increased = primary_datetime.isna().sum() > primary_nan_count
            secondary_nan_increased = secondary_datetime.isna().sum() > secondary_nan_count
            
            if not primary_nan_increased and not secondary_nan_increased:
                return primary_datetime, secondary_datetime, True
        except:
            pass
    
    return primary_series, secondary_series, False

def try_numeric_conversion_solo(series):
    """
    Try to convert a single series to numeric if it doesn't increase NaN count.
    
    Returns:
        tuple: Converted series and whether NaN count increased
    """
    # Count original NaNs
    nan_count = series.isna().sum()
    
    # Try conversion
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            numeric_series = pd.to_numeric(series, errors='coerce')
            nan_increased = numeric_series.isna().sum() > nan_count
            if not nan_increased:
                return numeric_series, False
        except:
            pass
    
    return series, True

def try_datetime_conversion_solo(series):
    """
    Try to convert a single series to datetime if it doesn't increase NaN count.
    
    Returns:
        tuple: Converted series and whether NaN count increased
    """
    # Count original NaNs
    nan_count = series.isna().sum()
    
    # Try conversion
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            datetime_series = pd.to_datetime(series, errors='coerce')
            nan_increased = datetime_series.isna().sum() > nan_count
            if not nan_increased:
                return datetime_series, False
        except:
            pass
    
    return series, True

def get_column_type_summary(column_types):
    """
    Generate a summary of column types for UI display.
    
    Args:
        column_types (dict): Dictionary of column names to their types
        
    Returns:
        dict: Summary with counts of each type
    """
    type_counts = {
        "Binary": 0,
        "Continuous": 0,
        "Datetime": 0,
        "Categorical": 0
    }
    
    for col_type in column_types.values():
        if col_type in type_counts:
            type_counts[col_type] += 1
    
    return type_counts


def identify_categorical_encoding_columns(primary_df, secondary_df=None, column_types=None):
    """
    Identify categorical columns that contain string values and will be encoded as numbers during analysis.
    
    Args:
        primary_df (DataFrame): The primary dataset
        secondary_df (DataFrame, optional): The secondary dataset if available
        column_types (dict, optional): Dictionary of column names to their types
        
    Returns:
        list: List of column names that are categorical with string values and will be encoded
    """
    categorical_encoding_columns = []
    
    if primary_df is None:
        return categorical_encoding_columns
    
    # Check each column in the primary dataset
    for col_name in primary_df.columns:
        # Check if this column is marked as categorical
        is_categorical = False
        if column_types and col_name in column_types:
            is_categorical = column_types[col_name] == 'Categorical'
        
        # If not explicitly marked, infer from data type
        if not is_categorical:
            is_categorical = (
                pd.api.types.is_object_dtype(primary_df[col_name]) or 
                pd.api.types.is_categorical_dtype(primary_df[col_name])
            )
        
        # If it's categorical, check if it contains string values
        if is_categorical:
            contains_strings = False
            
            # Check primary dataset
            if pd.api.types.is_object_dtype(primary_df[col_name]):
                # Sample some values to check if they're strings
                sample_values = primary_df[col_name].dropna().head(10)
                contains_strings = any(isinstance(val, str) for val in sample_values)
            
            # Also check secondary dataset if available
            if not contains_strings and secondary_df is not None and col_name in secondary_df.columns:
                if pd.api.types.is_object_dtype(secondary_df[col_name]):
                    sample_values = secondary_df[col_name].dropna().head(10)
                    contains_strings = any(isinstance(val, str) for val in sample_values)
            
            # If the column contains strings, it will be encoded
            if contains_strings:
                categorical_encoding_columns.append(col_name)
    
    return categorical_encoding_columns


def create_encoding_notification_text(categorical_columns, is_comprehensive=True):
    """
    Create a user-friendly notification text about categorical encoding.
    
    Args:
        categorical_columns (list): List of categorical column names that will be encoded
        is_comprehensive (bool): Whether to show comprehensive or brief explanation
        
    Returns:
        str: Formatted notification text
    """
    if not categorical_columns:
        return ""
    
    if len(categorical_columns) == 0:
        return ""
    
    # Format column names
    if len(categorical_columns) == 1:
        column_text = f"'{categorical_columns[0]}'"
    elif len(categorical_columns) <= 3:
        column_text = "', '".join(categorical_columns[:-1]) + f"' and '{categorical_columns[-1]}'"
        column_text = f"'{column_text}"
    else:
        column_text = "', '".join(categorical_columns[:3]) + "' and others"
        column_text = f"'{column_text}"
    
    if is_comprehensive:
        # Comprehensive explanation for upload stage
        base_text = f"\n\n💡 **Encoding Information**: The categorical column(s) {column_text} contain(s) text values that will be automatically converted to numbers (0, 1, 2, etc.) during statistical analysis. This conversion is necessary for mathematical calculations while preserving your original text values in the data preview."
    else:
        # Brief explanation for detect stage
        base_text = f"Categorical columns {column_text} have been encoded as numbers for statistical analysis."
    
    return base_text
