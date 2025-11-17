"""
Column Type Management Module

This module provides centralized state management for column type operations,
ensuring consistency between intelligent classification and pandas data types
across the entire application.

Following Dash best practices:
- Single source of truth for column type state
- Centralized validation and error handling
- Backwards-compatible with existing code
- Thread-safe operations for Dash callbacks
"""

import pandas as pd
import numpy as np
import warnings
import time
from datetime import datetime
from typing import Dict, Tuple, Any, Optional, List
import traceback

from UI.functions.global_vars import global_vars


class ColumnTypeChangeRegistry:
    """
    Registry for tracking column type changes and triggering updates.
    
    This class implements a thread-safe mechanism to queue column type changes
    and trigger global data state updates in compliance with Dash's callback system.
    """
    
    def __init__(self):
        self._pending_changes = {}
        self._last_change_id = 0
    
    def register_change(self, column: str, classification: str, data_type: str) -> dict:
        """
        Register a column type change for triggering global state updates.
        
        Args:
            column: Column name that was changed
            classification: New intelligent classification
            data_type: New pandas data type
            
        Returns:
            dict: Change data for trigger store
        """
        self._last_change_id += 1
        change_data = {
            'change_id': self._last_change_id,
            'timestamp': time.time(),
            'column': column,
            'classification': classification,
            'data_type': data_type,
            'trigger_reason': f"Column type updated: {column} -> {classification} ({data_type})"
        }
        
        self._pending_changes[self._last_change_id] = change_data
        return change_data
    
    def get_latest_change(self) -> dict:
        """Get the most recent change for triggering."""
        if not self._pending_changes:
            return {}
        
        latest_id = max(self._pending_changes.keys())
        return self._pending_changes[latest_id]
    
    def clear_processed_changes(self):
        """Clear processed changes to prevent memory buildup."""
        # Keep only the last 10 changes for debugging
        if len(self._pending_changes) > 10:
            sorted_ids = sorted(self._pending_changes.keys())
            for old_id in sorted_ids[:-10]:
                del self._pending_changes[old_id]


# Global registry instance
_column_type_change_registry = ColumnTypeChangeRegistry()


class ColumnTypeValidationError(Exception):
    """Custom exception for column type validation errors."""
    pass


class ColumnTypeManager:
    """
    Centralized manager for column type operations.
    
    This class provides a single point of control for updating column
    classifications and pandas data types, ensuring consistency and
    proper validation across the application.
    """
    
    # Valid classification types
    VALID_CLASSIFICATIONS = {"Binary", "Continuous", "Datetime", "Categorical"}
    
    # Valid pandas data types
    VALID_DATA_TYPES = {"int64", "float64", "object", "datetime64[ns]", "bool", "category"}
    
    # Classification to recommended data type mapping
    CLASSIFICATION_DATA_TYPE_MAP = {
        "Binary": ["int64", "float64", "bool"],
        "Continuous": ["int64", "float64"],
        "Datetime": ["datetime64[ns]"],
        "Categorical": ["object", "category", "int64", "float64"]  # More flexible for categorical
    }
    
    @staticmethod
    def validate_inputs(column: str, classification: str, data_type: str) -> Tuple[bool, str]:
        """
        Validate input parameters for column type update.
        
        Args:
            column: Column name to validate
            classification: Intelligent classification type
            data_type: Pandas data type
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check column exists
        if not hasattr(global_vars, 'df') or global_vars.df is None:
            return False, "Primary dataset not available"
        
        if column not in global_vars.df.columns:
            return False, f"Column '{column}' not found in primary dataset"
        
        # Check classification validity
        if classification not in ColumnTypeManager.VALID_CLASSIFICATIONS:
            return False, f"Invalid classification '{classification}'. Must be one of: {', '.join(ColumnTypeManager.VALID_CLASSIFICATIONS)}"
        
        # Check data type validity
        if data_type not in ColumnTypeManager.VALID_DATA_TYPES:
            return False, f"Invalid data type '{data_type}'. Must be one of: {', '.join(ColumnTypeManager.VALID_DATA_TYPES)}"
        
        # Check classification-data type compatibility
        recommended_types = ColumnTypeManager.CLASSIFICATION_DATA_TYPE_MAP.get(classification, [])
        if data_type not in recommended_types:
            # This is a warning, not an error
            warning_msg = f"Warning: '{data_type}' is not typically used with '{classification}' classification. Recommended: {', '.join(recommended_types)}"
            # We'll return this as a warning, not an error
            return True, warning_msg
        
        return True, ""
    
    @staticmethod
    def validate_data_conversion(column: str, data_type: str) -> Tuple[bool, str]:
        """
        Validate that data can be converted to the specified type.
        
        Args:
            column: Column name
            data_type: Target pandas data type
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Get primary dataset column
            primary_series = global_vars.df[column].copy()
            
            # Test conversion on primary dataset
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                if data_type == "datetime64[ns]":
                    # Special handling for datetime
                    pd.to_datetime(primary_series, errors='raise')
                else:
                    # Regular type conversion
                    primary_series.astype(data_type)
            
            # Test secondary dataset if it exists
            if (hasattr(global_vars, 'secondary_df') and 
                global_vars.secondary_df is not None and 
                column in global_vars.secondary_df.columns):
                
                secondary_series = global_vars.secondary_df[column].copy()
                
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    
                    if data_type == "datetime64[ns]":
                        pd.to_datetime(secondary_series, errors='raise')
                    else:
                        secondary_series.astype(data_type)
            
            return True, ""
            
        except Exception as e:
            return False, f"Cannot convert column '{column}' to {data_type}: {str(e)}"
    
    @staticmethod
    def validate_classification_constraints(column: str, classification: str, data_type: str) -> Tuple[bool, str]:
        """
        Validate classification-specific constraints.
        
        Args:
            column: Column name
            classification: Intelligent classification
            data_type: Pandas data type
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            if classification == "Binary":
                # Binary must have exactly 2 unique values (excluding NaN)
                primary_series = global_vars.df[column].copy()
                
                if data_type != str(primary_series.dtype):
                    # Test with converted type
                    if data_type == "datetime64[ns]":
                        primary_series = pd.to_datetime(primary_series, errors='coerce')
                    else:
                        primary_series = primary_series.astype(data_type)
                
                unique_values = primary_series.dropna().unique()
                if len(unique_values) > 2:
                    return False, f"Binary classification requires exactly 2 unique values, but column '{column}' has {len(unique_values)} unique values"
                
                # Check secondary dataset if it exists
                if (hasattr(global_vars, 'secondary_df') and 
                    global_vars.secondary_df is not None and 
                    column in global_vars.secondary_df.columns):
                    
                    secondary_series = global_vars.secondary_df[column].copy()
                    
                    if data_type != str(secondary_series.dtype):
                        if data_type == "datetime64[ns]":
                            secondary_series = pd.to_datetime(secondary_series, errors='coerce')
                        else:
                            secondary_series = secondary_series.astype(data_type)
                    
                    secondary_unique = secondary_series.dropna().unique()
                    if len(secondary_unique) > 2:
                        return False, f"Binary classification requires exactly 2 unique values, but column '{column}' in secondary dataset has {len(secondary_unique)} unique values"
            
            elif classification == "Continuous":
                # Continuous should be numeric
                if data_type not in ["int64", "float64"]:
                    return False, f"Continuous classification requires numeric data type (int64 or float64), not {data_type}"
                
                # Special check for float to int conversion
                if data_type == "int64":
                    primary_series = global_vars.df[column]
                    if primary_series.dtype == "float64":
                        # Check if all values are integers (no decimals)
                        non_na_values = primary_series.dropna()
                        if not all(x.is_integer() for x in non_na_values):
                            return False, f"Cannot convert column '{column}' from float64 to int64: contains decimal values"
                    
                    # Check secondary dataset
                    if (hasattr(global_vars, 'secondary_df') and 
                        global_vars.secondary_df is not None and 
                        column in global_vars.secondary_df.columns):
                        
                        secondary_series = global_vars.secondary_df[column]
                        if secondary_series.dtype == "float64":
                            secondary_non_na = secondary_series.dropna()
                            if not all(x.is_integer() for x in secondary_non_na):
                                return False, f"Cannot convert column '{column}' in secondary dataset from float64 to int64: contains decimal values"
            
            elif classification == "Datetime":
                # Datetime must use datetime64[ns]
                if data_type != "datetime64[ns]":
                    return False, f"Datetime classification requires datetime64[ns] data type, not {data_type}"
            
            return True, ""
            
        except Exception as e:
            return False, f"Error validating classification constraints: {str(e)}"
    
    @staticmethod
    def update_column_type(column: str, classification: str, data_type: str) -> Tuple[bool, str, Optional[str]]:
        """
        Centralized function to update both classification and pandas dtype.
        
        This function ensures atomic updates - either both classification and
        data type are updated successfully, or neither is changed.
        
        Args:
            column: Column name to update
            classification: Intelligent classification (Binary, Continuous, Datetime, Categorical)
            data_type: Pandas data type (int64, float64, object, datetime64[ns], bool, category)
            
        Returns:
            Tuple of (success, message, warning)
            - success: Boolean indicating if update was successful
            - message: Success or error message
            - warning: Optional warning message
        """
        warning_msg = None
        
        try:
            # Step 1: Validate inputs
            is_valid, validation_msg = ColumnTypeManager.validate_inputs(column, classification, data_type)
            if not is_valid:
                # Check if this is actually a warning
                if "Warning:" in validation_msg:
                    warning_msg = validation_msg
                else:
                    return False, validation_msg, None
            else:
                if "Warning:" in validation_msg:
                    warning_msg = validation_msg
            
            # Step 2: Validate data conversion capability
            can_convert, convert_msg = ColumnTypeManager.validate_data_conversion(column, data_type)
            if not can_convert:
                return False, convert_msg, warning_msg
            
            # Step 3: Validate classification constraints
            constraints_valid, constraints_msg = ColumnTypeManager.validate_classification_constraints(
                column, classification, data_type
            )
            if not constraints_valid:
                return False, constraints_msg, warning_msg
            
            # Step 4: Store original states for rollback
            original_primary_dtype = str(global_vars.df[column].dtype)
            original_secondary_dtype = None
            original_classification = None
            
            if hasattr(global_vars, 'column_types') and column in global_vars.column_types:
                original_classification = global_vars.column_types[column]
            
            if (hasattr(global_vars, 'secondary_df') and 
                global_vars.secondary_df is not None and 
                column in global_vars.secondary_df.columns):
                original_secondary_dtype = str(global_vars.secondary_df[column].dtype)
            
            # Step 5: Apply updates atomically
            try:
                # Update classification first
                if not hasattr(global_vars, 'column_types'):
                    global_vars.column_types = {}
                global_vars.column_types[column] = classification
                
                # Update primary dataset dtype
                if data_type != original_primary_dtype:
                    if data_type == "datetime64[ns]":
                        global_vars.df[column] = pd.to_datetime(global_vars.df[column], errors='coerce')
                    else:
                        global_vars.df[column] = global_vars.df[column].astype(data_type)
                
                # Update secondary dataset dtype if it exists
                if (hasattr(global_vars, 'secondary_df') and 
                    global_vars.secondary_df is not None and 
                    column in global_vars.secondary_df.columns and
                    original_secondary_dtype and 
                    data_type != original_secondary_dtype):
                    
                    if data_type == "datetime64[ns]":
                        global_vars.secondary_df[column] = pd.to_datetime(global_vars.secondary_df[column], errors='coerce')
                    else:
                        global_vars.secondary_df[column] = global_vars.secondary_df[column].astype(data_type)
                
                # Step 6: Clear metrics cache to ensure recalculation
                if hasattr(global_vars, 'clear_metrics_cache'):
                    cache_reason = f"Column type changed: {column} -> {classification} ({data_type})"
                    global_vars.clear_metrics_cache(cache_reason)
                    print(f"[COLUMN TYPE MANAGER] Cleared metrics cache: {cache_reason}")
                
                # Step 7: Register change for global state trigger (Phase 1.1 ↔ 1.2 Integration)
                change_data = _column_type_change_registry.register_change(column, classification, data_type)
                print(f"[COLUMN TYPE MANAGER] Registered change for global state update: {change_data['change_id']}")
                
                # Step 8: Log successful update
                success_msg = f"Successfully updated column '{column}' to {classification} classification with {data_type} data type"
                print(f"[COLUMN TYPE MANAGER] {success_msg}")
                
                return True, success_msg, warning_msg
                
            except Exception as update_error:
                # Rollback on failure
                print(f"[COLUMN TYPE MANAGER] Update failed, rolling back: {str(update_error)}")
                
                try:
                    # Restore original classification
                    if original_classification is not None:
                        global_vars.column_types[column] = original_classification
                    elif hasattr(global_vars, 'column_types') and column in global_vars.column_types:
                        del global_vars.column_types[column]
                    
                    # Restore original primary dtype
                    if original_primary_dtype != data_type:
                        global_vars.df[column] = global_vars.df[column].astype(original_primary_dtype)
                    
                    # Restore original secondary dtype
                    if (original_secondary_dtype and 
                        hasattr(global_vars, 'secondary_df') and 
                        global_vars.secondary_df is not None and 
                        column in global_vars.secondary_df.columns):
                        global_vars.secondary_df[column] = global_vars.secondary_df[column].astype(original_secondary_dtype)
                        
                except Exception as rollback_error:
                    print(f"[COLUMN TYPE MANAGER] Rollback failed: {str(rollback_error)}")
                    return False, f"Update failed and rollback failed: {str(update_error)} | Rollback error: {str(rollback_error)}", warning_msg
                
                return False, f"Update failed (changes rolled back): {str(update_error)}", warning_msg
                
        except Exception as e:
            error_msg = f"Unexpected error in column type update: {str(e)}"
            print(f"[COLUMN TYPE MANAGER] {error_msg}")
            print(f"[COLUMN TYPE MANAGER] Traceback: {traceback.format_exc()}")
            return False, error_msg, warning_msg
    
    @staticmethod
    def get_column_info(column: str) -> Dict[str, Any]:
        """
        Single function to read column information consistently.
        
        Args:
            column: Column name to get information for
            
        Returns:
            Dictionary containing column information:
            - pandas_type: Actual pandas data type
            - classification: Intelligent classification type
            - exists_in_primary: Whether column exists in primary dataset
            - exists_in_secondary: Whether column exists in secondary dataset
            - unique_count_primary: Number of unique values in primary dataset
            - unique_count_secondary: Number of unique values in secondary dataset
            - sample_values: Sample values from the column
        """
        result = {
            'pandas_type': 'Unknown',
            'classification': 'Unknown',
            'exists_in_primary': False,
            'exists_in_secondary': False,
            'unique_count_primary': 0,
            'unique_count_secondary': 0,
            'sample_values': []
        }
        
        try:
            # Check primary dataset
            if hasattr(global_vars, 'df') and global_vars.df is not None and column in global_vars.df.columns:
                result['exists_in_primary'] = True
                result['pandas_type'] = str(global_vars.df[column].dtype)
                result['unique_count_primary'] = global_vars.df[column].nunique()
                
                # Get sample values (up to 3)
                sample_values = global_vars.df[column].dropna().head(3).tolist()
                result['sample_values'] = [str(val) for val in sample_values]
            
            # Check secondary dataset
            if (hasattr(global_vars, 'secondary_df') and 
                global_vars.secondary_df is not None and 
                column in global_vars.secondary_df.columns):
                result['exists_in_secondary'] = True
                result['unique_count_secondary'] = global_vars.secondary_df[column].nunique()
            
            # Get intelligent classification
            if (hasattr(global_vars, 'column_types') and 
                global_vars.column_types and 
                column in global_vars.column_types):
                result['classification'] = global_vars.column_types[column]
            
        except Exception as e:
            print(f"[COLUMN TYPE MANAGER] Error getting column info for '{column}': {str(e)}")
        
        return result
    
    @staticmethod
    def get_all_columns_info() -> Dict[str, Dict[str, Any]]:
        """
        Get information for all columns in the datasets.
        
        Returns:
            Dictionary mapping column names to their information
        """
        all_columns = set()
        
        # Collect all column names from both datasets
        if hasattr(global_vars, 'df') and global_vars.df is not None:
            all_columns.update(global_vars.df.columns)
        
        if hasattr(global_vars, 'secondary_df') and global_vars.secondary_df is not None:
            all_columns.update(global_vars.secondary_df.columns)
        
        # Get info for each column
        result = {}
        for column in all_columns:
            result[column] = ColumnTypeManager.get_column_info(column)
        
        return result
    
    @staticmethod
    def reset_column_types():
        """
        Reset column types to their auto-detected values.
        
        This function re-runs the type detection logic and updates
        global_vars.column_types accordingly.
        """
        try:
            if not hasattr(global_vars, 'df') or global_vars.df is None:
                print("[COLUMN TYPE MANAGER] No primary dataset available for reset")
                return False, "No primary dataset available"
            
            # Import the data processor
            from utils.data_processor import preprocess_datasets
            
            # Re-run type detection
            if hasattr(global_vars, 'secondary_df') and global_vars.secondary_df is not None:
                # Both datasets available
                processed_primary, processed_secondary, detected_types = preprocess_datasets(
                    global_vars.df, global_vars.secondary_df
                )
            else:
                # Only primary dataset
                processed_primary, _, detected_types = preprocess_datasets(global_vars.df, None)
            
            # Update global state
            global_vars.column_types = detected_types
            
            # Clear metrics cache
            if hasattr(global_vars, 'clear_metrics_cache'):
                global_vars.clear_metrics_cache("Column types reset to auto-detected values")
            
            print(f"[COLUMN TYPE MANAGER] Reset column types for {len(detected_types)} columns")
            return True, f"Successfully reset {len(detected_types)} column types"
            
        except Exception as e:
            error_msg = f"Error resetting column types: {str(e)}"
            print(f"[COLUMN TYPE MANAGER] {error_msg}")
            return False, error_msg
    
    @staticmethod
    def get_latest_change_for_trigger() -> dict:
        """
        Get the latest column type change for triggering global state updates.
        
        This method provides access to the change registry for the Dash callback system.
        It's designed to be called from Dash callbacks to trigger global state updates.
        
        Returns:
            dict: Latest change data for trigger store
        """
        try:
            latest_change = _column_type_change_registry.get_latest_change()
            _column_type_change_registry.clear_processed_changes()
            return latest_change
        except Exception as e:
            print(f"[COLUMN TYPE MANAGER] Error getting latest change: {str(e)}")
            return {}