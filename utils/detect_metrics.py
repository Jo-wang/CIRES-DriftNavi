"""
Utility functions for calculating metrics in the Detect phase.

This module provides functions for computing normalized mutual information
between features and target attributes to assess target relevance.
"""
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import LabelEncoder
from scipy.stats import entropy


# def calculate_detect_target_relevance_mi(df, feature_name, target_name):
#     """
#     Calculate normalized mutual information between a feature and target attribute.
    
#     This function computes a normalized 0-1 score indicating how strongly the feature 
#     is related to the target attribute. Higher values indicate stronger relevance.
    
#     Args:
#         df (pandas.DataFrame): DataFrame containing both feature and target columns
#         feature_name (str): Name of the feature column
#         target_name (str): Name of the target column
        
#     Returns:
#         float or str: Normalized mutual information score between 0 and 1, 
#                      or "N/A" if calculation is not possible
#     """
#     # Handle missing case
#     if feature_name not in df.columns or target_name not in df.columns:
#         print(f"Column missing: feature='{feature_name}' or target='{target_name}'")
#         return "N/A"
    
#     # Extract feature and target series
#     feature = df[feature_name]
#     target = df[target_name]
    
#     # Debug: Print feature and target information
#     print(f"DEBUG - Feature '{feature_name}' info:")
#     print(f"  - Data type: {feature.dtype}")
#     print(f"  - Sample values: {feature.head(3).values}")
#     print(f"  - Unique count: {feature.nunique()}")
#     print(f"  - Contains null: {feature.isnull().sum() > 0}")
    
#     print(f"DEBUG - Target '{target_name}' info:")
#     print(f"  - Data type: {target.dtype}")
#     print(f"  - Sample values: {target.head(3).values}")
#     print(f"  - Unique count: {target.nunique()}")
#     print(f"  - Contains null: {target.isnull().sum() > 0}")
    
#     # Drop rows where either feature or target has NaN
#     valid_data = pd.DataFrame({
#         'feature': feature,
#         'target': target
#     }).dropna()
    
#     # If not enough data after dropping NaNs, return N/A
#     if len(valid_data) < 10:
#         print(f"Not enough valid data after dropping NaNs: {len(valid_data)} rows")
#         return "N/A"
    
#     feature = valid_data['feature']
#     target = valid_data['target']
    
#     # Handle different data types
#     try:
#         # Process feature based on its type
#         if pd.api.types.is_numeric_dtype(feature):
#             # Numeric feature - use as is
#             processed_feature = feature.values
#             feature_is_numeric = True
#             print(f"Feature '{feature_name}' treated as numeric")
#         elif pd.api.types.is_bool_dtype(feature) or (feature.nunique() <= 2):
#             # Boolean feature - convert to numeric
#             processed_feature = feature.astype(float).values
#             feature_is_numeric = True
#             print(f"Feature '{feature_name}' treated as boolean/binary")
#         elif pd.api.types.is_categorical_dtype(feature) or pd.api.types.is_object_dtype(feature):
#             # Categorical feature - always use label encoding for string values
#             print(f"Feature '{feature_name}' treated as categorical/object, applying label encoding")
#             le = LabelEncoder()
#             try:
#                 processed_feature = le.fit_transform(feature)
#                 print(f"  - Label encoding applied successfully")
#                 print(f"  - Unique labels: {le.classes_}")
#                 print(f"  - Transformed values (sample): {processed_feature[:5]}")
#                 feature_is_numeric = False
#             except Exception as e:
#                 print(f"  - ERROR in label encoding: {e}")
#                 return "N/A"
#         elif pd.api.types.is_datetime64_dtype(feature):
#             # Datetime feature - convert to numeric timestamps
#             processed_feature = pd.to_numeric(pd.to_datetime(feature)).values
#             feature_is_numeric = True
#             print(f"Feature '{feature_name}' treated as datetime")
#         else:
#             # Unsupported type
#             print(f"Feature '{feature_name}' has unsupported type: {feature.dtype}")
#             return "N/A"
        
#         # Process target based on its type
#         if pd.api.types.is_numeric_dtype(target):
#             # Numeric target - use regression MI
#             processed_target = target.values
#             target_is_numeric = True
#             print(f"Target '{target_name}' treated as numeric")
#         elif pd.api.types.is_bool_dtype(target) or pd.api.types.is_categorical_dtype(target) or pd.api.types.is_object_dtype(target):
#             # Categorical target - always use label encoding
#             print(f"Target '{target_name}' treated as categorical/object, applying label encoding")
#             le = LabelEncoder()
#             try:
#                 processed_target = le.fit_transform(target)
#                 print(f"  - Label encoding applied successfully")
#                 print(f"  - Unique labels: {le.classes_}")
#                 print(f"  - Transformed values (sample): {processed_target[:5]}")
#                 target_is_numeric = False
#             except Exception as e:
#                 print(f"  - ERROR in label encoding: {e}")
#                 return "N/A"
#         elif pd.api.types.is_datetime64_dtype(target):
#             # Datetime target - convert to numeric timestamps and use regression MI
#             processed_target = pd.to_numeric(pd.to_datetime(target)).values
#             target_is_numeric = True
#             print(f"Target '{target_name}' treated as datetime")
#         else:
#             # Unsupported type
#             print(f"Target '{target_name}' has unsupported type: {target.dtype}")
#             return "N/A"
            
#         # Select appropriate MI function based on target type
#         if target_is_numeric:
#             mi_func = mutual_info_regression
#             print(f"Using mutual_info_regression function")
#         else:
#             mi_func = mutual_info_classif
#             print(f"Using mutual_info_classif function")
        
#         # Calculate mutual information
#         try:
#             print(f"Calculating MI between {feature_name} ({processed_feature.shape}) and {target_name} ({processed_target.shape})")
#             mi_score = mi_func(
#                 processed_feature.reshape(-1, 1),
#                 processed_target
#             )[0]
#             print(f"MI calculation successful: {mi_score}")
#         except Exception as e:
#             print(f"Error in MI calculation for feature '{feature_name}' and target '{target_name}': {e}")
#             print(f"Feature type: {type(processed_feature)}, Shape: {processed_feature.shape}")
#             print(f"Feature values (sample): {processed_feature[:5]}")
#             print(f"Target type: {type(processed_target)}, Shape: {processed_target.shape}")
#             print(f"Target values (sample): {processed_target[:5]}")
#             return "N/A"
        
#         # Calculate entropy of feature and target for normalization
#         # For processed feature
#         if len(np.unique(processed_feature)) == 1:
#             feature_entropy = 0  # Zero entropy for constant features
#         else:
#             feature_entropy = entropy(
#                 np.histogram(processed_feature, bins=min(50, len(np.unique(processed_feature))))[0]
#             )
            
#         # For processed target
#         if len(np.unique(processed_target)) == 1:
#             target_entropy = 0  # Zero entropy for constant targets
#         else:
#             target_entropy = entropy(
#                 np.histogram(processed_target, bins=min(50, len(np.unique(processed_target))))[0]
#             )
        
#         # Normalize MI score
#         # Avoid division by zero
#         if feature_entropy == 0 or target_entropy == 0:
#             if mi_score > 0:
#                 return 1.0  # Perfect correlation with a constant
#             else:
#                 return 0.0  # No correlation
#         else:
#             # Normalized MI = MI / sqrt(H(X) * H(Y))
#             nmi = mi_score / np.sqrt(feature_entropy * target_entropy)
            
#             # Clip to [0, 1] range to handle any numerical issues
#             return max(0.0, min(1.0, nmi))
            
#     except Exception as e:
#         print(f"Error calculating MI between {feature_name} and {target_name}: {e}")
#         import traceback
#         traceback.print_exc()
#         return "N/A"


# def get_target_relevance_category(mi_value):
#     """
#     Convert normalized mutual information value to a relevance category.
    
#     Args:
#         mi_value (float or str): Normalized mutual information value or "N/A"
        
#     Returns:
#         str: Relevance category ("Target", "Very High", "High", "Medium", "Low", or "Unknown")
#     """
#     if mi_value == "N/A":
#         return "Unknown"
    
#     if mi_value == 1.0:
#         return "Target"  # Perfect correlation, probably the target itself
#     elif mi_value >= 0.5:
#         return "Very High"
#     elif mi_value >= 0.25:
#         return "High"
#     elif mi_value >= 0.1:
#         return "Medium"
#     else:
#         return "Low"
