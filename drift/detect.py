import pandas as pd
import numpy as np
from sklearn.metrics import normalized_mutual_info_score
from sklearn.preprocessing import LabelEncoder
from evidently import Report
from evidently.presets import DataDriftPreset
import pandas as pd
import json
import numpy as np

import evidently.metrics as metrics_det
from evidently import Report
from evidently.presets import DataDriftPreset
from evidently.metrics import ValueDrift
from evidently.metrics import DriftedColumnsCount
from evidently.metrics import DatasetMissingValueCount
from evidently.metrics import ColumnCount
from evidently.metrics import RowCount


from UI.functions.global_vars import global_vars


def format_metric_value(value):
    """
    Format metric values for display.
    
    Args:
        value (float or str): The value to format
        
    Returns:
        str: Formatted value
    """
    if value == "N/A":
        return "N/A"
    
    try:
        value = float(value)
        if value > 10000:
            return f"{value:.2e}"
        elif value < 0.001:
            return f"{value:.2e}"
        else:
            return f"{value:.4f}"
    except (ValueError, TypeError):
        return str(value)

def is_binary_feature(series):
    """Check if a feature is binary (has exactly 2 unique values)"""
    return series.nunique() == 2

def is_categorical_feature(series, max_categories=10):
    """Determine if a feature is categorical based on its properties"""
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_categorical_dtype(series):
        return True
    
    if series.nunique() <= max_categories:
        value_counts = series.value_counts(normalize=True)
        if (value_counts > 0.9).any():
            return False
        return True
    
    return False

def get_feature_type(series):
    """Determine if a feature is numerical or categorical"""
    if is_categorical_feature(series):
        return 'categorical'
    return 'numerical'

def convert_evidently_metrics_to_table_data(report_data, primary_df=None, secondary_df=None):
    """
    Convert Evidently's report data into a format suitable for displaying in the metrics table.
    
    Args:
        report_data (str): JSON string from Evidently report
        primary_df (pd.DataFrame, optional): Primary dataset for feature type detection
        secondary_df (pd.DataFrame, optional): Secondary dataset for feature type detection
        
    Returns:
        list: List of dictionaries containing drift metrics for each column
    """
    print("[DEBUG] Starting convert_evidently_metrics_to_table_data")
    print(f"[DEBUG] Report data type: {type(report_data)}")
    
    # Initialize a dictionary to store metrics for each column
    column_metrics = {}
    
    try:
        # Parse JSON data if it's a string
        if isinstance(report_data, str):
            import json
            report_data = json.loads(report_data)
        
        # Process each metric
        if 'metrics' not in report_data:
            print("[DEBUG] No 'metrics' key in report data")
            return []
        
        print(f"[DEBUG] Number of metrics: {len(report_data['metrics'])}")
        
        # First pass: collect all columns and initialize their metrics
        for metric in report_data['metrics']:
            metric_id = metric['metric_id']
            if not metric_id.startswith('ValueDrift'):
                continue
                
            try:
                # Extract column name from metric_id
                # Format: ValueDrift(column=column_name,method=method_name)
                column_name = metric_id.split('column=')[1].split(',')[0]
                
                if column_name not in column_metrics:
                    # Determine feature type if DataFrames are provided
                    feature_type = "Unknown"
                    if primary_df is not None and column_name in primary_df.columns:
                        feature_type = get_feature_type(primary_df[column_name])
                    
                    column_metrics[column_name] = {
                        "Attribute": column_name,
                        "Type": feature_type,
                        "JS_Divergence": "N/A",
                        "PSI": "N/A",
                        "Wasserstein": "N/A",
                        "Hellinger": "N/A",
                        "Empirical_MMD": "N/A",
                        "Chi_Square": "N/A",
                    }
                    print(f"[DEBUG] Initialized metrics for column: {column_name}")
            except Exception as e:
                print(f"[DEBUG] Error initializing metrics for {metric_id}: {str(e)}")
                continue
        
        print(f"[DEBUG] Initialized metrics for {len(column_metrics)} columns")
        
        # Second pass: update metrics with values
        for metric in report_data['metrics']:
            metric_id = metric['metric_id']
            if not metric_id.startswith('ValueDrift'):
                continue
                
            try:
                # Extract column name and method from metric_id
                parts = metric_id.replace('ValueDrift(', '').replace(')', '').split(',')
                column_name = parts[0].split('=')[1]
                method = parts[1].split('=')[1]
                value = metric['value']
                
                # print(f"[DEBUG] Processing metric: {metric_id}")
                # print(f"[DEBUG] Column: {column_name}, Method: {method}, Value: {value}")
                
                if column_name not in column_metrics:
                    print(f"[DEBUG] Warning: Column {column_name} not found in column_metrics")
                    continue
                
                # Update the corresponding metric
                if method == 'jensenshannon':
                    column_metrics[column_name]['JS_Divergence'] = format_metric_value(value)
                elif method == 'psi':
                    column_metrics[column_name]['PSI'] = format_metric_value(value)
                elif method == 'wasserstein':
                    column_metrics[column_name]['Wasserstein'] = format_metric_value(value)
                elif method == 'hellinger':
                    column_metrics[column_name]['Hellinger'] = format_metric_value(value)
                
                elif method == 'empirical_mmd':
                    column_metrics[column_name]['Empirical_MMD'] = format_metric_value(value)
                elif method == 'chisquare':
                    column_metrics[column_name]['Chi_Square'] = format_metric_value(value)
                else:
                    print(f"[DEBUG] Unknown method: {method}")
            except Exception as e:
                print(f"[DEBUG] Error processing metric {metric_id}: {str(e)}")
                continue
        
        # Convert dictionary to list
        metrics_data = list(column_metrics.values())
        print(f"[DEBUG] Generated metrics data for {len(metrics_data)} columns")
        
        # Print first row of metrics data for debugging
        if metrics_data:
            print(f"[DEBUG] First row of metrics data: {metrics_data[0]}")
        
        return metrics_data
        
    except Exception as e:
        print(f"[DEBUG] Error in convert_evidently_metrics_to_table_data: {str(e)}")
        import traceback
        print(f"[DEBUG] Traceback: {traceback.format_exc()}")
        return []

def resample_datasets(df1, df2, random_state=42):
    """
    Resample datasets to have equal lengths using the smaller dataset's size.
    
    Args:
        df1 (pd.DataFrame): First dataset
        df2 (pd.DataFrame): Second dataset
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (resampled_df1, resampled_df2)
    """
    n_samples = min(len(df1), len(df2))
    if n_samples > 10000:
        n_samples = 5000
    elif n_samples > 5000:
        n_samples = 2000
    elif n_samples > 1000:
        n_samples = 1000

    print(f"[DEBUG] Resampling datasets to {n_samples} records each")
    return df1.sample(n=n_samples, random_state=random_state), df2.sample(n=n_samples, random_state=random_state)

def encode_categorical_features(df1, df2):
    """
    Encode categorical features in both datasets.
    
    Args:
        df1 (pd.DataFrame): First dataset
        df2 (pd.DataFrame): Second dataset
        
    Returns:
        tuple: (encoded df1, encoded df2)
    """
    print(f"[DEBUG] Starting encode_categorical_features")
    print(f"[DEBUG] df1 shape: {df1.shape}, df2 shape: {df2.shape}")
    
    # ensure the two datasets have the same length (this is important!)
    if len(df1) != len(df2):
        print(f"[DEBUG] WARNING: DataFrames have different lengths! df1: {len(df1)}, df2: {len(df2)}")
        min_len = min(len(df1), len(df2))
        df1 = df1.head(min_len).copy()
        df2 = df2.head(min_len).copy()
        print(f"[DEBUG] Truncated both to length: {min_len}")
    
    # get common columns
    common_columns = list(set(df1.columns) & set(df2.columns))
    print(f"[DEBUG] Common columns: {common_columns}")
    
    # copy the datasets to avoid modifying the original data
    df1_encoded = df1.copy()
    df2_encoded = df2.copy()
    
    for column in common_columns:
        print(f"\n[DEBUG] Processing column: {column}")
        print(f"[DEBUG] df1[{column}] dtype: {df1[column].dtype}, unique count: {df1[column].nunique()}")
        print(f"[DEBUG] df2[{column}] dtype: {df2[column].dtype}, unique count: {df2[column].nunique()}")
        
        # check if the column is string or categorical
        is_string_or_categorical = (
            pd.api.types.is_object_dtype(df1[column]) or 
            pd.api.types.is_categorical_dtype(df1[column]) or
            pd.api.types.is_object_dtype(df2[column]) or 
            pd.api.types.is_categorical_dtype(df2[column])
        )
        
        print(f"[DEBUG] Column {column} is_string_or_categorical: {is_string_or_categorical}")
        
        if is_string_or_categorical:
            try:
                print(f"[DEBUG] Sample values from df1[{column}]: {df1[column].dropna().head(5).tolist()}")
                print(f"[DEBUG] Sample values from df2[{column}]: {df2[column].dropna().head(5).tolist()}")
                
                # concatenate the two datasets' column
                print(f"[DEBUG] Concatenating columns...")
                concatenated = pd.concat([df1[column], df2[column]])
                print(f"[DEBUG] Concatenated shape: {concatenated.shape}")
                
                print(f"[DEBUG] Dropping NaN values...")
                concatenated_no_na = concatenated.dropna()
                print(f"[DEBUG] After dropna shape: {concatenated_no_na.shape}")
                
                print(f"[DEBUG] Getting unique values...")
                all_values = concatenated_no_na.unique()
                print(f"[DEBUG] Unique values count: {len(all_values)}")
                print(f"[DEBUG] Sample unique values: {all_values[:min(10, len(all_values))].tolist()}")
                
                # create label encoder
                print(f"[DEBUG] Creating and fitting LabelEncoder...")
                encoder = LabelEncoder()
                encoder.fit(all_values)
                print(f"[DEBUG] Encoder classes count: {len(encoder.classes_)}")
                
            except Exception as e:
                print(f"[DEBUG] Error processing column {column}: {str(e)}")
                import traceback
                print(f"[DEBUG] Traceback: {traceback.format_exc()}")
                continue
            
                # encode the column for both datasets
                # handle missing values
                print(f"[DEBUG] Processing encoding for df1[{column}]...")
                df1_col_filled = df1[column].fillna('__MISSING__')
                print(f"[DEBUG] df1 column after fillna shape: {df1_col_filled.shape}")
                
                print(f"[DEBUG] Processing encoding for df2[{column}]...")
                df2_col_filled = df2[column].fillna('__MISSING__')
                print(f"[DEBUG] df2 column after fillna shape: {df2_col_filled.shape}")
                
                # safely encode, ensuring data integrity
                print(f"[DEBUG] Encoding df1[{column}]...")
                print(f"[DEBUG] df1_col_filled length: {len(df1_col_filled)}")
                
                # use encoder.transform, but handle unknown values
                df1_encoded_values = []
                unknown_values_df1 = set()
                for i, val in enumerate(df1_col_filled):
                    if pd.isna(val) or val == '__MISSING__':
                        df1_encoded_values.append(np.nan)
                    elif val in encoder.classes_:
                        df1_encoded_values.append(encoder.transform([val])[0])
                    else:
                        # for unknown values, we need to skip or use the nearest known value
                        # but to keep data integrity, we use a special encoding value
                        df1_encoded_values.append(len(encoder.classes_))  # use the next available encoding
                        unknown_values_df1.add(val)
                
                print(f"[DEBUG] df1 encoded values length: {len(df1_encoded_values)}")
                if unknown_values_df1:
                    print(f"[DEBUG] Found unknown values in df1[{column}]: {list(unknown_values_df1)}")
                
                print(f"[DEBUG] Encoding df2[{column}]...")
                print(f"[DEBUG] df2_col_filled length: {len(df2_col_filled)}")
                
                df2_encoded_values = []
                unknown_values_df2 = set()
                for i, val in enumerate(df2_col_filled):
                    if pd.isna(val) or val == '__MISSING__':
                        df2_encoded_values.append(np.nan)
                    elif val in encoder.classes_:
                        df2_encoded_values.append(encoder.transform([val])[0])
                    else:
                        # for unknown values, use the same encoding strategy
                        df2_encoded_values.append(len(encoder.classes_))  # use the next available encoding
                        unknown_values_df2.add(val)
                
                print(f"[DEBUG] df2 encoded values length: {len(df2_encoded_values)}")
                if unknown_values_df2:
                    print(f"[DEBUG] Found unknown values in df2[{column}]: {list(unknown_values_df2)}")
                
                # ensure the encoded data length is correct
                assert len(df1_encoded_values) == len(df1), f"df1 encoding length mismatch: {len(df1_encoded_values)} vs {len(df1)}"
                assert len(df2_encoded_values) == len(df2), f"df2 encoding length mismatch: {len(df2_encoded_values)} vs {len(df2)}"
                
                df1_encoded[column] = df1_encoded_values
                df2_encoded[column] = df2_encoded_values
                
                # final data type conversion and validation
                print(f"[DEBUG] Final data processing for {column}...")
                
                # ensure the encoded result is numeric, can contain NaN
                df1_encoded[column] = pd.Series(df1_encoded[column], dtype='float64', index=df1.index)
                df2_encoded[column] = pd.Series(df2_encoded[column], dtype='float64', index=df2.index)
                
                # final validation: ensure length consistency
                print(f"[DEBUG] Final lengths - df1[{column}]: {len(df1_encoded[column])}, df2[{column}]: {len(df2_encoded[column])}")
                print(f"[DEBUG] Expected lengths - df1: {len(df1)}, df2: {len(df2)}")
                
                # verify no unexpected data loss
                if len(df1_encoded[column]) != len(df1) or len(df2_encoded[column]) != len(df2):
                    raise ValueError(f"Data length mismatch after encoding column {column}")
                
                # verify the quality of the encoded data
                print(f"[DEBUG] Successfully encoded column '{column}': {len(encoder.classes_)} unique values")
                print(f"[DEBUG] df1[{column}] encoded unique count: {pd.Series(df1_encoded[column]).nunique()}")
                print(f"[DEBUG] df2[{column}] encoded unique count: {pd.Series(df2_encoded[column]).nunique()}")
                
                # check the range of the encoded values
                df1_values = pd.Series(df1_encoded[column]).dropna()
                df2_values = pd.Series(df2_encoded[column]).dropna()
                if len(df1_values) > 0:
                    print(f"[DEBUG] df1[{column}] encoded value range: {df1_values.min()} to {df1_values.max()}")
                if len(df2_values) > 0:
                    print(f"[DEBUG] df2[{column}] encoded value range: {df2_values.min()} to {df2_values.max()}")
                
                # check if the total count of the encoded data is consistent
                print(f"[DEBUG] df1[{column}] total count: {len(df1_encoded[column])}, non-null: {df1_encoded[column].notna().sum()}")
                print(f"[DEBUG] df2[{column}] total count: {len(df2_encoded[column])}, non-null: {df2_encoded[column].notna().sum()}")
                
            except Exception as e:
                print(f"[DEBUG] Error in encoding process for column {column}: {str(e)}")
                import traceback
                print(f"[DEBUG] Traceback: {traceback.format_exc()}")
                continue
    
    
    # ensure the shapes are exactly the same
    if df1_encoded.shape != df1.shape:
        raise ValueError(f"df1 shape changed during encoding: {df1.shape} -> {df1_encoded.shape}")
    if df2_encoded.shape != df2.shape:
        raise ValueError(f"df2 shape changed during encoding: {df2.shape} -> {df2_encoded.shape}")
    
    # ensure the two encoded datasets have the same length
    if len(df1_encoded) != len(df2_encoded):
        raise ValueError(f"Encoded datasets have different lengths: {len(df1_encoded)} vs {len(df2_encoded)}")
    
    print(f"[DEBUG] Data integrity check passed!")
    print(f"[DEBUG] Finished encode_categorical_features")
    return df1_encoded, df2_encoded

def generate_metrics_data():
    """
    Generate metrics data comparing distributions between two datasets using Evidently.
    
    Args:
        primary_df (pd.DataFrame): Primary dataset
        secondary_df (pd.DataFrame): Secondary dataset
        target_attribute (str or dict, optional): Target attribute for relevance calculation
        
    Returns:
        list: List of dictionaries containing drift metrics for each column
    """
    primary_df = global_vars.df
    secondary_df = global_vars.secondary_df
    target_attribute = global_vars.target_attribute
    
    if primary_df is None or secondary_df is None:
        print("[DEBUG] One or both DataFrames are None")
        return [], None
    
    # Get common columns
    common_columns = list(set(primary_df.columns) & set(secondary_df.columns))
    if not common_columns:
        print("[DEBUG] No common columns between datasets")
        return [], None
    
    print(f"[DEBUG] Common columns: {common_columns}")
    
    try:
        # Resample datasets to aligned sample size – important for drift metrics.
        primary_resampled, secondary_resampled = resample_datasets(primary_df, secondary_df)

        # Analyse datasets and obtain Evidently report (ColumnMapping handled inside).
        report, drift_columns = analyze_dataset(primary_resampled, secondary_resampled)
        report_json = report.json()
        print("[DEBUG] Successfully generated Evidently report")

        # Convert report to table data suitable for UI display.
        metrics_data = convert_evidently_metrics_to_table_data(report_json, primary_df, secondary_df)
        print(f"[DEBUG] Generated metrics data for {len(metrics_data)} columns")


        return metrics_data, (len(primary_resampled), len(secondary_resampled))
        
    except Exception as e:
        print(f"[DEBUG] Error in generate_metrics_data: {str(e)}")
        import traceback
        print(f"[DEBUG] Traceback: {traceback.format_exc()}")
        return [], None

# -----------------------------------------------------------------------------
# Supported drift methods matrix (feature_type → list of methods)
# Centralised here to ensure consistency across helper utilities.
# -----------------------------------------------------------------------------
SUPPORTED_DRIFT_METHODS = {
    "numerical": [
        "psi",                # Population Stability Index
        "wasserstein",        # Wasserstein distance (for larger samples)
        # "kl_div",             # Kullback-Leibler divergence
        "jensenshannon",      # Jensen-Shannon distance
        "hellinger",          # Hellinger distance
        "empirical_mmd"       # Maximum Mean Discrepancy
    ],
    "categorical": [
        "psi",                # PSI can work on categories (probability vectors)
        "jensenshannon",      # JS distance for probability distributions
        "chisquare",          # Chi-square test
    ]
}

# -----------------------------------------------------------------------------
# Helper: Retrieve applicable drift methods for a single feature
# -----------------------------------------------------------------------------

def get_drift_methods(series: pd.Series, reference_series: pd.Series, n_samples: int) -> list[str]:
    """Return a list of drift methods supported for the given feature.

    The list is derived from the central ``SUPPORTED_DRIFT_METHODS`` matrix and
    then lightly pruned based on sample-size heuristics (e.g., KS test is
    unreliable for small samples).
    """
    feature_type = get_feature_type(series)
    methods = SUPPORTED_DRIFT_METHODS.get(feature_type, []).copy()

    # Example heuristic – remove KS if sample size < 100.
    if feature_type == "numerical" and n_samples < 100 and "ks" in methods:
        methods.remove("ks")

    # For categorical binary variables restrict to tests that support binary-only.
    if feature_type == "categorical":
        is_binary_curr = is_binary_feature(series)
        is_binary_ref = is_binary_feature(reference_series)
        if not (is_binary_curr and is_binary_ref):
            # Remove binary-only tests if data is not binary.
            methods = [m for m in methods if m not in {"z", "fisher_exact"}]

    return methods

# -----------------------------------------------------------------------------
# Helper: Build a comprehensive list of ValueDrift metrics for Evidently
# -----------------------------------------------------------------------------

def create_drift_metrics(df: pd.DataFrame, reference_df: pd.DataFrame) -> list[ValueDrift]:
    """Generate ValueDrift metric objects for all columns based on support matrix."""
    metrics: list[ValueDrift] = []
    n_samples = len(df)
    for column in df.columns:
        for method in get_drift_methods(df[column], reference_df[column], n_samples):
            metrics.append(ValueDrift(column=column, method=method))
    return metrics

# -----------------------------------------------------------------------------
# Robust report creation helpers
# -----------------------------------------------------------------------------

def create_and_run_report_with_error_handling(
    df: pd.DataFrame,
    reference_df: pd.DataFrame,
    column_mapping=None,
):
    """Generate an Evidently report that skips failing metrics gracefully.

    1. Build a draft list of ``ValueDrift`` metrics via :pyfunc:`create_drift_metrics`.
    2. Attempt to run the report; if any metric raises, fall back to the
       *per-metric probing* strategy so we can isolate and drop problematic
       combinations.
    3. Return a fully-executed ``Report`` instance.
    """
    # First, attempt the optimistic path – run with all metrics at once.
    all_metrics = create_drift_metrics(df, reference_df)
    report = Report(all_metrics)
    try:
        return report.run(df, reference_df)
    except Exception as exc:
        print(f"[DEBUG] Initial full report failed → {exc}. Falling back to per-metric probing…")

    # Fallback: probe each metric individually to collect those that succeed.
    successful_metrics: list[ValueDrift] = []
    for metric in all_metrics:
        temp_report = Report([metric])
        try:
            temp_report.run(df, reference_df)
            successful_metrics.append(metric)
            print(f"[DEBUG] ✓ {metric.column}+{metric.method} succeeded in fallback")
        except Exception as err:
            print(f"[DEBUG] ✗ {metric.column}+{metric.method} failed in fallback: {err}")

    # Finally, build a report using only successful metrics.
    final_report = Report(successful_metrics)
    return final_report.run(df, reference_df)


def analyze_dataset(
    df: pd.DataFrame,
    reference_df: pd.DataFrame | None = None,
):
    """High-level convenience wrapper to analyse two datasets.

    Handles resampling, ColumnMapping construction, and invokes the robust
    report generation helper.
    """
    if reference_df is None:
        reference_df = df.copy()

    # 1. Resample to equal sizes so distribution metrics are comparable.
    df_resampled, ref_resampled = resample_datasets(df, reference_df)

    # 2. Column mapping so Evidently uses correct feature types.
    # NOTE: build_column_mapping removed – Evidently 0.7+ handles types via dtypes.

    # 3. Generate report resiliently.
    report = create_and_run_report_with_error_handling(
        df_resampled,
        ref_resampled,
    )

    # For now we do not compute drifted column counts separately; returning None
    return report, None



    