"""
Finetune data pipeline utilities.

This module provides data-only helpers for finetuning workflows. The function
implemented here selects a coreset from the primary dataset using CoreTab if
available; otherwise, it falls back to a stratified sampling scheme on the
target variable.

Public API:
- select_coreset(primary_df, percent, algo="dt", random_state=42) -> pd.DataFrame

Notes:
- The target attribute name is resolved from global state (global_vars.target_attribute).
- The function returns a DataFrame with the same schema as the input (including
  the target column), containing approximately the requested percentage of
  samples, with a minimum of one row.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def _resolve_target_column(primary_df: pd.DataFrame) -> str:
    """
    Resolve the target column name from global state and validate it exists.

    Returns:
        str: Target column name.

    Raises:
        ValueError: If the target attribute is not set or missing from the DataFrame.
    """
    try:
        from UI.functions.global_vars import global_vars
        target_col = getattr(global_vars, "target_attribute", None)
    except Exception:
        target_col = None

    if not target_col:
        raise ValueError("[FINETUNE] Target attribute is not set in global_vars.")
    if target_col not in primary_df.columns:
        raise ValueError(f"[FINETUNE] Target attribute '{target_col}' not found in primary_df columns.")
    return target_col


def _compute_examples_to_keep(total_rows: int, percent: float) -> int:
    """
    Compute the number of examples to keep based on the percentage.

    Ensures the result is at least 100 and at most total_rows.
    """
    if percent is None or percent <= 0 or percent > 100:
        raise ValueError("[FINETUNE] percent must be in the range (0, 100].")
    requested = int(round(total_rows * (percent / 100.0)))
    return max(100, min(total_rows, requested))


def _fallback_stratified_sample(
    df: pd.DataFrame,
    target_col: str,
    n_samples: int,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Fallback sampling when CoreTab is unavailable or fails.

    Strategy:
    - If target is categorical (or few unique values), stratify directly on target.
    - Else (likely regression), build quantile bins on target and stratify on bins.

    Returns:
        DataFrame: Sampled subset preserving schema and including the target column.
    """
    from sklearn.model_selection import StratifiedShuffleSplit

    y = df[target_col]
    rng = np.random.RandomState(random_state)

    # Decide stratification key
    if not pd.api.types.is_numeric_dtype(y) or y.nunique() <= 20:
        # Categorical or few unique values → stratify on y directly
        stratify_key = y
    else:
        # Numeric with many unique values → create quantile bins
        # Use up to 10 bins, avoid duplicates
        try:
            n_bins = min(10, max(2, int(np.sqrt(len(df)) // 10) or 10))
            stratify_key = pd.qcut(y, q=n_bins, duplicates="drop")
        except Exception:
            # If qcut fails, fallback to simple random sample
            return df.sample(n=n_samples, random_state=rng)

    # When classes are too small for exact train_size, fall back to proportional sampling per class
    try:
        splitter = StratifiedShuffleSplit(n_splits=1, train_size=n_samples, random_state=random_state)
        idx, _ = next(splitter.split(df, stratify_key))
        return df.iloc[idx].copy()
    except Exception:
        # Manual proportional sampling across strata
        parts = []
        counts = stratify_key.value_counts()
        total = len(df)
        for cls, cnt in counts.items():
            share = cnt / total
            need = max(1, int(round(share * n_samples)))
            seg = df.loc[stratify_key == cls]
            if len(seg) <= need:
                parts.append(seg)
            else:
                parts.append(seg.sample(n=need, random_state=rng))
        out = pd.concat(parts, axis=0)
        if len(out) > n_samples:
            out = out.sample(n=n_samples, random_state=rng)
        return out.copy()


def select_coreset(
    primary_df: pd.DataFrame,
    percent: float,
    algo: str = "dt",
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Select a coreset from the primary dataset using CoreTab if available.

    If CoreTab is unavailable or fails, falls back to a stratified sampling
    scheme on the target variable (with binning for regression-like targets).

    Args:
        primary_df: The primary dataset containing features and the target column.
        percent: Percentage of primary_df to keep in the coreset, (0, 100].
        algo: Coreset algorithm to use when CoreTab is available: 'dt' or 'xgb'.
        random_state: Random seed for reproducibility.

    Returns:
        pd.DataFrame: Subset of primary_df with the same schema, including the target column.

    Raises:
        ValueError: If inputs are invalid or the target column cannot be resolved.
    """
    if primary_df is None or len(primary_df) == 0:
        raise ValueError("[FINETUNE] primary_df is empty or None.")

    target_col = _resolve_target_column(primary_df)
    examples_to_keep = _compute_examples_to_keep(len(primary_df), percent)
    print(f"[FINETUNE] Examples to keep: {examples_to_keep}")
    # Attempt CoreTab-based coreset selection
    try:
        # Import lazily to avoid hard dependency when not used
        from coretab.coretab.coreset_algorithms import CoreTabDT, CoreTabXGB  # type: ignore

        # Prepare features/labels
        X = primary_df.drop(columns=[target_col])
        y = primary_df[target_col]

        if algo not in {"dt", "xgb"}:
            raise ValueError("[FINETUNE] algo must be one of {'dt', 'xgb'}.")

        # One-hot encode non-numeric columns for model consumption (preserve index)
        non_numeric_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
        X_enc = pd.get_dummies(X, columns=non_numeric_cols, drop_first=False) if non_numeric_cols else X

        if algo == "xgb":
            model = CoreTabXGB(examples_to_keep=examples_to_keep)
        else:
            model = CoreTabDT(examples_to_keep=examples_to_keep)

        # Run coreset selection on encoded features
        X_core, y_core = model.create_coreset(X_enc, y)

        # Use selected indices to slice original DataFrame to preserve schema
        coreset = primary_df.loc[X_core.index].copy()

        # Sanity: ensure at least one row
        if len(coreset) == 0:
            raise RuntimeError("[FINETUNE] CoreTab returned an empty coreset.")

        print(
            f"[FINETUNE] CoreTab coreset selected: algo={algo}, kept={len(coreset)}/{len(primary_df)}"
        )
        return coreset

    except Exception as e:
        # Fallback path: stratified sampling on target
        print(f"[FINETUNE] CoreTab coreset selection failed or unavailable: {e}. Using fallback.")
        fallback = _fallback_stratified_sample(
            df=primary_df,
            target_col=target_col,
            n_samples=examples_to_keep,
            random_state=random_state,
        )
        # Preserve original column order explicitly
        fallback = fallback[primary_df.columns]
        print(
            f"[FINETUNE] Fallback coreset selected: kept={len(fallback)}/{len(primary_df)}"
        )
        return fallback


def merge_datasets(
    primary_df: pd.DataFrame,
    secondary_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge primary and secondary datasets for retrain strategy.
    
    This function concatenates both datasets vertically, ensuring they have
    matching schemas. The result is a single DataFrame with all rows from
    both datasets, with the index reset for clean sequential numbering.
    
    Args:
        primary_df: Primary dataset (old/baseline data)
        secondary_df: Secondary dataset (new/drifted data)
        
    Returns:
        pd.DataFrame: Concatenated dataset with reset index
        
    Raises:
        ValueError: If inputs are invalid or schemas don't match
    """
    if primary_df is None or len(primary_df) == 0:
        raise ValueError("[RETRAIN] primary_df is empty or None.")
    if secondary_df is None or len(secondary_df) == 0:
        raise ValueError("[RETRAIN] secondary_df is empty or None.")
    
    # Verify schemas match (same columns, though order doesn't matter for concat)
    primary_cols = set(primary_df.columns)
    secondary_cols = set(secondary_df.columns)
    
    if primary_cols != secondary_cols:
        missing_in_secondary = primary_cols - secondary_cols
        missing_in_primary = secondary_cols - primary_cols
        error_msg = "[RETRAIN] Primary and secondary datasets have different schemas."
        if missing_in_secondary:
            error_msg += f"\n  Missing in secondary: {missing_in_secondary}"
        if missing_in_primary:
            error_msg += f"\n  Missing in primary: {missing_in_primary}"
        raise ValueError(error_msg)
    
    # Ensure column order matches (use primary's order as reference)
    secondary_df = secondary_df[primary_df.columns]
    
    # Merge and reset index
    merged_df = pd.concat([primary_df, secondary_df], axis=0, ignore_index=True)
    
    print(
        f"[RETRAIN] Datasets merged successfully: "
        f"primary={len(primary_df)}, secondary={len(secondary_df)}, total={len(merged_df)}"
    )
    
    return merged_df


