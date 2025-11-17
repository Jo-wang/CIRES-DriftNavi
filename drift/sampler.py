"""
Multi-Attribute Sampler: balance multiple attributes and target variables

Supports three resampling strategies:
- Oversampling
- Undersampling  
- SMOTE

Can balance multiple attributes (e.g., gender, race) and target variables (e.g., approved) simultaneously.
"""
from imblearn.over_sampling import SMOTENC
from sklearn.utils import resample
import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np


class Sampler:
    def __init__(self, df, target_cols):
        """
        Initialize the Sampler class.

        Parameters:
        - df (pd.DataFrame): The input dataframe.
        - target_col (str): The name of the target column.
        """
        self.df = df
        self.target_col = target_cols

    def _get_categorical_indices(self, X):
        """
        Identify indices of categorical features.

        Parameters:
        - X (pd.DataFrame): Feature DataFrame.

        Returns:
        - List[int]: Indices of categorical columns.
        """
        return [i for i, dtype in enumerate(X.dtypes) if dtype == 'object']

    def undersample(self, random_state=42):
        """
        Perform random undersampling to balance the classes.

        Parameters:
        - random_state (int): Random state for reproducibility.

        Returns:
        - pd.DataFrame: A new DataFrame with balanced classes.
        """
        min_class_size = self.df[self.target_col].value_counts().min()

        # Downsample each class
        balanced_dfs = []
        for class_label in self.df[self.target_col].unique():
            class_data = self.df[self.df[self.target_col] == class_label]
            downsampled_data = resample(
                class_data,
                replace=False,
                n_samples=min_class_size,
                random_state=random_state
            )
            balanced_dfs.append(downsampled_data)

        balanced_df = pd.concat(balanced_dfs)
        return balanced_df

    def oversample(self, random_state=42):
        """
        Perform random oversampling to balance the classes.

        Parameters:
        - random_state (int): Random state for reproducibility.

        Returns:
        - pd.DataFrame: A new DataFrame with balanced classes.
        """
        max_class_size = self.df[self.target_col].value_counts().max()

        # Oversample each class
        balanced_dfs = []
        for class_label in self.df[self.target_col].unique():
            class_data = self.df[self.df[self.target_col] == class_label]
            if class_data.empty:
                print(f"Warning: No samples found for class '{class_label}'. Skipping.")
                continue
            oversampled_data = resample(
                class_data,
                replace=True,
                n_samples=max_class_size,
                random_state=random_state
            )
            balanced_dfs.append(oversampled_data)

        balanced_df = pd.concat(balanced_dfs).reset_index(drop=True)
        return balanced_df

    def smote(self, random_state=42):
        """
        Perform SMOTE-NC to balance the classes in a dataset with mixed data types,
        and fallback to random oversampling for categorical-only datasets.

        Parameters:
        - random_state (int): Random state for reproducibility.

        Returns:
        - pd.DataFrame: A new DataFrame with balanced classes.
        """
        # Split data into features and target
        X = self.df.drop(columns=[self.target_col])
        if X.empty:
            raise ValueError("Input features are empty. Cannot perform SMOTE.")
        y = self.df[self.target_col]

        # Identify columns with all NaN values
        all_nan_columns = X.columns[X.isna().all()].tolist()

        # Remove columns with all NaN values for SMOTE
        X_non_nan = X.drop(columns=all_nan_columns)

        # Handle NaN values: Impute missing data for non-NaN columns
        imputer = SimpleImputer(strategy='most_frequent')
        X_imputed = pd.DataFrame(imputer.fit_transform(X_non_nan), columns=X_non_nan.columns)

        # Get categorical column indices
        categorical_indices = self._get_categorical_indices(X_imputed)

        # Check if the dataset has numerical features
        num_numerical_features = len(X_imputed.select_dtypes(include=['float64', 'int64']).columns)

        if num_numerical_features == 0:
            # Fallback to random oversampling if no numerical features exist
            print("No numerical features found. Falling back to random oversampling.")
            max_class_size = y.value_counts().max()

            # Oversample each class
            balanced_dfs = []
            for class_label in y.unique():
                class_data = self.df[self.df[self.target_col] == class_label]
                if class_data.empty:
                    print(f"Warning: No samples found for class '{class_label}'. Skipping.")
                    continue
                oversampled_data = resample(
                    class_data,
                    replace=True,
                    n_samples=max_class_size,
                    random_state=random_state
                )
                balanced_dfs.append(oversampled_data)

            balanced_df = pd.concat(balanced_dfs).reset_index(drop=True)
            return balanced_df

        # Determine the smallest class size
        class_counts = y.value_counts()
        min_class_size = class_counts.min()

        # Ensure k_neighbors is valid
        k_neighbors = max(1, min(5, min_class_size - 1))

        # Apply SMOTE-NC for mixed data
        smote_nc = SMOTENC(
            categorical_features=categorical_indices,
            k_neighbors=k_neighbors,
            random_state=random_state
        )
        X_resampled, y_resampled = smote_nc.fit_resample(X_imputed, y)

        # Combine resampled features and target into a DataFrame
        resampled_df = pd.concat(
            [pd.DataFrame(X_resampled, columns=X_imputed.columns),
             pd.DataFrame(y_resampled, columns=[self.target_col])],
            axis=1
        )

        # Add back all-NaN columns to the end
        for col in all_nan_columns:
            resampled_df[col] = np.nan

        # Reorder columns to place all-NaN columns at the end
        reordered_columns = [col for col in resampled_df.columns if col != self.target_col] + [self.target_col]
        resampled_df = resampled_df[reordered_columns]

        return resampled_df.reset_index(drop=True)


class MultiAttributeSampler(Sampler):
    """
    Extended sampler that supports balancing multiple attributes simultaneously.
    
    Can balance target variables and sensitive attributes (e.g., gender, race) simultaneously.
    """
    
    def __init__(self, df, target_col, protected_attrs=None):
        """
        Initialize the multi-attribute sampler.
        
        Parameters:
        - df (pd.DataFrame): Input dataframe
        - target_col (str): Target column name (the variable to predict)
        - protected_attrs (list): List of other attributes to balance simultaneously
                                  e.g., ['gender', 'race', 'age_group']
        
        Example:
        >>> sampler = MultiAttributeSampler(
        ...     df, 
        ...     target_col='approved',
        ...     protected_attrs=['gender', 'race']
        ... )
        """
        super().__init__(df, target_col)
        self.protected_attrs = protected_attrs or []
        
        # Verify all columns exist
        missing_cols = [col for col in [target_col] + self.protected_attrs 
                       if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Columns not found in dataframe: {missing_cols}")
    
    def _is_continuous(self, col_name, unique_threshold=10):
        """
        Automatically detect if a column is continuous.
        
        A column is considered continuous if:
        1. It has a numeric dtype (int or float)
        2. It has more unique values than the threshold
        
        Parameters:
        - col_name (str): Column name to check
        - unique_threshold (int): Minimum unique values to consider continuous
        
        Returns:
        - bool: True if column is continuous, False otherwise
        """
        if col_name not in self.df.columns:
            return False
        
        col = self.df[col_name]
        
        # Check if numeric type
        is_numeric = pd.api.types.is_numeric_dtype(col)
        
        if not is_numeric:
            return False
        
        # Check number of unique values (excluding NaN)
        n_unique = col.dropna().nunique()
        
        # If unique values exceed threshold, consider it continuous
        return n_unique > unique_threshold
    
    def _auto_bin_column(self, col_name, n_bins='auto'):
        """
        Automatically bin a continuous column into discrete categories.
        
        Uses quantile-based binning (qcut) to ensure balanced bin sizes.
        Falls back to equal-width binning (cut) if quantile binning fails.
        
        Parameters:
        - col_name (str): Column name to bin
        - n_bins (int or 'auto'): Number of bins. If 'auto', determines adaptively.
        
        Returns:
        - pd.Series: Binned column with string labels
        """
        col = self.df[col_name]
        
        # Handle missing values - will be converted to 'missing' category
        col_dropna = col.dropna()
        
        if len(col_dropna) == 0:
            # All values are NaN
            return pd.Series(['missing'] * len(col), index=col.index)
        
        # Determine number of bins adaptively
        if n_bins == 'auto':
            n_unique = col_dropna.nunique()
            n_samples = len(col_dropna)
            
            # Adaptive bin selection:
            # - At least 3 bins for basic distribution
            # - At most 10 bins to avoid over-fragmentation
            # - Ensure each bin has sufficient samples (at least 5% of data)
            max_bins_by_samples = max(3, min(10, n_samples // 20))
            max_bins_by_unique = min(10, n_unique // 2)
            
            n_bins = min(max_bins_by_samples, max_bins_by_unique)
            n_bins = max(3, n_bins)  # At least 3 bins
        
        try:
            # Try quantile-based binning (ensures equal-sized bins)
            binned = pd.qcut(
                col, 
                q=n_bins, 
                labels=[f"{col_name}_Q{i+1}" for i in range(n_bins)],
                duplicates='drop'  # Handle duplicate bin edges
            )
        except (ValueError, TypeError) as e:
            # Fallback to equal-width binning if qcut fails
            print(f"  [AUTO-BIN] Quantile binning failed for '{col_name}', using equal-width binning: {e}")
            try:
                binned = pd.cut(
                    col,
                    bins=n_bins,
                    labels=[f"{col_name}_B{i+1}" for i in range(n_bins)],
                    include_lowest=True
                )
            except (ValueError, TypeError) as e2:
                # Ultimate fallback: just use string representation
                print(f"  [AUTO-BIN] All binning failed for '{col_name}', using raw values: {e2}")
                return col.astype(str).fillna('missing')
        
        # Convert to string and handle NaN
        binned_str = binned.astype(str)
        binned_str = binned_str.fillna('missing')
        
        return binned_str
    
    def _create_combined_class(self):
        """
        Create a combined class column with automatic continuous variable binning.
        
        This method intelligently handles mixed data types:
        1. Automatically detects continuous variables (numeric with many unique values)
        2. Bins continuous variables into discrete categories
        3. Combines all columns (binned + categorical) into a single class label
        
        Returns:
        - str: The name of the combined column
        """
        # Ensure column names are unique
        combined_col_name = '_combined_class_temp_'
        
        # Combine all columns to balance
        cols_to_combine = self.protected_attrs + [self.target_col]
        
        print(f"[MULTI-ATTR] Creating combined class from: {cols_to_combine}")
        
        # Process each column - bin if continuous, use as-is if categorical
        processed_cols = []
        for col in cols_to_combine:
            if self._is_continuous(col):
                print(f"  [AUTO-BIN] '{col}' detected as continuous, binning...")
                binned_col = self._auto_bin_column(col)
                processed_cols.append(binned_col)
                
                # Show binning results
                n_bins = binned_col.nunique()
                print(f"  [AUTO-BIN] '{col}' binned into {n_bins} categories")
            else:
                print(f"  [AUTO-BIN] '{col}' detected as categorical, using raw values")
                processed_cols.append(self.df[col].astype(str))
        
        # Join all processed column values using underscores
        self.df[combined_col_name] = pd.concat(processed_cols, axis=1).agg('_'.join, axis=1)
        
        n_combined_classes = self.df[combined_col_name].nunique()
        print(f"[MULTI-ATTR] Created {n_combined_classes} combined classes")
        
        return combined_col_name
    
    def get_distribution_stats(self):
        """
        Get the distribution statistics of the current data.
        
        Returns:
        - dict: Distribution information for each attribute and combination
        """
        stats = {}
        
        # Target variable distribution
        stats['target'] = self.df[self.target_col].value_counts().to_dict()
        
        # Distribution of protected attributes
        for attr in self.protected_attrs:
            stats[attr] = self.df[attr].value_counts().to_dict()
        
        # Combined distribution
        if self.protected_attrs:
            combined_col = self._create_combined_class()
            stats['combined'] = self.df[combined_col].value_counts().to_dict()
            self.df = self.df.drop(combined_col, axis=1)
        
        return stats
    
    def oversample_multiattr(self, random_state=42, verbose=True):
        """
        Perform oversampling for multiple attributes.
        
        Boost all combined classes to the maximum number of samples in any combined class.
        
        Parameters:
        - random_state (int): Random seed
        - verbose (bool): Whether to print detailed information
        
        Returns:
        - pd.DataFrame: Balanced DataFrame
        """
        if verbose:
            print(f"Starting multi-attribute oversampling...")
            print(f"Target column: {self.target_col}")
            print(f"Protected attributes: {self.protected_attrs}")
        
        # Create combined classes
        combined_col = self._create_combined_class()
        
        # Get statistics
        value_counts = self.df[combined_col].value_counts()
        max_class_size = value_counts.max()
        
        if verbose:
            print(f"\nNumber of combined classes: {len(value_counts)}")
            print(f"Maximum combined class size: {max_class_size}")
            print(f"Minimum combined class size: {value_counts.min()}")
        
        # Oversample each combined class
        balanced_dfs = []
        for class_label in self.df[combined_col].unique():
            class_data = self.df[self.df[combined_col] == class_label]
            
            if class_data.empty:
                if verbose:
                    print(f"Warning: No samples found for class '{class_label}'. Skipping.")
                continue
            
            current_size = len(class_data)
            oversampled_data = resample(
                class_data,
                replace=True,
                n_samples=max_class_size,
                random_state=random_state
            )
            
            if verbose and current_size < max_class_size:
                print(f"  {class_label}: {current_size} -> {max_class_size}")
            
            balanced_dfs.append(oversampled_data)
        
        # Combine all data
        balanced_df = pd.concat(balanced_dfs).reset_index(drop=True)
        
        # Delete temporary combined column
        balanced_df = balanced_df.drop(combined_col, axis=1)
        
        if verbose:
            print(f"\n✓ Oversampling completed! Original sample size: {len(self.df)}, Balanced sample size: {len(balanced_df)}")
        
        return balanced_df
    
    def undersample_multiattr(self, random_state=42, verbose=True):
        """
        Perform undersampling for multiple attributes.
        
        Reduce all combined classes to the minimum number of samples in any combined class.
        
        Parameters:
        - random_state (int): Random seed
        - verbose (bool): Whether to print detailed information
        
        Returns:
        - pd.DataFrame: Balanced DataFrame
        """
        if verbose:
            print(f"Starting multi-attribute undersampling...")
            print(f"Target column: {self.target_col}")
            print(f"Protected attributes: {self.protected_attrs}")
        
        # Create combined classes
        combined_col = self._create_combined_class()
        
        # Get statistics
        value_counts = self.df[combined_col].value_counts()
        min_class_size = value_counts.min()
        
        if verbose:
            print(f"\nNumber of combined classes: {len(value_counts)}")
            print(f"Maximum combined class size: {value_counts.max()}")
            print(f"Minimum combined class size: {min_class_size}")
        
        # Undersample each combined class
        balanced_dfs = []
        for class_label in self.df[combined_col].unique():
            class_data = self.df[self.df[combined_col] == class_label]
            
            current_size = len(class_data)
            downsampled_data = resample(
                class_data,
                replace=False,
                n_samples=min_class_size,
                random_state=random_state
            )
            
            if verbose and current_size > min_class_size:
                print(f"  {class_label}: {current_size} -> {min_class_size}")
            
            balanced_dfs.append(downsampled_data)
        
        # Combine all data
        balanced_df = pd.concat(balanced_dfs).reset_index(drop=True)
        
        # Delete temporary combined column
        balanced_df = balanced_df.drop(combined_col, axis=1)
        
        if verbose:
            print(f"\n✓ Undersampling completed! Original sample size: {len(self.df)}, Balanced sample size: {len(balanced_df)}")
        
        return balanced_df
    
    def smote_multiattr(self, random_state=42, verbose=True):
        """
            Perform SMOTE for multiple attributes.
        
        Note: SMOTE requires numerical features. For purely categorical data, it will automatically downgrade to oversampling.
        
        Parameters:
        - random_state (int): Random seed
        - verbose (bool): Whether to print detailed information
        
        Returns:
        - pd.DataFrame: Balanced DataFrame
        """
        if verbose:
            print(f"Starting multi-attribute SMOTE sampling...")
            print(f"Target column: {self.target_col}")
            print(f"Protected attributes: {self.protected_attrs}")
        
        # Create combined classes as temporary target
        combined_col = self._create_combined_class()
        
        # Temporarily set combined column as target
        original_target = self.target_col
        self.target_col = combined_col
        
        # Use the SMOTE method of the parent class
        balanced_df = self.smote(random_state=random_state)
        
        # Restore original target column
        self.target_col = original_target
        
        # Delete temporary combined column
        if combined_col in balanced_df.columns:
            balanced_df = balanced_df.drop(combined_col, axis=1)
        
        if verbose:
            print(f"\n✓ SMOTE sampling completed! Original sample size: {len(self.df)}, Balanced sample size: {len(balanced_df)}")
        
        return balanced_df


