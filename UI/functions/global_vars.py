"""
Global variables and state management for the DriftNavi application.

This module provides centralized access to shared state across the application,
particularly enabling seamless integration between Detect and Explain functionalities.
"""

class GlobalVars:
    """
    Singleton class for managing global application state.
    
    This class stores shared state that needs to be accessible across different
    components of the application, especially for maintaining continuity between
    the Detect and Explain analysis pipeline stages.
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GlobalVars, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize default values for global variables."""
        # Basic app variables
        self.df = None  # Main dataset
        self.df_secondary = None  # Secondary dataset (legacy name)
        self.secondary_df = None  # Secondary dataset (preferred name)
        self.primary_dataset = None
        self.secondary_dataset = None
        self.target_attribute = None
        self.file_name = None  # Name of the primary dataset file
        self.secondary_file_name = None  # Name of the secondary dataset file
        self.conversation_session = None  # Conversation session ID
        
        # App state variables
        self.current_stage = "preview"  # Direct access for backward compatibility
        self.agent = None  # LLM agent instance
        
        # RAG feature variables
        self.rag = None
        self.use_rag = False
        self.rag_prompt = None
        
        # Chat history variables
        self.dialog = []  # Store chat dialog history
        self.suggested_questions = []  # Store suggested questions for chat UI
        
        # Analysis context for sharing state between Detect and Explain
        self.analysis_context = {
            "current_focus": None,      # Currently focused attribute for analysis
            "analysis_path": [],         # History of analysis actions for context
        }
        
        # Context data for explain phase (stores items added via "Add to Explain" buttons)
        self.explain_context_data = []
        
        # Store metrics results to avoid recalculation
        self.metrics_cache = None
        
        # Selected metrics from Detect phase
        self.selected_metrics = []
        
        # Target attribute related data
        self.target_attribute_stats = {
            "primary": {},   # Statistics for target in primary dataset
            "secondary": {}, # Statistics for target in secondary dataset
            "related_attributes": [],  # Attributes highly related to target
            "impact_scores": {},       # Impact scores of attributes on target
        }
        
        # UI state tracking to prevent duplicate buttons
        self.target_attribute_button_added = False
        
        # Dataset change tracking
        self.dataset_fingerprints = {
            'primary': None,
            'secondary': None
        }
        self.dataset_change_flags = {
            'primary_changed': False,
            'secondary_changed': False,
            'metrics_outdated': False
        }
        
        # Adaptation strategy variables
        self.adaptation_strategy = None  # Store user's selected strategy (retrain/finetune)
        self.adaptation_strategy_selected = False  # Track if strategy was selected
        self.adaptation_strategy_timestamp = None  # When strategy was selected
        
        # Finetune workflow data (stored separately from originals)
        self.primary_coreset_df = None
        self.secondary_resampled_df = None
        self.finetune_combined_df = None
        
        # Retrain workflow data (stored separately from originals)
        self.retrain_merged_df = None      # Primary + Secondary merged (before resampling)
        self.retrain_resampled_df = None   # After resampling on merged data

    def get_user_context(self, current_user):
        """
        Extract dynamic user context information from the actual user database record.
        
        This method reads real user profile data without any hardcoded fallbacks,
        returning None for missing information and allowing the analysis to adapt
        based on what's actually available.
        
        Args:
            current_user: Flask-Login current user object
            
        Returns:
            dict: User context with actual database values or None for missing fields
        """
        try:
            # Check if user is authenticated and has required attributes
            if not current_user or not hasattr(current_user, 'is_authenticated') or not current_user.is_authenticated:
                return {
                    "has_profile": False,
                    "professional_role": None,
                    "industry_sector": None,
                    "expertise_level": None,
                    "technical_level": None,
                    "drift_awareness": None,
                    "areas_of_interest": None,
                    "persona_prompt": None,
                    "system_prompt": None,
                    "prefix_prompt": None,
                    "profile_completeness": 0
                }
            
            # Extract actual user data directly from database fields
            user_context = {
                "has_profile": True,
                "professional_role": getattr(current_user, 'professional_role', None),
                "industry_sector": getattr(current_user, 'industry_sector', None),
                "expertise_level": getattr(current_user, 'expertise_level', None),
                "technical_level": getattr(current_user, 'technical_level', None),
                "drift_awareness": getattr(current_user, 'drift_awareness', None),
                "areas_of_interest": getattr(current_user, 'areas_of_interest', None),
                "persona_prompt": getattr(current_user, 'persona_prompt', None),
                "system_prompt": getattr(current_user, 'system_prompt', None),
                "prefix_prompt": getattr(current_user, 'prefix_prompt', None)
            }
            
            # Calculate profile completeness dynamically
            key_fields = ['professional_role', 'industry_sector', 'expertise_level', 
                         'technical_level', 'drift_awareness', 'areas_of_interest']
            completed_fields = sum(1 for field in key_fields if user_context[field] is not None)
            user_context["profile_completeness"] = (completed_fields / len(key_fields)) * 100
            
            return user_context
            
        except Exception as e:
            print(f"[USER CONTEXT] Error extracting user context: {str(e)}")
            # Return structure indicating no profile data available
            return {
                "has_profile": False,
                "professional_role": None,
                "industry_sector": None,
                "expertise_level": None,
                "technical_level": None,
                "drift_awareness": None,
                "areas_of_interest": None,
                "persona_prompt": None,
                "system_prompt": None,
                "prefix_prompt": None,
                "profile_completeness": 0,
                "error": str(e)
            }
    
    def reset(self):
        """Reset all global variables to their default state."""
        self._initialize()
    
    def store_metrics_results(self, metrics_data):
        """
        Store metrics calculation results for reuse across components.
        
        Args:
            metrics_data (list): The metrics data calculated in Detect phase
        """
        # Use the new cache_metrics method to maintain proper structure
        # If metrics_data is already cached properly, don't overwrite
        if metrics_data and not self.is_cache_valid()[0]:
            print(f"[METRICS CACHE] store_metrics_results called, using cache_metrics method")
            self.cache_metrics(metrics_data, None, force=True)
    
    def set_focus_attribute(self, attribute_name, source="detect"):
        """
        Set the current analysis focus to a specific attribute.
        
        Args:
            attribute_name (str): Name of the attribute to focus on
            source (str): Source of the focus change (detect, explain, etc.)
        """
        prev_focus = self.analysis_context.get("current_focus")
        
        # Only add to path if this is a new focus
        if prev_focus != attribute_name:
            self.analysis_context["current_focus"] = attribute_name
            
            # Record in analysis path
            self.analysis_context["analysis_path"].append({
                "action": "focus_change",
                "attribute": attribute_name,
                "source": source,
                "previous": prev_focus
            })
    
    
    def add_selected_metric(self, metric_name, metric_value):
        """
        Add a metric that the user has shown interest in.
        
        Args:
            metric_name (str): Name of the metric
            metric_value (float): Value of the metric
        """
        # Check if this metric is already selected
        for idx, metric in enumerate(self.selected_metrics):
            if metric["name"] == metric_name:
                # Update existing entry
                self.selected_metrics[idx]["value"] = metric_value
                return
                
        # Add new metric
        self.selected_metrics.append({
            "name": metric_name,
            "value": metric_value,
            "timestamp": None  # Could add timestamp if needed
        })
    
    def update_target_stats(self, primary_stats, secondary_stats):
        """
        Update statistics about the target attribute.
        
        Args:
            primary_stats (dict): Statistics from primary dataset
            secondary_stats (dict): Statistics from secondary dataset
        """
        self.target_attribute_stats["primary"] = primary_stats
        self.target_attribute_stats["secondary"] = secondary_stats
    
    # === METRICS CACHE MANAGEMENT METHODS ===
    
    def clear_metrics_cache(self, reason="Manual clear"):
        """
        Clear the metrics cache.
        
        Args:
            reason (str): Reason for clearing the cache for logging
        """
        if hasattr(self, 'metrics_cache') and self.metrics_cache is not None:
            print(f"[METRICS CACHE] Clearing cache: {reason}")
            self.metrics_cache = None
            return True
        return False
    
    def cache_metrics(self, metrics_data, data_length, force=False):
        """
        Cache metrics data with metadata.
        
        Args:
            metrics_data (list): The calculated metrics data
            data_length (tuple): Data length information
            force (bool): Whether to force caching even if cache exists
            
        Returns:
            bool: True if caching was successful
        """
        try:
            if self.metrics_cache is not None and not force:
                print("[METRICS CACHE] Cache already exists, use force=True to overwrite")
                return False
            
            if not metrics_data:
                print("[METRICS CACHE] Cannot cache empty metrics data")
                return False
            
            import pandas as pd
            self.metrics_cache = {
                'data': metrics_data,
                'data_length': data_length,
                'target_attribute': self.target_attribute,
                'calculated_at': pd.Timestamp.now(),
                'column_types_snapshot': self.column_types.copy() if hasattr(self, 'column_types') else {}
            }
            print(f"[METRICS CACHE] Successfully cached metrics for {len(metrics_data)} columns")
            return True
            
        except Exception as e:
            print(f"[METRICS CACHE] Error caching metrics: {str(e)}")
            return False
    
    def is_cache_valid(self, max_age_hours=1):
        """
        Check if the current metrics cache is valid.
        
        Args:
            max_age_hours (int): Maximum age of cache in hours
            
        Returns:
            tuple: (is_valid, reason) where reason explains why cache is invalid
        """
        if not hasattr(self, 'metrics_cache') or self.metrics_cache is None:
            return False, "No cache exists"
        
        cache = self.metrics_cache
        
        # Ensure cache is a dict and has the required structure
        if not isinstance(cache, dict):
            return False, f"Cache is not a dict (type: {type(cache)})"
        
        # Check target attribute
        cached_target = cache.get('target_attribute')
        if cached_target != self.target_attribute:
            return False, f"Target attribute changed (cached: {cached_target}, current: {self.target_attribute})"
        
        # Check column types
        current_column_types = self.column_types if hasattr(self, 'column_types') else {}
        cached_column_types = cache.get('column_types_snapshot', {})
        if current_column_types != cached_column_types:
            return False, "Column types changed"
        
        # Check age
        try:
            import pandas as pd
            calculated_at = cache.get('calculated_at')
            if calculated_at is None:
                return False, "Cache missing timestamp"
            
            if isinstance(calculated_at, str):
                calculated_at = pd.Timestamp(calculated_at)
            
            cache_age = pd.Timestamp.now() - calculated_at
            if cache_age.total_seconds() > (max_age_hours * 3600):
                return False, f"Cache too old ({cache_age})"
        except Exception as e:
            return False, f"Error checking cache age: {str(e)}"
        
        return True, "Valid"
    
    def get_cached_metrics(self):
        """
        Get cached metrics if valid, otherwise return None.
        
        Returns:
            tuple: (metrics_data, data_length) or (None, None) if invalid
        """
        is_valid, reason = self.is_cache_valid()
        
        if is_valid:
            cache = self.metrics_cache
            return cache['data'], cache['data_length']
        else:
            print(f"[METRICS CACHE] Cache invalid: {reason}")
            return None, None
    
    # === DATASET CHANGE DETECTION METHODS ===
    
    def generate_dataset_fingerprint(self, df):
        """
        Generate a lightweight fingerprint for dataset change detection.
        
        Args:
            df (pd.DataFrame): Dataset to fingerprint
            
        Returns:
            dict: Fingerprint containing shape, columns, dtypes, and content sample
        """
        if df is None or df.empty:
            return None
            
        try:
            import hashlib
            
            # Basic structure fingerprint
            fingerprint = {
                'shape': df.shape,
                'columns': list(df.columns),
                'dtypes_hash': hash(str(df.dtypes.to_dict())),
            }
            
            # Content fingerprint using sample data
            if len(df) > 0:
                sample_size = min(10, len(df))
                sample_data = df.head(sample_size)
                
                # Convert to string and hash for content comparison
                content_str = str(sample_data.values.tobytes())
                fingerprint['content_hash'] = hashlib.md5(content_str.encode()).hexdigest()
            else:
                fingerprint['content_hash'] = None
                
            return fingerprint
            
        except Exception as e:
            print(f"[DATASET FINGERPRINT] Error generating fingerprint: {str(e)}")
            return None
    
    def detect_dataset_changes(self, force_update=False):
        """
        Detect changes in primary and secondary datasets by comparing fingerprints.
        
        Args:
            force_update (bool): Force update fingerprints even if no changes detected
            
        Returns:
            dict: Change detection results
        """
        changes = {
            'primary_changed': False,
            'secondary_changed': False,
            'any_changed': False,
            'details': {}
        }
        
        try:
            # Check primary dataset
            if self.df is not None:
                current_primary_fp = self.generate_dataset_fingerprint(self.df)
                stored_primary_fp = self.dataset_fingerprints.get('primary')
                
                if stored_primary_fp is None or current_primary_fp != stored_primary_fp or force_update:
                    changes['primary_changed'] = True
                    changes['details']['primary'] = {
                        'previous': stored_primary_fp,
                        'current': current_primary_fp
                    }
                    if current_primary_fp:
                        self.dataset_fingerprints['primary'] = current_primary_fp
                        print(f"[DATASET CHANGE] Primary dataset changed: {current_primary_fp['shape']}")
            
            # Check secondary dataset with proper attribute checking
            # Try secondary_df first (preferred), fall back to df_secondary (legacy)
            secondary_dataset = None
            if hasattr(self, 'secondary_df') and self.secondary_df is not None:
                secondary_dataset = self.secondary_df
            elif hasattr(self, 'df_secondary') and self.df_secondary is not None:
                secondary_dataset = self.df_secondary
            
            if secondary_dataset is not None:
                current_secondary_fp = self.generate_dataset_fingerprint(secondary_dataset)
                stored_secondary_fp = self.dataset_fingerprints.get('secondary')
                
                if stored_secondary_fp is None or current_secondary_fp != stored_secondary_fp or force_update:
                    changes['secondary_changed'] = True
                    changes['details']['secondary'] = {
                        'previous': stored_secondary_fp,
                        'current': current_secondary_fp
                    }
                    if current_secondary_fp:
                        self.dataset_fingerprints['secondary'] = current_secondary_fp
                        print(f"[DATASET CHANGE] Secondary dataset changed: {current_secondary_fp['shape']}")
            
            # Update change flags
            changes['any_changed'] = changes['primary_changed'] or changes['secondary_changed']
            
            if changes['any_changed']:
                self.dataset_change_flags['primary_changed'] = changes['primary_changed']
                self.dataset_change_flags['secondary_changed'] = changes['secondary_changed']
                self.dataset_change_flags['metrics_outdated'] = True
                
                # Clear metrics cache when datasets change
                self.clear_metrics_cache("Dataset changes detected")
                
                print(f"[DATASET CHANGE] Changes detected - Primary: {changes['primary_changed']}, Secondary: {changes['secondary_changed']}")
            
            return changes
            
        except Exception as e:
            print(f"[DATASET CHANGE] Error detecting changes: {str(e)}")
            return changes
    
    def reset_change_flags(self):
        """Reset dataset change flags after metrics have been recomputed."""
        self.dataset_change_flags = {
            'primary_changed': False,
            'secondary_changed': False,
            'metrics_outdated': False
        }
        print("[DATASET CHANGE] Change flags reset")
    
    def initialize_dataset_fingerprints(self, force=False):
        """
        Initialize fingerprints for current datasets.
        
        Args:
            force (bool): Force re-initialization even if fingerprints exist
        """
        try:
            if self.df is not None and (self.dataset_fingerprints['primary'] is None or force):
                self.dataset_fingerprints['primary'] = self.generate_dataset_fingerprint(self.df)
                print(f"[DATASET FINGERPRINT] Primary dataset fingerprint initialized")
            
            # Initialize secondary fingerprint with proper attribute checking
            # Try secondary_df first (preferred), fall back to df_secondary (legacy)
            secondary_dataset = None
            if hasattr(self, 'secondary_df') and self.secondary_df is not None:
                secondary_dataset = self.secondary_df
            elif hasattr(self, 'df_secondary') and self.df_secondary is not None:
                secondary_dataset = self.df_secondary
            
            if secondary_dataset is not None and (self.dataset_fingerprints['secondary'] is None or force):
                self.dataset_fingerprints['secondary'] = self.generate_dataset_fingerprint(secondary_dataset)
                print(f"[DATASET FINGERPRINT] Secondary dataset fingerprint initialized")
                
        except Exception as e:
            print(f"[DATASET FINGERPRINT] Error initializing fingerprints: {str(e)}")
    
    def are_metrics_outdated(self):
        """
        Check if metrics need to be recomputed due to dataset changes.
        
        Returns:
            bool: True if metrics are outdated and need recomputation
        """
        # Always check for dataset changes first
        self.detect_dataset_changes()
        
        # Check if change flags indicate outdated metrics
        if self.dataset_change_flags.get('metrics_outdated', False):
            return True
            
        # Also check if cache is invalid for other reasons
        is_valid, reason = self.is_cache_valid()
        if not is_valid:
            print(f"[METRICS STATUS] Metrics outdated: {reason}")
            return True
            
        return False
    
    def set_adaptation_strategy(self, strategy):
        """
        Set the user's selected adaptation strategy.
        
        Args:
            strategy (str): Selected strategy ('retrain' or 'finetune')
        """
        import time
        if strategy in ['retrain', 'finetune']:
            self.adaptation_strategy = strategy
            self.adaptation_strategy_selected = True
            self.adaptation_strategy_timestamp = time.time()
            print(f"[ADAPTATION STRATEGY] Strategy set to: {strategy}")
        else:
            print(f"[ADAPTATION STRATEGY] Invalid strategy: {strategy}")
    
    def get_adaptation_strategy(self):
        """
        Get the current adaptation strategy.
        
        Returns:
            str or None: Current adaptation strategy, or None if not set
        """
        return self.adaptation_strategy
    
    def is_adaptation_strategy_selected(self):
        """
        Check if an adaptation strategy has been selected.
        
        Returns:
            bool: True if strategy has been selected
        """
        return self.adaptation_strategy_selected
    
    def reset_adaptation_strategy(self):
        """Reset adaptation strategy to default state."""
        self.adaptation_strategy = None
        self.adaptation_strategy_selected = False
        self.adaptation_strategy_timestamp = None
        print("[ADAPTATION STRATEGY] Strategy reset")


# Create global instance
global_vars = GlobalVars()
