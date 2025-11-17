"""
Shared constants for the DriftNavi application.

This module contains centralized constants used across the application
to avoid duplication and ensure consistency.
"""

# Pipeline stages for drift management
PIPELINE_STAGES = ["Detect", "Explain", "Adapt"]

# Default initial stage
DEFAULT_STAGE = "Detect"