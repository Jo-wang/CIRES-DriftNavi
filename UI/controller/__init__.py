"""
UI Controllers Package - Phase 3

This package contains centralized controllers for managing UI state and data flow.
All controllers follow the single responsibility principle and provide centralized
control over their respective domains.

Controllers:
- table_overview_controller: Master controller for all table-overview updates
"""

# Import all controllers for easy access
from . import table_overview_controller

__all__ = ['table_overview_controller']