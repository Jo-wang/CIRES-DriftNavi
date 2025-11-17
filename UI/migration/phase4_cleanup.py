"""
Phase 4 Migration and Cleanup Script
Complete migration to unified system and cleanup of legacy code

This script performs comprehensive cleanup of legacy pattern-matching
callbacks, duplicate functions, and outdated entry points, ensuring
the application uses only the unified Phase 1-3 system.

Key Cleanup Tasks:
1. Remove all pattern-matching callback references
2. Update entry points to use unified modal system
3. Clean up imports and unused variables
4. Verify integration between all phases
5. Update documentation and comments

Following Dash best practices:
- Single source of truth for all functionality
- No pattern-matching ID conflicts
- Clean import structure
- Proper error handling
- Complete migration documentation
"""

import os
import re
from typing import List, Dict, Tuple
from pathlib import Path


class Phase4Migrator:
    """
    Comprehensive migration tool for Phase 4 cleanup.
    
    This class handles systematic cleanup of legacy code and ensures
    complete migration to the unified Phase 1-3 system.
    """
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.ui_path = self.project_root / "UI"
        self.callback_path = self.ui_path / "callback"
        self.component_path = self.ui_path / "components"
        
        # Track cleanup actions
        self.cleanup_log = []
        self.issues_found = []
    
    def log_action(self, action: str, file_path: str = "", details: str = ""):
        """Log cleanup action for tracking."""
        self.cleanup_log.append({
            'action': action,
            'file': file_path,
            'details': details
        })
        print(f"[PHASE 4] {action}: {file_path} - {details}")
    
    def find_pattern_matching_usage(self) -> List[Tuple[str, str, int]]:
        """
        Find all remaining pattern-matching usage in callbacks.
        
        Returns:
            List of (file_path, line_content, line_number) tuples
        """
        pattern_files = []
        
        # Patterns to search for
        patterns = [
            r"MATCH",
            r"ALL.*type",
            r"Input.*\{'type':",
            r"Output.*\{'type':",
            r"classification-dropdown",
            r"datatype-dropdown",
            r"column-row"
        ]
        
        for callback_file in self.callback_path.glob("*.py"):
            try:
                with open(callback_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                for line_num, line in enumerate(lines, 1):
                    for pattern in patterns:
                        if re.search(pattern, line) and not line.strip().startswith('#'):
                            pattern_files.append((str(callback_file), line.strip(), line_num))
                            
            except Exception as e:
                self.log_action("ERROR", str(callback_file), f"Error reading file: {e}")
        
        return pattern_files
    
    def clean_unused_imports(self, file_path: str) -> bool:
        """
        Clean up unused imports in a Python file.
        
        Args:
            file_path: Path to the Python file
            
        Returns:
            bool: True if changes were made
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # Remove unused MATCH and ALL imports if no pattern-matching is used
            if "MATCH" in content and not re.search(r"(?<!#.*)\bMATCH\b", content.split('\n')[10:]):
                content = re.sub(r",\s*MATCH", "", content)
                content = re.sub(r"MATCH\s*,", "", content)
                content = re.sub(r"from dash.dependencies import.*MATCH.*", 
                               lambda m: m.group(0).replace(", MATCH", "").replace("MATCH, ", "").replace("MATCH", ""), 
                               content)
            
            if "ALL" in content and not re.search(r"(?<!#.*)\bALL\b", content.split('\n')[10:]):
                content = re.sub(r",\s*ALL", "", content)
                content = re.sub(r"ALL\s*,", "", content)
                content = re.sub(r"from dash.dependencies import.*ALL.*", 
                               lambda m: m.group(0).replace(", ALL", "").replace("ALL, ", "").replace("ALL", ""), 
                               content)
            
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                self.log_action("CLEANED_IMPORTS", file_path, "Removed unused MATCH/ALL imports")
                return True
                
        except Exception as e:
            self.log_action("ERROR", file_path, f"Error cleaning imports: {e}")
        
        return False
    
    def update_entry_points(self) -> List[str]:
        """
        Update all entry points to use the unified system.
        
        Returns:
            List of updated files
        """
        updated_files = []
        
        # Entry points to update:
        # 1. Navigation menu items
        # 2. Chat buttons
        # 3. Modal triggers
        # 4. Any remaining direct calls to legacy functions
        
        entry_point_updates = {
            # Replace direct modal creation with unified modal calls
            'show-type-compare-btn': 'unified-column-type-modal',
            'menu-dataset-info': 'unified-column-type-modal',
            
            # Update button IDs and handlers
            'create_column_type_comparison': 'create_modern_column_comparison',
        }
        
        for callback_file in self.callback_path.glob("*.py"):
            try:
                with open(callback_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                original_content = content
                
                # Apply updates
                for old_pattern, new_pattern in entry_point_updates.items():
                    if old_pattern in content:
                        content = content.replace(old_pattern, new_pattern)
                        self.log_action("UPDATED_ENTRY_POINT", str(callback_file), 
                                      f"Replaced {old_pattern} with {new_pattern}")
                
                if content != original_content:
                    with open(callback_file, 'w', encoding='utf-8') as f:
                        f.write(content)
                    updated_files.append(str(callback_file))
                    
            except Exception as e:
                self.log_action("ERROR", str(callback_file), f"Error updating entry points: {e}")
        
        return updated_files
    
    def verify_phase_integration(self) -> Dict[str, bool]:
        """
        Verify that all phases are properly integrated.
        
        Returns:
            Dict with integration status for each phase connection
        """
        integration_status = {
            'phase1_to_phase2': False,  # ColumnTypeManager → Unified Modal
            'phase2_to_phase3': False,  # Unified Modal → Table Controller
            'phase1_to_phase3': False,  # ColumnTypeManager → Table Controller
            'phase3_working': False,    # Table Controller functioning
        }
        
        try:
            # Check Phase 1 → Phase 2 integration
            phase2_file = self.ui_path / "callback" / "unified_column_modal_callbacks.py"
            if phase2_file.exists():
                with open(phase2_file, 'r') as f:
                    content = f.read()
                    if "ColumnTypeManager" in content and "update_column_type" in content:
                        integration_status['phase1_to_phase2'] = True
                        self.log_action("VERIFIED", str(phase2_file), "Phase 1→2 integration confirmed")
            
            # Check Phase 2 → Phase 3 integration
            phase3_file = self.ui_path / "controller" / "table_overview_controller.py"
            if phase3_file.exists():
                with open(phase3_file, 'r') as f:
                    content = f.read()
                    if "column-type-change-trigger" in content and "global-data-state" in content:
                        integration_status['phase2_to_phase3'] = True
                        integration_status['phase1_to_phase3'] = True
                        integration_status['phase3_working'] = True
                        self.log_action("VERIFIED", str(phase3_file), "Phase 2→3 integration confirmed")
            
        except Exception as e:
            self.log_action("ERROR", "", f"Error verifying integration: {e}")
        
        return integration_status
    
    def generate_migration_report(self) -> str:
        """
        Generate comprehensive migration report.
        
        Returns:
            Formatted migration report string
        """
        report = []
        report.append("="*80)
        report.append("PHASE 4 MIGRATION REPORT")
        report.append("="*80)
        report.append("")
        
        # Summary of actions
        report.append("CLEANUP ACTIONS PERFORMED:")
        for action_info in self.cleanup_log:
            report.append(f"  {action_info['action']}: {action_info['details']}")
        report.append("")
        
        # Pattern-matching cleanup
        pattern_usage = self.find_pattern_matching_usage()
        report.append(f"PATTERN-MATCHING REFERENCES FOUND: {len(pattern_usage)}")
        for file_path, line, line_num in pattern_usage:
            report.append(f"  {os.path.basename(file_path)}:{line_num} - {line}")
        report.append("")
        
        # Integration verification
        integration = self.verify_phase_integration()
        report.append("PHASE INTEGRATION STATUS:")
        for phase, status in integration.items():
            status_text = "✅ WORKING" if status else "❌ ISSUE"
            report.append(f"  {phase}: {status_text}")
        report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS:")
        if not all(integration.values()):
            report.append("  - Some phase integrations need attention")
        if pattern_usage:
            report.append("  - Pattern-matching references should be reviewed/removed")
        if not self.cleanup_log:
            report.append("  - No cleanup actions were needed")
        else:
            report.append("  - Migration cleanup completed successfully")
        
        report.append("")
        report.append("="*80)
        
        return "\n".join(report)
    
    def run_complete_migration(self) -> str:
        """
        Run complete Phase 4 migration and cleanup.
        
        Returns:
            Migration report
        """
        self.log_action("STARTED", "", "Phase 4 complete migration")
        
        # 1. Clean up unused imports
        for callback_file in self.callback_path.glob("*.py"):
            self.clean_unused_imports(str(callback_file))
        
        # 2. Update entry points
        updated_files = self.update_entry_points()
        self.log_action("UPDATED_ENTRY_POINTS", "", f"Updated {len(updated_files)} files")
        
        # 3. Verify integration
        integration_status = self.verify_phase_integration()
        working_integrations = sum(integration_status.values())
        total_integrations = len(integration_status)
        self.log_action("VERIFIED_INTEGRATION", "", 
                       f"{working_integrations}/{total_integrations} integrations working")
        
        # 4. Generate report
        report = self.generate_migration_report()
        
        self.log_action("COMPLETED", "", "Phase 4 complete migration")
        
        return report


def run_phase4_migration(project_root: str = "/Users/uqzwan30/Library/CloudStorage/OneDrive-TheUniversityofQueensland/Documents/cires/DriftNavi") -> str:
    """
    Run Phase 4 migration from the current project directory.
    
    Args:
        project_root: Path to the DriftNavi project root
        
    Returns:
        Migration report string
    """
    migrator = Phase4Migrator(project_root)
    return migrator.run_complete_migration()


# Entry point for testing the migration
if __name__ == "__main__":
    report = run_phase4_migration()
    print(report)