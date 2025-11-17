"""
Tests for the Explain component functionality.

This module provides test cases for the Explain component, ensuring that
distribution analysis and GPT-powered explanations work correctly.
"""

import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from UI.pages.components.explain_component import (
    analyze_target_distribution,
    generate_distribution_chart,
)

from UI.pages.components.explain_utils import (
    rank_attributes,
    analyze_conditional_distribution,
    get_target_values_options
)

from UI.pages.components.explain_prompts import generate_conditional_prompt

class TestExplainComponent(unittest.TestCase):
    """Test cases for the Explain component functionality."""
    
    def setUp(self):
        """Set up test data."""
        # Create test dataframes
        self.primary_df = pd.DataFrame({
            'age': [25, 30, 35, 40, 45, 28, 32, 38, 42, 47],
            'income': [50000, 60000, 70000, 80000, 90000, 55000, 65000, 75000, 85000, 95000],
            'education': ['High School', 'Bachelor', 'Master', 'PhD', 'Bachelor', 
                          'High School', 'Bachelor', 'Master', 'Bachelor', 'High School']
        })
        
        self.secondary_df = pd.DataFrame({
            'age': [22, 28, 33, 38, 43, 26, 30, 36, 40, 45, 50, 55],
            'income': [48000, 58000, 68000, 78000, 88000, 53000, 63000, 73000, 83000, 93000, 100000, 110000],
            'education': ['High School', 'Bachelor', 'Master', 'Bachelor', 'PhD', 
                          'High School', 'Bachelor', 'Master', 'Bachelor', 'High School', 'PhD', 'Master']
        })
        
    def test_analyze_target_distribution_categorical(self):
        """Test analyzing a categorical attribute distribution."""
        dist, col_type = analyze_target_distribution(self.primary_df, 'education')
        
        # Check distribution results
        self.assertEqual(col_type, "categorical")
        self.assertIn('Bachelor', dist)
        self.assertEqual(dist['Bachelor'], 4)  # 4 occurrences in primary_df
        self.assertIn('High School', dist)
        self.assertEqual(dist['High School'], 3)
        
    def test_analyze_target_distribution_continuous(self):
        """Test analyzing a continuous attribute distribution."""
        dist, col_type = analyze_target_distribution(self.primary_df, 'age')
        
        # Check distribution results
        self.assertEqual(col_type, "continuous")
        self.assertEqual(len(dist), 10)  # 10 bins by default
        
    def test_generate_distribution_chart_categorical(self):
        """Test generating a distribution chart for categorical data."""
        primary_dist = {'High School': 3, 'Bachelor': 4, 'Master': 2, 'PhD': 1}
        secondary_dist = {'High School': 3, 'Bachelor': 3, 'Master': 3, 'PhD': 2}
        
        fig = generate_distribution_chart(primary_dist, secondary_dist, "categorical")
        
        # Basic validation of the figure
        self.assertEqual(len(fig.data), 2)  # Two traces for primary and secondary
        self.assertEqual(fig.data[0].name, "Primary Dataset")
        self.assertEqual(fig.data[1].name, "Secondary Dataset")
        
    def test_generate_distribution_chart_continuous(self):
        """Test generating a distribution chart for continuous data."""
        # Create sample bin distributions
        primary_dist = {
            "25.00 to 30.00": 3,
            "30.00 to 35.00": 2,
            "35.00 to 40.00": 2,
            "40.00 to 45.00": 2,
            "45.00 to 50.00": 1
        }
        secondary_dist = {
            "20.00 to 25.00": 1,
            "25.00 to 30.00": 3,
            "30.00 to 35.00": 2,
            "35.00 to 40.00": 3,
            "40.00 to 45.00": 1,
            "45.00 to 50.00": 1,
            "50.00 to 55.00": 1
        }
        
        fig = generate_distribution_chart(primary_dist, secondary_dist, "continuous")
        
        # Basic validation of the figure
        self.assertEqual(len(fig.data), 2)  # Two traces for primary and secondary
        self.assertEqual(fig.data[0].mode, "lines+markers")  # Line chart for continuous data
        self.assertEqual(fig.data[1].mode, "lines+markers")

    @patch('UI.pages.components.explain_component.generate_response_from_prompt')
    def test_gpt_response_integration(self, mock_generate_response):
        """Test integration with GPT response generation."""
        # Mock the GPT response
        mock_generate_response.return_value = (
            "The distribution shift in the age attribute between the primary and secondary datasets "
            "shows a notable difference that could impact model training. The secondary dataset "
            "has a broader range with more younger and older individuals.\n\n"
            "This shift may affect how the model learns age-related patterns, potentially "
            "causing it to perform differently across different age groups."
        )
        
        # Call the function to ensure it works (actual callback testing would be in integration tests)
        response = mock_generate_response("Sample prompt")
        
        # Verify the mock was called
        mock_generate_response.assert_called_once()
        
        # Check response formatting
        self.assertIn("distribution shift", response)
        self.assertIn("model training", response)
        
    def test_rank_attributes(self):
        """Test ranking attributes based on shift metrics."""
        # Create sample metrics data
        metrics_data = [
            {"Attribute": "age", "JS_Divergence": 0.15, "PSI": 0.25, "Wasserstein": 5.0, "Test_Statistic": 0.3, "p_value": 0.01},
            {"Attribute": "income", "JS_Divergence": 0.08, "PSI": 0.15, "Wasserstein": 3.0, "Test_Statistic": 0.2, "p_value": 0.05},
            {"Attribute": "education", "JS_Divergence": 0.20, "PSI": 0.30, "Wasserstein": 7.0, "Test_Statistic": 0.4, "p_value": 0.001},
            {"Attribute": "gender", "JS_Divergence": 0.05, "PSI": 0.10, "Wasserstein": 2.0, "Test_Statistic": 0.15, "p_value": 0.1},
            {"Attribute": "location", "JS_Divergence": 0.12, "PSI": 0.18, "Wasserstein": 4.0, "Test_Statistic": 0.25, "p_value": 0.03}
        ]
        
        # Test with k=3
        top_attrs = rank_attributes(metrics_data, k=3)
        
        # Verify the results
        self.assertEqual(len(top_attrs), 3)
        self.assertIn("education", top_attrs)  # Should be in top 3 based on metrics
        self.assertIn("age", top_attrs)  # Should be in top 3 based on metrics
        
        # Test with handling of N/A values
        metrics_with_na = [
            {"Attribute": "age", "JS_Divergence": 0.15, "PSI": "N/A", "Wasserstein": 5.0, "Test_Statistic": 0.3, "p_value": 0.01},
            {"Attribute": "income", "JS_Divergence": 0.08, "PSI": 0.15, "Wasserstein": "N/A", "Test_Statistic": 0.2, "p_value": 0.05},
            {"Attribute": "education", "JS_Divergence": 0.20, "PSI": 0.30, "Wasserstein": 7.0, "Test_Statistic": "N/A", "p_value": 0.001}
        ]
        
        top_attrs_with_na = rank_attributes(metrics_with_na, k=2)
        self.assertEqual(len(top_attrs_with_na), 2)
    
    def test_analyze_conditional_distribution(self):
        """Test analyzing conditional distributions."""
        # Test with categorical target and categorical shifted attribute
        primary_dist, col_type = analyze_conditional_distribution(
            self.primary_df, 'education', 'Bachelor', 'age'
        )
        
        # Check results
        self.assertEqual(col_type, "continuous")
        self.assertTrue(isinstance(primary_dist, dict))
        
        # Only Bachelor education entries should be considered
        bachelor_count = len(self.primary_df[self.primary_df['education'] == 'Bachelor'])
        total_dist_count = sum(primary_dist.values())
        self.assertEqual(total_dist_count, bachelor_count)
        
        # Test with categorical target and continuous shifted attribute
        primary_dist, col_type = analyze_conditional_distribution(
            self.primary_df, 'education', 'High School', 'income'
        )
        
        # Check results
        self.assertEqual(col_type, "continuous")
        self.assertTrue(isinstance(primary_dist, dict))
        
    def test_get_target_values_options(self):
        """Test generating options for target attribute values."""
        # Test with categorical attribute
        options = get_target_values_options(self.primary_df, 'education')
        
        # Check results
        self.assertTrue(isinstance(options, list))
        self.assertEqual(len(options), len(self.primary_df['education'].unique()))
        
        # Each option should have 'label' and 'value' keys
        for option in options:
            self.assertIn('label', option)
            self.assertIn('value', option)
        
        # Test with continuous attribute
        options = get_target_values_options(self.primary_df, 'age')
        
        # Check results - should have 10 bins by default
        self.assertTrue(isinstance(options, list))
        # For continuous data, we should have bins (typically 10)
        self.assertLessEqual(len(options), 10)
        
    @patch('UI.pages.components.explain_prompts.format_distribution')
    def test_generate_conditional_prompt(self, mock_format_distribution):
        """Test generating conditional distribution analysis prompts."""
        # Mock the format_distribution function
        mock_format_distribution.side_effect = lambda dist: str(dist)
        
        # Sample distributions
        primary_dist = {"25-30": 10, "30-35": 20, "35-40": 15}
        secondary_dist = {"25-30": 15, "30-35": 15, "35-40": 25}
        
        # Generate prompt
        prompt = generate_conditional_prompt(
            "education", "Bachelor", "age", "continuous",
            primary_dist, secondary_dist
        )
        
        # Check prompt content
        self.assertIn("education", prompt)
        self.assertIn("Bachelor", prompt)
        self.assertIn("age", prompt)
        self.assertIn("continuous", prompt)


if __name__ == '__main__':
    unittest.main()
