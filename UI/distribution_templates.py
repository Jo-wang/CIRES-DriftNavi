"""
Distribution comparison templates for driftNavi.
This module provides template strings and functions to format distribution comparison 
analysis responses for the chatbot.
"""

def get_distribution_analysis_template(attribute_name, has_data_science_background=False):
    """
    Generate a template for GPT to analyze distribution comparisons.
    
    Args:
        attribute_name: The name of the attribute being compared
        has_data_science_background: Whether the user has a data science background
    
    Returns:
        A formatted template string
    """
    # Base template with core structure
    template = f"""
        # Analysis: Distribution Shift in {attribute_name}
        
        I've noticed you're analyzing the distribution differences for the **{attribute_name}** attribute. Here's an in-depth analysis of this comparison:
        
        ## Distribution Difference Overview
        
        Based on the distribution comparison data you've provided, I can see the following differences in `{attribute_name}` between the two datasets:
        - [Analysis of main differences between the two dataset distributions]
        - [Highlight the most notable statistical property changes, such as mean, variance, distribution shape, etc.]
        
        ## Meaning and Importance of the {attribute_name} Attribute
        
        `{attribute_name}` likely represents:
        - [Explanation of what this attribute might represent in the domain]
        - [Discussion of its importance and potential relationship with other attributes]
        
        ## Potential Impact of Distribution Shift
        """

    # Technical section - adjust based on user background
    if has_data_science_background:
        template += """
            ### Technical Analysis
            From a machine learning and statistical perspective, this distribution shift may lead to:
            - Decreased model performance, manifested as reduced [specific metrics]
            - Problems specific to the shift type, such as covariate shift or concept drift
            - Potential [specific problem types] in deployment environments

            ### Recommended Mitigation Strategies
            Consider the following approaches to mitigate this distribution shift:
            1. **Domain Adaptation Techniques**: Such as [specific technique names]
            2. **Feature Engineering**: [specific recommendations]
            3. **Model Retraining Strategies**: [specific strategies]
            """
    else:
        template += """
        ### Understanding the Impact of This Change
        This change in distribution means:
        - The model may not perform as well on new data as it did on old data
        - Some predictions may no longer be accurate
        - The model might produce drifted or unfair results for certain cases

        ### Possible Solutions
        Here are some solutions to consider:
        1. **Update Data**: Use more recent, representative data
        2. **Adjust Models**: Make the model more adaptable to new situations
        3. **Monitor Performance**: Regularly check how the model performs on different data
        """

    # Common recommendations section
    template += """
        ## Further Analysis Recommendations

        For a more comprehensive understanding of this distribution shift:
        1. Consider analyzing in conjunction with these attributes: [suggested related attributes]
        2. Try these statistical tests: [suggested testing methods]
        3. Visualization recommendations: [suggested visualization methods]

        I can help you explore these directions in depth, or answer any specific questions you have about this distribution shift.
        """
    return template

def create_distribution_analysis_message(summary_text, column_name):
    """
    Create a system message for distribution analysis that guides the agent's response.
    
    Args:
        summary_text: Summary of the distribution comparison
        column_name: The name of the attribute/column being compared
    
    Returns:
        Formatted system message for the agent
    """
    system_message = f"""
        The user has just added a distribution comparison analysis for the attribute: {column_name}

        Comparison Summary:
        {summary_text}

        As an AI assistant, you should:
        1. Analyze the distribution differences of this attribute between the two datasets
        2. Explain the potential impact of this distribution shift on model performance
        3. Suggest possible mitigation strategies and directions for further analysis
        4. Adjust the technical depth of your response based on whether the user has a data science background

        Please respond in a structured format, ensuring you provide both insights and practical next step recommendations.
        """
    return system_message
