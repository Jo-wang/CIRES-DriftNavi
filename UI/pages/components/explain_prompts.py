"""
Prompt templates for the Explain component in DriftNavi.

This module contains template definitions and utility functions for generating
prompts that are sent to GPT for analyzing distribution shifts between datasets.
It includes templates for general distribution analysis, training impact assessment,
adaptation needs analysis, and conditional distribution analysis.
"""

def generate_train_prompt(target_name, target_type, primary_dist, secondary_dist):
    """
    Generate a prompt for analyzing training risks based on label shift.
    
    Args:
        target_name (str): Name of the target attribute
        target_type (str): Type of the target attribute (categorical, continuous)
        primary_dist (dict): Distribution of the target in primary dataset {label: count}
        secondary_dist (dict): Distribution of the target in secondary dataset {label: count}
        
    Returns:
        str: Formatted prompt for GPT analysis
    """
    # Format type info for display
    type_info = f" ({target_type})" if target_type else ""
    
    # Format distributions as strings
    primary_dist_str = format_distribution(primary_dist)
    secondary_dist_str = format_distribution(secondary_dist)
    
    # Construct the prompt
    prompt = f"""
        The user is training a machine learning model using a primary dataset and wants to assess label shift risks based on the target attribute distribution in both primary and secondary datasets.
        
        Target attribute: {target_name}{type_info}
        Primary distribution: {primary_dist_str}
        Secondary distribution: {secondary_dist_str}
        
        Does this label shift indicate training risk, such as overfitting to majority labels? Should the user consider label balancing or resampling strategies?
        """
    return prompt.strip()


def generate_adapt_prompt(target_name, target_type, primary_dist, secondary_dist):
    """
    Generate a prompt for analyzing adaptation needs for models between datasets.
    
    Args:
        target_name (str): Name of the target attribute
        target_type (str): Type of the target attribute (categorical, continuous)
        primary_dist (dict): Distribution of the target in primary dataset {label: count}
        secondary_dist (dict): Distribution of the target in secondary dataset {label: count}
        
    Returns:
        str: Formatted prompt for GPT analysis
    """
    # Format type info for display
    type_info = f" ({target_type})" if target_type else ""
    
    # Format distributions as strings
    primary_dist_str = format_distribution(primary_dist)
    secondary_dist_str = format_distribution(secondary_dist)
    
    # Construct the prompt
    prompt = f"""
        The user wants to apply a model trained on the primary dataset to the secondary dataset.

        Target attribute: {target_name}{type_info}
        Primary distribution: {primary_dist_str}
        Secondary distribution: {secondary_dist_str}

        Should adaptation methods be considered? Is direct inference risky? Recommend strategies if needed (e.g., reweighting, pseudo-labeling, domain adaptation).
        """
    return prompt.strip()


def generate_conditional_prompt(target_name, target_value, shifted_attr_name, shifted_attr_type, primary_dist, secondary_dist):
    """
    Generate a prompt for analyzing conditional distribution shift between datasets.
    
    This analyzes how a specific target attribute value affects the distribution of another attribute,
    comparing this relationship between primary and secondary datasets.
    
    Args:
        target_name (str): Name of the target attribute
        target_value (str): Value or range of the target attribute being conditioned on
        shifted_attr_name (str): Name of the attribute showing distribution shift
        shifted_attr_type (str): Type of the shifted attribute (categorical, continuous)
        primary_dist (dict): Conditional distribution in primary dataset {value: count}
        secondary_dist (dict): Conditional distribution in secondary dataset {value: count}
        
    Returns:
        str: Formatted prompt for GPT analysis
    """
    # Format type info for display
    type_info = f" ({shifted_attr_type})" if shifted_attr_type else ""
    
    # Format distributions as strings
    primary_dist_str = format_distribution(primary_dist)
    secondary_dist_str = format_distribution(secondary_dist)
    
    # Construct the prompt
    prompt = f"""
        The user is analyzing how the distribution of one attribute shifts between datasets when conditioned on a specific value of another attribute.

        Target attribute: {target_name}
        Target value: {target_value}
        Shifted attribute: {shifted_attr_name}{type_info}

        Primary dataset distribution (when {target_name} = {target_value}):
        {primary_dist_str}

        Secondary dataset distribution (when {target_name} = {target_value}):
        {secondary_dist_str}

        Please analyze this conditional distribution shift and explain:
        1. What patterns or relationships exist between {target_name} = {target_value} and {shifted_attr_name}?
        2. How does this relationship differ between primary and secondary datasets?
        3. What might explain these differences and what are their implications?
        4. How might this conditional shift affect model performance or decision-making?
        """
    return prompt.strip()


def generate_adaptation_strategy_prompt(strategy, target_name, target_type, primary_dist, secondary_dist):
    """
    Generate a prompt for adaptation strategy analysis based on selected strategy.
    
    Args:
        strategy (str): Selected adaptation strategy ('retrain' or 'finetune')
        target_name (str): Name of the target attribute
        target_type (str): Type of the target attribute (categorical, continuous)
        primary_dist (dict): Distribution of the target in primary dataset {label: count}
        secondary_dist (dict): Distribution of the target in secondary dataset {label: count}
        
    Returns:
        str: Formatted prompt for GPT analysis
    """
    # Format type info for display
    type_info = f" ({target_type})" if target_type else ""
    
    # Format distributions as strings
    primary_dist_str = format_distribution(primary_dist)
    secondary_dist_str = format_distribution(secondary_dist)
    
    if strategy == "retrain":
        strategy_description = "complete model retraining"
        strategy_approach = "merging both datasets and rebuilding the model from scratch"
        strategy_considerations = [
            "Data merging and balancing strategies",
            "Feature engineering for combined dataset",
            "Model architecture considerations",
            "Performance validation on new data"
        ]
    elif strategy == "finetune":
        strategy_description = "incremental model fine-tuning"
        strategy_approach = "adapting the existing model using domain adaptation techniques"
        strategy_considerations = [
            "Domain adaptation methods",
            "Incremental learning approaches",
            "Model stability preservation",
            "Gradual adaptation techniques"
        ]
    else:
        strategy_description = "model adaptation"
        strategy_approach = "adapting the model to handle distribution shifts"
        strategy_considerations = [
            "Distribution alignment techniques",
            "Model adaptation strategies",
            "Performance monitoring",
            "Implementation considerations"
        ]
    
    # Construct the prompt
    prompt = f"""
        The user has selected a {strategy} strategy for adapting their model to handle distribution shifts between datasets.

        Target attribute: {target_name}{type_info}
        Primary dataset distribution: {primary_dist_str}
        Secondary dataset distribution: {secondary_dist_str}

        Selected Strategy: {strategy.upper()}
        Strategy Description: {strategy_description}
        Approach: {strategy_approach}

        Please provide a comprehensive analysis for this adaptation approach, including:

        1. **Data Preparation Strategy**: How should the data be prepared for {strategy}?
        2. **Distribution Analysis**: What specific distribution patterns should be addressed?
        3. **Implementation Steps**: What are the key steps for implementing this strategy?
        4. **Expected Outcomes**: What results can be expected from this approach?
        5. **Risk Assessment**: What are the potential challenges and how to mitigate them?
        6. **Monitoring Plan**: How should the adaptation process be monitored?

        Focus on practical, actionable recommendations that the user can implement immediately.
        """
    return prompt.strip()


def generate_adaptation_data_preparation_prompt(strategy, target_name, primary_df_info, secondary_df_info):
    """
    Generate a prompt for data preparation guidance based on adaptation strategy.
    
    Args:
        strategy (str): Selected adaptation strategy ('retrain' or 'finetune')
        target_name (str): Name of the target attribute
        primary_df_info (dict): Information about primary dataset
        secondary_df_info (dict): Information about secondary dataset
        
    Returns:
        str: Formatted prompt for data preparation guidance
    """
    if strategy == "retrain":
        focus_areas = [
            "Complete dataset merging and integration",
            "Data quality assessment and cleaning",
            "Feature engineering for combined dataset",
            "Balanced sampling and stratification"
        ]
    elif strategy == "finetune":
        focus_areas = [
            "Domain adaptation techniques",
            "Incremental data integration",
            "Distribution alignment methods",
            "Preservation of original data structure"
        ]
    else:
        focus_areas = [
            "Data preparation for adaptation",
            "Distribution analysis and alignment",
            "Quality assessment and validation"
        ]
    
    prompt = f"""
        Based on the selected {strategy} strategy, provide detailed data preparation guidance for adapting the model.

        Target Attribute: {target_name}
        Primary Dataset: {primary_df_info.get('shape', 'Unknown shape')} rows
        Secondary Dataset: {secondary_df_info.get('shape', 'Unknown shape')} rows

        Focus Areas for {strategy.upper()} Strategy:
        {chr(10).join([f"- {area}" for area in focus_areas])}

        Please provide:

        1. **Data Integration Plan**: How to combine the datasets effectively
        2. **Quality Checks**: What data quality issues to look for and fix
        3. **Feature Engineering**: What new features or transformations are needed
        4. **Sampling Strategy**: How to ensure balanced and representative data
        5. **Validation Approach**: How to validate the prepared data
        6. **Implementation Steps**: Step-by-step data preparation process

        Include specific techniques, tools, and code examples where applicable.
        """
    return prompt.strip()


def format_distribution(distribution):
    """
    Format a distribution dictionary into a readable string.
    
    Args:
        distribution (dict): Distribution as {label: count}
        
    Returns:
        str: Formatted string representation
    """
    if not distribution:
        return "No data available"
    
    # Format the distribution as a string
    items = [f"{key}: {value}" for key, value in distribution.items()]
    return "{" + ", ".join(items) + "}"
