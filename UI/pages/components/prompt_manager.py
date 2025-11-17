"""
Unified Prompt Manager for DriftNavi

This module provides a centralized, modular prompt management system
for GPT analysis across different components of the DriftNavi application.

Author: DriftNavi Team
Created: 2025
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod


# =============================================================================
# PROMPT TEMPLATE COMPONENTS
# =============================================================================

@dataclass
class UserContext:
    """User personalization context for prompt generation."""
    industry_sector: str = "Technology"
    professional_role: str = "Data Scientist"
    expertise_level: str = "Intermediate"
    domain_context: str = "Machine Learning Operations"


@dataclass
class DatasetContext:
    """Dataset information context for prompt generation."""
    primary_dataset: Dict[str, Any]
    secondary_dataset: Dict[str, Any]
    comparison_context: Dict[str, Any]
    
    def get_summary(self) -> str:
        """Get formatted dataset summary."""
        return f"""
**DATASET CONTEXT:**
Primary Dataset: {self.primary_dataset.get('dataset_name', 'Unknown')}
- Records: {self.primary_dataset.get('record_count', 0):,}
- Columns: {self.primary_dataset.get('column_count', 0)}
- Missing Data: {self.primary_dataset.get('missing_data_percentage', 0)}%

Secondary Dataset: {self.secondary_dataset.get('dataset_name', 'Unknown')}
- Records: {self.secondary_dataset.get('record_count', 0):,}
- Columns: {self.secondary_dataset.get('column_count', 0)}
- Missing Data: {self.secondary_dataset.get('missing_data_percentage', 0)}%

Common Columns: {len(self.comparison_context.get('common_columns', []))}
Target Attribute: {self.comparison_context.get('target_attribute', 'Not specified')}
"""


class PromptComponent(ABC):
    """Abstract base class for prompt components."""
    
    @abstractmethod
    def generate(self, context: Dict[str, Any]) -> str:
        """Generate prompt component content."""
        pass


class UserContextComponent(PromptComponent):
    """User personalization component for prompts."""
    
    def generate(self, context: Dict[str, Any]) -> str:
        user_ctx = context.get('user_context', UserContext())
        
        return f"""
                **USER CONTEXT:**
                Industry: {user_ctx.industry_sector}
                Role: {user_ctx.professional_role}
                Expertise Level: {user_ctx.expertise_level}
                Domain: {user_ctx.domain_context}

                Please tailor your analysis to this user's background and expertise level.
                """


class DatasetContextComponent(PromptComponent):
    """Dataset information component for prompts."""
    
    def generate(self, context: Dict[str, Any]) -> str:
        dataset_ctx = context.get('dataset_context')
        if not dataset_ctx:
            return "**DATASET CONTEXT:** Not available"
        
        return dataset_ctx.get_summary()


class ContextItemsComponent(PromptComponent):
    """Context items analysis component for prompts."""
    
    def generate(self, context: Dict[str, Any]) -> str:
        context_items = context.get('context_items', [])
        
        if not context_items:
            return "**DETECTED ISSUES:** No issues to analyze"
        
        prompt_section = "**DETECTED ISSUES TO ANALYZE:**\n"
        prompt_section += "Each item below represents a DIFFERENT type of issue. NEVER confuse conditional distributions with drift analysis.\n\n"
        
        for i, item in enumerate(context_items):
            context_type = item.get('type', 'unknown')
            prompt_section += f"\n{i}. **{context_type.replace('_', ' ').title()}** (context_type: \"{context_type}\")\n"
            
            if context_type == "drift_analysis":
                attr_name = item.get('attribute_name', 'Unknown')
                prompt_section += f"   - Type: Single attribute drift detection\n"
                prompt_section += f"   - Attribute: {attr_name}\n"
                prompt_section += f"   - Expected Title Format: 'Drift Analysis: {attr_name}'\n"
                prompt_section += f"   - Details: {item.get('metric_details', 'No details')}\n"
                prompt_section += f"   - Interpretation: {item.get('interpretation', 'No interpretation')}\n"
                
            elif context_type == "distribution_comparison":
                prompt_section += f"   - Type: Distribution comparison between datasets\n"
                prompt_section += f"   - Summary: {item.get('summary_text', 'No summary')}\n"
                prompt_section += f"   - Cell Info: {item.get('cell_info', 'No cell info')[:200]}...\n"
                prompt_section += f"   - Expected Title Format: 'Distribution Comparison: [ATTRIBUTE_NAME]' (extract attribute name from cell_info or summary)\n"
                
            elif context_type == "conditional_distribution":
                target_attr = item.get('target_attribute', 'Unknown')
                target_val = item.get('target_value', 'Unknown')
                compare_attr = item.get('compare_attribute', 'Unknown')
                prompt_section += f"   - Type: CONDITIONAL ANALYSIS - how {compare_attr} behaves when {target_attr}={target_val}\n"
                prompt_section += f"   - Target Condition: {target_attr} = {target_val}\n"
                prompt_section += f"   - Analyzed Attribute: {compare_attr}\n"
                prompt_section += f"   - Expected Title Format: 'Conditional Analysis: {target_attr} = {target_val}'\n"
                prompt_section += f"   - Summary: {item.get('summary_text', 'No summary')}\n"
                prompt_section += f"   - IMPORTANT: This is NOT a drift analysis! This analyzes {compare_attr} conditioned on {target_attr}={target_val}\n"
                
            elif context_type == "metric":
                metric_name = item.get('metric_name', 'Unknown')
                prompt_section += f"   - Type: Statistical metric analysis\n"
                prompt_section += f"   - Metric: {metric_name}\n"
                prompt_section += f"   - Expected Title Format: 'Metric Analysis: {metric_name}'\n"
                prompt_section += f"   - Attribute: {item.get('attribute_name', 'Unknown')}\n"
                prompt_section += f"   - Details: {item.get('metric_details', 'No details')}\n"
                
            elif context_type == "target_distribution":
                target_attr = item.get('target_attribute', 'Unknown')
                prompt_section += f"   - Type: Target attribute distribution analysis\n"
                prompt_section += f"   - Target Attribute: {target_attr}\n"
                prompt_section += f"   - Expected Title Format: 'Target Distribution Analysis: {target_attr}'\n"
                prompt_section += f"   - Summary: {item.get('summary_text', 'No summary')}\n"
        
        return prompt_section


# Legacy TaskInstructionComponent removed - now using ComprehensiveAnalysisInstructionComponent


# =============================================================================
# PROMPT TEMPLATES
# =============================================================================

class PromptTemplate:
    """Base class for prompt templates."""
    
    def __init__(self, components: List[PromptComponent]):
        self.components = components
    
    def generate(self, context: Dict[str, Any]) -> str:
        """Generate full prompt from components."""
        sections = []
        
        for component in self.components:
            section = component.generate(context)
            if section.strip():
                sections.append(section.strip())
        
        return "\n\n".join(sections)


class ComprehensiveAnalysisTemplate(PromptTemplate):
    """Enhanced template for comprehensive four-layer analysis."""
    
    def __init__(self):
        super().__init__([
            ComprehensiveSystemMessageComponent(),
            UserContextComponent(),
            DatasetContextComponent(),
            ContextItemsComponent(),
            ComprehensiveAnalysisInstructionComponent()
        ])


class ComprehensiveSystemMessageComponent(PromptComponent):
    """System message for comprehensive analysis."""
    
    def generate(self, context: Dict[str, Any]) -> str:
        return """You are an expert ML engineer and strategic advisor specializing in drift detection, drift analysis, and model maintenance strategies. You provide comprehensive, structured analysis that combines technical assessment with strategic recommendations."""


class ComprehensiveAnalysisInstructionComponent(PromptComponent):
    """Comprehensive four-layer analysis instructions."""
    
    def generate(self, context: Dict[str, Any]) -> str:
        user_ctx = context.get('user_context', UserContext())
        context_items = context.get('context_items', [])
        
        return f"""
                **YOUR TASK:**
                As an expert analyzing for a {user_ctx.professional_role} in {user_ctx.industry_sector}, 
                provide a comprehensive four-layer analysis covering severity assessment, individual context analysis, 
                joint analysis, and strategic recommendations.

                **RESPONSE FORMAT:**
                Please respond with a valid JSON object. Use simple string values and avoid complex nested structures to prevent parsing errors:

{{
    "layer1_severity_statistics": {{
        "high_count": <number>,
        "medium_count": <number>, 
        "low_count": <number>,
        "overall_risk_level": "<Low|Medium|High|Critical>",
        "confidence_score": <0.0-1.0>,
        "summary": "<brief overall assessment in 1-2 sentences>"
    }},
    "layer2_context_analysis": [
        {{
            "context_id": <0-based_item_index>,
            "context_type": "<use_internal_key_format_not_display_name>",
            "severity_score": <0-100>,
            "risk_level": "<Low|Medium|High>",
            "title": "<concise title including SPECIFIC attribute name from context item>",
            "explanation": {{
                "beginner": "<simple explanation referencing SPECIFIC attribute/metric from this context item>",
                "intermediate": "<detailed explanation with SPECIFIC metric values from this context item>", 
                "advanced": "<technical explanation using ACTUAL attribute names and values from this context item>"
            }},
            "business_impact": "<business implications specific to THIS attribute/metric, not generic>",
            "technical_details": "<technical analysis with ACTUAL metric names and values from this specific context item>",
            "action_required": "<specific actions needed for THIS particular attribute/issue>",
            "detailed_analysis": "<comprehensive domain-specific analysis (200-400 words) covering deployment risks, failure modes, target attribute reliability impacts, and real-world scenarios specific to THIS context item in the user's {user_ctx.industry_sector} domain - include actual metric values and attribute names from this specific context item>"
        }}
        // GENERATE ONE ENTRY FOR EACH CONTEXT ITEM WITH DIFFERENT CONTENT
    ],
    "layer3_joint_analysis": {{
        "overall_assessment": "<systematic assessment of primary-to-secondary inference risks>"
    }},
    "layer4_strategy_selection": {{
        "recommended_strategy": "<monitor|retrain|finetune>",
        "strategy_overview": "<concise description of the recommended strategy approach>",
        "confidence": <0.0-1.0>,
        "reasoning": "<detailed reasoning based on Layer 3 analysis>"
    }}
}}

    **ANALYSIS GUIDELINES:**

    **CRITICAL - CONTEXT_TYPE FORMAT:** For the context_type field in layer2_context_analysis, you MUST use the exact internal key format, NOT display names:
    - Use "drift_analysis" (NOT "Drift Analysis") - for single attribute drift detection
    - Use "distribution_comparison" (NOT "Distribution Comparison") - for distribution comparisons between datasets
    - Use "conditional_distribution" (NOT "Conditional Distribution") - for conditional analysis (how one attribute behaves when another equals a specific value)
    - Use "target_distribution" (NOT "Target Distribution") - for target attribute distribution analysis
    - Use "metric" (NOT "Metric Analysis") - for statistical metric analysis
    The context_type must match the original detected issue type exactly with underscores and lowercase.

    **CRITICAL - DO NOT CONFUSE ANALYSIS TYPES:**
    - "drift_analysis": Single attribute drift detection (e.g., "age" attribute showing drift)
    - "conditional_distribution": Conditional relationship analysis (e.g., how "income" behaves when "education"="college")
    - "distribution_comparison": Dataset distribution comparison (e.g., comparing overall distributions)
    - NEVER label a conditional_distribution as drift_analysis or vice versa!

    **CRITICAL - CONTEXT_ID MAPPING:** Each context_id MUST correspond to the exact same item index from the **DETECTED ISSUES TO ANALYZE** section:
    - context_id 0 → item "0. **Type**" 
    - context_id 1 → item "1. **Type**"
    - context_id 2 → item "2. **Type**"
    - etc.
    The context_type field MUST exactly match the type shown in parentheses for each item (e.g., if item shows "(context_type: "conditional_distribution")" then use "conditional_distribution").

    **Layer 1:** Classify each issue as High (≥70), Medium (40-69), or Low (<40). Assess overall risk level.

    **Layer 2:** CRITICAL - You MUST analyze each of the {len(context_items)} context items individually and generate UNIQUE, SPECIFIC analysis for each item. DO NOT provide generic analysis:

    - Each context item has different attributes, metrics, and characteristics shown in the **DETECTED ISSUES TO ANALYZE** section above
    - For context_id 0: Focus specifically on the first item (0. **Type**) - reference its specific attribute name, metric values, and details
    - For context_id 1: Focus specifically on the second item (1. **Type**) - reference its specific attribute name, metric values, and details
    - Continue for all {len(context_items)} items, ensuring each analysis is UNIQUE and addresses that specific item's content

    **CRITICAL - TITLE FORMAT REQUIREMENTS:** The title field MUST follow these EXACT formats based on context type:
    - For "drift_analysis": "Drift Analysis: [SPECIFIC_ATTRIBUTE_NAME]" (use attribute_name field)
    - For "distribution_comparison": "Distribution Comparison: [SPECIFIC_ATTRIBUTE_NAME]" (extract from cell_info or summary)
    - For "conditional_distribution": "Conditional Analysis: [TARGET_ATTRIBUTE] = [TARGET_VALUE]" (use target_attribute and target_value fields, NOT compare_attribute)
    - For "target_distribution": "Target Distribution Analysis: [TARGET_ATTRIBUTE]" (use target_attribute field)
    - For "metric": "Metric Analysis: [METRIC_NAME]" (use metric_name field)

    - The technical_details MUST include specific metric values, attribute names, and details from that context item
    - The business_impact MUST be tailored to what that specific context item indicates
    - DO NOT use generic phrases like 'detected issue' - use the actual attribute names and specific values from each context item
    - Each layer2_context_analysis item must be clearly distinguishable and reference different attributes/metrics

    **CRITICAL - DETAILED_ANALYSIS FIELD:** For each context item, the detailed_analysis field must contain comprehensive domain-specific analysis (200-400 words) covering:
    - Domain-specific impact in {user_ctx.industry_sector} applications and target attribute predictions
    - Model deployment risks when primary-trained model is applied to secondary dataset  
    - Target attribute reliability implications (false positives/negatives, calibration issues)
    - Technical consequences during inference (distribution mismatch, performance degradation)
    - Real-world scenarios specific to {user_ctx.industry_sector} where this issue causes business problems
    - Reference ACTUAL metric values and attribute names from the specific context item
    - Write at {user_ctx.expertise_level} level without numbered sections - use flowing paragraphs
    - Keep between 200-400 words with concrete examples and actionable insights

    **Layer 3:** Provide comprehensive systematic assessment (400-800 characters) of primary-to-secondary inference risks. This is the CORE analysis combining user context, dataset metadata, and all detected issues. Include:
    - Detailed assessment of how ALL detected issues compound when deploying primary-trained model to secondary dataset
    - Specific risks to target attribute prediction reliability in the user's {user_ctx.industry_sector} domain
    - Technical failure modes when model encounters secondary data patterns without adaptation
    - Domain-specific implications considering user's {user_ctx.professional_role} perspective
    - Cumulative impact assessment across all {len(context_items)} detected issues

    **Layer 4:** Choose strategy based EXCLUSIVELY on Layer 3 analysis and overall risk assessment. DO NOT use any pre-existing drift or strategy preference:
    - **MONITOR:** Issues are manageable, model performance acceptable with enhanced monitoring and validation systems
    - **RETRAIN:** Significant risks identified requiring model rebuild for reliable deployment to secondary dataset
    - **FINETUNE:** Specific drift patterns detected where targeted adaptation can effectively address identified risks

    CRITICAL: Your strategy recommendation must be data-driven based on the severity and nature of detected issues, NOT any predetermined preference. Consider:
    - Risk severity: High issues → RETRAIN, Medium → FINETUNE, Low → MONITOR
    - Issue types: Distribution shifts → FINETUNE, Data quality problems → RETRAIN, Minor deviations → MONITOR
    - Deployment context: High-stakes applications may require lower risk tolerance

    Provide strategy_overview as a concise 1-2 sentence description explaining the approach and when it's suitable.
    Provide detailed reasoning connecting to specific Layer 3 findings and risk assessment.

    **CRITICAL REQUIREMENTS:**
    - Respond ONLY with valid JSON
    - Keep layer1 and layer4 fields concise (max 200 characters each)
    - Layer3 overall_assessment should be comprehensive (400-800 characters) - this is the core analysis
    - Use simple language
    - ABSOLUTELY NO DOUBLE QUOTES within any string values - use single quotes, parentheses, or simple words instead
    - For metrics, use format like: JS Divergence 0.0325, PSI 0.0085 (NO quotes around metric names)
    - For quoted text, use single quotes: 'high risk' not "high risk"
    - NO special characters that break JSON parsing (avoid: " \ inside strings)
    - NO line breaks within string values
    - If you need to reference terms, use parentheses: (critical) instead of "critical"
    - ABSOLUTELY NO NUMBERED SECTIONS in your analysis text (avoid 1. 2. 3. format)
    - Use natural paragraph breaks instead of numbered lists
    - Keep analysis flowing without section markers or numbered points
    """


class BusinessAnalysisTemplate(PromptTemplate):
    """Template for business perspective analysis."""
    
    def __init__(self):
        super().__init__([
            BusinessSystemMessageComponent(),
            UserContextComponent(),
            DatasetContextComponent(),
            ContextItemsComponent(),
            BusinessAnalysisInstructionComponent()
        ])


class BusinessSystemMessageComponent(PromptComponent):
    """System message for business analysis."""
    
    def generate(self, context: Dict[str, Any]) -> str:
        return """You are a professional data scientist, specializing in transforming technical analysis results into business insights."""


class BusinessAnalysisInstructionComponent(PromptComponent):
    """Business analysis specific instructions."""
    
    def generate(self, context: Dict[str, Any]) -> str:
        user_ctx = context.get('user_context', UserContext())
        
        return f"""
Please analyze the following business scenarios based on the user's {user_ctx.industry_sector} industry background:

1. **Business Scenario Identification:**
   - Based on the {user_ctx.industry_sector} industry characteristics, what is the most likely business scenario?
   - Consider the business significance of detected issues in the {user_ctx.industry_sector} industry
   - Propose the most likely application scenarios (A/B testing, model monitoring, market analysis, etc.)

2. **Business Impact Translation:**
   - Translate statistical metrics into specific business risks in the {user_ctx.industry_sector} industry
   - Estimate the impact on key business metrics
   - Provide specific impact estimates related to {user_ctx.industry_sector}

3. **Strategic Recommendations:**
   - Provide actionable recommendations for {user_ctx.professional_role}
   - Consider resource implications and implementation timeline
   - Prioritize based on business value and risk mitigation
"""


class ExecutiveAnalysisTemplate(PromptTemplate):
    """Template for executive perspective analysis."""
    
    def __init__(self):
        super().__init__([
            ExecutiveSystemMessageComponent(),
            UserContextComponent(),
            DatasetContextComponent(),
            ContextItemsComponent(),
            ExecutiveAnalysisInstructionComponent()
        ])


class ExecutiveSystemMessageComponent(PromptComponent):
    """System message for executive analysis."""
    
    def generate(self, context: Dict[str, Any]) -> str:
        return """You are an executive consultant specializing in translating technical data analysis into strategic business insights for C-level decision makers."""


class ExecutiveAnalysisInstructionComponent(PromptComponent):
    """Executive analysis specific instructions."""
    
    def generate(self, context: Dict[str, Any]) -> str:
        user_ctx = context.get('user_context', UserContext())
        
        return f"""
Provide an executive summary suitable for C-level stakeholders in {user_ctx.industry_sector}:

1. **Key Findings:**
   - High-level summary of critical issues (3-5 bullet points)
   - Quantify business impact where possible
   - Highlight immediate risks and opportunities

2. **Strategic Recommendations:**
   - Primary recommendation with clear rationale
   - Alternative strategies with pros/cons
   - Timeline and resource requirements

3. **Resource Requirements:**
   - Budget implications and team requirements
   - ROI considerations and risk mitigation
   - Implementation timeline and milestones
"""


class TrainAnalysisTemplate(PromptTemplate):
    """Template for training impact analysis."""
    
    def __init__(self):
        super().__init__([
            TrainAnalysisInstructionComponent()
        ])


class AdaptAnalysisTemplate(PromptTemplate):
    """Template for adaptation needs analysis."""
    
    def __init__(self):
        super().__init__([
            AdaptAnalysisInstructionComponent()
        ])


class ConditionalAnalysisTemplate(PromptTemplate):
    """Template for conditional distribution analysis."""
    
    def __init__(self):
        super().__init__([
            ConditionalAnalysisInstructionComponent()
        ])


class TrainAnalysisInstructionComponent(PromptComponent):
    """Training analysis specific instructions."""
    
    def generate(self, context: Dict[str, Any]) -> str:
        target_name = context.get('target_name', 'target')
        target_type = context.get('target_type', 'unknown')
        primary_dist = context.get('primary_dist', {})
        secondary_dist = context.get('secondary_dist', {})
        
        # Format type info for display
        type_info = f" ({target_type})" if target_type else ""
        
        # Format distributions as strings
        primary_dist_str = format_distribution(primary_dist)
        secondary_dist_str = format_distribution(secondary_dist)
        
        return f"""
                The user is training a machine learning model using a primary dataset and wants to assess label shift risks based on the target attribute distribution in both primary and secondary datasets.

                Target attribute: {target_name}{type_info}
                Primary distribution: {primary_dist_str}
                Secondary distribution: {secondary_dist_str}

                Does this label shift indicate training risk, such as overfitting to majority labels? Should the user consider label balancing or resampling strategies?
                """


class AdaptAnalysisInstructionComponent(PromptComponent):
    """Adaptation analysis specific instructions."""
    
    def generate(self, context: Dict[str, Any]) -> str:
        target_name = context.get('target_name', 'target')
        target_type = context.get('target_type', 'unknown')
        primary_dist = context.get('primary_dist', {})
        secondary_dist = context.get('secondary_dist', {})
        
        # Format type info for display
        type_info = f" ({target_type})" if target_type else ""
        
        # Format distributions as strings
        primary_dist_str = format_distribution(primary_dist)
        secondary_dist_str = format_distribution(secondary_dist)
        
        return f"""
                The user wants to apply a model trained on the primary dataset to the secondary dataset.

                Target attribute: {target_name}{type_info}
                Primary distribution: {primary_dist_str}
                Secondary distribution: {secondary_dist_str}

                Should adaptation methods be considered? Is direct inference risky? Recommend strategies if needed (e.g., reweighting, pseudo-labeling, domain adaptation).
                """


class ConditionalAnalysisInstructionComponent(PromptComponent):
    """Conditional distribution analysis specific instructions."""
    
    def generate(self, context: Dict[str, Any]) -> str:
        target_name = context.get('target_name', 'target')
        target_value = context.get('target_value', 'unknown')
        shifted_attr_name = context.get('shifted_attr_name', 'attribute')
        shifted_attr_type = context.get('shifted_attr_type', 'unknown')
        primary_dist = context.get('primary_dist', {})
        secondary_dist = context.get('secondary_dist', {})
        
        # Format type info for display
        type_info = f" ({shifted_attr_type})" if shifted_attr_type else ""
        
        # Format distributions as strings
        primary_dist_str = format_distribution(primary_dist)
        secondary_dist_str = format_distribution(secondary_dist)
        
        return f"""
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


def format_distribution(dist: Dict) -> str:
    """Format distribution dictionary as string for display."""
    if not dist:
        return "No distribution data available"
    
    if isinstance(dist, dict):
        items = []
        for key, value in dist.items():
            if isinstance(value, (int, float)):
                items.append(f"{key}: {value}")
            else:
                items.append(f"{key}: {str(value)}")
        return ", ".join(items)
    else:
        return str(dist)


# =============================================================================
# PROMPT MANAGER
# =============================================================================

class PromptManager:
    """Centralized prompt management system."""
    
    def __init__(self):
        self.templates = {
            'comprehensive_analysis': ComprehensiveAnalysisTemplate(),
            'severity_analysis': ComprehensiveAnalysisTemplate(),  # Alias for backward compatibility
            'business_analysis': BusinessAnalysisTemplate(),
            'executive_analysis': ExecutiveAnalysisTemplate(),
            'train_analysis': TrainAnalysisTemplate(),
            'adapt_analysis': AdaptAnalysisTemplate(),
            'conditional_analysis': ConditionalAnalysisTemplate(),
            'context_item_detailed_analysis': ContextItemDetailedAnalysisTemplate()  # New template
        }
    
    def generate_prompt(self, template_name: str, context: Dict[str, Any]) -> str:
        """Generate prompt using specified template."""
        if template_name not in self.templates:
            raise ValueError(f"Unknown template: {template_name}")
        
        template = self.templates[template_name]
        return template.generate(context)
    
    def add_template(self, name: str, template: PromptTemplate):
        """Add custom template to manager."""
        self.templates[name] = template
    
    def get_available_templates(self) -> List[str]:
        """Get list of available template names."""
        return list(self.templates.keys())


# =============================================================================
# NEW TEMPLATE FOR CONTEXT ITEM DETAILED ANALYSIS
# =============================================================================

class ContextItemDetailedAnalysisTemplate(PromptTemplate):
    """Template for generating detailed domain-specific analysis of a single context item."""
    
    def __init__(self):
        super().__init__([
            ContextItemDetailedSystemMessageComponent(),
            UserContextComponent(),
            DatasetContextComponent(),
            SingleContextItemComponent(),
            ContextItemDetailedInstructionComponent()
        ])


class ContextItemDetailedSystemMessageComponent(PromptComponent):
    """System message for detailed context item analysis."""
    
    def generate(self, context: Dict[str, Any]) -> str:
        return """You are an expert ML engineer specializing in model reliability, drift detection, and domain-specific risk assessment. You provide detailed, technical analysis focused on real-world deployment implications and domain-specific risks."""


class SingleContextItemComponent(PromptComponent):
    """Component for describing a single context item in detail."""
    
    def generate(self, context: Dict[str, Any]) -> str:
        context_item = context.get('context_item', {})
        context_type = context_item.get('type', 'unknown')
        
        prompt_section = "**CONTEXT ITEM TO ANALYZE:**\n"
        prompt_section += f"Type: {context_type.replace('_', ' ').title()}\n"
        
        if context_type == "drift_analysis":
            prompt_section += f"Attribute: {context_item.get('attribute_name', 'Unknown')}\n"
            prompt_section += f"Details: {context_item.get('metric_details', 'No details')}\n"
            prompt_section += f"Interpretation: {context_item.get('interpretation', 'No interpretation')}\n"
            
        elif context_type == "distribution_comparison":
            prompt_section += f"Summary: {context_item.get('summary_text', 'No summary')}\n"
            prompt_section += f"Details: {context_item.get('cell_info', 'No details')}\n"
            
        elif context_type == "conditional_distribution":
            prompt_section += f"Target: {context_item.get('target_attribute', 'Unknown')} = {context_item.get('target_value', 'Unknown')}\n"
            prompt_section += f"Compare With: {context_item.get('compare_attribute', 'Unknown')}\n"
            prompt_section += f"Summary: {context_item.get('summary_text', 'No summary')}\n"
            
        elif context_type == "metric":
            prompt_section += f"Metric: {context_item.get('metric_name', 'Unknown')}\n"
            prompt_section += f"Attribute: {context_item.get('attribute_name', 'Unknown')}\n"
            prompt_section += f"Details: {context_item.get('metric_details', 'No details')}\n"
            
        elif context_type == "target_distribution":
            prompt_section += f"Target Attribute: {context_item.get('target_attribute', 'Unknown')}\n"
            prompt_section += f"Summary: {context_item.get('summary_text', 'No summary')}\n"
        
        return prompt_section


class ContextItemDetailedInstructionComponent(PromptComponent):
    """Instructions for generating detailed domain-specific analysis."""
    
    def generate(self, context: Dict[str, Any]) -> str:
        user_ctx = context.get('user_context', UserContext())
        dataset_ctx = context.get('dataset_context')
        context_item = context.get('context_item', {})
        
        # Extract target attribute information
        target_attribute = None
        dataset_info = ""
        if dataset_ctx:
            target_attribute = dataset_ctx.comparison_context.get('target_attribute', 'Unknown')
            dataset_info = f"""
                            TARGET ATTRIBUTE: {target_attribute}
                            Primary Dataset: {dataset_ctx.primary_dataset.get('dataset_name', 'Unknown')} ({dataset_ctx.primary_dataset.get('record_count', 0):,} records)
                            Secondary Dataset: {dataset_ctx.secondary_dataset.get('dataset_name', 'Unknown')} ({dataset_ctx.secondary_dataset.get('record_count', 0):,} records)
                            """
        
        return f"""
                **YOUR TASK:**
                As an expert analyzing for a {user_ctx.professional_role} in the {user_ctx.industry_sector} industry, provide a comprehensive technical analysis of this specific context item focusing on domain-specific implications.
                
                {dataset_info}
                
                **REQUIRED ANALYSIS:**
                Provide a detailed technical explanation (200-400 words) covering:
                
                1. **Domain-Specific Impact**: How does this issue specifically affect {user_ctx.industry_sector} applications and {target_attribute} predictions?
                
                2. **Model Deployment Risks**: What are the concrete risks if a model trained on the primary dataset is directly applied to predict the secondary dataset? Include specific failure modes relevant to {user_ctx.industry_sector}.
                
                3. **Target Attribute Reliability**: How might this issue compromise the reliability of {target_attribute} predictions in production? Consider both false positives and false negatives.
                
                4. **Technical Consequences**: What specific technical problems could arise during inference? Consider distribution mismatch, calibration issues, and performance degradation.
                
                5. **Real-World Scenarios**: Provide {user_ctx.industry_sector}-specific examples of when this issue could cause problems in actual business operations.
                
                **RESPONSE FORMAT:**
                Provide ONLY the detailed analysis text (no JSON, no special formatting). Write in a technical but accessible tone appropriate for a {user_ctx.professional_role}. Focus on actionable insights and concrete risks rather than general statements.
                
                **CRITICAL REQUIREMENTS:**
                - Keep the response between 200-400 words
                - Focus specifically on {user_ctx.industry_sector} domain implications
                - Include concrete examples relevant to {target_attribute} prediction
                - Avoid generic statements - be specific about risks and consequences
                - Write for {user_ctx.expertise_level} expertise level
                - Use flowing paragraphs - NO numbered sections or bullet points (avoid 1. 2. 3. format)
                - Structure your analysis with natural paragraph breaks instead of numbered lists
                - Keep analysis narrative and coherent without section markers
                """


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_user_context(user_info: Dict[str, Any]) -> UserContext:
    """Create UserContext from user information, handling dynamic user data without hardcoded defaults."""
    
    # Helper function to get value or provide contextual default
    def get_contextual_value(key: str, fallback: str) -> str:
        value = user_info.get(key)
        # Handle both missing keys and explicit None values
        if value is None or value == "":
            return fallback
        return value
    
    # Check if we have profile information available
    has_profile = user_info.get('has_profile', False)
    profile_completeness = user_info.get('profile_completeness', 0)
    
    if not has_profile or profile_completeness == 0:
        # No profile available - use general context indicators
        return UserContext(
            industry_sector="General Business",
            professional_role="Professional",
            expertise_level="General",
            domain_context="General business analysis - user profile information not available"
        )
    
    # Profile partially or fully available - use actual values with contextual fallbacks
    industry = get_contextual_value('industry_sector', 'General Industry')
    role = get_contextual_value('professional_role', 'Professional')
    expertise = get_contextual_value('expertise_level', 'General')
    
    # Create domain context based on available information
    domain_parts = []
    if industry != 'General Industry':
        domain_parts.append(f"{industry} industry")
    if role != 'Professional':
        domain_parts.append(f"{role} role")
    
    if domain_parts:
        domain_context = f"{' - '.join(domain_parts)} context"
    else:
        domain_context = "General professional context"
    
    return UserContext(
        industry_sector=industry,
        professional_role=role,
        expertise_level=expertise,
        domain_context=domain_context
    )


def create_dataset_context(dataset_metadata: Dict[str, Any]) -> DatasetContext:
    """Create DatasetContext from dataset metadata."""
    return DatasetContext(
        primary_dataset=dataset_metadata.get('primary_dataset', {}),
        secondary_dataset=dataset_metadata.get('secondary_dataset', {}),
        comparison_context=dataset_metadata.get('comparison_context', {})
    )


# =============================================================================
# GLOBAL PROMPT MANAGER INSTANCE
# =============================================================================

# Global instance for easy access
prompt_manager = PromptManager() 