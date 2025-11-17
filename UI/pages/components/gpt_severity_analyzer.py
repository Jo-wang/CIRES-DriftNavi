"""
GPT-based Severity Analysis Module

This module provides intelligent severity assessment for context items using OpenAI's GPT-4.
Instead of rule-based thresholds, it leverages AI to understand the context and impact
of different issues based on dataset characteristics and strategy requirements.

Author: DriftNavi Team
Created: 2025
"""

import json
from typing import Dict, List, Any, Optional, Tuple
from functools import lru_cache
import hashlib
import time
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from dash import dcc, html
from flask_login import current_user
from UI.functions.global_vars import global_vars


class GPTSeverityAnalyzer:
    """
    Comprehensive GPT-powered severity analysis system for drift detection.
    
    This class provides multi-layered analysis of detected drift and distribution
    shifts, offering contextualized insights based on user profile and domain expertise.
    """
    
    @staticmethod
    def _get_user_context() -> Dict[str, Any]:
        """
        Extract user context information for personalized analysis using dynamic user data.
        
        Returns:
            Dict containing actual user background information or None for missing fields
        """
        try:
            # Use the dynamic user context extraction from global_vars
            user_context = global_vars.get_user_context(current_user)
            
            print(f"[GPT SEVERITY] Retrieved user context - Profile Complete: {user_context.get('profile_completeness', 0):.1f}%")
            
            return user_context
            
        except Exception as e:
            print(f"[GPT SEVERITY] Error retrieving user context: {str(e)}")
            # Return minimal structure indicating no user context available
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
    
    @staticmethod
    def _build_adaptive_prompt_context(user_context: Dict[str, Any]) -> str:
        """
        Build adaptive prompt context based on available user information.
        
        Args:
            user_context: User context dictionary from _get_user_context
            
        Returns:
            str: Adaptive context string for GPT prompting
        """
        # Start with basic context
        context_parts = []
        
        # Add user-specific context only if available
        if user_context.get("has_profile") and user_context.get("profile_completeness", 0) > 0:
            context_parts.append("## User Context")
            
            # Professional role context
            if user_context.get("professional_role"):
                context_parts.append(f"**Professional Role**: {user_context['professional_role']}")
            
            # Industry context
            if user_context.get("industry_sector"):
                context_parts.append(f"**Industry**: {user_context['industry_sector']}")
            
            # Expertise level context
            if user_context.get("expertise_level"):
                context_parts.append(f"**Expertise Level**: {user_context['expertise_level']}")
            
            # Technical level context
            if user_context.get("technical_level"):
                context_parts.append(f"**Technical Background**: {user_context['technical_level']}")
            
            # drift awareness context
            if user_context.get("drift_awareness"):
                context_parts.append(f"**drift Awareness**: {user_context['drift_awareness']}")
            
            # Areas of interest
            if user_context.get("areas_of_interest") and isinstance(user_context["areas_of_interest"], list):
                if len(user_context["areas_of_interest"]) > 0:
                    interests = ", ".join(user_context["areas_of_interest"])
                    context_parts.append(f"**Areas of Interest**: {interests}")
            
            # Custom prompts if available
            if user_context.get("persona_prompt"):
                context_parts.append(f"**User Persona**: {user_context['persona_prompt']}")
            
        else:
            # No user profile available - adapt accordingly
            context_parts.append("## General Analysis Context")
            context_parts.append("User profile information is not available. Provide analysis suitable for a general audience with clear explanations of technical concepts.")
        
        return "\n".join(context_parts) if context_parts else ""
    
    @staticmethod
    def _generate_adaptive_analysis_instructions(user_context: Dict[str, Any]) -> str:
        """
        Generate analysis instructions that adapt to user's background.
        
        Args:
            user_context: User context dictionary
            
        Returns:
            str: Adaptive analysis instructions
        """
        instructions = []
        
        # Base instructions
        instructions.append("## Analysis Instructions")
        
        # Adapt based on technical level
        technical_level = user_context.get("technical_level")
        if technical_level == "Expert" or technical_level == "Advanced":
            instructions.append("- Provide detailed statistical analysis with technical metrics")
            instructions.append("- Include specific algorithmic recommendations")
            instructions.append("- Reference relevant academic literature or methodologies")
        elif technical_level == "Intermediate":
            instructions.append("- Balance technical detail with clear explanations")
            instructions.append("- Provide actionable recommendations with reasoning")
            instructions.append("- Explain statistical concepts when necessary")
        elif technical_level == "Beginner" or technical_level is None:
            instructions.append("- Focus on clear, non-technical explanations")
            instructions.append("- Provide practical insights with minimal jargon")
            instructions.append("- Explain all statistical concepts in simple terms")
        
        # Adapt based on professional role
        role = user_context.get("professional_role")
        if role:
            if "Manager" in role or "Executive" in role:
                instructions.append(f"- Focus on business impact and strategic implications")
                instructions.append(f"- Provide executive-level recommendations")
            elif "Scientist" in role or "Analyst" in role:
                instructions.append(f"- Emphasize methodology and statistical rigor")
                instructions.append(f"- Provide detailed analytical insights")
            elif "Engineer" in role:
                instructions.append(f"- Focus on implementation details and technical solutions")
                instructions.append(f"- Provide system-level recommendations")
        
        # Adapt based on industry
        industry = user_context.get("industry_sector")
        if industry:
            instructions.append(f"- Consider implications specific to the {industry} industry")
            instructions.append(f"- Reference relevant {industry} standards and practices where applicable")
        
        return "\n".join(instructions)

    def __init__(self, openai_api_key: str = None):
        """
        Initialize the GPT severity analyzer.
        
        Args:
            openai_api_key: OpenAI API key. If None, will use environment variable.
        """
        # OpenAI client will automatically use OPENAI_API_KEY environment variable
        pass
        
    @staticmethod
    def get_dataset_metadata() -> Dict[str, Any]:
        """
        Extract comprehensive dataset metadata for GPT analysis.
        
        Returns:
            Dict containing dataset metadata
        """
        try:
            from UI.functions.global_vars import global_vars
            import pandas as pd
            
            metadata = {
                "primary_dataset": {},
                "secondary_dataset": {},
                "comparison_context": {}
            }
            
            # Primary dataset metadata
            if hasattr(global_vars, 'df') and global_vars.df is not None:
                df = global_vars.df
                metadata["primary_dataset"] = {
                    "record_count": len(df),
                    "column_count": len(df.columns),
                    "column_names": list(df.columns),
                    "column_types": {},
                    "missing_data_percentage": round(df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100, 2),
                    "numeric_columns": list(df.select_dtypes(include=['number']).columns),
                    "categorical_columns": list(df.select_dtypes(include=['object', 'category']).columns),
                    "dataset_name": getattr(global_vars, 'file_name', 'Primary Dataset')
                }
                
                # Detailed column type analysis
                for col in df.columns:
                    unique_count = df[col].nunique()
                    null_count = df[col].isnull().sum()
                    
                    if pd.api.types.is_numeric_dtype(df[col]):
                        if unique_count == 2:
                            col_category = "Binary"
                        elif unique_count <= 10:
                            col_category = "Discrete Numeric"
                        else:
                            col_category = "Continuous Numeric"
                    elif pd.api.types.is_datetime64_any_dtype(df[col]):
                        col_category = "Datetime"
                    else:
                        if unique_count <= 2:
                            col_category = "Binary Categorical"
                        elif unique_count <= 20:
                            col_category = "Categorical"
                        else:
                            col_category = "High Cardinality Text"
                    
                    metadata["primary_dataset"]["column_types"][col] = {
                        "category": col_category,
                        "unique_count": unique_count,
                        "null_percentage": round(null_count / len(df) * 100, 2),
                        "dtype": str(df[col].dtype)
                    }
            
            # Secondary dataset metadata
            if hasattr(global_vars, 'secondary_df') and global_vars.secondary_df is not None:
                df_sec = global_vars.secondary_df
                metadata["secondary_dataset"] = {
                    "record_count": len(df_sec),
                    "column_count": len(df_sec.columns),
                    "column_names": list(df_sec.columns),
                    "missing_data_percentage": round(df_sec.isnull().sum().sum() / (len(df_sec) * len(df_sec.columns)) * 100, 2),
                    "numeric_columns": list(df_sec.select_dtypes(include=['number']).columns),
                    "categorical_columns": list(df_sec.select_dtypes(include=['object', 'category']).columns),
                    "dataset_name": getattr(global_vars, 'secondary_file_name', 'Secondary Dataset')
                }
            
            # Comparison context
            if metadata["primary_dataset"] and metadata["secondary_dataset"]:
                primary_cols = set(metadata["primary_dataset"]["column_names"])
                secondary_cols = set(metadata["secondary_dataset"]["column_names"])
                
                metadata["comparison_context"] = {
                    "common_columns": list(primary_cols & secondary_cols),
                    "primary_only_columns": list(primary_cols - secondary_cols),
                    "secondary_only_columns": list(secondary_cols - primary_cols),
                    "size_ratio": metadata["secondary_dataset"]["record_count"] / metadata["primary_dataset"]["record_count"] if metadata["primary_dataset"]["record_count"] > 0 else 0,
                    "target_attribute": getattr(global_vars, 'target_attribute', None)
                }
            
            return metadata
            
        except Exception as e:
            print(f"[GPT SEVERITY] Error extracting dataset metadata: {str(e)}")
            return {
                "primary_dataset": {"record_count": 0, "column_names": []},
                "secondary_dataset": {"record_count": 0, "column_names": []},
                "comparison_context": {}
            }

    def _create_severity_assessment_prompt(
        self, 
        context_items: List[Dict[str, Any]], 
        dataset_metadata: Dict[str, Any],
        strategy_focus: str,
        user_context: Dict[str, Any] = None
    ) -> str:
        """Create comprehensive prompt for GPT severity assessment with adaptive user context."""
        
        # Get user context for personalization
        if user_context is None:
            user_context = self._get_user_context()
        
        # Build adaptive user context and instructions
        adaptive_context = self._build_adaptive_prompt_context(user_context)
        adaptive_instructions = self._generate_adaptive_analysis_instructions(user_context)
        
        # Try to use prompt manager if available, otherwise create direct prompt
        try:
            from .prompt_manager import (
                prompt_manager, 
                create_user_context, 
                create_dataset_context
            )
            
            # Prepare context data for prompt manager - convert user context dict to UserContext dataclass
            prompt_context = {
                'user_context': create_user_context(user_context),  # Convert dict to UserContext dataclass
                'dataset_context': create_dataset_context(dataset_metadata),
                'context_items': context_items,
                'strategy_focus': strategy_focus,
                'analysis_instructions': adaptive_instructions
            }
            
            # Generate prompt using unified template system
            return prompt_manager.generate_prompt('comprehensive_analysis', prompt_context)
            
        except ImportError:
            # Fallback: Create direct prompt if prompt manager is not available
            return self._create_direct_severity_prompt(
                context_items, dataset_metadata, strategy_focus, 
                adaptive_context, adaptive_instructions
            )
    
    def _create_direct_severity_prompt(
        self, 
        context_items: List[Dict[str, Any]], 
        dataset_metadata: Dict[str, Any],
        strategy_focus: str,
        adaptive_context: str,
        adaptive_instructions: str
    ) -> str:
        """Create direct severity assessment prompt when prompt manager is unavailable."""
        
        # Build context items description with explicit indexing
        context_descriptions = []
        for i, item in enumerate(context_items):
            context_descriptions.append(f"{i}. **{item.get('type', 'Unknown')}**: {item.get('description', 'No description')}")
        
        # Debug: Log the indexed context descriptions
        print(f"[GPT SEVERITY] 📝 Context items with explicit indexing:")
        for desc in context_descriptions:
            print(f"[GPT SEVERITY]   {desc}")
        
        # Create analysis task instruction based on strategy_focus
        if strategy_focus:
            analysis_instruction = f"Analyze the severity of these detected issues with focus on {strategy_focus} strategy."
        else:
            analysis_instruction = "Analyze the severity of these detected issues and intelligently recommend the most appropriate strategy (monitor/retrain/finetune) based on the analysis."
        
        # Create comprehensive prompt
        prompt = f"""
{adaptive_context}

{adaptive_instructions}

## Dataset Information
Primary Dataset: {dataset_metadata.get('primary_dataset', {}).get('record_count', 0)} records
Secondary Dataset: {dataset_metadata.get('secondary_dataset', {}).get('record_count', 0)} records

## Detected Issues
{chr(10).join(context_descriptions)}

## Analysis Task
{analysis_instruction}

Please provide a JSON response with:
1. Overall severity assessment (critical/moderate/minor)
2. Individual item analysis with severity scores - IMPORTANT: Use context_id to match the exact numbered items above (0, 1, 2, etc.)
3. Intelligent strategy recommendation based on the analysis (monitor/retrain/finetune)
4. Personalized insights appropriate for the user's background

CRITICAL: For layer2_context_analysis, each context_id MUST correspond exactly to the numbered items:
- context_id 0 → item "0. **Type**"
- context_id 1 → item "1. **Type**"
- etc.

Return only valid JSON format.
"""
        return prompt.strip()
    
    @lru_cache(maxsize=32)
    def _get_cached_gpt_response(self, prompt_hash: str, prompt: str) -> Optional[str]:
        """Get cached GPT response or make new API call."""
        try:
            # Get the user's selected model from global_vars.agent
            from UI.functions.global_vars import global_vars
            
            # Use the user's selected model if available, otherwise fall back to gpt-4o-mini
            if hasattr(global_vars, 'agent') and global_vars.agent and hasattr(global_vars.agent, 'model_name'):
                selected_model = global_vars.agent.model_name
                print(f"[GPT SEVERITY] Using user's selected model: {selected_model}")
            else:
                selected_model = "gpt-4o-mini"
                print(f"[GPT SEVERITY] No agent found, using fallback model: {selected_model}")
            
            # Use OpenAI client for v1.0+ compatibility
            from openai import OpenAI
            client = OpenAI()
            
            response = client.chat.completions.create(
                model=selected_model,  # Use user's selected model instead of hardcoded "gpt-4"
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an expert ML engineer specializing in drift detection, drift analysis, and model maintenance strategies. Provide precise, actionable analysis."
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=4000,  # Increased to accommodate detailed analysis for each context item
                temperature=0.1
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"[GPT SEVERITY] API call failed: {str(e)}")
            return None
    
    def analyze_context_severity(
        self, 
        context_items: List[Dict[str, Any]],
        strategy_focus: str = None,  # Let GPT decide strategy based on analysis, not drift with defaults
        user_context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Analyze context items using GPT for intelligent severity assessment with user personalization."""
        if not context_items:
            return {
                "gpt_analysis": "No issues to analyze",
                "individual_scores": {},
                "overall_assessment": {
                    "high_count": 0,
                    "medium_count": 0,
                    "low_count": 0,
                    "recommendation": "monitor"  # Default when no issues exist
                },
                "analysis_timestamp": time.time()
            }
        
        # Get dataset metadata
        dataset_metadata = self.get_dataset_metadata()
        
        # Get user context for personalization
        if user_context is None:
            user_context = self._get_user_context()
        
        # Create informative log message based on available user context
        role = user_context.get('professional_role', 'Unknown Role')
        industry = user_context.get('industry_sector', 'Unknown Industry')
        profile_complete = user_context.get('profile_completeness', 0)
        
        if user_context.get('has_profile', False):
            print(f"[GPT SEVERITY] Analyzing for {role} in {industry} (Profile: {profile_complete:.0f}% complete)")
        else:
            print(f"[GPT SEVERITY] Analyzing with general context (no user profile available)")
        
        # Create comprehensive prompt with user personalization
        prompt = self._create_severity_assessment_prompt(
            context_items, dataset_metadata, strategy_focus, user_context
        )
        
        # DEBUG: Log prompt details
        print(f"[GPT SEVERITY] 📝 Generated prompt length: {len(prompt)} characters")
        print(f"[GPT SEVERITY] Prompt preview: {prompt[:300]}...")
        print(f"[GPT SEVERITY] Context items count: {len(context_items)}")
        print(f"[GPT SEVERITY] Strategy focus: {strategy_focus}")
        if "layer4_strategy_selection" in prompt:
            print(f"[GPT SEVERITY] ✅ Prompt contains layer4_strategy_selection instructions")
        else:
            print(f"[GPT SEVERITY] ❌ Prompt missing layer4_strategy_selection instructions")
        
        # Generate hash for caching (include relevant user context in hash)
        user_cache_parts = [
            user_context.get('professional_role', ''),
            user_context.get('industry_sector', ''),
            user_context.get('technical_level', ''),
            user_context.get('expertise_level', ''),
            str(user_context.get('profile_completeness', 0))
        ]
        user_cache_string = "_".join(filter(None, user_cache_parts))  # Filter out None/empty values
        
        cache_key = f"{prompt}_{user_cache_string}"
        prompt_hash = hashlib.md5(
            (cache_key + str(time.time() // 3600)).encode()  # Cache for 1 hour
        ).hexdigest()
        
        # Get GPT analysis
        gpt_response = self._get_cached_gpt_response(prompt_hash, prompt)
        
        # DEBUG: Log GPT response details
        if gpt_response:
            print(f"[GPT SEVERITY] ✅ GPT API SUCCESS - Response length: {len(gpt_response)} characters")
            print(f"[GPT SEVERITY] Response preview: {gpt_response[:200]}...")
            if '{' in gpt_response:
                json_start = gpt_response.find('{')
                print(f"[GPT SEVERITY] JSON starts at position: {json_start}")
                print(f"[GPT SEVERITY] JSON preview: {gpt_response[json_start:json_start+300]}...")
            else:
                print(f"[GPT SEVERITY] ❌ NO JSON FOUND in response!")
        else:
            print(f"[GPT SEVERITY] ❌ GPT API FAILED - No response received")
        
        if not gpt_response:
            # Create comprehensive fallback data when API fails
            api_fallback_data = {
                "layer1_severity_statistics": {
                    "high_count": 0,
                    "medium_count": 0,
                    "low_count": len(context_items),
                    "overall_risk_level": "Medium",
                    "confidence_score": 0.3,
                    "summary": f"API analysis unavailable (quota/error). {len(context_items)} items detected requiring review."
                },
                "layer2_context_analysis": [
                    {
                        "context_id": i,
                        "title": f"Analysis for {context_items[i].get('type', 'Unknown').replace('_', ' ').title()}",
                        "severity_score": 50,
                        "risk_level": "Medium",
                        "explanation": {
                            "beginner": f"API unavailable: This {context_items[i].get('type', 'unknown')} issue needs manual review.",
                            "intermediate": f"API unavailable: The {context_items[i].get('type', 'unknown')} requires technical assessment while GPT is offline.",
                            "advanced": f"API unavailable: Technical evaluation needed for {context_items[i].get('type', 'unknown')} - check system diagnostics."
                        },
                        "business_impact": f"Impact assessment pending for {context_items[i].get('type', 'unknown')} due to API unavailability. Recommend manual review.",
                        "technical_details": f"Technical analysis for {context_items[i].get('type', 'unknown')} requires API access. Please check OpenAI configuration and quotas.",
                        "action_required": f"Manual review required for {context_items[i].get('type', 'unknown')} while API is unavailable."
                    }
                    for i in range(len(context_items))
                ],
                "layer3_joint_analysis": {
                    "overall_assessment": f"Systematic assessment of {len(context_items)} detected issues pending due to API unavailability. Without AI analysis, manual assessment is required to evaluate the compound risks of deploying a model trained on primary dataset directly to secondary dataset. Consider potential failure modes where detected distribution shifts, correlation changes, or data quality issues could compound to create significant prediction reliability problems. The cumulative impact across multiple detected issues may threaten target attribute accuracy when the model encounters secondary dataset patterns not present during training. Immediate manual review recommended to assess primary-to-secondary inference risks and determine appropriate adaptation strategy before model deployment."
                },
                "layer4_strategy_selection": {
                    "recommended_strategy": "monitor",
                    "strategy_overview": "Continue monitoring model performance with enhanced alerting and validation systems. Suitable when issues are manageable and model performance remains acceptable.",
                    "confidence": 0.3,
                    "reasoning": "Default monitoring strategy recommended while API is unavailable. Manual assessment needed for optimal strategy selection.",
                    "alternative_strategies": [
                        {"strategy": "manual_review", "confidence": 0.7, "rationale": "Best option while API unavailable"},
                        {"strategy": "delay_decision", "confidence": 0.5, "rationale": "Wait for API access to resume"}
                    ],
                    "implementation_roadmap": {
                        "phase1_immediate": ["Check OpenAI API status", "Verify quota limits", "Review error logs"],
                        "phase2_short_term": ["Manual item assessment", "Document findings", "Retry API access"],
                        "phase3_long_term": ["Comprehensive analysis when API available", "Update strategies based on results"]
                    },
                    "success_metrics": ["API restoration", "Manual review completion"],
                    "risk_factors": ["Analysis delay", "Potential oversight without AI assistance"]
                }
            }
            
            # Convert to the expected format with comprehensive_data
            api_fallback_legacy = self._create_legacy_format_from_fallback(api_fallback_data)
            
            return {
                "gpt_analysis": "Analysis temporarily unavailable. Please check your OpenAI API configuration and quota limits.",
                "individual_scores": api_fallback_legacy.get("individual_scores", {}),
                "overall_assessment": api_fallback_legacy.get("overall_assessment", {}),
                "comprehensive_data": api_fallback_data,  # ← 关键：添加这个字段
                "analysis_timestamp": time.time(),
                "error": "GPT API unavailable"
            }
        
        # Parse GPT response for structured data
        print(f"[GPT SEVERITY] 🔄 Starting JSON parsing...")
        parsed_data = self._parse_gpt_severity_response(gpt_response, context_items)
        
        # Debug: Check what parsed_data contains
        print(f"[GPT SEVERITY] ✅ JSON parsing completed")
        print(f"[GPT SEVERITY] parsed_data keys: {list(parsed_data.keys())}")
        if "comprehensive_data" in parsed_data:
            comp_data = parsed_data["comprehensive_data"]
            layer4_data = comp_data.get("layer4_strategy_selection", {})
            layer2_data = comp_data.get("layer2_context_analysis", [])
            layer2_count = len(layer2_data)
            print(f"[GPT SEVERITY] SUCCESS PATH - Layer2 data count: {layer2_count}")
            print(f"[GPT SEVERITY] SUCCESS PATH - Layer4 strategy: {layer4_data.get('recommended_strategy', 'unknown')}")
            print(f"[GPT SEVERITY] SUCCESS PATH - Layer4 reasoning: {layer4_data.get('reasoning', 'unknown')[:100]}...")
            
            # DEBUG: Print each layer2 item to verify uniqueness
            print(f"[GPT SEVERITY] 📝 LAYER2 CONTENT VERIFICATION:")
            for i, item in enumerate(layer2_data):
                title = item.get('title', 'Unknown')
                context_id = item.get('context_id', 'unknown')
                risk_level = item.get('risk_level', 'unknown')
                business_impact = item.get('business_impact', '')
                detailed_analysis = item.get('detailed_analysis', '')
                print(f"[GPT SEVERITY] Layer2[{i}] ID:{context_id} - {title} ({risk_level})")
                print(f"[GPT SEVERITY]   Business: {business_impact[:80]}{'...' if len(business_impact) > 80 else ''}")
                print(f"[GPT SEVERITY]   Detailed Analysis: {'✅ Present' if detailed_analysis else '❌ Missing'} ({len(detailed_analysis)} chars)")
                if detailed_analysis:
                    print(f"[GPT SEVERITY]   Analysis Preview: {detailed_analysis[:100]}{'...' if len(detailed_analysis) > 100 else ''}")
        else:
            print(f"[GPT SEVERITY] ❌ WARNING - No comprehensive_data in parsed_data!")
            if parsed_data.get("overall_assessment", {}).get("recommendation"):
                print(f"[GPT SEVERITY] ❌ FALLBACK PATH DETECTED - Using fallback data")
        
        result = {
            "gpt_analysis": gpt_response,
            "individual_scores": parsed_data["individual_scores"],
            "overall_assessment": parsed_data["overall_assessment"],
            "comprehensive_data": parsed_data.get("comprehensive_data", {}),  # ← 关键：添加这个字段
            "analysis_timestamp": time.time(),
            "dataset_metadata": dataset_metadata,
            "user_context": user_context  # Include user context in response
        }
        
        print(f"[GPT SEVERITY] Final result has comprehensive_data: {'comprehensive_data' in result}")
        return result
    
    def _parse_gpt_severity_response(
        self, 
        gpt_response: str, 
        context_items: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Parse GPT JSON response to extract comprehensive four-layer analysis data."""
        import json
        
        # Default fallback data with proper layer2 content
        fallback_data = {
            "layer1_severity_statistics": {
                "high_count": 0,
                "medium_count": 0,
                "low_count": len(context_items),
                "overall_risk_level": "Low",
                "confidence_score": 0.5,
                "summary": "GPT analysis failed - JSON parsing error or API unavailable. Manual review required."
            },
            "layer2_context_analysis": [
                {
                    "context_id": i,
                    "context_type": context_items[i].get('type', 'unknown'),  # Ensure internal key format
                    "title": f"System Error: {context_items[i].get('type', 'Unknown').replace('_', ' ').title()}",
                    "severity_score": 25,  # Conservative low score
                    "risk_level": "Low",   # Conservative due to uncertainty
                    "explanation": {
                        "beginner": f"System error prevented analysis of {context_items[i].get('type', 'unknown')}. Manual review needed.",
                        "intermediate": f"Technical failure: {context_items[i].get('type', 'unknown')} analysis unavailable due to system error.",
                        "advanced": f"GPT analysis system failure for {context_items[i].get('type', 'unknown')} - requires manual technical evaluation."
                    },
                    "business_impact": f"Business impact assessment for {context_items[i].get('type', 'unknown')} unavailable due to system failure. Manual evaluation required.",
                    "technical_details": f"Technical analysis for {context_items[i].get('type', 'unknown')} failed due to GPT system error or JSON parsing failure.",
                    "action_required": f"URGENT: Manual review required for {context_items[i].get('type', 'unknown')} - automated analysis system unavailable."
                }
                for i in range(len(context_items))
            ],
            "layer3_joint_analysis": {
                "overall_assessment": f"SYSTEM FAILURE: Automated joint analysis of {len(context_items)} detected issues is unavailable due to GPT analysis system error. Manual assessment is required to evaluate compound risks for primary-to-secondary dataset inference. The systematic evaluation of distribution shifts, correlation changes, and data quality issues could not be completed due to technical failure. This represents a system limitation, not an assessment of actual risk severity. Manual expert review is urgently required to determine adaptation strategies and deployment safety."
            },
            "layer4_strategy_selection": {
                "recommended_strategy": "manual_review",
                "strategy_overview": "SYSTEM ERROR: Automated strategy recommendation unavailable. Manual expert review required immediately due to analysis system failure.",
                "confidence": 0.0,  # No confidence due to system failure
                "reasoning": "GPT analysis system failed - no reliable automated recommendation possible. All detected issues require manual expert evaluation.",
                "alternative_strategies": [
                    {"strategy": "delay_deployment", "confidence": 0.8, "rationale": "Safest option until manual analysis completed"},
                    {"strategy": "conservative_monitoring", "confidence": 0.3, "rationale": "High-risk option without proper analysis"}
                ],
                "implementation_roadmap": {
                    "phase1_immediate": ["Stop automated deployment", "Alert technical team", "Begin manual review"],
                    "phase2_short_term": ["Complete manual analysis", "Fix GPT analysis system", "Validate results"],
                    "phase3_long_term": ["Resume automated analysis", "Implement system monitoring"]
                },
                "success_metrics": ["Manual analysis completion", "System restoration", "Analysis validation"],
                "risk_factors": ["Analysis system unreliable", "Manual review delay", "Potential oversight without AI assistance"]
            }
        }
        
        try:
            # First, try to extract JSON from the response
            print(f"[JSON PARSE] 🔍 Searching for JSON boundaries...")
            json_start = gpt_response.find('{')
            json_end = gpt_response.rfind('}') + 1
            print(f"[JSON PARSE] JSON boundaries: start={json_start}, end={json_end}")
            
            if json_start == -1 or json_end == 0:
                print(f"[JSON PARSE] ❌ No JSON found in response")
                print(f"[JSON PARSE] Full response: {gpt_response[:500]}...")
                print(f"[JSON PARSE] FALLBACK PATH 1 - Using default fallback with {len(fallback_data['layer2_context_analysis'])} layer2 items")
                fallback_result = self._create_legacy_format_from_fallback(fallback_data)
                print(f"[JSON PARSE] FALLBACK PATH 1 - Result has comprehensive_data: {'comprehensive_data' in fallback_result}")
                return fallback_result
            
            json_text = gpt_response[json_start:json_end]
            print(f"[JSON PARSE] 📝 Extracted JSON length: {len(json_text)} characters")
            print(f"[JSON PARSE] Raw JSON preview: {json_text[:300]}...")
            
            # FIRST: Try to parse the original JSON without any fixes
            print(f"[JSON PARSE] 🔍 Testing original JSON without fixes...")
            try:
                parsed_data = json.loads(json_text)
                print(f"[JSON PARSE] ✅ SUCCESS: Original JSON is valid!")
                print(f"[JSON PARSE] Parsed keys: {list(parsed_data.keys())}")
                
                # Validate required structure
                required_keys = ['layer1_severity_statistics', 'layer2_context_analysis', 
                               'layer3_joint_analysis', 'layer4_strategy_selection']
                
                for key in required_keys:
                    if key not in parsed_data:
                        print(f"[JSON PARSE] ❌ Missing required key: {key}")
                        raise ValueError(f"Missing key: {key}")
                
                print(f"[JSON PARSE] ✅ All required keys found - using original JSON!")
                # Check layer4 content specifically
                layer4_data = parsed_data.get('layer4_strategy_selection', {})
                print(f"[JSON PARSE] Layer4 preview: strategy={layer4_data.get('recommended_strategy')}, reasoning={layer4_data.get('reasoning', '')[:50]}...")
                
                # Convert to legacy format for backward compatibility
                print(f"[JSON PARSE] 🔄 Converting to legacy format...")
                return self._convert_comprehensive_to_legacy_format(parsed_data)
                
            except json.JSONDecodeError as original_error:
                print(f"[JSON PARSE] ❌ Original JSON has errors: {str(original_error)}")
                print(f"[JSON PARSE] 🔧 Will attempt JSON fixes...")
                
                # Only apply fixes if the original JSON failed
                original_length = len(json_text)
                json_text = self._fix_common_json_issues(json_text)
                fixed_length = len(json_text)
                print(f"[JSON PARSE] JSON fix applied: {original_length} → {fixed_length} chars")
                print(f"[JSON PARSE] Fixed JSON preview: {json_text[:300]}...")
                
                print(f"[JSON PARSE] 📊 Attempting JSON.loads() on fixed JSON...")
                parsed_data = json.loads(json_text)
                print(f"[JSON PARSE] ✅ Fixed JSON parsing successful!")
                
                # Validate required structure
                required_keys = ['layer1_severity_statistics', 'layer2_context_analysis', 
                               'layer3_joint_analysis', 'layer4_strategy_selection']
                
                for key in required_keys:
                    if key not in parsed_data:
                        print(f"[JSON PARSE] ❌ Missing required key after fixing: {key}")
                        raise ValueError(f"Missing key: {key}")
                
                print(f"[JSON PARSE] ✅ All required keys found in fixed JSON!")
                layer4_data = parsed_data.get('layer4_strategy_selection', {})
                print(f"[JSON PARSE] Layer4 preview: strategy={layer4_data.get('recommended_strategy')}, reasoning={layer4_data.get('reasoning', '')[:50]}...")
                
                # Convert to legacy format for backward compatibility
                print(f"[JSON PARSE] 🔄 Converting fixed JSON to legacy format...")
                return self._convert_comprehensive_to_legacy_format(parsed_data)
            
        except json.JSONDecodeError as e:
            print(f"[JSON PARSE] ❌ JSON DECODE ERROR: {str(e)}")
            print(f"[JSON PARSE] Error position: line {e.lineno}, column {e.colno}")
            print(f"[JSON PARSE] Original JSON snippet: {gpt_response[json_start:json_start+300]}...")
            print(f"[JSON PARSE] Fixed JSON snippet: {json_text[:300]}...")
            
            # Show the problematic area around the error position
            if hasattr(e, 'pos') and e.pos < len(json_text):
                error_start = max(0, e.pos - 100)
                error_end = min(len(json_text), e.pos + 100)
                print(f"[JSON PARSE] Error context: ...{json_text[error_start:error_end]}...")
                print(f"[JSON PARSE] Error position marked: {' ' * (e.pos - error_start)}^")
            
            # Try a more aggressive JSON fix
            try:
                print(f"[JSON PARSE] 🔧 Trying aggressive JSON fix...")
                aggressive_fix = self._aggressive_json_fix(json_text)
                print(f"[JSON PARSE] Aggressive fix result length: {len(aggressive_fix)}")
                print(f"[JSON PARSE] Aggressive fix preview: {aggressive_fix[:300]}...")
                
                # Check if aggressive fix resolved quote issues
                quote_issues = aggressive_fix.count('""') + aggressive_fix.count('" "')
                print(f"[JSON PARSE] Aggressive fix quote issue indicators: {quote_issues}")
                if quote_issues == 0:
                    print(f"[JSON PARSE] ✅ Aggressive fix appears to have resolved quote issues")
                else:
                    print(f"[JSON PARSE] ⚠️ Aggressive fix may still have quote issues")
                
                parsed_data = json.loads(aggressive_fix)
                
                # If successful, validate and return
                required_keys = ['layer1_severity_statistics', 'layer2_context_analysis', 
                               'layer3_joint_analysis', 'layer4_strategy_selection']
                
                print(f"[JSON PARSE] ✅ Aggressive fix JSON parsing successful!")
                for key in required_keys:
                    if key not in parsed_data:
                        print(f"[JSON PARSE] ❌ Missing required key after aggressive fix: {key}")
                        raise ValueError(f"Missing key: {key}")
                
                print(f"[JSON PARSE] ✅ Aggressive JSON fix succeeded!")
                return self._convert_comprehensive_to_legacy_format(parsed_data)
                
            except Exception as aggressive_error:
                print(f"[JSON PARSE] ❌ Aggressive fix also failed: {str(aggressive_error)}")
                print(f"[JSON PARSE] 🚨 FALLBACK PATH 3 - JSON decode error fallback triggered!")
                print(f"[JSON PARSE] This will result in 'Default recommendation due to parsing error'")
            
            # Create honest fallback indicating parsing failure
            enhanced_fallback = fallback_data.copy()
            enhanced_fallback["layer2_context_analysis"] = [
                {
                    "context_id": i,
                    "context_type": context_items[i].get('type', 'unknown'),  # Use internal key format
                    "title": f"Analysis Unavailable: {context_items[i].get('type', 'Unknown').replace('_', ' ').title()}",
                    "severity_score": 30,  # Conservative low score due to parsing failure
                    "risk_level": "Low",  # All marked as low due to uncertainty
                    "explanation": {
                        "beginner": f"GPT analysis for {context_items[i].get('type', 'unknown')} failed due to parsing error. Manual review needed.",
                        "intermediate": f"JSON parsing failed for {context_items[i].get('type', 'unknown')} analysis. Technical assessment required.",
                        "advanced": f"System error: JSON response parsing failed for {context_items[i].get('type', 'unknown')}. Raw GPT analysis unavailable."
                    },
                    "business_impact": f"Impact assessment for {context_items[i].get('type', 'unknown')} unavailable due to parsing failure. Manual evaluation required.",
                    "technical_details": f"Technical analysis for {context_items[i].get('type', 'unknown')} could not be parsed from GPT response. System requires debugging.",
                    "action_required": f"Manual review required for {context_items[i].get('type', 'unknown')} - automated analysis failed due to JSON parsing error."
                }
                for i in range(len(context_items))
            ]
            
            # Update layer1 stats to honestly reflect parsing failure
            enhanced_fallback["layer1_severity_statistics"] = {
                "high_count": 0,
                "medium_count": 0,
                "low_count": len(context_items),  # All items marked as low due to parsing failure
                "overall_risk_level": "Unknown",
                "confidence_score": 0.1,  # Very low confidence due to parsing failure
                "summary": f"JSON parsing failed after GPT analysis. {len(context_items)} items require manual review - severity assessment unavailable."
            }
            
            # Update layer3 with comprehensive joint analysis reflecting systematic assessment
            issue_types = [item.get('type', 'unknown').replace('_', ' ') for item in context_items]
            unique_types = list(set(issue_types))
            
            enhanced_fallback["layer3_joint_analysis"] = {
                "overall_assessment": f"SYSTEM ERROR: JSON parsing failed after GPT analysis completed. {len(context_items)} detected issues ({', '.join(unique_types[:3])}) require manual assessment as automated joint analysis is unavailable. The GPT response could not be parsed due to JSON formatting errors, preventing systematic evaluation of compound risks and failure pathways. Manual review is required to assess primary-to-secondary dataset inference risks and determine appropriate adaptation strategies. This is a technical system issue, not a reflection of the actual risk severity of the detected issues."
            }
            
            print(f"[GPT SEVERITY] Created enhanced fallback with {len(enhanced_fallback['layer2_context_analysis'])} layer2 items")
            
            enhanced_legacy = self._create_legacy_format_from_fallback(enhanced_fallback)
            print(f"[GPT SEVERITY] Enhanced legacy format has comprehensive_data: {'comprehensive_data' in enhanced_legacy}")
            
            return enhanced_legacy
        except Exception as e:
            print(f"[GPT SEVERITY] Unexpected error parsing response: {str(e)}")
            print(f"[GPT SEVERITY] FALLBACK PATH 4 - Unexpected error fallback with {len(fallback_data['layer2_context_analysis'])} layer2 items")
            fallback_result = self._create_legacy_format_from_fallback(fallback_data)
            print(f"[GPT SEVERITY] FALLBACK PATH 4 - Result has comprehensive_data: {'comprehensive_data' in fallback_result}")
            return fallback_result
    
    def _fix_common_json_issues(self, json_text: str) -> str:
        """Fix common JSON syntax issues in GPT responses with conservative approach."""
        import re
        
        print(f"[JSON FIX] Starting conservative JSON repair...")
        original_text = json_text
        
        # 1. Clean control characters only (very safe)
        json_text = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', json_text)
        
        # 2. Fix ONLY the specific issue: unescaped quotes WITHIN string values
        # Pattern: "key": "value with "inner" quotes" -> "key": "value with \"inner\" quotes"
        # This targets the exact problem from the logs
        def fix_unescaped_inner_quotes(text):
            """Fix unescaped quotes specifically within JSON string values."""
            result = []
            i = 0
            
            while i < len(text):
                # Look for JSON key-value pattern: "key": "
                if i < len(text) - 10:  # Need some lookahead
                    # Match pattern: "word": "
                    key_match = re.match(r'"[^"]+"\s*:\s*"', text[i:])
                    if key_match:
                        # Found start of a string value
                        key_part = key_match.group(0)
                        result.append(key_part)
                        i += len(key_part)
                        
                        # Now collect the string value until we find the REAL end quote
                        value_chars = []
                        in_string_value = True
                        
                        while i < len(text) and in_string_value:
                            char = text[i]
                            
                            if char == '"':
                                # Found a quote - is this the end or an inner quote?
                                # Look ahead to see what follows
                                remaining = text[i+1:i+20].strip()
                                
                                # If followed by comma, brace, or bracket -> it's the end
                                if remaining.startswith(',') or remaining.startswith('}') or remaining.startswith(']'):
                                    # This is the end quote
                                    value_chars.append('"')
                                    in_string_value = False
                                else:
                                    # This is an inner quote - escape it
                                    value_chars.append('\\"')
                                    print(f"[JSON FIX] Escaped inner quote at position {i}")
                            else:
                                value_chars.append(char)
                            
                            i += 1
                        
                        result.extend(value_chars)
                        continue
                
                # Not in a string value pattern, just copy the character
                result.append(text[i])
                i += 1
            
            return ''.join(result)
        
        # Apply the targeted fix
        json_text = fix_unescaped_inner_quotes(json_text)
        
        # 3. Fix only critical structural issues (very conservative)
        # Remove trailing commas (safe)
        json_text = re.sub(r',(\s*[}\]])', r'\1', json_text)
        
        # Fix missing commas between objects/arrays (safe)
        json_text = re.sub(r'}\s*{', r'}, {', json_text)
        json_text = re.sub(r']\s*\[', r'], [', json_text)
        
        # Only fix clearly unquoted keys (very conservative pattern)
        json_text = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', json_text)
        
        if json_text != original_text:
            print(f"[JSON FIX] Applied conservative fixes: {len(original_text)} -> {len(json_text)} chars")
        else:
            print(f"[JSON FIX] No fixes needed")
        
        return json_text
    
    def _aggressive_json_fix(self, json_text: str) -> str:
        """More aggressive but still targeted JSON fixing for severely malformed responses."""
        import re
        
        print(f"[AGGRESSIVE FIX] Input length: {len(json_text)} characters")
        
        # DON'T apply basic fixes again - they might have caused the problem
        # Just apply targeted aggressive fixes
        
        # Remove problematic control characters
        json_text = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', json_text)
        
        # Targeted fix: Replace internal quotes with single quotes
        # This is more drastic but safer than trying to escape them
        def replace_inner_quotes_with_singles(text):
            """Replace inner quotes with single quotes - more reliable than escaping."""
            result = []
            i = 0
            
            while i < len(text):
                # Look for string value patterns
                if i < len(text) - 10:
                    # Match: "key": "
                    key_match = re.match(r'"[^"]+"\s*:\s*"', text[i:])
                    if key_match:
                        # Found start of a string value
                        key_part = key_match.group(0)
                        result.append(key_part)
                        i += len(key_part)
                        
                        # Collect the value and replace inner quotes with singles
                        value_chars = []
                        found_end = False
                        
                        while i < len(text) and not found_end:
                            char = text[i]
                            
                            if char == '"':
                                # Check if this could be the end
                                remaining = text[i+1:i+20].strip()
                                if remaining.startswith(',') or remaining.startswith('}') or remaining.startswith(']') or remaining == '':
                                    # This is the end quote
                                    value_chars.append('"')
                                    found_end = True
                                else:
                                    # Replace inner quote with single quote
                                    value_chars.append("'")
                                    print(f"[AGGRESSIVE FIX] Replaced inner quote with single quote at position {i}")
                            else:
                                value_chars.append(char)
                            
                            i += 1
                        
                        result.extend(value_chars)
                        continue
                
                # Not in a string pattern
                result.append(text[i])
                i += 1
            
            return ''.join(result)
        
        json_text = replace_inner_quotes_with_singles(json_text)
        
        # Fix common GPT output issues
        json_text = re.sub(r'"\s*:\s*"([^"]*)"([^,}])', r'": "\1",', json_text)  # Missing commas
        json_text = re.sub(r'}\s*"([^"]+)":', r'}, "\1":', json_text)  # Missing commas after objects
        json_text = re.sub(r']\s*"([^"]+)":', r'], "\1":', json_text)  # Missing commas after arrays
        
        # Handle long responses that may be truncated
        if len(json_text) > 8000:  # If response is very long
            print(f"[AGGRESSIVE FIX] Long response detected, attempting smart truncation")
            
            # Try to find the end of layer2_context_analysis array
            layer2_end = json_text.find('"layer3_joint_analysis"')
            if layer2_end > 0:
                # Take everything up to layer3 and try to close it properly
                truncated = json_text[:layer2_end-1]  # Remove comma before layer3
                
                # Count unclosed structures
                open_braces = truncated.count('{') - truncated.count('}')
                open_brackets = truncated.count('[') - truncated.count(']')
                
                # Close structures properly
                if truncated.rstrip().endswith(','):
                    truncated = truncated.rstrip()[:-1]  # Remove trailing comma
                
                truncated += ']' * max(0, open_brackets)  # Close arrays
                truncated += '}' * max(0, open_braces)    # Close objects
                
                print(f"[AGGRESSIVE FIX] Smart truncation: {len(truncated)} chars")
                json_text = truncated
        
        # Handle incomplete JSON structure
        open_braces = json_text.count('{') - json_text.count('}')
        open_brackets = json_text.count('[') - json_text.count(']')
        
        if open_braces > 0 or open_brackets > 0:
            print(f"[AGGRESSIVE FIX] Closing {open_brackets} brackets and {open_braces} braces")
            
            # Remove trailing comma if exists
            json_text = re.sub(r',\s*$', '', json_text.rstrip())
            
            # Close all open structures
            json_text += ']' * max(0, open_brackets)
            json_text += '}' * max(0, open_braces)
        
        # Ensure valid JSON structure
        json_text = json_text.strip()
        if not json_text.startswith('{'):
            json_text = '{' + json_text
        if not json_text.endswith('}'):
            json_text = json_text + '}'
        
        # Final validation: check for basic structure
        if '"layer1_severity_statistics"' not in json_text:
            print(f"[AGGRESSIVE FIX] Missing layer1, creating minimal structure")
            json_text = '{"layer1_severity_statistics":{"high_count":1,"medium_count":1,"low_count":1,"overall_risk_level":"Medium","confidence_score":0.5,"summary":"Parsing error occurred"}}'
        
        print(f"[AGGRESSIVE FIX] Output length: {len(json_text)} characters")
        return json_text
    
    def _validate_and_fix_titles(self, layer2_data: List[Dict[str, Any]]) -> None:
        """
        Validate and fix both context types and titles in layer2 data to match expected format.
        
        Args:
            layer2_data: List of layer2 context analysis items to validate
        """
        # Get original context items from global state for reference
        original_context_items = []
        try:
            from UI.functions.global_vars import global_vars
            
            # Try multiple possible sources for context items
            context_sources = [
                getattr(global_vars, 'current_explain_context', None),
                getattr(global_vars, 'explain_context_data', None), 
                getattr(global_vars, 'chat_context_data', None),
                getattr(global_vars, 'context_items', None)
            ]
            
            for source in context_sources:
                if isinstance(source, list) and len(source) > 0:
                    original_context_items = source
                    print(f"[VALIDATION] ✅ Found {len(original_context_items)} context items from global state")
                    break
                    
            if not original_context_items:
                print(f"[VALIDATION] ⚠️ No context items found in global state, trying callback context...")
                # Try to get context from Dash callback context
                try:
                    import dash
                    ctx = dash.callback_context
                    if hasattr(ctx, 'states') and ctx.states:
                        for state in ctx.states:
                            if 'context-data' in state.get('id', ''):
                                state_value = state.get('value', [])
                                if isinstance(state_value, list) and len(state_value) > 0:
                                    original_context_items = state_value
                                    print(f"[VALIDATION] ✅ Found {len(original_context_items)} items from callback context")
                                    break
                except Exception as e:
                    print(f"[VALIDATION] Could not access callback context: {e}")

        except Exception as e:
            print(f"[VALIDATION] ⚠️ Could not access original context items: {e}")
        
        print(f"[VALIDATION] 🔍 Validating {len(layer2_data)} GPT items against {len(original_context_items)} original items")
        
        for item in layer2_data:
            context_id = item.get('context_id', 0)
            gpt_context_type = item.get('context_type', '')
            current_title = item.get('title', '')
            
            # Get corresponding original context item if available
            original_item = None
            expected_context_type = None
            if context_id < len(original_context_items):
                original_item = original_context_items[context_id]
                expected_context_type = original_item.get('type', '')
                
                print(f"[VALIDATION] Item {context_id}: GPT='{gpt_context_type}' vs Expected='{expected_context_type}'")
                
                # CRITICAL: Validate and fix context type mismatch
                if gpt_context_type != expected_context_type:
                    print(f"[VALIDATION] ⚠️ CONTEXT TYPE MISMATCH for context_id {context_id}")
                    print(f"[VALIDATION] GPT returned: '{gpt_context_type}'")
                    print(f"[VALIDATION] Expected: '{expected_context_type}'")
                    
                    # Fix the context type
                    item['context_type'] = expected_context_type
                    print(f"[VALIDATION] ✅ Fixed context_type for context_id {context_id}: '{gpt_context_type}' → '{expected_context_type}'")
                    
                    # Update gpt_context_type for title generation
                    gpt_context_type = expected_context_type
                else:
                    print(f"[VALIDATION] ✅ Valid context_type for context_id {context_id}: '{gpt_context_type}'")
            else:
                print(f"[VALIDATION] ⚠️ No original item found for context_id {context_id}")
                expected_context_type = gpt_context_type  # Use GPT's type as fallback
            
            # Generate expected title based on correct context type and available data
            expected_title = self._generate_expected_title(gpt_context_type, original_item, current_title)
            
            # Validate current title against expected format
            if not self._is_title_format_valid(current_title, gpt_context_type, expected_title):
                print(f"[VALIDATION] ⚠️ Invalid title for context_id {context_id}")
                print(f"[VALIDATION] Current: '{current_title}'")
                print(f"[VALIDATION] Expected: '{expected_title}'")
                
                # Fix the title
                item['title'] = expected_title
                print(f"[VALIDATION] ✅ Fixed title for context_id {context_id}")
            else:
                print(f"[VALIDATION] ✅ Valid title for context_id {context_id}: '{current_title}'")
    
    def _generate_expected_title(self, context_type: str, original_item: Dict[str, Any] = None, current_title: str = '') -> str:
        """
        Generate expected title based on context type and original item data.
        
        Args:
            context_type: The context type (drift_analysis, conditional_distribution, etc.)
            original_item: Original context item data if available
            current_title: Current title from GPT (may contain useful attribute names)
            
        Returns:
            str: Expected title format
        """
        if context_type == "drift_analysis":
            if original_item:
                attr_name = original_item.get('attribute_name', 'Unknown')
            else:
                # Try to extract attribute name from current title
                attr_name = self._extract_attribute_from_title(current_title, "drift")
            return f"Drift Analysis: {attr_name}"
            
        elif context_type == "distribution_comparison":
            if original_item:
                # Try to extract attribute from cell_info or other fields
                attr_name = self._extract_attribute_from_context_item(original_item)
            else:
                attr_name = self._extract_attribute_from_title(current_title, "distribution")
            return f"Distribution Comparison: {attr_name}"
            
        elif context_type == "conditional_distribution":
            if original_item:
                target_attr = original_item.get('target_attribute', 'target')
                target_value = original_item.get('target_value', 'unknown')
                return f"Conditional Analysis: {target_attr} = {target_value}"
            else:
                # Try to parse from current title or create generic
                if "=" in current_title:
                    # Try to reformat existing conditional analysis
                    parts = current_title.split("=")
                    if len(parts) >= 2:
                        attr_part = parts[0].split(":")[-1].strip() if ":" in parts[0] else parts[0].strip()
                        value_part = parts[1].split()[0].strip() if parts[1] else "unknown"
                        return f"Conditional Analysis: {attr_part} = {value_part}"
                return f"Conditional Analysis: target = unknown"
                
        elif context_type == "target_distribution":
            if original_item:
                target_attr = original_item.get('target_attribute', 'target')
            else:
                attr_name = self._extract_attribute_from_title(current_title, "target")
            return f"Target Distribution Analysis: {target_attr if original_item else attr_name}"
            
        elif context_type == "metric":
            if original_item:
                metric_name = original_item.get('metric_name', 'Unknown')
            else:
                metric_name = self._extract_attribute_from_title(current_title, "metric")
            return f"Metric Analysis: {metric_name}"
            
        else:
            return current_title  # Keep as is for unknown types
    
    def _extract_attribute_from_context_item(self, context_item: Dict[str, Any]) -> str:
        """Extract attribute name from context item using multiple strategies."""
        # Try direct attribute fields
        for field in ['attribute_name', 'target_attribute', 'compare_attribute', 'metric_name']:
            value = context_item.get(field)
            if value and value != 'Unknown':
                return value
        
        # Try to extract from cell_info
        cell_info = context_item.get('cell_info', '')
        if cell_info:
            lines = cell_info.split('\n')
            for line in lines:
                if "Column:" in line:
                    parts = line.split("Column:")
                    if len(parts) > 1:
                        column_part = parts[1].strip()
                        if ", Value:" in column_part:
                            return column_part.split(", Value:")[0].strip()
                        else:
                            return column_part.strip()
        
        # Try to extract from summary_text
        summary_text = context_item.get('summary_text', '')
        if summary_text:
            # Look for common patterns in summary text
            import re
            attr_patterns = [
                r"attribute:\s*([^\n,]+)",
                r"column:\s*([^\n,]+)",
                r"variable:\s*([^\n,]+)",
                r"feature:\s*([^\n,]+)"
            ]
            for pattern in attr_patterns:
                match = re.search(pattern, summary_text, re.IGNORECASE)
                if match:
                    return match.group(1).strip()
        
        return "Unknown"
    
    def _extract_attribute_from_title(self, title: str, analysis_type: str) -> str:
        """Extract attribute name from current title."""
        if not title:
            return "Unknown"
        
        # Try to extract attribute name after colon or common patterns
        if ":" in title:
            parts = title.split(":", 1)
            if len(parts) > 1:
                attr_part = parts[1].strip()
                # Remove common prefixes/suffixes
                attr_part = attr_part.replace("Analysis", "").replace("Shift", "").replace("in", "").strip()
                if attr_part and len(attr_part) < 50:  # Reasonable attribute name length
                    return attr_part
        
        # For conditional analysis, try to extract from patterns
        if analysis_type == "conditional" and "=" in title:
            parts = title.split("=")[0]
            attr_name = parts.split()[-1].strip() if parts else "target"
            return attr_name
        
        # Look for common attribute-like words in the title
        import re
        words = re.findall(r'\b[A-Za-z][A-Za-z0-9_]*\b', title)
        for word in words:
            if word.lower() not in ['analysis', 'distribution', 'drift', 'conditional', 'target', 'comparison', 'metric', 'shift', 'the', 'of', 'in', 'and', 'or']:
                return word
        
        return "Unknown"
    
    def _is_title_format_valid(self, current_title: str, context_type: str, expected_title: str) -> bool:
        """
        Check if current title follows the expected format.
        
        Args:
            current_title: Current title from GPT
            context_type: Context type
            expected_title: Expected title format
            
        Returns:
            bool: True if title format is valid
        """
        if not current_title:
            return False
        
        # Define expected prefixes for each context type
        expected_prefixes = {
            "drift_analysis": "Drift Analysis:",
            "distribution_comparison": "Distribution Comparison:",
            "conditional_distribution": "Conditional Analysis:",
            "target_distribution": "Target Distribution Analysis:",
            "metric": "Metric Analysis:"
        }
        
        expected_prefix = expected_prefixes.get(context_type, "")
        if not expected_prefix:
            return True  # Unknown type, accept any title
        
        # Check if title starts with expected prefix
        if not current_title.startswith(expected_prefix):
            return False
        
        # For conditional distribution, also check for "=" pattern
        if context_type == "conditional_distribution":
            if "=" not in current_title:
                return False
        
        return True
    
    def _convert_comprehensive_to_legacy_format(self, comprehensive_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert new comprehensive format to legacy format for backward compatibility."""
        layer1 = comprehensive_data.get('layer1_severity_statistics', {})
        layer2 = comprehensive_data.get('layer2_context_analysis', [])
        layer4 = comprehensive_data.get('layer4_strategy_selection', {})
        
        # Create display name to internal key mapping for context types
        context_type_mapping = {
            "Drift Analysis": "drift_analysis",
            "Distribution Comparison": "distribution_comparison", 
            "Conditional Distribution": "conditional_distribution",
            "Target Distribution": "target_distribution",
            "Metric Analysis": "metric",
            "Metric": "metric",
            "Distribution": "distribution_comparison",
            "Conditional": "conditional_distribution",
            "Target": "target_distribution"
        }
        
        # Normalize context_type in layer2 data to fix modal matching
        for item in layer2:
            current_type = item.get('context_type', '')
            if current_type in context_type_mapping:
                normalized_type = context_type_mapping[current_type]
                item['context_type'] = normalized_type
                print(f"[TYPE NORMALIZATION] Fixed: '{current_type}' → '{normalized_type}'")
            elif current_type and current_type not in ['drift_analysis', 'distribution_comparison', 
                                                      'conditional_distribution', 'target_distribution', 'metric']:
                print(f"[TYPE NORMALIZATION] ⚠️ Unknown type '{current_type}' - keeping as is")
        
        # Validate and fix titles based on expected format
        self._validate_and_fix_titles(layer2)
        
        # Build individual scores from layer2 using consistent High/Medium/Low
        individual_scores = {}
        for item in layer2:
            context_id = item.get('context_id', f"Item {len(individual_scores) + 1}")
            individual_scores[f"Issue {context_id}"] = {
                "severity_score": item.get('severity_score', 50),
                "severity_level": item.get('risk_level', 'Medium').lower(),
                "title": item.get('title', 'Unknown Issue'),
                "explanation": item.get('explanation', {}),
                "business_impact": item.get('business_impact', ''),
                "technical_details": item.get('technical_details', ''),
                "action_required": item.get('action_required', '')
            }
        
        # Build overall assessment using consistent High/Medium/Low terminology
        overall_assessment = {
            "high_count": layer1.get('high_count', 0),
            "medium_count": layer1.get('medium_count', 0),
            "low_count": layer1.get('low_count', 0),
            "recommendation": layer4.get('recommended_strategy', 'monitor'),
            "overall_risk_level": layer1.get('overall_risk_level', 'Medium'),
            "confidence_score": layer1.get('confidence_score', 0.5),
            "summary": layer1.get('summary', '')
        }
        
        return {
            "individual_scores": individual_scores,
            "overall_assessment": overall_assessment,
            "comprehensive_data": comprehensive_data  # Keep full data for new UI
        }
    
    def _create_legacy_format_from_fallback(self, fallback_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create legacy format from fallback data, ensuring all layers are properly populated."""
        # Ensure layer3 has proper content if it's empty
        layer3_data = fallback_data.get('layer3_joint_analysis', {})
        if not layer3_data.get('overall_assessment'):
            # Get context items info for generating layer3 content
            layer2_data = fallback_data.get('layer2_context_analysis', [])
            context_count = len(layer2_data)
            
            # Extract issue types from layer2 data
            issue_types = []
            for item in layer2_data:
                context_type = item.get('context_type', item.get('type', 'unknown'))
                issue_types.append(context_type.replace('_', ' '))
            unique_types = list(set(issue_types))
            
            # Generate comprehensive layer3 content
            fallback_data['layer3_joint_analysis'] = {
                "overall_assessment": f"Systematic assessment of {context_count} detected issues ({', '.join(unique_types[:3])}) reveals significant compound risks for direct model deployment from primary to secondary dataset. These issues create multiple failure pathways: distribution shifts may cause prediction drift toward primary dataset patterns, correlation changes could invalidate feature relationships learned during training, and missing data patterns may trigger unexpected model behaviors during inference. The cumulative effect threatens target attribute prediction reliability, potentially causing systematic errors when the model encounters secondary dataset characteristics it wasn't trained to handle. Without adaptation strategies, model performance degradation is highly likely, particularly affecting prediction accuracy and calibration across different population segments present in the secondary dataset."
            }
        
        return self._convert_comprehensive_to_legacy_format(fallback_data)


# Global instance for reuse
gpt_severity_analyzer = GPTSeverityAnalyzer()
