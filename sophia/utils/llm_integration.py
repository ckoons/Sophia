"""
Tekton LLM Client Integration for Sophia

This module provides integration with the tekton-llm-client for standardized
access to LLM capabilities following the Tekton shared architecture patterns.
"""

import os
import json
import logging
import asyncio
from typing import Dict, List, Any, Optional, Union, Callable, AsyncIterator

# Import tekton-llm-client
from tekton_llm_client import Client as TektonLLMClient
from tekton_llm_client.models import (
    ChatMessage, 
    ChatCompletionOptions,
    StreamingChunk
)
from tekton_llm_client.adapters import FallbackAdapter

# Import Sophia utilities
from sophia.utils.tekton_utils import get_config, get_logger

# Set up logging
logger = get_logger("sophia.utils.llm_integration")

# System prompts for different task types
SYSTEM_PROMPTS = {
    "default": "You are Sophia, Tekton's machine learning and continuous improvement component. "
              "You analyze metrics, patterns, and performance data to provide insights and recommendations.",
              
    "analysis": "You are Sophia's Analysis Engine. Examine the following metrics and identify patterns, "
                "trends, anomalies, and potential optimization opportunities. Format your analysis "
                "in a structured, factual manner.",
                
    "recommendation": "You are Sophia's Recommendation System. Generate specific, actionable recommendations "
                      "based on the provided analysis. Focus on concrete improvements with clear impact "
                      "and implementation steps.",
                     
    "experiment": "You are Sophia's Experiment Design System. Design experiments to test hypotheses "
                 "about component performance and optimization. Focus on clear methodology, measurable "
                 "outcomes, and controlled variables.",
                 
    "explanation": "You are Sophia's Explanation System. Translate technical metrics and analysis "
                  "into clear, concise explanations accessible to users with varying technical backgrounds.",
                  
    "intelligence": "You are Sophia's Intelligence Measurement System. Analyze component capabilities "
                   "across multiple intelligence dimensions and provide objective assessments."
}

# Prompt templates for different tasks
PROMPT_TEMPLATES = {
    "metrics_analysis": "Analyze the following metrics data from {component_id}:\n\n{metrics_json}\n\n"
                       "Identify patterns, anomalies, and potential optimization opportunities.",
    
    "experiment_design": "Design an experiment to test the hypothesis: {hypothesis}\n\n"
                        "Available components: {components_list}\n"
                        "Recent metrics: {metrics_summary}",
    
    "recommendation_generation": "Based on the following analysis:\n\n{analysis_summary}\n\n"
                               "Generate {count} specific recommendations to improve {target}.",
                               
    "natural_language_query": "Context information:\n{context_json}\n\nUser query: {query}",
    
    "intelligence_assessment": "Assess the intelligence capabilities of {component_id} "
                              "based on the following data:\n\n{data_json}\n\n"
                              "Provide ratings for each intelligence dimension with justification."
}

# Model configuration for different tasks
MODEL_CONFIGURATION = {
    "analysis": {
        "preferred_model": "claude-3-opus-20240229",
        "fallback_model": "claude-3-haiku-20240307",
        "local_fallback": "mistral-7b-instruct",
        "temperature": 0.2,
        "max_tokens": 2000
    },
    "recommendation": {
        "preferred_model": "claude-3-sonnet-20240229",
        "fallback_model": "claude-3-haiku-20240307",
        "local_fallback": "llama-3-8b",
        "temperature": 0.4,
        "max_tokens": 1000
    },
    "experiment": {
        "preferred_model": "claude-3-sonnet-20240229",
        "fallback_model": "claude-3-haiku-20240307",
        "local_fallback": "mistral-7b-instruct",
        "temperature": 0.3,
        "max_tokens": 1500
    },
    "explanation": {
        "preferred_model": "claude-3-sonnet-20240229",
        "fallback_model": "gpt-3.5-turbo",
        "local_fallback": "phi-2",
        "temperature": 0.7,
        "max_tokens": 1500
    },
    "intelligence": {
        "preferred_model": "claude-3-opus-20240229",
        "fallback_model": "claude-3-sonnet-20240229",
        "local_fallback": "mistral-7b-instruct",
        "temperature": 0.3,
        "max_tokens": 2000
    }
}

class SophiaLLMIntegration:
    """
    LLM integration for Sophia using the tekton-llm-client.
    
    This class provides methods for all LLM-related tasks in Sophia,
    ensuring standardized access to LLM capabilities with proper
    error handling and retries.
    """
    
    def __init__(self, component_id: str = "sophia"):
        """
        Initialize the Sophia LLM integration.
        
        Args:
            component_id: Component ID for tracking
        """
        self.component_id = component_id
        self.is_initialized = False
        self.clients = {}  # Task-specific clients
        
        # Get default URL and provider from config
        self.base_url = get_config("TEKTON_LLM_URL", "http://localhost:8001")
        self.default_provider = get_config("TEKTON_LLM_PROVIDER", "anthropic")
        self.default_model = get_config("TEKTON_LLM_MODEL", "claude-3-sonnet-20240229")
        
        logger.info(f"Initialized Sophia LLM Integration with URL {self.base_url}")
        
    async def initialize(self) -> bool:
        """
        Initialize the LLM clients.
        
        Returns:
            True if initialization was successful
        """
        if self.is_initialized:
            return True
            
        logger.info("Initializing Sophia LLM Integration...")
        
        try:
            # Initialize task-specific clients
            for task_type, config in MODEL_CONFIGURATION.items():
                client = TektonLLMClient(
                    component_id=f"{self.component_id}-{task_type}",
                    rhetor_url=self.base_url,
                    provider_id=self.default_provider,
                    model_id=config["preferred_model"],
                    timeout=60,
                    max_retries=3,
                    use_fallback=True
                )
                
                # Initialize the client to verify connection
                await client.initialize()
                
                self.clients[task_type] = client
                logger.info(f"Initialized LLM client for {task_type}")
                
            self.is_initialized = True
            logger.info("Sophia LLM Integration initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Sophia LLM Integration: {e}")
            return False
            
    async def shutdown(self) -> None:
        """
        Shut down all LLM clients.
        """
        for task_type, client in self.clients.items():
            try:
                await client.shutdown()
                logger.info(f"Shut down LLM client for {task_type}")
            except Exception as e:
                logger.error(f"Error shutting down LLM client for {task_type}: {e}")
                
        self.is_initialized = False
        logger.info("Sophia LLM Integration shut down")
        
    async def get_client(self, task_type: Optional[str] = None) -> TektonLLMClient:
        """
        Get an LLM client for a specific task type.
        
        Args:
            task_type: Type of task (analysis, recommendation, etc.)
            
        Returns:
            TektonLLMClient for the task
        """
        if not self.is_initialized:
            await self.initialize()
            
        # Use the task-specific client if available, or default to analysis
        client = self.clients.get(task_type) if task_type in self.clients else self.clients.get("analysis")
        
        # If no clients are available, create one for the requested task
        if client is None:
            config = MODEL_CONFIGURATION.get(task_type, MODEL_CONFIGURATION["analysis"])
            
            client = TektonLLMClient(
                component_id=f"{self.component_id}-{task_type or 'default'}",
                rhetor_url=self.base_url,
                provider_id=self.default_provider,
                model_id=config["preferred_model"],
                timeout=60,
                max_retries=3,
                use_fallback=True
            )
            
            # Initialize the client
            await client.initialize()
            
            # Store the client for future use
            if task_type:
                self.clients[task_type] = client
                
        return client
        
    async def analyze_metrics(
        self, 
        metrics_data: Dict[str, Any], 
        component_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze metrics data using LLM.
        
        Args:
            metrics_data: Metrics data to analyze
            component_id: Optional component ID for context
            
        Returns:
            Analysis results
        """
        client = await self.get_client("analysis")
        
        try:
            # Format prompt using template
            prompt = PROMPT_TEMPLATES["metrics_analysis"].format(
                component_id=component_id or "all components",
                metrics_json=json.dumps(metrics_data, indent=2)
            )
            
            # Create system prompt
            system_prompt = SYSTEM_PROMPTS["analysis"]
            
            # Generate response
            response = await client.generate_text(
                prompt=prompt,
                system_prompt=system_prompt,
                options={
                    "temperature": MODEL_CONFIGURATION["analysis"]["temperature"],
                    "max_tokens": MODEL_CONFIGURATION["analysis"]["max_tokens"]
                }
            )
            
            # Extract and structure the response
            return {
                "analysis": response.content,
                "structured": await self._extract_structured_analysis(response.content),
                "component_id": component_id,
                "model": response.model,
                "provider": response.provider
            }
            
        except Exception as e:
            logger.error(f"Error analyzing metrics with LLM: {e}")
            return {
                "error": str(e),
                "component_id": component_id,
                "fallback": "Unable to analyze metrics due to LLM service error"
            }
            
    async def generate_recommendations(
        self, 
        analysis_results: Dict[str, Any], 
        target_component: Optional[str] = None,
        count: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Generate recommendations based on analysis results.
        
        Args:
            analysis_results: Analysis data to base recommendations on
            target_component: Optional target component for recommendations
            count: Number of recommendations to generate
            
        Returns:
            List of recommendation objects
        """
        client = await self.get_client("recommendation")
        
        try:
            # Create summary of analysis results
            analysis_summary = (
                analysis_results["analysis"] 
                if "analysis" in analysis_results 
                else json.dumps(analysis_results, indent=2)
            )
            
            # Format prompt using template
            prompt = PROMPT_TEMPLATES["recommendation_generation"].format(
                analysis_summary=analysis_summary,
                count=count,
                target=target_component or "the Tekton ecosystem"
            )
            
            # Create system prompt with formatting instructions
            system_prompt = (
                SYSTEM_PROMPTS["recommendation"] + "\n\n" + 
                "Format your response as a JSON array of recommendation objects with " +
                "'title', 'description', 'impact', 'effort', and 'implementation_steps' fields."
            )
            
            # Generate response
            response = await client.generate_text(
                prompt=prompt,
                system_prompt=system_prompt,
                options={
                    "temperature": MODEL_CONFIGURATION["recommendation"]["temperature"],
                    "max_tokens": MODEL_CONFIGURATION["recommendation"]["max_tokens"]
                }
            )
            
            # Extract recommendations
            return await self._extract_json_recommendations(response.content)
            
        except Exception as e:
            logger.error(f"Error generating recommendations with LLM: {e}")
            return [{
                "error": str(e),
                "title": "Recommendation Error",
                "description": "Unable to generate recommendations due to LLM service error"
            }]
            
    async def design_experiment(
        self, 
        hypothesis: str, 
        available_components: Optional[List[str]] = None,
        metrics_summary: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Design an experiment to test a hypothesis.
        
        Args:
            hypothesis: The hypothesis to test
            available_components: List of components available for the experiment
            metrics_summary: Summary of recent metrics
            
        Returns:
            Experiment design
        """
        client = await self.get_client("experiment")
        
        try:
            # Format components list
            components_list = (
                ", ".join(available_components) 
                if available_components 
                else "all available components"
            )
            
            # Format metrics summary
            metrics_json = (
                json.dumps(metrics_summary, indent=2)
                if metrics_summary
                else "No metrics summary provided"
            )
            
            # Format prompt using template
            prompt = PROMPT_TEMPLATES["experiment_design"].format(
                hypothesis=hypothesis,
                components_list=components_list,
                metrics_summary=metrics_json
            )
            
            # Create system prompt with formatting instructions
            system_prompt = (
                SYSTEM_PROMPTS["experiment"] + "\n\n" + 
                "Format your response as a JSON object with 'title', 'hypothesis', " +
                "'methodology', 'components', 'metrics', 'variables', 'control_condition', " +
                "'test_condition', and 'success_criteria' fields."
            )
            
            # Generate response
            response = await client.generate_text(
                prompt=prompt,
                system_prompt=system_prompt,
                options={
                    "temperature": MODEL_CONFIGURATION["experiment"]["temperature"],
                    "max_tokens": MODEL_CONFIGURATION["experiment"]["max_tokens"]
                }
            )
            
            # Extract experiment design
            return await self._extract_json_experiment(response.content)
            
        except Exception as e:
            logger.error(f"Error designing experiment with LLM: {e}")
            return {
                "error": str(e),
                "title": "Experiment Design Error",
                "hypothesis": hypothesis,
                "message": "Unable to design experiment due to LLM service error"
            }
            
    async def explain_analysis(
        self, 
        analysis_data: Dict[str, Any],
        audience: str = "technical"
    ) -> str:
        """
        Explain analysis results in human-readable form.
        
        Args:
            analysis_data: Analysis data to explain
            audience: Target audience ("technical", "executive", "general")
            
        Returns:
            Human-readable explanation
        """
        client = await self.get_client("explanation")
        
        try:
            # Create system prompt with audience targeting
            system_prompt = SYSTEM_PROMPTS["explanation"] + f"\n\nTarget audience: {audience}"
            
            # Create prompt with analysis data
            prompt = f"Explain the following analysis results in a way appropriate for a {audience} "
                   f"audience:\n\n{json.dumps(analysis_data, indent=2)}"
            
            # Generate response
            response = await client.generate_text(
                prompt=prompt,
                system_prompt=system_prompt,
                options={
                    "temperature": MODEL_CONFIGURATION["explanation"]["temperature"],
                    "max_tokens": MODEL_CONFIGURATION["explanation"]["max_tokens"]
                }
            )
            
            return response.content
            
        except Exception as e:
            logger.error(f"Error explaining analysis with LLM: {e}")
            return f"Unable to generate explanation due to service error: {str(e)}"
            
    async def process_natural_language_query(
        self, 
        query: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a natural language query about metrics or analysis.
        
        Args:
            query: The natural language query
            context: Optional context information
            
        Returns:
            Response to the query
        """
        client = await self.get_client()
        
        try:
            # Create context if not provided
            if context is None:
                context = {"query_time": "current time", "system_state": "normal"}
                
            # Format prompt using template
            prompt = PROMPT_TEMPLATES["natural_language_query"].format(
                context_json=json.dumps(context, indent=2),
                query=query
            )
            
            # Generate response
            response = await client.generate_text(
                prompt=prompt,
                system_prompt=SYSTEM_PROMPTS["default"],
                options={
                    "temperature": 0.5,
                    "max_tokens": 1500
                }
            )
            
            return {
                "query": query,
                "response": response.content,
                "type": "natural_language"
            }
            
        except Exception as e:
            logger.error(f"Error processing natural language query with LLM: {e}")
            return {
                "query": query,
                "response": f"I'm sorry, I couldn't process your query due to a service error: {str(e)}",
                "type": "error"
            }
            
    async def assess_intelligence(
        self,
        component_id: str,
        component_data: Dict[str, Any],
        dimensions: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Assess the intelligence of a component across dimensions.
        
        Args:
            component_id: ID of the component to assess
            component_data: Data about the component's capabilities
            dimensions: Specific dimensions to assess (all if None)
            
        Returns:
            Intelligence assessment with scores and justifications
        """
        client = await self.get_client("intelligence")
        
        try:
            # Add dimensions to the component data if provided
            if dimensions:
                component_data["dimensions_to_assess"] = dimensions
                
            # Format prompt using template
            prompt = PROMPT_TEMPLATES["intelligence_assessment"].format(
                component_id=component_id,
                data_json=json.dumps(component_data, indent=2)
            )
            
            # Create system prompt with instructions
            system_prompt = (
                SYSTEM_PROMPTS["intelligence"] + "\n\n" +
                "Format your response as a JSON object with a 'dimensions' field containing " +
                "ratings and justifications for each intelligence dimension."
            )
            
            # Generate response
            response = await client.generate_text(
                prompt=prompt,
                system_prompt=system_prompt,
                options={
                    "temperature": MODEL_CONFIGURATION["intelligence"]["temperature"],
                    "max_tokens": MODEL_CONFIGURATION["intelligence"]["max_tokens"]
                }
            )
            
            # Extract assessment
            return await self._extract_json_assessment(response.content)
            
        except Exception as e:
            logger.error(f"Error assessing intelligence with LLM: {e}")
            return {
                "error": str(e),
                "component_id": component_id,
                "message": "Unable to assess intelligence due to LLM service error"
            }
            
    async def stream_explanation(
        self, 
        analysis_id: str,
        callback: Callable[[str], None]
    ) -> None:
        """
        Stream an explanation of the analysis, delivering chunks via callback.
        
        Args:
            analysis_id: ID of the analysis to explain
            callback: Function to call with each content chunk
        """
        client = await self.get_client("explanation")
        
        try:
            # Create system prompt
            system_prompt = SYSTEM_PROMPTS["explanation"]
            
            # Create prompt
            prompt = f"Explain analysis {analysis_id} in detail, covering the methods used, "
                   f"findings, and recommendations."
            
            # Start streaming
            async for chunk in client.generate_text(
                prompt=prompt,
                system_prompt=system_prompt,
                streaming=True,
                options={
                    "temperature": MODEL_CONFIGURATION["explanation"]["temperature"],
                    "max_tokens": MODEL_CONFIGURATION["explanation"]["max_tokens"]
                }
            ):
                # Only process text chunks
                if hasattr(chunk, 'chunk') and chunk.chunk:
                    callback(chunk.chunk)
                    
        except Exception as e:
            logger.error(f"Error streaming explanation with LLM: {e}")
            callback(f"\nError: Unable to complete explanation due to service error: {str(e)}")
            
    async def _extract_structured_analysis(self, content: str) -> Dict[str, Any]:
        """
        Extract structured analysis from LLM response.
        
        Args:
            content: Raw LLM response
            
        Returns:
            Structured analysis data
        """
        # Look for JSON content
        try:
            # Check if the response contains JSON (common format)
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = content[start_idx:end_idx]
                return json.loads(json_str)
                
            # If no JSON detected, create a basic structure
            return {
                "text_analysis": content,
                "patterns_detected": await self._extract_key_points(content, "pattern"),
                "anomalies": await self._extract_key_points(content, "anomaly"),
                "insights": await self._extract_key_points(content, "insight")
            }
            
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON from analysis response")
            return {"text_analysis": content}
        except Exception as e:
            logger.error(f"Error extracting structured analysis: {e}")
            return {"text_analysis": content, "error": str(e)}
            
    async def _extract_json_recommendations(self, content: str) -> List[Dict[str, Any]]:
        """
        Extract JSON recommendations from LLM response.
        
        Args:
            content: Raw LLM response
            
        Returns:
            List of recommendation objects
        """
        try:
            # Find JSON array in the content
            start_idx = content.find('[')
            end_idx = content.rfind(']') + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = content[start_idx:end_idx]
                return json.loads(json_str)
            else:
                # Fallback to basic parsing if JSON not found
                return [{
                    "title": "Recommendation", 
                    "description": content,
                    "impact": "unknown",
                    "effort": "unknown"
                }]
                
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON recommendations")
            return [{
                "title": "Recommendation", 
                "description": content,
                "impact": "unknown",
                "effort": "unknown",
                "parsing_error": "Could not parse structured recommendations"
            }]
        except Exception as e:
            logger.error(f"Error extracting JSON recommendations: {e}")
            return [{
                "title": "Error in recommendation parsing", 
                "description": f"Error: {str(e)}",
                "impact": "unknown",
                "effort": "unknown"
            }]
            
    async def _extract_json_experiment(self, content: str) -> Dict[str, Any]:
        """
        Extract JSON experiment design from LLM response.
        
        Args:
            content: Raw LLM response
            
        Returns:
            Experiment design object
        """
        try:
            # Find JSON object in the content
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = content[start_idx:end_idx]
                return json.loads(json_str)
            else:
                # Fallback to basic parsing if JSON not found
                return {
                    "title": "Experiment Design", 
                    "description": content,
                    "type": "unstructured"
                }
                
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON experiment design")
            return {
                "title": "Experiment Design", 
                "description": content,
                "type": "unstructured",
                "parsing_error": "Could not parse structured experiment design"
            }
        except Exception as e:
            logger.error(f"Error extracting JSON experiment design: {e}")
            return {
                "title": "Error in experiment design parsing", 
                "description": f"Error: {str(e)}",
                "type": "error"
            }
            
    async def _extract_json_assessment(self, content: str) -> Dict[str, Any]:
        """
        Extract JSON intelligence assessment from LLM response.
        
        Args:
            content: Raw LLM response
            
        Returns:
            Intelligence assessment object
        """
        try:
            # Find JSON object in the content
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = content[start_idx:end_idx]
                return json.loads(json_str)
            else:
                # Fallback to basic parsing if JSON not found
                return {
                    "dimensions": {},
                    "overall_assessment": content,
                    "type": "unstructured"
                }
                
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON intelligence assessment")
            return {
                "dimensions": {},
                "overall_assessment": content,
                "type": "unstructured",
                "parsing_error": "Could not parse structured intelligence assessment"
            }
        except Exception as e:
            logger.error(f"Error extracting JSON intelligence assessment: {e}")
            return {
                "dimensions": {},
                "overall_assessment": f"Error: {str(e)}",
                "type": "error"
            }
            
    async def _extract_key_points(self, content: str, point_type: str) -> List[str]:
        """
        Extract key points of a specific type from content.
        
        Args:
            content: Content to analyze
            point_type: Type of points to extract (pattern, anomaly, insight)
            
        Returns:
            List of extracted points
        """
        # Simple extraction based on line analysis
        # In a production system, this would use more sophisticated NLP
        points = []
        
        # Convert to lowercase for case-insensitive matching
        content_lower = content.lower()
        point_type_lower = point_type.lower()
        
        # Split into lines and look for relevant content
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            line_lower = line.lower()
            if point_type_lower in line_lower:
                # Found a line with the point type
                points.append(line.strip())
                # Also include the next line if it doesn't have a different point type
                if i+1 < len(lines) and not any(pt in lines[i+1].lower() for pt in ["pattern", "anomaly", "insight"]):
                    points.append(lines[i+1].strip())
                    
        return points
        
# Global singleton instance
_llm_integration = SophiaLLMIntegration()

async def get_llm_integration() -> SophiaLLMIntegration:
    """
    Get the global LLM integration instance.
    
    Returns:
        SophiaLLMIntegration instance
    """
    if not _llm_integration.is_initialized:
        await _llm_integration.initialize()
    return _llm_integration