"""
MCP capabilities for Sophia.

This module defines the Model Context Protocol capabilities that Sophia provides
for ML/AI analysis, research management, and intelligence measurement.
"""

from typing import Dict, Any, List
from tekton.mcp.fastmcp.schema import MCPCapability


class MLAnalysisCapability(MCPCapability):
    """Capability for machine learning and AI analysis."""
    
    name = "ml_analysis"
    description = "Perform ML/AI analysis, pattern extraction, and predictive modeling"
    version = "1.0.0"
    
    @classmethod
    def get_supported_operations(cls) -> List[str]:
        """Get list of supported operations."""
        return [
            "analyze_component_performance",
            "extract_patterns",
            "predict_optimization_impact",
            "design_ml_experiment",
            "analyze_ecosystem_trends",
            "forecast_system_evolution"
        ]
    
    @classmethod
    def get_capability_metadata(cls) -> Dict[str, Any]:
        """Get capability metadata."""
        return {
            "category": "ml_analysis",
            "provider": "sophia",
            "requires_auth": False,
            "rate_limited": True,
            "analysis_types": ["performance", "trend", "pattern", "prediction"],
            "ml_algorithms": ["clustering", "classification", "regression", "time_series"],
            "data_formats": ["metrics", "logs", "events", "time_series"],
            "prediction_horizon": ["short_term", "medium_term", "long_term"]
        }


class ResearchManagementCapability(MCPCapability):
    """Capability for research project and experiment management."""
    
    name = "research_management"
    description = "Manage research projects, experiments, and knowledge discovery"
    version = "1.0.0"
    
    @classmethod
    def get_supported_operations(cls) -> List[str]:
        """Get list of supported operations."""
        return [
            "create_research_project",
            "manage_experiment_lifecycle",
            "validate_optimization_results",
            "generate_research_recommendations",
            "track_research_progress",
            "synthesize_research_findings"
        ]
    
    @classmethod
    def get_capability_metadata(cls) -> Dict[str, Any]:
        """Get capability metadata."""
        return {
            "category": "research_management",
            "provider": "sophia",
            "requires_auth": False,
            "project_types": ["exploratory", "confirmatory", "optimization", "evaluation"],
            "experiment_methods": ["a_b_testing", "controlled_trial", "observational", "simulation"],
            "research_phases": ["planning", "execution", "analysis", "reporting"],
            "knowledge_domains": ["performance", "behavior", "efficiency", "intelligence"]
        }


class IntelligenceMeasurementCapability(MCPCapability):
    """Capability for measuring and tracking component intelligence."""
    
    name = "intelligence_measurement"
    description = "Measure, compare, and track intelligence across components"
    version = "1.0.0"
    
    @classmethod
    def get_supported_operations(cls) -> List[str]:
        """Get list of supported operations."""
        return [
            "measure_component_intelligence",
            "compare_intelligence_profiles",
            "track_intelligence_evolution",
            "generate_intelligence_insights"
        ]
    
    @classmethod
    def get_capability_metadata(cls) -> Dict[str, Any]:
        """Get capability metadata."""
        return {
            "category": "intelligence_measurement",
            "provider": "sophia",
            "requires_auth": False,
            "intelligence_dimensions": ["reasoning", "learning", "adaptation", "creativity", "problem_solving"],
            "measurement_methods": ["behavioral_analysis", "performance_testing", "capability_assessment"],
            "comparison_types": ["peer", "historical", "benchmark", "relative"],
            "tracking_intervals": ["real_time", "hourly", "daily", "weekly", "monthly"]
        }


# Export all capabilities
__all__ = [
    "MLAnalysisCapability",
    "ResearchManagementCapability",
    "IntelligenceMeasurementCapability"
]