"""
Sophia MCP (Model Context Protocol) integration.

This module provides MCP capabilities and tools for Sophia's ML/AI analysis,
research management, and intelligence measurement features.
"""

from .capabilities import (
    MLAnalysisCapability,
    ResearchManagementCapability,
    IntelligenceMeasurementCapability
)

from .tools import (
    ml_analysis_tools,
    research_management_tools,
    intelligence_measurement_tools
)


def get_all_capabilities():
    """Get all Sophia MCP capabilities."""
    return [
        MLAnalysisCapability,
        ResearchManagementCapability,
        IntelligenceMeasurementCapability
    ]


def get_all_tools():
    """Get all Sophia MCP tools."""
    return ml_analysis_tools + research_management_tools + intelligence_measurement_tools


__all__ = [
    "MLAnalysisCapability",
    "ResearchManagementCapability",
    "IntelligenceMeasurementCapability",
    "ml_analysis_tools",
    "research_management_tools",
    "intelligence_measurement_tools",
    "get_all_capabilities",
    "get_all_tools"
]