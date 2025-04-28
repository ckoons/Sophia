#!/usr/bin/env python3
"""
Register Sophia with Hermes.

This script registers the Sophia component with the Hermes service registry.
"""

import os
import sys
import json
import logging
import argparse
import requests
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("sophia.register")

# Default Hermes URL
DEFAULT_HERMES_URL = "http://localhost:8000"

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Register Sophia with Hermes")
    parser.add_argument(
        "--hermes-url",
        type=str,
        default=os.environ.get("HERMES_URL", DEFAULT_HERMES_URL),
        help=f"Hermes service URL (default: {DEFAULT_HERMES_URL})"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("SOPHIA_PORT", "8006")),
        help="Port on which Sophia is running (default: 8006)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default=os.environ.get("SOPHIA_HOST", "localhost"),
        help="Host on which Sophia is running (default: localhost)"
    )
    parser.add_argument(
        "--version",
        type=str,
        default="0.1.0",
        help="Sophia version (default: 0.1.0)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-registration even if already registered"
    )
    return parser.parse_args()

def check_hermes_availability(hermes_url):
    """Check if Hermes is available."""
    try:
        response = requests.get(f"{hermes_url}/health")
        if response.status_code == 200:
            logger.info("Hermes is available")
            return True
        else:
            logger.error(f"Hermes returned non-200 status code: {response.status_code}")
            return False
    except requests.RequestException as e:
        logger.error(f"Error connecting to Hermes: {e}")
        return False

def check_registration(hermes_url, component_id):
    """Check if component is already registered."""
    try:
        response = requests.get(f"{hermes_url}/api/registry/components/{component_id}")
        if response.status_code == 200:
            logger.info(f"Component {component_id} is already registered")
            return True
        elif response.status_code == 404:
            logger.info(f"Component {component_id} is not registered")
            return False
        else:
            logger.error(f"Unexpected status code: {response.status_code}")
            return False
    except requests.RequestException as e:
        logger.error(f"Error checking registration: {e}")
        return False

def register_component(hermes_url, registration_data):
    """Register component with Hermes."""
    try:
        response = requests.post(
            f"{hermes_url}/api/registry/components",
            json=registration_data
        )
        
        if response.status_code == 200 or response.status_code == 201:
            logger.info(f"Successfully registered component {registration_data['component_id']}")
            return True
        else:
            logger.error(f"Failed to register component: {response.status_code} - {response.text}")
            return False
    except requests.RequestException as e:
        logger.error(f"Error registering component: {e}")
        return False

def main():
    """Main entry point."""
    args = parse_args()
    
    # Check if Hermes is available
    if not check_hermes_availability(args.hermes_url):
        logger.error("Cannot connect to Hermes. Please ensure Hermes is running.")
        sys.exit(1)
    
    # Component ID for Sophia
    component_id = "sophia"
    
    # Check if already registered
    if check_registration(args.hermes_url, component_id) and not args.force:
        logger.info("Component already registered. Use --force to re-register.")
        sys.exit(0)
    
    # Base URL for Sophia API
    base_url = f"http://{args.host}:{args.port}"
    
    # Prepare registration data
    registration_data = {
        "component_id": component_id,
        "name": "Sophia",
        "description": "Machine learning and continuous improvement component for Tekton",
        "version": args.version,
        "base_url": base_url,
        "health_endpoint": f"{base_url}/health",
        "api_endpoints": [
            {
                "path": "/api/metrics",
                "method": "POST",
                "description": "Submit metrics data"
            },
            {
                "path": "/api/metrics",
                "method": "GET",
                "description": "Query metrics data"
            },
            {
                "path": "/api/experiments",
                "method": "POST",
                "description": "Create a new experiment"
            },
            {
                "path": "/api/experiments",
                "method": "GET",
                "description": "Query experiments"
            },
            {
                "path": "/api/recommendations",
                "method": "POST",
                "description": "Create a new recommendation"
            },
            {
                "path": "/api/recommendations",
                "method": "GET",
                "description": "Query recommendations"
            },
            {
                "path": "/api/intelligence/measurements",
                "method": "POST",
                "description": "Create a new intelligence measurement"
            },
            {
                "path": "/api/intelligence/measurements",
                "method": "GET",
                "description": "Query intelligence measurements"
            },
            {
                "path": "/api/components/register",
                "method": "POST",
                "description": "Register a component with Sophia"
            },
            {
                "path": "/api/components",
                "method": "GET",
                "description": "Query registered components"
            },
            {
                "path": "/api/research/projects",
                "method": "POST",
                "description": "Create a new research project"
            },
            {
                "path": "/api/research/projects",
                "method": "GET",
                "description": "Query research projects"
            }
        ],
        "capabilities": [
            "metrics_collection",
            "metrics_analysis",
            "experimentation",
            "recommendation_generation",
            "intelligence_measurement",
            "component_analysis",
            "ai_research"
        ],
        "dependencies": [
            "hermes"
        ],
        "websocket_endpoints": [
            {
                "path": "/ws",
                "description": "WebSocket endpoint for real-time updates"
            }
        ],
        "ui_components": [
            {
                "name": "sophia-dashboard",
                "path": "/ui/sophia-component.html",
                "description": "Sophia UI dashboard component"
            }
        ],
        "tags": [
            "metrics",
            "analysis",
            "ml",
            "ai",
            "intelligence",
            "research",
            "experiments",
            "recommendations"
        ]
    }
    
    # Register component
    if register_component(args.hermes_url, registration_data):
        logger.info("Sophia successfully registered with Hermes")
    else:
        logger.error("Failed to register Sophia with Hermes")
        sys.exit(1)

if __name__ == "__main__":
    main()