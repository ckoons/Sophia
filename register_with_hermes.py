#!/usr/bin/env python3
"""
Registers Sophia with the Hermes service registry.

This script registers the Sophia ML service with the Hermes centralized
service registry so other components can discover and use it.
"""

import os
import sys
import asyncio
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("sophia_registration")

# Get the directory where this script is located
script_dir = Path(__file__).parent.absolute()

# Check if we're in a virtual environment
venv_dir = os.path.join(script_dir, "venv")
if os.path.exists(venv_dir):
    # Activate the virtual environment if not already activated
    if not os.environ.get("VIRTUAL_ENV"):
        print(f"Please run this script within the Sophia virtual environment:")
        print(f"source {venv_dir}/bin/activate")
        print(f"python {os.path.basename(__file__)}")
        sys.exit(1)

# Find Hermes directory (prioritize environment variable if set)
hermes_dir = os.environ.get("HERMES_DIR")
if not hermes_dir or not os.path.exists(hermes_dir):
    # Try to find Hermes relative to this script
    potential_hermes_dir = os.path.normpath(os.path.join(script_dir, "../Hermes"))
    if os.path.exists(potential_hermes_dir):
        hermes_dir = potential_hermes_dir
    else:
        print(f"Hermes directory not found. Please set the HERMES_DIR environment variable.")
        sys.exit(1)

# Add Hermes to the Python path
sys.path.insert(0, hermes_dir)

# Try to import Hermes modules
try:
    from hermes.core.service_discovery import ServiceRegistry
    from hermes.core.registration.client import RegistrationClient
    logger.info(f"Successfully imported Hermes modules from {hermes_dir}")
except ImportError as e:
    logger.error(f"Error importing Hermes modules: {e}")
    logger.error(f"Make sure Hermes is properly installed and accessible")
    sys.exit(1)

async def register_with_hermes():
    """Register Sophia services with Hermes."""
    try:
        # Initialize the service registry
        registry = ServiceRegistry()
        await registry.start()
        
        # Register the ML engine service
        success = await registry.register(
            service_id="sophia-ml-engine",
            name="Sophia ML Engine",
            version="0.1.0",
            endpoint="http://localhost:5500",  # Default endpoint for Sophia API
            capabilities=[
                "machine_learning", 
                "embedding", 
                "classification", 
                "model_management"
            ],
            metadata={
                "component": "sophia",
                "description": "Machine learning component of Tekton"
            }
        )
        
        if success:
            logger.info("Registered Sophia ML Engine with Hermes service registry")
        else:
            logger.error("Failed to register Sophia ML Engine")
            return False
        
        # Register the embedding service specifically
        success = await registry.register(
            service_id="sophia-embedding",
            name="Sophia Embedding Service",
            version="0.1.0",
            endpoint="http://localhost:5500/embed",
            capabilities=["embedding", "vector_encoding"],
            metadata={
                "component": "sophia",
                "description": "Text embedding service",
                "dimensions": 384,
                "model": "sophia-embedding-small"
            }
        )
        
        if success:
            logger.info("Registered Sophia Embedding Service with Hermes service registry")
        else:
            logger.error("Failed to register Sophia Embedding Service")
            return False
            
        await registry.stop()
        logger.info("Registration with Hermes completed successfully")
        return True
    except Exception as e:
        logger.error(f"Error during Hermes registration: {e}")
        return False

# Main execution
if __name__ == "__main__":
    logger.info("Registering Sophia with Hermes service registry...")
    success = asyncio.run(register_with_hermes())
    sys.exit(0 if success else 1)
