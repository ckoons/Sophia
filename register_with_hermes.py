#!/usr/bin/env python3
"""
Register Sophia with Hermes Service Registry

This script registers the Sophia ML component with the Hermes service registry,
allowing other components to discover and use its capabilities.

Usage:
    python register_with_hermes.py [options]

Environment Variables:
    HERMES_URL: URL of the Hermes API (default: http://localhost:8000/api)
    STARTUP_INSTRUCTIONS_FILE: Path to JSON file with startup instructions
    SOPHIA_API_ENDPOINT: API endpoint for Sophia (optional)

Options:
    --hermes-url: URL of the Hermes API (overrides HERMES_URL env var)
    --instructions-file: Path to startup instructions JSON file
    --endpoint: API endpoint for Sophia
    --help: Show this help message
"""

import os
import sys
import asyncio
import signal
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("sophia.registration")

# Get the directory where this script is located
script_dir = Path(__file__).parent.absolute()

# Add parent directories to path
component_dir = str(script_dir)
tekton_dir = os.path.abspath(os.path.join(component_dir, ".."))
tekton_core_dir = os.path.join(tekton_dir, "tekton-core")

# Add to Python path
sys.path.insert(0, component_dir)
sys.path.insert(0, tekton_dir)
sys.path.insert(0, tekton_core_dir)

# Check if we're in a virtual environment
in_venv = sys.prefix != sys.base_prefix
if not in_venv:
    venv_dir = os.path.join(component_dir, "venv")
    if os.path.exists(venv_dir):
        logger.warning(f"Not running in the Sophia virtual environment.")
        logger.warning(f"Consider activating it with: source {venv_dir}/bin/activate")

# Import registration utilities
try:
    # Try to import from tekton-core first (preferred)
    from tekton.utils.hermes_registration import (
        HermesRegistrationClient,
        register_component,
        load_startup_instructions
    )
    REGISTRATION_UTILS_AVAILABLE = True
    logger.info("Successfully imported Tekton registration utilities")
except ImportError:
    logger.warning("Could not import Tekton registration utilities.")
    logger.warning("Falling back to direct Hermes client import.")
    REGISTRATION_UTILS_AVAILABLE = False

    # Try to import from Hermes directly
    try:
        hermes_dir = os.environ.get("HERMES_DIR")
        if not hermes_dir or not os.path.exists(hermes_dir):
            potential_hermes_dir = os.path.normpath(os.path.join(script_dir, "../Hermes"))
            if os.path.exists(potential_hermes_dir):
                hermes_dir = potential_hermes_dir
                sys.path.insert(0, hermes_dir)
                
        # Import from hermes directly
        from hermes.core.service_discovery import ServiceRegistry
        logger.info(f"Successfully imported Hermes modules from {hermes_dir}")
    except ImportError as e:
        logger.error(f"Error importing Hermes modules: {e}")
        logger.error(f"Make sure Hermes is properly installed and accessible")
        sys.exit(1)

# Import Sophia-specific modules
try:
    from sophia.core.ml_engine import get_ml_engine
    logger.info("Successfully imported Sophia ML engine")
except ImportError as e:
    logger.error(f"Error importing Sophia modules: {e}")
    logger.error(f"Make sure Sophia is properly installed and accessible")
    sys.exit(1)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Register Sophia with Hermes Service Registry"
    )
    parser.add_argument(
        "--hermes-url",
        help="URL of the Hermes API",
        default=os.environ.get("HERMES_URL", "http://localhost:8000/api")
    )
    parser.add_argument(
        "--instructions-file",
        help="Path to startup instructions JSON file",
        default=os.environ.get("STARTUP_INSTRUCTIONS_FILE")
    )
    parser.add_argument(
        "--endpoint",
        help="API endpoint for Sophia",
        default=os.environ.get("SOPHIA_API_ENDPOINT", "http://localhost:5500")
    )
    
    return parser.parse_args()

async def register_sophia_with_hermes(
    hermes_url: Optional[str] = None,
    instructions_file: Optional[str] = None,
    endpoint: Optional[str] = None
) -> bool:
    """
    Register Sophia with Hermes service registry.
    
    Args:
        hermes_url: URL of the Hermes API
        instructions_file: Path to JSON file with startup instructions
        endpoint: API endpoint for Sophia
        
    Returns:
        True if registration was successful
    """
    # Check for startup instructions file
    if instructions_file and os.path.isfile(instructions_file):
        logger.info(f"Loading startup instructions from {instructions_file}")
        instructions = load_startup_instructions(instructions_file)
        # Extract relevant information from instructions
    else:
        instructions = {}
    
    # Define component information
    component_id = instructions.get("component_id", "sophia.ml")
    component_name = instructions.get("name", "Sophia ML Engine")
    component_type = instructions.get("type", "machine_learning")
    component_version = instructions.get("version", "0.1.0")
    
    # Define capabilities specific to Sophia
    capabilities = [
        {
            "name": "generate_embeddings",
            "description": "Generate vector embeddings for text",
            "parameters": {
                "text": "string",
                "model_id": "string (optional)"
            }
        },
        {
            "name": "classify_text",
            "description": "Classify text into categories",
            "parameters": {
                "text": "string",
                "categories": "array of strings",
                "model_id": "string (optional)"
            }
        },
        {
            "name": "register_model",
            "description": "Register a new machine learning model",
            "parameters": {
                "model_id": "string",
                "model_type": "string",
                "provider": "string",
                "capabilities": "array of strings",
                "metadata": "object (optional)"
            }
        },
        {
            "name": "get_model_status",
            "description": "Get the status of all registered models",
            "parameters": {}
        }
    ]
    
    # Define dependencies
    dependencies = instructions.get("dependencies", ["hermes.core.database"])
    
    # Define additional metadata
    metadata = {
        "description": "Machine learning and embedding component for Tekton",
        "models": {
            "embedding": {
                "default": "sophia-embedding-small",
                "dimensions": 384
            },
            "classification": {
                "default": "sophia-classification-base"
            }
        }
    }
    if instructions.get("metadata"):
        metadata.update(instructions["metadata"])
    
    # If endpoint is not provided, use a default or from instructions
    if not endpoint:
        endpoint = instructions.get("endpoint", "http://localhost:5500")
    
    try:
        # Initialize the ML engine to make sure models are registered
        engine = await get_ml_engine()
        await engine.start()
        
        # Update metadata with actual model information
        model_status = await engine.get_model_status()
        metadata["registered_models"] = model_status["registered_models"]
        metadata["active_models"] = model_status["active_models"]
        metadata["default_models"] = model_status["default_models"]
        
        # Use standardized registration utility if available
        if REGISTRATION_UTILS_AVAILABLE:
            client = await register_component(
                component_id=component_id,
                component_name=component_name,
                component_type=component_type,
                component_version=component_version,
                capabilities=capabilities,
                hermes_url=hermes_url,
                dependencies=dependencies,
                endpoint=endpoint,
                additional_metadata=metadata
            )
            
            if client:
                logger.info(f"Successfully registered {component_name} with Hermes")
                
                # Set up signal handlers
                loop = asyncio.get_event_loop()
                client.setup_signal_handlers(loop)
                
                # Keep the registration active until interrupted
                stop_event = asyncio.Event()
                
                def handle_signal(sig):
                    logger.info(f"Received signal {sig.name}, shutting down")
                    asyncio.create_task(shutdown(client, engine, stop_event))
                
                for sig in (signal.SIGINT, signal.SIGTERM):
                    loop.add_signal_handler(sig, lambda s=sig: handle_signal(s))
                
                logger.info("Registration active. Press Ctrl+C to unregister and exit...")
                try:
                    await stop_event.wait()
                except Exception as e:
                    logger.error(f"Error during registration: {e}")
                    await shutdown(client, engine, stop_event)
                
                return True
            else:
                logger.error(f"Failed to register {component_name} with Hermes")
                await engine.stop()
                return False
        else:
            # Fall back to direct Hermes registration
            registry = ServiceRegistry()
            await registry.start()
            
            # Register the ML engine service
            ml_success = await registry.register(
                service_id=component_id,
                name=component_name,
                version=component_version,
                endpoint=endpoint,
                capabilities=[cap["name"] for cap in capabilities],
                metadata=metadata
            )
            
            if ml_success:
                logger.info(f"Registered {component_name} with Hermes service registry")
            else:
                logger.error(f"Failed to register {component_name}")
                await registry.stop()
                await engine.stop()
                return False
            
            # Register the embedding service specifically
            embed_success = await registry.register(
                service_id="sophia.embedding",
                name="Sophia Embedding Service",
                version=component_version,
                endpoint=f"{endpoint}/embed",
                capabilities=["embedding", "vector_encoding"],
                metadata={
                    "component": "sophia",
                    "description": "Text embedding service",
                    "dimensions": 384,
                    "model": "sophia-embedding-small"
                }
            )
            
            if embed_success:
                logger.info("Registered Sophia Embedding Service with Hermes service registry")
            else:
                logger.error("Failed to register Sophia Embedding Service")
                await registry.stop()
                await engine.stop()
                return False
                
            # Keep the registration active until interrupted
            try:
                logger.info("Registration active. Press Ctrl+C to unregister and exit...")
                # Run indefinitely until interrupted
                while True:
                    await asyncio.sleep(60)
                    logger.info(f"{component_name} registration still active...")
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received, unregistering...")
            finally:
                # Unregister from Hermes before exiting
                await registry.stop()
                await engine.stop()
                logger.info(f"{component_name} unregistered from Hermes")
            
            return True
    except Exception as e:
        logger.error(f"Error during registration: {e}")
        return False

async def shutdown(client, engine, stop_event):
    """
    Perform graceful shutdown.
    
    Args:
        client: HermesRegistrationClient instance
        engine: ML engine instance
        stop_event: Asyncio event to signal shutdown
    """
    logger.info("Shutting down Sophia...")
    
    # Stop the ML engine
    await engine.stop()
    logger.info("ML Engine stopped")
    
    # Unregister from Hermes
    if client:
        await client.close()
        logger.info("Unregistered from Hermes")
    
    # Signal to stop the main loop
    stop_event.set()
    logger.info("Shutdown complete")

async def main():
    """Main entry point."""
    args = parse_arguments()
    
    logger.info("Registering Sophia with Hermes service registry...")
    
    success = await register_sophia_with_hermes(
        hermes_url=args.hermes_url,
        instructions_file=args.instructions_file,
        endpoint=args.endpoint
    )
    
    if success:
        logger.info("Sophia registration process complete")
    else:
        logger.error("Failed to register Sophia with Hermes")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())