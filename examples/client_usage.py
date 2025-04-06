#!/usr/bin/env python3
"""
Example Usage of the Sophia Client

This script demonstrates how to use the SophiaClient to interact with the Sophia ML component.
"""

import asyncio
import logging
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("sophia_example")

# Try to import from the sophia package
try:
    from sophia.client import SophiaClient, get_sophia_client
except ImportError:
    import sys
    import os
    
    # Add the parent directory to sys.path
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    
    # Try importing again
    from sophia.client import SophiaClient, get_sophia_client


async def text_embedding_example():
    """Example of using the Sophia client for text embedding."""
    logger.info("=== Text Embedding Example ===")
    
    # Create a Sophia client
    client = await get_sophia_client()
    
    try:
        # Get available embedding models
        embedding_models = await client.get_embedding_models()
        logger.info(f"Available embedding models: {embedding_models}")
        
        # Generate embeddings for some text
        text = "This is an example text for embedding with Sophia."
        
        # Use the default model
        embeddings = await client.generate_embeddings(text)
        logger.info(f"Generated embeddings with {len(embeddings)} dimensions")
        
        # Use a specific model if available
        if embedding_models:
            model_id = embedding_models[0]
            embeddings = await client.generate_embeddings(text, model_id=model_id)
            logger.info(f"Generated embeddings with model {model_id}: {len(embeddings)} dimensions")
    
    except Exception as e:
        logger.error(f"Error in text embedding example: {e}")
    
    finally:
        # Close the client
        await client.close()


async def text_classification_example():
    """Example of using the Sophia client for text classification."""
    logger.info("=== Text Classification Example ===")
    
    # Create a Sophia client
    client = await get_sophia_client()
    
    try:
        # Get available classification models
        classification_models = await client.get_classification_models()
        logger.info(f"Available classification models: {classification_models}")
        
        # Classify some text
        text = "This project is focused on machine learning and AI integration."
        categories = ["technology", "science", "business", "health"]
        
        # Use the default model
        classifications = await client.classify_text(text, categories)
        logger.info(f"Text classifications: {classifications}")
        
        # Use a specific model if available
        if classification_models:
            model_id = classification_models[0]
            classifications = await client.classify_text(text, categories, model_id=model_id)
            logger.info(f"Text classifications with model {model_id}: {classifications}")
    
    except Exception as e:
        logger.error(f"Error in text classification example: {e}")
    
    finally:
        # Close the client
        await client.close()


async def model_status_example():
    """Example of getting model status information."""
    logger.info("=== Model Status Example ===")
    
    # Create a Sophia client
    client = await get_sophia_client()
    
    try:
        # Get model status
        status = await client.get_model_status()
        
        # Log key information
        logger.info(f"Registered models: {status.get('registered_models', 0)}")
        logger.info(f"Active models: {status.get('active_models', 0)}")
        logger.info(f"Default models: {status.get('default_models', {})}")
        
        # Print model details
        models = status.get("models", {})
        for model_id, model_info in models.items():
            logger.info(f"Model: {model_id}")
            logger.info(f"  Type: {model_info.get('model_type')}")
            logger.info(f"  Provider: {model_info.get('provider')}")
            logger.info(f"  Status: {model_info.get('status')}")
            logger.info(f"  Capabilities: {model_info.get('capabilities', [])}")
    
    except Exception as e:
        logger.error(f"Error in model status example: {e}")
    
    finally:
        # Close the client
        await client.close()


async def error_handling_example():
    """Example of handling errors with the Sophia client."""
    logger.info("=== Error Handling Example ===")
    
    # Create a Sophia client with a non-existent component ID
    try:
        client = await get_sophia_client(component_id="sophia.nonexistent")
        # This should raise ComponentNotFoundError
        
    except Exception as e:
        logger.info(f"Caught expected error: {type(e).__name__}: {e}")
    
    # Create a valid client
    client = await get_sophia_client()
    
    try:
        # Try to invoke a non-existent capability
        try:
            await client.invoke_capability("nonexistent_capability", {})
        except Exception as e:
            logger.info(f"Caught expected error: {type(e).__name__}: {e}")
        
        # Try to generate embeddings with a non-existent model
        try:
            await client.generate_embeddings("test", model_id="nonexistent_model")
        except Exception as e:
            logger.info(f"Caught expected error: {type(e).__name__}: {e}")
    
    finally:
        # Close the client
        await client.close()


async def main():
    """Run all examples."""
    try:
        await text_embedding_example()
        await text_classification_example()
        await model_status_example()
        await error_handling_example()
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}")


if __name__ == "__main__":
    asyncio.run(main())