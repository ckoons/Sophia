"""
Sophia Client - Client for interacting with the Sophia ML component.

This module provides a client for interacting with Sophia's machine learning capabilities
through the standardized Tekton component client interface.
"""

import os
import logging
from typing import Dict, List, Any, Optional, Union

# Try to import from tekton-core first
try:
    from tekton.utils.component_client import (
        ComponentClient,
        ComponentError,
        ComponentNotFoundError,
        CapabilityNotFoundError,
        CapabilityInvocationError,
        ComponentUnavailableError,
        SecurityContext,
        RetryPolicy,
    )
except ImportError:
    # If tekton-core is not available, use a minimal implementation
    from .utils.component_client import (
        ComponentClient,
        ComponentError,
        ComponentNotFoundError,
        CapabilityNotFoundError,
        CapabilityInvocationError,
        ComponentUnavailableError,
        SecurityContext,
        RetryPolicy,
    )

# Configure logger
logger = logging.getLogger(__name__)


class SophiaClient(ComponentClient):
    """Client for the Sophia ML component."""
    
    def __init__(
        self,
        component_id: str = "sophia.ml",
        hermes_url: Optional[str] = None,
        security_context: Optional[SecurityContext] = None,
        retry_policy: Optional[RetryPolicy] = None
    ):
        """
        Initialize the Sophia client.
        
        Args:
            component_id: ID of the Sophia component to connect to (default: "sophia.ml")
            hermes_url: URL of the Hermes API
            security_context: Security context for authentication/authorization
            retry_policy: Policy for retrying capability invocations
        """
        super().__init__(
            component_id=component_id,
            hermes_url=hermes_url,
            security_context=security_context,
            retry_policy=retry_policy
        )
    
    async def generate_embeddings(
        self,
        text: str,
        model_id: Optional[str] = None
    ) -> List[float]:
        """
        Generate vector embeddings for text.
        
        Args:
            text: The text to generate embeddings for
            model_id: ID of the embedding model to use (optional)
            
        Returns:
            List of embedding values
            
        Raises:
            CapabilityInvocationError: If the embedding generation fails
            ComponentUnavailableError: If the Sophia component is unavailable
        """
        parameters = {"text": text}
        if model_id:
            parameters["model_id"] = model_id
            
        result = await self.invoke_capability("generate_embeddings", parameters)
        
        if not isinstance(result, dict) or "embeddings" not in result:
            raise CapabilityInvocationError(
                "Unexpected response format from Sophia",
                result
            )
            
        return result["embeddings"]
    
    async def classify_text(
        self,
        text: str,
        categories: List[str],
        model_id: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Classify text into provided categories.
        
        Args:
            text: The text to classify
            categories: List of categories to classify into
            model_id: ID of the classification model to use (optional)
            
        Returns:
            Dictionary mapping categories to confidence scores
            
        Raises:
            CapabilityInvocationError: If the classification fails
            ComponentUnavailableError: If the Sophia component is unavailable
        """
        parameters = {
            "text": text,
            "categories": categories
        }
        if model_id:
            parameters["model_id"] = model_id
            
        result = await self.invoke_capability("classify_text", parameters)
        
        if not isinstance(result, dict) or "classifications" not in result:
            raise CapabilityInvocationError(
                "Unexpected response format from Sophia",
                result
            )
            
        return result["classifications"]
    
    async def register_model(
        self,
        model_id: str,
        model_type: str,
        provider: str,
        capabilities: List[str],
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Register a new machine learning model.
        
        Args:
            model_id: Unique identifier for the model
            model_type: Type of model (embedding, classification, etc.)
            provider: The provider of the model (local, huggingface, etc.)
            capabilities: List of capabilities the model provides
            metadata: Additional model information
            
        Returns:
            True if registration was successful
            
        Raises:
            CapabilityInvocationError: If the model registration fails
            ComponentUnavailableError: If the Sophia component is unavailable
        """
        parameters = {
            "model_id": model_id,
            "model_type": model_type,
            "provider": provider,
            "capabilities": capabilities
        }
        if metadata:
            parameters["metadata"] = metadata
            
        result = await self.invoke_capability("register_model", parameters)
        
        if not isinstance(result, dict) or "success" not in result:
            raise CapabilityInvocationError(
                "Unexpected response format from Sophia",
                result
            )
            
        return result["success"]
    
    async def get_model_status(self) -> Dict[str, Any]:
        """
        Get the status of all registered models.
        
        Returns:
            Dictionary with model status information
            
        Raises:
            CapabilityInvocationError: If the status retrieval fails
            ComponentUnavailableError: If the Sophia component is unavailable
        """
        result = await self.invoke_capability("get_model_status", {})
        
        if not isinstance(result, dict):
            raise CapabilityInvocationError(
                "Unexpected response format from Sophia",
                result
            )
            
        return result
    
    async def get_embedding_models(self) -> List[str]:
        """
        Get a list of available embedding models.
        
        Returns:
            List of embedding model IDs
            
        Raises:
            CapabilityInvocationError: If the model retrieval fails
            ComponentUnavailableError: If the Sophia component is unavailable
        """
        status = await self.get_model_status()
        
        models = status.get("models", {})
        embedding_models = []
        
        for model_id, model_info in models.items():
            if model_info.get("model_type") == "embedding":
                embedding_models.append(model_id)
                
        return embedding_models
    
    async def get_classification_models(self) -> List[str]:
        """
        Get a list of available classification models.
        
        Returns:
            List of classification model IDs
            
        Raises:
            CapabilityInvocationError: If the model retrieval fails
            ComponentUnavailableError: If the Sophia component is unavailable
        """
        status = await self.get_model_status()
        
        models = status.get("models", {})
        classification_models = []
        
        for model_id, model_info in models.items():
            if model_info.get("model_type") == "classification":
                classification_models.append(model_id)
                
        return classification_models


async def get_sophia_client(
    component_id: str = "sophia.ml",
    hermes_url: Optional[str] = None,
    security_context: Optional[SecurityContext] = None,
    retry_policy: Optional[RetryPolicy] = None
) -> SophiaClient:
    """
    Create a client for the Sophia ML component.
    
    Args:
        component_id: ID of the Sophia component to connect to (default: "sophia.ml")
        hermes_url: URL of the Hermes API
        security_context: Security context for authentication/authorization
        retry_policy: Policy for retrying capability invocations
        
    Returns:
        SophiaClient instance
        
    Raises:
        ComponentNotFoundError: If the Sophia component is not found
        ComponentUnavailableError: If the Hermes API is unavailable
    """
    # Try to import from tekton-core first
    try:
        from tekton.utils.component_client import discover_component
    except ImportError:
        # If tekton-core is not available, use a minimal implementation
        from .utils.component_client import discover_component
    
    # Check if the component exists
    await discover_component(component_id, hermes_url)
    
    # Create the client
    return SophiaClient(
        component_id=component_id,
        hermes_url=hermes_url,
        security_context=security_context,
        retry_policy=retry_policy
    )