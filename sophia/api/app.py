"""
Sophia API Server

This module implements the Sophia API server using FastAPI with Single Port Architecture,
providing HTTP and WebSocket endpoints for accessing Sophia's capabilities.
"""

import os
import json
import asyncio
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

from fastapi import FastAPI, APIRouter, HTTPException, WebSocket, WebSocketDisconnect, Depends, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Import Sophia core modules
from sophia.core.metrics_engine import get_metrics_engine
from sophia.core.analysis_engine import get_analysis_engine
from sophia.core.experiment_framework import get_experiment_framework
from sophia.core.recommendation_system import get_recommendation_system
from sophia.core.intelligence_measurement import get_intelligence_measurement, IntelligenceDimension
from sophia.core.ml_engine import get_ml_engine

# Import models
from sophia.models.metrics import MetricSubmission, MetricQuery, MetricResponse, MetricAggregationQuery, MetricDefinition
from sophia.models.experiment import ExperimentCreate, ExperimentUpdate, ExperimentQuery, ExperimentResponse, ExperimentResult
from sophia.models.recommendation import RecommendationCreate, RecommendationUpdate, RecommendationQuery, RecommendationResponse
from sophia.models.intelligence import IntelligenceMeasurementCreate, IntelligenceMeasurementQuery, IntelligenceMeasurementResponse, ComponentIntelligenceProfile
from sophia.models.component import ComponentRegister, ComponentUpdate, ComponentQuery, ComponentResponse
from sophia.models.research import ResearchProjectCreate, ResearchProjectUpdate, ResearchProjectQuery, ResearchProjectResponse

# Import Sophia utilities
try:
    from sophia.utils.tekton_utils import setup_logging, get_logger, get_config
    from sophia.utils.llm_integration import get_llm_integration
    
    # Setup logging with Tekton shared utilities
    setup_logging("sophia")
    logger = get_logger("sophia.api")
except ImportError:
    # Fallback to standard logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger("sophia.api")

# Create FastAPI app
app = FastAPI(
    title="Sophia API",
    description="API for Sophia, the machine learning and continuous improvement component of Tekton",
    version="0.1.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import API endpoint routers
from sophia.api.endpoints import api_router

# Import FastMCP endpoints
try:
    from sophia.api.fastmcp_endpoints import fastmcp_router
    fastmcp_available = True
    logger.info("FastMCP endpoints loaded successfully")
except ImportError as e:
    fastmcp_available = False
    logger.warning(f"FastMCP endpoints not available: {e}")

# Create system router for system-level endpoints
system_router = APIRouter(tags=["System"])

# WebSocket connections
active_connections: List[WebSocket] = []

# ------------------------
# Dependency Injections
# ------------------------

async def get_metrics_engine_dep():
    """Dependency injection for metrics engine."""
    return await get_metrics_engine()

async def get_analysis_engine_dep():
    """Dependency injection for analysis engine."""
    return await get_analysis_engine()

async def get_experiment_framework_dep():
    """Dependency injection for experiment framework."""
    return await get_experiment_framework()

async def get_recommendation_system_dep():
    """Dependency injection for recommendation system."""
    return await get_recommendation_system()

async def get_intelligence_measurement_dep():
    """Dependency injection for intelligence measurement."""
    return await get_intelligence_measurement()

async def get_ml_engine_dep():
    """Dependency injection for ML engine."""
    return await get_ml_engine()

# ------------------------
# Metrics API Routes
# ------------------------


# ------------------------
# WebSocket Connection
# ------------------------

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await websocket.accept()
    active_connections.append(websocket)
    
    try:
        while True:
            # Wait for client message
            data = await websocket.receive_text()
            
            # Parse message
            try:
                message = json.loads(data)
                message_type = message.get("type", "")
                
                if message_type == "subscribe":
                    # Handle subscription requests
                    await handle_subscription(websocket, message)
                elif message_type == "ping":
                    # Handle ping messages
                    await websocket.send_json({
                        "type": "pong",
                        "timestamp": datetime.utcnow().isoformat() + "Z"
                    })
                else:
                    # Unknown message type
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Unknown message type: {message_type}"
                    })
            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "message": "Invalid JSON message"
                })
    except WebSocketDisconnect:
        # Client disconnected
        if websocket in active_connections:
            active_connections.remove(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        # Remove connection on error
        if websocket in active_connections:
            active_connections.remove(websocket)

async def handle_subscription(websocket: WebSocket, message: Dict[str, Any]):
    """Handle WebSocket subscription request."""
    channel = message.get("channel", "")
    filters = message.get("filters", {})
    
    # Send confirmation
    await websocket.send_json({
        "type": "subscription_confirmed",
        "channel": channel,
        "filters": filters,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    })

async def notify_clients(data: Dict[str, Any]):
    """Send notification to all connected WebSocket clients."""
    for connection in active_connections:
        try:
            await connection.send_json(data)
        except Exception as e:
            logger.error(f"Error sending WebSocket notification: {e}")
            # Remove failed connection
            if connection in active_connections:
                active_connections.remove(connection)

# ------------------------
# Health Check Endpoint
# ------------------------

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Check core engines
        metrics_engine = await get_metrics_engine()
        analysis_engine = await get_analysis_engine()
        experiment_framework = await get_experiment_framework()
        recommendation_system = await get_recommendation_system()
        intelligence_measurement = await get_intelligence_measurement()
        ml_engine = await get_ml_engine()
        
        # All engines should be initialized
        all_initialized = (
            metrics_engine.is_initialized and
            analysis_engine.is_initialized and
            experiment_framework.is_initialized and
            recommendation_system.is_initialized and
            intelligence_measurement.is_initialized and
            ml_engine.is_initialized
        )
        
        if all_initialized:
            return {
                "status": "healthy",
                "message": "All systems operational",
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "components": {
                    "metrics_engine": metrics_engine.is_initialized,
                    "analysis_engine": analysis_engine.is_initialized,
                    "experiment_framework": experiment_framework.is_initialized,
                    "recommendation_system": recommendation_system.is_initialized,
                    "intelligence_measurement": intelligence_measurement.is_initialized,
                    "ml_engine": ml_engine.is_initialized
                }
            }
        else:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={
                    "status": "degraded",
                    "message": "Some components are not initialized",
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "components": {
                        "metrics_engine": metrics_engine.is_initialized,
                        "analysis_engine": analysis_engine.is_initialized,
                        "experiment_framework": experiment_framework.is_initialized,
                        "recommendation_system": recommendation_system.is_initialized,
                        "intelligence_measurement": intelligence_measurement.is_initialized,
                        "ml_engine": ml_engine.is_initialized
                    }
                }
            )
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "status": "error",
                "message": f"Health check failed: {str(e)}",
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
        )

# Include routers
app.include_router(api_router, prefix="/api")
app.include_router(system_router)

# Include FastMCP endpoints if available
if fastmcp_available:
    app.include_router(fastmcp_router, prefix="/mcp", tags=["MCP"])
    logger.info("FastMCP router included at /mcp")

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize components on startup."""
    try:
        # Initialize core engines
        metrics_engine = await get_metrics_engine()
        await metrics_engine.start()
        
        analysis_engine = await get_analysis_engine()
        await analysis_engine.start()
        
        experiment_framework = await get_experiment_framework()
        await experiment_framework.start()
        
        recommendation_system = await get_recommendation_system()
        await recommendation_system.start()
        
        intelligence_measurement = await get_intelligence_measurement()
        await intelligence_measurement.start()
        
        ml_engine = await get_ml_engine()
        await ml_engine.start()
        
        # Initialize LLM integration if available
        try:
            llm_integration = await get_llm_integration()
            await llm_integration.initialize()
            logger.info("LLM Integration initialized successfully")
        except Exception as llm_error:
            logger.warning(f"Failed to initialize LLM Integration: {llm_error}")
        
        # Register with Hermes if available
        try:
            from sophia.utils.tekton_utils import register_with_hermes
            success = await register_with_hermes()
            if success:
                logger.info("Registered with Hermes successfully")
            else:
                logger.warning("Failed to register with Hermes")
        except Exception as hermes_error:
            logger.warning(f"Failed to register with Hermes: {hermes_error}")
        
        logger.info("Sophia API server started successfully")
    except Exception as e:
        logger.error(f"Error during startup: {e}")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    try:
        # Stop core engines
        metrics_engine = await get_metrics_engine()
        await metrics_engine.stop()
        
        analysis_engine = await get_analysis_engine()
        await analysis_engine.stop()
        
        experiment_framework = await get_experiment_framework()
        await experiment_framework.stop()
        
        recommendation_system = await get_recommendation_system()
        await recommendation_system.stop()
        
        intelligence_measurement = await get_intelligence_measurement()
        await intelligence_measurement.stop()
        
        ml_engine = await get_ml_engine()
        await ml_engine.stop()
        
        # Shutdown LLM integration if available
        try:
            llm_integration = await get_llm_integration()
            await llm_integration.shutdown()
            logger.info("LLM Integration shut down successfully")
        except Exception as llm_error:
            logger.warning(f"Error shutting down LLM Integration: {llm_error}")
        
        # Close any active WebSocket connections
        for connection in active_connections:
            try:
                await connection.close()
            except Exception:
                pass
        
        logger.info("Sophia API server shutdown complete")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")