"""
Sophia API Server

This module implements the Sophia API server using FastAPI with Single Port Architecture,
providing HTTP and WebSocket endpoints for accessing Sophia's capabilities.
"""

import os
import sys
import json
import asyncio
import time
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, APIRouter, HTTPException, WebSocket, WebSocketDisconnect, Depends, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Add Tekton root to path if not already present
tekton_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if tekton_root not in sys.path:
    sys.path.insert(0, tekton_root)

# Import shared utilities
from shared.utils.hermes_registration import HermesRegistration, heartbeat_loop
from shared.utils.logging_setup import setup_component_logging
from shared.utils.env_config import get_component_config
from shared.utils.errors import StartupError
from shared.utils.startup import component_startup, StartupMetrics
from shared.utils.shutdown import GracefulShutdown

# Import shared API utilities
from shared.api.documentation import get_openapi_configuration
from shared.api.endpoints import create_ready_endpoint, create_discovery_endpoint, EndpointInfo
from shared.api.routers import create_standard_routers, mount_standard_routers

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

# Import Sophia version
from sophia import __version__

# Import Sophia utilities
try:
    from sophia.utils.tekton_utils import get_config
    from sophia.utils.llm_integration import get_llm_integration
except ImportError:
    pass

# Set up logging
logger = setup_component_logging("sophia")

# Component configuration
COMPONENT_NAME = "sophia"
COMPONENT_VERSION = "0.1.0"
COMPONENT_DESCRIPTION = "Machine learning and continuous improvement system for Tekton ecosystem"

# Global start time for readiness checks
start_time = time.time()

# WebSocket connections (global for access in lifespan)
active_connections: List[WebSocket] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for Sophia"""
    # Startup
    logger.info("Starting Sophia API server...")
    
    # Get configuration
    config = get_component_config()
    try:
        port = config.sophia.port
    except (AttributeError, TypeError):
        port = int(os.environ.get("SOPHIA_PORT"))
    
    try:
        # Initialize core engines
        metrics_engine = await get_metrics_engine()
        await metrics_engine.start()
        logger.info("Metrics engine started")
        
        analysis_engine = await get_analysis_engine()
        await analysis_engine.start()
        logger.info("Analysis engine started")
        
        experiment_framework = await get_experiment_framework()
        await experiment_framework.start()
        logger.info("Experiment framework started")
        
        recommendation_system = await get_recommendation_system()
        await recommendation_system.start()
        logger.info("Recommendation system started")
        
        intelligence_measurement = await get_intelligence_measurement()
        await intelligence_measurement.start()
        logger.info("Intelligence measurement started")
        
        ml_engine = await get_ml_engine()
        await ml_engine.start()
        logger.info("ML engine started")
        
        # Initialize LLM integration if available
        try:
            llm_integration = await get_llm_integration()
            await llm_integration.initialize()
            logger.info("LLM Integration initialized successfully")
        except Exception as llm_error:
            logger.warning(f"Failed to initialize LLM Integration: {llm_error}")
        
        # Register with Hermes
        hermes_registration = HermesRegistration()
        await hermes_registration.register_component(
            component_name=COMPONENT_NAME,
            port=port,
            version=COMPONENT_VERSION,
            capabilities=["metrics", "analysis", "experiments", "recommendations", "intelligence", "ml"],
            metadata={
                "description": COMPONENT_DESCRIPTION,
                "category": "analytics"
            }
        )
        app.state.hermes_registration = hermes_registration
        
        # Start heartbeat task
        if hermes_registration.is_registered:
            heartbeat_task = asyncio.create_task(heartbeat_loop(hermes_registration, "sophia"))
        
        # Initialize Hermes MCP Bridge
        try:
            from sophia.core.mcp.hermes_bridge import SophiaMCPBridge
            mcp_bridge = SophiaMCPBridge(ml_engine)
            await mcp_bridge.initialize()
            app.state.mcp_bridge = mcp_bridge
            logger.info("Initialized Hermes MCP Bridge for FastMCP tools")
        except Exception as e:
            logger.warning(f"Failed to initialize MCP Bridge: {e}")
        
        logger.info(f"Sophia API server started successfully on port {port}")
        
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise StartupError(f"Failed to start Sophia: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Sophia API server...")
    
    # Cancel heartbeat task if running
    if hermes_registration.is_registered and 'heartbeat_task' in locals():
        heartbeat_task.cancel()
        try:
            await heartbeat_task
        except asyncio.CancelledError:
            pass
    
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
        
        # Cleanup MCP bridge
        if hasattr(app.state, "mcp_bridge") and app.state.mcp_bridge:
            try:
                await app.state.mcp_bridge.shutdown()
                logger.info("MCP bridge cleaned up")
            except Exception as e:
                logger.warning(f"Error cleaning up MCP bridge: {e}")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")
    
    # Deregister from Hermes
    if hasattr(app.state, "hermes_registration") and app.state.hermes_registration:
        await app.state.hermes_registration.deregister("sophia")
    
    # Give sockets time to close on macOS
    await asyncio.sleep(0.5)
    
    logger.info("Sophia API server shutdown complete")


# Create FastAPI app with OpenAPI configuration
app = FastAPI(
    **get_openapi_configuration(
        component_name=COMPONENT_NAME,
        component_version=COMPONENT_VERSION,
        component_description=COMPONENT_DESCRIPTION
    ),
    lifespan=lifespan
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create standard routers
routers = create_standard_routers(COMPONENT_NAME)

# Add infrastructure endpoints
@routers.root.get("/ready")
async def ready():
    """Readiness check endpoint."""
    # Check if all engines are initialized
    try:
        metrics_engine = await get_metrics_engine()
        analysis_engine = await get_analysis_engine()
        experiment_framework = await get_experiment_framework()
        recommendation_system = await get_recommendation_system()
        intelligence_measurement = await get_intelligence_measurement()
        ml_engine = await get_ml_engine()
        
        all_initialized = (
            metrics_engine.is_initialized and
            analysis_engine.is_initialized and
            experiment_framework.is_initialized and
            recommendation_system.is_initialized and
            intelligence_measurement.is_initialized and
            ml_engine.is_initialized
        )
    except:
        all_initialized = False
    
    ready_check = create_ready_endpoint(
        component_name=COMPONENT_NAME,
        component_version=COMPONENT_VERSION,
        start_time=start_time,
        readiness_check=lambda: all_initialized
    )
    return await ready_check()


@routers.root.get("/discovery")
async def discovery():
    """Service discovery endpoint."""
    discovery_check = create_discovery_endpoint(
        component_name=COMPONENT_NAME,
        component_version=COMPONENT_VERSION,
        component_description=COMPONENT_DESCRIPTION,
        endpoints=[
            EndpointInfo(path="/health", method="GET", description="Health check"),
            EndpointInfo(path="/ready", method="GET", description="Readiness check"),
            EndpointInfo(path="/discovery", method="GET", description="Service discovery"),
            EndpointInfo(path="/api/v1/metrics", method="*", description="Metrics management"),
            EndpointInfo(path="/api/v1/analysis", method="*", description="Analysis operations"),
            EndpointInfo(path="/api/v1/experiments", method="*", description="Experiment management"),
            EndpointInfo(path="/api/v1/recommendations", method="*", description="Recommendation system"),
            EndpointInfo(path="/api/v1/intelligence", method="*", description="Intelligence measurement"),
            EndpointInfo(path="/api/v1/ml", method="*", description="Machine learning operations"),
            EndpointInfo(path="/api/v1/components", method="*", description="Component management"),
            EndpointInfo(path="/api/v1/research", method="*", description="Research projects"),
            EndpointInfo(path="/api/v1/mcp", method="*", description="MCP endpoints"),
            EndpointInfo(path="/ws", method="WS", description="WebSocket for real-time updates")
        ],
        capabilities=[
            "metrics",
            "analysis",
            "experiments",
            "recommendations",
            "intelligence",
            "ml"
        ],
        metadata={
            "category": "analytics",
            "intelligence_dimensions": [dim.value for dim in IntelligenceDimension]
        }
    )
    return await discovery_check()

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
    """Health check endpoint following Tekton standards."""
    try:
        # Get port from config
        config = get_component_config()
        try:
            port = config.sophia.port
        except (AttributeError, TypeError):
            port = int(os.environ.get("SOPHIA_PORT"))
        
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
                "component": "sophia",
                "version": __version__,
                "port": port,
                "message": "Sophia is running normally",
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
                    "component": "sophia",
                    "version": __version__,
                    "port": port,
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
                "component": "sophia",
                "version": __version__,
                "message": f"Health check failed: {str(e)}",
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
        )

# Mount standard routers
mount_standard_routers(app, routers)

# Include business logic routers under v1
routers.v1.include_router(api_router)
app.include_router(system_router)

# Include FastMCP endpoints if available
if fastmcp_available:
    routers.v1.include_router(fastmcp_router, prefix="/mcp", tags=["MCP"])
    logger.info("FastMCP router included at /api/v1/mcp")

if __name__ == "__main__":
    from shared.utils.socket_server import run_component_server
    
    config = get_component_config()
    try:
        port = config.sophia.port
    except (AttributeError, TypeError):
        port = int(os.environ.get("SOPHIA_PORT"))
    
    run_component_server(
        component_name="sophia",
        app_module="sophia.api.app",
        default_port=port,
        reload=False
    )