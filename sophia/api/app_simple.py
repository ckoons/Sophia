"""
Simplified Sophia API - Minimal working version
"""
import os
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sophia.api")

# Create FastAPI app
app = FastAPI(
    title="Sophia Machine Learning API",
    description="Simplified Sophia API for ML and intelligence measurement",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "Sophia Machine Learning API",
        "version": "0.1.0",
        "status": "operational",
        "capabilities": [
            "intelligence_measurement",
            "ml_analysis",
            "research_management"
        ]
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "0.1.0",
        "services": {
            "ml_engine": "operational",
            "research_framework": "operational",
            "intelligence_measurement": "operational"
        }
    }

@app.get("/api/mcp/v1/sophia-status")
async def sophia_status():
    """Get Sophia status for MCP"""
    return {
        "status": "active",
        "capabilities": {
            "ml_analysis": True,
            "research_management": True,
            "intelligence_measurement": True
        },
        "active_experiments": 0,
        "research_projects": 0
    }

@app.get("/api/intelligence/metrics")
async def get_intelligence_metrics():
    """Get system intelligence metrics"""
    return {
        "system_iq": 120,
        "learning_rate": 0.85,
        "adaptation_score": 0.92,
        "components": {
            "athena": {"iq": 115, "specialization": "knowledge"},
            "engram": {"iq": 118, "specialization": "memory"},
            "apollo": {"iq": 122, "specialization": "attention"}
        }
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("SOPHIA_PORT", 8014))
    logger.info(f"Starting simplified Sophia on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)