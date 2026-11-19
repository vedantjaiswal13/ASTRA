"""
Main Application Entry Point
============================
File: app/main.py

Purpose:
--------
Initializes the FastAPI application instance for the ASTRA backend.
Configures CORS middleware to support browser-based frontends,
mounts API routes (/health, /sample-pair, /generate), and serves
the interactive ASTRA research dashboard at root GET `/`.
"""

from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from app.api import api_router

# ------------------------------------------------------------------------------
# FastAPI Application Initialization
# ------------------------------------------------------------------------------
app = FastAPI(
    title="ASTRA Backend API",
    description="Scientific data ingestion and image processing backend for ASTRA.",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ------------------------------------------------------------------------------
# CORS Middleware Configuration
# Allows cross-origin requests from any frontend client or dev server
# ------------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (e.g. localhost:3000, localhost:5173, file://)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],  # Allows all request headers
)

# Path to frontend dashboard HTML
FRONTEND_HTML = Path(__file__).resolve().parent.parent.parent / "astra-frontend" / "index.html"


# ------------------------------------------------------------------------------
# Serve Interactive Dashboard UI at Root
# ------------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_dashboard():
    """Serves the ASTRA research dashboard UI."""
    if FRONTEND_HTML.exists():
        return FileResponse(FRONTEND_HTML)
    return HTMLResponse("<h1>ASTRA Backend Operational</h1><p>Visit <a href='/docs'>/docs</a> for API Swagger UI.</p>")


# ------------------------------------------------------------------------------
# Mount API Routers
# Mounts routes at both root (/) and (/api) for maximum client flexibility
# ------------------------------------------------------------------------------
app.include_router(api_router)
app.include_router(api_router, prefix="/api")


if __name__ == "__main__":
    import uvicorn

    # Run the application server locally with hot-reloading enabled
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
