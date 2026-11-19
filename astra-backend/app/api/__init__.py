"""
API Routing Module
==================
File: app/api/__init__.py

Purpose:
--------
This package defines all REST API route definitions, endpoint controllers, and request/response models.
Endpoints defined in submodules (such as `routes.py`) are aggregated here into an `APIRouter` instance
to be mounted onto the primary FastAPI application in `app/main.py`.
"""

from fastapi import APIRouter
from app.api.routes import router as core_router

# Primary API Router aggregating route submodules
api_router = APIRouter()

# Include core routes (e.g. /health, etc.)
api_router.include_router(core_router)
