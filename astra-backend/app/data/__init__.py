"""
Data Ingestion Package
======================
File: app/data/__init__.py

Purpose:
--------
Houses modules responsible for downloading, streaming, and ingesting raw geospatial,
satellite, and scientific datasets (e.g., NetCDF, HDF5, GeoTIFF, S3 buckets).

Exports:
--------
- GOES16Fetcher: Client for anonymous NOAA GOES-16 S3 downloads.
- run_ingestion_pipeline: Complete download, extraction, and preprocessing pipeline.
"""

from app.data.goes16_fetch import (
    DEFAULT_DATE_HOUR_PAIRS,
    GOES16Fetcher,
    batch_download_triplets,
    download_single_triplet,
    run_ingestion_pipeline,
)
from app.data.ingestion import DataIngestionService

__all__ = [
    "GOES16Fetcher",
    "run_ingestion_pipeline",
    "batch_download_triplets",
    "download_single_triplet",
    "DEFAULT_DATE_HOUR_PAIRS",
    "DataIngestionService",
]
