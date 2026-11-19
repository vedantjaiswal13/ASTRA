"""
NOAA GOES-16 Satellite Data Ingestion Pipeline
==============================================
File: app/data/goes16_fetch.py

Purpose:
--------
Connects to the public NOAA GOES-16 AWS S3 bucket (noaa-goes16) anonymously via HTTPS,
lists, downloads, parses, and normalizes ABI-L2-CMIPF (Cloud and Moisture Imagery
Full Disk) Channel 13 (Clean IR Longwave Window ~10.3 µm) data products using h5py.

Scientific Background & Metadata Guide:
---------------------------------------
1. Brightness Temperature (CMI Variable, Kelvin 'K'):
   - Brightness Temperature (Tb) is the equivalent physical temperature a blackbody
     radiator would need to have in order to produce the observed spectral radiance
     at the sensor's wavelength (~10.35 µm for Channel 13).
   - Band 13 is a "clean" infrared window channel with minimal absorption by atmospheric water vapor:
     * High, thick convective clouds/storm tops appear very COLD (low Tb, ~180K - 230K).
     * Warm ocean and land surfaces appear WARM (high Tb, ~280K - 320K).
     * Space pixels (outside Earth's disc) contain NaN / fill values.
"""

import argparse
import datetime
import json
import logging
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import xml.etree.ElementTree as ET
import urllib.request
import urllib.error

import cv2
import h5py
import numpy as np

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("goes16_ingest")


def _clean_h5_attr(v: Any) -> Any:
    """Helper to convert h5py attributes to standard JSON-serializable primitives."""
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace")
    if isinstance(v, (np.ndarray, list)):
        if len(v) == 1:
            val = v[0]
            if isinstance(val, (np.integer, int)):
                return int(val)
            if isinstance(val, (np.floating, float)):
                return float(val)
            if isinstance(val, bytes):
                return val.decode("utf-8", errors="replace")
            return str(val)
        return [_clean_h5_attr(x) for x in v]
    if isinstance(v, (np.integer, int)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return float(v)
    return str(v)


class GOES16Fetcher:
    """
    Client for querying, downloading, and preprocessing NOAA GOES-16 satellite imagery
    from public AWS S3 bucket (noaa-goes16) via anonymous HTTPS REST requests.
    """

    BUCKET_URL = "https://noaa-goes16.s3.amazonaws.com"
    BUCKET_NAME = "noaa-goes16"
    PRODUCT = "ABI-L2-CMIPF"  # ABI Level 2 Cloud and Moisture Imagery Full Disk

    def __init__(self, s3_client=None):
        logger.info("Initialized GOES16Fetcher with public NOAA S3 HTTPS connection (%s).", self.BUCKET_URL)

    def ping(self) -> Dict[str, Any]:
        """Check live connection latency and status to NOAA AWS S3."""
        t0 = time.time()
        try:
            req = urllib.request.Request(
                f"{self.BUCKET_URL}/?prefix={self.PRODUCT}/2024/001/18/",
                headers={"User-Agent": "ASTRA-Satellite-Workbench/2.0"}
            )
            with urllib.request.urlopen(req, timeout=8) as res:
                latency_ms = round((time.time() - t0) * 1000, 1)
                return {
                    "status": "connected",
                    "http_code": res.status,
                    "bucket": self.BUCKET_NAME,
                    "region": "us-east-1",
                    "latency_ms": latency_ms,
                    "endpoint": self.BUCKET_URL
                }
        except Exception as e:
            logger.warning("Ping to NOAA S3 failed: %s", e)
            return {
                "status": "error",
                "error": str(e),
                "bucket": self.BUCKET_NAME,
                "endpoint": self.BUCKET_URL
            }

    def list_channel_files(
        self,
        year: int,
        day_of_year: int,
        hour: int,
        channel: int = 13
    ) -> List[str]:
        """
        List available ABI NetCDF scan files for a specified date, hour, and channel
        from the public NOAA S3 bucket.
        """
        prefix = f"{self.PRODUCT}/{year:04d}/{day_of_year:03d}/{hour:02d}/"
        url = f"{self.BUCKET_URL}/?prefix={prefix}"
        logger.info("Listing S3 files via HTTP: %s", url)

        req = urllib.request.Request(url, headers={"User-Agent": "ASTRA-Satellite-Workbench/2.0"})
        with urllib.request.urlopen(req, timeout=15) as res:
            xml_data = res.read()

        root = ET.fromstring(xml_data)
        ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
        channel_tag = f"C{channel:02d}"

        all_keys = [elem.text for elem in root.findall(".//s3:Key", ns) if elem.text]
        channel_files = sorted([k for k in all_keys if channel_tag in k])

        logger.info(
            "Found %d files for Channel %02d at %04d-DOY%03d %02d:00 UTC",
            len(channel_files), channel, year, day_of_year, hour
        )
        return channel_files

    def select_triplet_scans(self, files: List[str]) -> Tuple[str, str, str]:
        """
        Selects two boundary scans and one scan roughly halfway between them in time.
        """
        if len(files) < 3:
            raise ValueError(
                f"Need at least 3 scan files to select a triplet with a halfway scan. Found {len(files)}."
            )

        idx_start = 0
        idx_end = 2 if len(files) >= 3 else len(files) - 1
        idx_mid = (idx_start + idx_end) // 2

        selected = (files[idx_start], files[idx_mid], files[idx_end])
        logger.info(
            "Selected scan triplet:\n  [Start]: %s\n  [Mid  ]: %s\n  [End  ]: %s",
            selected[0].split("/")[-1],
            selected[1].split("/")[-1],
            selected[2].split("/")[-1],
        )
        return selected

    def download_file(self, s3_path: str, destination_dir: Path) -> Path:
        """
        Download a NetCDF file from NOAA public S3 to local filesystem via HTTPS.
        """
        destination_dir.mkdir(parents=True, exist_ok=True)
        filename = s3_path.split("/")[-1]
        local_path = destination_dir / filename

        if local_path.exists() and local_path.stat().st_size > 1024 * 1024:
            logger.info("File already exists locally: %s (%d bytes)", local_path.name, local_path.stat().st_size)
            return local_path

        # If s3_path starts with bucket name, strip it
        clean_key = s3_path
        if clean_key.startswith(f"{self.BUCKET_NAME}/"):
            clean_key = clean_key[len(self.BUCKET_NAME) + 1:]

        file_url = f"{self.BUCKET_URL}/{clean_key}"
        logger.info("Downloading NOAA S3 object via HTTP: %s -> %s", file_url, local_path)

        req = urllib.request.Request(file_url, headers={"User-Agent": "ASTRA-Satellite-Workbench/2.0"})
        with urllib.request.urlopen(req, timeout=60) as response, open(local_path, "wb") as out_f:
            chunk_size = 1024 * 1024
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                out_f.write(chunk)

        logger.info("Downloaded %s successfully (%d bytes)", local_path.name, local_path.stat().st_size)
        return local_path

    @staticmethod
    def process_netcdf(
        nc_path: Path,
        output_dir: Path,
        save_prefix: str = "goes16_c13"
    ) -> Dict:
        """
        Opens a NetCDF-4 file with h5py, extracts the CMI (brightness temperature) variable,
        reads and formats metadata, normalizes to 0-255 grayscale, and saves as .npy and .png.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        base_stem = nc_path.stem

        logger.info("Reading NetCDF file with h5py: %s", nc_path.name)
        with h5py.File(nc_path, "r") as f:
            cmi_ds = f["CMI"]
            sf = float(cmi_ds.attrs.get("scale_factor", [1.0])[0])
            ao = float(cmi_ds.attrs.get("add_offset", [0.0])[0])
            fill_val = int(cmi_ds.attrs.get("_FillValue", [-1])[0])

            raw_cmi = cmi_ds[:]
            valid_mask = raw_cmi != fill_val

            cmi_data = np.full(raw_cmi.shape, np.nan, dtype=np.float32)
            cmi_data[valid_mask] = raw_cmi[valid_mask].astype(np.float32) * sf + ao

            # Extract projection metadata
            proj = f["goes_imager_projection"]
            projection_info = {
                "grid_mapping_name": _clean_h5_attr(proj.attrs.get("grid_mapping_name", "geostationary")),
                "perspective_point_height_m": float(_clean_h5_attr(proj.attrs.get("perspective_point_height", 35786023.0))),
                "semi_major_axis_m": float(_clean_h5_attr(proj.attrs.get("semi_major_axis", 6378137.0))),
                "semi_minor_axis_m": float(_clean_h5_attr(proj.attrs.get("semi_minor_axis", 6356752.31414))),
                "inverse_flattening": float(_clean_h5_attr(proj.attrs.get("inverse_flattening", 298.2572221))),
                "latitude_of_projection_origin_deg": float(_clean_h5_attr(proj.attrs.get("latitude_of_projection_origin", 0.0))),
                "longitude_of_projection_origin_deg": float(_clean_h5_attr(proj.attrs.get("longitude_of_projection_origin", -75.0))),
                "sweep_angle_axis": _clean_h5_attr(proj.attrs.get("sweep_angle_axis", "x")),
            }

            metadata = {
                "source_file": nc_path.name,
                "product": _clean_h5_attr(f.attrs.get("title", "ABI L2+ Cloud and Moisture Imagery")),
                "channel": 13,
                "channel_name": "Clean IR Longwave Window (~10.3 µm)",
                "time_coverage_start": _clean_h5_attr(f.attrs.get("time_coverage_start", "")),
                "time_coverage_end": _clean_h5_attr(f.attrs.get("time_coverage_end", "")),
                "spatial_resolution": _clean_h5_attr(cmi_ds.attrs.get("resolution", "2km at nadir")),
                "units": _clean_h5_attr(cmi_ds.attrs.get("units", "K")),
                "raw_shape": list(cmi_data.shape),
                "projection_info": projection_info,
            }

        # Grayscale normalization
        if np.any(valid_mask):
            t_min = float(np.nanmin(cmi_data[valid_mask]))
            t_max = float(np.nanmax(cmi_data[valid_mask]))
            norm_data = np.zeros_like(cmi_data, dtype=np.float32)
            norm_data[valid_mask] = (cmi_data[valid_mask] - t_min) / (t_max - t_min + 1e-6)
            norm_uint8 = np.clip(norm_data * 255.0, 0, 255).astype(np.uint8)
        else:
            t_min, t_max = 0.0, 0.0
            norm_uint8 = np.zeros(cmi_data.shape, dtype=np.uint8)

        metadata["stats"] = {
            "min_brightness_temp_kelvin": t_min,
            "max_brightness_temp_kelvin": t_max,
            "valid_pixels_count": int(np.count_nonzero(valid_mask)),
        }

        npy_path = output_dir / f"{save_prefix}_{base_stem}.npy"
        png_path = output_dir / f"{save_prefix}_{base_stem}.png"
        json_path = output_dir / f"{save_prefix}_{base_stem}_meta.json"

        np.save(npy_path, norm_uint8)
        cv2.imwrite(str(png_path), norm_uint8)

        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(metadata, jf, indent=2)

        logger.info("Saved normalized products: %s, %s", npy_path.name, png_path.name)
        return {
            "metadata": metadata,
            "npy_path": str(npy_path),
            "png_path": str(png_path),
            "json_path": str(json_path),
        }


# ----------------------------------------------------------------------
# Curated Default Dataset: 50 diverse triplets across 10 dates in 2024
# ----------------------------------------------------------------------
DEFAULT_DATE_HOUR_PAIRS: List[Tuple[str, int]] = [
    ("2024-01-15", 0), ("2024-01-15", 6), ("2024-01-15", 12), ("2024-01-15", 18), ("2024-01-15", 21),
    ("2024-02-15", 0), ("2024-02-15", 6), ("2024-02-15", 12), ("2024-02-15", 18), ("2024-02-15", 21),
    ("2024-03-15", 0), ("2024-03-15", 6), ("2024-03-15", 12), ("2024-03-15", 18), ("2024-03-15", 21),
    ("2024-04-15", 0), ("2024-04-15", 6), ("2024-04-15", 12), ("2024-04-15", 18), ("2024-04-15", 21),
    ("2024-05-15", 0), ("2024-05-15", 6), ("2024-05-15", 12), ("2024-05-15", 18), ("2024-05-15", 21),
    ("2024-06-15", 0), ("2024-06-15", 6), ("2024-06-15", 12), ("2024-06-15", 18), ("2024-06-15", 21),
    ("2024-07-15", 0), ("2024-07-15", 6), ("2024-07-15", 12), ("2024-07-15", 18), ("2024-07-15", 21),
    ("2024-08-15", 0), ("2024-08-15", 6), ("2024-08-15", 12), ("2024-08-15", 18), ("2024-08-15", 21),
    ("2024-09-15", 0), ("2024-09-15", 6), ("2024-09-15", 12), ("2024-09-15", 18), ("2024-09-15", 21),
    ("2024-10-15", 0), ("2024-10-15", 6), ("2024-10-15", 12), ("2024-10-15", 18), ("2024-10-15", 21),
]


def parse_date_input(date_input: Union[str, datetime.date, datetime.datetime]) -> Tuple[int, int, str]:
    if isinstance(date_input, (datetime.date, datetime.datetime)):
        return date_input.year, date_input.timetuple().tm_yday, date_input.strftime("%Y-%m-%d")
    elif isinstance(date_input, str):
        s = date_input.strip()
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d"):
            try:
                dt = datetime.datetime.strptime(s, fmt).date()
                return dt.year, dt.timetuple().tm_yday, dt.strftime("%Y-%m-%d")
            except ValueError:
                pass
        parts = s.split("-")
        if len(parts) == 2:
            try:
                year = int(parts[0])
                doy = int(parts[1])
                dt = datetime.date(year, 1, 1) + datetime.timedelta(days=doy - 1)
                return year, doy, dt.strftime("%Y-%m-%d")
            except Exception:
                pass
        raise ValueError(f"Unrecognized date format: '{date_input}'. Expected YYYY-MM-DD or YYYY-DOY.")
    else:
        raise TypeError(f"Date must be a string or datetime.date, got {type(date_input)}")


def download_single_triplet(
    fetcher: GOES16Fetcher,
    year: int,
    day_of_year: int,
    hour: int,
    triplet_dir: Path,
    channel: int = 13,
    keep_raw_nc: bool = False,
    date_str: Optional[str] = None
) -> Dict[str, Any]:
    triplet_dir.mkdir(parents=True, exist_ok=True)
    raw_nc_dir = triplet_dir / "raw_netcdf"

    files = fetcher.list_channel_files(year=year, day_of_year=day_of_year, hour=hour, channel=channel)
    if len(files) < 3:
        raise ValueError(
            f"Insufficient Channel {channel} files at Year {year}, DOY {day_of_year}, Hour {hour} UTC (found {len(files)})."
        )

    triplet_s3_paths = fetcher.select_triplet_scans(files)
    labels = ["scan_0_start", "scan_1_halfway", "scan_2_end"]
    keys = ["t0", "t_mid", "t1"]
    file_records = {}

    for idx, (label, key, s3_path) in enumerate(zip(labels, keys, triplet_s3_paths)):
        local_nc = fetcher.download_file(s3_path, raw_nc_dir)
        processed = fetcher.process_netcdf(
            nc_path=local_nc,
            output_dir=triplet_dir,
            save_prefix=f"{idx:02d}_{label}"
        )
        if not keep_raw_nc and local_nc.exists():
            try:
                local_nc.unlink()
            except Exception:
                pass

        file_records[key] = {
            "label": label,
            "npy_path": str(processed["npy_path"]),
            "png_path": str(processed["png_path"]),
            "json_path": str(processed["json_path"]),
            "timestamp": processed["metadata"].get("time_coverage_start", ""),
            "source_file": processed["metadata"].get("source_file", ""),
        }

    date_label = date_str or f"{year}-{day_of_year:03d}"
    triplet_id = f"{date_label}_{hour:02d}"

    return {
        "triplet_id": triplet_id,
        "date": date_label,
        "hour": hour,
        "year": year,
        "day_of_year": day_of_year,
        "channel": channel,
        "directory": str(triplet_dir),
        "files": file_records,
    }


def batch_download_triplets(
    date_hour_pairs: Optional[List[Tuple[str, int]]] = None,
    output_base_dir: Union[str, Path] = "sample_data/triplets",
    channel: int = 13,
    keep_raw_nc: bool = False,
    max_triplets: Optional[int] = None,
    overwrite: bool = False
) -> Dict[str, Any]:
    base_dir = Path(output_base_dir).resolve()
    base_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = base_dir / "manifest.json"

    pairs = list(date_hour_pairs or DEFAULT_DATE_HOUR_PAIRS)
    if max_triplets is not None and max_triplets > 0:
        pairs = pairs[:max_triplets]

    fetcher = GOES16Fetcher()
    manifest_data = {
        "title": "NOAA GOES-16 ABI-L2-CMIPF Channel 13 Triplets Manifest",
        "description": "Catalog of downloaded GOES-16 Clean IR Channel 13 scan triplets.",
        "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "summary": {"total_target": len(pairs), "attempted": 0, "succeeded": 0, "failed": 0},
        "triplets": [],
        "failed_attempts": []
    }

    if manifest_path.exists() and not overwrite:
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
                manifest_data["triplets"] = existing.get("triplets", [])
        except Exception:
            pass

    for date_str, hour in pairs:
        triplet_id = f"{date_str}_{hour:02d}"
        triplet_dir = base_dir / triplet_id
        manifest_data["summary"]["attempted"] += 1

        if not overwrite and any(t.get("triplet_id") == triplet_id for t in manifest_data["triplets"]):
            logger.info("Triplet %s already recorded in manifest, skipping.", triplet_id)
            manifest_data["summary"]["succeeded"] += 1
            continue

        try:
            year, doy, d_str = parse_date_input(date_str)
            record = download_single_triplet(
                fetcher=fetcher,
                year=year,
                day_of_year=doy,
                hour=hour,
                triplet_dir=triplet_dir,
                channel=channel,
                keep_raw_nc=keep_raw_nc,
                date_str=d_str
            )
            manifest_data["triplets"].append(record)
            manifest_data["summary"]["succeeded"] += 1
            logger.info("Successfully ingested triplet: %s", triplet_id)
        except Exception as e:
            logger.error("Failed downloading triplet %s: %s", triplet_id, e)
            manifest_data["failed_attempts"].append({"triplet_id": triplet_id, "error": str(e)})
            manifest_data["summary"]["failed"] += 1

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    return manifest_data


def run_ingestion_pipeline(
    year: int = 2024,
    day_of_year: int = 1,
    hour: int = 18,
    channel: int = 13,
    output_dir: Union[str, Path] = "sample_data",
    keep_raw_nc: bool = False
) -> List[Dict]:
    out_path = Path(output_dir).resolve()
    raw_nc_dir = out_path / "raw_netcdf"

    fetcher = GOES16Fetcher()
    files = fetcher.list_channel_files(year=year, day_of_year=day_of_year, hour=hour, channel=channel)
    if not files:
        raise FileNotFoundError(f"No Channel {channel} files found for Year {year}, DOY {day_of_year}, Hour {hour} UTC.")

    triplet_s3_paths = fetcher.select_triplet_scans(files)
    results = []
    labels = ["scan_0_start", "scan_1_halfway", "scan_2_end"]

    for idx, (label, s3_path) in enumerate(zip(labels, triplet_s3_paths)):
        logger.info("--- Processing [%s] ---", label)
        local_nc = fetcher.download_file(s3_path, raw_nc_dir)
        processed = fetcher.process_netcdf(
            nc_path=local_nc,
            output_dir=out_path,
            save_prefix=f"{idx:02d}_{label}"
        )
        results.append(processed)
        if not keep_raw_nc and local_nc.exists():
            try:
                local_nc.unlink()
            except Exception:
                pass

    logger.info("GOES-16 Ingestion pipeline completed successfully! Output in: %s", out_path)
    return results


def main():
    parser = argparse.ArgumentParser(description="NOAA GOES-16 ABI-L2-CMIPF Channel 13 S3 Data Ingestion Pipeline")
    parser.add_argument("--date", type=str, default="2024-01-01", help="Date YYYY-MM-DD")
    parser.add_argument("--hour", type=int, default=18, help="UTC Hour (0-23)")
    parser.add_argument("--channel", type=int, default=13, help="Spectral Channel (default: 13)")
    parser.add_argument("--output-dir", type=str, default="sample_data", help="Output directory")
    parser.add_argument("--ping", action="store_true", help="Ping NOAA S3 and check latency")

    args = parser.parse_args()
    fetcher = GOES16Fetcher()

    if args.ping:
        status = fetcher.ping()
        print("NOAA S3 Status:", json.dumps(status, indent=2))
        return

    y, d, dt_str = parse_date_input(args.date)
    run_ingestion_pipeline(year=y, day_of_year=d, hour=args.hour, channel=args.channel, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
