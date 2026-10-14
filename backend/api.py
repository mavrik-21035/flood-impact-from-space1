"""HTTP wrapper around pipeline.py, analysis.py, and report.py."""

import asyncio
import json
import re
import shutil
import sys
import uuid
from datetime import date
from enum import Enum
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus
from urllib.request import Request as UrllibRequest, urlopen

import numpy as np
import rasterio
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, Field, field_validator, model_validator
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.warp import reproject
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "runs"
PIPELINE = ROOT / "pipeline.py"
ANALYSIS = ROOT / "analysis.py"
REPORT = ROOT / "report.py"
TILE_ORIGIN = 20037508.342789244
TILE_SPAN = TILE_ORIGIN * 2
TILE_SIZE = 256

app = FastAPI(title="Flood analysis API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
        # Docker Compose (nginx on host port 3000)
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost",
        "http://127.0.0.1",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

RUNS: dict[str, dict[str, Any]] = {}
LAYER_FILES = {
    "flood_mask": "flood_extent.geojson",
    "damaged_buildings": "buildings.geojson",
    "roads": "roads.geojson",
    "bridges": "bridges.geojson",
    "settlements": "settlements.geojson",
}


def _suggest_missing_decimal(value: float | str, axis: str) -> str | None:
    if value in (None, ""):
        return None
    clean = str(value).strip().replace(",", "")
    if not re.fullmatch(r"[-+]?\d+(?:\.\d+)?", clean):
        return None
    numeric = float(clean)
    max_abs = 90 if axis == "lat" else 180
    if abs(numeric) <= max_abs:
        return None
    digits = re.sub(r"[^0-9]", "", clean)
    if len(digits) < 4:
        return None
    decimal_places = 3 if len(digits) >= 5 else 2
    candidate = float(f"{digits[:-decimal_places]}.{digits[-decimal_places:]}")
    if abs(candidate) <= max_abs:
        return str(candidate)
    return None


class RunRequest(BaseModel):
    west: float = Field(ge=-180, le=180)
    east: float = Field(ge=-180, le=180)
    south: float = Field(ge=-90, le=90)
    north: float = Field(ge=-90, le=90)
    flood_date: date
    raw_flood_class_id: int
    checkpoint: str
    channel_names: str
    input_scale: str

    @field_validator("west", "east", mode="before")
    @classmethod
    def normalize_longitude(cls, value):
        if isinstance(value, str):
            value = value.strip().replace(",", "")
        if value in (None, ""):
            raise ValueError("Longitude is required")
        return value

    @field_validator("west", "east")
    @classmethod
    def validate_longitude(cls, value: float):
        if not -180 <= value <= 180:
            suggestion = _suggest_missing_decimal(value, "lon")
            if suggestion:
                raise ValueError(f"Longitude looks like a missing decimal point. Did you mean {suggestion}?")
            raise ValueError("Longitude must be between -180 and 180 degrees")
        return value

    @field_validator("south", "north", mode="before")
    @classmethod
    def normalize_latitude(cls, value):
        if isinstance(value, str):
            value = value.strip().replace(",", "")
        if value in (None, ""):
            raise ValueError("Latitude is required")
        return value

    @field_validator("south", "north")
    @classmethod
    def validate_latitude(cls, value: float):
        if not -90 <= value <= 90:
            suggestion = _suggest_missing_decimal(value, "lat")
            if suggestion:
                raise ValueError(f"Latitude looks like a missing decimal point. Did you mean {suggestion}?")
            raise ValueError("Latitude must be between -90 and 90 degrees")
        return value

    @model_validator(mode="after")
    def validate_bbox_order(self):
        if self.west >= self.east:
            raise ValueError("west must be less than east")
        if self.south >= self.north:
            raise ValueError("south must be less than north")
        return self


class RunStatus(str, Enum):
    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"


def _run_dir(run_id: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{32}", run_id):
        raise HTTPException(status_code=404, detail="Unknown run")
    return RUNS_DIR / run_id


def _set_progress(run_id: str, progress: int, stage: str, status: RunStatus | None = None):
    state = RUNS.get(run_id)
    if state is None:
        return
    state["progress"] = progress
    state["stage"] = stage
    if status is not None:
        state["status"] = status.value


async def _execute(command: list[str], run_id: str, progress: int, stage: str) -> str:
    _set_progress(run_id, progress, stage, RunStatus.running)
    process = await asyncio.create_subprocess_exec(
        *command,
        cwd=ROOT,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    output, _ = await process.communicate()
    text = output.decode("utf-8", errors="replace")
    if process.returncode:
        tail = text[-8000:]
        raise RuntimeError(f"{stage} failed (exit {process.returncode}).\n{tail}")
    return text


def _read_collection(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Expected analysis output is missing: {path.name}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("type") != "FeatureCollection":
        raise ValueError(f"Unexpected GeoJSON type in {path.name}")
    return payload


def _feature_id(feature: dict[str, Any]) -> str:
    properties = feature.get("properties") or {}
    value = properties.get("osm_id", feature.get("id", ""))
    return str(value)


def _join_layer(source: dict[str, Any], hits: dict[str, Any], flag: str) -> dict[str, Any]:
    hit_by_id = {_feature_id(feature): feature for feature in hits.get("features", [])}
    joined = []
    for original in source.get("features", []):
        feature = dict(original)
        properties = dict(feature.get("properties") or {})
        hit = hit_by_id.get(_feature_id(feature))
        properties[flag] = hit is not None
        if hit is not None:
            hit_properties = hit.get("properties") or {}
            if "geometric_hit_confidence" in hit_properties:
                properties["geometric_hit_confidence"] = hit_properties["geometric_hit_confidence"]
            if "affected_length_m" in hit_properties:
                properties["affected_length_m"] = hit_properties["affected_length_m"]
        feature["properties"] = properties
        joined.append(feature)
    return {**source, "features": joined}


def _settlement_layer(analysis_dir: Path) -> dict[str, Any]:
    groups = (
        ("settlements_connected.geojson", "connected", False),
        ("settlements_disconnected.geojson", "disconnected", True),
        ("settlements_no_baseline_route.geojson", "no_baseline_path", None),
    )
    features: dict[str, dict[str, Any]] = {}
    for filename, connectivity, cut_off in groups:
        for feature in _read_collection(analysis_dir / filename).get("features", []):
            feature = dict(feature)
            properties = dict(feature.get("properties") or {})
            properties["connectivity"] = connectivity
            properties["cut_off"] = cut_off
            feature["properties"] = properties
            features[_feature_id(feature)] = feature
    return {
        "type": "FeatureCollection",
        "name": "settlements",
        "features": list(features.values()),
    }


def _prepare_web_layers(analysis_dir: Path) -> None:
    web_dir = analysis_dir / "web"
    web_dir.mkdir(parents=True, exist_ok=True)
    buildings = _join_layer(
        _read_collection(analysis_dir / "osm_buildings.geojson"),
        _read_collection(analysis_dir / "buildings_hit.geojson"),
        "cut",
    )
    roads = _join_layer(
        _read_collection(analysis_dir / "osm_roads.geojson"),
        _read_collection(analysis_dir / "roads_hit.geojson"),
        "cut",
    )
    bridges = _join_layer(
        _read_collection(analysis_dir / "osm_bridges.geojson"),
        _read_collection(analysis_dir / "bridges_hit.geojson"),
        "cut",
    )
    for feature in buildings.get("features", []):
        properties = feature.get("properties") or {}
        tags = properties.get("tags") or {}
        height = tags.get("height")
        levels = tags.get("building:levels")
        try:
            properties["height_m"] = float(re.match(r"\s*([0-9]+(?:\.[0-9]+)?)", str(height)).group(1)) if height else None
        except (AttributeError, ValueError):
            properties["height_m"] = None
        try:
            properties["building_levels"] = float(levels) if levels else None
        except ValueError:
            properties["building_levels"] = None

    settlements = _settlement_layer(analysis_dir)
    flood = _read_collection(analysis_dir / LAYER_FILES["flood_mask"])
    for key, payload in (
        ("damaged_buildings", buildings),
        ("roads", roads),
        ("bridges", bridges),
        ("settlements", settlements),
        ("flood_mask", flood),
    ):
        (web_dir / LAYER_FILES[key]).write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )


def _add_flood_area(summary_path: Path, mask_path: Path, flood_class: int, bbox, event_date) -> dict[str, Any]:
    with rasterio.open(mask_path) as mask:
        pixels = mask.read(1)
        crs = rasterio.crs.CRS.from_user_input(mask.crs)
        if not crs.is_projected:
            raise ValueError("Flood-mask CRS must be projected to calculate area in square kilometres")
        area_per_pixel = abs(mask.transform.a * mask.transform.e - mask.transform.b * mask.transform.d)
        area_per_pixel *= crs.linear_units_factor[1] ** 2
        flooded_area_km2 = float(np.count_nonzero(pixels == flood_class) * area_per_pixel / 1_000_000)

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.setdefault("impacts", {})["flooded_area_km2"] = round(flooded_area_km2, 6)
    summary["event_date"] = event_date.isoformat()
    summary["requested_area_bbox_wgs84"] = list(bbox)
    summary["flood_mask"] = str(mask_path)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


async def _run_job(run_id: str, request_data: RunRequest) -> None:
    run_dir = _run_dir(run_id)
    analysis_dir = run_dir / "analysis"
    mask_path = run_dir / "flood_mask.tif"
    work_dir = run_dir / "work"
    bbox = (request_data.west, request_data.south, request_data.east, request_data.north)
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        pipeline_command = [
            sys.executable, str(PIPELINE),
            str(request_data.west), str(request_data.south),
            str(request_data.east), str(request_data.north),
            request_data.flood_date.isoformat(),
            "--checkpoint", str(Path(request_data.checkpoint).expanduser()),
            "--output", str(mask_path),
            "--work-dir", str(work_dir),
            "--trained-channel-names", request_data.channel_names,
            "--trained-input-scale", request_data.input_scale,
        ]
        await _execute(pipeline_command, run_id, 5, "Sentinel-1 search, preprocessing, and inference")

        analysis_command = [
            sys.executable, str(ANALYSIS), str(mask_path),
            "--flood-class", str(request_data.raw_flood_class_id),
            "--output-dir", str(analysis_dir),
        ]
        await _execute(analysis_command, run_id, 55, "ohsome impact and connectivity analysis")
        await asyncio.to_thread(
            _prepare_web_layers, analysis_dir
        )
        summary_path = analysis_dir / "summary.json"
        summary = await asyncio.to_thread(
            _add_flood_area, summary_path, mask_path,
            request_data.raw_flood_class_id, bbox, request_data.flood_date,
        )

        _set_progress(run_id, 88, "Generating bilingual PDF report", RunStatus.running)
        report_path = run_dir / "situation_report.pdf"
        await _execute(
            [sys.executable, str(REPORT), str(summary_path), "--output", str(report_path)],
            run_id, 88, "Generating bilingual PDF report",
        )
        state = RUNS[run_id]
        state.update({
            "status": RunStatus.completed.value,
            "progress": 100,
            "stage": "Complete",
            "error": None,
            "summary": summary,
        })
    except Exception as error:
        state = RUNS[run_id]
        state.update({
            "status": RunStatus.failed.value,
            "stage": "Failed",
            "error": str(error),
        })


KNOWN_PLACES = {
    "nepal": (28.3949, 84.1240, "Nepal (Country Overview)"),
    "syaphrubesi": (28.1601, 85.3374, "Syabrubesi (Syaphrubesi), Rasuwa, Bagmati, Nepal"),
    "syabrubesi": (28.1601, 85.3374, "Syabrubesi, Rasuwa, Bagmati, Nepal"),
    "syaphru": (28.1402, 85.3520, "Syaphru, Rasuwa, Bagmati, Nepal"),
    "dhunche": (28.1132, 85.3028, "Dhunche, Rasuwa District HQ, Bagmati, Nepal"),
    "trishuli": (27.9575, 85.1481, "Trishuli Bazar, Nuwakot, Bagmati, Nepal"),
    "trishuli bazar": (27.9575, 85.1481, "Trishuli Bazar, Nuwakot, Bagmati, Nepal"),
    "bidur": (27.9130, 85.1630, "Bidur, Nuwakot District HQ, Bagmati, Nepal"),
    "betrawati": (27.9860, 85.1830, "Betrawati, Nuwakot, Bagmati, Nepal"),
    "mailung": (28.0580, 85.2360, "Mailung, Rasuwa, Bagmati, Nepal"),
    "ramche": (28.0670, 85.2670, "Ramche, Rasuwa, Bagmati, Nepal"),
    "timure": (28.2780, 85.3780, "Timure (Rasuwagadhi Border), Rasuwa, Nepal"),
    "rasuwagadhi": (28.2830, 85.3830, "Rasuwagadhi Border, Rasuwa, Nepal"),
    "langtang": (28.2170, 85.5010, "Langtang Valley, Rasuwa, Nepal"),
    "kathmandu": (27.7172, 85.3240, "Kathmandu, Bagmati, Nepal"),
    "lalitpur": (27.6710, 85.3250, "Lalitpur (Patan), Bagmati, Nepal"),
    "bhaktapur": (27.6710, 85.4298, "Bhaktapur, Bagmati, Nepal"),
    "pokhara": (28.2096, 83.9856, "Pokhara, Gandaki, Nepal"),
    "melamchi": (27.8300, 85.5800, "Melamchi, Sindhupalchok, Nepal"),
    "barhabise": (27.7900, 85.9000, "Barhabise, Sindhupalchok, Nepal"),
    "bhote koshi": (28.1800, 85.3500, "Bhote Koshi River Corridor, Rasuwa, Nepal"),
    "bhotekoshi": (28.1800, 85.3500, "Bhote Koshi River Corridor, Rasuwa, Nepal"),
    "biratnagar": (26.4525, 87.2718, "Biratnagar, Morang, Koshi, Nepal"),
    "birgunj": (27.0104, 84.8774, "Birgunj, Parsa, Madhesh, Nepal"),
    "bharatpur": (27.6833, 84.4333, "Bharatpur, Chitwan, Bagmati, Nepal"),
    "chitwan": (27.6833, 84.4333, "Chitwan (Bharatpur), Bagmati, Nepal"),
    "dharan": (26.8126, 87.2834, "Dharan, Sunsari, Koshi, Nepal"),
    "janakpur": (26.7288, 85.9244, "Janakpurdham, Dhanusha, Madhesh, Nepal"),
    "hetauda": (27.4287, 85.0331, "Hetauda, Makwanpur, Bagmati, Nepal"),
    "butwal": (27.7006, 83.4484, "Butwal, Rupandehi, Lumbini, Nepal"),
    "nepalgunj": (28.0500, 81.6167, "Nepalgunj, Banke, Lumbini, Nepal"),
    "dhangadhi": (28.6852, 80.6078, "Dhangadhi, Kailali, Sudurpashchim, Nepal"),
    "itahari": (26.6667, 87.2833, "Itahari, Sunsari, Koshi, Nepal"),
    "birendranagar": (28.6019, 81.6339, "Birendranagar, Surkhet, Karnali, Nepal"),
    "surkhet": (28.6019, 81.6339, "Surkhet (Birendranagar), Karnali, Nepal"),
    "gorkha": (28.0000, 84.6300, "Gorkha, Gandaki, Nepal"),
    "besisahar": (28.2300, 84.3750, "Besisahar, Lamjung, Gandaki, Nepal"),
    "beni": (28.3440, 83.5650, "Beni, Myagdi, Gandaki, Nepal"),
    "baglung": (28.2720, 83.5930, "Baglung, Gandaki, Nepal"),
    "tansen": (27.8670, 83.5500, "Tansen, Palpa, Lumbini, Nepal"),
    "bhairahawa": (27.5050, 83.4500, "Bhairahawa, Rupandehi, Lumbini, Nepal"),
    "damak": (26.6600, 87.7000, "Damak, Jhapa, Koshi, Nepal"),
    "birtamod": (26.6400, 87.9900, "Birtamod, Jhapa, Koshi, Nepal"),
    "lahan": (26.7200, 86.4800, "Lahan, Siraha, Madhesh, Nepal"),
    "rajbiraj": (26.5400, 86.7500, "Rajbiraj, Saptari, Koshi, Nepal"),
    "gaur": (26.7600, 85.2700, "Gaur, Rautahat, Madhesh, Nepal"),
    "kalaiya": (27.0300, 85.0000, "Kalaiya, Bara, Madhesh, Nepal"),
    "malangwa": (26.8500, 85.5600, "Malangwa, Sarlahi, Madhesh, Nepal"),
    "jaleshwar": (26.6400, 85.8000, "Jaleshwar, Mahottari, Madhesh, Nepal"),
    "kapilvastu": (27.5500, 83.0500, "Kapilvastu, Lumbini, Nepal"),
    "gulariya": (28.2300, 81.3300, "Gulariya, Bardiya, Lumbini, Nepal"),
    "tikapur": (28.5200, 81.1300, "Tikapur, Kailali, Sudurpashchim, Nepal"),
    "mahendranagar": (28.9600, 80.1800, "Mahendranagar, Kanchanpur, Nepal"),
    "darchula": (29.8400, 80.5300, "Darchula, Sudurpashchim, Nepal"),
    "kerala": (10.8505, 76.2711, "Kerala, India"),
}


@app.get("/places")
async def list_places(q: str = ""):
    query = (q or "").strip().lower()
    matches = []
    for key, (lat, lon, name) in KNOWN_PLACES.items():
        if not query or query in key or query in name.lower():
            matches.append({
                "id": key,
                "name": name.split(",")[0],
                "display_name": name,
                "latitude": lat,
                "longitude": lon,
            })
        if len(matches) >= 12:
            break
    return matches


@app.get("/geocode")
async def geocode_place(q: str):
    query = (q or "").strip()
    if not query:
        raise HTTPException(status_code=422, detail="Place name is required")

    normalized = re.sub(r"[,\.\-]+", " ", query.lower()).strip()
    normalized_no_country = re.sub(r"\b(nepal|india)\b", "", normalized).strip()

    # Exact or stripped match
    for key, (lat, lon, name) in KNOWN_PLACES.items():
        if key == normalized or (normalized_no_country and key == normalized_no_country):
            return {"display_name": name, "latitude": lat, "longitude": lon}

    # Substring match
    for key, (lat, lon, name) in KNOWN_PLACES.items():
        if key in normalized or (normalized_no_country and key in normalized_no_country):
            return {"display_name": name, "latitude": lat, "longitude": lon}

    # 1. Try Open-Meteo Geocoding
    try:
        search_term = query.split(",")[0].strip()
        open_meteo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={quote_plus(search_term)}&count=1"
        req = UrllibRequest(open_meteo_url, headers={"User-Agent": "DisasterResponseCopilot/1.0"})
        with urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode("utf-8"))
            results = data.get("results")
            if results and len(results) > 0:
                first = results[0]
                country = first.get("country", "")
                admin1 = first.get("admin1", "")
                full_name = f"{first['name']}, {admin1}, {country}".replace(", ,", ",").strip(", ")
                return {
                    "display_name": full_name or first["name"],
                    "latitude": float(first["latitude"]),
                    "longitude": float(first["longitude"]),
                }
    except Exception:
        pass

    # 2. Try Photon (Komoot OSM)
    try:
        photon_url = f"https://photon.komoot.io/api/?q={quote_plus(query)}&limit=1"
        req = UrllibRequest(photon_url, headers={"User-Agent": "DisasterResponseCopilot/1.0"})
        with urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode("utf-8"))
            features = data.get("features", [])
            if features:
                props = features[0].get("properties", {})
                coords = features[0]["geometry"]["coordinates"]
                name = props.get("name") or query
                country = props.get("country") or ""
                return {
                    "display_name": f"{name}, {country}".strip(", "),
                    "latitude": float(coords[1]),
                    "longitude": float(coords[0]),
                }
    except Exception:
        pass

    # 3. Try Nominatim
    try:
        nominatim_url = f"https://nominatim.openstreetmap.org/search?format=jsonv2&limit=1&q={quote_plus(query)}"
        req = UrllibRequest(
            nominatim_url,
            headers={
                "User-Agent": "DisasterResponseCopilot/1.0 (contact: support@disastermapper.org)",
                "Accept": "application/json",
            },
        )
        with urlopen(req, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
            if payload:
                first = payload[0]
                return {
                    "display_name": first.get("display_name", query),
                    "latitude": float(first["lat"]),
                    "longitude": float(first["lon"]),
                }
    except Exception:
        pass

    raise HTTPException(status_code=404, detail=f"No place found for '{query}'")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/model-metadata")
async def model_metadata():
    checkpoint_path = ROOT / "checkpoints" / "best_unet.pt"
    if not checkpoint_path.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"Model checkpoint not found: {checkpoint_path}",
        )

    import torch

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        channel_names = checkpoint.get("input_channel_names")
        if isinstance(channel_names, str):
            channel_names = [name.strip() for name in channel_names.split(",") if name.strip()]
        if not isinstance(channel_names, list):
            raise ValueError("Checkpoint is missing input_channel_names metadata")
        return {
            "raw_flood_class_id": int(checkpoint["flood_class"]),
            "channel_names": [str(name) for name in channel_names],
            "input_scale": str(checkpoint["input_scale"]).lower(),
        }
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        raise HTTPException(
            status_code=422,
            detail=f"Checkpoint input metadata is invalid: {error}",
        ) from error


@app.post("/run", status_code=202)
async def run_pipeline(payload: RunRequest, background_tasks: BackgroundTasks):
    if payload.west >= payload.east or payload.south >= payload.north:
        raise HTTPException(status_code=422, detail="Bounding box must satisfy west < east and south < north")
    if payload.input_scale not in ("db", "linear", "unknown"):
        raise HTTPException(status_code=422, detail="input_scale must be db, linear, or unknown")

    checkpoint_path = Path(payload.checkpoint).expanduser()
    if not checkpoint_path.is_absolute():
        checkpoint_path = ROOT / checkpoint_path
    checkpoint_path = checkpoint_path.resolve()
    if not checkpoint_path.is_file():
        raise HTTPException(
            status_code=422,
            detail=(
                f"Model checkpoint not found: {checkpoint_path}. "
                "Train the model to create checkpoints/best_unet.pt, or enter the absolute path "
                "to an existing .pt checkpoint."
            ),
        )
    payload.checkpoint = str(checkpoint_path)

    run_id = uuid.uuid4().hex
    RUNS[run_id] = {
        "run_id": run_id,
        "status": RunStatus.queued.value,
        "progress": 0,
        "stage": "Queued",
        "error": None,
    }
    background_tasks.add_task(_run_job, run_id, payload)
    return {"run_id": run_id, "status": RunStatus.queued.value}


@app.get("/status/{run_id}")
async def status(run_id: str):
    state = RUNS.get(run_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Unknown run")
    return {key: state[key] for key in ("run_id", "status", "progress", "stage", "error")}


@app.get("/results/{run_id}")
async def results(run_id: str, request: Request):
    state = RUNS.get(run_id)
    if state is None or state.get("status") != RunStatus.completed.value:
        raise HTTPException(status_code=409, detail="Run is not complete")
    base_url = str(request.base_url).rstrip("/")
    layer_urls = {
        key: f"{base_url}/files/{run_id}/{filename}"
        for key, filename in LAYER_FILES.items()
    }
    layer_urls["dem_tiles"] = f"{base_url}/dem/{run_id}/{{z}}/{{x}}/{{y}}.png"
    layer_urls["report"] = f"{base_url}/report/{run_id}?lang=en"
    return {"run_id": run_id, "summary": state["summary"], "urls": layer_urls}


@app.get("/files/{run_id}/{filename}")
async def serve_layer(run_id: str, filename: str):
    if filename not in LAYER_FILES.values():
        raise HTTPException(status_code=404, detail="Unknown result layer")
    path = _run_dir(run_id) / "analysis" / "web" / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Result layer not available")
    return FileResponse(path, media_type="application/geo+json")


def _elevation_band(dataset) -> int:
    for index, description in enumerate(dataset.descriptions, start=1):
        if description and "elevation" in description.lower():
            return index
    raise RuntimeError("Copernicus DEM elevation band was not found in the SNAP terrain-corrected raster")


def _tile_bounds_mercator(z: int, x: int, y: int):
    count = 1 << z
    west = x / count * TILE_SPAN - TILE_ORIGIN
    east = (x + 1) / count * TILE_SPAN - TILE_ORIGIN
    north = TILE_ORIGIN - y / count * TILE_SPAN
    south = TILE_ORIGIN - (y + 1) / count * TILE_SPAN
    return west, south, east, north


def _make_dem_tile(dem_path: Path, tile_path: Path, z: int, x: int, y: int) -> None:
    if tile_path.is_file():
        return
    tile_path.parent.mkdir(parents=True, exist_ok=True)
    destination = np.full((TILE_SIZE, TILE_SIZE), np.nan, dtype=np.float32)
    west, south, east, north = _tile_bounds_mercator(z, x, y)
    with rasterio.open(dem_path) as source:
        band = _elevation_band(source)
        reproject(
            source=rasterio.band(source, band),
            destination=destination,
            src_transform=source.transform,
            src_crs=source.crs,
            src_nodata=source.nodata,
            dst_transform=from_bounds(west, south, east, north, TILE_SIZE, TILE_SIZE),
            dst_crs="EPSG:3857",
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )

    valid = np.isfinite(destination)
    elevation_code = np.zeros(destination.shape, dtype=np.uint32)
    elevation_code[valid] = np.clip(
        np.rint((destination[valid] + 10000.0) * 10.0), 0, 16777215
    ).astype(np.uint32)
    rgb = np.zeros((TILE_SIZE, TILE_SIZE, 4), dtype=np.uint8)
    rgb[:, :, 0] = ((elevation_code >> 16) & 255).astype(np.uint8)
    rgb[:, :, 1] = ((elevation_code >> 8) & 255).astype(np.uint8)
    rgb[:, :, 2] = (elevation_code & 255).astype(np.uint8)
    rgb[:, :, 3] = np.where(valid, 255, 0).astype(np.uint8)
    Image.fromarray(rgb, mode="RGBA").save(tile_path, format="PNG")


@app.get("/dem/{run_id}/{z}/{x}/{y}.png")
async def dem_tile(run_id: str, z: int, x: int, y: int):
    if z < 0 or z > 20:
        raise HTTPException(status_code=404, detail="Tile zoom is outside the supported range")
    tile_count = 1 << z
    if x < 0 or y < 0 or x >= tile_count or y >= tile_count:
        raise HTTPException(status_code=404, detail="Tile coordinate is outside the world grid")
    dem_path = _run_dir(run_id) / "work" / "post_terrain_corrected.tif"
    if not dem_path.is_file():
        raise HTTPException(status_code=404, detail="Copernicus DEM is not available for this run")
    tile_path = _run_dir(run_id) / "analysis" / "web" / "dem_tiles" / str(z) / str(x) / f"{y}.png"
    try:
        await asyncio.to_thread(_make_dem_tile, dem_path, tile_path, z, x, y)
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
    return FileResponse(tile_path, media_type="image/png", headers={"Cache-Control": "public, max-age=86400"})


@app.get("/report/{run_id}")
async def report(run_id: str, lang: str = "en"):
    if lang not in ("en", "ne"):
        raise HTTPException(status_code=422, detail="lang must be en or ne")
    path = _run_dir(run_id) / "situation_report.pdf"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Situation report is not available")
    return FileResponse(
        path,
        media_type="application/pdf",
        filename=f"situation_report_{lang}.pdf",
        headers={"X-Report-Content-Language": "en,ne"},
    )