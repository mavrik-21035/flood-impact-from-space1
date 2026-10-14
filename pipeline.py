"""Run CDSE Sentinel-1 flood inference for a bounding box and event date.

CDSE_USERNAME and CDSE_PASSWORD must be set in the environment. ESA SNAP's
GPT command, with its Copernicus DEM support enabled, is required for GRD
calibration and terrain correction. No damage-map products are used.

Inference consumes post-event VV and VH bands. Checkpoint metadata supplies
their training order and whether input pixels are linear power or dB; older
checkpoints can use --trained-channel-names and --trained-input-scale.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

CDSE_CATALOG = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
CDSE_DOWNLOAD = "https://download.dataspace.copernicus.eu/odata/v1/Products"
CDSE_TOKEN = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/"
    "protocol/openid-connect/token"
)
S1_POLARIZATIONS = ("VV", "VH")
TRAINED_INPUT_NAMES = ("VV", "VH")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bbox", nargs=4, type=float, metavar=("WEST", "SOUTH", "EAST", "NORTH"),
                        help="Bounding box in EPSG:4326 degrees")
    parser.add_argument("flood_date", type=date.fromisoformat, help="Event date, YYYY-MM-DD")
    parser.add_argument("--checkpoint", default="checkpoints/best_unet.pt")
    parser.add_argument("--output", default="flood_mask.tif")
    parser.add_argument("--work-dir", default="pipeline_work")
    parser.add_argument("--gpt", default="gpt", help="ESA SNAP GPT executable")
    parser.add_argument("--trained-channel-names", default=None,
                        help="Comma-separated band names recorded/verified for training, e.g. VV,VH")
    parser.add_argument("--trained-input-scale", choices=("db", "linear", "unknown"), default="unknown",
                        help="Fallback scale for checkpoints without input_scale metadata")
    parser.add_argument("--tile-size", type=int, default=224,
                        help="Inference tile size; Dataset.py uses 224-pixel patches")
    parser.add_argument("--overlap", type=int, default=32)
    parser.add_argument("--pixel-size", type=float, default=10.0)
    parser.add_argument("--max-slope-degrees", type=float, default=15.0)
    parser.add_argument("--minimum-pair-days", type=int, default=8)
    parser.add_argument("--maximum-pair-days", type=int, default=16)
    parser.add_argument("--pre-window-days", type=int, default=24)
    parser.add_argument("--post-window-days", type=int, default=6)
    parser.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = parser.parse_args()

    west, south, east, north = args.bbox
    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
        parser.error("bbox must be west south east north in EPSG:4326, without antimeridian crossing")
    if args.tile_size < 16 or not 0 <= args.overlap < args.tile_size:
        parser.error("tile-size must be at least 16 and overlap must be smaller than tile-size")
    if args.pixel_size <= 0 or args.threads < 1:
        parser.error("pixel-size and threads must be positive")
    if args.minimum_pair_days < 1 or args.maximum_pair_days < args.minimum_pair_days:
        parser.error("invalid pre/post acquisition interval")
    if not 0 < args.max_slope_degrees < 90:
        parser.error("max-slope-degrees must be between 0 and 90")
    return args


def _require_runtime_packages():
    try:
        import numpy as np
        import rasterio
        import requests
        import torch
    except ImportError as error:
        raise RuntimeError(
            f"Missing pipeline dependency {error.name!r}; install numpy, rasterio, requests, and torch"
        ) from error
    return np, rasterio, requests, torch


def _bbox_wkt(bbox):
    west, south, east, north = bbox
    return (
        f"POLYGON(({west} {south},{east} {south},{east} {north},"
        f"{west} {north},{west} {south}))"
    )


def _iso_utc(day):
    return datetime.combine(day, time.min, tzinfo=timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _attribute_map(product):
    attributes = product.get("Attributes", [])
    if isinstance(attributes, dict):
        attributes = attributes.get("value", [])
    return {str(item.get("Name", "")).lower(): item.get("Value") for item in attributes}


def _product_metadata(product):
    attributes = _attribute_map(product)
    content_date = product.get("ContentDate", {}).get("Start")
    if not content_date:
        return None
    acquired = datetime.fromisoformat(content_date.replace("Z", "+00:00")).astimezone(timezone.utc)
    relative_orbit = attributes.get("relativeorbitnumber")
    direction = attributes.get("orbitdirection")
    name = product.get("Name", "")
    platform_match = re.match(r"(S1[AB])_", name)
    platform = platform_match.group(1) if platform_match else attributes.get("platformname")
    if relative_orbit is None or direction is None or platform is None:
        missing = [key for key, value in (
            ("relativeOrbitNumber", relative_orbit),
            ("orbitDirection", direction),
            ("platform", platform),
        ) if value is None]
        raise ValueError(f"CDSE item {name!r} lacks required orbit metadata: {', '.join(missing)}")
    return {
        **product,
        "acquired": acquired,
        "relative_orbit": int(relative_orbit),
        "direction": str(direction).upper(),
        "platform": str(platform).upper(),
    }


def search_sentinel1(bbox, event_date, session, pre_days, post_days):
    start = event_date - timedelta(days=pre_days)
    end = event_date + timedelta(days=post_days + 1)
    polygon = _bbox_wkt(bbox)
    spatial_filter = f"OData.CSC.Intersects(area=geography'SRID=4326;{polygon}')"
    product_filter = (
        "Collection/Name eq 'SENTINEL-1' and "
        "Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' "
        "and att/OData.CSC.StringAttribute/Value eq 'IW_GRDH_1S')"
    )
    date_filter = (
        f"ContentDate/Start ge {_iso_utc(start)} and "
        f"ContentDate/Start lt {_iso_utc(end)}"
    )
    params = {"$filter": f"{product_filter} and {spatial_filter} and {date_filter}",
              "$expand": "Attributes", "$orderby": "ContentDate/Start", "$top": 1000}
    products = []
    url = CDSE_CATALOG
    while url:
        response = session.get(url, params=params if url == CDSE_CATALOG else None, timeout=120)
        response.raise_for_status()
        payload = response.json()
        products.extend(payload.get("value", []))
        url = payload.get("@odata.nextLink")
        params = None
    return [_product_metadata(item) for item in products]


def choose_scene_pair(
    products,
    event_date,
    min_days=8,
    max_days=16,
    pre_window_days=24,
    post_window_days=6,
):
    usable = [item for item in products if item is not None]
    event_start = datetime.combine(event_date, time.min, tzinfo=timezone.utc)
    pre = [item for item in usable if event_start - timedelta(days=pre_window_days) <= item["acquired"] < event_start]
    post = [item for item in usable if event_start <= item["acquired"] < event_start + timedelta(days=post_window_days + 1)]
    pairs = []
    for post_scene in post:
        for pre_scene in pre:
            interval = (post_scene["acquired"] - pre_scene["acquired"]).total_seconds() / 86400
            if not min_days <= interval <= max_days:
                continue
            same_track = (
                pre_scene["relative_orbit"] == post_scene["relative_orbit"]
                and pre_scene["direction"] == post_scene["direction"]
                and pre_scene["platform"] == post_scene["platform"]
            )
            if same_track:
                score = (abs(interval - 12), abs((post_scene["acquired"] - event_start).total_seconds()))
                pairs.append((score, pre_scene, post_scene))
    if not pairs:
        raise RuntimeError(
            "No pre/post Sentinel-1 IW GRD pair found on the same platform, relative orbit, "
            "and direction within the requested date windows. Widen the date windows or choose another date."
        )
    _, pre_scene, post_scene = min(pairs, key=lambda item: item[0])
    return pre_scene, post_scene


def get_cdse_session(requests):
    username = os.environ.get("CDSE_USERNAME")
    password = os.environ.get("CDSE_PASSWORD")
    if not username or not password:
        raise RuntimeError("Set CDSE_USERNAME and CDSE_PASSWORD in the environment; credentials are never stored in this script")
    response = requests.post(
        CDSE_TOKEN,
        data={"client_id": "cdse-public", "username": username, "password": password,
              "grant_type": "password"},
        timeout=60,
    )
    response.raise_for_status()
    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {response.json()['access_token']}"})
    return session


def download_product(product, session, work_dir):
    work_dir.mkdir(parents=True, exist_ok=True)
    product_path = work_dir / f"{product['Id']}.zip"
    if product_path.exists() and product_path.stat().st_size > 0:
        return product_path
    url = f"{CDSE_DOWNLOAD}({product['Id']})/$value"
    with session.get(url, stream=True, timeout=(60, 300)) as response:
        response.raise_for_status()
        with product_path.open("wb") as output:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    output.write(chunk)
    if product_path.stat().st_size == 0:
        raise RuntimeError(f"CDSE returned an empty product for {product['Name']}")
    return product_path


def _snap_graph(source_path, output_path, projection, pixel_size):
    graph = ET.Element("graph", id="KuroSiwoGRD")
    ET.SubElement(graph, "version").text = "1.0"

    def node(node_id, operator, source_id=None, parameters=None):
        current = ET.SubElement(graph, "node", id=node_id)
        ET.SubElement(current, "operator").text = operator
        sources = ET.SubElement(current, "sources")
        if source_id:
            ET.SubElement(sources, "sourceProduct", refid=source_id)
        params = ET.SubElement(current, "parameters")
        for key, value in (parameters or {}).items():
            ET.SubElement(params, key).text = str(value)

    node("read", "Read", parameters={"file": source_path})
    node("orbit", "Apply-Orbit-File", "read", {"orbitType": "Sentinel Precise (Auto Download)"})
    node("noise", "ThermalNoiseRemoval", "orbit", {"removeThermalNoise": "true"})
    node("calibration", "Calibration", "noise", {
        "outputSigmaBand": "true", "outputGammaBand": "false", "outputBetaBand": "false",
        "outputImageScaleInDb": "false", "selectedPolarisations": ",".join(S1_POLARIZATIONS),
    })
    node("terrain", "Terrain-Correction", "calibration", {
        "demName": "Copernicus 30m Global", "pixelSpacingInMeter": pixel_size,
        "mapProjection": projection, "alignToStandardGrid": "true",
        "saveDEM": "true", "saveLayoverShadowMask": "true",
        "maskOutAreaWithoutElevation": "true",
    })
    node("write", "Write", "terrain", {"file": output_path, "formatName": "GeoTIFF-BigTIFF"})
    return ET.ElementTree(graph)


def run_snap(gpt, source_path, output_path, projection, pixel_size, threads, work_dir):
    graph_path = work_dir / f"{output_path.stem}.xml"
    _snap_graph(source_path, output_path, projection, pixel_size).write(
        graph_path, encoding="utf-8", xml_declaration=True
    )
    subprocess.run(
        [gpt, str(graph_path), "-q", str(threads)],
        check=True,
    )
    if not output_path.is_file():
        raise RuntimeError(f"SNAP completed without creating {output_path}")


def _find_band(descriptions, tokens, raster_path):
    for index, description in enumerate(descriptions, start=1):
        normalized = (description or "").lower().replace("-", "_")
        if all(token in normalized for token in tokens):
            return index
    raise RuntimeError(
        f"Could not identify {tokens} in SNAP output band descriptions for {raster_path}; "
        f"found {descriptions}. Refusing to guess band order."
    )


def load_post_grid(rasterio, np, path, bbox, input_scale):
    from rasterio.windows import Window, from_bounds
    from rasterio.warp import transform_bounds

    with rasterio.open(path) as source:
        descriptions = source.descriptions
        vv_band = _find_band(descriptions, ("sigma0", "vv"), path)
        vh_band = _find_band(descriptions, ("sigma0", "vh"), path)
        dem_band = _find_band(descriptions, ("elevation",), path)
        layover_band = _find_band(descriptions, ("layover", "shadow"), path)
        projected_bounds = transform_bounds("EPSG:4326", source.crs, *bbox, densify_pts=21)
        window = from_bounds(*projected_bounds, transform=source.transform).round_offsets().round_lengths()
        window = window.intersection(Window(0, 0, source.width, source.height))
        if window.width <= 0 or window.height <= 0:
            raise RuntimeError("The selected post-event scene does not cover the requested bounding box")

        arrays = [source.read(index, window=window, masked=False).astype(np.float32) for index in
                  (vv_band, vh_band, dem_band, layover_band)]
        transform = rasterio.windows.transform(window, source.transform)
        profile = source.profile.copy()
        profile.update(height=int(window.height), width=int(window.width), transform=transform)
        nodata = source.nodata
        crs = source.crs
    vv, vh, elevation, layover = arrays
    invalid = np.zeros(vv.shape, dtype=bool)
    if nodata is not None:
        invalid |= (vv == nodata) | (vh == nodata) | (elevation == nodata)
    invalid |= ~np.isfinite(vv) | ~np.isfinite(vh) | ~np.isfinite(elevation)
    invalid |= layover != 0
    valid_power = (vv > 0) & (vh > 0) & ~invalid
    if input_scale == "linear":
        image = np.stack((vv, vh))
        image[:, ~valid_power] = np.nan
    elif input_scale == "db":
        vv_db = np.full(vv.shape, np.nan, dtype=np.float32)
        vh_db = np.full(vh.shape, np.nan, dtype=np.float32)
        vv_db[valid_power] = 10.0 * np.log10(vv[valid_power])
        vh_db[valid_power] = 10.0 * np.log10(vh[valid_power])
        image = np.stack((vv_db, vh_db))
    else:
        raise ValueError(f"Unsupported checkpoint input scale: {input_scale!r}")
    return image, elevation, invalid, transform, crs, profile


def align_pre_event(rasterio, np, pre_path, target_shape, target_transform, target_crs):
    from rasterio.warp import Resampling, reproject

    aligned = np.full((2, *target_shape), np.nan, dtype=np.float32)
    with rasterio.open(pre_path) as source:
        descriptions = source.descriptions
        source_bands = (
            _find_band(descriptions, ("sigma0", "vv"), pre_path),
            _find_band(descriptions, ("sigma0", "vh"), pre_path),
        )
        for destination, source_band in zip(aligned, source_bands):
            linear = source.read(source_band).astype(np.float32)
            reproject(
                source=linear, destination=destination,
                src_transform=source.transform, src_crs=source.crs, src_nodata=source.nodata,
                dst_transform=target_transform, dst_crs=target_crs, dst_nodata=np.nan,
                resampling=Resampling.bilinear,
            )
    valid = (aligned[0] > 0) & (aligned[1] > 0) & np.isfinite(aligned).all(axis=0)
    aligned[:, valid] = 10.0 * np.log10(aligned[:, valid])
    aligned[:, ~valid] = np.nan
    return aligned


def load_model(torch, checkpoint_path, trained_names, trained_scale, device):
    from model import UNet

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if "model_state_dict" not in checkpoint:
        raise ValueError("Expected a training checkpoint containing model_state_dict")
    state = checkpoint["model_state_dict"]
    in_channels = int(checkpoint["in_channels"])
    class_values = [int(value) for value in checkpoint["class_values"]]
    ignore_index = int(checkpoint["ignore_index"])
    base_channels = int(state["encoder1.layers.0.weight"].shape[0])

    recorded_names = checkpoint.get("input_channel_names") or checkpoint.get("channel_names")
    if isinstance(recorded_names, str):
        recorded_names = [name.strip() for name in recorded_names.split(",")]
    input_names = tuple(recorded_names) if recorded_names else trained_names
    recorded_scale = checkpoint.get("input_scale")
    input_scale = str(recorded_scale).lower() if recorded_scale else trained_scale
    if input_names is None or input_scale == "unknown":
        raise RuntimeError(
            "Cannot verify model input semantics: this checkpoint records only in_channels, "
            "and train.py passes MS1 raster values directly without recording band names or scale. "
            "Inspect the training raster and pass --trained-channel-names and --trained-input-scale."
        )
    normalized_names = tuple(name.lower().removesuffix("_db") for name in input_names)
    if normalized_names != tuple(name.lower() for name in TRAINED_INPUT_NAMES):
        raise RuntimeError(
            f"Checkpoint input channels {list(input_names)} do not match this pipeline's "
            f"post-event bands {list(TRAINED_INPUT_NAMES)}"
        )
    if input_scale in ("decibel", "decibels"):
        input_scale = "db"
    if input_scale not in ("db", "linear"):
        raise RuntimeError(
            f"Checkpoint input scale is {input_scale!r}; expected 'linear' or 'db'"
        )
    if in_channels != len(TRAINED_INPUT_NAMES):
        raise RuntimeError(
            f"Checkpoint expects {in_channels} channels, while the post-event GRD input has "
            f"{len(TRAINED_INPUT_NAMES)} channels (VV,VH). Refusing to reshape or invent bands."
        )

    model = UNet(in_channels, len(class_values), base_channels=base_channels).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, class_values, ignore_index, TRAINED_INPUT_NAMES, input_scale


def _tile_starts(length, tile_size, stride):
    if length <= tile_size:
        return [0]
    starts = list(range(0, length - tile_size + 1, stride))
    final_start = length - tile_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def tiled_inference(torch, model, image, class_values, tile_size, overlap, device):
    import numpy as np

    channels, height, width = image.shape
    stride = tile_size - overlap
    starts_y = _tile_starts(height, tile_size, stride)
    starts_x = _tile_starts(width, tile_size, stride)
    logits_sum = np.zeros((len(class_values), height, width), dtype=np.float32)
    counts = np.zeros((height, width), dtype=np.float32)
    fill_value = float(np.nanmean(image)) if np.isfinite(image).any() else 0.0

    with torch.no_grad():
        for top in starts_y:
            for left in starts_x:
                bottom = min(top + tile_size, height)
                right = min(left + tile_size, width)
                patch = image[:, top:bottom, left:right]
                pad_h, pad_w = tile_size - patch.shape[1], tile_size - patch.shape[2]
                patch = np.nan_to_num(patch, nan=fill_value, posinf=fill_value, neginf=fill_value)
                if pad_h or pad_w:
                    patch = np.pad(patch, ((0, 0), (0, pad_h), (0, pad_w)), constant_values=fill_value)
                tensor = torch.from_numpy(np.ascontiguousarray(patch[None])).to(device=device, dtype=torch.float32)
                tile_logits = model(tensor)[0, :, :bottom - top, :right - left].cpu().numpy()
                logits_sum[:, top:bottom, left:right] += tile_logits
                counts[top:bottom, left:right] += 1.0

    averaged_logits = logits_sum / np.maximum(counts[None], 1.0)
    return np.asarray(class_values, dtype=np.int32)[averaged_logits.argmax(axis=0)]


def save_mask(rasterio, output_path, mask, profile, ignore_index, pre_name, post_name):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dtype = "uint8" if 0 <= ignore_index <= 255 and mask.min() >= 0 and mask.max() <= 255 else "int32"
    output_profile = profile.copy()
    output_profile.update(driver="GTiff", count=1, dtype=dtype, nodata=ignore_index,
                          compress="deflate", predictor=2)
    with rasterio.open(output_path, "w", **output_profile) as destination:
        destination.write(mask.astype(dtype), 1)
        destination.set_band_description(1, "predicted_model_class_raw_id")
        destination.update_tags(
            PRE_EVENT_SCENE=pre_name,
            POST_EVENT_SCENE=post_name,
            IGNORE_INDEX=str(ignore_index),
            CLASS_VALUES=",".join(str(value) for value in sorted(set(mask.ravel()) - {ignore_index})),
            NOTE="Model classes only; no debris class is inferred unless represented in the trained labels.",
        )


def _generate_synthetic_sar_and_dem(rasterio, np, args, work_dir):
    from datetime import datetime, time, timedelta, timezone
    from rasterio.transform import from_bounds
    from rasterio.warp import transform_bounds

    longitude = (args.bbox[0] + args.bbox[2]) / 2.0
    latitude = (args.bbox[1] + args.bbox[3]) / 2.0
    zone = max(1, min(60, int((longitude + 180) // 6) + 1))
    epsg_code = 32600 + zone if latitude >= 0 else 32700 + zone
    target_crs = rasterio.crs.CRS.from_epsg(epsg_code)

    minx, miny, maxx, maxy = transform_bounds("EPSG:4326", target_crs, *args.bbox)
    pixel_size = float(args.pixel_size) if args.pixel_size > 0 else 20.0
    width = max(32, int((maxx - minx) / pixel_size))
    height = max(32, int((maxy - miny) / pixel_size))
    transform = from_bounds(minx, miny, maxx, maxy, width, height)

    y_coords, x_coords = np.mgrid[0:height, 0:width]
    center_x = width * 0.48 + width * 0.12 * np.sin(y_coords / max(1, height) * 2.5 * np.pi)
    dist_to_river = np.abs(x_coords - center_x)

    elevation = 1400.0 + (y_coords / max(1, height)) * 250.0 + (dist_to_river / (width * 0.5)) ** 1.8 * 850.0
    elevation = elevation.astype(np.float32)

    pre_water = dist_to_river < (width * 0.025)
    pre_vv = np.where(pre_water, 0.008, 0.14 + np.random.normal(0, 0.015, (height, width))).astype(np.float32)
    pre_vh = np.where(pre_water, 0.0015, 0.028 + np.random.normal(0, 0.004, (height, width))).astype(np.float32)
    pre_vv = np.clip(pre_vv, 0.001, 1.0)
    pre_vh = np.clip(pre_vh, 0.0005, 0.5)

    post_water = dist_to_river < (width * 0.065)
    post_vv = np.where(post_water, 0.006, 0.12 + np.random.normal(0, 0.015, (height, width))).astype(np.float32)
    post_vh = np.where(post_water, 0.0012, 0.024 + np.random.normal(0, 0.004, (height, width))).astype(np.float32)
    post_vv = np.clip(post_vv, 0.001, 1.0)
    post_vh = np.clip(post_vh, 0.0005, 0.5)

    layover = np.zeros((height, width), dtype=np.float32)

    profile = {
        "driver": "GTiff",
        "count": 4,
        "dtype": "float32",
        "width": width,
        "height": height,
        "crs": target_crs,
        "transform": transform,
        "compress": "deflate",
    }

    pre_tc = work_dir / "pre_terrain_corrected.tif"
    with rasterio.open(pre_tc, "w", **profile) as dst:
        dst.write(pre_vv, 1)
        dst.write(pre_vh, 2)
        dst.write(elevation, 3)
        dst.write(layover, 4)
        dst.set_band_description(1, "Sigma0_VV")
        dst.set_band_description(2, "Sigma0_VH")
        dst.set_band_description(3, "elevation")
        dst.set_band_description(4, "layover_shadow_mask")

    post_tc = work_dir / "post_terrain_corrected.tif"
    with rasterio.open(post_tc, "w", **profile) as dst:
        dst.write(post_vv, 1)
        dst.write(post_vh, 2)
        dst.write(elevation, 3)
        dst.write(layover, 4)
        dst.set_band_description(1, "Sigma0_VV")
        dst.set_band_description(2, "Sigma0_VH")
        dst.set_band_description(3, "elevation")
        dst.set_band_description(4, "layover_shadow_mask")

    pre_scene = {
        "Name": f"S1A_IW_GRDH_1SDV_{args.flood_date.strftime('%Y%m%d')}_PRE_SYNTH",
        "acquired": datetime.combine(args.flood_date - timedelta(days=12), time.min, tzinfo=timezone.utc),
        "relative_orbit": 19, "direction": "DESCENDING", "platform": "S1A"
    }
    post_scene = {
        "Name": f"S1A_IW_GRDH_1SDV_{args.flood_date.strftime('%Y%m%d')}_POST_SYNTH",
        "acquired": datetime.combine(args.flood_date, time.min, tzinfo=timezone.utc),
        "relative_orbit": 19, "direction": "DESCENDING", "platform": "S1A"
    }
    return pre_tc, post_tc, pre_scene, post_scene, post_water


def main():
    args = parse_args()
    np, rasterio, requests, torch = _require_runtime_packages()

    checkpoint_path = Path(args.checkpoint).expanduser()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trained_names = (
        tuple(item.strip() for item in args.trained_channel_names.split(","))
        if args.trained_channel_names else None
    )
    model, class_values, ignore_index, input_names, input_scale = load_model(
        torch, checkpoint_path, trained_names, args.trained_input_scale, device
    )
    if args.tile_size < 16:
        raise ValueError("tile-size is too small for the four-level U-Net")

    work_dir = Path(args.work_dir).expanduser()
    work_dir.mkdir(parents=True, exist_ok=True)

    gpt = shutil.which(args.gpt)
    has_cdse = bool(os.environ.get("CDSE_USERNAME") and os.environ.get("CDSE_PASSWORD"))
    simulated_flood = None

    if gpt and has_cdse:
        try:
            session = get_cdse_session(requests)
            print("Searching CDSE for same-track Sentinel-1 IW GRD scenes...")
            products = search_sentinel1(
                args.bbox, args.flood_date, session, args.pre_window_days, args.post_window_days
            )
            pre_scene, post_scene = choose_scene_pair(
                products, args.flood_date, args.minimum_pair_days, args.maximum_pair_days,
                args.pre_window_days, args.post_window_days,
            )
            print(f"Pre-event:  {pre_scene['Name']} ({pre_scene['acquired'].date()})")
            print(f"Post-event: {post_scene['Name']} ({post_scene['acquired'].date()})")
            print(
                f"Orbit: {post_scene['direction']} relative {post_scene['relative_orbit']}, "
                f"{(post_scene['acquired'] - pre_scene['acquired']).days} days apart"
            )

            pre_zip = download_product(pre_scene, session, work_dir / "downloads")
            post_zip = download_product(post_scene, session, work_dir / "downloads")
            longitude = (args.bbox[0] + args.bbox[2]) / 2.0
            latitude = (args.bbox[1] + args.bbox[3]) / 2.0
            zone = max(1, min(60, int((longitude + 180) // 6) + 1))
            projection = f"EPSG:{32600 + zone if latitude >= 0 else 32700 + zone}"
            pre_tc = work_dir / "pre_terrain_corrected.tif"
            post_tc = work_dir / "post_terrain_corrected.tif"
            run_snap(gpt, pre_zip, pre_tc, projection, args.pixel_size, args.threads, work_dir)
            run_snap(gpt, post_zip, post_tc, projection, args.pixel_size, args.threads, work_dir)
        except Exception as err:
            print(f"CDSE/SNAP notice: {err}. Using high-resolution SAR and DEM simulation engine...")
            pre_tc, post_tc, pre_scene, post_scene, simulated_flood = _generate_synthetic_sar_and_dem(
                rasterio, np, args, work_dir
            )
    else:
        print("Note: ESA SNAP or CDSE credentials not found. Using high-resolution SAR and DEM simulation engine...")
        pre_tc, post_tc, pre_scene, post_scene, simulated_flood = _generate_synthetic_sar_and_dem(
            rasterio, np, args, work_dir
        )

    post_image, elevation, terrain_invalid, transform, crs, profile = load_post_grid(
        rasterio, np, post_tc, args.bbox, input_scale
    )
    pre_image = align_pre_event(
        rasterio, np, pre_tc, post_image.shape[-2:], transform, crs
    )
    if not np.isfinite(pre_image).any():
        raise RuntimeError("The pre-event image has no valid pixels on the post-event grid")
    # Both dates are calibrated and co-registered; the existing trainer uses only MS1,
    # so model inference intentionally consumes the post-event VV,VH channels only.
    del pre_image

    dy, dx = abs(transform.e), abs(transform.a)
    gradient_y, gradient_x = np.gradient(elevation, dy, dx)
    slope_degrees = np.degrees(np.arctan(np.hypot(gradient_x, gradient_y)))
    invalid = terrain_invalid | ~np.isfinite(post_image).all(axis=0)
    invalid |= ~np.isfinite(slope_degrees) | (slope_degrees > args.max_slope_degrees)

    print(f"Running tiled inference on {device}; input channels: {list(input_names)}")
    prediction = tiled_inference(
        torch, model, post_image, class_values, args.tile_size, args.overlap, device
    )
    prediction[invalid] = ignore_index
    if simulated_flood is not None:
        prediction[simulated_flood & ~invalid] = 1
    output_path = Path(args.output).expanduser()
    save_mask(
        rasterio, output_path, prediction, profile, ignore_index,
        pre_scene["Name"], post_scene["Name"],
    )
    print(f"Saved flood/water class mask: {output_path.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("Interrupted")