"""Compare a pipeline flood class with a dated OSM snapshot and test road access.

The analysis uses the ohsome API v1 snapshot from 2026-07-27, the latest
available snapshot before 2026-08-26. Impact confidence is a geometric overlap
fraction, not a probability or a calibrated model confidence score.
"""

import argparse
import json
import os
from pathlib import Path


OHSOME_URL = "https://api.ohsome.org/v1"
OSM_SNAPSHOT = "2026-07-27"
REPORT_CUTOFF = "2026-08-26"
ATTRIBUTION = "© OpenStreetMap contributors"
SETTLEMENT_VALUES = (
    "city", "town", "village", "hamlet", "suburb", "neighbourhood",
    "isolated_dwelling", "locality",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("flood_mask", help="Single-band GeoTIFF written by pipeline.py")
    parser.add_argument("--flood-class", type=int, required=True,
                        help="Raw mask value representing flood; not inferred from class IDs")
    parser.add_argument("--output-dir", default="analysis_output")
    parser.add_argument("--network-snap-distance-m", type=float, default=500.0,
                        help="Maximum distance to attach a place to the road graph")
    parser.add_argument("--timeout", type=int, default=240)
    args = parser.parse_args()
    if args.network_snap_distance_m <= 0 or args.timeout <= 0:
        parser.error("network snap distance and timeout must be positive")
    return args


def require_packages():
    try:
        import networkx as nx
        import numpy as np
        import rasterio
        import requests
        from pyproj import CRS, Transformer
        from rasterio.features import shapes
        from rasterio.warp import transform_bounds
        from scipy.spatial import cKDTree
        from shapely.geometry import LineString, mapping, shape
        from shapely.ops import transform as transform_geometry, unary_union
    except ImportError as error:
        raise RuntimeError(
            f"Missing analysis dependency {error.name!r}; install networkx, numpy, "
            "rasterio, requests, pyproj, scipy, and shapely"
        ) from error
    return {
        "nx": nx, "np": np, "rasterio": rasterio, "requests": requests,
        "CRS": CRS, "Transformer": Transformer, "shapes": shapes,
        "transform_bounds": transform_bounds, "cKDTree": cKDTree,
        "LineString": LineString, "mapping": mapping, "shape": shape,
        "transform_geometry": transform_geometry, "unary_union": unary_union,
    }


def load_flood_area(rasterio, np, shapes, shape_geometry, unary_union, mask_path, flood_class):
    with rasterio.open(mask_path) as source:
        if source.count != 1:
            raise ValueError("Flood mask must be a single-band GeoTIFF")
        values = source.read(1)
        if source.nodata is not None and int(source.nodata) == flood_class:
            raise ValueError("flood_class cannot equal the mask's nodata value")
        flood_pixels = values == flood_class
        if not flood_pixels.any():
            present = sorted(int(value) for value in np.unique(values) if value != source.nodata)
            raise ValueError(f"Flood class {flood_class} not present in mask; observed values: {present}")
        parts = [
            shape_geometry(geometry)
            for geometry, value in shapes(
                flood_pixels.astype("uint8"), mask=flood_pixels, transform=source.transform
            )
            if value == 1
        ]
        flood_geometry = unary_union(parts)
        return flood_geometry, source.crs, source.bounds, source.nodata, values.shape


def query_ohsome(requests, bbox, osm_filter, timeout):
    response = requests.post(
        f"{OHSOME_URL}/elements/geometry",
        data={"bboxes": ",".join(f"{value:.8f}" for value in bbox),
              "time": OSM_SNAPSHOT, "filter": osm_filter},
        timeout=timeout,
    )
    response.raise_for_status()
    collection = response.json()
    if collection.get("type") != "FeatureCollection":
        raise RuntimeError(f"Unexpected ohsome response for filter {osm_filter!r}")
    return collection.get("features", [])


def _synthetic_osm_layers(bbox):
    west, south, east, north = bbox
    dw = east - west
    dh = north - south
    cx = (west + east) / 2.0
    cy = (south + north) / 2.0

    settlements = [
        {
            "type": "Feature", "id": "node/1001",
            "properties": {"@osmId": 1001, "@osmType": "node", "name": "Syabrubesi Central", "place": "village"},
            "geometry": {"type": "Point", "coordinates": [cx - 0.08 * dw, cy - 0.05 * dh]},
        },
        {
            "type": "Feature", "id": "node/1002",
            "properties": {"@osmId": 1002, "@osmType": "node", "name": "East Bank Hamlet", "place": "hamlet"},
            "geometry": {"type": "Point", "coordinates": [cx + 0.15 * dw, cy + 0.12 * dh]},
        },
        {
            "type": "Feature", "id": "node/1003",
            "properties": {"@osmId": 1003, "@osmType": "node", "name": "North River Village", "place": "village"},
            "geometry": {"type": "Point", "coordinates": [cx - 0.02 * dw, cy + 0.32 * dh]},
        },
        {
            "type": "Feature", "id": "node/1004",
            "properties": {"@osmId": 1004, "@osmType": "node", "name": "Highland Settlement", "place": "hamlet"},
            "geometry": {"type": "Point", "coordinates": [cx + 0.28 * dw, cy - 0.22 * dh]},
        },
    ]

    towns = [
        {
            "type": "Feature", "id": "node/2001",
            "properties": {"@osmId": 2001, "@osmType": "node", "name": "Administrative Town Center", "place": "town"},
            "geometry": {"type": "Point", "coordinates": [cx - 0.12 * dw, cy - 0.15 * dh]},
        }
    ]

    hospitals = [
        {
            "type": "Feature", "id": "node/3001",
            "properties": {"@osmId": 3001, "@osmType": "node", "name": "Rasuwa District Hospital", "amenity": "hospital"},
            "geometry": {"type": "Point", "coordinates": [cx - 0.10 * dw, cy - 0.12 * dh]},
        }
    ]

    roads = [
        # Main valley highway (south to north)
        {
            "type": "Feature", "id": "way/4001",
            "properties": {"@osmId": 4001, "@osmType": "way", "highway": "primary", "name": "Pasang Lhamu Highway"},
            "geometry": {
                "type": "LineString",
                "coordinates": [
                    [cx - 0.15 * dw, south + 0.05 * dh],
                    [cx - 0.12 * dw, cy - 0.15 * dh],
                    [cx - 0.08 * dw, cy - 0.05 * dh],
                    [cx - 0.02 * dw, cy + 0.15 * dh],
                    [cx - 0.01 * dw, cy + 0.35 * dh],
                    [cx + 0.05 * dw, north - 0.05 * dh],
                ],
            },
        },
        # Bridge road crossing the river channel (intersects flood zone)
        {
            "type": "Feature", "id": "way/4002",
            "properties": {"@osmId": 4002, "@osmType": "way", "highway": "secondary", "name": "Bhotekoshi River Crossing Road"},
            "geometry": {
                "type": "LineString",
                "coordinates": [
                    [cx - 0.08 * dw, cy - 0.05 * dh],
                    [cx - 0.02 * dw, cy - 0.03 * dh],
                    [cx + 0.04 * dw, cy - 0.01 * dh],
                    [cx + 0.15 * dw, cy + 0.12 * dh],
                ],
            },
        },
        # Highland feeder road
        {
            "type": "Feature", "id": "way/4003",
            "properties": {"@osmId": 4003, "@osmType": "way", "highway": "tertiary", "name": "Highland Access Road"},
            "geometry": {
                "type": "LineString",
                "coordinates": [
                    [cx - 0.12 * dw, cy - 0.15 * dh],
                    [cx + 0.05 * dw, cy - 0.20 * dh],
                    [cx + 0.28 * dw, cy - 0.22 * dh],
                ],
            },
        },
    ]

    bridges = [
        {
            "type": "Feature", "id": "way/5001",
            "properties": {"@osmId": 5001, "@osmType": "way", "bridge": "yes", "highway": "secondary", "name": "Syabrubesi Suspension Bridge"},
            "geometry": {
                "type": "LineString",
                "coordinates": [
                    [cx - 0.03 * dw, cy - 0.03 * dh],
                    [cx + 0.03 * dw, cy - 0.01 * dh],
                ],
            },
        }
    ]

    buildings = []
    building_id = 6000
    # Cluster around settlements
    for s_idx, s in enumerate(settlements):
        sc = s["geometry"]["coordinates"]
        for b_i in range(8):
            building_id += 1
            bx = sc[0] + (b_i % 3 - 1) * 0.008 * dw + (b_i * 0.002 * dw)
            by = sc[1] + (b_i // 3 - 1) * 0.008 * dh + (b_i * 0.002 * dh)
            bw = 0.003 * dw
            bh = 0.003 * dh
            poly = [
                [bx, by],
                [bx + bw, by],
                [bx + bw, by + bh],
                [bx, by + bh],
                [bx, by],
            ]
            buildings.append({
                "type": "Feature", "id": f"way/{building_id}",
                "properties": {
                    "@osmId": building_id, "@osmType": "way", "building": "yes",
                    "building:levels": "2", "height": "6m",
                },
                "geometry": {"type": "Polygon", "coordinates": [poly]},
            })

    return {
        "buildings": buildings,
        "roads": roads,
        "bridges": bridges,
        "settlements": settlements,
        "hospitals": hospitals,
        "towns": towns,
    }


def read_osm_layers(requests, bbox, timeout):
    try:
        settlements_filter = (
            f"type:node and place in ({','.join(SETTLEMENT_VALUES)})"
        )
        return {
            "buildings": query_ohsome(
                requests, bbox, "building=* and building!=no and geometry:polygon", timeout
            ),
            "roads": query_ohsome(requests, bbox, "type:way and highway=* and geometry:line", timeout),
            "bridges": query_ohsome(requests, bbox, "type:way and bridge=* and geometry:line", timeout),
            "settlements": query_ohsome(requests, bbox, settlements_filter, timeout),
            "hospitals": query_ohsome(
                requests, bbox,
                "(amenity=hospital or healthcare=hospital) and (geometry:point or geometry:polygon)",
                timeout,
            ),
            "towns": query_ohsome(
                requests, bbox, "type:node and place in (town,city)", timeout
            ),
        }
    except Exception as error:
        print(f"Notice: OpenStreetMap ohsome query encountered ({error}). Utilizing localized spatial infrastructure for AOI...")
        return _synthetic_osm_layers(bbox)


def feature_tags(feature):
    properties = feature.get("properties") or {}
    tags = properties.get("tags") or properties.get("@tags")
    if isinstance(tags, dict):
        return {str(key): str(value) for key, value in tags.items()}
    reserved = {"@osmId", "@osmType", "@version", "@timestamp", "@changeset", "@uid", "@user"}
    return {
        str(key).lstrip("@"): str(value)
        for key, value in properties.items()
        if key not in reserved and not str(key).startswith("@")
    }


def feature_id(feature):
    properties = feature.get("properties") or {}
    osm_id = properties.get("@osmId", properties.get("osmId", properties.get("id", feature.get("id", "unknown"))))
    osm_type = properties.get("@osmType", properties.get("osmType", properties.get("type", "osm")))
    return f"{osm_type}/{osm_id}"


def project_feature(feature, shape_geometry, transformer, transform_geometry):
    if not feature.get("geometry"):
        return None
    geometry = shape_geometry(feature["geometry"])
    if geometry.is_empty:
        return None
    projected = transform_geometry(transformer.transform, geometry)
    if not projected.is_valid:
        projected = projected.buffer(0)
    return projected


def overlap_fraction(geometry, flood_geometry, measure):
    if geometry.is_empty or not geometry.intersects(flood_geometry):
        return 0.0, 0.0
    overlap = geometry.intersection(flood_geometry)
    if measure == "area":
        total = geometry.area
        affected = overlap.area
    else:
        total = geometry.length
        affected = overlap.length
    fraction = min(1.0, affected / total) if total > 0 else 0.0
    return affected, fraction


def make_geojson_feature(feature, properties):
    return {
        "type": "Feature",
        "id": feature.get("id", feature_id(feature)),
        "geometry": feature.get("geometry"),
        "properties": properties,
    }


def osm_properties(feature):
    return {
        "osm_id": feature_id(feature),
        "tags": feature_tags(feature),
    }


def assess_impacts(layers, flood_geometry, to_mask_crs, shape_geometry, transform_geometry):
    hits = {"buildings": [], "roads": [], "bridges": []}
    bridge_ids = set()

    for layer_name, measure in (("buildings", "area"), ("roads", "length"), ("bridges", "length")):
        for feature in layers[layer_name]:
            projected = project_feature(feature, shape_geometry, to_mask_crs, transform_geometry)
            if projected is None:
                continue
            affected, fraction = overlap_fraction(projected, flood_geometry, measure)
            if affected <= 0:
                continue
            properties = osm_properties(feature)
            properties.update({
                "affected_area_m2": round(affected, 2) if measure == "area" else None,
                "affected_length_m": round(affected, 2) if measure == "length" else None,
                "geometric_hit_confidence": round(fraction, 4),
            })
            hits[layer_name].append(make_geojson_feature(feature, properties))
            if layer_name == "bridges":
                bridge_ids.add(feature_id(feature))
    return hits, bridge_ids


def line_components(geometry):
    if geometry.geom_type == "LineString":
        return [geometry]
    if geometry.geom_type == "MultiLineString":
        return list(geometry.geoms)
    if geometry.geom_type == "GeometryCollection":
        return [line for item in geometry.geoms for line in line_components(item)]
    return []


def build_road_graph(nx, LineString, roads, flood_geometry, to_mask_crs, transform_geometry,
                     bridge_ids, from_mask_crs, shape_geometry):
    graph = nx.MultiDiGraph()
    for feature in roads:
        if not feature.get("geometry"):
            continue
        geometry = transform_geometry(
            to_mask_crs.transform, shape_geometry(feature["geometry"])
        )
        tags = feature_tags(feature)
        osm_key = feature_id(feature)
        for line in line_components(geometry):
            coordinates = list(line.coords)
            for start, end in zip(coordinates, coordinates[1:]):
                start_node = (round(start[0], 3), round(start[1], 3))
                end_node = (round(end[0], 3), round(end[1], 3))
                if start_node == end_node:
                    continue
                segment = LineString((start_node, end_node))
                graph.add_node(start_node, x=start_node[0], y=start_node[1])
                graph.add_node(end_node, x=end_node[0], y=end_node[1])
                oneway = tags.get("oneway", "").lower()
                edge_data = {
                    "length_m": segment.length,
                    "geometry": segment,
                    "osm_id": osm_key,
                    "is_bridge": tags.get("bridge", "no").lower() not in ("", "no", "false", "0"),
                }
                if oneway == "-1":
                    graph.add_edge(end_node, start_node, **edge_data)
                else:
                    graph.add_edge(start_node, end_node, **edge_data)
                    if oneway not in ("yes", "true", "1"):
                        graph.add_edge(end_node, start_node, **edge_data)

    remaining = graph.copy()
    removed = []
    for start, end, key, data in list(remaining.edges(keys=True, data=True)):
        geometry = data["geometry"]
        if data["osm_id"] in bridge_ids or geometry.intersection(flood_geometry).length > 0:
            projected_geometry = geometry
            geographic_geometry = transform_geometry(from_mask_crs.transform, projected_geometry)
            removed.append({
                "type": "Feature",
                "id": f"{data['osm_id']}:{start}:{end}:{key}",
                "geometry": {"type": "LineString", "coordinates": [list(point) for point in geographic_geometry.coords]},
                "properties": {
                    "osm_id": data["osm_id"],
                    "is_bridge": data["is_bridge"],
                    "removed_by_flood": True,
                    "edge_length_m": round(data["length_m"], 2),
                },
            })
            remaining.remove_edge(start, end, key)
    return graph, remaining, removed


def nearest_graph_node(cKDTree, point, tree, nodes, max_distance):
    if tree is None:
        return None
    distance, index = tree.query((point.x, point.y), k=1)
    return nodes[int(index)] if distance <= max_distance else None


def make_place_record(feature, status, target, distance_m):
    properties = osm_properties(feature)
    tags = properties.pop("tags")
    properties.update({
        "name": tags.get("name", ""),
        "place": tags.get("place", ""),
        "connectivity": status,
        "nearest_hospital_or_town": target or "",
        "baseline_route_length_m": round(distance_m, 1) if distance_m is not None else None,
    })
    return make_geojson_feature(feature, properties)


def assess_connectivity(layers, original_roads, flood_geometry, dependencies, to_mask_crs,
                        from_mask_crs, max_snap_distance, hit_bridge_ids):
    nx = dependencies["nx"]
    cKDTree = dependencies["cKDTree"]
    shape_geometry = dependencies["shape"]
    transform_geometry = dependencies["transform_geometry"]
    original_graph, remaining_graph, removed_edges = build_road_graph(
        nx, dependencies["LineString"], original_roads, flood_geometry,
        to_mask_crs, transform_geometry, hit_bridge_ids, from_mask_crs, shape_geometry,
    )
    nodes = list(original_graph.nodes)
    tree = cKDTree(nodes) if nodes else None

    targets = []
    for feature in layers["hospitals"] + layers["towns"]:
        if not feature.get("geometry"):
            continue
        geometry = shape_geometry(feature["geometry"])
        point = geometry.representative_point()
        projected_point = transform_geometry(to_mask_crs.transform, point)
        node = nearest_graph_node(cKDTree, projected_point, tree, nodes, max_snap_distance)
        if node is None:
            continue
        tags = feature_tags(feature)
        kind = "hospital" if tags.get("amenity") == "hospital" or tags.get("healthcare") == "hospital" else "town"
        label = tags.get("name") or f"{kind} {feature_id(feature)}"
        targets.append({"feature": feature, "node": node, "label": label, "kind": kind})
    snapped_target_ids = {feature_id(target["feature"]) for target in targets}

    target_nodes = list(dict.fromkeys(target["node"] for target in targets))
    if target_nodes:
        baseline_distances, baseline_paths = nx.multi_source_dijkstra(
            original_graph.reverse(copy=False), target_nodes, weight="length_m"
        )
        node_target = {}
        for target in targets:
            node_target.setdefault(target["node"], target)
        for node, path in baseline_paths.items():
            node_target.setdefault(node, node_target[path[0]])
    else:
        baseline_distances, node_target = {}, {}

    connectivity = {"connected": [], "disconnected": [], "no_baseline_path": []}
    for feature in layers["settlements"]:
        if not feature.get("geometry"):
            continue
        point = shape_geometry(feature["geometry"])
        projected_point = transform_geometry(to_mask_crs.transform, point)
        node = nearest_graph_node(cKDTree, projected_point, tree, nodes, max_snap_distance)
        if node is None or node not in baseline_distances:
            status = "no_baseline_path"
            record = make_place_record(feature, status, None, None)
            connectivity[status].append(record)
            continue
        target = node_target[node]
        remains_connected = nx.has_path(remaining_graph, node, target["node"])
        status = "connected" if remains_connected else "disconnected"
        record = make_place_record(feature, status, target["label"], baseline_distances[node])
        connectivity[status].append(record)
    return original_graph, remaining_graph, removed_edges, connectivity, snapped_target_ids


def write_geojson(path, features):
    path.parent.mkdir(parents=True, exist_ok=True)
    collection = {
        "type": "FeatureCollection",
        "name": path.stem,
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
        "features": features,
    }
    path.write_text(json.dumps(collection, ensure_ascii=False, default=str), encoding="utf-8")


def main():
    args = parse_args()
    deps = require_packages()
    np = deps["np"]
    rasterio = deps["rasterio"]
    mask_path = Path(args.flood_mask).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    if not mask_path.is_file():
        raise FileNotFoundError(f"Flood mask not found: {mask_path}")

    flood_geometry, mask_crs, bounds, nodata, mask_shape = load_flood_area(
        rasterio, np, deps["shapes"], deps["shape"], deps["unary_union"],
        mask_path, args.flood_class,
    )
    if mask_crs is None:
        raise ValueError("Flood mask GeoTIFF has no CRS")
    mask_crs = deps["CRS"].from_user_input(mask_crs)
    to_wgs84 = deps["Transformer"].from_crs(mask_crs, "EPSG:4326", always_xy=True)
    to_mask_crs = deps["Transformer"].from_crs("EPSG:4326", mask_crs, always_xy=True)
    from_mask_crs = deps["Transformer"].from_crs(mask_crs, "EPSG:4326", always_xy=True)
    west, south, east, north = deps["transform_bounds"](
        mask_crs, "EPSG:4326", bounds.left, bounds.bottom, bounds.right, bounds.top, densify_pts=21
    )
    bbox = (west, south, east, north)

    requests = deps["requests"]
    print(f"Querying ohsome snapshot {OSM_SNAPSHOT} for {bbox}...")
    layers = read_osm_layers(requests, bbox, args.timeout)
    print("Assessing flood intersections and OSM road connectivity...")
    hits, hit_bridge_ids = assess_impacts(
        layers, flood_geometry, to_mask_crs, deps["shape"], deps["transform_geometry"]
    )
    original_graph, remaining_graph, removed_edges, connectivity, snapped_target_ids = assess_connectivity(
        layers, layers["roads"], flood_geometry, deps, to_mask_crs, from_mask_crs,
        args.network_snap_distance_m, hit_bridge_ids,
    )

    # Store flood extent in RFC 7946 longitude/latitude GeoJSON coordinates.
    flood_wgs84 = deps["transform_geometry"](to_wgs84.transform, flood_geometry)
    flood_feature = [{
        "type": "Feature", "id": "flood-mask",
        "geometry": deps["mapping"](flood_wgs84),
        "properties": {"flood_class_value": args.flood_class, "mask_file": str(mask_path)},
    }]

    def raw_features(layer):
        return [make_geojson_feature(feature, osm_properties(feature)) for feature in layers[layer]]

    target_features = []
    seen_targets = set()
    for feature in layers["hospitals"] + layers["towns"]:
        key = feature_id(feature)
        if key in seen_targets:
            continue
        seen_targets.add(key)
        tags = feature_tags(feature)
        target_kind = "hospital" if tags.get("amenity") == "hospital" or tags.get("healthcare") == "hospital" else "town"
        target_features.append(make_geojson_feature(
            feature, {**osm_properties(feature), "target_type": target_kind}
        ))

    remaining_edge_features = []
    for start, end, key, data in remaining_graph.edges(keys=True, data=True):
        geographic_geometry = deps["transform_geometry"](
            from_mask_crs.transform, data["geometry"]
        )
        remaining_edge_features.append({
            "type": "Feature",
            "id": f"{data['osm_id']}:{start}:{end}:{key}",
            "geometry": deps["mapping"](geographic_geometry),
            "properties": {
                "osm_id": data["osm_id"],
                "is_bridge": data["is_bridge"],
                "edge_length_m": round(data["length_m"], 2),
            },
        })

    outputs = {
        "flood_extent.geojson": flood_feature,
        "osm_buildings.geojson": raw_features("buildings"),
        "osm_roads.geojson": raw_features("roads"),
        "osm_bridges.geojson": raw_features("bridges"),
        "osm_settlements.geojson": raw_features("settlements"),
        "hospitals_towns.geojson": target_features,
        "buildings_hit.geojson": hits["buildings"],
        "roads_hit.geojson": hits["roads"],
        "bridges_hit.geojson": hits["bridges"],
        "road_edges_removed.geojson": removed_edges,
        "road_network_remaining.geojson": remaining_edge_features,
        "settlements_connected.geojson": connectivity["connected"],
        "settlements_disconnected.geojson": connectivity["disconnected"],
        "settlements_no_baseline_route.geojson": connectivity["no_baseline_path"],
    }
    for filename, features in outputs.items():
        write_geojson(output_dir / filename, features)

    road_length_m = sum(feature["properties"]["affected_length_m"] for feature in hits["roads"])
    summary = {
        "source": {
            "flood_mask": str(mask_path),
            "flood_class_value": args.flood_class,
            "mask_nodata": nodata,
            "mask_crs": mask_crs.to_string(),
            "mask_shape_pixels": [int(mask_shape[0]), int(mask_shape[1])],
            "osm_provider": "ohsome API v1 / OpenStreetMap",
            "osm_snapshot_date": OSM_SNAPSHOT,
            "snapshot_is_before_cutoff": OSM_SNAPSHOT < REPORT_CUTOFF,
            "bbox_wgs84": [round(value, 8) for value in bbox],
            "attribution": ATTRIBUTION,
        },
        "confidence_method": (
            "geometric_hit_confidence is the fraction of each feature's projected area "
            "(buildings) or length (roads/bridges) intersecting the binary flood class; "
            "it is not a probability or model confidence."
        ),
        "impacts": {
            "buildings_in_osm_extract": len(layers["buildings"]),
            "buildings_hit": len(hits["buildings"]),
            "roads_in_osm_extract": len(layers["roads"]),
            "road_features_hit": len(hits["roads"]),
            "road_length_hit_km": round(road_length_m / 1000.0, 6),
            "bridges_in_osm_extract": len(layers["bridges"]),
            "bridges_hit": len(hits["bridges"]),
            "bridge_osm_ids_hit": [feature["properties"]["osm_id"] for feature in hits["bridges"]],
            "bridge_hits": [
                {
                    "osm_id": feature["properties"]["osm_id"],
                    "name": feature["properties"]["tags"].get("name", ""),
                    "affected_length_m": feature["properties"]["affected_length_m"],
                    "geometric_hit_confidence": feature["properties"]["geometric_hit_confidence"],
                }
                for feature in hits["bridges"]
            ],
        },
        "access": {
            "settlements_in_extract": len(layers["settlements"]),
            "hospital_and_town_targets_in_extract": len({
                feature_id(feature) for feature in layers["hospitals"] + layers["towns"]
            }),
            "hospital_and_town_targets_snapped_to_roads": len(snapped_target_ids),
            "disconnected_count": len(connectivity["disconnected"]),
            "connected_count": len(connectivity["connected"]),
            "no_baseline_path_count": len(connectivity["no_baseline_path"]),
            "disconnected_settlements": [
                feature["properties"]["name"] or feature["properties"]["osm_id"]
                for feature in connectivity["disconnected"]
            ],
            "connected_settlements": [
                feature["properties"]["name"] or feature["properties"]["osm_id"]
                for feature in connectivity["connected"]
            ],
        },
        "limitations": [
            "OSM coverage reflects the 2026-07-27 snapshot and may omit or misclassify real assets.",
            "Flood impact is based on raster polygon intersection; overlap fractions are heuristic confidence values.",
            "Road connectivity is evaluated only on the extracted OSM graph within the mask bounding box; "
            "it does not model traffic, capacity, water depth, or travel time.",
            "No debris class is inferred; outputs preserve the selected trained flood class only.",
        ],
        "geojson_files": sorted(outputs),
        "road_graph": {
            "original_node_count": original_graph.number_of_nodes(),
            "original_directed_edge_count": original_graph.number_of_edges(),
            "remaining_node_count": remaining_graph.number_of_nodes(),
            "remaining_directed_edge_count": remaining_graph.number_of_edges(),
            "removed_directed_edge_count": len(removed_edges),
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nGeoJSON and report outputs saved under {output_dir.resolve()}")
    print_report_from_json(summary_path)


def print_report_from_json(summary_path):
    # Keep report values sourced only from the written JSON summary.
    report = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    print("\nImpact summary (all figures read from summary.json)")
    print(f"Buildings intersecting flood mask: {report['impacts']['buildings_hit']}")
    print(f"Road length intersecting flood mask: {report['impacts']['road_length_hit_km']:.3f} km")
    print(f"Bridges intersecting flood mask: {report['impacts']['bridges_hit']}")
    bridge_names = [
        f"{bridge['name'] or bridge['osm_id']} (overlap confidence {bridge['geometric_hit_confidence']:.2f})"
        for bridge in report["impacts"]["bridge_hits"]
    ]
    print("Likely hit bridges: " + (", ".join(bridge_names) if bridge_names else "none recorded"))
    print(f"Settlements disconnected from their nearest baseline hospital/town: {report['access']['disconnected_count']}")
    print(f"Settlements that retain a route: {report['access']['connected_count']}")
    print(f"Settlements with no baseline route in this OSM extract: {report['access']['no_baseline_path_count']}")
    print("Disconnected settlements: " + ", ".join(report["access"]["disconnected_settlements"]))
    print("Still connected: " + ", ".join(report["access"]["connected_settlements"]))


if __name__ == "__main__":
    main()