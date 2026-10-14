import { useEffect, useRef } from "react";
import {
  Map as MapLibreMap,
  NavigationControl,
  GlobeControl,
  Marker,
  type GeoJSONSource,
  type MapMouseEvent,
} from "maplibre-gl";
import type {
  FeatureCollectionData,
  LayerVisibility,
  BBox,
  RunResults,
  SettlementPoint,
  LocationPin,
  BasemapMode,
} from "./types";

type MapViewProps = {
  bbox: BBox | null;
  results: RunResults | null;
  drawMode: boolean;
  visibility: LayerVisibility;
  basemap?: BasemapMode;
  locationPin?: LocationPin | null;
  onBboxDrawn: (bbox: BBox) => void;
  onDrawProgress: (message: string) => void;
  onMapCenterSelected: (center: { lat: number; lon: number }) => void;
  onSettlements: (points: SettlementPoint[]) => void;
  focusSettlement: SettlementPoint | null;
};

const EMPTY: FeatureCollectionData = { type: "FeatureCollection", features: [] };
const SOURCE_IDS = ["flood", "buildings", "roads", "bridges", "settlements"] as const;

// 100% Free basemap tile sources — zero API key required, zero watermarks!
const BASE_STYLE = {
  version: 8 as const,
  glyphs: "https://demotiles.maplibre.org/font/{fontstack}/{range}.pbf",
  sprite: "",
  sources: {
    // 1. High-resolution satellite imagery (photorealistic 3D Earth globe from space)
    "satellite-tiles": {
      type: "raster" as const,
      tiles: [
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
      ],
      tileSize: 256,
      attribution: "© Esri, Maxar, Earthstar Geographics",
      maxzoom: 19,
    },
    // 2. High-contrast Dark Gray Canvas (sleek disaster assessment view)
    "dark-tiles": {
      type: "raster" as const,
      tiles: [
        "https://services.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}",
      ],
      tileSize: 256,
      attribution: "© Esri, HERE, Garmin",
      maxzoom: 16,
    },
    // 3. OpenStreetMap standard streets / landmarks
    "osm-tiles": {
      type: "raster" as const,
      tiles: [
        "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
      ],
      tileSize: 256,
      attribution: "© OpenStreetMap contributors",
      maxzoom: 19,
    },
    // 4. Global 3D elevation terrain-RGB (Mapzen Terrarium open AWS public data)
    "terrain-dem": {
      type: "raster-dem" as const,
      tiles: ["https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"],
      tileSize: 256,
      encoding: "terrarium" as const,
      maxzoom: 15,
      attribution: "Terrain tiles by Mapzen",
    },
  },
  terrain: { source: "terrain-dem", exaggeration: 1.8 },
  sky: {
    "sky-color": "#020713",
    "horizon-color": "#0c253d",
    "fog-color": "#061524",
    "sky-horizon-blend": 0.45,
    "atmosphere-blend": 0.85,
  },
  layers: [
    {
      id: "background",
      type: "background" as const,
      paint: { "background-color": "#030712" },
    },
    // Satellite layer (default active)
    {
      id: "satellite-base",
      type: "raster" as const,
      source: "satellite-tiles",
      paint: { "raster-opacity": 1.0 },
    },
    // Dark canvas layer
    {
      id: "dark-base",
      type: "raster" as const,
      source: "dark-tiles",
      layout: { visibility: "none" as const },
      paint: { "raster-opacity": 0.95 },
    },
    // OSM street layer
    {
      id: "osm-base",
      type: "raster" as const,
      source: "osm-tiles",
      layout: { visibility: "none" as const },
      paint: { "raster-opacity": 0.92 },
    },
    // 3D terrain hillshade
    {
      id: "dem-hillshade",
      type: "hillshade" as const,
      source: "terrain-dem",
      paint: {
        "hillshade-exaggeration": 0.55,
        "hillshade-shadow-color": "#050d14",
        "hillshade-highlight-color": "#ffffff",
        "hillshade-accent-color": "#112a3d",
        "hillshade-illumination-direction": 335,
      },
    },
  ],
};

export default function MapView({
  bbox,
  results,
  drawMode,
  visibility,
  basemap = "satellite",
  locationPin = null,
  onBboxDrawn,
  onDrawProgress,
  onMapCenterSelected,
  onSettlements,
  focusSettlement,
}: MapViewProps) {
  const container = useRef<HTMLDivElement>(null);
  const mapRef = useRef<MapLibreMap | null>(null);
  const markerRef = useRef<Marker | null>(null);
  const drawModeRef = useRef(drawMode);
  const drawCallbackRef = useRef(onBboxDrawn);
  const progressCallbackRef = useRef(onDrawProgress);
  const mapCenterCallbackRef = useRef(onMapCenterSelected);
  const settlementsCallbackRef = useRef(onSettlements);
  const firstCorner = useRef<[number, number] | null>(null);
  const latestResults = useRef(results);
  const latestVisibility = useRef(visibility);
  const latestBasemap = useRef(basemap);

  useEffect(() => {
    drawModeRef.current = drawMode;
    if (!drawMode) firstCorner.current = null;
    if (mapRef.current) mapRef.current.getCanvas().style.cursor = drawMode ? "crosshair" : "grab";
  }, [drawMode]);
  useEffect(() => { drawCallbackRef.current = onBboxDrawn; }, [onBboxDrawn]);
  useEffect(() => { progressCallbackRef.current = onDrawProgress; }, [onDrawProgress]);
  useEffect(() => { mapCenterCallbackRef.current = onMapCenterSelected; }, [onMapCenterSelected]);
  useEffect(() => { settlementsCallbackRef.current = onSettlements; }, [onSettlements]);
  useEffect(() => { latestResults.current = results; }, [results]);
  useEffect(() => { latestVisibility.current = visibility; }, [visibility]);
  useEffect(() => { latestBasemap.current = basemap; }, [basemap]);

  useEffect(() => {
    if (!container.current || mapRef.current) return;

    // If bbox is provided, center on it; otherwise show global view centered on Nepal/Asia
    const initialCenter: [number, number] = bbox
      ? [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
      : [84.1240, 28.3949];

    const initialZoom = bbox ? 11 : 2.6;
    const initialPitch = bbox ? 60 : 22;

    const map = new MapLibreMap({
      container: container.current,
      style: BASE_STYLE as unknown as ConstructorParameters<typeof MapLibreMap>[0]["style"],
      center: initialCenter,
      zoom: initialZoom,
      pitch: initialPitch,
      bearing: 0,
      canvasContextAttributes: { antialias: true },
      attributionControl: false,
    });
    mapRef.current = map;

    // Controls: Navigation + Globe / Flat Toggle Control
    map.addControl(new NavigationControl({ visualizePitch: true }), "top-right");
    map.addControl(new GlobeControl(), "top-right");

    map.on("style.load", () => {
      // Configure 3D Globe Projection
      try {
        map.setProjection({ type: "globe" });
      } catch {
        // Fallback if globe not supported
      }

      // Configure atmospheric glow around planet Earth
      try {
        map.setSky({
          "sky-color": "#020713",
          "horizon-color": "#0c253d",
          "fog-color": "#061524",
          "sky-horizon-blend": 0.45,
          "atmosphere-blend": 0.85,
        });
      } catch {
        // Sky unsupported
      }

      // Sync initial basemap layer
      syncBasemap(map, latestBasemap.current);

      // Data sources for result layers
      map.addSource("area-box", { type: "geojson", data: EMPTY });
      map.addLayer({
        id: "area-box-fill",
        type: "fill",
        source: "area-box",
        paint: {
          "fill-color": "#54d1df",
          "fill-opacity": 0.12,
        },
      });
      map.addLayer({
        id: "area-box-line",
        type: "line",
        source: "area-box",
        paint: {
          "line-color": "#54d1df",
          "line-width": 2.5,
          "line-dasharray": [3, 2],
        },
      });

      for (const id of SOURCE_IDS) map.addSource(id, { type: "geojson", data: EMPTY });

      // Flood fill layer
      map.addLayer({
        id: "flood-fill", type: "fill", source: "flood",
        paint: {
          "fill-color": "#30c7df",
          "fill-opacity": 0.55,
          "fill-outline-color": "#b6f5fb",
        },
      });

      // 3D buildings
      map.addLayer({
        id: "buildings-3d", type: "fill-extrusion", source: "buildings",
        paint: {
          "fill-extrusion-color": ["case", ["==", ["get", "cut"], true], "#ef5a55", "#4a8fa0"],
          "fill-extrusion-height": [
            "coalesce",
            ["to-number", ["get", "height_m"]],
            ["*", ["to-number", ["get", "building_levels"]], 3],
            6,
          ],
          "fill-extrusion-base": 0,
          "fill-extrusion-opacity": 0.92,
          "fill-extrusion-vertical-gradient": true,
        },
      });

      // Roads
      map.addLayer({
        id: "roads-line", type: "line", source: "roads",
        paint: {
          "line-color": ["case", ["==", ["get", "cut"], true], "#ff6b6b", "#f0e68c"],
          "line-width": ["interpolate", ["linear"], ["zoom"], 8, 1.5, 15, 5],
          "line-opacity": 0.95,
          "line-blur": 0.5,
        },
      });

      // Bridges
      map.addLayer({
        id: "bridges-line", type: "line", source: "bridges",
        paint: {
          "line-color": ["case", ["==", ["get", "cut"], true], "#ff4841", "#f5d87a"],
          "line-width": ["interpolate", ["linear"], ["zoom"], 8, 2.5, 15, 7],
          "line-gap-width": 1,
        },
      });

      // Settlement glow (halo)
      map.addLayer({
        id: "settlements-glow", type: "circle", source: "settlements",
        paint: {
          "circle-color": ["case",
            ["==", ["get", "cut_off"], true], "#f2544d",
            ["==", ["get", "cut_off"], false], "#49c48a", "#c6a75b",
          ],
          "circle-radius": ["interpolate", ["linear"], ["zoom"], 7, 10, 13, 18],
          "circle-opacity": 0.25,
          "circle-blur": 1,
        },
      });

      // Settlement points
      map.addLayer({
        id: "settlements-points", type: "circle", source: "settlements",
        paint: {
          "circle-color": ["case",
            ["==", ["get", "cut_off"], true], "#f2544d",
            ["==", ["get", "cut_off"], false], "#49c48a", "#c6a75b",
          ],
          "circle-radius": ["interpolate", ["linear"], ["zoom"], 7, 5, 13, 9],
          "circle-stroke-color": "#0d1b24",
          "circle-stroke-width": 2,
        },
      });

      syncVisibility(map, latestVisibility.current);
      if (bbox) setAreaBox(map, bbox);

      const current = latestResults.current;
      if (current) void loadResults(map, current, settlementsCallbackRef.current, latestVisibility.current);
    });

    map.on("click", (event: MapMouseEvent) => {
      if (!drawModeRef.current) return;
      const corner: [number, number] = [event.lngLat.lng, event.lngLat.lat];
      if (!firstCorner.current) {
        firstCorner.current = corner;
        mapCenterCallbackRef.current({ lat: corner[1], lon: corner[0] });
        progressCallbackRef.current("Click the opposite corner to finish the box.");
        return;
      }
      const first = firstCorner.current;
      firstCorner.current = null;
      const drawn: BBox = [
        Math.min(first[0], corner[0]), Math.min(first[1], corner[1]),
        Math.max(first[0], corner[0]), Math.max(first[1], corner[1]),
      ];
      drawCallbackRef.current(drawn);
      progressCallbackRef.current("");
    });

    return () => {
      markerRef.current?.remove();
      markerRef.current = null;
      map.remove();
      mapRef.current = null;
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Update area box when bbox changes
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    const isLoaded = map.isStyleLoaded();
    const updateBox = () => {
      if (bbox) {
        setAreaBox(map, bbox);
        map.fitBounds([[bbox[0], bbox[1]], [bbox[2], bbox[3]]], {
          padding: 80,
          maxZoom: 14,
          duration: 1100,
        });
      } else {
        clearAreaBox(map);
      }
    };
    if (isLoaded) {
      updateBox();
    } else {
      map.once("style.load", updateBox);
    }
  }, [bbox?.join(",")]);

  // Sync Basemap layer switcher
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    if (map.isStyleLoaded()) {
      syncBasemap(map, basemap);
    } else {
      map.once("style.load", () => syncBasemap(map, basemap));
    }
  }, [basemap]);

  // Update verified Location Pin & trigger smooth flyTo
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;

    // Remove existing marker
    if (markerRef.current) {
      markerRef.current.remove();
      markerRef.current = null;
    }

    if (!locationPin) return;

    // Create custom pin element
    const el = document.createElement("div");
    el.className = "verified-location-pin";
    el.innerHTML = `
      <div class="pin-pulse"></div>
      <div class="pin-icon">📍</div>
      <div class="pin-badge">
        <strong>${escapeHtml(locationPin.name)}</strong>
        <span>${locationPin.lat.toFixed(4)}° N, ${locationPin.lon.toFixed(4)}° E</span>
      </div>
    `;

    const marker = new Marker({ element: el, anchor: "bottom" })
      .setLngLat([locationPin.lon, locationPin.lat])
      .addTo(map);

    markerRef.current = marker;

    // Cinematic flyTo the verified location
    const targetZoom = locationPin.zoom ?? 11.5;
    map.flyTo({
      center: [locationPin.lon, locationPin.lat],
      zoom: targetZoom,
      pitch: 52,
      bearing: -12,
      duration: 1600,
      essential: true,
    });
  }, [locationPin?.lat, locationPin?.lon, locationPin?.name, locationPin?.zoom]);

  // Layer visibility
  useEffect(() => {
    const map = mapRef.current;
    if (map) syncVisibility(map, visibility);
  }, [visibility]);

  // Load results
  useEffect(() => {
    const map = mapRef.current;
    if (map && results) {
      const apply = () => void loadResults(map, results, settlementsCallbackRef.current, latestVisibility.current);
      if (map.isStyleLoaded()) apply();
      else map.once("style.load", apply);
    }
  }, [results?.run_id]);

  // Focus specific settlement
  useEffect(() => {
    if (!mapRef.current || !focusSettlement) return;
    mapRef.current.flyTo({
      center: focusSettlement.coordinates,
      zoom: 15,
      pitch: 65,
      bearing: 20,
      essential: true,
      duration: 1200,
    });
  }, [focusSettlement]);

  return <div className="map-root" ref={container} aria-label="Interactive 3D Earth flood map" />;
}

function escapeHtml(text: string): string {
  return text.replace(/[&<>"']/g, (match) => {
    switch (match) {
      case "&": return "&amp;";
      case "<": return "&lt;";
      case ">": return "&gt;";
      case '"': return "&quot;";
      case "'": return "&#39;";
      default: return match;
    }
  });
}

function syncBasemap(map: MapLibreMap, basemap: BasemapMode) {
  if (map.getLayer("satellite-base")) {
    map.setLayoutProperty("satellite-base", "visibility", basemap === "satellite" ? "visible" : "none");
  }
  if (map.getLayer("dark-base")) {
    map.setLayoutProperty("dark-base", "visibility", basemap === "dark" ? "visible" : "none");
  }
  if (map.getLayer("osm-base")) {
    map.setLayoutProperty("osm-base", "visibility", basemap === "streets" ? "visible" : "none");
  }
}

function clearAreaBox(map: MapLibreMap) {
  const source = map.getSource("area-box") as GeoJSONSource | undefined;
  if (!source) return;
  source.setData(EMPTY);
}

function setAreaBox(map: MapLibreMap, bbox: BBox) {
  const source = map.getSource("area-box") as GeoJSONSource | undefined;
  if (!source) return;
  source.setData({
    type: "FeatureCollection",
    features: [{
      type: "Feature",
      properties: {},
      geometry: {
        type: "Polygon",
        coordinates: [[
          [bbox[0], bbox[1]], [bbox[2], bbox[1]], [bbox[2], bbox[3]],
          [bbox[0], bbox[3]], [bbox[0], bbox[1]],
        ]],
      },
    }],
  });
}

function syncVisibility(map: MapLibreMap, visibility: LayerVisibility) {
  const layerMap: Record<keyof LayerVisibility, string[]> = {
    flood: ["flood-fill"],
    buildings: ["buildings-3d"],
    roads: ["roads-line"],
    bridges: ["bridges-line"],
    settlements: ["settlements-points", "settlements-glow"],
    hillshade: ["dem-hillshade"],
  };
  for (const [key, layerIds] of Object.entries(layerMap)) {
    for (const layerId of layerIds) {
      if (map.getLayer(layerId)) {
        map.setLayoutProperty(layerId, "visibility", visibility[key as keyof LayerVisibility] ? "visible" : "none");
      }
    }
  }
}

async function loadResults(
  map: MapLibreMap,
  results: RunResults,
  onSettlements: (points: SettlementPoint[]) => void,
  visibility: LayerVisibility,
) {
  const layerEntries = [
    ["flood", results.urls.flood_mask],
    ["buildings", results.urls.damaged_buildings],
    ["roads", results.urls.roads],
    ["bridges", results.urls.bridges],
    ["settlements", results.urls.settlements],
  ] as const;

  await Promise.all(layerEntries.map(async ([sourceId, url]) => {
    try {
      const response = await fetch(url);
      if (!response.ok) return;
      const collection = await response.json() as FeatureCollectionData;
      const source = map.getSource(sourceId) as GeoJSONSource | undefined;
      source?.setData(collection);
      if (sourceId === "settlements") {
        const points = collection.features.flatMap((feature) => {
          if (feature.geometry.type !== "Point") return [];
          const properties = feature.properties ?? {};
          return [{
            id: String(properties.osm_id ?? feature.id ?? ""),
            name: String(properties.name ?? properties.osm_id ?? "Unnamed settlement"),
            coordinates: feature.geometry.coordinates.slice(0, 2) as [number, number],
          }];
        });
        onSettlements(points);
      }
    } catch {
      // Skip failed layers gracefully
    }
  }));

  // Add run-specific DEM terrain if available
  if (results.urls.dem_tiles && !map.getSource("dem-run")) {
    try {
      map.addSource("dem-run", {
        type: "raster-dem",
        tiles: [results.urls.dem_tiles],
        tileSize: 256,
        encoding: "mapbox",
        maxzoom: 16,
      });
      map.setTerrain({ source: "dem-run", exaggeration: 1.8 });
    } catch {
      // DEM source already added or failed
    }
    syncVisibility(map, visibility);
  }
}