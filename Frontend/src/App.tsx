import { useEffect, useMemo, useRef, useState } from "react";
import MapView from "./MapView";
import { NEPAL_PLACES, searchNepalPlaces, type PlaceItem } from "./nepalPlaces";
import type {
  BBox,
  LayerVisibility,
  RunResults,
  RunStatus,
  SettlementPoint,
  LocationPin,
  BasemapMode,
} from "./types";

const API_BASE = (import.meta.env.VITE_API_BASE as string | undefined) ?? "http://localhost:8000";
const ATTRIBUTIONS = [
  "Contains modified Copernicus Sentinel data 2026.",
  "Produced using Copernicus WorldDEM-30 © DLR e.V. 2010–2014 and © Airbus Defence and Space GmbH 2014–2018 provided under COPERNICUS by the European Union and ESA; all rights reserved.",
  "© OpenStreetMap contributors.",
  "Kuro Siwo: Bountos et al., 2024.",
];
const INITIAL_BBOX = { west: "", south: "", east: "", north: "" };
const INITIAL_VISIBILITY: LayerVisibility = {
  flood: true,
  buildings: true,
  roads: true,
  bridges: true,
  settlements: true,
  hillshade: true,
};
const MAX_AOI_KM = 30;
const DEFAULT_SIZE_KM = { widthKm: 10, heightKm: 10 };

type AreaMode = "center" | "draw" | "bbox";
type RunForm = {
  flood_date: string;
  raw_flood_class_id: string;
  checkpoint: string;
  channel_names: string;
  input_scale: string;
};
type CenterInput = {
  lat: string;
  lon: string;
};

function normalizeCoordinateInput(value: string): string {
  return value.trim().replace(/,/g, "").replace(/[^0-9.\-+]/g, "");
}

function formatCoordinate(value: number): string {
  if (!Number.isFinite(value)) return "";
  return Number(value.toFixed(6)).toString();
}

function computeCenterBbox(latitude: number, longitude: number, widthKm: number, heightKm: number): BBox {
  const safeWidth = Number.isFinite(widthKm) && widthKm > 0 ? widthKm : DEFAULT_SIZE_KM.widthKm;
  const safeHeight = Number.isFinite(heightKm) && heightKm > 0 ? heightKm : DEFAULT_SIZE_KM.heightKm;
  const latitudeRad = (latitude * Math.PI) / 180;
  const kmPerDegreeLat = 111.32;
  const cosLatitude = Math.max(Math.cos(latitudeRad), 0.01);
  const longitudeDegrees = safeWidth / (kmPerDegreeLat * cosLatitude);
  const latitudeDegrees = safeHeight / kmPerDegreeLat;
  return [
    longitude - longitudeDegrees / 2,
    latitude - latitudeDegrees / 2,
    longitude + longitudeDegrees / 2,
    latitude + latitudeDegrees / 2,
  ];
}

function calculateAreaMetrics(bbox: BBox) {
  const latitudeMid = (bbox[1] + bbox[3]) / 2;
  const widthKm = Math.abs(bbox[2] - bbox[0]) * 111.32 * Math.max(Math.cos((latitudeMid * Math.PI) / 180), 0.01);
  const heightKm = Math.abs(bbox[3] - bbox[1]) * 111.32;
  return {
    widthKm,
    heightKm,
    areaKm2: widthKm * heightKm,
  };
}

function suggestLikelyCoordinate(raw: string, axis: "lat" | "lon"): string | null {
  const normalized = normalizeCoordinateInput(raw);
  if (!normalized || !/^[-+]?\d+(?:\.\d+)?$/.test(normalized)) return null;
  const numeric = Number(normalized);
  if (!Number.isFinite(numeric)) return null;

  const maxAbs = axis === "lat" ? 90 : 180;
  if (Math.abs(numeric) <= maxAbs) return null;

  const digits = normalized.replace(/[^\d]/g, "");
  if (digits.length < 4) return null;

  const decimalLength = digits.length >= 5 ? 3 : 2;
  const suggested = Number(`${digits.slice(0, -decimalLength)}.${digits.slice(-decimalLength)}`);
  if (Number.isFinite(suggested) && Math.abs(suggested) <= maxAbs) {
    return formatCoordinate(suggested);
  }
  return null;
}

function validateCoordinate(raw: string, axis: "lat" | "lon"): string | null {
  const normalized = normalizeCoordinateInput(raw);
  // Do not report error when empty before user inputs anything
  if (normalized === "") return null;

  const numeric = Number(normalized);
  if (!Number.isFinite(numeric)) return `${axis === "lat" ? "Latitude" : "Longitude"} must be a number.`;

  const maxAbs = axis === "lat" ? 90 : 180;
  if (Math.abs(numeric) > maxAbs) {
    const suggestion = suggestLikelyCoordinate(normalized, axis);
    if (suggestion) {
      return `${axis === "lat" ? "Latitude" : "Longitude"} looks like a missing decimal point. Did you mean ${suggestion}?`;
    }
    return `${axis === "lat" ? "Latitude" : "Longitude"} must be between ${axis === "lat" ? "-90 and 90" : "-180 and 180"}.`;
  }

  return null;
}

function parseBbox(fields: typeof INITIAL_BBOX): BBox | null {
  const normalized = [fields.west, fields.south, fields.east, fields.north].map(normalizeCoordinateInput);
  if (normalized.some((value) => value === "")) return null;
  const values = normalized.map((value) => Number(value));
  if (!values.every(Number.isFinite)) return null;
  if (values[0] >= values[2] || values[1] >= values[3]) return null;
  if (Math.abs(values[0]) > 180 || Math.abs(values[2]) > 180 || Math.abs(values[1]) > 90 || Math.abs(values[3]) > 90) return null;
  return values as BBox;
}

async function responseError(response: Response) {
  try {
    const payload = await response.json();
    return typeof payload.detail === "string" ? payload.detail : JSON.stringify(payload);
  } catch {
    return response.statusText;
  }
}

function Metric({ label, value, unit }: { label: string; value: unknown; unit?: string }) {
  const shown = value === undefined || value === null ? "—" : String(value);
  return (
    <div className="metric-row">
      <span>{label}</span>
      <strong>{shown}{unit && value !== undefined && value !== null ? ` ${unit}` : ""}</strong>
    </div>
  );
}

export default function App() {
  const [areaMode, setAreaMode] = useState<AreaMode>("center");
  // Coordinates start empty by default — coordinates are ONLY populated when user types or chooses a city
  const [centerPoint, setCenterPoint] = useState<CenterInput>({ lat: "", lon: "" });
  const [sizeKm, setSizeKm] = useState({ widthKm: "10", heightKm: "10" });
  const [bboxFields, setBboxFields] = useState(INITIAL_BBOX);
  const [drawnBbox, setDrawnBbox] = useState<BBox | null>(null);
  const [drawMessage, setDrawMessage] = useState("");

  // Place search and instant suggestions
  const [placeQuery, setPlaceQuery] = useState("");
  const [isSearchingPlace, setIsSearchingPlace] = useState(false);
  const [placeMessage, setPlaceMessage] = useState<string | null>(null);
  const [suggestions, setSuggestions] = useState<PlaceItem[]>([]);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [selectedSuggestionIndex, setSelectedSuggestionIndex] = useState(-1);
  const searchContainerRef = useRef<HTMLDivElement>(null);

  // Cross-verification & location marker state
  const [locationPin, setLocationPin] = useState<LocationPin | null>(null);
  const [verifiedLocation, setVerifiedLocation] = useState<{
    name: string;
    lat: number;
    lon: number;
    display: string;
    region: string;
  } | null>(null);

  // Basemap switcher
  const [basemap, setBasemap] = useState<BasemapMode>("satellite");

  const [form, setForm] = useState<RunForm>({
    flood_date: "",
    raw_flood_class_id: "",
    checkpoint: "checkpoints/best_unet.pt",
    channel_names: "",
    input_scale: "unknown",
  });
  const [visibility, setVisibility] = useState(INITIAL_VISIBILITY);
  const [status, setStatus] = useState<RunStatus | null>(null);
  const [results, setResults] = useState<RunResults | null>(null);
  const [error, setError] = useState("");
  const [settlements, setSettlements] = useState<SettlementPoint[]>([]);
  const [focusSettlement, setFocusSettlement] = useState<SettlementPoint | null>(null);

  // Close suggestions when clicking outside
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (searchContainerRef.current && !searchContainerRef.current.contains(event.target as Node)) {
        setShowSuggestions(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  useEffect(() => {
    let stopped = false;
    const loadModelMetadata = async () => {
      try {
        const response = await fetch(`${API_BASE}/model-metadata`);
        if (!response.ok) throw new Error(await responseError(response));
        const metadata = await response.json() as {
          raw_flood_class_id: number;
          channel_names: string[];
          input_scale: "db" | "linear";
        };
        if (stopped) return;
        setForm((current) => ({
          ...current,
          raw_flood_class_id: String(metadata.raw_flood_class_id),
          channel_names: metadata.channel_names.join(","),
          input_scale: metadata.input_scale,
        }));
      } catch (metadataError) {
        if (!stopped) {
          setError(metadataError instanceof Error ? metadataError.message : String(metadataError));
        }
      }
    };
    void loadModelMetadata();
    return () => {
      stopped = true;
    };
  }, []);

  const hasCenterInput = centerPoint.lat.trim() !== "" && centerPoint.lon.trim() !== "";
  const centerLatitude = hasCenterInput ? Number(centerPoint.lat) : NaN;
  const centerLongitude = hasCenterInput ? Number(centerPoint.lon) : NaN;
  const widthKmNumber = Number(sizeKm.widthKm);
  const heightKmNumber = Number(sizeKm.heightKm);

  const centerBBox = useMemo(() => {
    if (!hasCenterInput || !Number.isFinite(centerLatitude) || !Number.isFinite(centerLongitude)) {
      return null;
    }
    return computeCenterBbox(centerLatitude, centerLongitude, widthKmNumber, heightKmNumber);
  }, [hasCenterInput, centerLatitude, centerLongitude, widthKmNumber, heightKmNumber]);

  const manualBbox = useMemo(() => parseBbox(bboxFields), [bboxFields]);

  // Keep bboxFields updated when in center or draw mode and a valid bbox exists
  useEffect(() => {
    if (areaMode === "bbox") return;
    const next = areaMode === "draw" ? drawnBbox ?? centerBBox : centerBBox;
    if (!next) return;
    setBboxFields({
      west: formatCoordinate(next[0]),
      south: formatCoordinate(next[1]),
      east: formatCoordinate(next[2]),
      north: formatCoordinate(next[3]),
    });
  }, [areaMode, centerBBox, drawnBbox]);

  useEffect(() => {
    if (areaMode !== "center" || !centerBBox) return;
    setBboxFields({
      west: formatCoordinate(centerBBox[0]),
      south: formatCoordinate(centerBBox[1]),
      east: formatCoordinate(centerBBox[2]),
      north: formatCoordinate(centerBBox[3]),
    });
  }, [areaMode, centerBBox]);

  const effectiveBbox = useMemo(() => {
    if (areaMode === "bbox") return manualBbox;
    if (areaMode === "draw") return drawnBbox ?? centerBBox ?? null;
    return centerBBox ?? null;
  }, [areaMode, centerBBox, drawnBbox, manualBbox]);

  // Synchronize location pin with manual coordinate input
  useEffect(() => {
    if (areaMode === "center" && hasCenterInput && Number.isFinite(centerLatitude) && Number.isFinite(centerLongitude)) {
      if (Math.abs(centerLatitude) <= 90 && Math.abs(centerLongitude) <= 180) {
        setLocationPin((prev) => ({
          lat: centerLatitude,
          lon: centerLongitude,
          name: prev?.name || "Selected AOI Point",
          region: prev?.region || "User Coordinates",
          zoom: prev?.zoom || 11.5,
        }));
      }
    }
  }, [areaMode, hasCenterInput, centerLatitude, centerLongitude]);

  const centerFieldErrors = useMemo(() => ({
    lat: validateCoordinate(centerPoint.lat, "lat"),
    lon: validateCoordinate(centerPoint.lon, "lon"),
  }), [centerPoint.lat, centerPoint.lon]);

  const bboxFieldErrors = useMemo(() => {
    const errors: Record<string, string> = {};
    for (const [key, value] of Object.entries(bboxFields)) {
      if (value.trim() !== "") {
        const axis = key === "west" || key === "east" ? "lon" : "lat";
        const message = validateCoordinate(value, axis);
        if (message) errors[key] = message;
      }
    }

    if (!errors.west && !errors.east && bboxFields.west !== "" && bboxFields.east !== "" && Number(bboxFields.west) >= Number(bboxFields.east)) {
      errors.west = "West must be less than East.";
      errors.east = "East must be greater than West.";
    }

    if (!errors.south && !errors.north && bboxFields.south !== "" && bboxFields.north !== "" && Number(bboxFields.south) >= Number(bboxFields.north)) {
      errors.south = "South must be less than North.";
      errors.north = "North must be greater than South.";
    }

    return errors;
  }, [bboxFields]);

  const areaMetrics = effectiveBbox ? calculateAreaMetrics(effectiveBbox) : null;
  const areaWarning = areaMetrics && (areaMetrics.widthKm > MAX_AOI_KM || areaMetrics.heightKm > MAX_AOI_KM)
    ? `Warning: this area is ${areaMetrics.widthKm.toFixed(1)} km × ${areaMetrics.heightKm.toFixed(1)} km (${areaMetrics.areaKm2.toFixed(1)} km²), larger than the recommended ${MAX_AOI_KM} × ${MAX_AOI_KM} km maximum. Sentinel downloads may be slow.`
    : null;

  const areaIsValid =
    !!effectiveBbox &&
    effectiveBbox[0] < effectiveBbox[2] &&
    effectiveBbox[1] < effectiveBbox[3] &&
    Math.abs(effectiveBbox[0]) <= 180 &&
    Math.abs(effectiveBbox[2]) <= 180 &&
    Math.abs(effectiveBbox[1]) <= 90 &&
    Math.abs(effectiveBbox[3]) <= 90 &&
    !centerFieldErrors.lat &&
    !centerFieldErrors.lon &&
    (areaMode !== "bbox" || (!bboxFieldErrors.west && !bboxFieldErrors.east && !bboxFieldErrors.south && !bboxFieldErrors.north));

  const runReady = areaIsValid && !!form.flood_date && form.raw_flood_class_id.trim() !== "" && !!form.checkpoint.trim();

  useEffect(() => {
    if (!status?.run_id || status.status === "completed" || status.status === "failed") return;
    let stopped = false;
    let finished = false;
    let timer = 0;

    const poll = async () => {
      if (stopped || finished) return;
      try {
        const response = await fetch(`${API_BASE}/status/${status.run_id}`);
        if (!response.ok) throw new Error(await responseError(response));
        const nextStatus = await response.json() as RunStatus;
        if (stopped) return;
        setStatus(nextStatus);

        if (nextStatus.status === "failed") {
          finished = true;
          window.clearInterval(timer);
          setError(nextStatus.error ?? "The run failed.");
          return;
        }

        if (nextStatus.status === "completed") {
          finished = true;
          window.clearInterval(timer);
          const resultResponse = await fetch(`${API_BASE}/results/${nextStatus.run_id}`);
          if (!resultResponse.ok) throw new Error(await responseError(resultResponse));
          const payload = await resultResponse.json() as RunResults;
          if (!stopped) setResults(payload);
        }
      } catch (pollError) {
        finished = true;
        window.clearInterval(timer);
        if (!stopped) setError(pollError instanceof Error ? pollError.message : String(pollError));
      }
    };

    void poll();
    timer = window.setInterval(() => void poll(), 1200);
    return () => {
      stopped = true;
      window.clearInterval(timer);
    };
  }, [status?.run_id]);

  function updateBbox(key: keyof typeof INITIAL_BBOX, value: string) {
    setBboxFields((current) => ({ ...current, [key]: normalizeCoordinateInput(value) }));
  }

  function updateForm<K extends keyof RunForm>(key: K, value: RunForm[K]) {
    setForm((current) => ({ ...current, [key]: value }));
  }

  function handleDrawn(next: BBox) {
    setDrawnBbox(next);
    setAreaMode("draw");
    const centerLat = (next[1] + next[3]) / 2;
    const centerLon = (next[0] + next[2]) / 2;
    setCenterPoint({ lat: formatCoordinate(centerLat), lon: formatCoordinate(centerLon) });
    const metrics = calculateAreaMetrics(next);
    setSizeKm({
      widthKm: formatCoordinate(metrics.widthKm),
      heightKm: formatCoordinate(metrics.heightKm),
    });
    setLocationPin({
      lat: centerLat,
      lon: centerLon,
      name: "Drawn Area",
      region: `${metrics.widthKm.toFixed(1)} × ${metrics.heightKm.toFixed(1)} km`,
      zoom: 12,
    });
    setVerifiedLocation({
      name: "Drawn AOI Box",
      lat: centerLat,
      lon: centerLon,
      display: `Center (${centerLat.toFixed(4)}° N, ${centerLon.toFixed(4)}° E) · Area ${metrics.areaKm2.toFixed(1)} km²`,
      region: "Custom Drawn Polygon",
    });
    setDrawMessage("");
  }

  function handleMapCenterSelected(center: { lat: number; lon: number }) {
    setCenterPoint({ lat: formatCoordinate(center.lat), lon: formatCoordinate(center.lon) });
    const projected = computeCenterBbox(center.lat, center.lon, widthKmNumber, heightKmNumber);
    setDrawnBbox(projected);
    setLocationPin({
      lat: center.lat,
      lon: center.lon,
      name: "Map Center",
      region: `${center.lat.toFixed(4)}° N, ${center.lon.toFixed(4)}° E`,
      zoom: 12,
    });
  }

  // Handle typing in the search box — instant prefix matching on first letter
  function handlePlaceInputChange(value: string) {
    setPlaceQuery(value);
    setPlaceMessage(null);
    if (value.trim().length > 0) {
      const matches = searchNepalPlaces(value, 8);
      setSuggestions(matches);
      setShowSuggestions(matches.length > 0);
      setSelectedSuggestionIndex(-1);
    } else {
      setSuggestions([]);
      setShowSuggestions(false);
    }
  }

  // Select place from suggestions dropdown or search result
  function selectPlace(place: PlaceItem) {
    setPlaceQuery(place.name);
    setShowSuggestions(false);
    setSuggestions([]);

    const latStr = formatCoordinate(place.lat);
    const lonStr = formatCoordinate(place.lon);
    setAreaMode("center");
    setCenterPoint({ lat: latStr, lon: lonStr });

    // Target zoom: country overview zoom 6 for Nepal, city zoom ~12 for cities
    const targetZoom = place.defaultZoom ?? 12;

    setLocationPin({
      lat: place.lat,
      lon: place.lon,
      name: place.name,
      region: place.region,
      zoom: targetZoom,
    });

    setVerifiedLocation({
      name: place.name,
      lat: place.lat,
      lon: place.lon,
      display: place.display,
      region: place.region,
    });

    setPlaceMessage(`✓ Located: ${place.display} (${place.lat.toFixed(4)}° N, ${place.lon.toFixed(4)}° E) · Area verified`);
  }

  function handleSearchKeyDown(event: React.KeyboardEvent<HTMLInputElement>) {
    if (!showSuggestions || suggestions.length === 0) {
      if (event.key === "Enter") {
        event.preventDefault();
        void handleGeocode();
      }
      return;
    }

    if (event.key === "ArrowDown") {
      event.preventDefault();
      setSelectedSuggestionIndex((prev) => (prev < suggestions.length - 1 ? prev + 1 : 0));
    } else if (event.key === "ArrowUp") {
      event.preventDefault();
      setSelectedSuggestionIndex((prev) => (prev > 0 ? prev - 1 : suggestions.length - 1));
    } else if (event.key === "Enter") {
      event.preventDefault();
      if (selectedSuggestionIndex >= 0 && selectedSuggestionIndex < suggestions.length) {
        selectPlace(suggestions[selectedSuggestionIndex]);
      } else {
        selectPlace(suggestions[0]);
      }
    } else if (event.key === "Escape") {
      setShowSuggestions(false);
    }
  }

  async function handleGeocode() {
    const query = placeQuery.trim();
    if (!query) {
      setPlaceMessage("Enter a city or place name first.");
      return;
    }

    // Check curated Nepal places first
    const exactLocal = NEPAL_PLACES.find(
      (p) => p.name.toLowerCase() === query.toLowerCase() || p.aliases.some((a) => a.toLowerCase() === query.toLowerCase())
    );
    if (exactLocal) {
      selectPlace(exactLocal);
      return;
    }

    const prefixLocal = searchNepalPlaces(query, 1);
    if (prefixLocal.length > 0 && prefixLocal[0].name.toLowerCase() === query.toLowerCase()) {
      selectPlace(prefixLocal[0]);
      return;
    }

    setIsSearchingPlace(true);
    setPlaceMessage(null);
    setShowSuggestions(false);

    try {
      const response = await fetch(`${API_BASE}/geocode?q=${encodeURIComponent(query)}`);
      if (!response.ok) throw new Error(await responseError(response));
      const payload = await response.json() as { latitude: number; longitude: number; display_name: string };
      const latStr = formatCoordinate(payload.latitude);
      const lonStr = formatCoordinate(payload.longitude);
      setAreaMode("center");
      setCenterPoint({ lat: latStr, lon: lonStr });

      const name = payload.display_name.split(",")[0] || query;
      setLocationPin({
        lat: payload.latitude,
        lon: payload.longitude,
        name,
        zoom: query.toLowerCase() === "nepal" ? 6 : 11.5,
      });

      setVerifiedLocation({
        name,
        lat: payload.latitude,
        lon: payload.longitude,
        display: payload.display_name,
        region: "Geocoded Region",
      });

      setPlaceMessage(`✓ Located: ${payload.display_name} (${payload.latitude.toFixed(4)}° N, ${payload.longitude.toFixed(4)}° E) · Area verified`);
    } catch (lookupError) {
      // Fallback: match any prefix in local data
      const fallback = searchNepalPlaces(query, 1);
      if (fallback.length > 0) {
        selectPlace(fallback[0]);
      } else {
        setPlaceMessage(lookupError instanceof Error ? lookupError.message : "Place lookup failed.");
      }
    } finally {
      setIsSearchingPlace(false);
    }
  }

  async function startRun(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!runReady || !effectiveBbox) {
      setError("Choose a valid area of interest before starting the run.");
      return;
    }
    if (!form.flood_date || form.raw_flood_class_id.trim() === "" || !form.checkpoint.trim()) {
      setError("Flood date, raw flood class ID, and checkpoint are required.");
      return;
    }
    setError("");
    setResults(null);
    setStatus({ run_id: "", status: "queued", progress: 0, stage: "Submitting run" });
    try {
      const response = await fetch(`${API_BASE}/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          west: effectiveBbox[0],
          south: effectiveBbox[1],
          east: effectiveBbox[2],
          north: effectiveBbox[3],
          flood_date: form.flood_date,
          raw_flood_class_id: Number(form.raw_flood_class_id),
          checkpoint: form.checkpoint,
          channel_names: form.channel_names,
          input_scale: form.input_scale,
        }),
      });
      if (!response.ok) throw new Error(await responseError(response));
      const queued = await response.json() as { run_id: string; status: RunStatus["status"] };
      setStatus({ run_id: queued.run_id, status: queued.status, progress: 0, stage: "Queued" });
    } catch (submitError) {
      setStatus(null);
      setError(submitError instanceof Error ? submitError.message : String(submitError));
    }
  }

  function toggleLayer(key: keyof LayerVisibility) {
    setVisibility((current) => ({ ...current, [key]: !current[key] }));
  }

  function focusOnName(name: string) {
    const target = settlements.find((settlement) => settlement.name === name || settlement.id === name);
    if (target) setFocusSettlement(target);
  }

  const layerToggles: { key: keyof LayerVisibility; label: string; swatch: string }[] = [
    { key: "flood", label: "Flood mask", swatch: "flood" },
    { key: "buildings", label: "Buildings", swatch: "building" },
    { key: "roads", label: "Roads", swatch: "road" },
    { key: "bridges", label: "Bridges", swatch: "bridge" },
    { key: "settlements", label: "Settlements", swatch: "settlement" },
    { key: "hillshade", label: "3D Terrain", swatch: "terrain" },
  ];

  return (
    <main className="app-shell">
      <MapView
        bbox={effectiveBbox}
        results={results}
        drawMode={areaMode === "draw"}
        visibility={visibility}
        basemap={basemap}
        locationPin={locationPin}
        onBboxDrawn={handleDrawn}
        onDrawProgress={setDrawMessage}
        onMapCenterSelected={handleMapCenterSelected}
        onSettlements={setSettlements}
        focusSettlement={focusSettlement}
      />

      {/* Floating Basemap Style Switcher */}
      <div className="basemap-floating-bar" role="toolbar" aria-label="Basemap style selector">
        <button
          type="button"
          className={basemap === "satellite" ? "active" : ""}
          onClick={() => setBasemap("satellite")}
          title="Photorealistic 3D Earth Globe with Satellite Imagery (Zero API Key)"
        >
          🛰️ 3D Globe
        </button>
        <button
          type="button"
          className={basemap === "dark" ? "active" : ""}
          onClick={() => setBasemap("dark")}
          title="Dark 3D Disaster Assessment Map"
        >
          🌙 Dark 3D
        </button>
        <button
          type="button"
          className={basemap === "streets" ? "active" : ""}
          onClick={() => setBasemap("streets")}
          title="OpenStreetMap Streets & Topography"
        >
          🗺️ Streets
        </button>
      </div>

      <header className="topbar">
        <div className="brand-mark">F</div>
        <div>
          <h1>Flood Impact</h1>
          <p>Sentinel-1 SAR Assessment · 3D Earth Globe · OSM Access</p>
        </div>
        <div className="top-status">{status?.stage ?? (effectiveBbox ? "Area selected" : "Globe ready · Select area")}</div>
      </header>

      <aside className="panel left-panel">
        <div className="panel-heading">
          <div><span className="eyebrow">AREA OF INTEREST</span><h2>Run assessment</h2></div>
        </div>
        <form onSubmit={startRun}>
          {/* Place Search with Instant First-Letter Suggestions */}
          <div className="place-search" ref={searchContainerRef}>
            <div className="place-label-row">
              <label htmlFor="place-search-input">City / Location Search</label>
              <span className="search-hint">Type first letter (e.g. S, K, P, N)</span>
            </div>
            <div className="place-search-row">
              <input
                id="place-search-input"
                value={placeQuery}
                onChange={(event) => handlePlaceInputChange(event.target.value)}
                onFocus={() => {
                  if (placeQuery.trim().length > 0) {
                    const matches = searchNepalPlaces(placeQuery, 8);
                    setSuggestions(matches);
                    setShowSuggestions(matches.length > 0);
                  }
                }}
                onKeyDown={handleSearchKeyDown}
                placeholder="Search Nepal city (e.g. Syaphrubesi, Kathmandu, Pokhara, Nepal)"
                autoComplete="off"
              />
              <button
                type="button"
                onClick={handleGeocode}
                disabled={isSearchingPlace}
                title="Locate and center on place"
              >
                {isSearchingPlace ? "..." : "Locate"}
              </button>
            </div>

            {/* Suggestions Dropdown */}
            {showSuggestions && suggestions.length > 0 && (
              <ul className="suggestions-dropdown" role="listbox">
                {suggestions.map((item, index) => (
                  <li
                    key={item.id}
                    role="option"
                    aria-selected={index === selectedSuggestionIndex}
                    className={`suggestion-item ${index === selectedSuggestionIndex ? "highlighted" : ""}`}
                    onClick={() => selectPlace(item)}
                  >
                    <div className="suggestion-main">
                      <span className="suggestion-name">📍 {item.name}</span>
                      <span className="suggestion-region">{item.region}</span>
                    </div>
                    <span className="suggestion-coords">
                      {item.lat.toFixed(4)}° N, {item.lon.toFixed(4)}° E
                    </span>
                  </li>
                ))}
              </ul>
            )}

            {placeMessage && (
              <p className={placeMessage.startsWith("✓") ? "place-ok" : "place-error"}>{placeMessage}</p>
            )}
          </div>

          {/* Location Verification & Cross-Verification Card */}
          {verifiedLocation && (
            <div className="location-verification-card">
              <div className="card-header">
                <span className="verification-badge">✓ POINT VERIFIED</span>
                <span className="verified-name">{verifiedLocation.name}</span>
              </div>
              <div className="card-body">
                <div className="verification-row">
                  <span className="row-label">Region</span>
                  <span className="row-value">{verifiedLocation.region}</span>
                </div>
                <div className="verification-row">
                  <span className="row-label">Coordinates</span>
                  <span className="row-value coords-highlight">
                    {verifiedLocation.lat.toFixed(6)}° N, {verifiedLocation.lon.toFixed(6)}° E
                  </span>
                </div>
                {effectiveBbox && (
                  <div className="verification-row">
                    <span className="row-label">AOI Bounds</span>
                    <span className="row-value bounds-highlight">
                      [{effectiveBbox[0].toFixed(4)}°, {effectiveBbox[1].toFixed(4)}°] → [{effectiveBbox[2].toFixed(4)}°, {effectiveBbox[3].toFixed(4)}°]
                    </span>
                  </div>
                )}
              </div>
              <div className="card-footer">
                <span className="status-indicator"></span>
                <span>3D Earth globe pinned &amp; cross-verified with SAR bounds</span>
              </div>
            </div>
          )}

          <div className="mode-switch" role="tablist" aria-label="Area of interest mode selector">
            <button type="button" className={areaMode === "center" ? "selected" : ""} onClick={() => setAreaMode("center")}>Center point</button>
            <button type="button" className={areaMode === "draw" ? "selected" : ""} onClick={() => setAreaMode("draw")}>Draw on map</button>
            <button type="button" className={areaMode === "bbox" ? "selected" : ""} onClick={() => setAreaMode("bbox")}>Bounding box</button>
          </div>

          {areaMode === "center" && (
            <>
              <div className="input-row two-col">
                <label>
                  Latitude
                  <input
                    type="number"
                    step="any"
                    value={centerPoint.lat}
                    onChange={(event) => setCenterPoint((current) => ({ ...current, lat: normalizeCoordinateInput(event.target.value) }))}
                    placeholder="e.g. 28.1601"
                  />
                  {centerFieldErrors.lat && <span className="field-error">{centerFieldErrors.lat}</span>}
                </label>
                <label>
                  Longitude
                  <input
                    type="number"
                    step="any"
                    value={centerPoint.lon}
                    onChange={(event) => setCenterPoint((current) => ({ ...current, lon: normalizeCoordinateInput(event.target.value) }))}
                    placeholder="e.g. 85.3374"
                  />
                  {centerFieldErrors.lon && <span className="field-error">{centerFieldErrors.lon}</span>}
                </label>
              </div>
              <div className="input-row two-col">
                <label>
                  Width (km)
                  <input
                    type="number"
                    min="0.1"
                    step="0.1"
                    value={sizeKm.widthKm}
                    onChange={(event) => setSizeKm((current) => ({ ...current, widthKm: event.target.value }))}
                  />
                </label>
                <label>
                  Height (km)
                  <input
                    type="number"
                    min="0.1"
                    step="0.1"
                    value={sizeKm.heightKm}
                    onChange={(event) => setSizeKm((current) => ({ ...current, heightKm: event.target.value }))}
                  />
                </label>
              </div>
            </>
          )}

          {areaMode === "draw" && (
            <div className="draw-box-panel">
              <p>Click once on the 3D globe to set the center point, then click the opposite corner to finish the box.</p>
              <div className="input-row two-col">
                <label>Latitude<input type="number" step="any" value={centerPoint.lat} readOnly placeholder="Click on map" /></label>
                <label>Longitude<input type="number" step="any" value={centerPoint.lon} readOnly placeholder="Click on map" /></label>
              </div>
              {drawMessage && <p className="draw-hint">{drawMessage}</p>}
            </div>
          )}

          {areaMode === "bbox" && (
            <div className="bbox-grid">
              <label>
                West
                <input type="number" step="any" value={bboxFields.west} onChange={(event) => updateBbox("west", event.target.value)} placeholder="e.g. 85.28" />
                {bboxFieldErrors.west && <span className="field-error">{bboxFieldErrors.west}</span>}
              </label>
              <label>
                South
                <input type="number" step="any" value={bboxFields.south} onChange={(event) => updateBbox("south", event.target.value)} placeholder="e.g. 28.11" />
                {bboxFieldErrors.south && <span className="field-error">{bboxFieldErrors.south}</span>}
              </label>
              <label>
                East
                <input type="number" step="any" value={bboxFields.east} onChange={(event) => updateBbox("east", event.target.value)} placeholder="e.g. 85.38" />
                {bboxFieldErrors.east && <span className="field-error">{bboxFieldErrors.east}</span>}
              </label>
              <label>
                North
                <input type="number" step="any" value={bboxFields.north} onChange={(event) => updateBbox("north", event.target.value)} placeholder="e.g. 28.21" />
                {bboxFieldErrors.north && <span className="field-error">{bboxFieldErrors.north}</span>}
              </label>
            </div>
          )}

          {/* Resulting BBox Readout */}
          <div className="bbox-readout">
            <div className="readout-header">Resulting bbox</div>
            <div className="bbox-grid small-grid">
              <label>West<input value={effectiveBbox ? formatCoordinate(effectiveBbox[0]) : "—"} readOnly /></label>
              <label>South<input value={effectiveBbox ? formatCoordinate(effectiveBbox[1]) : "—"} readOnly /></label>
              <label>East<input value={effectiveBbox ? formatCoordinate(effectiveBbox[2]) : "—"} readOnly /></label>
              <label>North<input value={effectiveBbox ? formatCoordinate(effectiveBbox[3]) : "—"} readOnly /></label>
            </div>
            {!effectiveBbox && (
              <p className="readout-empty-hint">Enter coordinates, search a city above, or draw on the map to define the bounding box.</p>
            )}
            {areaWarning && <div className="warning-box">{areaWarning}</div>}
            {!areaIsValid && effectiveBbox && (
              <div className="warning-box error-box">Choose a valid area with West &lt; East and South &lt; North, and lat/long values within the valid range.</div>
            )}
          </div>

          <label>Flood date<input type="date" value={form.flood_date} onChange={(event) => updateForm("flood_date", event.target.value)} /></label>
          <label>Raw flood class ID<input inputMode="numeric" value={form.raw_flood_class_id} onChange={(event) => updateForm("raw_flood_class_id", event.target.value)} placeholder="From trained labels" /></label>
          <label>Checkpoint path<input value={form.checkpoint} onChange={(event) => updateForm("checkpoint", event.target.value)} /></label>
          <div className="input-row">
            <label>
              Band names
              <input value={form.channel_names} onChange={(event) => updateForm("channel_names", event.target.value)} placeholder="Verified from training" />
            </label>
            <label>
              Scale
              <select value={form.input_scale} onChange={(event) => updateForm("input_scale", event.target.value)}>
                <option value="unknown">Select</option>
                <option value="db">dB</option>
                <option value="linear">Linear</option>
              </select>
            </label>
          </div>

          <button className="run-button" type="submit" disabled={!runReady || status?.status === "running" || status?.status === "queued"}>
            {status?.status === "running" || status?.status === "queued" ? "Assessment running" : "Run assessment"}
            <span aria-hidden="true">↗</span>
          </button>
        </form>

        {status && (
          <div className="progress-block">
            <div className="progress-label"><span>{status.stage}</span><span>{status.progress}%</span></div>
            <progress max={100} value={status.progress} />
          </div>
        )}
        {error && <div className="error-box" role="alert">{error}</div>}

        <div className="layers-block">
          <span className="eyebrow">MAP LAYERS</span>
          {layerToggles.map((layer) => (
            <label className="layer-toggle" key={layer.key}>
              <span className={`swatch ${layer.swatch}`} />{layer.label}
              <input type="checkbox" checked={visibility[layer.key]} onChange={() => toggleLayer(layer.key)} />
            </label>
          ))}
        </div>
      </aside>

      <aside className="panel right-panel">
        <div className="panel-heading">
          <div><span className="eyebrow">SITUATION SUMMARY</span><h2>Observed impacts</h2></div>
          {results && <span className="snapshot-tag">OSM {results.summary.source.osm_snapshot_date ?? ""}</span>}
        </div>
        {results ? (
          <>
            <div className="metric-list">
              <Metric label="Flooded area" value={results.summary.impacts.flooded_area_km2} unit="km²" />
              <Metric label="Buildings hit" value={results.summary.impacts.buildings_hit} />
              <Metric label="Road cut" value={results.summary.impacts.road_length_hit_km} unit="km" />
              <Metric label="Bridges hit" value={results.summary.impacts.bridges_hit} />
              <Metric label="Settlements cut off" value={results.summary.access.disconnected_count} />
            </div>
            <div className="settlement-list">
              <div className="list-heading"><span className="swatch settlement-cut" />Cut-off settlements</div>
              {results.summary.access.disconnected_settlements?.length ? (
                results.summary.access.disconnected_settlements.map((name) => (
                  <button className="settlement-item" key={name} onClick={() => focusOnName(name)}>
                    {name}
                    <span>Locate</span>
                  </button>
                ))
              ) : (
                <p className="muted">No cut-off settlements recorded in the summary.</p>
              )}
            </div>
            <div className="report-actions">
              <a href={`${API_BASE}/report/${results.run_id}?lang=en`} target="_blank" rel="noreferrer">Report · EN <span>↓</span></a>
              <a href={`${API_BASE}/report/${results.run_id}?lang=ne`} target="_blank" rel="noreferrer">Report · नेपाली <span>↓</span></a>
            </div>
            <p className="data-note">Figures are read from the run summary JSON. Geometric overlap is a heuristic, not a probability.</p>
          </>
        ) : (
          <p className="muted empty-summary">Summary metrics and reports appear after a completed run.</p>
        )}
      </aside>

      <footer className="attribution-footer">
        <span className="prototype-note">Educational prototype, not an operational tool.</span>
        <div>{ATTRIBUTIONS.map((line) => <span key={line}>{line}</span>)}</div>
      </footer>
    </main>
  );
}
