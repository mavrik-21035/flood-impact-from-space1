import type { FeatureCollection, Geometry } from "geojson";

export type BBox = [west: number, south: number, east: number, north: number];

export type ImpactSummary = {
  flooded_area_km2?: number;
  buildings_hit?: number;
  road_length_hit_km?: number;
  bridges_hit?: number;
};

export type AccessSummary = {
  disconnected_count?: number;
  connected_count?: number;
  no_baseline_path_count?: number;
  disconnected_settlements?: string[];
};

export type RunSummary = {
  source: {
    bbox_wgs84?: BBox;
    osm_snapshot_date?: string;
    flood_class_value?: number;
  };
  event_date?: string;
  impacts: ImpactSummary;
  access: AccessSummary;
};

export type RunResults = {
  run_id: string;
  summary: RunSummary;
  urls: {
    flood_mask: string;
    damaged_buildings: string;
    roads: string;
    bridges: string;
    settlements: string;
    dem_tiles: string;
    report: string;
  };
};

export type RunStatus = {
  run_id: string;
  status: "queued" | "running" | "completed" | "failed";
  progress: number;
  stage: string;
  error?: string | null;
};

export type SettlementPoint = {
  id: string;
  name: string;
  coordinates: [number, number];
};

export type FeatureCollectionData = FeatureCollection<Geometry, Record<string, unknown>>;

export type LayerVisibility = {
  flood: boolean;
  buildings: boolean;
  roads: boolean;
  bridges: boolean;
  settlements: boolean;
  hillshade: boolean;
};

export type LocationPin = {
  lat: number;
  lon: number;
  name: string;
  region?: string;
  zoom?: number;
};

export type BasemapMode = "satellite" | "dark" | "streets";