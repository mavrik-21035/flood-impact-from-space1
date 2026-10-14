import React from "react";
import ReactDOM from "react-dom/client";
import * as maplibregl from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import App from "./App";
import "./styles.css";

maplibregl.setWorkerUrl(
  /* @vite-ignore */ new URL("maplibre-gl/dist/maplibre-gl-worker.mjs", import.meta.url).toString(),
);

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);