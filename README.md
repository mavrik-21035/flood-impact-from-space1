# Flood Impact Map

MapLibre 3D frontend and FastAPI wrapper for the existing Sentinel-1 flood pipeline, OSM impact analysis, and bilingual PDF report. The wrapper does not alter `pipeline.py`, `analysis.py`, or `report.py`.

## 🐳 Quick Start with Docker (Recommended)

> **Only requirement:** [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running.

```bash
# 1. Clone the repository
git clone <repo-url> && cd HAKATHON

# 2. (Optional) Set Copernicus credentials for real satellite data.
#    Skip this to use the built-in simulation engine.
cp .env.example .env
# Edit .env with your CDSE_USERNAME and CDSE_PASSWORD

# 3. Build and start everything with a single command
docker compose up --build

# 4. Open in your browser
#    → http://localhost:3000
```

**That's it!** The Docker setup bundles Python, Node.js, PyTorch, GDAL, ESA SNAP GPT, and all other dependencies — no manual installs needed.

To stop the containers:

```bash
docker compose down
```

---

## Prerequisites

- Python with the packages listed in `backend/requirements.txt`
- Node.js and npm
- ESA SNAP GPT installed and available on `PATH`, with Copernicus DEM support enabled
- CDSE credentials in the backend process environment: `CDSE_USERNAME` and `CDSE_PASSWORD`
- A trained checkpoint and verified training band/scale metadata

The trainer saves the input channel names and input scale in the checkpoint. It passes MS1 raster pixel values through unchanged; the default channel order is `VV,VH`, and the default scale metadata is `linear`. If the source rasters already store dB values, pass `--input-scale db`. Verify the raster encoding before training because scale conversion is not performed by the loader. The inference pipeline reads the checkpoint metadata and supports both linear and dB inputs. When `checkpoints/best_unet.pt` is present, the dashboard reads the checkpoint metadata and pre-fills the raw flood class ID, band names, and scale.

## Download and train Kuro Siwo

The dataset is not included in this repository. The upstream KuroSiwo repository provides `download_kuro_siwo.sh`; on Windows, run it from Git Bash or WSL:

```bash
git clone https://github.com/Orion-AI-Lab/KuroSiwo.git
cd KuroSiwo
bash download_kuro_siwo.sh "$HOME/Downloads"
```

The script creates a `KuroSiwo` directory under the supplied root and downloads/extracts the dataset there. Once downloaded, find the dataset root that contains sample folders with `MS1*.tif`/`.tiff` and `MK0_MLU*.tif`/`.tiff` files. `MK0_MNA` validity rasters are optional for this trainer. The label convention is `0 = no water`, `1 = permanent water`, `2 = flood`, and `3 = invalid pixels`; invalid pixels are excluded by the validity mask when present.

From the project root in PowerShell, run a short smoke training pass (replace the placeholder with the verified dataset root):

```powershell
& .\.venv\Scripts\python.exe .\train.py "<dataset-root>" --flood-class 2 --epochs 1 --max-batches 5
```

For full training, omit `--max-batches` and increase `--epochs`, for example:

```powershell
& .\.venv\Scripts\python.exe .\train.py "<dataset-root>" --flood-class 2 --epochs 20
```

Training time depends on dataset size, batch size, CPU/GPU, and available memory; CPU training can take substantially longer than a CUDA-capable GPU.

## Run the backend

From the workspace root in PowerShell:

```powershell
python -m pip install -r backend/requirements.txt
$env:CDSE_USERNAME = "your-CDSE-username"
$env:CDSE_PASSWORD = "your-CDSE-password"
python -m uvicorn backend.api:app --host 127.0.0.1 --port 8000
```

Keep the backend terminal open while using the app. API docs are at `http://127.0.0.1:8000/docs`.

## Run the frontend

In a second terminal from the workspace root:

```powershell
cd frontend
npm install
npm run dev
```

Open the Vite URL printed by the command, normally `http://localhost:5173`.

## Build the frontend

```powershell
cd frontend
npm run build
```

## Outputs

Runs are written under `runs/<run-id>/`. The API status is held in memory, so keep the backend process running for the lifetime of a run and to serve its layers, DEM tiles, and report downloads. Terrain-RGB tiles are generated on demand from the SNAP terrain-corrected Copernicus DEM.