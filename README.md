# DRAI Offline Inference Application

A professional, offline-capable web application for processing, merging, and analyzing DRI (Direct Reduced Iron) furnace data with real-time simulation and ML model inference.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [ML Models and Frequencies](#ml-models-and-frequencies)
- [Data Pipeline](#data-pipeline)
- [API Endpoints](#api-endpoints)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Development](#development)

---

## Overview

This application is designed for **air-gapped environments** where internet connectivity is not available. It provides a complete offline solution for:

- **Batch processing** of multiple CSV/Excel files (Process Tags, Pellet Data, MD/Quality Data)
- **Automatic data merging** with intelligent encoding and delimiter detection
- **Jalali to Gregorian date conversion** for Persian calendar support
- **Real-time simulation** of furnace operations
- **ML model inference** for predicting MD (Metallization Degree) and C (Carbon) values
- **Residence time calculations** based on production rates (WTH15)

**Use cases:** industrial data analysis in secure/isolated environments, offline monitoring and prediction, batch processing of historical furnace data, real-time simulation, and quality prediction.

---

## Features

### Core Capabilities

| Feature | Description |
|--------|-------------|
| **100% Offline** | No CDN dependencies; all assets bundled locally |
| **Batch Upload** | Process 50+ files per category in one run |
| **Robust File Handling** | Auto encoding (UTF-8, CP1256, Latin1, etc.) and delimiter detection |
| **Multi-format** | CSV and Excel (.xlsx) |
| **Jalali Support** | Automatic Persian-to-Gregorian date conversion |
| **Residence Time** | Physics-based time synchronization |
| **Simulation** | Step-through with configurable intervals |
| **ML Inference** | Predict MD and C using pre-trained models (1h, 15T, 30T) |
| **Status Validation** | Shutdown/startup detection (e.g. WTH15 &lt; 80) |

### User Interface

- Responsive dashboard (Tailwind CSS)
- Real-time updates via HTMX
- Persian (RTL) UI support
- File upload with progress; simulation controls (Start, Pause, Resume, Reset)

---

## Technology Stack

| Layer | Technologies |
|-------|--------------|
| **Backend** | Flask 3.0+, Pandas 2.0+, NumPy, Joblib, LightGBM, Scikit-learn |
| **Frontend** | HTMX, Alpine.js, Tailwind CSS (standalone, no CDN) |
| **Data** | jdatetime, persiantools, openpyxl |

Architecture: pure Python core logic, session-based in-memory state, modular design.

---

## Project Structure

```
DRAI-Offline-Prototype/
├── app.py                  # Flask app, routes, session, model discovery
├── core_logic.py           # Data processing, merging, inference
├── simulation.py           # Simulation engine
├── setup_assets.py         # Download HTMX, Alpine, Tailwind to static/vendor
├── requirements.txt
├── README.md
│
├── analysis/               # (Optional) Pipeline outputs; add to .gitignore
│   ├── darts_pipeline_freq_1h_*/
│   │   ├── MDNC_M_D/model_MDNC_M_D.pkl
│   │   └── MDNC_C/model_MDNC_C.pkl
│   ├── darts_pipeline_freq_15T_*/
│   └── darts_pipeline_freq_30T_*/
│
├── models/                 # Optional fallback for 30T models
│   ├── model_MDNC_M_D.pkl
│   └── model_MDNC_C.pkl
│
├── static/vendor/          # Offline JS/CSS (from setup_assets.py)
│   ├── htmx.min.js
│   ├── alpine.min.js
│   └── tailwind.js
│
├── templates/
│   ├── index.html
│   └── partials/
│       ├── dashboard.html
│       └── results.html
│
└── uploads/                 # Temporary uploads
```

---

## Installation

### Prerequisites

- Python 3.8+
- pip

### Steps

```bash
# 1. Go to project root
cd /path/to/DRAI-Offline-Prototype

# 2. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download offline frontend assets (run once with network)
python setup_assets.py
```

### Optional: ML models

- **Preferred:** Place pipeline outputs in `analysis/` (see [ML Models and Frequencies](#ml-models-and-frequencies)). The app discovers models by frequency (1h, 15T, 30T).
- **Fallback for 30T only:** Put `model_MDNC_M_D.pkl` and `model_MDNC_C.pkl` in `models/`.
- **Alternative:** Use a sibling project at `../DRAI-Modeling/data/analysis/` with the same pipeline folder structure; the app will look there if `analysis/` is missing.

---

## Usage

### Start the server

```bash
python app.py
```

- **Local:** http://127.0.0.1:8000  
- **Network:** http://0.0.0.0:8000

### Workflow

1. **Open** the app in a browser.
2. **Select model frequency** (Prediction interval): **30 min (30T)**, **15 min (15T)**, or **1 hour (1h)**. This chooses which pair of MD/C models is used.
3. **Upload files** for each category (Process Tags, Pellet Data, MD/Quality Data), then click **Upload and Process**.
4. **Start simulation**; the dashboard updates periodically with current data and MD/C predictions.
5. **Control:** Pause, Resume, or Reset as needed.

---

## ML Models and Frequencies

The app uses **two models per run:** one for **MD** (Metallization Degree) and one for **C** (Carbon). Models are grouped by **time frequency**:

| Frequency | Folder pattern | Description |
|----------|----------------|-------------|
| **30T**  | `darts_pipeline_freq_30T_*` | 30-minute interval |
| **15T**  | `darts_pipeline_freq_15T_*` | 15-minute interval |
| **1h**   | `darts_pipeline_freq_1h_*`  | 1-hour interval |

Each pipeline folder must contain:

- `MDNC_M_D/model_MDNC_M_D.pkl`
- `MDNC_C/model_MDNC_C.pkl`

**Discovery order:**

1. `analysis/` inside the project (if present)
2. `../DRAI-Modeling/data/analysis/` (sibling project)
3. For **30T only:** fallback to `models/model_MDNC_M_D.pkl` and `models/model_MDNC_C.pkl`

The user selects the frequency in the upload form; that choice is stored in the session and used for both upload-time processing and dashboard inference.

---

## Data Pipeline

### Input file types

| Type | Format | Content |
|------|--------|---------|
| **Process Tags** | CSV/Excel | Sensor data (date/time, WTH15, etc.); often 5-min intervals |
| **Pellet Data** | CSV/Excel | Lab data for input pellets (date/time, %FeO, CCS, etc.) |
| **MD/Quality Data** | CSV/Excel | Output quality (date/time, M_D, C, etc.) |

### Processing steps

1. **Load:** try multiple encodings and delimiters; handle malformed lines.
2. **Dates:** detect date columns; convert Jalali to Gregorian; sort by time.
3. **Stack:** merge files per category; deduplicate by timestamp.
4. **Merge:** 5-minute reference grid; merge Process Tags, Pellet, and MD/Quality (e.g. `merge_asof` for Pellet and MD).
5. **Residence time:**  
   `enter_deltatime_hours = 600 / ((WTH15 + 270) / 2)`  
   `exit_deltatime_hours = 1200 / ((WTH15 + 270) / 2)`
6. **Validation:** e.g. mark "Shutdown" when WTH15 &lt; 80, "Normal" otherwise.

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Main dashboard page |
| POST | `/upload` | Upload and process files. Body: `multipart/form-data` with `process_files`, `pellet_files`, `md_files`, and optional `model_frequency` (`1h`, `15T`, `30T`). Returns JSON with success and stats. |
| POST | `/simulation/start` | Start simulation |
| POST | `/simulation/pause` | Pause simulation |
| POST | `/simulation/resume` | Resume simulation |
| POST | `/simulation/reset` | Reset simulation |
| GET | `/update-dashboard` | HTMX polling; returns HTML fragment with current state and predictions |

---

## Configuration

- **Upload size limit:** 500 MB (set in `app.py` via `MAX_CONTENT_LENGTH`).
- **Model paths:** See [ML Models and Frequencies](#ml-models-and-frequencies). No config file; discovery is automatic from `analysis/`, `DRAI-Modeling`, and `models/`.
- **Sessions:** In-memory. For production, consider Redis, a database, or file-based sessions.

---

## Troubleshooting

| Issue | Cause | Action |
|-------|--------|--------|
| "No valid data found" | Encoding/delimiter or bad file | Check CSV/Excel format and encoding; see server logs. |
| "Pellet file must have georgian_datetime" | Date column not detected | Ensure date/time columns; names may include `date`, `time`, `Year`, `Month`, `Day`, or `georgian_datetime`. |
| Encoding errors | Unrecognized encoding | App tries several encodings; verify file encoding if all fail. |
| Simulation not starting | No data or expired session | Re-upload files; check browser console and session. |
| No predictions | No models for selected frequency | Ensure `analysis/` (or fallback for 30T) contains the correct `.pkl` files for the chosen frequency. |

**Debug:** Watch terminal output for file loading, encoding, merge stats, and tracebacks.

---

## Development

- **app.py:** Routes, session, model discovery, upload/simulation endpoints.
- **core_logic.py:** File loading, merging, residence time, inference.
- **simulation.py:** Step-through simulation state.
- **templates:** Jinja2; use partials for HTMX fragments.

**Production:** Do not use the Flask dev server. Use e.g. **Gunicorn:**  
`gunicorn -w 4 -b 0.0.0.0:8000 app:app`  
and a reverse proxy (Nginx/Apache) as needed.

**Security:** Set a strong `SECRET_KEY`; validate file types and sizes; consider authentication.

---

## Related Projects

- **DRAI-Modeling:** data science pipeline and model training. Model outputs can be placed in `DRAI-Modeling/data/analysis/` with the same folder layout; this app will use them when `analysis/` is not used.

---

## License and Contributors

Part of the DRAI (Direct Reduced Iron) analysis system, for offline/air-gapped industrial data analysis.

**Last updated:** February 2026
