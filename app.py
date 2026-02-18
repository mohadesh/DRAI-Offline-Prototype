"""
Flask Server - Routing and Webhook Handling.
Logic for processing is delegated to 'core_logic.py'.
Simulation state is managed by 'simulation.py'.

OPTIMIZED VERSION: Includes Global Model Caching to prevent disk I/O bottlenecks.
"""

import os
import uuid
import logging
from pathlib import Path
from typing import Optional, Tuple, Any

from flask import Flask, render_template, request, jsonify, session
import pandas as pd

# Internal modules
import core_logic
import simulation

# --- CONFIGURATION ---
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB Limit
app.config["SECRET_KEY"] = "drai-offline-secret-key-change-in-production"

# --- LOGGING SETUP ---
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# --- PATHS ---
BASE_DIR = Path(__file__).resolve().parent
UPLOADS_DIR = BASE_DIR / "uploads"
MODELS_DIR = BASE_DIR / "models"
ANALYSIS_DIR = BASE_DIR / "analysis"
DRAI_MODELING_DIR = BASE_DIR.parent / "DRAI-Modeling" / "data" / "analysis"

# Ensure directories exist
UPLOADS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# --- MODEL SETTINGS ---
FREQUENCY_OPTIONS = ("1h", "15T", "30T")
DEFAULT_MODEL_FREQUENCY = "30T"

FREQUENCY_PATTERNS = {
    "1h": "darts_pipeline_freq_1h_*",
    "15T": "darts_pipeline_freq_15T_*",
    "30T": "darts_pipeline_freq_30T_*",
}

# --- GLOBAL CACHE ---
# Stores loaded model objects in memory to avoid repeated disk reads.
# Key: str(file_path), Value: Model Object
MODEL_CACHE = {}

# Stores SimulationRunner instances per session
# Key: session_id, Value: SimulationRunner instance
_simulation_runners = {}


# --- HELPER FUNCTIONS ---

def _find_pipeline_base(frequency: str) -> Optional[Path]:
    """Finds the directory containing the Darts pipeline for a given frequency."""
    pattern = FREQUENCY_PATTERNS.get(frequency)
    if not pattern:
        return None
    
    # Search priorities: 1. Local 'analysis' folder, 2. Sibling 'DRAI-Modeling' folder
    for base in (ANALYSIS_DIR, DRAI_MODELING_DIR):
        if not base.exists():
            continue
        matches = list(base.glob(pattern))
        if matches:
            # Sort to get the latest folder if multiple exist (alphabetical order usually works for dates)
            matches.sort()
            return matches[-1]
    return None


def get_model_paths_for_frequency(frequency: str) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Resolves file paths for MD and C models based on frequency.
    """
    model_base = _find_pipeline_base(frequency)
    
    # 1. Try loading from Pipeline structure
    if model_base:
        md_path = model_base / "MDNC_M_D" / "model_MDNC_M_D.pkl"
        c_path = model_base / "MDNC_C" / "model_MDNC_C.pkl"
        return (md_path if md_path.exists() else None, c_path if c_path.exists() else None)
    
    # 2. Fallback: Legacy structure in 'models/' folder (Only for 30T)
    if frequency == "30T":
        md = MODELS_DIR / "model_MDNC_M_D.pkl"
        c = MODELS_DIR / "model_MDNC_C.pkl"
        return (md if md.exists() else None, c if c.exists() else None)
        
    return (None, None)


def get_cached_model(path: Optional[Path]) -> Any:
    """
    Efficiently retrieves a model.
    1. Checks if model is already in RAM (MODEL_CACHE).
    2. If not, loads from Disk and saves to Cache.
    """
    if not path:
        return None
    
    path_str = str(path)
    
    # Check Cache
    if path_str in MODEL_CACHE:
        return MODEL_CACHE[path_str]
    
    # Load from Disk
    try:
        logger.info(f"⏳ Loading model into memory (First time): {path_str}")
        model = core_logic.load_model(path_str)
        if model is not None:
            MODEL_CACHE[path_str] = model
            logger.info("✅ Model cached successfully.")
            return model
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        
    return None


def _log_model_locations():
    """Debug utility to print search paths."""
    print("="*80)
    print("[Models] Search locations:")
    print(f"  1) {ANALYSIS_DIR} (Local project)")
    print(f"  2) {DRAI_MODELING_DIR} (Sibling project)")
    print(f"  3) {MODELS_DIR} (Legacy fallback for 30T)")
    print("-" * 40)
    for freq in FREQUENCY_OPTIONS:
        md, c = get_model_paths_for_frequency(freq)
        status = "OK" if (md and c) else "MISSING"
        print(f"  [{freq}] {status}")
        if md: print(f"      MD: {md}")
        if c:  print(f"      C : {c}")
    print("="*80)


# --- ROUTES ---

@app.route("/")
def index():
    """Renders the main dashboard."""
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    """
    Handles batch file uploads (Process, Pellet, MD).
    Processes data and initializes the Simulation Runner.
    """
    # 1. Retrieve Files
    process_files = request.files.getlist("process_files")
    pellet_files = request.files.getlist("pellet_files")
    md_files = request.files.getlist("md_files")
    
    # 2. Filter Valid Files
    def is_valid(f):
        return f and f.filename and f.filename.lower().endswith(('.csv', '.xlsx'))
        
    process_files = [f for f in process_files if is_valid(f)]
    pellet_files = [f for f in pellet_files if is_valid(f)]
    md_files = [f for f in md_files if is_valid(f)]
    
    # 3. Validation
    if not (process_files and pellet_files and md_files):
        return jsonify({"success": False, "error": "Please upload at least one file for each category."}), 400

    # 4. Session & Frequency Setup
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    session_id = session['session_id']
    
    model_freq = request.form.get("model_frequency", DEFAULT_MODEL_FREQUENCY)
    if model_freq not in FREQUENCY_OPTIONS:
        model_freq = DEFAULT_MODEL_FREQUENCY
    session["model_frequency"] = model_freq
    
    # 5. Resolve Model Paths (for reference)
    model_md_path, model_c_path = get_model_paths_for_frequency(model_freq)
    
    try:
        logger.info(f"Processing upload for session {session_id} @ {model_freq}")
        
        # 6. Core Processing (Merging & Cleaning)
        result = core_logic.process_data(
            process_files=process_files,
            pellet_files=pellet_files,
            md_files=md_files,
            model_md_path=None, # We load models lazily/cached later
            model_c_path=None
        )
        
        # 7. Initialize Simulation
        runner = simulation.SimulationRunner(result['merged_df'])
        _simulation_runners[session_id] = runner
        
        # 8. Pre-load models into cache (Optional optimization)
        # This ensures the first simulation step is fast
        if model_md_path: get_cached_model(model_md_path)
        if model_c_path: get_cached_model(model_c_path)
        
        return jsonify({
            "success": True,
            "message": f"Processed {result['stats']['rows']:,} rows successfully.",
            "stats": result['stats']
        })
        
    except Exception as e:
        logger.exception("Error during processing")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/simulation/start", methods=["POST"])
def simulation_start():
    session_id = session.get('session_id')
    runner = _simulation_runners.get(session_id)
    if not runner:
        return jsonify({"success": False, "error": "No data loaded."}), 400
    runner.start()
    return jsonify({"success": True, "message": "Simulation started"})


@app.route("/simulation/pause", methods=["POST"])
def simulation_pause():
    session_id = session.get('session_id')
    runner = _simulation_runners.get(session_id)
    if not runner: return jsonify({"success": False, "error": "No simulation found."}), 400
    runner.pause()
    return jsonify({"success": True, "message": "Simulation paused"})


@app.route("/simulation/resume", methods=["POST"])
def simulation_resume():
    session_id = session.get('session_id')
    runner = _simulation_runners.get(session_id)
    if not runner: return jsonify({"success": False, "error": "No simulation found."}), 400
    runner.resume()
    return jsonify({"success": True, "message": "Simulation resumed"})


@app.route("/simulation/reset", methods=["POST"])
def simulation_reset():
    session_id = session.get('session_id')
    runner = _simulation_runners.get(session_id)
    if not runner: return jsonify({"success": False, "error": "No simulation found."}), 400
    runner.reset()
    return jsonify({"success": True, "message": "Simulation reset"})


@app.route("/update-dashboard", methods=["GET"])
def update_dashboard():
    """
    High-frequency polling endpoint (HTMX).
    Fetches the current simulation step and runs inference.
    """
    if 'session_id' not in session:
        return render_template("partials/dashboard.html", error="Session expired")
        
    session_id = session['session_id']
    runner = _simulation_runners.get(session_id)
    
    if not runner:
        return render_template("partials/dashboard.html", error="No data loaded")

    # 1. Get Simulation State
    current_data = runner.get_current_step()
    progress = runner.get_progress()
    
    predictions = None
    
    # 2. Run Inference (Only if simulation is active/has data)
    if current_data:
        try:
            # Prepare single-row DataFrame
            df_row = pd.DataFrame([current_data])
            
            # Identify models
            freq = session.get("model_frequency", DEFAULT_MODEL_FREQUENCY)
            md_path, c_path = get_model_paths_for_frequency(freq)
            
            # Retrieve from CACHE (Fast!)
            model_md = get_cached_model(md_path)
            model_c = get_cached_model(c_path)
            
            if model_md or model_c:
                # Perform inference
                predictions = core_logic.run_inference_for_md_c(model_md, model_c, df_row)
        except Exception as e:
            logger.error(f"Inference error: {e}")
            # Do not crash the dashboard, just show empty predictions
            pass

    # 3. Advance Simulation (if running)
    if runner.is_running and not runner.is_paused:
        runner.get_next_step()

    return render_template(
        "partials/dashboard.html",
        current_data=current_data,
        predictions=predictions,
        progress=progress,
        error=None
    )


@app.route("/analyze", methods=["POST"])
def analyze():
    """Legacy redirect."""
    return upload()


def _find_free_port(start_port: int, max_tries: int = 10) -> int:
    import socket
    for i in range(max_tries):
        port = start_port + i
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", port))
                return port
        except OSError:
            continue
    raise OSError(f"No free port available starting from {start_port}")


if __name__ == "__main__":
    # Disable Werkzeug logs in production-like run to keep terminal clean
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    
    # Port Setup
    base_port = int(os.environ.get("PORT", "8000"))
    port = _find_free_port(base_port)
    
    print("\n" + "="*80)
    print("DRAI OFFLINE INFERENCE APP (OPTIMIZED)")
    print("="*80)
    print(f"Server URL: http://127.0.0.1:{port}")
    _log_model_locations()
    
    app.run(debug=True, host="0.0.0.0", port=port, use_reloader=False)
