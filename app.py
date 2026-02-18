"""
سرور Flask — فقط مسئول روتینگ و وب‌هوک.
منطق پردازش در core_logic قرار دارد.
"""

import os
from pathlib import Path
from typing import Optional, Tuple
from flask import Flask, render_template, request, jsonify, session
import uuid

import core_logic
import simulation

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB (increased for batch uploads)
app.config["SECRET_KEY"] = "drai-offline-secret-key-change-in-production"  # For sessions

# Project paths
BASE_DIR = Path(__file__).resolve().parent
UPLOADS_DIR = BASE_DIR / "uploads"
MODELS_DIR = BASE_DIR / "models"
# Analysis locations: in-project analysis/ first, then sibling DRAI-Modeling
ANALYSIS_DIR = BASE_DIR / "analysis"
DRAI_MODELING_DIR = BASE_DIR.parent / "DRAI-Modeling" / "data" / "analysis"

# Supported model frequencies: 1h, 15T, 30T (each has MD and C models)
FREQUENCY_OPTIONS = ("1h", "15T", "30T")
# Glob patterns for pipeline folders per frequency
FREQUENCY_PATTERNS = {
    "1h": "darts_pipeline_freq_1h_*",
    "15T": "darts_pipeline_freq_15T_*",
    "30T": "darts_pipeline_freq_30T_*",
}


def _find_pipeline_base(frequency: str) -> Optional[Path]:
    """Return first matching pipeline directory for the given frequency, or None."""
    pattern = FREQUENCY_PATTERNS.get(frequency)
    if not pattern:
        return None
    for base in (ANALYSIS_DIR, DRAI_MODELING_DIR):
        if not base.exists():
            continue
        matches = list(base.glob(pattern))
        if matches:
            return matches[0]
    return None


def get_model_paths_for_frequency(frequency: str) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Return (model_MD_path, model_C_path) for the given frequency (1h, 15T, 30T).
    Search order:
      1) analysis/darts_pipeline_freq_<freq>_*/ (inside my_inference_app)
      2) ../DRAI-Modeling/data/analysis/darts_pipeline_freq_<freq>_*/
      3) For 30T only: models/model_MDNC_M_D.pkl and models/model_MDNC_C.pkl
    """
    model_base = _find_pipeline_base(frequency)
    if model_base:
        md_path = model_base / "MDNC_M_D" / "model_MDNC_M_D.pkl"
        c_path = model_base / "MDNC_C" / "model_MDNC_C.pkl"
        return (md_path if md_path.exists() else None, c_path if c_path.exists() else None)
    # Fallback: local models/ only for 30T (legacy single set)
    if frequency == "30T":
        md = MODELS_DIR / "model_MDNC_M_D.pkl"
        c = MODELS_DIR / "model_MDNC_C.pkl"
        return (md if md.exists() else None, c if c.exists() else None)
    return (None, None)


# Default frequency when not selected (30T to match previous behavior)
DEFAULT_MODEL_FREQUENCY = "30T"
MODELS_BY_FREQUENCY = {
    freq: get_model_paths_for_frequency(freq) for freq in FREQUENCY_OPTIONS
}

UPLOADS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)


def _log_model_locations():
    """Print where the app looks for models (for user reference)."""
    print("[Models] Search locations:")
    print(f"  1) {ANALYSIS_DIR} (in-project analysis/)")
    print(f"  2) {DRAI_MODELING_DIR} (DRAI-Modeling/data/analysis/)")
    print(f"  3) For 30T only: {MODELS_DIR} (model_MDNC_M_D.pkl, model_MDNC_C.pkl)")
    for freq in FREQUENCY_OPTIONS:
        md, c = get_model_paths_for_frequency(freq)
        status = "OK" if (md and c) else "MISSING"
        print(f"  [{freq}] {status}  MD={md or '-'}  C={c or '-'}")
    print()

# In-memory storage for simulation runners (in production, use Redis or database)
# Key: session_id, Value: SimulationRunner instance
_simulation_runners = {}


@app.route("/")
def index():
    """صفحه اصلی داشبورد."""
    # Initialize session if not exists
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    """
    دریافت چندین فایل CSV برای هر دسته (Process Tags, Pellet, MD/Quality) و پردازش.
    """
    # Get file lists (multiple files per category)
    process_files = request.files.getlist("process_files")
    pellet_files = request.files.getlist("pellet_files")
    md_files = request.files.getlist("md_files")
    
    # Debug: log what we received
    print(f"\n[Upload Debug]")
    print(f"  • process_files received: {len(process_files)} items")
    print(f"  • pellet_files received: {len(pellet_files)} items")
    print(f"  • md_files received: {len(md_files)} items")
    
    # Also check all file keys in request
    all_file_keys = list(request.files.keys())
    print(f"  • All file keys in request: {all_file_keys}")
    
    # Debug: print filenames
    if process_files:
        print(f"  • Process filenames: {[f.filename for f in process_files[:5]]}")
    if pellet_files:
        print(f"  • Pellet filenames: {[f.filename for f in pellet_files[:5]]}")
    if md_files:
        print(f"  • MD filenames: {[f.filename for f in md_files[:5]]}")
    
    # Filter out empty files - accept both .csv and .xlsx
    def is_valid_file(f):
        if not f or not f.filename:
            return False
        filename_lower = f.filename.lower()
        return filename_lower.endswith('.csv') or filename_lower.endswith('.xlsx')
    
    process_files = [f for f in process_files if is_valid_file(f)]
    pellet_files = [f for f in pellet_files if is_valid_file(f)]
    md_files = [f for f in md_files if is_valid_file(f)]
    
    print(f"  • After filtering - process: {len(process_files)}, pellet: {len(pellet_files)}, md: {len(md_files)}")
    
    if not process_files:
        return jsonify({
            "success": False,
            "error": "لطفاً حداقل یک فایل Process Tags آپلود کنید."
        }), 400
    
    if not pellet_files:
        return jsonify({
            "success": False,
            "error": "لطفاً حداقل یک فایل Pellet آپلود کنید."
        }), 400
    
    if not md_files:
        return jsonify({
            "success": False,
            "error": "لطفاً حداقل یک فایل MD/Quality آپلود کنید."
        }), 400
    
    # Ensure session exists
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    session_id = session['session_id']
    # Model frequency: 1h, 15T, or 30T (default 30T)
    model_frequency = request.form.get("model_frequency", DEFAULT_MODEL_FREQUENCY)
    if model_frequency not in FREQUENCY_OPTIONS:
        model_frequency = DEFAULT_MODEL_FREQUENCY
    session["model_frequency"] = model_frequency
    model_md_path, model_c_path = get_model_paths_for_frequency(model_frequency)

    # Log model paths so user can verify location (see README / راهنمای مدل‌ها)
    print(f"[Models] frequency={model_frequency} → MD: {model_md_path or 'NOT FOUND'}, C: {model_c_path or 'NOT FOUND'}")

    try:
        # Process and merge files directly from file objects
        # core_logic will handle temporary file saving internally
        print(f"\n[Upload] Processing files for session {session_id} (model frequency: {model_frequency})...")
        print(f"  • Process files: {len(process_files)}")
        print(f"  • Pellet files: {len(pellet_files)}")
        print(f"  • MD/Quality files: {len(md_files)}")
        
        result = core_logic.process_data(
            process_files=process_files,
            pellet_files=pellet_files,
            md_files=md_files,
            model_md_path=str(model_md_path) if model_md_path else None,
            model_c_path=str(model_c_path) if model_c_path else None
        )
        
        # Create simulation runner
        runner = simulation.SimulationRunner(result['merged_df'])
        _simulation_runners[session_id] = runner
        
        return jsonify({
            "success": True,
            "message": (
                f"فایل‌ها با موفقیت پردازش شدند. "
                f"{result['stats']['rows']:,} ردیف داده آماده است. "
                f"(Process: {result['stats']['process_files']}, "
                f"Pellet: {result['stats']['pellet_files']}, "
                f"MD: {result['stats']['md_files']})"
            ),
            "stats": result['stats']
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": f"خطا در پردازش: {str(e)}"
        }), 500


@app.route("/simulation/start", methods=["POST"])
def simulation_start():
    """شروع شبیه‌سازی."""
    if 'session_id' not in session:
        return jsonify({"success": False, "error": "Session not found"}), 400
    
    session_id = session['session_id']
    runner = _simulation_runners.get(session_id)
    
    if runner is None:
        return jsonify({"success": False, "error": "No data loaded. Please upload files first."}), 400
    
    runner.start()
    return jsonify({"success": True, "message": "Simulation started"})


@app.route("/simulation/pause", methods=["POST"])
def simulation_pause():
    """توقف موقت شبیه‌سازی."""
    if 'session_id' not in session:
        return jsonify({"success": False, "error": "Session not found"}), 400
    
    session_id = session['session_id']
    runner = _simulation_runners.get(session_id)
    
    if runner is None:
        return jsonify({"success": False, "error": "No simulation running"}), 400
    
    runner.pause()
    return jsonify({"success": True, "message": "Simulation paused"})


@app.route("/simulation/resume", methods=["POST"])
def simulation_resume():
    """ادامه شبیه‌سازی."""
    if 'session_id' not in session:
        return jsonify({"success": False, "error": "Session not found"}), 400
    
    session_id = session['session_id']
    runner = _simulation_runners.get(session_id)
    
    if runner is None:
        return jsonify({"success": False, "error": "No simulation running"}), 400
    
    runner.resume()
    return jsonify({"success": True, "message": "Simulation resumed"})


@app.route("/simulation/reset", methods=["POST"])
def simulation_reset():
    """بازنشانی شبیه‌سازی."""
    if 'session_id' not in session:
        return jsonify({"success": False, "error": "Session not found"}), 400
    
    session_id = session['session_id']
    runner = _simulation_runners.get(session_id)
    
    if runner is None:
        return jsonify({"success": False, "error": "No simulation running"}), 400
    
    runner.reset()
    return jsonify({"success": True, "message": "Simulation reset"})


@app.route("/update-dashboard", methods=["GET"])
def update_dashboard():
    """
    Endpoint for HTMX polling to get current simulation state.
    Returns HTML fragment to update dashboard.
    """
    print("\n[DEBUG] 1. Entering /update-dashboard...") # DEBUG PRINT

    if 'session_id' not in session:
        return render_template("partials/dashboard.html", error="Session not found")

    session_id = session['session_id']
    runner = _simulation_runners.get(session_id)

    if runner is None:
        return render_template("partials/dashboard.html", error="No data loaded")

    # Get current step
    current_data = runner.get_current_step()
    progress = runner.get_progress()

    # Run inference using models for the session's selected frequency
    predictions = None
    if current_data:
        import pandas as pd
        df_row = pd.DataFrame([current_data])
        freq = session.get("model_frequency", DEFAULT_MODEL_FREQUENCY)
        model_md_path, model_c_path = get_model_paths_for_frequency(freq)

        model_md = None
        model_c = None

        print(f"[DEBUG] 2. Attempting to load models for frequency: {freq}") # DEBUG PRINT
        if model_md_path:
            print(f"[DEBUG] 2a. Loading MD model from: {model_md_path}") # DEBUG PRINT
            model_md = core_logic.load_model(str(model_md_path))
            if model_md:
                print("[DEBUG] 2b. MD model LOADED SUCCESSFULLY.") # DEBUG PRINT
            else:
                print("[DEBUG] 2b. MD model FAILED to load.") # DEBUG PRINT

        if model_c_path:
            print(f"[DEBUG] 2c. Loading C model from: {model_c_path}") # DEBUG PRINT
            model_c = core_logic.load_model(str(model_c_path))
            if model_c:
                print("[DEBUG] 2d. C model LOADED SUCCESSFULLY.") # DEBUG PRINT
            else:
                print("[DEBUG] 2d. C model FAILED to load.") # DEBUG PRINT

        if model_md or model_c:
            print("[DEBUG] 3. Running inference...") # DEBUG PRINT
            predictions = core_logic.run_inference_for_md_c(model_md, model_c, df_row)
            print("[DEBUG] 4. Inference COMPLETE.") # DEBUG PRINT
        else:
             print("[DEBUG] 3. Skipping inference as no models were loaded.") # DEBUG PRINT


    # Advance to next step (for next poll)
    if runner.is_running and not runner.is_paused:
        runner.get_next_step()

    print("[DEBUG] 5. Rendering template...") # DEBUG PRINT
    return render_template(
        "partials/dashboard.html",
        current_data=current_data,
        predictions=predictions,
        progress=progress,
        error=None
    )


@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Legacy endpoint for backward compatibility.
    Redirects to new upload endpoint.
    """
    return upload()


def _find_free_port(start_port: int, max_tries: int = 10) -> int:
    """Return first port in [start_port, start_port+max_tries) that is free to bind."""
    import socket
    for i in range(max_tries):
        port = start_port + i
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", port))
                return port
        except OSError:
            continue
    raise OSError(f"No free port in range {start_port}-{start_port + max_tries - 1}")


if __name__ == "__main__":
    # Suppress Flask development server warning for offline/air-gapped environments
    import logging
    logging.getLogger('werkzeug').setLevel(logging.ERROR)

    # Port: from env PORT, or first free port in 8000, 8001, ...
    base_port = int(os.environ.get("PORT", "8000"))
    port = _find_free_port(base_port)
    if port != base_port:
        print(f"Port {base_port} is in use, using port {port} instead.\n")

    print("\n" + "="*80)
    print("DRAI OFFLINE INFERENCE APP")
    print("="*80)
    print(f"Server running on: http://127.0.0.1:{port}")
    print(f"Network access: http://0.0.0.0:{port}")
    print("="*80)
    _log_model_locations()

    app.run(debug=True, host="0.0.0.0", port=port, use_reloader=False)
