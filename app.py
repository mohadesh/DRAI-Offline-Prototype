# app.py
import os
import uuid
import logging
import socket
from pathlib import Path
from flask import Flask, render_template, request, jsonify, session
import pandas as pd

# ماژول‌های داخلی
import core_logic
import simulation
import model_loader

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024
app.config["SECRET_KEY"] = "secret-key-offline"

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
ANALYSIS_DIR = BASE_DIR / "analysis"
DRAI_MODELING_DIR = BASE_DIR.parent / "DRAI-Modeling" / "data" / "analysis"

# تنظیمات فرکانس
FREQUENCY_OPTIONS = ("1h", "15T", "30T")
DEFAULT_MODEL_FREQUENCY = "30T"
FREQUENCY_PATTERNS = {
    "1h": "darts_pipeline_freq_1h_*",
    "15T": "darts_pipeline_freq_15T_*",
    "30T": "darts_pipeline_freq_30T_*",
}

# کش مدل‌ها در رم
MODEL_CACHE = {}
_simulation_runners = {}

def get_model_paths_for_frequency(frequency):
    pattern = FREQUENCY_PATTERNS.get(frequency)
    if not pattern: return None, None
    for base in (ANALYSIS_DIR, DRAI_MODELING_DIR):
        if not base.exists(): continue
        matches = list(base.glob(pattern))
        if matches:
            matches.sort()
            latest_folder = matches[-1]
            md_path = latest_folder / "MDNC_M_D" / "model_MDNC_M_D.pkl"
            c_path = latest_folder / "MDNC_C" / "model_MDNC_C.pkl"
            return (md_path, c_path)
    if frequency == "30T":
        md = MODELS_DIR / "model_MDNC_M_D.pkl"
        c = MODELS_DIR / "model_MDNC_C.pkl"
        return (md if md.exists() else None, c if c.exists() else None)
    return (None, None)

def get_cached_model(path):
    if not path: return None
    path_str = str(path)
    if path_str in MODEL_CACHE:
        return MODEL_CACHE[path_str]
    model = model_loader.load_model(path_str)
    if model:
        MODEL_CACHE[path_str] = model
    return model

@app.route("/")
def index():
    if 'session_id' not in session: session['session_id'] = str(uuid.uuid4())
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    p_files = request.files.getlist("process_files")
    pl_files = request.files.getlist("pellet_files")
    m_files = request.files.getlist("md_files")

    valid = lambda f: f and f.filename.endswith(('.csv', '.xlsx'))
    p_files = [f for f in p_files if valid(f)]
    pl_files = [f for f in pl_files if valid(f)]
    m_files = [f for f in m_files if valid(f)]

    if not (p_files and pl_files and m_files):
        return jsonify({"success": False, "error": "Missing files"}), 400

    if 'session_id' not in session: session['session_id'] = str(uuid.uuid4())
    sid = session['session_id']

    model_freq = request.form.get("model_frequency", DEFAULT_MODEL_FREQUENCY)
    if model_freq not in FREQUENCY_OPTIONS: model_freq = DEFAULT_MODEL_FREQUENCY
    session["model_frequency"] = model_freq

    try:
        result = core_logic.process_data(
            process_files=p_files,
            pellet_files=pl_files,
            md_files=m_files,
            resample_rate=model_freq  # Pass the frequency selec by the user
        )
        
        runner = simulation.SimulationRunner(result['merged_df'])
        _simulation_runners[sid] = runner

        md_path, c_path = get_model_paths_for_frequency(model_freq)
        if md_path: get_cached_model(md_path)
        if c_path: get_cached_model(c_path)

        return jsonify({"success": True, "message": "Processed successfully", "stats": result['stats']})
    except Exception as e:
        logger.error(f"Upload failed: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500

# (سایر روت‌های سیمولیشن بدون تغییر)
@app.route("/simulation/start", methods=["POST"])
def sim_start():
    runner = _simulation_runners.get(session.get('session_id'))
    if runner: runner.start(); return jsonify({"success": True})
    return jsonify({"success": False, "error": "No data"}), 400

@app.route("/simulation/pause", methods=["POST"])
def sim_pause():
    runner = _simulation_runners.get(session.get('session_id'))
    if runner: runner.pause(); return jsonify({"success": True})
    return jsonify({"success": False}), 400

@app.route("/simulation/resume", methods=["POST"])
def sim_resume():
    runner = _simulation_runners.get(session.get('session_id'))
    if runner: runner.resume(); return jsonify({"success": True})
    return jsonify({"success": False}), 400

@app.route("/simulation/reset", methods=["POST"])
def sim_reset():
    runner = _simulation_runners.get(session.get('session_id'))
    if runner: runner.reset(); return jsonify({"success": True})
    return jsonify({"success": False}), 400


@app.route("/update-dashboard", methods=["GET"])
def update_dashboard():
    """
    Endpoint for HTMX polling to get current simulation state.
    Sends a HISTORY WINDOW to the inference engine.
    """
    if 'session_id' not in session:
        return render_template("partials/dashboard.html", error="Session not found")

    session_id = session['session_id']
    runner = _simulation_runners.get(session_id)

    if runner is None:
        return render_template("partials/dashboard.html", error="No data loaded")

    # Get current step data (for display)
    current_data = runner.get_current_step()
    progress = runner.get_progress()

    predictions = None
    if current_data:
        WINDOW_SIZE = 120 # Safe buffer (e.g. 10 hours of data at 5T)

        # Slice from runner.df using current index
        start_idx = max(0, runner.current_index - WINDOW_SIZE)
        end_idx = runner.current_index + 1

        # Prepare the dataframe window
        df_window = runner.df.iloc[start_idx:end_idx].copy()
        
        # --- FIX APPLIED HERE ---
        # Ensure the frequency is retrieved from the session and passed to the inference function.
        freq = session.get("model_frequency", DEFAULT_MODEL_FREQUENCY)
        md_path, c_path = get_model_paths_for_frequency(freq)

        model_md = get_cached_model(md_path)
        model_c = get_cached_model(c_path)

        if model_md or model_c:
            # Pass the WINDOW and the FREQUENCY to core_logic
            predictions = core_logic.run_inference_for_md_c(
                model_md=model_md, 
                model_c=model_c, 
                df_window=df_window, 
                frequency=freq  # Pass the frequency here
            )

    # Advance to next step
    if runner.is_running and not runner.is_paused:
        runner.get_next_step()

    return render_template(
        "partials/dashboard.html",
        current_data=current_data,
        predictions=predictions,
        progress=progress,
        error=None
    )
def find_free_port(start_port=8000, max_tries=100):
    for port in range(start_port, start_port + max_tries):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", port))
                return port
        except OSError:
            continue
    raise IOError("No free port found")

if __name__ == "__main__":
    try:
        port = find_free_port()
        print(f"Server starting on http://127.0.0.1:{port}")
        app.run(debug=True, port=port, use_reloader=False)
    except IOError as e:
        print(f"Error: {e}")
