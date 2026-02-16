"""
سرور Flask — فقط مسئول روتینگ و وب‌هوک.
منطق پردازش در core_logic قرار دارد.
"""

import os
from pathlib import Path
from flask import Flask, render_template, request, jsonify, session
import uuid

import core_logic
import simulation

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB (increased for batch uploads)
app.config["SECRET_KEY"] = "drai-offline-secret-key-change-in-production"  # For sessions

# مسیرهای پروژه
BASE_DIR = Path(__file__).resolve().parent
UPLOADS_DIR = BASE_DIR / "uploads"
MODELS_DIR = BASE_DIR / "models"
# Try to find models in local models/ directory first, then check DRAI-Modeling
DRAI_MODELING_DIR = BASE_DIR.parent / "DRAI-Modeling" / "data" / "analysis"
# Look for a model directory (using 30T frequency as default)
_model_subdirs = list(DRAI_MODELING_DIR.glob("darts_pipeline_freq_30T_*")) if DRAI_MODELING_DIR.exists() else []
DEFAULT_MODEL_MD = None
DEFAULT_MODEL_C = None
if _model_subdirs:
    # Use the first matching directory
    model_base = _model_subdirs[0]
    md_path = model_base / "MDNC_M_D" / "model_MDNC_M_D.pkl"
    c_path = model_base / "MDNC_C" / "model_MDNC_C.pkl"
    if md_path.exists():
        DEFAULT_MODEL_MD = md_path
    if c_path.exists():
        DEFAULT_MODEL_C = c_path
# Fallback to local models directory
if DEFAULT_MODEL_MD is None:
    DEFAULT_MODEL_MD = MODELS_DIR / "model_MDNC_M_D.pkl"
if DEFAULT_MODEL_C is None:
    DEFAULT_MODEL_C = MODELS_DIR / "model_MDNC_C.pkl"

UPLOADS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

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
    
    try:
        # Process and merge files directly from file objects
        # core_logic will handle temporary file saving internally
        print(f"\n[Upload] Processing files for session {session_id}...")
        print(f"  • Process files: {len(process_files)}")
        print(f"  • Pellet files: {len(pellet_files)}")
        print(f"  • MD/Quality files: {len(md_files)}")
        
        result = core_logic.process_data(
            process_files=process_files,
            pellet_files=pellet_files,
            md_files=md_files,
            model_md_path=str(DEFAULT_MODEL_MD) if DEFAULT_MODEL_MD and DEFAULT_MODEL_MD.exists() else None,
            model_c_path=str(DEFAULT_MODEL_C) if DEFAULT_MODEL_C and DEFAULT_MODEL_C.exists() else None
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
    if 'session_id' not in session:
        return render_template("partials/dashboard.html", error="Session not found")
    
    session_id = session['session_id']
    runner = _simulation_runners.get(session_id)
    
    if runner is None:
        return render_template("partials/dashboard.html", error="No data loaded")
    
    # Get current step
    current_data = runner.get_current_step()
    progress = runner.get_progress()
    
    # Run inference if models are available
    predictions = None
    if current_data:
        # Create a single-row dataframe for inference
        import pandas as pd
        df_row = pd.DataFrame([current_data])
        
        model_md = core_logic.load_model(str(DEFAULT_MODEL_MD)) if DEFAULT_MODEL_MD and DEFAULT_MODEL_MD.exists() else None
        model_c = core_logic.load_model(str(DEFAULT_MODEL_C)) if DEFAULT_MODEL_C and DEFAULT_MODEL_C.exists() else None
        
        if model_md or model_c:
            predictions = core_logic.run_inference_for_md_c(model_md, model_c, df_row)
    
    # Advance to next step (for next poll)
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
    """
    Legacy endpoint for backward compatibility.
    Redirects to new upload endpoint.
    """
    return upload()


if __name__ == "__main__":
    # Suppress Flask development server warning for offline/air-gapped environments
    import logging
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    
    print("\n" + "="*80)
    print("DRAI OFFLINE INFERENCE APP")
    print("="*80)
    print(f"Server running on: http://127.0.0.1:5000")
    print(f"Network access: http://0.0.0.0:5000")
    print("="*80 + "\n")
    
    app.run(debug=True, host="0.0.0.0", port=5000, use_reloader=False)
