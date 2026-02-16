# DRAI Offline Inference Application

<div dir="rtl">

# سیستم مانیتورینگ و پیش‌بینی آفلاین DRAI

</div>

A professional, offline-capable web application for processing, merging, and analyzing DRI (Direct Reduced Iron) furnace data with real-time simulation and ML model inference capabilities.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Data Pipeline](#data-pipeline)
- [API Endpoints](#api-endpoints)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Development](#development)

---

## 🎯 Overview

This application is designed for **air-gapped environments** where internet connectivity is not available. It provides a complete offline solution for:

- **Batch processing** of multiple CSV/Excel files (Process Tags, Pellet Data, MD/Quality Data)
- **Automatic data merging** with intelligent encoding and delimiter detection
- **Jalali to Gregorian date conversion** for Persian calendar support
- **Real-time simulation** of furnace operations
- **ML model inference** for predicting MD (Metallization Degree) and C (Carbon) values
- **Residence time calculations** based on production rates (WTH15)

### Use Cases

- Industrial data analysis in secure/isolated environments
- Offline monitoring and prediction systems
- Batch processing of historical furnace data
- Real-time simulation of production processes
- Quality prediction and process optimization

---

## ✨ Features

### Core Capabilities

- ✅ **100% Offline Operation** - No CDN dependencies, all assets bundled locally
- ✅ **Batch File Upload** - Process 50+ files simultaneously per category
- ✅ **Robust File Handling** - Automatic encoding detection (UTF-8, CP1256, Latin1, etc.)
- ✅ **Multi-format Support** - CSV and Excel (.xlsx) files
- ✅ **Smart Delimiter Detection** - Handles comma, semicolon, and tab-separated files
- ✅ **Jalali Calendar Support** - Automatic conversion from Persian to Gregorian dates
- ✅ **Data Deduplication** - Removes duplicate timestamps automatically
- ✅ **Residence Time Calculation** - Physics-based time synchronization
- ✅ **Real-time Simulation** - Step-through simulation with 5-minute intervals
- ✅ **ML Model Inference** - Predict MD and C values using pre-trained models
- ✅ **Status Validation** - Automatic detection of shutdown/startup periods (WTH15 < 80)

### User Interface

- Modern, responsive dashboard with Tailwind CSS
- Real-time updates via HTMX polling
- Persian (RTL) language support
- Drag & drop file upload
- Progress tracking and status indicators
- Interactive simulation controls

---

## 🛠 Technology Stack

### Backend
- **Flask 3.0+** - Web framework and routing
- **Pandas 2.0+** - Data processing and manipulation
- **NumPy** - Numerical computations
- **Joblib** - Model serialization
- **LightGBM** - Machine learning models
- **Scikit-learn** - ML utilities

### Frontend
- **HTMX** - Dynamic HTML updates without JavaScript framework
- **Alpine.js** - Lightweight reactive components
- **Tailwind CSS** - Utility-first CSS framework (standalone)

### Data Processing
- **jdatetime** - Jalali calendar support
- **persiantools** - Persian date utilities
- **openpyxl** - Excel file reading

### Architecture
- **Pure Python Logic** - Core processing separated from web layer
- **Session-based State** - In-memory simulation runners
- **Modular Design** - Easy to extend and maintain

---

## 📁 Project Structure

```
my_inference_app/
├── app.py                 # Flask application and routing
├── core_logic.py          # Core data processing (pure Python)
├── simulation.py          # Simulation engine
├── setup_assets.py        # Asset downloader for offline setup
├── requirements.txt       # Python dependencies
├── README.md              # This file
│
├── models/                # ML model storage
│   └── dummy_model.pkl    # Placeholder model
│
├── static/
│   └── vendor/            # Offline JavaScript/CSS libraries
│       ├── htmx.min.js
│       ├── alpine.min.js
│       └── tailwind.js
│
├── templates/
│   ├── index.html         # Main dashboard
│   └── partials/
│       ├── dashboard.html # Real-time dashboard fragment
│       └── results.html   # Results display
│
└── uploads/               # Temporary file storage (auto-cleaned)
```

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Step 1: Clone or Navigate to Project

```bash
cd /path/to/DRAI-OFFLINE/my_inference_app
```

### Step 2: Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Setup Offline Assets

Download required frontend libraries (HTMX, Alpine.js, Tailwind CSS):

```bash
python setup_assets.py
```

This will download and save all JavaScript/CSS files to `static/vendor/` for offline operation.

### Step 5: (Optional) Add ML Models

Place your trained models in the `models/` directory:

```
models/
├── model_MDNC_M_D.pkl  # Model for MD prediction
└── model_MDNC_C.pkl    # Model for C prediction
```

Or the application will auto-detect models from:
```
../DRAI-Modeling/data/analysis/darts_pipeline_freq_30T_*/MDNC_*/
```

---

## 📖 Usage

### Starting the Server

```bash
python app.py
```

The server will start on:
- Local: `http://127.0.0.1:8000`
- Network: `http://0.0.0.0:8000` (accessible from other devices)

### Using the Application

1. **Open Browser**: Navigate to `http://127.0.0.1:8000`

2. **Upload Files**:
   - **Process Tags**: Select one or more CSV/Excel files containing sensor data
   - **Pellet Data**: Select one or more files with pellet input lab data
   - **MD/Quality Data**: Select one or more files with output quality measurements

3. **Process Data**: Click "آپلود و پردازش" (Upload and Process)
   - System will automatically:
     - Detect file encoding and delimiter
     - Convert Jalali dates to Gregorian
     - Merge all files
     - Calculate residence times
     - Validate data

4. **Start Simulation**: Click "شروع شبیه‌سازی" (Start Simulation)
   - Dashboard updates every 5 seconds
   - Shows current timestamp, WTH15, predicted MD/C, and system status

5. **Control Simulation**:
   - **Pause**: Temporarily stop simulation
   - **Resume**: Continue from where paused
   - **Reset**: Start from beginning

---

## 🔄 Data Pipeline

### Input Files

#### Process Tags Files
- **Format**: CSV or Excel
- **Required Columns**: Date/Time columns (Jalali or Gregorian)
- **Content**: Sensor readings (WTH15, temperatures, pressures, etc.)
- **Frequency**: Typically 5-minute intervals

#### Pellet Data Files
- **Format**: CSV or Excel
- **Required Columns**: Date/Time, Pellet properties (%FeO, CCS, etc.)
- **Content**: Laboratory analysis of input pellets
- **Frequency**: Daily or monthly (will be resampled)

#### MD/Quality Data Files
- **Format**: CSV or Excel
- **Required Columns**: Date/Time, Quality metrics (M_D, C, etc.)
- **Content**: Output quality measurements
- **Frequency**: Lab results (irregular intervals)

### Processing Steps

1. **File Loading**:
   - Try multiple encodings: UTF-8, CP1256, Latin1, UTF-16, etc.
   - Try multiple delimiters: comma, semicolon, tab
   - Handle malformed lines gracefully

2. **Date Conversion**:
   - Auto-detect date columns (case-insensitive)
   - Convert Jalali (Shamsi) to Gregorian datetime
   - Sort by timestamp

3. **Stacking**:
   - Combine all files from each category
   - Remove duplicates based on timestamp
   - Add source file tracking

4. **Merging**:
   - Create 5-minute reference time grid
   - Merge Process Tags (prefix: `INST_`)
   - Merge Pellet data (prefix: `PELLET_`) using `merge_asof`
   - Merge MD/Quality data (prefix: `MDNC_`) using `merge_asof`

5. **Residence Time Calculation**:
   ```
   enter_deltatime_hours = 600 / ((WTH15 + 270) / 2)
   exit_deltatime_hours = 1200 / ((WTH15 + 270) / 2)
   ```

6. **Validation**:
   - Mark rows as "Shutdown" if WTH15 < 80
   - Mark rows as "Normal" if WTH15 >= 80

---

## 🔌 API Endpoints

### Web Interface
- `GET /` - Main dashboard page

### File Upload
- `POST /upload` - Upload and process files
  - **Body**: `multipart/form-data`
  - **Fields**: `process_files[]`, `pellet_files[]`, `md_files[]`
  - **Response**: JSON with success status and statistics

### Simulation Control
- `POST /simulation/start` - Start simulation
- `POST /simulation/pause` - Pause simulation
- `POST /simulation/resume` - Resume simulation
- `POST /simulation/reset` - Reset to beginning

### Dashboard Updates
- `GET /update-dashboard` - Get current simulation state (HTMX polling)
  - **Returns**: HTML fragment with current data and predictions

---

## ⚙️ Configuration

### File Size Limits

Default maximum upload size: **500 MB** (configurable in `app.py`)

```python
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB
```

### Model Paths

The application searches for models in this order:

1. `models/model_MDNC_M_D.pkl` and `models/model_MDNC_C.pkl`
2. `../DRAI-Modeling/data/analysis/darts_pipeline_freq_30T_*/MDNC_*/model_*.pkl`

### Session Management

Sessions are stored in-memory. For production, consider:
- Redis for session storage
- Database for persistent state
- File-based session storage

---

## 🐛 Troubleshooting

### Common Issues

#### 1. "No valid data found" Error

**Cause**: Files couldn't be read due to encoding/delimiter issues

**Solution**:
- Check file format (CSV or Excel)
- Verify file is not corrupted
- Check console logs for specific error messages

#### 2. "Pellet file must have georgian_datetime" Error

**Cause**: Date columns not detected

**Solution**:
- Ensure files have date/time columns
- Column names should include: `date`, `time`, `Year`, `Month`, `Day`, or `georgian_datetime`
- Check that dates are in valid format

#### 3. Encoding Errors

**Cause**: File encoding not recognized

**Solution**:
- System tries multiple encodings automatically
- If all fail, file will be read with error replacement
- Check file encoding manually if needed

#### 4. Simulation Not Starting

**Cause**: No data loaded or session expired

**Solution**:
- Re-upload files
- Check browser console for errors
- Verify session is active

### Debug Mode

Enable detailed logging by checking terminal output. The application logs:
- File loading progress
- Encoding/delimiter detection
- Date conversion results
- Merge statistics
- Error messages with stack traces

---

## 💻 Development

### Code Structure

- **`app.py`**: Web layer - routing, sessions, API endpoints
- **`core_logic.py`**: Business logic - data processing, merging, inference
- **`simulation.py`**: Simulation engine - step-through data playback
- **`templates/`**: Jinja2 templates for HTML rendering

### Adding New Features

1. **New Data Source**: Extend `load_and_stack_csvs()` in `core_logic.py`
2. **New Calculations**: Add functions to `core_logic.py` (keep pure Python)
3. **New UI Components**: Add to `templates/index.html` or create new partials
4. **New Endpoints**: Add routes to `app.py`

### Testing

For offline testing:
1. Use sample CSV files with known structure
2. Verify date conversion accuracy
3. Test with various encodings and delimiters
4. Validate merge results

### Production Deployment

**Warning**: The Flask development server is not suitable for production.

For production, use:
- **Gunicorn**: `gunicorn -w 4 -b 0.0.0.0:8000 app:app`
- **uWSGI**: Configure with proper workers
- **Docker**: Containerize the application
- **Reverse Proxy**: Use Nginx or Apache

---

## 📝 Notes

### Offline Requirements

- All JavaScript/CSS libraries must be in `static/vendor/`
- No CDN dependencies allowed
- Models must be local or in sibling directory

### Performance

- Large file batches (50+ files) may take several minutes to process
- Simulation updates every 5 seconds (configurable)
- Memory usage scales with dataset size

### Security

- Change `SECRET_KEY` in production
- Validate file types and sizes
- Sanitize file names
- Consider authentication for production use

---

## 📄 License

This project is part of the DRAI (Direct Reduced Iron) analysis system.

---

## 👥 Contributors

Developed for offline/air-gapped industrial data analysis environments.

---

## 🔗 Related Projects

- `DRAI-Modeling/` - Data science pipeline and model training
- Model files location: `DRAI-Modeling/data/analysis/`

---

<div dir="rtl">

## 📞 پشتیبانی

برای سوالات و مشکلات، لطفاً لاگ‌های ترمینال و مرورگر را بررسی کنید.

</div>

---

**Last Updated**: February 2026

