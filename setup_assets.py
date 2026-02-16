"""
Offline environment setup script.

1) Downloads HTMX, Alpine.js, and Tailwind and saves them under static/vendor.
2) If no model exists, creates a dummy model for testing and saves it under models/.

Run this script once with internet access to download assets; the app then runs fully offline.
"""

import os
import ssl
import urllib.error
import urllib.request
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent
VENDOR_DIR = BASE_DIR / "static" / "vendor"
MODELS_DIR = BASE_DIR / "models"
DUMMY_MODEL_PATH = MODELS_DIR / "dummy_model.pkl"

# URLs for downloading (used only by this script; HTML references local paths)
ASSETS = [
    {
        "url": "https://unpkg.com/htmx.org@1.9.10/dist/htmx.min.js",
        "filename": "htmx.min.js",
    },
    {
        "url": "https://unpkg.com/alpinejs@3.14.3/dist/cdn.min.js",
        "filename": "alpine.min.js",
    },
    {
        "url": "https://cdn.tailwindcss.com",
        "filename": "tailwind.js",
    },
]

# Timeout in seconds for each download
DOWNLOAD_TIMEOUT = 60


def download_file(url: str, dest_path: Path) -> bool:
    """
    Download a file from URL and save to the given path.
    Returns True on success, False on failure.
    """
    print(f"  Downloading: {url}")
    try:
        # Use default SSL context; allow unverified only if needed for air-gapped copy
        ctx = ssl.create_default_context()
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; DRAI-Setup/1.0)"},
        )
        with urllib.request.urlopen(req, timeout=DOWNLOAD_TIMEOUT, context=ctx) as resp:
            if resp.status != 200:
                print(f"  Error: HTTP {resp.status}")
                return False
            content = resp.read()
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_bytes(content)
        print(f"  Saved: {dest_path.relative_to(BASE_DIR)}")
        return True
    except urllib.error.HTTPError as e:
        print(f"  Error: HTTP {e.code} - {e.reason}")
        return False
    except urllib.error.URLError as e:
        print(f"  Error: {e.reason}")
        return False
    except OSError as e:
        print(f"  Error: {e}")
        return False


def create_dummy_model() -> None:
    """Create a dummy sklearn model and save with joblib for testing when no real model exists."""
    if DUMMY_MODEL_PATH.exists():
        print("  Dummy model already exists; skipping.")
        return
    try:
        import joblib
        import numpy as np
        from sklearn.dummy import DummyClassifier

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        X = np.random.randn(20, 3)
        y = np.array([0, 1] * 10)
        model = DummyClassifier(strategy="stratified")
        model.fit(X, y)
        joblib.dump(model, DUMMY_MODEL_PATH)
        print(f"  Dummy model created: {DUMMY_MODEL_PATH.relative_to(BASE_DIR)}")
    except ImportError as e:
        print(f"  Skipping dummy model: scikit-learn and joblib required. ({e})")


def main() -> None:
    print("Setting up offline assets...\n")

    VENDOR_DIR.mkdir(parents=True, exist_ok=True)

    ok = 0
    for asset in ASSETS:
        dest = VENDOR_DIR / asset["filename"]
        if download_file(asset["url"], dest):
            ok += 1
    print(f"\nDownloaded {ok}/{len(ASSETS)} asset(s).")

    print("\nChecking dummy model...")
    create_dummy_model()

    print("\nDone. Assets are in static/vendor; dummy model (if created) is in models/.")


if __name__ == "__main__":
    main()
