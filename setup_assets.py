"""
Offline environment setup script.

Downloads HTMX, Alpine.js, and Tailwind and saves them under static/vendor.

Run this script once with internet access to download assets; the app then runs fully offline.
Place real models (model_MDNC_M_D.pkl, model_MDNC_C.pkl) in models/ for inference.
"""

import ssl
import urllib.error
import urllib.request
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent
VENDOR_DIR = BASE_DIR / "static" / "vendor"

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


def main() -> None:
    print("Setting up offline assets...\n")

    VENDOR_DIR.mkdir(parents=True, exist_ok=True)

    ok = 0
    for asset in ASSETS:
        dest = VENDOR_DIR / asset["filename"]
        if download_file(asset["url"], dest):
            ok += 1
    print(f"\nDownloaded {ok}/{len(ASSETS)} asset(s).")
    print("\nDone. Assets are in static/vendor.")


if __name__ == "__main__":
    main()
