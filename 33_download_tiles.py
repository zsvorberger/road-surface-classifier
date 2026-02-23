"""
30_full_download_tiles_progress.py
----------------------------------
Final version of the NAIP → Azure uploader.
Downloads every tile listed in output/naip_index.csv and uploads to your Azure
Blob Storage container, showing live progress and a summary on completion.

Dependencies (pip install):
  - tqdm
  - azure-storage-blob
  - requests
  - pandas
"""

import os
import pandas as pd
import requests
import threading
import time
from io import BytesIO
from pathlib import Path
try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - fallback when tqdm isn't installed
    def tqdm(iterable=None, **kwargs):
        return iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from azure.storage.blob import BlobServiceClient

# === CONFIGURATION ===
SCRIPT_DIR = Path(__file__).resolve().parent
NAIP_INDEX = SCRIPT_DIR / "naip_index.csv"

# New storage account connection string
AZURE_CONNECTION_STRING = os.environ.get(
    "AZURE_STORAGE_CONNECTION_STRING",
    "DefaultEndpointsProtocol=https;"
    "AccountName=tilesstorage01;"
    "AccountKey=Rwn0JdynK1cTlgKuEhoxWMdd1IC5fMPmOtr7Ac0qbn/KHd/6xNPzpWIrvEmBKPlEQv1q+e7qDN7G+AStfxMNKQ==;"
    "EndpointSuffix=core.windows.net",
)

CONTAINER_NAME = "patiles"
BLOB_PREFIX = "naip/"

MAX_WORKERS = 8          # parallel uploads; adjust 6-10 for balance
TIMEOUT = 60             # seconds for slow NAIP servers
HEARTBEAT_INTERVAL = 30  # seconds between "still working" updates


# === HEARTBEAT THREAD ===
def heartbeat():
    start = time.time()
    while True:
        time.sleep(HEARTBEAT_INTERVAL)
        elapsed = int(time.time() - start)
        active = threading.active_count() - 1
        print(f"💓 Still working... {active} threads active | Elapsed: {elapsed}s")


threading.Thread(target=heartbeat, daemon=True).start()


# === AZURE CONNECTION ===
print("🔗 Connecting to Azure Blob Storage...")
blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
container_client = blob_service_client.get_container_client(CONTAINER_NAME)

try:
    container_client.create_container()
    print(f"🪣 Created new container '{CONTAINER_NAME}'")
except Exception:
    print(f"📦 Using existing container '{CONTAINER_NAME}'")

session = requests.Session()


# === LOAD TILE LIST ===
if not NAIP_INDEX.exists():
    raise FileNotFoundError(f"Missing file: {NAIP_INDEX}")

tiles = pd.read_csv(NAIP_INDEX, usecols=["tile_url"], low_memory=False)
tile_urls = tiles["tile_url"].dropna().unique().tolist()
print(f"🧩 Loaded {len(tile_urls)} unique tile URLs for upload.\n")
print("ℹ️  Tiles are streamed directly to Azure Blob Storage and not saved locally.\n")


# === FETCH EXISTING BLOBS (ONE-TIME) ===
print("📋 Checking Azure for existing tiles (one-time)...")
existing_blobs = {b.name for b in container_client.list_blobs(name_starts_with=BLOB_PREFIX)}
print(f"✅ Found {len(existing_blobs)} existing blobs in Azure.\n")


# === UPLOAD FUNCTION ===
def upload_tile(tile_url):
    """Download and upload a single tile, with detailed logging."""
    blob_name = f"{BLOB_PREFIX}{os.path.basename(tile_url.split('?')[0])}"

    # 1️⃣ Skip if already uploaded
    if blob_name in existing_blobs:
        print(f"🟡 Skipping (already exists): {blob_name}")
        return "skipped"

    # 2️⃣ Download from NAIP / Planetary Computer
    print(f"🌐 Starting download: {tile_url}")
    try:
        with session.get(tile_url, stream=True, timeout=TIMEOUT) as r:
            r.raise_for_status()
            data = BytesIO()
            for chunk in r.iter_content(chunk_size=1024 * 256):
                data.write(chunk)
        data.seek(0)
        size_mb = len(data.getvalue()) / 1_000_000
        print(f"📦 Finished download: {blob_name} ({size_mb:.2f} MB)")
    except Exception as e:
        print(f"❌ Download error for {blob_name}: {e}")
        return "download_error"

    # 3️⃣ Upload to Azure
    try:
        blob_client = container_client.get_blob_client(blob_name)
        print(f"⬆️  Uploading to Azure: {blob_name}")
        blob_client.upload_blob(data, overwrite=False)
        print(f"✅ Upload complete: {blob_name}")
        return "uploaded"
    except Exception as e:
        print(f"❌ Upload error for {blob_name}: {e}")
        return "upload_error"


# === MAIN EXECUTION ===
def main():
    print(f"=== Starting full NAIP → Azure upload ({MAX_WORKERS} threads) ===")

    uploaded = skipped = failed = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(upload_tile, url): url for url in tile_urls}

        with tqdm(total=len(tile_urls), desc="Uploading tiles", unit="tile") as pbar:
            for future in as_completed(futures):
                result = future.result()
                if result == "uploaded":
                    uploaded += 1
                elif result == "skipped":
                    skipped += 1
                else:
                    failed += 1
                pbar.update(1)

    print("\n=== Summary ===")
    print(f"✅ Uploaded: {uploaded}")
    print(f"🟡 Skipped (already existed): {skipped}")
    print(f"❌ Failed: {failed}")
    print(f"All tiles saved to '{CONTAINER_NAME}/{BLOB_PREFIX}'.")


if __name__ == "__main__":
    main()
