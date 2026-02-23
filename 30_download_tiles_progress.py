"""
30_debug_download_tiles_progress.py
-----------------------------------
Diagnostic version of NAIP → Azure uploader.
Shows detailed logs for every step so you can see exactly where it's stuck.
Uses in-memory duplicate check (no .exists() calls).
"""

import os
import pandas as pd
import requests
import threading
import time
from io import BytesIO
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from azure.storage.blob import BlobServiceClient

# === CONFIGURATION ===
NAIP_INDEX = "output/naip_index.csv"
AZURE_CONNECTION_STRING = "DefaultEndpointsProtocol=https;AccountName=mapstoragepa;AccountKey=5Vn4LGabS+qBqyKqgtPOSooAr3MNzRUsGbiC792YADRdScC17178H/Ogv4ShoXIJdQF3eDQE1cND+AStC0ZWKg==;EndpointSuffix=core.windows.net"
CONTAINER_NAME = "naip-tiles"
BLOB_PREFIX = "pa/"
MAX_WORKERS = 4         # number of parallel uploads
TIMEOUT = 20            # shorter timeout for quick diagnostics
HEARTBEAT_INTERVAL = 10 # seconds

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
    pass

session = requests.Session()

# === LOAD TILE LIST ===
if not os.path.exists(NAIP_INDEX):
    raise FileNotFoundError(f"Missing file: {NAIP_INDEX}")

tiles = pd.read_csv(NAIP_INDEX, usecols=["tile_url"], low_memory=False)
tile_urls = tiles["tile_url"].dropna().unique().tolist()[:20]  # limit to 20 for quick test
print(f"🧩 Loaded {len(tile_urls)} unique tile URLs for test.\n")

# === FETCH EXISTING BLOBS (ONE-TIME) ===
print("📋 Checking Azure for existing tiles (one-time)...")
existing_blobs = {b.name for b in container_client.list_blobs(name_starts_with=BLOB_PREFIX)}
print(f"✅ Found {len(existing_blobs)} existing blobs in Azure.\n")

# === UPLOAD FUNCTION ===
def upload_tile(tile_url):
    """Download and upload a single tile, with detailed logging."""
    blob_name = f"{BLOB_PREFIX}{os.path.basename(tile_url.split('?')[0])}"

    # Step 1: duplicate check
    if blob_name in existing_blobs:
        print(f"🟡 Skipping (already exists): {blob_name}")
        return "skipped"

    # Step 2: download from NAIP
    print(f"🌐 Starting download: {tile_url}")
    try:
        with session.get(tile_url, stream=True, timeout=TIMEOUT) as r:
            r.raise_for_status()
            data = BytesIO()
            for chunk in r.iter_content(chunk_size=1024 * 256):
                data.write(chunk)
        data.seek(0)
        print(f"📦 Finished download: {blob_name} ({len(data.getvalue())/1_000_000:.2f} MB)")
    except Exception as e:
        print(f"❌ Download error for {blob_name}: {e}")
        return "download_error"

    # Step 3: upload to Azure
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
    print(f"=== Starting DEBUG NAIP → Azure Upload ({MAX_WORKERS} threads) ===")

    uploaded = 0
    skipped = 0
    failed = 0

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
    print(f"🟡 Skipped: {skipped}")
    print(f"❌ Failed: {failed}")
    print(f"All tiles saved to '{CONTAINER_NAME}/{BLOB_PREFIX}'.")

if __name__ == "__main__":
    main()
