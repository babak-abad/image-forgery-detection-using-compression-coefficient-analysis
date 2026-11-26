# download_vision.py
"""Download a balanced, PAIRED subset of the VISION dataset (Univ. of Florence).

VISION (https://lesc.dinfo.unifi.it/VISION/, CC BY-SA 4.0, 35 devices) stores,
per device folder Dxx_<name>:

  images/nat/    native camera JPEGs          -> k = 1 (single-compressed)
  images/natWA/  WhatsApp re-encodes of nat   -> k = 2 (double-compressed)
  images/flat|natFBH|natFBL/                  -> screen / Facebook versions (unused)

File names encode the photo identity: <Dxx>_I_nat_<NNNN>.jpg in nat/ and
<Dxx>_I_natWA_<NNNN>.jpg in natWA/ are the SAME photo (same 4-digit index).
This downloader samples the SAME indices in both pools per device, so every
downloaded photo exists in both compression histories. dataset_sources.py
groups files by (device, index) so a photo and its WhatsApp twin can never
land in different train/val/test splits (content leakage guard).

Local layout (VISION_ROOT):
  <root>/nat/<device>/<original file name>
  <root>/natWA/<device>/<original file name>

Every successful file is appended to sources_manifest.csv using the existing
schema: file,source,url,width,height,bpp,bytes,sha256,license,downloaded_at.
Re-runs resume: files that already exist and pass a PIL structural check are
re-used, and rows whose file path is already in the manifest are not duplicated.

Run from the project root with the venv python:
  python src/download_vision.py --dry-run
  python src/download_vision.py --devices D01 --per-pool 2   # smoke test
  python src/download_vision.py                              # full subset

While a run is in progress (also fine fully in the background), a second
terminal can show a live progress bar with the remaining number of images and
an ETA:
  python src/download_progress.py
It reads vision_download_status.json, which this script rewrites after every
finished file; runs started before this status output existed are monitored
by counting files on disk instead.
"""
import argparse
import csv
import hashlib
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from PIL import Image
from tqdm import tqdm

VISION_BASE = "https://lesc.dinfo.unifi.it/VISION/dataset/"
VISION_ROOT = "D:/projects/datasets/double compress checking/vision"
MANIFEST_PATH = "D:/projects/datasets/double compress checking/sources_manifest.csv"
STATUS_PATH = "D:/projects/datasets/double compress checking/vision_download_status.json"
MANIFEST_COLUMNS = [
    "file",
    "source",
    "url",
    "width",
    "height",
    "bpp",
    "bytes",
    "sha256",
    "license",
    "downloaded_at",
]
LICENSE = "CC BY-SA 4.0"
POOLS = ("nat", "natWA")
POOL_SOURCE_NAMES = {"nat": "vision_nat", "natWA": "vision_natwa"}
USER_AGENT = "image-forgery-detection/1.0 (dataset fetcher)"
DEVICE_LINK_RE = re.compile(r'href="(D\d+[^"/]*/)"')
FILE_LINK_RE = re.compile(r'href="([^"]+\.jpe?g)"', re.IGNORECASE)
TRAILING_INDEX_RE = re.compile(r"(\d+)$")


def http_get(url, timeout=60, retries=6):
    """GET `url` and return the raw body; retry with backoff.

    HTTP 429 (rate limit) uses long exponential waits (20s, 40s, 80s, ...),
    other errors use short linear waits.
    """
    last_exc = None
    for attempt in range(retries):
        try:
            time.sleep(random.uniform(0.05, 0.25))  # politeness
            request = Request(url, headers={"User-Agent": USER_AGENT})
            with urlopen(request, timeout=timeout) as response:
                return response.read()
        except HTTPError as exc:
            last_exc = exc
            if exc.code == 429:
                wait = 20.0 * (2 ** attempt)
                print(
                    f"  rate limited (429), waiting {wait:.0f}s "
                    f"(attempt {attempt + 1}/{retries}): {url}",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(wait)
            else:
                time.sleep(1.5 * (attempt + 1))
        except Exception as exc:
            last_exc = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"GET failed after {retries} attempts: {url} ({last_exc})")


def list_devices():
    """Parse the device folders (Dxx_*) from the dataset index listing."""
    html = http_get(VISION_BASE).decode("utf-8", "replace")
    devices = DEVICE_LINK_RE.findall(html)
    if not devices:
        raise RuntimeError(f"no device folders found in index: {VISION_BASE}")
    return devices


def list_pool_files(device):
    """Return {pool: {photo_index: file_name}} for nat/ and natWA/ of a device."""
    pools = {}
    for pool in POOLS:
        url = f"{VISION_BASE}{device}images/{pool}/"
        html = http_get(url).decode("utf-8", "replace")
        by_index = {}
        for name in FILE_LINK_RE.findall(html):
            stem = os.path.splitext(name)[0]
            match = TRAILING_INDEX_RE.search(stem)
            index = match.group(1) if match else stem
            by_index[index] = name
        if not by_index:
            raise RuntimeError(f"no images listed under {url}")
        pools[pool] = by_index
    return pools


def sample_keys(keys_sorted, k):
    """Evenly spaced deterministic sample of `k` keys from a sorted list."""
    keys = sorted(keys_sorted)
    if len(keys) <= k:
        return keys
    if k == 1:
        return [keys[len(keys) // 2]]
    positions = sorted({round(i * (len(keys) - 1) / (k - 1)) for i in range(k)})
    return [keys[position] for position in positions]

def is_valid_image(path):
    """Structural JPEG check (PIL verify, no full decode)."""
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        return False
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


def download_file(url, dest):
    """Fetch `url` -> `dest` atomically (via .part). Returns (status, bytes)."""
    if is_valid_image(dest):
        return "exists", os.path.getsize(dest)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    data = http_get(url, timeout=180)
    part_path = dest + ".part"
    with open(part_path, "wb") as handle:
        handle.write(data)
    os.replace(part_path, dest)
    if not is_valid_image(dest):
        os.remove(dest)
        raise RuntimeError(f"downloaded file failed PIL verification: {dest}")
    return "downloaded", len(data)


def sha256_file(path):
    """Streamed SHA-256 hex digest of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest_row(path, source_name, url):
    """One sources_manifest.csv row for a downloaded file (PIL header read)."""
    with Image.open(path) as image:
        width, height = image.size
        bpp = len(image.getbands()) * 8
    return {
        "file": path,
        "source": source_name,
        "url": url,
        "width": width,
        "height": height,
        "bpp": bpp,
        "bytes": os.path.getsize(path),
        "sha256": sha256_file(path),
        "license": LICENSE,
        "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def load_manifest_files():
    """Set of (case-normalised) file paths already present in the manifest."""
    if not os.path.isfile(MANIFEST_PATH):
        return set()
    with open(MANIFEST_PATH, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return {os.path.normcase(row["file"]) for row in reader if row.get("file")}


def append_manifest_rows(rows):
    """Append rows to the manifest (creating it with header when missing)."""
    if not rows:
        print("No new manifest rows to append.")
        return
    is_new = not os.path.isfile(MANIFEST_PATH)
    os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)
    with open(MANIFEST_PATH, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerows(rows)
    print(f"Appended {len(rows)} rows to {MANIFEST_PATH}")


def utc_now_iso():
    """UTC timestamp string used in the status file."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_status(status):
    """Atomically dump the progress state consumed by download_progress.py."""
    os.makedirs(os.path.dirname(STATUS_PATH), exist_ok=True)
    tmp_path = STATUS_PATH + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(status, handle, indent=2)
    os.replace(tmp_path, STATUS_PATH)


def fetch_job(job):
    """Worker: download one planned file and build its manifest row."""
    status, _byte_count = download_file(url=job["url"], dest=job["dest"])
    row = build_manifest_row(path=job["dest"], source_name=job["source"], url=job["url"])
    return "ok", status, row

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--per-pool",
        type=int,
        default=40,
        help="images to sample per device per pool (nat and natWA, default 40)",
    )
    parser.add_argument(
        "--devices",
        type=str,
        default="",
        help="comma-separated subset of device folders (e.g. D01,D02); default all",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="parallel download threads (default 4)",
    )
    parser.add_argument("--dry-run", action="store_true", help="show the plan only")
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"VISION base:  {VISION_BASE}")
    print(f"Local root:   {VISION_ROOT}")
    print(f"Manifest:     {MANIFEST_PATH}")

    devices = list_devices()
    if args.devices:
        wanted = [item.strip().lower() for item in args.devices.split(",") if item.strip()]
        selected = []
        for token in wanted:
            matches = [
                device
                for device in devices
                if device.rstrip("/").lower().startswith(token)
                or token in device.rstrip("/").lower()
            ]
            if not matches:
                known = ", ".join(sorted(device.rstrip("/") for device in devices))
                sys.exit(f"unknown device filter: {token}; known: {known}")
            selected.extend(device for device in matches if device not in selected)
        devices = selected

    print(f"Devices:      {len(devices)} (per-pool target {args.per_pool})")

    progress_state = {
        "phase": "listing",
        "started_at": utc_now_iso(),
        "updated_at": utc_now_iso(),
        "finished_at": None,
        "total": None,
        "planned": {},
        "downloaded": 0,
        "exists": 0,
        "failed": 0,
        "devices": len(devices),
        "per_pool": args.per_pool,
    }
    if not args.dry_run:
        write_status(status=progress_state)

    pools_by_device = {}
    skipped_devices = []
    for device in tqdm(devices, desc="Listing", unit="dev"):
        for attempt in range(3):
            try:
                pools_by_device[device] = list_pool_files(device)
                break
            except Exception as exc:
                if attempt == 2:
                    skipped_devices.append(device.rstrip("/"))
                    print(
                        f"  warning: skipping {device.rstrip('/')} "
                        f"(listing failed: {exc})",
                        file=sys.stderr,
                        flush=True,
                    )
                else:
                    time.sleep(5.0 * (attempt + 1))
        time.sleep(0.8)  # pacing: avoid bursts of index requests

    if skipped_devices:
        print(
            f"Skipping {len(skipped_devices)} devices with failed listings "
            f"(re-run later to resume): {', '.join(skipped_devices)}"
        )

    plan = []
    for device in devices:
        pools = pools_by_device.get(device)
        if not pools:
            continue
        keys = sample_keys(keys_sorted=pools["nat"].keys(), k=args.per_pool)
        device_name = device.rstrip("/")
        for pool in POOLS:
            for key in keys:
                file_name = pools[pool].get(key)
                if file_name is None:
                    continue
                plan.append(
                    {
                        "url": f"{VISION_BASE}{device}images/{pool}/{file_name}",
                        "dest": os.path.join(VISION_ROOT, pool, device_name, file_name),
                        "source": POOL_SOURCE_NAMES[pool],
                        "device": device_name,
                    }
                )

    if args.dry_run:
        for device in devices:
            pools = pools_by_device[device]
            device_name = device.rstrip("/")
            planned = {
                pool: sum(
                    1
                    for job in plan
                    if job["device"] == device_name
                    and job["source"] == POOL_SOURCE_NAMES[pool]
                )
                for pool in POOLS
            }
            print(
                f"  {device_name}: nat {planned['nat']}/{len(pools['nat'])}, "
                f"natWA {planned['natWA']}/{len(pools['natWA'])}"
            )
        per_pool = {
            pool: sum(1 for job in plan if job["source"] == POOL_SOURCE_NAMES[pool])
            for pool in POOLS
        }
        print(
            f"Plan: {len(plan)} files total "
            f"(nat {per_pool['nat']} + natWA {per_pool['natWA']}); no downloads in dry-run."
        )
        return

    existing_files = load_manifest_files()
    planned_counts = {
        pool: sum(1 for job in plan if job["source"] == POOL_SOURCE_NAMES[pool])
        for pool in POOLS
    }
    progress_state.update(
        phase="downloading",
        total=len(plan),
        planned=planned_counts,
        updated_at=utc_now_iso(),
    )
    write_status(status=progress_state)
    stats = {"downloaded": 0, "exists": 0, "failed": 0}
    failures = []
    new_rows = []
    rows_flushed = 0
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(fetch_job, job): job for job in plan}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Fetching", unit="img"):
                job = futures[future]
                try:
                    outcome, status, payload = future.result()
                    if outcome == "error":
                        stats["failed"] += 1
                        failures.append(f"{job['url']} -> {payload}")
                        continue
                    stats[status] += 1
                    if os.path.normcase(job["dest"]) not in existing_files:
                        new_rows.append(payload)
                        if len(new_rows) >= 25:
                            rows_flushed += len(new_rows)
                            append_manifest_rows(rows=new_rows)
                            new_rows.clear()
                except Exception as exc:
                    stats["failed"] += 1
                    failures.append(f"{job['url']} -> {exc}")
                    continue
                finally:
                    progress_state.update(stats)
                    progress_state["updated_at"] = utc_now_iso()
                    write_status(status=progress_state)
    except BaseException:
        progress_state["phase"] = "aborted"
        progress_state["updated_at"] = utc_now_iso()
        write_status(status=progress_state)
        raise

    rows_flushed += len(new_rows)
    append_manifest_rows(rows=new_rows)
    progress_state["phase"] = "done"
    progress_state["finished_at"] = utc_now_iso()
    progress_state["updated_at"] = progress_state["finished_at"]
    write_status(status=progress_state)
    print(
        f"Done: {stats['downloaded']} downloaded, {stats['exists']} reused, "
        f"{stats['failed']} failed, {rows_flushed} new manifest rows."
    )
    if failures:
        print("Failures:")
        for line in failures:
            print(f"  {line}")
        sys.exit(1)


if __name__ == "__main__":
    main()

