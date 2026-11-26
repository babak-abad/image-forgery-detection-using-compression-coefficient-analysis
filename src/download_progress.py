# download_progress.py
"""Live progress monitor for the VISION dataset downloader (download_vision.py).

Run download_vision.py in one terminal (or fully in the background), then start
this monitor in a second terminal from the project root:

  python src/download_progress.py

It renders one refreshing line with a progress bar, the remaining number of
images, the download rate and the estimated remaining time (ETA); Ctrl+C exits.

Two data sources, auto-selected on every refresh:

  status mode  reads vision_download_status.json, which download_vision.py
               rewrites after every finished file -> exact total and counts.
  watch mode   no status file (e.g. a download started before the downloader
               gained status output) -> counts *.jpg/*.jpeg files under
               VISION_ROOT/nat and natWA and compares them with an ESTIMATED
               total (devices x pools x per-pool). Override with --total,
               --per-pool or --devices when your run differs.

The monitor is read-only: it never writes to the dataset or the manifest, so
it is always safe to run beside an active download. --once prints a single
line without the header, for use in scripts.
"""
import argparse
import json
import os
import shutil
import time
from collections import deque
from datetime import datetime, timezone

from download_vision import POOLS, STATUS_PATH, VISION_ROOT

IMAGE_EXTENSIONS = (".jpg", ".jpeg")
PART_SUFFIX = ".part"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="seconds between refreshes (default 2.0)",
    )
    parser.add_argument(
        "--total",
        type=int,
        default=0,
        help="expected total images (overrides the per-pool estimate)",
    )
    parser.add_argument(
        "--per-pool",
        type=int,
        default=40,
        help="images per device per pool for the estimate (default 40)",
    )
    parser.add_argument(
        "--devices",
        type=int,
        default=35,
        help="device count for the estimate (default 35)",
    )
    parser.add_argument(
        "--bar-width",
        type=int,
        default=28,
        help="progress bar width in characters (default 28)",
    )
    parser.add_argument(
        "--window",
        type=float,
        default=180.0,
        help="seconds of history used for the rate estimate (default 180)",
    )
    parser.add_argument(
        "--stall-warn",
        type=float,
        default=90.0,
        help="seconds without a new image before warning (default 90)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="print a single line and exit (no live refresh)",
    )
    return parser.parse_args()


def count_images():
    """Count finished images and in-flight .part files under the VISION pools."""
    done = 0
    in_flight = 0
    for pool in POOLS:
        pool_dir = os.path.join(VISION_ROOT, pool)
        if not os.path.isdir(pool_dir):
            continue
        for _dirpath, _dirnames, file_names in os.walk(pool_dir):
            for name in file_names:
                if name.lower().endswith(PART_SUFFIX):
                    in_flight += 1
                elif os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS:
                    done += 1
    return done, in_flight


def load_status():
    """Parsed status JSON from the downloader, or None when absent/invalid."""
    try:
        with open(STATUS_PATH, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def parse_iso_timestamp(value):
    """UTC-aware datetime from an ISO string, or None."""
    if not value:
        return None
    try:
        stamp = datetime.fromisoformat(value)
    except ValueError:
        return None
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp


def format_duration(seconds):
    """Compact duration: '1h 57m', '38m 12s', '12s'; unknown -> '--:--'."""
    if seconds is None or seconds < 0:
        return "--:--"
    total_seconds = int(round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def format_rate(rate_per_second):
    """Human download rate in images per minute."""
    if rate_per_second is None or rate_per_second <= 0:
        return "-- img/min"
    return f"{rate_per_second * 60.0:.1f} img/min"


def render_bar(done, total, width):
    """ASCII progress bar for done/total."""
    if width <= 0:
        return ""
    if total <= 0:
        return "[" + "-" * width + "]"
    filled = round(width * min(done, total) / total)
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def collect_state(estimated_total):
    """One observation of the download, from the status file or the disk."""
    status = load_status()
    disk_done, in_flight = count_images()
    wall_now = time.time()
    if status and status.get("phase"):
        started = parse_iso_timestamp(value=status.get("started_at"))
        updated = parse_iso_timestamp(value=status.get("updated_at"))
        elapsed = wall_now - started.timestamp() if started else None
        downloaded = status.get("downloaded") or 0
        return {
            "mode": "status",
            "phase": status["phase"],
            "done": downloaded + (status.get("exists") or 0),
            "total": status.get("total") or estimated_total,
            "failed": status.get("failed") or 0,
            "in_flight": in_flight,
            "elapsed": elapsed,
            "stale_for": max(wall_now - updated.timestamp(), 0.0) if updated else None,
            "avg_rate": (
                downloaded / elapsed
                if elapsed is not None and elapsed > 0 and downloaded > 0
                else None
            ),
        }
    return {
        "mode": "watch",
        "phase": "downloading",
        "done": disk_done,
        "total": estimated_total,
        "failed": None,
        "in_flight": in_flight,
        "elapsed": None,
        "stale_for": None,
        "avg_rate": None,
    }


def build_line(state, percent, remaining, rate, eta, elapsed, stalled_for, bar_width, stall_warn):
    """The single live console line shown by the monitor."""
    segments = [
        f"{render_bar(done=state['done'], total=state['total'], width=bar_width)} {percent:5.1f}%",
        f"{state['done']}/{state['total']}",
        f"remaining {remaining} img",
        f"ETA {format_duration(seconds=eta)}",
        format_rate(rate_per_second=rate),
        f"elapsed {format_duration(seconds=elapsed)}",
    ]
    if state["in_flight"]:
        segments.append(f"{state['in_flight']} in flight")
    if state["failed"]:
        segments.append(f"failed {state['failed']}")
    tag = f"{state['mode']}:{state['phase']}"
    if state["stale_for"] is not None and state["stale_for"] > 60.0:
        tag += f", stale {format_duration(seconds=state['stale_for'])}"
    segments.append(f"[{tag}]")
    line = " | ".join(segments)
    if stalled_for is not None and stalled_for > stall_warn:
        line += f"  ! no new image for {format_duration(seconds=stalled_for)}"
    return line


def main():
    args = parse_args()
    estimated_total = (
        args.total if args.total > 0 else args.per_pool * args.devices * len(POOLS)
    )
    if not args.once:
        print(
            f"VISION download monitor (refresh every {args.interval:g}s, "
            "Ctrl+C to exit)"
        )
        print(f"  dataset: {VISION_ROOT}")
        print(f"  status:  {STATUS_PATH}")
        if args.total > 0:
            print(f"  total:   {args.total} (fixed via --total)")
        else:
            print(
                f"  total:   {estimated_total} (estimated: {args.devices} devices x "
                f"{len(POOLS)} pools x {args.per_pool} per-pool; override with --total)"
            )

    samples = deque()
    monitor_started = time.monotonic()
    window = max(args.window, 2.0 * args.interval)
    last_done = None
    last_progress = time.monotonic()
    last_line_len = 0

    try:
        while True:
            now = time.monotonic()
            state = collect_state(estimated_total=estimated_total)
            done, total = state["done"], state["total"]

            if last_done is not None and done < last_done:
                samples.clear()  # a fresh run reset the counters
            if last_done is not None and done > last_done:
                last_progress = now
            last_done = done

            samples.append((now, done))
            while samples and now - samples[0][0] > window:
                samples.popleft()
            rate = None
            if len(samples) >= 2 and now > samples[0][0]:
                gain = done - samples[0][1]
                if gain > 0:
                    rate = gain / (now - samples[0][0])
            if rate is None:
                rate = state["avg_rate"]

            elapsed = (
                state["elapsed"] if state["elapsed"] is not None
                else now - monitor_started
            )
            remaining = max(total - done, 0)
            eta = remaining / rate if rate and rate > 0 else None
            percent = 100.0 * done / total if total > 0 else 0.0
            stalled_for = now - last_progress

            line = build_line(
                state=state,
                percent=percent,
                remaining=remaining,
                rate=rate,
                eta=eta,
                elapsed=elapsed,
                stalled_for=stalled_for,
                bar_width=args.bar_width,
                stall_warn=args.stall_warn,
            )
            if args.once:
                print(line)
            else:
                max_width = shutil.get_terminal_size(fallback=(100, 20)).columns - 1
                line = line[:max_width]
                print("\r" + line.ljust(last_line_len), end="", flush=True)
                last_line_len = len(line)

            if state["phase"] in ("done", "aborted") or (total > 0 and done >= total):
                if not args.once:
                    print()
                failed_part = f", {state['failed']} failed" if state["failed"] else ""
                verdict = (
                    "Download complete"
                    if state["phase"] != "aborted"
                    else "Downloader was interrupted"
                )
                print(
                    f"{verdict}: {done}/{total} images{failed_part} "
                    f"(elapsed {format_duration(seconds=elapsed)})."
                )
                return
            if args.once:
                return
            time.sleep(args.interval)
    except KeyboardInterrupt:
        if not args.once:
            print()
        print("Monitor stopped.")


if __name__ == "__main__":
    main()

