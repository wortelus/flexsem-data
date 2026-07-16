#!/usr/bin/env python3
"""Reconstruct hysteresis dataset records from corrector run logs.

I used this for run70, where the .jsonl files didn't generate because of a crash, so I had to reconstruct
the points from a log...

The current writer appends one JSON object per line, despite the .jsonl suffix.
Some old/broken files may contain a JSON array, so this script can write both.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

try:
    from zoneinfo import ZoneInfo
    from zoneinfo import ZoneInfoNotFoundError
except ImportError:  # pragma: no cover
    ZoneInfo = None
    ZoneInfoNotFoundError = None


LOG_TS_RE = re.compile(r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})")
BASE_RE = re.compile(
    r"Base position: Distance\(nanometers=(?P<x>-?\d+)\), Distance\(nanometers=(?P<y>-?\d+)\)"
)
MOVED_BASE_RE = re.compile(
    r"Moved to base position for experiment: x=(?P<x>-?\d+), y=(?P<y>-?\d+)"
)
ITER_RE = re.compile(
    r"--- Starting Iteration (?P<iteration>\d+) at Base: \((?P<x>-?\d+), (?P<y>-?\d+)\) ---"
)
STEP_RE = re.compile(
    r"Step (?P<step>\d+)/(?P<total>\d+): Moving to target x=(?P<x>-?\d+), y=(?P<y>-?\d+)"
)
DRIFT_RE = re.compile(
    r"Drift detected: \(dx=(?P<dx>-?\d+), dy=(?P<dy>-?\d+)\) with conf=(?P<conf>[0-9.]+)"
)
FINISHED_RE = re.compile(r"Trajectory '(?P<experiment>[^']+)' finished\. Returning to base position\.")
DATASET_RE = re.compile(r"Dataset will be saved to (?P<name>\S+)")
IMG_EXPERIMENT_RE = re.compile(
    r"temp[\\/]\d{5}_(?P<experiment>.+?)_(?:\d{3,}|fin)_[0-9.]+\.bmp"
)


@dataclass
class PendingStep:
    iteration: int
    step: int
    total: int | None
    x_target: int
    y_target: int
    ts: datetime
    final: bool = False


def parse_log_timestamp(line: str, tz_name: str | None) -> datetime | None:
    match = LOG_TS_RE.search(line)
    if not match:
        return None
    dt = datetime.strptime(match.group("ts"), "%Y-%m-%d %H:%M:%S,%f")
    if tz_name:
        if ZoneInfo is None:
            raise RuntimeError("zoneinfo is not available in this Python build")
        try:
            tz = ZoneInfo(tz_name)
        except Exception as exc:
            if ZoneInfoNotFoundError is not None and isinstance(exc, ZoneInfoNotFoundError):
                tz = fallback_timezone(tz_name, dt)
            else:
                raise
        dt = dt.replace(tzinfo=tz)
    return dt


def fallback_timezone(tz_name: str, dt: datetime):
    if tz_name not in {"Europe/Prague", "CET", "CEST"}:
        raise RuntimeError(
            f"Timezone '{tz_name}' is unavailable. Install tzdata or omit --timezone."
        )
    if tz_name == "CET":
        return timezone(timedelta(hours=1), "CET")
    if tz_name == "CEST":
        return timezone(timedelta(hours=2), "CEST")

    # Windows Python often lacks the IANA tz database. This covers Europe/Prague
    # well enough for local log timestamps: CET outside DST, CEST inside DST.
    dst_start = last_sunday(dt.year, 3).replace(hour=2, minute=0, second=0, microsecond=0)
    dst_end = last_sunday(dt.year, 10).replace(hour=3, minute=0, second=0, microsecond=0)
    if dst_start <= dt < dst_end:
        return timezone(timedelta(hours=2), "CEST")
    return timezone(timedelta(hours=1), "CET")


def last_sunday(year: int, month: int) -> datetime:
    if month == 12:
        day = datetime(year + 1, 1, 1) - timedelta(days=1)
    else:
        day = datetime(year, month + 1, 1) - timedelta(days=1)
    while day.weekday() != 6:
        day -= timedelta(days=1)
    return day


def to_epoch(dt: datetime) -> float:
    return dt.timestamp()


def clean_timestamp(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".")


def first_timestamp(path: Path, tz_name: str | None) -> datetime:
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            ts = parse_log_timestamp(line, tz_name)
            if ts is not None:
                return ts
    return datetime.max


def normalize_log_dir(path: Path) -> Path:
    if path.is_file():
        return path
    original_logs = path / "logs" / "original"
    if original_logs.is_dir():
        return original_logs
    logs = path / "logs"
    if logs.is_dir():
        return logs
    return path


def collect_log_files(paths: Iterable[Path], tz_name: str | None) -> list[Path]:
    files: list[Path] = []
    for raw_path in paths:
        path = normalize_log_dir(raw_path)
        if path.is_file():
            files.append(path)
            continue
        files.extend(p for p in path.glob("last.log*") if p.is_file())

    unique = sorted(set(files), key=lambda p: (first_timestamp(p, tz_name), str(p)))
    if not unique:
        raise FileNotFoundError("No last.log* files found in the requested path(s)")
    return unique


def infer_default_output(input_path: Path) -> Path:
    path = normalize_log_dir(input_path)
    if path.is_file():
        return path.with_name("hysteresis_dataset_reconstructed.jsonl")
    if path.name == "original" and path.parent.name == "logs":
        return path.parent.parent / "hysteresis_dataset_reconstructed.jsonl"
    if path.name == "logs":
        return path.parent / "hysteresis_dataset_reconstructed.jsonl"
    return path / "hysteresis_dataset_reconstructed.jsonl"


def make_img_path(
    *,
    iteration: int,
    experiment_name: str,
    step: int,
    timestamp: float,
    final: bool,
    img_path_mode: str,
    image_ext: str,
) -> str | None:
    if img_path_mode == "none":
        return None
    if final:
        suffix = "fin"
    elif step == 0:
        suffix = "000_reference"
    else:
        suffix = f"{step:03d}"
    return f"temp/{iteration:05d}_{experiment_name}_{suffix}_{clean_timestamp(timestamp)}.{image_ext}"


def build_record(
    *,
    pending: PendingStep,
    base_x: int,
    base_y: int,
    dx: int,
    dy: int,
    conf: float,
    experiment_name: str,
    timestamp: float,
    img_path_mode: str,
    image_ext: str,
) -> dict:
    record = {
        "timestamp": timestamp,
        "experiment_name": experiment_name,
        "iteration": pending.iteration,
        "step": pending.step,
        "x_target_abs": pending.x_target,
        "y_target_abs": pending.y_target,
        "x_actual_abs": base_x - dx,
        "y_actual_abs": base_y - dy,
        "confidence": f"{conf:.4f}",
    }
    img_path = make_img_path(
        iteration=pending.iteration,
        experiment_name=experiment_name,
        step=pending.step,
        timestamp=timestamp,
        final=pending.final,
        img_path_mode=img_path_mode,
        image_ext=image_ext,
    )
    if img_path_mode != "omit":
        record["img_path"] = img_path
    return record


def add_reference_record(
    *,
    records: dict[tuple[int, int], dict],
    iteration: int,
    base_x: int,
    base_y: int,
    experiment_name: str,
    timestamp: float,
    img_path_mode: str,
    image_ext: str,
) -> None:
    if (iteration, 0) in records:
        return
    record = {
        "timestamp": timestamp,
        "experiment_name": experiment_name,
        "iteration": iteration,
        "step": 0,
        "x_target_abs": base_x,
        "y_target_abs": base_y,
        "x_actual_abs": base_x,
        "y_actual_abs": base_y,
        "confidence": "1.0000",
    }
    img_path = make_img_path(
        iteration=iteration,
        experiment_name=experiment_name,
        step=0,
        timestamp=timestamp,
        final=False,
        img_path_mode=img_path_mode,
        image_ext=image_ext,
    )
    if img_path_mode != "omit":
        record["img_path"] = img_path
    records[(iteration, 0)] = record


def reconstruct(args: argparse.Namespace) -> tuple[list[dict], dict]:
    log_files = collect_log_files([Path(p) for p in args.inputs], args.timezone)
    base_x = args.base_x
    base_y = args.base_y
    current_iteration = args.iteration
    experiment_name = args.experiment_name
    dataset_name = None
    first_step_ts: datetime | None = None
    moved_base_ts: datetime | None = None
    pending: PendingStep | None = None
    records: dict[tuple[int, int], dict] = {}
    duplicates = 0
    missing_drift = 0
    inferred_experiment_names: list[str] = []

    for log_file in log_files:
        with log_file.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                ts = parse_log_timestamp(line, args.timezone)
                if ts is None:
                    continue

                if match := DATASET_RE.search(line):
                    dataset_name = match.group("name")

                if match := IMG_EXPERIMENT_RE.search(line):
                    inferred_experiment_names.append(match.group("experiment"))

                if match := BASE_RE.search(line):
                    base_x = int(match.group("x"))
                    base_y = int(match.group("y"))

                if match := ITER_RE.search(line):
                    current_iteration = int(match.group("iteration"))
                    base_x = int(match.group("x"))
                    base_y = int(match.group("y"))

                if match := MOVED_BASE_RE.search(line):
                    moved_base_ts = ts
                    if base_x is None:
                        base_x = int(match.group("x"))
                    if base_y is None:
                        base_y = int(match.group("y"))

                if match := FINISHED_RE.search(line):
                    if pending is not None:
                        missing_drift += 1
                    if base_x is None or base_y is None:
                        pending = None
                    else:
                        last_step = max((step for _, step in records), default=0)
                        pending = PendingStep(
                            iteration=current_iteration,
                            step=last_step + 1,
                            total=None,
                            x_target=base_x,
                            y_target=base_y,
                            ts=ts,
                            final=True,
                        )
                    if experiment_name is None:
                        experiment_name = match.group("experiment")
                    continue

                if match := STEP_RE.search(line):
                    if pending is not None:
                        missing_drift += 1
                    step = int(match.group("step"))
                    if first_step_ts is None:
                        first_step_ts = ts
                    if base_x is None and step == 1:
                        base_x = int(match.group("x"))
                    if base_y is None and step == 1:
                        base_y = int(match.group("y"))
                    pending = PendingStep(
                        iteration=current_iteration,
                        step=step,
                        total=int(match.group("total")),
                        x_target=int(match.group("x")),
                        y_target=int(match.group("y")),
                        ts=ts,
                    )
                    continue

                if match := DRIFT_RE.search(line):
                    if pending is None:
                        continue
                    if base_x is None or base_y is None:
                        raise ValueError(
                            "Base position is missing. Pass --base-x and --base-y "
                            "or include the beginning of the run log."
                        )
                    if experiment_name is None and inferred_experiment_names:
                        experiment_name = inferred_experiment_names[-1]
                    if experiment_name is None:
                        experiment_name = f"random_walk_{ts.date().isoformat()}"

                    record = build_record(
                        pending=pending,
                        base_x=base_x,
                        base_y=base_y,
                        dx=int(match.group("dx")),
                        dy=int(match.group("dy")),
                        conf=float(match.group("conf")),
                        experiment_name=experiment_name,
                        timestamp=to_epoch(ts),
                        img_path_mode=args.img_path_mode,
                        image_ext=args.image_ext,
                    )
                    key = (pending.iteration, pending.step)
                    if key in records and records[key] != record:
                        duplicates += 1
                    records.setdefault(key, record)
                    pending = None

    if base_x is not None and base_y is not None and args.include_reference:
        min_step = min((step for _, step in records), default=None)
        if min_step is not None and min_step <= 1:
            if experiment_name is None:
                experiment_name = "random_walk_unknown"
            ref_ts = to_epoch(first_step_ts or moved_base_ts or first_timestamp(log_files[0], args.timezone))
            add_reference_record(
                records=records,
                iteration=current_iteration,
                base_x=base_x,
                base_y=base_y,
                experiment_name=experiment_name,
                timestamp=ref_ts,
                img_path_mode=args.img_path_mode,
                image_ext=args.image_ext,
            )

    if pending is not None:
        missing_drift += 1

    ordered_records = [records[key] for key in sorted(records)]
    stats = {
        "base_x": base_x,
        "base_y": base_y,
        "dataset_name": dataset_name,
        "duplicates": duplicates,
        "log_files": [str(p) for p in log_files],
        "missing_drift": missing_drift,
        "records": len(ordered_records),
        "first_step": ordered_records[0]["step"] if ordered_records else None,
        "last_step": ordered_records[-1]["step"] if ordered_records else None,
        "experiment_name": experiment_name,
    }
    return ordered_records, stats


def write_records(path: Path, records: list[dict], output_format: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        if output_format == "json-array":
            json.dump(records, fh, indent=2)
            fh.write("\n")
        else:
            for record in records:
                fh.write(json.dumps(record) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reconstruct hysteresis JSONL records from corrector run logs."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Run directory, logs directory, or individual last.log* file(s).",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output path. Defaults to hysteresis_dataset_reconstructed.jsonl near the input logs.",
    )
    parser.add_argument(
        "--format",
        choices=("jsonl", "json-array"),
        default="jsonl",
        help="Write one JSON object per line or a single JSON array.",
    )
    parser.add_argument("--experiment-name", help="Override experiment_name.")
    parser.add_argument("--iteration", type=int, default=1, help="Default iteration when logs do not state it.")
    parser.add_argument("--base-x", type=int, help="Override or provide base X in nanometers.")
    parser.add_argument("--base-y", type=int, help="Override or provide base Y in nanometers.")
    parser.add_argument(
        "--timezone",
        help="Timezone for log timestamps, for example Europe/Prague. Defaults to local time.",
    )
    parser.add_argument(
        "--img-path-mode",
        choices=("approximate", "none", "omit"),
        default="approximate",
        help="Reconstruct approximate img_path, set it to null, or omit the key.",
    )
    parser.add_argument("--image-ext", default="bmp", help="Image extension used in reconstructed img_path.")
    parser.add_argument(
        "--no-reference",
        dest="include_reference",
        action="store_false",
        help="Do not synthesize the step 0 reference record.",
    )
    parser.set_defaults(include_reference=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if (args.base_x is None) != (args.base_y is None):
        parser.error("--base-x and --base-y must be provided together")

    records, stats = reconstruct(args)
    output = Path(args.output) if args.output else infer_default_output(Path(args.inputs[0]))
    write_records(output, records, args.format)

    print(f"Wrote {stats['records']} records to {output}")
    print(f"Experiment: {stats['experiment_name']}")
    print(f"Base: x={stats['base_x']}, y={stats['base_y']}")
    print(f"Steps: {stats['first_step']}..{stats['last_step']}")
    if stats["dataset_name"]:
        print(f"Dataset mentioned in logs: {stats['dataset_name']}")
    if stats["duplicates"]:
        print(f"Warning: ignored {stats['duplicates']} conflicting duplicate step(s)", file=sys.stderr)
    if stats["missing_drift"]:
        print(f"Warning: {stats['missing_drift']} step(s) had no following drift record", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
