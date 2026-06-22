"""Split an experiment dataset into contiguous high-confidence JSONL segments.

The confidence used for splitting comes from ``positions.json``.  It is not
the ``confidence`` field already present in experiment records, which was
produced by a different measurement.

Both historical JSON arrays and actual JSONL files are accepted as input.
Every produced segment is actual JSONL: one complete JSON object per line.
JSONL has no comment syntax, so the position confidence is stored in each
record as ``position_confidence`` and run metadata is written to
``manifest.json``.

Example:
    python split_by_position_confidence.py experiment.jsonl positions.json \
        --min-confidence 0.8 --output-dir experiment_segments
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any


DEFAULT_MIN_CONFIDENCE = 0.8
DEFAULT_CONFIDENCE_FIELD = "position_confidence"


def load_experiment(path: Path) -> tuple[list[dict[str, Any]], str]:
    """Load either a JSON array/object or one-object-per-line JSONL."""
    text = path.read_text(encoding="utf-8-sig").strip()
    if not text:
        return [], "empty"

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        records: list[dict[str, Any]] = []
        for line_number, line in enumerate(text.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL on line {line_number}: {exc}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(
                    f"JSONL line {line_number} is not a JSON object"
                )
            records.append(record)
        return records, "jsonl"

    if isinstance(parsed, list):
        if not all(isinstance(record, dict) for record in parsed):
            raise ValueError("JSON array contains a value that is not an object")
        return parsed, "json-array"
    if isinstance(parsed, dict):
        return [parsed], "json-object"
    raise ValueError("Experiment input must contain JSON objects")


def load_positions(path: Path) -> dict[str, dict[str, Any]]:
    try:
        positions = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid positions JSON: {exc}") from exc
    if not isinstance(positions, dict):
        raise ValueError("positions.json must contain an object keyed by filename")
    return positions


def image_basename(record: dict[str, Any], index: int) -> str:
    try:
        image_path = str(record["img_path"])
    except KeyError as exc:
        raise ValueError(f"Record {index} has no img_path") from exc
    # Accept both slash styles regardless of the operating system.
    return os.path.basename(image_path.replace("\\", "/"))


def position_confidence(
    record: dict[str, Any],
    index: int,
    positions: dict[str, dict[str, Any]],
) -> tuple[str, float | None]:
    filename = image_basename(record, index)
    position = positions.get(filename)
    if position is None:
        return filename, None
    try:
        return filename, float(position["confidence"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Position for {filename!r} has no numeric confidence"
        ) from exc


def build_segments(
    records: list[dict[str, Any]],
    positions: dict[str, dict[str, Any]],
    min_confidence: float,
    confidence_field: str | None,
) -> tuple[list[list[dict[str, Any]]], list[dict[str, Any]]]:
    segments: list[list[dict[str, Any]]] = []
    discarded: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []

    for index, source_record in enumerate(records):
        filename, confidence = position_confidence(source_record, index, positions)
        if confidence is None or confidence < min_confidence:
            if current:
                segments.append(current)
                current = []
            discarded.append(
                {
                    "source_index": index,
                    "step": source_record.get("step"),
                    "img_path": source_record.get("img_path"),
                    "filename": filename,
                    "position_confidence": confidence,
                    "reason": "missing-position" if confidence is None else "below-threshold",
                }
            )
            continue

        record = dict(source_record)
        if confidence_field is not None:
            record[confidence_field] = confidence
        current.append(record)

    if current:
        segments.append(current)

    return segments, discarded


def safe_label(value: Any) -> str:
    if isinstance(value, int):
        return f"{value:04d}"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def segment_filename(source: Path, part: int, records: list[dict[str, Any]]) -> str:
    first_step = safe_label(records[0].get("step", "unknown"))
    last_step = safe_label(records[-1].get("step", "unknown"))
    return f"{source.stem}_part{part:03d}_steps{first_step}-{last_step}.jsonl"


def write_jsonl_atomic(path: Path, records: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            stream.write("\n")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path, help="Experiment JSON array or JSONL")
    parser.add_argument("positions", type=Path, help="positions.json")
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=DEFAULT_MIN_CONFIDENCE,
        help=f"Minimum accepted position confidence (default: {DEFAULT_MIN_CONFIDENCE})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <experiment_stem>_segments)",
    )
    parser.add_argument(
        "--min-segment-length",
        type=int,
        default=1,
        help="Do not write segments shorter than this many records (default: 1)",
    )
    parser.add_argument(
        "--confidence-field",
        default=DEFAULT_CONFIDENCE_FIELD,
        help=(
            "Field added to every output record; use an empty value to disable "
            f"(default: {DEFAULT_CONFIDENCE_FIELD})"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing existing segment files and manifest.json",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned split without writing files",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0.0 <= args.min_confidence <= 1.0:
        print("ERROR: --min-confidence must be between 0 and 1", file=sys.stderr)
        return 2
    if args.min_segment_length < 1:
        print("ERROR: --min-segment-length must be at least 1", file=sys.stderr)
        return 2

    try:
        records, input_format = load_experiment(args.experiment)
        positions = load_positions(args.positions)
        confidence_field = args.confidence_field or None
        all_segments, discarded = build_segments(
            records,
            positions,
            args.min_confidence,
            confidence_field,
        )
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    kept_segments = [
        segment
        for segment in all_segments
        if len(segment) >= args.min_segment_length
    ]
    short_segments = [
        segment
        for segment in all_segments
        if len(segment) < args.min_segment_length
    ]
    output_dir = args.output_dir or args.experiment.with_name(
        f"{args.experiment.stem}_segments"
    )

    plans: list[dict[str, Any]] = []
    for part, segment in enumerate(kept_segments, 1):
        filename = segment_filename(args.experiment, part, segment)
        plans.append(
            {
                "file": filename,
                "records": len(segment),
                "first_step": segment[0].get("step"),
                "last_step": segment[-1].get("step"),
            }
        )

    print(f"Input format: {input_format}")
    print(f"Input records: {len(records)}")
    print(f"Position entries: {len(positions)}")
    print(f"Minimum confidence: {args.min_confidence}")
    print(f"Discarded records: {len(discarded)}")
    print(f"Contiguous high-confidence segments: {len(all_segments)}")
    print(f"Segments to write: {len(kept_segments)}")
    print(f"Records to write: {sum(len(segment) for segment in kept_segments)}")
    if short_segments:
        print(
            f"Short segments omitted: {len(short_segments)} "
            f"({sum(len(segment) for segment in short_segments)} records)"
        )
    print(f"Output directory: {output_dir}")
    for plan in plans:
        print(
            f"  {plan['file']}: {plan['records']} records "
            f"(steps {plan['first_step']}..{plan['last_step']})"
        )

    if args.dry_run:
        print("Dry run: no files written.")
        return 0

    manifest_path = output_dir / "manifest.json"
    targets = [output_dir / plan["file"] for plan in plans] + [manifest_path]
    existing = [path for path in targets if path.exists()]
    if existing and not args.overwrite:
        print(
            "ERROR: output files already exist; use --overwrite to replace them:\n  "
            + "\n  ".join(str(path) for path in existing),
            file=sys.stderr,
        )
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    for plan, segment in zip(plans, kept_segments):
        write_jsonl_atomic(output_dir / plan["file"], segment)

    manifest = {
        "source_experiment": str(args.experiment),
        "source_positions": str(args.positions),
        "source_format": input_format,
        "min_confidence": args.min_confidence,
        "confidence_field": confidence_field,
        "min_segment_length": args.min_segment_length,
        "input_records": len(records),
        "position_entries": len(positions),
        "written_records": sum(len(segment) for segment in kept_segments),
        "discarded_records": discarded,
        "omitted_short_segments": [
            {
                "records": len(segment),
                "first_step": segment[0].get("step"),
                "last_step": segment[-1].get("step"),
            }
            for segment in short_segments
        ],
        "segments": plans,
    }
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary_manifest.replace(manifest_path)
    print(f"Wrote {len(kept_segments)} segments and {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
