"""Split run34 around confirmed bad reconstructed-actual positions.

The original experiment is preserved.  Each contiguous range between rejected
records is written as a separate JSONL file together with a manifest.  Short
ranges are intentionally retained in the output, but the manifest marks those
that cannot produce a window for the configured sequence length.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    REPO_ROOT
    / "data_original"
    / "run34-random678mag"
    / "hysteresis_dataset_20251114_102017.jsonl"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "data_original"
    / "run34-random678mag"
    / "actual_outlier_segments"
)
SEQUENCE_LENGTH = 16
OFFSET_THRESHOLD_NM = 10_000.0
CONFIDENCE_SANITY_MAX = 0.4
BAD_STEPS_BY_ITERATION = {1: {4, 84, 88, 218, 426, 450, 458}}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace files already present in the output directory.",
    )
    return parser.parse_args()


def load_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8-sig").strip()
    if not text:
        raise ValueError(f"Source is empty: {path}")

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = [json.loads(line) for line in text.splitlines() if line.strip()]

    if not isinstance(parsed, list) or not all(
        isinstance(record, dict) for record in parsed
    ):
        raise ValueError("Source must contain a JSON array or JSONL objects")
    return parsed


def numeric_step(record: dict[str, Any]) -> int | None:
    value = record.get("step")
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def actual_target_offset_nm(record: dict[str, Any]) -> float:
    dx = float(record["x_actual_abs"]) - float(record["x_target_abs"])
    dy = float(record["y_actual_abs"]) - float(record["y_target_abs"])
    return math.hypot(dx, dy)


def is_configured_bad(record: dict[str, Any]) -> bool:
    iteration = int(record["iteration"])
    step = numeric_step(record)
    return step in BAD_STEPS_BY_ITERATION.get(iteration, set())


def validate_bad_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    configured = [record for record in records if is_configured_bad(record)]
    configured_keys = {
        (int(record["iteration"]), numeric_step(record)) for record in configured
    }
    expected_keys = {
        (iteration, step)
        for iteration, steps in BAD_STEPS_BY_ITERATION.items()
        for step in steps
    }
    if configured_keys != expected_keys:
        raise ValueError(
            "Configured bad records do not match source: "
            f"expected={sorted(expected_keys)}, found={sorted(configured_keys)}"
        )

    threshold_keys = {
        (int(record["iteration"]), numeric_step(record))
        for record in records
        if actual_target_offset_nm(record) > OFFSET_THRESHOLD_NM
    }
    if threshold_keys != expected_keys:
        raise ValueError(
            f"Offset > {OFFSET_THRESHOLD_NM:g} nm no longer selects exactly the "
            f"configured records: {sorted(threshold_keys)}"
        )

    for record in configured:
        confidence = float(record["confidence"])
        if confidence >= CONFIDENCE_SANITY_MAX:
            raise ValueError(
                "Configured bad record no longer satisfies confidence sanity "
                f"check: iteration={record['iteration']} step={record['step']} "
                f"confidence={confidence}"
            )
    return configured


def split_records(
    records: list[dict[str, Any]],
) -> list[tuple[int, list[dict[str, Any]]]]:
    chunks: list[tuple[int, list[dict[str, Any]]]] = []
    current: list[dict[str, Any]] = []
    current_iteration: int | None = None

    def finish_current() -> None:
        nonlocal current
        if current:
            assert current_iteration is not None
            chunks.append((current_iteration, current))
            current = []

    for record in records:
        iteration = int(record["iteration"])
        if current_iteration is None:
            current_iteration = iteration
        elif iteration != current_iteration:
            finish_current()
            current_iteration = iteration

        if is_configured_bad(record):
            finish_current()
            continue
        current.append(record)

    finish_current()
    return chunks


def step_token(value: Any) -> str:
    step = numeric_step({"step": value})
    return f"{step:04d}" if step is not None else str(value)


def write_jsonl_atomic(path: Path, records: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(
                json.dumps(record, ensure_ascii=False, separators=(",", ":"))
            )
            stream.write("\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output_dir = args.output_dir.resolve()
    records = load_records(source)
    bad_records = validate_bad_records(records)
    chunks = split_records(records)

    output_dir.mkdir(parents=True, exist_ok=True)
    existing = list(output_dir.iterdir())
    if existing and not args.overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. Use --overwrite."
        )

    expected_names: set[str] = set()
    manifest_chunks: list[dict[str, Any]] = []
    parts_by_iteration: dict[int, int] = {}
    for iteration, chunk in chunks:
        parts_by_iteration[iteration] = parts_by_iteration.get(iteration, 0) + 1
        part = parts_by_iteration[iteration]
        start = step_token(chunk[0].get("step"))
        end = step_token(chunk[-1].get("step"))
        filename = (
            f"run34_iteration{iteration}_part{part:03d}_steps{start}-{end}.jsonl"
        )
        expected_names.add(filename)
        output_path = output_dir / filename
        write_jsonl_atomic(output_path, chunk)
        generated_windows = max(0, len(chunk) - SEQUENCE_LENGTH + 1)
        manifest_chunks.append(
            {
                "file": filename,
                "iteration": iteration,
                "part": part,
                "start_step": chunk[0].get("step"),
                "end_step": chunk[-1].get("step"),
                "rows": len(chunk),
                "generated_windows_seq16": generated_windows,
                "usable_for_seq16": generated_windows > 0,
                "sha256": sha256(output_path),
            }
        )

    manifest_name = "manifest.json"
    expected_names.add(manifest_name)
    if args.overwrite:
        for existing_path in output_dir.iterdir():
            if existing_path.is_file() and existing_path.name not in expected_names:
                existing_path.unlink()

    excluded = []
    for record in bad_records:
        excluded.append(
            {
                "iteration": int(record["iteration"]),
                "step": numeric_step(record),
                "confidence": float(record["confidence"]),
                "actual_target_offset_nm": actual_target_offset_nm(record),
            }
        )

    manifest = {
        "version": 1,
        "source": os.path.relpath(source, output_dir),
        "source_sha256": sha256(source),
        "sequence_length": SEQUENCE_LENGTH,
        "rule": (
            "Remove the seven confirmed reconstructed-actual outliers; each "
            "removed record creates a hard trajectory boundary."
        ),
        "verification": {
            "actual_target_offset_threshold_nm": OFFSET_THRESHOLD_NM,
            "confidence_sanity_max": CONFIDENCE_SANITY_MAX,
        },
        "source_rows": len(records),
        "excluded_rows": len(excluded),
        "output_rows": sum(chunk["rows"] for chunk in manifest_chunks),
        "excluded_records": excluded,
        "chunks": manifest_chunks,
    }
    manifest_path = output_dir / manifest_name
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"Source rows:   {len(records)}")
    print(f"Excluded rows: {len(excluded)}")
    print(f"Output rows:   {manifest['output_rows']}")
    print(f"Output chunks: {len(manifest_chunks)}")
    for chunk in manifest_chunks:
        status = "usable" if chunk["usable_for_seq16"] else "too short"
        print(
            f"  {chunk['file']}: rows={chunk['rows']}, "
            f"windows={chunk['generated_windows_seq16']} [{status}]"
        )
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
