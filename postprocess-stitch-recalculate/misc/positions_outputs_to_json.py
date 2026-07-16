#!/usr/bin/env python3
"""Convert copied composite_positions.py output into positions.json.

I used this when I ran multiple composite_positions.py in pararell and the positions.json got overwritten...
Not needed for normal pipeline ops :)

Default mode is tailored for run70:

    python positions_outputs_to_json.py
    python positions_outputs_to_json.py --write

It reads positions_1, positions_2, positions_3, detects outputs that say
"Saved: run70-subN_stitched...", filters entries to the BMP files present in
that sub-run temp directory, and writes:

    data_original/run70-data-feast-overnight-subN/positions.json

Generic single-output conversion:

    python positions_outputs_to_json.py positions_2 --output positions.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


POSITION_RE = re.compile(
    r'^\s*"(?P<name>[^"]+)": '
    r"\((?P<x>-?\d+), (?P<y>-?\d+)\),\s*"
    r"# (?P<w>\d+)x(?P<h>\d+), conf=(?P<confidence>[0-9.]+)\s*$"
)
SAVED_RE = re.compile(r"Saved:\s*(?P<path>.+)$")
SUB_RE = re.compile(r"run70-sub(?P<sub>[0-2])", re.IGNORECASE)


@dataclass(frozen=True)
class ParsedOutput:
    path: Path
    entries: "OrderedDict[str, dict[str, int | float]]"
    saved_paths: tuple[str, ...]


def parse_positions_output(path: Path) -> ParsedOutput:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    saved_paths = tuple(
        match.group("path").strip()
        for line in lines
        if (match := SAVED_RE.search(line))
    )

    starts = [i for i, line in enumerate(lines) if line.strip() == "positions = {"]
    if not starts:
        raise ValueError(f"{path}: missing 'positions = {{' block")

    entries: "OrderedDict[str, dict[str, int | float]]" = OrderedDict()
    bad_lines: list[str] = []
    for line in lines[starts[-1] + 1 :]:
        stripped = line.strip()
        if stripped == "}":
            break
        if not stripped:
            continue
        match = POSITION_RE.match(line)
        if not match:
            bad_lines.append(line)
            continue
        entries[match.group("name")] = {
            "x": int(match.group("x")),
            "y": int(match.group("y")),
            "w": int(match.group("w")),
            "h": int(match.group("h")),
            "confidence": float(match.group("confidence")),
        }

    if bad_lines:
        sample = "\n".join(bad_lines[:5])
        raise ValueError(f"{path}: {len(bad_lines)} unparsable dict lines, first lines:\n{sample}")
    if not entries:
        raise ValueError(f"{path}: parsed positions dict is empty")

    return ParsedOutput(path=path, entries=entries, saved_paths=saved_paths)


def write_json(path: Path, entries: "OrderedDict[str, dict[str, int | float]]") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(entries, handle, indent=2)
        handle.write("\n")


def expected_bmps(temp_dir: Path) -> set[str]:
    if not temp_dir.is_dir():
        raise ValueError(f"Missing temp directory: {temp_dir}")
    return {path.name for path in temp_dir.glob("*.bmp")}


def filter_entries(
    parsed: ParsedOutput,
    names: set[str],
) -> tuple["OrderedDict[str, dict[str, int | float]]", set[str], set[str]]:
    filtered = OrderedDict((name, info) for name, info in parsed.entries.items() if name in names)
    missing = names - set(filtered)
    extra = set(parsed.entries) - names
    return filtered, missing, extra


def detect_run70_sub(parsed: ParsedOutput) -> str | None:
    for saved_path in parsed.saved_paths:
        match = SUB_RE.search(saved_path)
        if match:
            return f"sub{match.group('sub')}"
    return None


def parse_manual_maps(values: Iterable[str]) -> dict[str, Path]:
    maps: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid --map value {value!r}; use subN=positions_file")
        sub, path = value.split("=", 1)
        sub = sub.strip().lower()
        if sub not in {"sub0", "sub1", "sub2"}:
            raise ValueError(f"Invalid sub-run {sub!r}; expected sub0, sub1, or sub2")
        maps[sub] = Path(path.strip())
    return maps


def merge_entries(
    parsed_outputs: Iterable[ParsedOutput],
    conflict: str,
) -> "OrderedDict[str, dict[str, int | float]]":
    merged: "OrderedDict[str, dict[str, int | float]]" = OrderedDict()
    source_by_name: dict[str, Path] = {}
    for parsed in parsed_outputs:
        for name, info in parsed.entries.items():
            if name not in merged:
                merged[name] = info
                source_by_name[name] = parsed.path
                continue
            if merged[name] == info:
                continue
            if conflict == "error":
                raise ValueError(
                    f"Conflicting entry for {name}: {source_by_name[name]} and {parsed.path}"
                )
            if conflict == "first":
                continue
            if conflict == "last":
                merged[name] = info
                source_by_name[name] = parsed.path
                continue
            if conflict == "best-confidence":
                if float(info["confidence"]) > float(merged[name]["confidence"]):
                    merged[name] = info
                    source_by_name[name] = parsed.path
                continue
            raise AssertionError(conflict)
    return merged


def run_single_output_mode(args: argparse.Namespace, parsed_outputs: list[ParsedOutput]) -> int:
    entries = merge_entries(parsed_outputs, args.conflict)

    if args.filter_dir is not None:
        names = expected_bmps(args.filter_dir)
        entries = OrderedDict((name, info) for name, info in entries.items() if name in names)
        missing = names - set(entries)
        if missing:
            print(f"ERROR: {args.filter_dir} has {len(missing)} BMP files missing from parsed output")
            print("First missing:", ", ".join(sorted(missing)[:5]))
            return 2

    if args.dry_run:
        print(f"Would write {len(entries)} entries to {args.output}")
    else:
        write_json(args.output, entries)
        print(f"Wrote {len(entries)} entries to {args.output}")
    return 0


def run_run70_mode(args: argparse.Namespace, parsed_outputs: list[ParsedOutput]) -> int:
    by_path = {parsed.path: parsed for parsed in parsed_outputs}
    sub_sources: dict[str, ParsedOutput] = {}

    for parsed in parsed_outputs:
        sub = detect_run70_sub(parsed)
        if sub:
            sub_sources[sub] = parsed

    for sub, path in parse_manual_maps(args.map).items():
        try:
            sub_sources[sub] = by_path[path]
        except KeyError:
            sub_sources[sub] = parse_positions_output(path)

    print("Parsed inputs:")
    for parsed in parsed_outputs:
        detected = detect_run70_sub(parsed) or "not run70-sub detected"
        saved = ", ".join(parsed.saved_paths) if parsed.saved_paths else "no Saved lines"
        print(f"  {parsed.path}: {len(parsed.entries)} entries; {detected}; saved: {saved}")

    outputs: list[tuple[str, Path, OrderedDict[str, dict[str, int | float]], int]] = []
    errors: list[str] = []
    for sub in ("sub0", "sub1", "sub2"):
        run_dir = args.run70_root / f"run70-data-feast-overnight-{sub}"
        temp_dir = run_dir / "temp"
        output_path = run_dir / "positions.json"

        names = expected_bmps(temp_dir)
        parsed = sub_sources.get(sub)
        if parsed is None:
            errors.append(f"{sub}: no source detected; pass --map {sub}=positions_N if needed")
            continue

        filtered, missing, extra = filter_entries(parsed, names)
        if missing:
            errors.append(
                f"{sub}: {parsed.path} is missing {len(missing)} of {len(names)} BMPs "
                f"(first missing: {', '.join(sorted(missing)[:5])})"
            )
            continue
        outputs.append((sub, output_path, filtered, len(extra)))

    if errors and not args.allow_partial:
        print("ERROR: run70 conversion is incomplete, not writing anything.")
        for error in errors:
            print(f"  {error}")
        print("Use --allow-partial to write the complete sub-runs only.")
        return 2

    for error in errors:
        print(f"WARNING: {error}")

    for sub, output_path, entries, extra_count in outputs:
        suffix = f"; ignored {extra_count} extra entries" if extra_count else ""
        if args.dry_run:
            print(f"Would write {len(entries)} entries to {output_path}{suffix}")
        else:
            write_json(output_path, entries)
            print(f"Wrote {len(entries)} entries to {output_path}{suffix}")
    return 0 if not errors else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        default=[Path("positions_1"), Path("positions_2"), Path("positions_3")],
        help="Copied composite_positions.py output files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write a single positions.json instead of run70 sub-run outputs.",
    )
    parser.add_argument(
        "--filter-dir",
        type=Path,
        help="Only include BMP filenames present in this directory. Used with --output.",
    )
    parser.add_argument(
        "--conflict",
        choices=("error", "first", "last", "best-confidence"),
        default="error",
        help="How --output mode handles duplicate filenames with different values.",
    )
    parser.add_argument(
        "--run70-root",
        type=Path,
        default=Path("data_original"),
        help="Parent directory containing run70-data-feast-overnight-subN directories.",
    )
    parser.add_argument(
        "--map",
        action="append",
        default=[],
        metavar="subN=positions_file",
        help="Manually assign an input file to sub0, sub1, or sub2.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="In run70 mode, write complete sub-runs even if another sub-run is missing.",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Actually write files. Without this, the script only reports what it would do.",
    )
    args = parser.parse_args()
    args.dry_run = not args.write

    try:
        parsed_outputs = [parse_positions_output(path) for path in args.inputs]
        if args.output is not None:
            return run_single_output_mode(args, parsed_outputs)
        return run_run70_mode(args, parsed_outputs)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
