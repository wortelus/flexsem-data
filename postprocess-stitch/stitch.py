"""Stitch SEM tiles using commanded positions as priors.

The commanded ``x_target_abs``/``y_target_abs`` coordinates provide the rough
layout.  Image registration only searches close to the relative commanded
position of two overlapping tiles.  Accepted pairwise registrations are then
solved together as one robust pose graph, so an error cannot simply accumulate
from one tile to the next.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from functools import lru_cache

import cv2
import numpy as np


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RUN_DIR = os.path.join(SCRIPT_DIR, "run72-data-feast-overnight-3")
JSONL_PATH = os.path.join(RUN_DIR, "hysteresis_dataset_20260303_203815.jsonl")
IMAGE_BASE_DIR = RUN_DIR
OUTPUT_PATH = os.path.join(RUN_DIR, "run72_stitched.png")

# SEM calibration from the per-image metadata.
NM_PER_PX = 12.40234

# The image overlay occupies the bottom of the source bitmap.
CROP_BOTTOM_PX = 60

# 0 means all records.  Use --max-tiles for a quick smoke test.
MAX_TILES = 0
TILE_STEP = 1

# Commanded positions may be wrong by roughly 2-3 um.  This is the maximum
# pairwise correction searched around the commanded relative displacement.
PAIR_SEARCH_RADIUS_NM = 3000.0
PAIR_NEIGHBORS = 4
PAIR_SAME_DIRECTION_NEIGHBORS = 6
PAIR_RECENT_NEIGHBORS = 2
PAIR_RECENT_WINDOW = 30

# Template matching.  Large predicted overlaps use a large patch; overlaps
# close to an edge automatically use a smaller patch.
TM_METHOD = cv2.TM_CCOEFF_NORMED
TM_TEMPLATE_FRACTION = 0.35
TM_OVERLAP_FRACTION = 0.60
TM_MIN_TEMPLATE_PX = 128
TM_PREPROCESS_MEDIAN_KSIZE = 5
TM_MIN_CONFIDENCE = 0.72
TM_MIN_PEAK_MARGIN = 0.04
TM_PEAK_EXCLUSION_PX = 12
TM_PATCH_AGREEMENT_PX = 2.5
TM_MIN_PATCH_SUPPORT = 2
TM_EARLY_PATCH_SUPPORT = 3
TM_EARLY_ACCEPT_CONFIDENCE = 0.84
TM_EARLY_ACCEPT_MARGIN = 0.06

# Translation-only cycle consistency.  A pairwise match must participate in
# several independently closed image triangles before it can enter the solver.
CYCLE_CLOSURE_PX = 1.5
MIN_EDGE_CYCLE_SUPPORT = 3
MAX_COMPONENT_SEEDS = 80
GROWTH_AGREEMENT_PX = 1.5
GROWTH_MAX_COMMAND_CORRECTION_NM = 6000.0

# Robust global pose-graph solve.
POSE_PRIOR_WEIGHT = 0.02
POSE_ANCHOR_WEIGHT = 2.0
POSE_HUBER_PX = 6.0
POSE_IRLS_ITERATIONS = 5
POSE_CG_ITERATIONS = 600
POSE_CG_TOLERANCE = 1e-7

# A tile is rendered only when it belongs to this mutually supported k-core.
# Every accepted tile must have at least three accepted neighboring tiles whose
# measured pair displacement agrees with the solved graph within 4 px.
MIN_AGREEING_NEIGHBORS = 3
CONSENSUS_EDGE_RESIDUAL_PX = 1.5

# Rendering.
BLEND_MODE = "feather"  # "sharp" is available for seam diagnostics.
BLEND_FEATHER_PX = 64
CANVAS_MARGIN_PX = 32
SAVE_FULL_CANVAS = False
SAVE_CROPPED = True
OUTPUT_FORMAT = ".png"

DEBUG = True
PROGRESS_EVERY = 100


# -----------------------------------------------------------------------------
# Input
# -----------------------------------------------------------------------------

def load_jsonl(path: str) -> list[dict]:
    records: list[dict] = []
    with open(path, "r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"WARNING: skipping JSONL line {line_number}: {exc}")
    return records


def prepare_tile_list(records: list[dict], max_tiles: int) -> tuple[list[dict], int]:
    if not records:
        raise ValueError("No records in JSONL")

    selected = records[::TILE_STEP]
    if max_tiles > 0:
        selected = selected[:max_tiles]

    anchor = selected[0]
    anchor_x_nm = float(anchor["x_target_abs"])
    anchor_y_nm = float(anchor["y_target_abs"])

    tiles: list[dict] = []
    missing = 0
    for record in selected:
        image_path = os.path.join(IMAGE_BASE_DIR, record["img_path"])
        if not os.path.isfile(image_path):
            missing += 1
            continue

        # SEM commanded coordinates: +X moves the image left, +Y down.
        x_px = -(float(record["x_target_abs"]) - anchor_x_nm) / NM_PER_PX
        y_px = (float(record["y_target_abs"]) - anchor_y_nm) / NM_PER_PX
        tiles.append(
            {
                "image_path": image_path,
                "img_path": record["img_path"],
                "x_px": x_px,
                "y_px": y_px,
                "step": record.get("step", 0),
                "iteration": record.get("iteration", 0),
            }
        )

    if not tiles:
        raise ValueError("None of the selected records has an image on disk")
    return tiles, missing


def load_and_crop(path: str) -> np.ndarray | None:
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    if CROP_BOTTOM_PX > 0:
        if image.shape[0] <= CROP_BOTTOM_PX:
            return None
        image = image[:-CROP_BOTTOM_PX, :]
    return image


@lru_cache(maxsize=48)
def load_preprocessed(path: str) -> np.ndarray | None:
    image = load_and_crop(path)
    if image is None:
        return None
    kernel = TM_PREPROCESS_MEDIAN_KSIZE
    if kernel > 1:
        image = cv2.medianBlur(image, kernel)
    return image


# -----------------------------------------------------------------------------
# Pairwise registration
# -----------------------------------------------------------------------------

def _patch_positions(low: float, high: float) -> list[int]:
    """Return center first, then both extremes, without duplicates."""
    low_i = int(round(low))
    high_i = int(round(high))
    if high_i < low_i:
        return []
    center = int(round((low_i + high_i) / 2))
    result: list[int] = []
    for value in (center, low_i, high_i):
        if value not in result:
            result.append(value)
    return result


def _second_peak(result: np.ndarray, peak: tuple[int, int]) -> float:
    if result.size <= 1:
        return -1.0
    suppressed = result.copy()
    x, y = peak
    radius = TM_PEAK_EXCLUSION_PX
    suppressed[
        max(0, y - radius): min(result.shape[0], y + radius + 1),
        max(0, x - radius): min(result.shape[1], x + radius + 1),
    ] = -1.0
    return float(np.max(suppressed))


def _subpixel_peak(result: np.ndarray, peak: tuple[int, int]) -> tuple[float, float]:
    """Refine a correlation maximum with independent 1-D parabolas."""
    x, y = peak

    def parabola(left: float, center: float, right: float) -> float:
        denominator = left - 2.0 * center + right
        if abs(denominator) < 1e-12:
            return 0.0
        offset = 0.5 * (left - right) / denominator
        return float(np.clip(offset, -1.0, 1.0))

    offset_x = 0.0
    offset_y = 0.0
    if 0 < x < result.shape[1] - 1:
        offset_x = parabola(
            float(result[y, x - 1]),
            float(result[y, x]),
            float(result[y, x + 1]),
        )
    if 0 < y < result.shape[0] - 1:
        offset_y = parabola(
            float(result[y - 1, x]),
            float(result[y, x]),
            float(result[y + 1, x]),
        )
    return offset_x, offset_y


def refine_pair_tm(
    ref_image: np.ndarray,
    new_image: np.ndarray,
    expected_delta: tuple[float, float],
) -> dict | None:
    """Measure ``new_position - ref_position`` close to a commanded prior.

    Multiple patches are tried inside the predicted overlap.  Searching around
    the expected patch location makes repeated round features far less likely
    to produce an unrelated, high-correlation match.
    """
    if ref_image.shape != new_image.shape:
        return None

    height, width = ref_image.shape
    expected_dx, expected_dy = expected_delta
    overlap_width = width - abs(expected_dx)
    overlap_height = height - abs(expected_dy)
    if overlap_width < TM_MIN_TEMPLATE_PX or overlap_height < TM_MIN_TEMPLATE_PX:
        return None

    max_patch_width = max(TM_MIN_TEMPLATE_PX, int(width * TM_TEMPLATE_FRACTION))
    max_patch_height = max(TM_MIN_TEMPLATE_PX, int(height * TM_TEMPLATE_FRACTION))
    patch_width = min(
        max_patch_width,
        max(TM_MIN_TEMPLATE_PX, int(overlap_width * TM_OVERLAP_FRACTION)),
    )
    patch_height = min(
        max_patch_height,
        max(TM_MIN_TEMPLATE_PX, int(overlap_height * TM_OVERLAP_FRACTION)),
    )

    # Range of template top-left positions inside the predicted overlap,
    # expressed in new-image coordinates.
    overlap_new_x1 = max(0.0, -expected_dx)
    overlap_new_x2 = min(float(width), width - expected_dx)
    overlap_new_y1 = max(0.0, -expected_dy)
    overlap_new_y2 = min(float(height), height - expected_dy)

    template_xs = _patch_positions(
        overlap_new_x1,
        min(width - patch_width, overlap_new_x2 - patch_width),
    )
    template_ys = _patch_positions(
        overlap_new_y1,
        min(height - patch_height, overlap_new_y2 - patch_height),
    )
    if not template_xs or not template_ys:
        return None

    search_radius = int(math.ceil(PAIR_SEARCH_RADIUS_NM / NM_PER_PX))
    candidates: list[dict] = []

    def combine_agreeing_patches(minimum_support: int) -> dict | None:
        valid = [
            candidate
            for candidate in candidates
            if candidate["score"] >= TM_MIN_CONFIDENCE
            and candidate["peak_margin"] >= TM_MIN_PEAK_MARGIN
        ]
        if not valid:
            return None

        best_cluster: list[dict] = []
        best_quality = -math.inf
        for seed in valid:
            cluster = [
                candidate
                for candidate in valid
                if math.hypot(
                    candidate["dx"] - seed["dx"],
                    candidate["dy"] - seed["dy"],
                ) <= TM_PATCH_AGREEMENT_PX
            ]
            quality = sum(
                candidate["score"] + candidate["peak_margin"]
                for candidate in cluster
            )
            if len(cluster) > len(best_cluster) or (
                len(cluster) == len(best_cluster) and quality > best_quality
            ):
                best_cluster = cluster
                best_quality = quality

        if len(best_cluster) < minimum_support:
            return None
        cluster_weight = np.asarray(
            [
                max(1e-6, candidate["score"] ** 4)
                * max(0.20, candidate["area_ratio"])
                for candidate in best_cluster
            ],
            dtype=np.float64,
        )
        cluster_weight /= np.sum(cluster_weight)
        return {
            "dx": float(np.dot(cluster_weight, [c["dx"] for c in best_cluster])),
            "dy": float(np.dot(cluster_weight, [c["dy"] for c in best_cluster])),
            "score": float(np.median([c["score"] for c in best_cluster])),
            "peak_margin": float(
                np.median([c["peak_margin"] for c in best_cluster])
            ),
            "area_ratio": float(
                np.mean([c["area_ratio"] for c in best_cluster])
            ),
            "patch_support": len(best_cluster),
        }

    for template_y in template_ys:
        for template_x in template_xs:
            expected_ref_x = int(round(template_x + expected_dx))
            expected_ref_y = int(round(template_y + expected_dy))
            search_x1 = max(0, expected_ref_x - search_radius)
            search_y1 = max(0, expected_ref_y - search_radius)
            search_x2 = min(width, expected_ref_x + patch_width + search_radius)
            search_y2 = min(height, expected_ref_y + patch_height + search_radius)
            if search_x2 - search_x1 < patch_width or search_y2 - search_y1 < patch_height:
                continue

            template = new_image[
                template_y: template_y + patch_height,
                template_x: template_x + patch_width,
            ]
            search = ref_image[search_y1:search_y2, search_x1:search_x2]
            result = cv2.matchTemplate(search, template, TM_METHOD)
            _, score, _, peak = cv2.minMaxLoc(result)
            second = _second_peak(result, peak)
            margin = float(score - second)

            subpixel_x, subpixel_y = _subpixel_peak(result, peak)

            relative_x = search_x1 + peak[0] + subpixel_x - template_x
            relative_y = search_y1 + peak[1] + subpixel_y - template_y
            candidate = {
                "dx": float(relative_x),
                "dy": float(relative_y),
                "score": float(score),
                "peak_margin": margin,
                "area_ratio": (patch_width * patch_height)
                / float(max_patch_width * max_patch_height),
            }
            candidates.append(candidate)

            agreed = combine_agreeing_patches(TM_EARLY_PATCH_SUPPORT)
            if (
                agreed is not None
                and agreed["score"] >= TM_EARLY_ACCEPT_CONFIDENCE
                and agreed["peak_margin"] >= TM_EARLY_ACCEPT_MARGIN
            ):
                return agreed

    if not candidates:
        return None
    return combine_agreeing_patches(TM_MIN_PATCH_SUPPORT)


def build_neighbor_pairs(
    target_positions: np.ndarray,
    tile_width: int,
    tile_height: int,
) -> list[tuple[int, int]]:
    """Build a chronological graph whose old edges never change.

    Each tile looks only backwards.  We mix nearest commanded positions,
    neighbors approached from the same XY direction, and a few recent records.
    Adding more records can therefore only append edges; it cannot replace the
    successful graph of an earlier prefix.
    """
    count = len(target_positions)
    if count < 2:
        return []

    approach_direction = np.zeros((count, 2), dtype=np.int8)
    if count > 1:
        approach_direction[1:] = np.sign(np.diff(target_positions, axis=0)).astype(np.int8)

    def nearest_indices(index: int, pool: np.ndarray, limit: int) -> list[int]:
        if limit <= 0 or not len(pool):
            return []
        delta = target_positions[pool] - target_positions[index]
        distance_squared = np.einsum("ij,ij->i", delta, delta)
        take = min(limit, len(pool))
        selected = np.argpartition(distance_squared, take - 1)[:take]
        selected = selected[np.argsort(distance_squared[selected])]
        return [int(pool[position]) for position in selected]

    pairs: set[tuple[int, int]] = set()
    for index in range(1, count):
        previous = np.arange(index, dtype=np.int32)
        selected: set[int] = set(nearest_indices(index, previous, PAIR_NEIGHBORS))

        same_direction_mask = np.all(
            approach_direction[:index] == approach_direction[index], axis=1
        )
        same_direction = previous[same_direction_mask]
        selected.update(
            nearest_indices(index, same_direction, PAIR_SAME_DIRECTION_NEIGHBORS)
        )

        recent = np.arange(max(0, index - PAIR_RECENT_WINDOW), index, dtype=np.int32)
        selected.update(nearest_indices(index, recent, PAIR_RECENT_NEIGHBORS))

        for other in selected:
            dx, dy = target_positions[other] - target_positions[index]
            if (
                tile_width - abs(dx) < TM_MIN_TEMPLATE_PX
                or tile_height - abs(dy) < TM_MIN_TEMPLATE_PX
            ):
                continue
            pairs.add((other, index))
    return sorted(pairs)


def measure_edges(
    tiles: list[dict],
    target_positions: np.ndarray,
    pairs: list[tuple[int, int]],
) -> list[dict]:
    edges: list[dict] = []
    start = time.time()
    for pair_number, (ref_index, new_index) in enumerate(pairs, 1):
        ref_image = load_preprocessed(tiles[ref_index]["image_path"])
        new_image = load_preprocessed(tiles[new_index]["image_path"])
        if ref_image is None or new_image is None:
            continue

        expected = target_positions[new_index] - target_positions[ref_index]
        match = refine_pair_tm(ref_image, new_image, (expected[0], expected[1]))
        if match is not None:
            # Small edge templates carry less global influence even when their
            # local correlation is high.
            base_weight = (
                (match["score"] ** 4)
                * max(0.20, match["area_ratio"])
                * min(1.0, match.get("patch_support", 1) / 3.0)
            )
            edges.append(
                {
                    "i": ref_index,
                    "j": new_index,
                    "dx": match["dx"],
                    "dy": match["dy"],
                    "score": match["score"],
                    "peak_margin": match["peak_margin"],
                    "base_weight": base_weight,
                }
            )

        if pair_number % PROGRESS_EVERY == 0 or pair_number == len(pairs):
            elapsed = time.time() - start
            print(
                f"  matching {pair_number}/{len(pairs)}: "
                f"accepted={len(edges)}, elapsed={elapsed:.1f}s"
            )
    return edges


def filter_cycle_consistent_edges(
    edges: list[dict],
) -> tuple[list[dict], list[tuple[int, int, int, float]]]:
    """Reject pair matches that do not close a translation triangle."""
    if not edges:
        return [], []

    edge_lookup: dict[tuple[int, int], int] = {}
    forward_neighbors: dict[int, set[int]] = {}
    for index, edge in enumerate(edges):
        left, right = int(edge["i"]), int(edge["j"])
        if left > right:
            raise ValueError("Cycle filter expects edges oriented from lower to higher index")
        edge_lookup[(left, right)] = index
        forward_neighbors.setdefault(left, set()).add(right)

    support = np.zeros(len(edges), dtype=np.int32)
    cycle_errors: list[list[float]] = [[] for _ in edges]
    triangles: list[tuple[int, int, int, float]] = []
    for left, left_neighbors in forward_neighbors.items():
        for middle in left_neighbors:
            middle_neighbors = forward_neighbors.get(middle)
            if not middle_neighbors:
                continue
            for right in left_neighbors.intersection(middle_neighbors):
                edge_lm_index = edge_lookup[(left, middle)]
                edge_mr_index = edge_lookup[(middle, right)]
                edge_lr_index = edge_lookup[(left, right)]
                edge_lm = edges[edge_lm_index]
                edge_mr = edges[edge_mr_index]
                edge_lr = edges[edge_lr_index]
                closure_x = edge_lm["dx"] + edge_mr["dx"] - edge_lr["dx"]
                closure_y = edge_lm["dy"] + edge_mr["dy"] - edge_lr["dy"]
                closure_error = math.hypot(closure_x, closure_y)
                if closure_error <= CYCLE_CLOSURE_PX:
                    triangles.append((left, middle, right, closure_error))
                    for edge_index in (edge_lm_index, edge_mr_index, edge_lr_index):
                        support[edge_index] += 1
                        cycle_errors[edge_index].append(closure_error)

    filtered: list[dict] = []
    for index, edge in enumerate(edges):
        if support[index] < MIN_EDGE_CYCLE_SUPPORT:
            continue
        accepted = dict(edge)
        accepted["cycle_support"] = int(support[index])
        accepted["cycle_error_median"] = float(np.median(cycle_errors[index]))
        accepted["base_weight"] *= min(2.0, 1.0 + 0.15 * support[index])
        filtered.append(accepted)
    retained_pairs = {(edge["i"], edge["j"]) for edge in filtered}
    triangles = [
        triangle
        for triangle in triangles
        if (triangle[0], triangle[1]) in retained_pairs
        and (triangle[1], triangle[2]) in retained_pairs
        and (triangle[0], triangle[2]) in retained_pairs
    ]
    return filtered, triangles


def grow_translation_component(
    target_positions: np.ndarray,
    edges: list[dict],
    triangles: list[tuple[int, int, int, float]],
    minimum_neighbors: int,
) -> tuple[np.ndarray, np.ndarray, list[dict], np.ndarray]:
    """Grow the largest component using only mutually agreeing translations."""
    tile_count = len(target_positions)
    if not edges or not triangles:
        return (
            target_positions.copy(),
            np.zeros(tile_count, dtype=bool),
            [],
            np.zeros(tile_count, dtype=np.int32),
        )

    edge_lookup: dict[tuple[int, int], int] = {}
    adjacency: list[list[tuple[int, int]]] = [[] for _ in range(tile_count)]
    for edge_index, edge in enumerate(edges):
        left, right = int(edge["i"]), int(edge["j"])
        edge_lookup[(left, right)] = edge_index
        adjacency[left].append((right, edge_index))
        adjacency[right].append((left, edge_index))

    def measured_delta(start: int, end: int) -> np.ndarray:
        if start < end:
            edge = edges[edge_lookup[(start, end)]]
            return np.asarray((edge["dx"], edge["dy"]), dtype=np.float64)
        edge = edges[edge_lookup[(end, start)]]
        return -np.asarray((edge["dx"], edge["dy"]), dtype=np.float64)

    def cluster_proposals(proposals: list[tuple[np.ndarray, float, int]]) -> tuple[np.ndarray, list[int]] | None:
        if len(proposals) < minimum_neighbors:
            return None
        best: list[tuple[np.ndarray, float, int]] = []
        best_quality = -math.inf
        for seed_position, _, _ in proposals:
            cluster = [
                proposal
                for proposal in proposals
                if float(np.linalg.norm(proposal[0] - seed_position))
                <= GROWTH_AGREEMENT_PX
            ]
            quality = sum(proposal[1] for proposal in cluster)
            if len(cluster) > len(best) or (
                len(cluster) == len(best) and quality > best_quality
            ):
                best = cluster
                best_quality = quality
        if len(best) < minimum_neighbors:
            return None
        weights = np.asarray([proposal[1] for proposal in best], dtype=np.float64)
        weights /= np.sum(weights)
        position = np.sum(
            np.asarray([proposal[0] for proposal in best]) * weights[:, None], axis=0
        )
        return position, [proposal[2] for proposal in best]

    triangle_quality: list[tuple[float, tuple[int, int, int, float]]] = []
    for triangle in triangles:
        left, middle, right, error = triangle
        involved = (
            edges[edge_lookup[(left, middle)]],
            edges[edge_lookup[(middle, right)]],
            edges[edge_lookup[(left, right)]],
        )
        quality = sum(
            edge.get("cycle_support", 0) + 3.0 * edge["base_weight"]
            for edge in involved
        ) - error
        triangle_quality.append((quality, triangle))
    triangle_quality.sort(key=lambda item: item[0], reverse=True)

    max_correction_px = GROWTH_MAX_COMMAND_CORRECTION_NM / NM_PER_PX
    best_positions = target_positions.copy()
    best_mask = np.zeros(tile_count, dtype=bool)
    best_edges: list[dict] = []
    best_degree = np.zeros(tile_count, dtype=np.int32)
    seen_components: set[frozenset[int]] = set()

    attempted_seeds = 0
    for _, triangle in triangle_quality:
        first, second, third, _ = triangle
        if any(
            first in component and second in component and third in component
            for component in seen_components
        ):
            continue
        attempted_seeds += 1
        if attempted_seeds > MAX_COMPONENT_SEEDS:
            break
        positions = target_positions.copy()
        accepted = np.zeros(tile_count, dtype=bool)
        accepted[[first, second, third]] = True
        positions[first] = target_positions[first]
        positions[second] = positions[first] + measured_delta(first, second)
        positions[third] = positions[first] + measured_delta(first, third)

        accepted_neighbor_count = np.zeros(tile_count, dtype=np.int32)
        queued = np.zeros(tile_count, dtype=bool)
        queue: list[int] = []
        queue_cursor = 0

        def expose(new_node: int) -> None:
            for neighbor, _ in adjacency[new_node]:
                if accepted[neighbor]:
                    continue
                accepted_neighbor_count[neighbor] += 1
                if accepted_neighbor_count[neighbor] >= minimum_neighbors and not queued[neighbor]:
                    queued[neighbor] = True
                    queue.append(neighbor)

        for seed_node in (first, second, third):
            expose(seed_node)

        while queue_cursor < len(queue):
            node = queue[queue_cursor]
            queue_cursor += 1
            queued[node] = False
            if accepted[node]:
                continue

            proposals: list[tuple[np.ndarray, float, int]] = []
            for neighbor, edge_index in adjacency[node]:
                if not accepted[neighbor]:
                    continue
                proposal = positions[neighbor] + measured_delta(neighbor, node)
                proposals.append((proposal, edges[edge_index]["base_weight"], edge_index))
            clustered = cluster_proposals(proposals)
            if clustered is None:
                continue
            proposed_position, _ = clustered
            if float(np.linalg.norm(proposed_position - target_positions[node])) > max_correction_px:
                continue
            positions[node] = proposed_position
            accepted[node] = True
            expose(node)

        signature = frozenset(int(index) for index in np.flatnonzero(accepted))
        if signature in seen_components:
            continue
        seen_components.add(signature)

        component_edges: list[dict] = []
        degree = np.zeros(tile_count, dtype=np.int32)
        for edge in edges:
            left, right = edge["i"], edge["j"]
            if not accepted[left] or not accepted[right]:
                continue
            residual = np.linalg.norm(
                positions[right] - positions[left] - np.asarray((edge["dx"], edge["dy"]))
            )
            if residual <= CONSENSUS_EDGE_RESIDUAL_PX:
                component_edges.append(edge)
                degree[left] += 1
                degree[right] += 1

        score = (int(np.count_nonzero(accepted)), len(component_edges))
        best_score = (int(np.count_nonzero(best_mask)), len(best_edges))
        if score > best_score:
            best_positions = positions
            best_mask = accepted
            best_edges = component_edges
            best_degree = degree

    return best_positions, best_mask, best_edges, best_degree


# -----------------------------------------------------------------------------
# Robust pose-graph solve (NumPy-only preconditioned conjugate gradient)
# -----------------------------------------------------------------------------

def _graph_matvec(
    vector: np.ndarray,
    prior_weight: np.ndarray,
    edge_i: np.ndarray,
    edge_j: np.ndarray,
    edge_weight: np.ndarray,
) -> np.ndarray:
    output = prior_weight * vector
    difference = vector[edge_i] - vector[edge_j]
    np.add.at(output, edge_i, edge_weight * difference)
    np.add.at(output, edge_j, -edge_weight * difference)
    return output


def _solve_axis_cg(
    target: np.ndarray,
    measured_delta: np.ndarray,
    prior_weight: np.ndarray,
    edge_i: np.ndarray,
    edge_j: np.ndarray,
    edge_weight: np.ndarray,
    initial: np.ndarray,
) -> np.ndarray:
    rhs = prior_weight * target
    np.add.at(rhs, edge_i, -edge_weight * measured_delta)
    np.add.at(rhs, edge_j, edge_weight * measured_delta)

    diagonal = prior_weight.copy()
    np.add.at(diagonal, edge_i, edge_weight)
    np.add.at(diagonal, edge_j, edge_weight)
    diagonal = np.maximum(diagonal, 1e-12)

    solution = initial.astype(np.float64, copy=True)
    residual = rhs - _graph_matvec(solution, prior_weight, edge_i, edge_j, edge_weight)
    preconditioned = residual / diagonal
    direction = preconditioned.copy()
    residual_dot = float(np.dot(residual, preconditioned))
    rhs_norm = max(float(np.linalg.norm(rhs)), 1.0)

    for _ in range(POSE_CG_ITERATIONS):
        product = _graph_matvec(direction, prior_weight, edge_i, edge_j, edge_weight)
        denominator = float(np.dot(direction, product))
        if abs(denominator) < 1e-20:
            break
        alpha = residual_dot / denominator
        solution += alpha * direction
        residual -= alpha * product
        if float(np.linalg.norm(residual)) <= POSE_CG_TOLERANCE * rhs_norm:
            break
        preconditioned = residual / diagonal
        new_residual_dot = float(np.dot(residual, preconditioned))
        if abs(residual_dot) < 1e-30:
            break
        direction = preconditioned + (new_residual_dot / residual_dot) * direction
        residual_dot = new_residual_dot
    return solution


def solve_pose_graph(
    target_positions: np.ndarray,
    edges: list[dict],
    pose_prior_weight: float = POSE_PRIOR_WEIGHT,
    pose_anchor_weight: float = POSE_ANCHOR_WEIGHT,
) -> np.ndarray:
    if not edges:
        print("WARNING: no image registrations accepted; using commanded positions")
        return target_positions.copy()

    count = len(target_positions)
    edge_i = np.asarray([edge["i"] for edge in edges], dtype=np.int32)
    edge_j = np.asarray([edge["j"] for edge in edges], dtype=np.int32)
    measured_x = np.asarray([edge["dx"] for edge in edges], dtype=np.float64)
    measured_y = np.asarray([edge["dy"] for edge in edges], dtype=np.float64)
    base_weight = np.asarray([edge["base_weight"] for edge in edges], dtype=np.float64)

    prior_weight = np.full(count, pose_prior_weight, dtype=np.float64)
    prior_weight[0] = pose_anchor_weight
    positions = target_positions.astype(np.float64, copy=True)
    robust_weight = np.ones(len(edges), dtype=np.float64)

    for iteration in range(POSE_IRLS_ITERATIONS):
        edge_weight = base_weight * robust_weight
        positions[:, 0] = _solve_axis_cg(
            target_positions[:, 0], measured_x, prior_weight,
            edge_i, edge_j, edge_weight, positions[:, 0],
        )
        positions[:, 1] = _solve_axis_cg(
            target_positions[:, 1], measured_y, prior_weight,
            edge_i, edge_j, edge_weight, positions[:, 1],
        )

        residual_x = positions[edge_j, 0] - positions[edge_i, 0] - measured_x
        residual_y = positions[edge_j, 1] - positions[edge_i, 1] - measured_y
        residual = np.hypot(residual_x, residual_y)
        robust_weight = np.minimum(1.0, POSE_HUBER_PX / np.maximum(residual, 1e-9))
        print(
            f"  pose iteration {iteration + 1}/{POSE_IRLS_ITERATIONS}: "
            f"median edge residual={np.median(residual):.2f}px, "
            f"p90={np.percentile(residual, 90):.2f}px"
        )
    return positions


def select_consensus_core(
    positions: np.ndarray,
    edges: list[dict],
    minimum_neighbors: int,
    maximum_residual: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a mutually supported tile k-core and its active edges.

    An edge first has to agree with the solved pose graph.  Nodes with fewer
    than ``minimum_neighbors`` remaining edges are then removed repeatedly.
    Repetition matters: a neighbor that was itself rejected cannot count as
    support for another tile.
    """
    tile_count = len(positions)
    if minimum_neighbors <= 0:
        return (
            np.ones(tile_count, dtype=bool),
            np.ones(len(edges), dtype=bool),
            np.zeros(tile_count, dtype=np.int32),
        )
    if not edges:
        return (
            np.zeros(tile_count, dtype=bool),
            np.zeros(0, dtype=bool),
            np.zeros(tile_count, dtype=np.int32),
        )

    edge_i = np.asarray([edge["i"] for edge in edges], dtype=np.int32)
    edge_j = np.asarray([edge["j"] for edge in edges], dtype=np.int32)
    measured_x = np.asarray([edge["dx"] for edge in edges], dtype=np.float64)
    measured_y = np.asarray([edge["dy"] for edge in edges], dtype=np.float64)
    residual = np.hypot(
        positions[edge_j, 0] - positions[edge_i, 0] - measured_x,
        positions[edge_j, 1] - positions[edge_i, 1] - measured_y,
    )
    geometrically_valid = residual <= maximum_residual
    active_tiles = np.ones(tile_count, dtype=bool)

    while True:
        active_edges = (
            geometrically_valid
            & active_tiles[edge_i]
            & active_tiles[edge_j]
        )
        degree = np.zeros(tile_count, dtype=np.int32)
        np.add.at(degree, edge_i[active_edges], 1)
        np.add.at(degree, edge_j[active_edges], 1)
        rejected = active_tiles & (degree < minimum_neighbors)
        if not np.any(rejected):
            break
        active_tiles[rejected] = False

    active_edges = geometrically_valid & active_tiles[edge_i] & active_tiles[edge_j]
    degree = np.zeros(tile_count, dtype=np.int32)
    np.add.at(degree, edge_i[active_edges], 1)
    np.add.at(degree, edge_j[active_edges], 1)
    return active_tiles, active_edges, degree


def enforce_neighbor_consensus(
    target_positions: np.ndarray,
    positions: np.ndarray,
    edges: list[dict],
    minimum_neighbors: int,
    maximum_residual: float,
) -> tuple[np.ndarray, np.ndarray, list[dict], np.ndarray]:
    """Keep and re-solve only the largest strictly consistent translation graph."""

    def largest_component(
        tile_mask: np.ndarray,
        candidate_edges: list[dict],
    ) -> np.ndarray:
        parent = np.arange(len(tile_mask), dtype=np.int32)
        size = np.ones(len(tile_mask), dtype=np.int32)

        def find(node: int) -> int:
            while parent[node] != node:
                parent[node] = parent[parent[node]]
                node = int(parent[node])
            return node

        def union(left: int, right: int) -> None:
            left_root = find(left)
            right_root = find(right)
            if left_root == right_root:
                return
            if size[left_root] < size[right_root]:
                left_root, right_root = right_root, left_root
            parent[right_root] = left_root
            size[left_root] += size[right_root]

        for edge in candidate_edges:
            left, right = edge["i"], edge["j"]
            if tile_mask[left] and tile_mask[right]:
                union(left, right)

        active = np.flatnonzero(tile_mask)
        if not len(active):
            return np.zeros_like(tile_mask)
        roots = np.asarray([find(int(node)) for node in active], dtype=np.int32)
        unique_roots, counts = np.unique(roots, return_counts=True)
        winning_root = int(unique_roots[int(np.argmax(counts))])
        result = np.zeros_like(tile_mask)
        result[active[roots == winning_root]] = True
        return result

    def solve_component(
        tile_mask: np.ndarray,
        component_edges: list[dict],
    ) -> np.ndarray:
        indices = np.flatnonzero(tile_mask)
        if not len(indices):
            return positions.copy()
        remap = np.full(len(tile_mask), -1, dtype=np.int32)
        remap[indices] = np.arange(len(indices), dtype=np.int32)
        local_edges: list[dict] = []
        for edge in component_edges:
            if tile_mask[edge["i"]] and tile_mask[edge["j"]]:
                local_edge = dict(edge)
                local_edge["i"] = int(remap[edge["i"]])
                local_edge["j"] = int(remap[edge["j"]])
                local_edges.append(local_edge)
        result = positions.copy()
        result[indices] = solve_pose_graph(
            target_positions[indices],
            local_edges,
            pose_prior_weight=0.0,
            pose_anchor_weight=POSE_ANCHOR_WEIGHT,
        )
        return result

    current_positions = positions.copy()
    accepted_tiles = np.zeros(len(positions), dtype=bool)
    accepted_edges: list[dict] = []
    degree = np.zeros(len(positions), dtype=np.int32)

    # Two prune/re-solve rounds remove edges that only looked consistent while
    # the full graph was compromising between mutually incompatible groups.
    candidate_edges = edges
    for _ in range(2):
        core_tiles, core_edge_mask, _ = select_consensus_core(
            current_positions,
            candidate_edges,
            minimum_neighbors,
            maximum_residual,
        )
        core_edges = [
            edge
            for edge, accepted in zip(candidate_edges, core_edge_mask)
            if accepted
        ]
        accepted_tiles = largest_component(core_tiles, core_edges)
        accepted_edges = [
            edge
            for edge in core_edges
            if accepted_tiles[edge["i"]] and accepted_tiles[edge["j"]]
        ]
        if not accepted_edges:
            break
        current_positions = solve_component(accepted_tiles, accepted_edges)
        candidate_edges = accepted_edges

    if accepted_edges:
        # Final guarantee after the last component-only solve.
        core_tiles, core_edge_mask, degree = select_consensus_core(
            current_positions,
            accepted_edges,
            minimum_neighbors,
            maximum_residual,
        )
        accepted_tiles &= core_tiles
        accepted_edges = [
            edge
            for edge, accepted in zip(accepted_edges, core_edge_mask)
            if accepted and accepted_tiles[edge["i"]] and accepted_tiles[edge["j"]]
        ]
        degree = np.zeros(len(positions), dtype=np.int32)
        for edge in accepted_edges:
            degree[edge["i"]] += 1
            degree[edge["j"]] += 1

    return current_positions, accepted_tiles, accepted_edges, degree


# -----------------------------------------------------------------------------
# Rendering
# -----------------------------------------------------------------------------

@lru_cache(maxsize=8)
def feather_mask(height: int, width: int) -> np.ndarray:
    feather = max(1, BLEND_FEATHER_PX)
    x = np.minimum(np.arange(width) + 1, np.arange(width, 0, -1))
    y = np.minimum(np.arange(height) + 1, np.arange(height, 0, -1))
    x_weight = np.minimum(1.0, x / float(feather))
    y_weight = np.minimum(1.0, y / float(feather))
    return np.minimum(y_weight[:, None], x_weight[None, :]).astype(np.float32)


def add_tile(
    accumulation: np.ndarray,
    weight: np.ndarray,
    tile: np.ndarray,
    x: int,
    y: int,
    blend_mode: str,
) -> None:
    tile_height, tile_width = tile.shape
    canvas_height, canvas_width = accumulation.shape
    canvas_x1 = max(0, x)
    canvas_y1 = max(0, y)
    canvas_x2 = min(canvas_width, x + tile_width)
    canvas_y2 = min(canvas_height, y + tile_height)
    if canvas_x1 >= canvas_x2 or canvas_y1 >= canvas_y2:
        return

    tile_x1 = canvas_x1 - x
    tile_y1 = canvas_y1 - y
    tile_x2 = tile_x1 + canvas_x2 - canvas_x1
    tile_y2 = tile_y1 + canvas_y2 - canvas_y1
    local_weight = feather_mask(tile_height, tile_width)[tile_y1:tile_y2, tile_x1:tile_x2]
    tile_roi = tile[tile_y1:tile_y2, tile_x1:tile_x2].astype(np.float32)
    accumulation_roi = accumulation[canvas_y1:canvas_y2, canvas_x1:canvas_x2]
    weight_roi = weight[canvas_y1:canvas_y2, canvas_x1:canvas_x2]

    if blend_mode == "sharp":
        # Pick the tile whose pixel lies furthest from a tile boundary.  Unlike
        # feather averaging this cannot create a doubled high-contrast edge
        # when the scans contain a small affine/raster distortion.
        replace = local_weight > weight_roi
        accumulation_roi[replace] = tile_roi[replace]
        weight_roi[replace] = local_weight[replace]
    else:
        accumulation_roi += tile_roi * local_weight
        weight_roi += local_weight


def render_mosaic(
    tiles: list[dict],
    positions: np.ndarray,
    tile_shape: tuple[int, int],
    output_path: str,
    render_mask: np.ndarray | None = None,
    blend_mode: str = BLEND_MODE,
) -> tuple[str | None, str | None]:
    tile_height, tile_width = tile_shape
    if render_mask is None:
        render_mask = np.ones(len(tiles), dtype=bool)
    rendered_positions = positions[render_mask]
    if not len(rendered_positions):
        raise ValueError("No tiles satisfy the rendering acceptance criteria")

    min_x = math.floor(float(np.min(rendered_positions[:, 0])))
    min_y = math.floor(float(np.min(rendered_positions[:, 1])))
    max_x = math.ceil(float(np.max(rendered_positions[:, 0])))
    max_y = math.ceil(float(np.max(rendered_positions[:, 1])))
    canvas_width = max_x - min_x + tile_width + 2 * CANVAS_MARGIN_PX
    canvas_height = max_y - min_y + tile_height + 2 * CANVAS_MARGIN_PX
    offset_x = -min_x + CANVAS_MARGIN_PX
    offset_y = -min_y + CANVAS_MARGIN_PX

    print(f"Canvas size: {canvas_width}x{canvas_height}")
    accumulation = np.zeros((canvas_height, canvas_width), dtype=np.float32)
    weight = np.zeros((canvas_height, canvas_width), dtype=np.float32)

    render_total = int(np.count_nonzero(render_mask))
    rendered = 0
    for index, tile_info in enumerate(tiles):
        if not render_mask[index]:
            continue
        image = load_and_crop(tile_info["image_path"])
        if image is None:
            continue
        x = int(round(positions[index, 0])) + offset_x
        y = int(round(positions[index, 1])) + offset_y
        add_tile(accumulation, weight, image, x, y, blend_mode)
        rendered += 1
        if rendered % PROGRESS_EVERY == 0 or rendered == render_total:
            print(f"  rendering {rendered}/{render_total}")

    covered = weight > 0
    canvas = np.zeros_like(accumulation, dtype=np.uint8)
    if blend_mode == "sharp":
        canvas[covered] = np.clip(accumulation[covered], 0, 255).astype(np.uint8)
    else:
        canvas[covered] = np.clip(
            accumulation[covered] / weight[covered], 0, 255
        ).astype(np.uint8)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    base, _ = os.path.splitext(output_path)
    full_path: str | None = None
    cropped_path: str | None = None

    if SAVE_FULL_CANVAS:
        full_path = f"{base}_full{OUTPUT_FORMAT}"
        cv2.imwrite(full_path, canvas)
        print(f"Saved full canvas: {full_path}")

    if SAVE_CROPPED:
        if np.any(covered):
            rows = np.flatnonzero(np.any(covered, axis=1))
            columns = np.flatnonzero(np.any(covered, axis=0))
            padding = 10
            y1 = max(0, int(rows[0]) - padding)
            y2 = min(canvas_height, int(rows[-1]) + padding + 1)
            x1 = max(0, int(columns[0]) - padding)
            x2 = min(canvas_width, int(columns[-1]) + padding + 1)
            cropped = canvas[y1:y2, x1:x2]
        else:
            cropped = canvas
        cropped_path = f"{base}_cropped{OUTPUT_FORMAT}"
        cv2.imwrite(cropped_path, cropped)
        print(f"Saved cropped mosaic: {cropped_path} ({cropped.shape[1]}x{cropped.shape[0]})")

    return full_path, cropped_path


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-tiles", type=int, default=MAX_TILES,
        help="maximum JSONL records to use; 0 uses all records",
    )
    parser.add_argument(
        "--output", default=OUTPUT_PATH,
        help="output base path (the cropped image gets a _cropped suffix)",
    )
    parser.add_argument(
        "--no-refine", action="store_true",
        help="render commanded positions without image registration",
    )
    parser.add_argument(
        "--min-agreeing-neighbors", type=int, default=MIN_AGREEING_NEIGHBORS,
        help="minimum mutually accepted image neighbors required per rendered tile",
    )
    parser.add_argument(
        "--consensus-residual-px", type=float, default=CONSENSUS_EDGE_RESIDUAL_PX,
        help="maximum pose-graph edge residual counted as an agreeing neighbor",
    )
    parser.add_argument(
        "--blend-mode", choices=("sharp", "feather"), default=BLEND_MODE,
        help="sharp selects one tile per pixel; feather averages overlaps",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.time()
    print(f"Loading JSONL: {JSONL_PATH}")
    records = load_jsonl(JSONL_PATH)
    tiles, missing = prepare_tile_list(records, args.max_tiles)
    print(
        f"Records: {len(records)}, selected images: {len(tiles)}, "
        f"missing selected images: {missing}"
    )
    print(f"Scale: {NM_PER_PX} nm/px; pair search: {PAIR_SEARCH_RADIUS_NM} nm")

    first_image = load_and_crop(tiles[0]["image_path"])
    if first_image is None:
        print(f"ERROR: cannot load first image: {tiles[0]['image_path']}")
        return 1
    tile_height, tile_width = first_image.shape
    for tile in tiles[:10]:
        sample = load_and_crop(tile["image_path"])
        if sample is None or sample.shape != first_image.shape:
            print("ERROR: selected images do not share one readable tile shape")
            return 1
    print(f"Tile shape after overlay crop: {tile_width}x{tile_height}")

    target_positions = np.asarray(
        [(tile["x_px"], tile["y_px"]) for tile in tiles], dtype=np.float64
    )
    if args.no_refine or len(tiles) < 2:
        final_positions = target_positions
        render_mask = np.ones(len(tiles), dtype=bool)
    else:
        pairs = build_neighbor_pairs(target_positions, tile_width, tile_height)
        print(f"Candidate overlapping pairs: {len(pairs)}")
        edges = measure_edges(tiles, target_positions, pairs)
        load_preprocessed.cache_clear()
        print(f"Accepted registrations: {len(edges)}/{len(pairs)}")
        cycle_edges, triangles = filter_cycle_consistent_edges(edges)
        print(
            f"Cycle-consistent registrations "
            f"(>= {MIN_EDGE_CYCLE_SUPPORT} triangles, "
            f"closure <= {CYCLE_CLOSURE_PX:.1f}px): "
            f"{len(cycle_edges)}/{len(edges)} edges, {len(triangles)} triangles"
        )
        if not cycle_edges or not triangles:
            print("ERROR: no pairwise registrations passed cycle consistency")
            return 2
        grown_positions, grown_mask, grown_edges, _ = grow_translation_component(
            target_positions,
            cycle_edges,
            triangles,
            args.min_agreeing_neighbors,
        )
        print(
            f"Largest monotonically grown component: "
            f"tiles={int(np.count_nonzero(grown_mask))}/{len(tiles)}, "
            f"edges={len(grown_edges)}/{len(cycle_edges)}"
        )
        if not grown_edges:
            print("ERROR: no component could be grown from consistent triangles")
            return 2
        final_positions, render_mask, consensus_edges, degree = enforce_neighbor_consensus(
            target_positions,
            grown_positions,
            grown_edges,
            args.min_agreeing_neighbors,
            args.consensus_residual_px,
        )
        print(
            f"Consensus {args.min_agreeing_neighbors}-core "
            f"(edge residual <= {args.consensus_residual_px:.1f}px): "
            f"tiles={int(np.count_nonzero(render_mask))}/{len(tiles)}, "
            f"edges={len(consensus_edges)}/{len(edges)}"
        )
        if np.any(render_mask):
            accepted_degree = degree[render_mask]
            print(
                f"Agreeing neighbors per rendered tile: "
                f"min={int(np.min(accepted_degree))}, "
                f"median={float(np.median(accepted_degree)):.1f}"
            )
        else:
            print("ERROR: no tile has enough mutually agreeing neighbors")
            return 2
        correction = np.linalg.norm(
            final_positions[render_mask] - target_positions[render_mask], axis=1
        )
        print(
            f"Command correction: median={np.median(correction):.1f}px, "
            f"p90={np.percentile(correction, 90):.1f}px, "
            f"max={np.max(correction):.1f}px"
        )

    render_mosaic(
        tiles,
        final_positions,
        (tile_height, tile_width),
        os.path.abspath(args.output),
        render_mask,
        args.blend_mode,
    )
    print(f"Total time: {time.time() - started:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
