#!/usr/bin/env python3
"""Visualize the complete GCN TSP10 inference process on the first test graph."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
TEST_DATASET = ROOT / "tsp-data" / "tsp10_test_concorde.txt"
GCN_RESULTS = ROOT / "Results" / "gcn_tsp10.csv"
OUTPUT_DIR = ROOT / "Visualizations" / "output"

WHITE = (255, 255, 255)
INK = (17, 24, 39)
MUTED = (100, 116, 139)
EDGE = (220, 226, 235)
PREFIX = (37, 99, 235)
PREDICT = (22, 163, 74)
CLOSE = (220, 38, 38)
CURRENT = (234, 88, 12)
TARGET = (22, 163, 74)
VISITED = (148, 163, 184)
UNVISITED = (255, 255, 255)
GRID = (226, 232, 240)


@dataclass(frozen=True)
class Instance:
    coords: list[tuple[float, float]]


@dataclass(frozen=True)
class Step:
    index: int
    current: int
    target: int
    visited: set[int]
    prefix: list[int]
    closing: bool


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    if Path(path).exists():
        return ImageFont.truetype(path, size)
    return ImageFont.load_default()


FONT_XS = font(14)
FONT_SM = font(17)
FONT_STEP = font(22, bold=True)


def load_first_test_instance() -> Instance:
    line = TEST_DATASET.read_text(encoding="utf-8").splitlines()[0]
    parts = line.split()
    coords = [(float(parts[i]), float(parts[i + 1])) for i in range(0, 20, 2)]
    return Instance(coords=coords)


def parse_tour(text: str) -> list[int]:
    match = re.fullmatch(r"\{([\d,\s]+)\}", text.strip())
    if not match:
        raise ValueError(f"Invalid tour: {text}")
    return [int(value.strip()) - 1 for value in match.group(1).split(",")]


def load_gcn_tour() -> list[int]:
    with GCN_RESULTS.open("r", encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    return parse_tour(row["gcn_tour"])


def build_steps(tour: list[int]) -> list[Step]:
    steps: list[Step] = []
    visited: set[int] = set()
    start = tour[0]
    for idx in range(len(tour) - 1):
        current = tour[idx]
        target = tour[idx + 1]
        if idx > 0:
            visited.add(tour[idx - 1])
        steps.append(
            Step(
                index=idx + 1,
                current=current,
                target=target,
                visited=set(visited),
                prefix=tour[: idx + 1],
                closing=target == start,
            )
        )
    return steps


def mapper(coords: list[tuple[float, float]], box: tuple[int, int, int, int]):
    left, top, right, bottom = box
    width = right - left
    height = bottom - top
    if width != height:
        raise ValueError("Graph box must be square to preserve the 1x1 TSP coordinate space.")

    def map_point(point: tuple[float, float]) -> tuple[int, int]:
        x, y = point
        px = left + x * width
        py = bottom - y * height
        return int(px), int(py)

    return map_point


def draw_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, fnt: ImageFont.ImageFont, fill=INK) -> None:
    draw.text(xy, text, font=fnt, fill=fill)


def draw_step_label(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    points: list[tuple[int, int]],
    text: str,
) -> None:
    pad_x, pad_y = 8, 5
    text_bbox = draw.textbbox((0, 0), text, font=FONT_STEP)
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]
    left, top, right, bottom = box
    candidates = [
        (left + 18, top + 16),
        (right - text_w - 18, top + 16),
        (left + 18, bottom - text_h - 20),
        (right - text_w - 18, bottom - text_h - 20),
    ]

    def clear_score(candidate: tuple[int, int]) -> int:
        x, y = candidate
        label_box = (x - 18, y - 18, x + text_w + 18, y + text_h + 18)
        return sum(label_box[0] <= px <= label_box[2] and label_box[1] <= py <= label_box[3] for px, py in points)

    x, y = min(candidates, key=clear_score)
    bbox = draw.textbbox((x, y), text, font=FONT_STEP)
    draw.rectangle(
        (bbox[0] - pad_x, bbox[1] - pad_y, bbox[2] + pad_x, bbox[3] + pad_y),
        fill=WHITE,
    )
    draw_text(draw, (x, y), text, FONT_STEP, INK)


def draw_legend(draw: ImageDraw.ImageDraw, x: int, y: int) -> None:
    entries = [
        (CURRENT, "current node"),
        (TARGET, "next node"),
        (VISITED, "visited node"),
        (UNVISITED, "unvisited node"),
        (PREFIX, "built tour"),
        (PREDICT, "next edge"),
        (CLOSE, "closing edge"),
    ]
    cursor = x
    for color, label in entries[:4]:
        draw.ellipse((cursor, y, cursor + 18, y + 18), fill=color, outline=INK, width=2)
        draw_text(draw, (cursor + 28, y - 2), label, FONT_SM, MUTED)
        cursor += 205
    cursor = x
    y += 34
    for color, label in entries[4:]:
        draw.line((cursor, y + 9, cursor + 46, y + 9), fill=color, width=5)
        draw_text(draw, (cursor + 58, y - 2), label, FONT_SM, MUTED)
        cursor += 210


def node_color(node: int, step: Step) -> tuple[int, int, int]:
    if node == step.current:
        return CURRENT
    if node == step.target:
        return CLOSE if step.closing else TARGET
    if node in step.visited:
        return VISITED
    return UNVISITED


def draw_graph(draw: ImageDraw.ImageDraw, instance: Instance, step: Step, box: tuple[int, int, int, int]) -> None:
    map_point = mapper(instance.coords, box)
    points = [map_point(point) for point in instance.coords]

    draw.rectangle(box, fill=WHITE, outline=GRID, width=2)

    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            draw.line((points[i], points[j]), fill=EDGE, width=1)

    for a, b in zip(step.prefix, step.prefix[1:]):
        draw.line((points[a], points[b]), fill=PREFIX, width=4)
    draw.line((points[step.current], points[step.target]), fill=CLOSE if step.closing else PREDICT, width=5)

    for node, (x, y) in enumerate(points):
        fill = node_color(node, step)
        draw.ellipse((x - 15, y - 15, x + 15, y + 15), fill=fill, outline=INK, width=2)
        label = str(node + 1)
        bbox = draw.textbbox((0, 0), label, font=FONT_XS)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        label_fill = WHITE if fill in (CURRENT, TARGET, VISITED, CLOSE) else INK
        draw_text(draw, (x - tw // 2, y - th // 2 - 1), label, FONT_XS, label_fill)
    draw_step_label(draw, box, points, f"Step {step.index}")


def render_single_step(instance: Instance, step: Step, output_path: Path) -> None:
    img = Image.new("RGB", (900, 960), WHITE)
    draw = ImageDraw.Draw(img)
    draw_graph(draw, instance, step, (70, 50, 830, 810))
    draw_legend(draw, 70, 850)
    img.save(output_path)


def generate() -> list[Path]:
    inference_dir = OUTPUT_DIR / "gcn_inference"
    inference_dir.mkdir(parents=True, exist_ok=True)
    instance = load_first_test_instance()
    tour = load_gcn_tour()
    steps = build_steps(tour)

    paths: list[Path] = []
    for step in steps:
        output = inference_dir / f"tsp10_gcn_inference_step_{step.index:02d}.png"
        render_single_step(instance, step, output)
        paths.append(output)
    return paths


def main() -> None:
    paths = generate()
    print("Generated GCN inference figures:")
    for path in paths:
        print(f"  {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
