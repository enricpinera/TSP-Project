"""Visualize how the first TSP10 training tour becomes sequential training instances."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "tsp-data" / "tsp10_train_concorde.txt"
OUTPUT_DIR = ROOT / "Visualizations" / "output"

WHITE = (255, 255, 255)
INK = (17, 24, 39)
MUTED = (100, 116, 139)
EDGE = (220, 226, 235)
PREFIX = (37, 99, 235)
PREDICT = (22, 163, 74)
CURRENT = (234, 88, 12)
TARGET = (22, 163, 74)
VISITED = (148, 163, 184)
UNVISITED = (255, 255, 255)
GRID = (226, 232, 240)


@dataclass(frozen=True)
class Instance:
    coords: list[tuple[float, float]]
    tour: list[int]


@dataclass(frozen=True)
class Decision:
    index: int
    current: int
    target: int
    visited: set[int]
    prefix: list[int]


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    if Path(path).exists():
        return ImageFont.truetype(path, size)
    return ImageFont.load_default()


FONT_XS = font(14)
FONT_SM = font(17)
FONT_STEP = font(22, bold=True)


def parse_first_instance() -> Instance:
    line = DATASET.read_text(encoding="utf-8").splitlines()[0]
    parts = line.split()
    output_idx = parts.index("output")
    coords = [(float(parts[i]), float(parts[i + 1])) for i in range(0, 20, 2)]
    tour = [int(value) - 1 for value in parts[output_idx + 1 :]]
    if tour[-1] == tour[0]:
        tour = tour[:-1]
    return Instance(coords=coords, tour=tour)


def build_decisions(instance: Instance) -> list[Decision]:
    decisions: list[Decision] = []
    visited: set[int] = set()
    for idx in range(len(instance.tour) - 2):
        current = instance.tour[idx]
        target = instance.tour[idx + 1]
        decisions.append(
            Decision(
                index=idx + 1,
                current=current,
                target=target,
                visited=set(visited),
                prefix=instance.tour[: idx + 1],
            )
        )
        visited.add(current)
    return decisions


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
        (TARGET, "target node"),
        (VISITED, "visited node"),
        (UNVISITED, "unvisited node"),
        (PREFIX, "tour prefix"),
        (PREDICT, "edge to predict"),
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
        cursor += 230


def node_color(node: int, decision: Decision) -> tuple[int, int, int]:
    if node == decision.current:
        return CURRENT
    if node == decision.target:
        return TARGET
    if node in decision.visited:
        return VISITED
    return UNVISITED


def draw_graph(draw: ImageDraw.ImageDraw, instance: Instance, decision: Decision, box: tuple[int, int, int, int]) -> None:
    map_point = mapper(instance.coords, box)
    points = [map_point(point) for point in instance.coords]

    draw.rectangle(box, fill=WHITE, outline=GRID, width=2)

    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            draw.line((points[i], points[j]), fill=EDGE, width=1)

    for a, b in zip(decision.prefix, decision.prefix[1:]):
        draw.line((points[a], points[b]), fill=PREFIX, width=4)
    draw.line((points[decision.current], points[decision.target]), fill=PREDICT, width=5)

    for node, (x, y) in enumerate(points):
        fill = node_color(node, decision)
        draw.ellipse((x - 16, y - 16, x + 16, y + 16), fill=fill, outline=INK, width=2)
        label = str(node + 1)
        bbox = draw.textbbox((0, 0), label, font=FONT_XS)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        label_fill = WHITE if fill in (CURRENT, TARGET, VISITED) else INK
        draw_text(draw, (x - tw // 2, y - th // 2 - 1), label, FONT_XS, label_fill)
    draw_step_label(draw, box, points, f"Step {decision.index}")


def render_single_decision(instance: Instance, decision: Decision, output_path: Path) -> None:
    img = Image.new("RGB", (900, 960), WHITE)
    draw = ImageDraw.Draw(img)
    draw_graph(draw, instance, decision, (70, 50, 830, 810))
    draw_legend(draw, 70, 850)
    img.save(output_path)


def generate() -> list[Path]:
    decisions_dir = OUTPUT_DIR / "training_instances"
    decisions_dir.mkdir(parents=True, exist_ok=True)
    instance = parse_first_instance()
    decisions = build_decisions(instance)

    paths: list[Path] = []
    for decision in decisions:
        output = decisions_dir / f"tsp10_training_decision_{decision.index:02d}.png"
        render_single_decision(instance, decision, output)
        paths.append(output)
    return paths


def main() -> None:
    paths = generate()
    print("Generated TSP10 training-instance figures:")
    for path in paths:
        print(f"  {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
