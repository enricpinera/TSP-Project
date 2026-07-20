#!/usr/bin/env python3
"""Generate score and time boxplots by algorithm for each evaluated TSP size."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "Results"
OUTPUT_ROOT = ROOT / "Visualizations" / "output"

SIZES = [10, 20, 30]
ALGORITHMS = [
    "nearest_neighbor",
    "greedy",
    "christofides",
    "2opt",
    "3opt",
    "simulated_annealing",
    "threshold_accepting",
    "gcn",
    "transformer",
]

LABELS = {
    "nearest_neighbor": "Nearest\nneighbor",
    "greedy": "Greedy",
    "christofides": "Christofides",
    "2opt": "2-opt",
    "3opt": "3-opt",
    "simulated_annealing": "Simulated\nannealing",
    "threshold_accepting": "Threshold\naccepting",
    "gcn": "GCN",
    "transformer": "Transformer",
}

WHITE = (255, 255, 255)
INK = (17, 24, 39)
MUTED = (100, 116, 139)
GRID = (226, 232, 240)
AXIS = (71, 85, 105)
BOX_FILL = (219, 234, 254)
BOX_EDGE = (37, 99, 235)
TIME_BOX_FILL = (220, 252, 231)
TIME_BOX_EDGE = (22, 163, 74)
MEDIAN = (220, 38, 38)
WHISKER = (51, 65, 85)
ZERO = (22, 163, 74)
GCN_COLOR = (37, 99, 235)
TRANSFORMER_COLOR = (22, 163, 74)
SERIES_COLORS = [
    (37, 99, 235),
    (22, 163, 74),
    (220, 38, 38),
    (147, 51, 234),
    (234, 88, 12),
    (8, 145, 178),
    (202, 138, 4),
    (79, 70, 229),
    (15, 118, 110),
]


@dataclass(frozen=True)
class BoxStats:
    algorithm: str
    values: list[float]
    minimum: float
    q1: float
    median: float
    q3: float
    maximum: float
    mean: float
    count: int


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
    ]
    for path in paths:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


FONT_XS = font(13)
FONT_SM = font(16)
FONT_MD = font(21)
FONT_TITLE = font(32, bold=True)


def percentile(values: list[float], percent: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    idx = (len(ordered) - 1) * percent
    lower = int(idx)
    upper = min(lower + 1, len(ordered) - 1)
    weight = idx - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


@dataclass(frozen=True)
class MetricConfig:
    name: str
    csv_column: str
    axis_label: str
    lower_is_better: bool
    output_dir: str
    file_prefix: str
    zero_line: bool = False


METRICS = [
    MetricConfig(
        name="score",
        csv_column="score",
        axis_label="score",
        lower_is_better=True,
        output_dir="score_boxplots",
        file_prefix="score_boxplot",
        zero_line=True,
    ),
    MetricConfig(
        name="time",
        csv_column="time",
        axis_label="temps (s)",
        lower_is_better=True,
        output_dir="time_boxplots",
        file_prefix="time_boxplot",
    ),
]


def read_metric_values(num_nodes: int, algorithm: str, metric: MetricConfig) -> list[float]:
    path = RESULTS_DIR / f"{algorithm}_tsp{num_nodes}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [float(row[metric.csv_column]) for row in reader]


def compute_box_stats(num_nodes: int, metric: MetricConfig) -> list[BoxStats]:
    stats: list[BoxStats] = []
    for algorithm in ALGORITHMS:
        values = read_metric_values(num_nodes, algorithm, metric)
        if not values:
            raise ValueError(f"No {metric.name} values found for {algorithm} TSP{num_nodes}")
        stats.append(
            BoxStats(
                algorithm=algorithm,
                values=values,
                minimum=min(values),
                q1=percentile(values, 0.25),
                median=percentile(values, 0.50),
                q3=percentile(values, 0.75),
                maximum=max(values),
                mean=sum(values) / len(values),
                count=len(values),
            )
        )
    return sorted(stats, key=lambda item: item.median, reverse=not metric.lower_is_better)


def draw_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, fnt: ImageFont.ImageFont, fill=INK) -> None:
    draw.text(xy, text, font=fnt, fill=fill)


def draw_centered_text(
    draw: ImageDraw.ImageDraw,
    center_x: int,
    y: int,
    text: str,
    fnt: ImageFont.ImageFont,
    fill=INK,
    line_gap: int = 2,
) -> None:
    lines = text.split("\n")
    heights = []
    widths = []
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=fnt)
        widths.append(bbox[2] - bbox[0])
        heights.append(bbox[3] - bbox[1])
    cursor_y = y
    for line, width, height in zip(lines, widths, heights):
        draw_text(draw, (center_x - width // 2, cursor_y), line, fnt, fill)
        cursor_y += height + line_gap


def draw_box(
    draw: ImageDraw.ImageDraw,
    cx: int,
    stats: BoxStats,
    map_y,
    box_w: int,
    cap_w: int,
    fill: tuple[int, int, int],
    edge: tuple[int, int, int],
    clip_min: float | None = None,
    clip_max: float | None = None,
) -> None:
    minimum = max(stats.minimum, clip_min) if clip_min is not None else stats.minimum
    maximum = min(stats.maximum, clip_max) if clip_max is not None else stats.maximum
    q1 = min(max(stats.q1, minimum), maximum)
    median = min(max(stats.median, minimum), maximum)
    q3 = min(max(stats.q3, minimum), maximum)
    mean = min(max(stats.mean, minimum), maximum)

    y_minimum = map_y(minimum)
    y_q1 = map_y(q1)
    y_median = map_y(median)
    y_q3 = map_y(q3)
    y_maximum = map_y(maximum)
    y_mean = map_y(mean)

    draw.line((cx, y_maximum, cx, y_q3), fill=WHISKER, width=2)
    draw.line((cx, y_q1, cx, y_minimum), fill=WHISKER, width=2)
    draw.line((cx - cap_w // 2, y_maximum, cx + cap_w // 2, y_maximum), fill=WHISKER, width=2)
    draw.line((cx - cap_w // 2, y_minimum, cx + cap_w // 2, y_minimum), fill=WHISKER, width=2)
    draw.rectangle((cx - box_w // 2, y_q3, cx + box_w // 2, y_q1), fill=fill, outline=edge, width=2)
    draw.line((cx - box_w // 2, y_median, cx + box_w // 2, y_median), fill=MEDIAN, width=3)
    draw.ellipse((cx - 3, y_mean - 3, cx + 3, y_mean + 3), fill=INK)


def nice_upper_bound(value: float) -> float:
    if value <= 0.1:
        return 0.1
    if value <= 0.25:
        return 0.25
    if value <= 0.5:
        return 0.5
    return value


def metric_upper_bound(value: float, metric: MetricConfig) -> float:
    if metric.name == "score":
        return nice_upper_bound(value)
    return value


def render_boxplot_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    num_nodes: int,
    stats: list[BoxStats],
    metric: MetricConfig,
) -> None:
    left, top, right, bottom = box
    title_h = 36
    label_h = 78
    axis_left = left + 84
    axis_top = top + title_h
    axis_right = right - 28
    axis_bottom = bottom - label_h

    y_min = min(0.0, min(item.minimum for item in stats)) if metric.zero_line else 0.0
    y_max = metric_upper_bound(max(item.maximum for item in stats), metric)
    padding = (y_max - y_min) * 0.06
    if metric.zero_line:
        y_min -= padding
    y_max += padding

    def map_y(value: float) -> int:
        return int(axis_bottom - (value - y_min) / (y_max - y_min) * (axis_bottom - axis_top))

    tick_count = 5
    for i in range(tick_count + 1):
        value = y_min + i * (y_max - y_min) / tick_count
        y = map_y(value)
        draw.line((axis_left, y, axis_right, y), fill=GRID, width=1)
        label = f"{value:.2f}"
        bbox = draw.textbbox((0, 0), label, font=FONT_XS)
        draw_text(draw, (axis_left - 12 - (bbox[2] - bbox[0]), y - 8), label, FONT_XS, MUTED)

    if metric.zero_line:
        zero_y = map_y(0.0)
        draw.line((axis_left, zero_y, axis_right, zero_y), fill=ZERO, width=2)
    draw.line((axis_left, axis_top, axis_left, axis_bottom), fill=AXIS, width=2)
    draw.line((axis_left, axis_bottom, axis_right, axis_bottom), fill=AXIS, width=2)
    draw_text(draw, (left, axis_top - 24), metric.axis_label, FONT_SM, MUTED)

    slot_w = (axis_right - axis_left) / len(stats)
    box_w = min(54, int(slot_w * 0.48))
    cap_w = int(box_w * 0.68)

    for idx, item in enumerate(stats):
        cx = int(axis_left + slot_w * (idx + 0.5))
        fill = TIME_BOX_FILL if metric.name == "time" else BOX_FILL
        edge = TIME_BOX_EDGE if metric.name == "time" else BOX_EDGE
        draw_box(draw, cx, item, map_y, box_w, cap_w, fill, edge)
        draw_centered_text(draw, cx, axis_bottom + 16, LABELS[item.algorithm], FONT_XS, MUTED)

    legend_y = bottom - 24
    draw.line((right - 360, legend_y + 8, right - 318, legend_y + 8), fill=MEDIAN, width=3)
    draw_text(draw, (right - 308, legend_y), "median", FONT_XS, MUTED)
    draw.ellipse((right - 222, legend_y + 5, right - 216, legend_y + 11), fill=INK)
    draw_text(draw, (right - 206, legend_y), "mean", FONT_XS, MUTED)
    if metric.zero_line:
        draw.line((right - 140, legend_y + 8, right - 98, legend_y + 8), fill=ZERO, width=2)
        draw_text(draw, (right - 88, legend_y), "score 0", FONT_XS, MUTED)


def render_single(num_nodes: int, stats: list[BoxStats], metric: MetricConfig, output_path: Path) -> None:
    img = Image.new("RGB", (1320, 760), WHITE)
    draw = ImageDraw.Draw(img)
    render_boxplot_panel(draw, (52, 42, 1288, 720), num_nodes, stats, metric)
    img.save(output_path)


def render_combined(all_stats: dict[int, list[BoxStats]], metric: MetricConfig, output_path: Path) -> None:
    img = Image.new("RGB", (1320, 1800), WHITE)
    draw = ImageDraw.Draw(img)
    panel_top = 42
    for num_nodes in SIZES:
        render_boxplot_panel(
            draw,
            (52, panel_top, 1288, panel_top + 520),
            num_nodes,
            all_stats[num_nodes],
            metric,
        )
        panel_top += 540
    img.save(output_path)


def stats_by_algorithm(stats: list[BoxStats]) -> dict[str, BoxStats]:
    return {item.algorithm: item for item in stats}


def render_score_time_panel(num_nodes: int, output_path: Path) -> None:
    score_stats = compute_box_stats(num_nodes, METRICS[0])
    time_by_algorithm = stats_by_algorithm(compute_box_stats(num_nodes, METRICS[1]))
    time_stats = [time_by_algorithm[item.algorithm] for item in score_stats]

    img = Image.new("RGB", (1480, 820), WHITE)
    draw = ImageDraw.Draw(img)
    left, top, right, bottom = 60, 58, 1418, 760
    label_h = 90
    axis_left = left + 84
    axis_top = top + 36
    axis_right = right - 92
    axis_bottom = bottom - label_h

    score_min = min(0.0, min(item.minimum for item in score_stats))
    score_max = nice_upper_bound(max(item.maximum for item in score_stats))
    score_padding = (score_max - score_min) * 0.06
    score_min -= score_padding
    score_max += score_padding

    time_min = 0.0
    time_max = max(percentile(item.values, 0.95) for item in time_stats)
    time_max += (time_max - time_min) * 0.06

    def map_score(value: float) -> int:
        return int(axis_bottom - (value - score_min) / (score_max - score_min) * (axis_bottom - axis_top))

    def map_time(value: float) -> int:
        return int(axis_bottom - (value - time_min) / (time_max - time_min) * (axis_bottom - axis_top))

    tick_count = 5
    for i in range(tick_count + 1):
        score_value = score_min + i * (score_max - score_min) / tick_count
        y = map_score(score_value)
        draw.line((axis_left, y, axis_right, y), fill=GRID, width=1)

        score_label = f"{score_value:.2f}"
        score_bbox = draw.textbbox((0, 0), score_label, font=FONT_XS)
        draw_text(draw, (axis_left - 12 - (score_bbox[2] - score_bbox[0]), y - 8), score_label, FONT_XS, MUTED)

        time_value = time_min + i * (time_max - time_min) / tick_count
        time_label = f"{time_value:.3f}" if time_max < 1 else f"{time_value:.2f}"
        draw_text(draw, (axis_right + 12, y - 8), time_label, FONT_XS, MUTED)

    zero_y = map_score(0.0)
    draw.line((axis_left, zero_y, axis_right, zero_y), fill=ZERO, width=2)
    draw.line((axis_left, axis_top, axis_left, axis_bottom), fill=AXIS, width=2)
    draw.line((axis_right, axis_top, axis_right, axis_bottom), fill=AXIS, width=2)
    draw.line((axis_left, axis_bottom, axis_right, axis_bottom), fill=AXIS, width=2)
    draw_text(draw, (left, axis_top - 24), "score", FONT_SM, MUTED)
    draw_text(draw, (axis_right + 24, axis_top - 24), "temps (s)", FONT_SM, MUTED)

    slot_w = (axis_right - axis_left) / len(score_stats)
    box_w = min(32, int(slot_w * 0.26))
    cap_w = int(box_w * 0.72)
    pair_gap = box_w + 6

    for idx, (score_item, time_item) in enumerate(zip(score_stats, time_stats)):
        cx = int(axis_left + slot_w * (idx + 0.5))
        score_x = cx - pair_gap // 2
        time_x = cx + pair_gap // 2
        draw_box(draw, score_x, score_item, map_score, box_w, cap_w, BOX_FILL, BOX_EDGE)
        draw_box(
            draw,
            time_x,
            time_item,
            map_time,
            box_w,
            cap_w,
            TIME_BOX_FILL,
            TIME_BOX_EDGE,
            clip_min=time_min,
            clip_max=time_max,
        )
        draw_centered_text(draw, cx, axis_bottom + 18, LABELS[score_item.algorithm], FONT_XS, MUTED)

    legend_y = bottom - 28
    draw.rectangle((axis_right - 344, legend_y + 1, axis_right - 322, legend_y + 15), fill=BOX_FILL, outline=BOX_EDGE, width=2)
    draw_text(draw, (axis_right - 312, legend_y), "score", FONT_XS, MUTED)
    draw.rectangle((axis_right - 230, legend_y + 1, axis_right - 208, legend_y + 15), fill=TIME_BOX_FILL, outline=TIME_BOX_EDGE, width=2)
    draw_text(draw, (axis_right - 198, legend_y), "temps", FONT_XS, MUTED)
    draw.line((axis_right - 106, legend_y + 8, axis_right - 64, legend_y + 8), fill=MEDIAN, width=3)
    draw_text(draw, (axis_right - 54, legend_y), "median", FONT_XS, MUTED)

    img.save(output_path)


def generate_score_time_boxplots() -> list[Path]:
    output_dir = OUTPUT_ROOT / "score_time_boxplots"
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for num_nodes in SIZES:
        output = output_dir / f"score_time_boxplot_tsp{num_nodes}.png"
        render_score_time_panel(num_nodes, output)
        paths.append(output)
    return paths


def read_summary_value(num_nodes: int, algorithm: str, column: str) -> float:
    path = RESULTS_DIR / f"summary_tsp{num_nodes}.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["algorithm"] == algorithm:
                return float(row[column])
    raise ValueError(f"Could not find {algorithm} in {path}")


def render_evolution_chart(
    values_by_model: dict[str, list[float]],
    axis_label: str,
    output_path: Path,
    series_order: list[str],
    image_size: tuple[int, int] = (980, 620),
) -> None:
    img = Image.new("RGB", image_size, WHITE)
    draw = ImageDraw.Draw(img)

    width, height = image_size
    left, top, right, bottom = 92, 76, width - 210, height - 92
    all_values = [value for values in values_by_model.values() for value in values]
    y_min = 0.0
    y_max = max(all_values) * 1.12
    if y_max == 0:
        y_max = 1.0

    def map_x(size: int) -> int:
        return int(left + (size - min(SIZES)) / (max(SIZES) - min(SIZES)) * (right - left))

    def map_y(value: float) -> int:
        return int(bottom - (value - y_min) / (y_max - y_min) * (bottom - top))

    for i in range(6):
        value = y_min + i * (y_max - y_min) / 5
        y = map_y(value)
        draw.line((left, y, right, y), fill=GRID, width=1)
        label = f"{value:.4f}" if y_max < 0.1 else f"{value:.3f}"
        bbox = draw.textbbox((0, 0), label, font=FONT_XS)
        draw_text(draw, (left - 12 - (bbox[2] - bbox[0]), y - 8), label, FONT_XS, MUTED)

    for size in SIZES:
        x = map_x(size)
        draw.line((x, top, x, bottom), fill=GRID, width=1)
        draw_centered_text(draw, x, bottom + 18, str(size), FONT_SM, MUTED)

    draw.line((left, bottom, right, bottom), fill=AXIS, width=2)
    draw.line((left, top, left, bottom), fill=AXIS, width=2)
    draw_text(draw, (52, top - 26), axis_label, FONT_SM, MUTED)
    draw_centered_text(draw, (left + right) // 2, bottom + 52, "nodes", FONT_SM, MUTED)

    label_positions: list[tuple[int, str, tuple[int, int, int], int]] = []
    for idx, algorithm in enumerate(series_order):
        values = values_by_model[algorithm]
        color = SERIES_COLORS[idx % len(SERIES_COLORS)]
        points = [(map_x(size), map_y(value)) for size, value in zip(SIZES, values)]
        draw.line(points, fill=color, width=3)
        for x, y in points:
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=WHITE, outline=color, width=3)
        label = LABELS[algorithm].replace("\n", " ")
        label_positions.append((points[-1][1], label, color, points[-1][0]))

    min_label_gap = 16
    ordered_positions = sorted(label_positions, key=lambda item: item[0])
    adjusted: list[tuple[int, str, tuple[int, int, int], int]] = []
    for y, label, color, x in ordered_positions:
        if adjusted and y - adjusted[-1][0] < min_label_gap:
            y = adjusted[-1][0] + min_label_gap
        adjusted.append((y, label, color, x))

    overflow = adjusted[-1][0] - (bottom - 12) if adjusted else 0
    if overflow > 0:
        adjusted = [(y - overflow, label, color, x) for y, label, color, x in adjusted]
    if adjusted and adjusted[0][0] < top + 8:
        shift = top + 8 - adjusted[0][0]
        adjusted = [(y + shift, label, color, x) for y, label, color, x in adjusted]

    for y, label, color, x in adjusted:
        draw.line((x + 8, y, x + 28, y), fill=color, width=2)
        draw_text(draw, (x + 34, y - 8), label, FONT_XS, color)
    img.save(output_path)


def generate_evolution_charts() -> list[Path]:
    output_dir = OUTPUT_ROOT / "model_evolution"
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    charts = [
        ("median_score", "score", "median_score_evolution.png"),
        ("median_time", "temps (s)", "median_time_evolution.png"),
    ]
    for column, axis_label, filename in charts:
        values_by_model = {
            algorithm: [read_summary_value(size, algorithm, column) for size in SIZES]
            for algorithm in ["gcn", "transformer"]
        }
        output = output_dir / filename
        render_evolution_chart(values_by_model, axis_label, output, ["gcn", "transformer"])
        paths.append(output)

    all_method_charts = [
        ("median_score", "score", "median_score_evolution_all_methods.png"),
        ("median_time", "temps (s)", "median_time_evolution_all_methods.png"),
    ]
    for column, axis_label, filename in all_method_charts:
        values_by_model = {
            algorithm: [read_summary_value(size, algorithm, column) for size in SIZES]
            for algorithm in ALGORITHMS
        }
        output = output_dir / filename
        ordered_algorithms = sorted(ALGORITHMS, key=lambda algorithm: values_by_model[algorithm][-1])
        render_evolution_chart(
            values_by_model,
            axis_label,
            output,
            ordered_algorithms,
            image_size=(1180, 700),
        )
        paths.append(output)
    return paths


def generate_metric(metric: MetricConfig) -> list[Path]:
    output_dir = OUTPUT_ROOT / metric.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    all_stats = {num_nodes: compute_box_stats(num_nodes, metric) for num_nodes in SIZES}

    paths: list[Path] = []
    for num_nodes, stats in all_stats.items():
        output = output_dir / f"{metric.file_prefix}_tsp{num_nodes}.png"
        render_single(num_nodes, stats, metric, output)
        paths.append(output)

    combined = output_dir / f"{metric.file_prefix}s_all_sizes.png"
    render_combined(all_stats, metric, combined)
    paths.append(combined)
    return paths


def generate() -> list[Path]:
    paths: list[Path] = []
    for metric in METRICS:
        paths.extend(generate_metric(metric))
    paths.extend(generate_score_time_boxplots())
    paths.extend(generate_evolution_charts())
    return paths


def main() -> None:
    paths = generate()
    print("Generated boxplot figures:")
    for path in paths:
        print(f"  {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
