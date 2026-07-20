"""Generate learning-curve figures for the six trained neural models."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "Visualizations" / "output"
REPORTS = {
    "GCN": ROOT / "GCNReports.txt",
    "Transformer": ROOT / "TransformersReports.txt",
}
SIZES = [10, 20, 30]

WHITE = (255, 255, 255)
INK = (17, 24, 39)
MUTED = (100, 116, 139)
GRID = (226, 232, 240)
AXIS = (71, 85, 105)
TRAIN = (37, 99, 235)
VAL = (22, 163, 74)


@dataclass(frozen=True)
class Curve:
    model: str
    num_nodes: int
    epochs: list[int]
    train: list[float]
    val: list[float]


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    if Path(path).exists():
        return ImageFont.truetype(path, size)
    return ImageFont.load_default()


FONT_XS = font(14)
FONT_SM = font(17)
FONT_MD = font(22)
FONT_TITLE = font(34, bold=True)


def parse_report(model: str, report_path: Path, num_nodes: int) -> Curve:
    epochs: list[int] = []
    train: list[float] = []
    val: list[float] = []
    in_section = False

    num_nodes_re = re.compile(r"num_nodes\s*=\s*(\d+)")
    train_re = re.compile(r"Epoch\s+(\d+)\s+\|\s+Train Loss:\s+([0-9.]+)")
    val_re = re.compile(r"Epoch\s+(\d+)\s+\|\s+Val Loss:\s+([0-9.]+)")

    for line in report_path.read_text(encoding="utf-8").splitlines():
        match = num_nodes_re.search(line)
        if match:
            if in_section and epochs:
                break
            in_section = int(match.group(1)) == num_nodes
            continue
        if not in_section:
            continue

        match = train_re.search(line)
        if match:
            epochs.append(int(match.group(1)))
            train.append(float(match.group(2)))
            continue

        match = val_re.search(line)
        if match:
            val.append(float(match.group(2)))

    if not epochs or len(epochs) != len(train) or len(train) != len(val):
        raise ValueError(f"Could not parse complete {model} TSP{num_nodes} curve from {report_path}")
    return Curve(model=model, num_nodes=num_nodes, epochs=epochs, train=train, val=val)


def draw_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, fnt: ImageFont.ImageFont, fill=INK) -> None:
    draw.text(xy, text, font=fnt, fill=fill)


def draw_curve(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], curve: Curve) -> None:
    left, top, right, bottom = box
    draw.rounded_rectangle(box, radius=6, fill=WHITE, outline=GRID, width=1)
    draw_text(draw, (left + 24, top + 18), f"{curve.model} TSP{curve.num_nodes}", FONT_MD)

    plot = (left + 68, top + 76, right - 30, bottom - 58)
    px0, py0, px1, py1 = plot
    losses = curve.train + curve.val
    y_min = min(losses) - 0.02
    y_max = max(losses) + 0.02
    x_min, x_max = min(curve.epochs), max(curve.epochs)

    def map_x(epoch: int) -> int:
        return int(px0 + (epoch - x_min) / (x_max - x_min) * (px1 - px0))

    def map_y(loss: float) -> int:
        return int(py1 - (loss - y_min) / (y_max - y_min) * (py1 - py0))

    for i in range(5):
        y = int(py0 + i * (py1 - py0) / 4)
        draw.line((px0, y, px1, y), fill=GRID, width=1)
    for epoch in range(0, x_max + 1, 20):
        if epoch >= x_min:
            x = map_x(epoch)
            draw.line((x, py0, x, py1), fill=GRID, width=1)
            draw_text(draw, (x - 10, py1 + 12), str(epoch), FONT_XS, MUTED)

    draw.line((px0, py1, px1, py1), fill=AXIS, width=2)
    draw.line((px0, py0, px0, py1), fill=AXIS, width=2)

    train_points = [(map_x(e), map_y(v)) for e, v in zip(curve.epochs, curve.train)]
    val_points = [(map_x(e), map_y(v)) for e, v in zip(curve.epochs, curve.val)]
    draw.line(train_points, fill=TRAIN, width=3)
    draw.line(val_points, fill=VAL, width=3)

    best_val = min(curve.val)
    best_epoch = curve.epochs[curve.val.index(best_val)]
    draw_text(draw, (left + 24, bottom - 38), f"best val: {best_val:.4f} @ epoch {best_epoch}", FONT_SM, MUTED)


def render_single_curve(curve: Curve, output_path: Path) -> None:
    img = Image.new("RGB", (980, 560), WHITE)
    draw = ImageDraw.Draw(img)
    draw.line((650, 48, 705, 48), fill=TRAIN, width=5)
    draw_text(draw, (718, 37), "Train", FONT_SM, MUTED)
    draw.line((805, 48, 860, 48), fill=VAL, width=5)
    draw_text(draw, (873, 37), "Validation", FONT_SM, MUTED)
    draw_curve(draw, (55, 80, 925, 510), curve)
    img.save(output_path)


def generate() -> list[Path]:
    curves_dir = OUTPUT_DIR / "learning_curves"
    curves_dir.mkdir(parents=True, exist_ok=True)
    curves = [
        parse_report(model, report, size)
        for model, report in REPORTS.items()
        for size in SIZES
    ]

    paths: list[Path] = []
    for curve in curves:
        filename = f"{curve.model.lower()}_tsp{curve.num_nodes}_learning_curve.png"
        output = curves_dir / filename
        render_single_curve(curve, output)
        paths.append(output)
    return paths


def main() -> None:
    paths = generate()
    print("Generated learning-curve figures:")
    for path in paths:
        print(f"  {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
