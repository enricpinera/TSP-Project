import os
import re
import math
import statistics
from typing import Dict, List, Tuple

import pandas as pd


CSV_DECIMALS = 6
SCORE_TOL = 5e-6
EXACT_SCORE_TOL = 5e-6
TARGET_NUM_NODES = [10, 20, 30]
EXPECTED_ALGORITHMS = [
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
SUMMARY_COLUMNS = [
    "num_nodes",
    "algorithm",
    "file",
    "rows",
    "valid_rows",
    "invalid_rows",
    "mean_score",
    "median_score",
    "std_score",
    "min_score",
    "p25_score",
    "p75_score",
    "p90_score",
    "p95_score",
    "max_score",
    "ci95_score_lower",
    "ci95_score_upper",
    "exact_rows",
    "exact_rate",
    "negative_rows",
    "negative_rate",
    "mean_time",
    "median_time",
    "std_time",
    "min_time",
    "p25_time",
    "p75_time",
    "p90_time",
    "p95_time",
    "max_time",
    "ci95_time_lower",
    "ci95_time_upper",
    "total_time",
    "rank_quality",
    "rank_speed",
    "status",
]

def format_csv_float(value: float) -> str:
    rounded = round(float(value), CSV_DECIMALS)
    if rounded == 0.0:
        rounded = 0.0
    return format(rounded, f".{CSV_DECIMALS}f")


def format_optional_float(value: float | None) -> str:
    if value is None:
        return ""
    return format_csv_float(value)


def percentile(values: List[float], percent: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    idx = (len(ordered) - 1) * percent
    lower = int(idx)
    upper = min(lower + 1, len(ordered) - 1)
    weight = idx - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def confidence_interval_95(values: List[float]) -> Tuple[float | None, float | None]:
    if not values:
        return None, None
    mean_value = sum(values) / len(values)
    if len(values) == 1:
        return mean_value, mean_value
    sample_std = statistics.stdev(values)
    margin = 1.96 * sample_std / math.sqrt(len(values))
    return mean_value - margin, mean_value + margin


def parse_tour(tour_text: str) -> List[int]:
    if tour_text is None:
        raise ValueError("missing tour")
    text = tour_text.strip()
    if not (text.startswith("{") and text.endswith("}")):
        raise ValueError(f"invalid tour format: {tour_text}")
    inner = text[1:-1].strip()
    if inner == "":
        return []
    parts = [p.strip() for p in inner.split(",")]
    return [int(p) for p in parts]


def validate_cycle(tour: List[int], label: str) -> List[str]:
    errors = []
    if len(tour) < 2:
        errors.append(f"{label}: tour must have at least 2 nodes")
        return errors
    if tour[0] != tour[-1]:
        errors.append(f"{label}: first and last node differ ({tour[0]} != {tour[-1]})")

    inner = tour[:-1]
    if len(set(inner)) != len(inner):
        errors.append(f"{label}: repeated node before closing the cycle")
    if any(node <= 0 for node in inner):
        errors.append(f"{label}: node ids must be >= 1")
    return errors


def detect_algorithm_columns(fieldnames: List[str]) -> Tuple[str, str, str]:
    tour_cols = [c for c in fieldnames if c.endswith("_tour") and c != "optimal_tour"]
    length_cols = [c for c in fieldnames if c.endswith("_tour_length") and c != "optimal_tour_length"]

    if len(tour_cols) != 1 or len(length_cols) != 1:
        raise ValueError("expected exactly one algorithm tour column and one algorithm tour_length column")

    tour_col = tour_cols[0]
    length_col = length_cols[0]
    algorithm_from_tour = tour_col[: -len("_tour")]
    algorithm_from_length = length_col[: -len("_tour_length")]
    if algorithm_from_tour != algorithm_from_length:
        raise ValueError("algorithm tour and tour_length columns do not match")

    return algorithm_from_tour, tour_col, length_col


def extract_num_nodes_from_filename(csv_path: str) -> int:
    name = os.path.basename(csv_path)
    match = re.search(r"_tsp(\d+)\.csv$", name)
    if not match:
        raise ValueError(f"cannot extract num_nodes from filename: {name}")
    return int(match.group(1))


def audit_file(csv_path: str, num_nodes: int) -> Tuple[Dict[str, str], List[str]]:
    base_required = {"optimal_tour", "optimal_tour_length", "score", "time"}
    errors: List[str] = []
    total_rows = 0
    valid_rows = 0
    scores: List[float] = []
    times: List[float] = []

    df = pd.read_csv(csv_path)
    fieldnames = list(df.columns)

    missing = sorted(base_required - set(fieldnames))
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    algorithm, tour_col, length_col = detect_algorithm_columns(fieldnames)

    for idx, row in df.iterrows():
        total_rows += 1
        line_no = idx + 2
        line_issues: List[str] = []

        try:
            optimal_tour = parse_tour(str(row["optimal_tour"]))
            pred_tour = parse_tour(str(row[tour_col]))
        except Exception as exc:
            line_issues.append(f"line {line_no}: cannot parse tour ({exc})")
            optimal_tour = []
            pred_tour = []

        line_issues.extend([f"line {line_no}: {msg}" for msg in validate_cycle(optimal_tour, "optimal_tour")])
        line_issues.extend([f"line {line_no}: {msg}" for msg in validate_cycle(pred_tour, tour_col)])

        if optimal_tour and pred_tour:
            opt_nodes = optimal_tour[:-1]
            pred_nodes = pred_tour[:-1]
            if len(opt_nodes) != len(pred_nodes):
                line_issues.append(
                    f"line {line_no}: node count mismatch ({len(opt_nodes)} vs {len(pred_nodes)})"
                )
            if set(opt_nodes) != set(pred_nodes):
                line_issues.append(f"line {line_no}: predicted node set differs from optimal node set")

        try:
            opt_len = float(row["optimal_tour_length"])
            pred_len = float(row[length_col])
            score = float(row["score"])
            duration = float(row["time"])
        except Exception as exc:
            line_issues.append(f"line {line_no}: cannot parse numeric columns ({exc})")
            opt_len = pred_len = score = duration = 0.0

        if duration < 0:
            line_issues.append(f"line {line_no}: time is negative ({duration})")

        if opt_len <= 0:
            line_issues.append(f"line {line_no}: optimal_tour_length must be > 0 ({opt_len})")
        else:
            expected_score = pred_len / opt_len - 1.0
            if abs(score - expected_score) > SCORE_TOL:
                line_issues.append(
                    f"line {line_no}: score mismatch (csv={score}, expected={expected_score})"
                )

        if line_issues:
            errors.extend(line_issues)
        else:
            valid_rows += 1
            scores.append(score)
            times.append(duration)

    invalid_rows = total_rows - valid_rows
    mean_score = sum(scores) / len(scores) if scores else None
    median_score = statistics.median(scores) if scores else None
    std_score = statistics.pstdev(scores) if len(scores) > 1 else (0.0 if scores else None)
    min_score = min(scores) if scores else None
    p25_score = percentile(scores, 0.25)
    p75_score = percentile(scores, 0.75)
    p90_score = percentile(scores, 0.90)
    p95_score = percentile(scores, 0.95)
    max_score = max(scores) if scores else None
    ci95_score_lower, ci95_score_upper = confidence_interval_95(scores)
    exact_rows = sum(1 for score in scores if abs(score) <= EXACT_SCORE_TOL)
    negative_rows = sum(1 for score in scores if score < -EXACT_SCORE_TOL)
    exact_rate = exact_rows / len(scores) if scores else None
    negative_rate = negative_rows / len(scores) if scores else None
    mean_time = sum(times) / len(times) if times else None
    median_time = statistics.median(times) if times else None
    std_time = statistics.pstdev(times) if len(times) > 1 else (0.0 if times else None)
    min_time = min(times) if times else None
    p25_time = percentile(times, 0.25)
    p75_time = percentile(times, 0.75)
    p90_time = percentile(times, 0.90)
    p95_time = percentile(times, 0.95)
    max_time = max(times) if times else None
    ci95_time_lower, ci95_time_upper = confidence_interval_95(times)
    total_time = sum(times) if times else None
    status = "ok" if invalid_rows == 0 else "issues_found"
    summary = {
        "num_nodes": str(num_nodes),
        "algorithm": algorithm,
        "file": os.path.basename(csv_path),
        "rows": str(total_rows),
        "valid_rows": str(valid_rows),
        "invalid_rows": str(invalid_rows),
        "mean_score": format_optional_float(mean_score),
        "median_score": format_optional_float(median_score),
        "std_score": format_optional_float(std_score),
        "min_score": format_optional_float(min_score),
        "p25_score": format_optional_float(p25_score),
        "p75_score": format_optional_float(p75_score),
        "p90_score": format_optional_float(p90_score),
        "p95_score": format_optional_float(p95_score),
        "max_score": format_optional_float(max_score),
        "ci95_score_lower": format_optional_float(ci95_score_lower),
        "ci95_score_upper": format_optional_float(ci95_score_upper),
        "exact_rows": str(exact_rows),
        "exact_rate": format_optional_float(exact_rate),
        "negative_rows": str(negative_rows),
        "negative_rate": format_optional_float(negative_rate),
        "mean_time": format_optional_float(mean_time),
        "median_time": format_optional_float(median_time),
        "std_time": format_optional_float(std_time),
        "min_time": format_optional_float(min_time),
        "p25_time": format_optional_float(p25_time),
        "p75_time": format_optional_float(p75_time),
        "p90_time": format_optional_float(p90_time),
        "p95_time": format_optional_float(p95_time),
        "max_time": format_optional_float(max_time),
        "ci95_time_lower": format_optional_float(ci95_time_lower),
        "ci95_time_upper": format_optional_float(ci95_time_upper),
        "total_time": format_optional_float(total_time),
        "rank_quality": "",
        "rank_speed": "",
        "status": status,
    }
    return summary, errors


def missing_summary(num_nodes: int, algorithm: str) -> Dict[str, str]:
    row = {column: "" for column in SUMMARY_COLUMNS}
    row.update(
        {
            "num_nodes": str(num_nodes),
            "algorithm": algorithm,
            "status": "missing_result_file",
        }
    )
    return row


def add_ranks(summaries: List[Dict[str, str]]) -> None:
    valid_rows = [row for row in summaries if row["status"] == "ok" and row["mean_score"] and row["mean_time"]]

    quality_sorted = sorted(valid_rows, key=lambda row: float(row["mean_score"]))
    for rank, row in enumerate(quality_sorted, start=1):
        row["rank_quality"] = str(rank)

    speed_sorted = sorted(valid_rows, key=lambda row: float(row["mean_time"]))
    for rank, row in enumerate(speed_sorted, start=1):
        row["rank_speed"] = str(rank)


def print_summary_csv(summary_path: str) -> None:
    summary_df = pd.read_csv(summary_path, dtype=str, keep_default_na=False)
    print(summary_df.to_csv(index=False).strip())


def main():
    results_dir = "Results"
    csv_files = [
        os.path.join(results_dir, name)
        for name in sorted(os.listdir(results_dir))
        if name.endswith(".csv") and not name.startswith("summary")
    ]

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {results_dir}")

    grouped_files: Dict[int, List[str]] = {}
    for csv_path in csv_files:
        num_nodes = extract_num_nodes_from_filename(csv_path)
        grouped_files.setdefault(num_nodes, []).append(csv_path)

    any_errors = False
    for num_nodes in TARGET_NUM_NODES:
        summaries = []
        all_errors = {}
        summary_filename = f"summary_tsp{num_nodes}.csv"
        summary_path = os.path.join(results_dir, summary_filename)
        files_by_algorithm = {}

        for csv_path in sorted(grouped_files.get(num_nodes, [])):
            summary, errors = audit_file(csv_path, num_nodes)
            files_by_algorithm[summary["algorithm"]] = summary
            if errors:
                all_errors[os.path.basename(csv_path)] = errors

        for algorithm in EXPECTED_ALGORITHMS:
            summaries.append(files_by_algorithm.get(algorithm, missing_summary(num_nodes, algorithm)))

        add_ranks(summaries)

        summary_df = pd.DataFrame(summaries, columns=SUMMARY_COLUMNS, dtype=str)
        summary_df.to_csv(summary_path, index=False)

        print(f"Summary written to: {summary_path}")
        print("")
        print_summary_csv(summary_path)
        print("")

        if all_errors:
            any_errors = True
            print(f"Validation errors found for num_nodes={num_nodes}:")
            for file_name in sorted(all_errors.keys()):
                issues = all_errors[file_name]
                print(f"- {file_name}: {len(issues)} issue(s)")
                for issue in issues[:10]:
                    print(f"  {issue}")
                if len(issues) > 10:
                    print(f"  ... and {len(issues) - 10} more")
        else:
            print(f"All available CSV files passed validation for num_nodes={num_nodes}.")

        missing = [row["algorithm"] for row in summaries if row["status"] == "missing_result_file"]
        if missing:
            print(f"Missing result files for num_nodes={num_nodes}: {', '.join(missing)}")
        else:
            print(f"All expected result files are present for num_nodes={num_nodes}.")
        print("")

    if not any_errors:
        print("All summaries completed without validation errors.")


if __name__ == "__main__":
    main()
