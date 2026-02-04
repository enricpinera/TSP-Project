from pathlib import Path
import shutil
import tarfile
import gzip
import json
import csv
import time
import tsplib95
import networkx as nx


def extract_tsp_archive(tar_path: Path, extract_path: Path):
    """
    Extract a TSPLIB .tar archive and decompress any .gz files inside it.
    Returns a list of .tsp file paths.
    """

    if not tar_path.exists():
        raise FileNotFoundError(f"Archive not found: {tar_path}")

    if extract_path.exists():
        shutil.rmtree(extract_path)
    extract_path.mkdir(parents=True, exist_ok=True)

    with tarfile.open(tar_path, "r") as tar:
        tar.extractall(path=extract_path)

    for gz_file in extract_path.glob("*.gz"):
        output_file = extract_path / gz_file.stem
        with gzip.open(gz_file, "rb") as f_in, open(output_file, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        gz_file.unlink()

    print("Archive extracted and .gz files decompressed.")

    return sorted(extract_path.glob("*.tsp"))


def validate_tour(G, tour):
    """
    Validate a TSP tour for a given graph.
    Returns True if valid, otherwise False.
    Prints diagnostics when invalid and a confirmation when valid.
    """

    expected_nodes = set(G.nodes())
    tour_set = set(tour)

    # Wrong node set
    if tour_set != expected_nodes:
        print(f"{G.name}: ❌ Invalid tour")
        missing = expected_nodes - tour_set
        extra = tour_set - expected_nodes
        if missing:
            print(f"  - Missing nodes: {missing}")
        if extra:
            print(f"  - Extra nodes: {extra}")
        return False

    # Wrong length
    if len(tour) != len(expected_nodes):
        print(f"{G.name}: ❌ Invalid tour")
        print(f"  - Incorrect length: {len(tour)} (expected {len(expected_nodes)})")
        return False

    print("✅ Valid tour")
    return True


def nearest_neighbor_tour(G):
    """
    Construct a TSP tour using the Nearest Neighbor heuristic.

    The algorithm starts from an arbitrary node and repeatedly selects
    the closest unvisited node until all nodes have been visited.
    Returns the resulting tour as an ordered list of node identifiers.
    """

    nodes = list(G.nodes())
    start = nodes[0]
    unvisited = set(nodes)
    unvisited.remove(start)

    tour = [start]
    current = start

    while unvisited:
        next_node = min(unvisited, key=lambda x: G[current][x]['weight'])
        tour.append(next_node)
        unvisited.remove(next_node)
        current = next_node

    return tour


def two_opt(G, initial_tour):
    """
    Apply an optimized 2‑Opt local search heuristic to improve a TSP tour.

    This implementation uses:
      - Precomputed distance lookup for fast edge evaluation.
      - Incremental cost updates (O(1) per swap).
      - First‑improvement strategy to accelerate convergence.

    Returns the best tour discovered.
    """

    # Precompute distances for O(1) lookup
    dist = dict(nx.all_pairs_dijkstra_path_length(G))

    def edge_cost(a, b):
        return dist[a][b]

    def tour_cost_change(tour, i, j):
        """
        Compute the cost difference produced by reversing the segment tour[i:j].
        Only the affected edges are evaluated.
        """
        a, b = tour[i - 1], tour[i]
        c, d = tour[j - 1], tour[j]

        before = edge_cost(a, b) + edge_cost(c, d)
        after = edge_cost(a, c) + edge_cost(b, d)

        return after - before

    tour = initial_tour.copy()
    n = len(tour)
    improved = True

    while improved:
        improved = False

        for i in range(1, n - 2):
            for j in range(i + 2, n - 1):

                delta = tour_cost_change(tour, i, j)
                if delta < 0:
                    # Apply the improving swap
                    tour[i:j] = reversed(tour[i:j])
                    improved = True
                    break  # First‑improvement: restart search
            if improved:
                break

    return tour


def main():
    """
    Execute a full TSP heuristic benchmark over a collection of TSPLIB instances.
    
    The function performs the following steps:
        1. Extracts and decompresses a TSPLIB .tar archive.
        2. Loads each TSP instance and constructs its corresponding graph.
        3. Computes tours using several classical heuristics:
            - Nearest Neighbor
            - 2‑Opt local search (optimized implementation)
            - Greedy heuristic
            - Christofides algorithm
            - Simulated Annealing
            - Threshold Accepting
        4. Validates each tour and prints diagnostic information.
        5. Stores all tours and execution times in separate CSV files.
        6. Cleans up temporary extracted files.
    
    This script provides a unified framework for comparing heuristic performance across multiple TSP instances.
    """

    tar_path = Path("Datasets/ALL_tsp.tar")
    extract_path = Path("Datasets/ALL_tsp")
    tsp_files = extract_tsp_archive(tar_path, extract_path)

    tours = []
    times = []

    for i, tsp_file in enumerate(tsp_files):
        problem = tsplib95.load(tsp_file)
        print(f"\n---Graph {i}: {problem.name}---")

        # Process only symmetric TSP instances
        if problem.type != "TSP":
            print(f"⚠️ Skipped (TYPE: {problem.type})")
            continue

        # Skip large instances
        if problem.dimension > 1000:
            print(f"⚠️ Skipped (DIMENSION: {problem.dimension})")
            continue

        # Load graph to NetworkX
        print("Loading graph...")
        G = problem.get_graph()
        G = nx.Graph(G) # Ensure graph is undirected

        # Nearest Neighbor
        print("Nearest Neighbor Tour:")
        start = time.perf_counter()
        tour_nn = nearest_neighbor_tour(G)
        nn_time = time.perf_counter() - start
        valid_nn = validate_tour(G, tour_nn)

        # 2-Opt
        print("2-Opt Improved Tour:")
        start = time.perf_counter()
        tour_2opt = two_opt(G, tour_nn)
        twoopt_time = time.perf_counter() - start
        valid_2opt = validate_tour(G, tour_2opt)

        # Greedy
        print("Greedy Tour:")
        start = time.perf_counter()
        tour_greedy = nx.algorithms.approximation.greedy_tsp(G)[:-1]
        greedy_time = time.perf_counter() - start
        valid_greedy = validate_tour(G, tour_greedy)

        # Christfoides
        print("Christofides Tour:")
        start = time.perf_counter()
        tour_christofides = nx.algorithms.approximation.christofides(G)[:-1]
        christofides_time = time.perf_counter() - start
        valid_christofides = validate_tour(G, tour_christofides)

        # Simulated Annealing
        print("Simulated Annealing Tour:")
        start = time.perf_counter()
        tour_sa = nx.algorithms.approximation.simulated_annealing_tsp(G, nx.algorithms.approximation.greedy_tsp(G))[:-1]
        sa_time = time.perf_counter() - start
        valid_sa = validate_tour(G, tour_sa)

        # Threshold Accepting
        print("Threshold Accepting Tour:")
        start = time.perf_counter()
        tour_ta = nx.algorithms.approximation.threshold_accepting_tsp(G, nx.algorithms.approximation.greedy_tsp(G))[:-1]
        ta_time = time.perf_counter() - start
        valid_ta = validate_tour(G, tour_ta)

        # Store results
        tours.append({
            "graph": G.name,
            "nn": json.dumps(tour_nn) if valid_nn else None,
            "two_opt": json.dumps(tour_2opt) if valid_2opt else None,
            "greedy": json.dumps(tour_greedy) if valid_greedy else None,
            "christofides": json.dumps(tour_christofides) if valid_christofides else None,
            "simulated_annealing": json.dumps(tour_sa) if valid_sa else None,
            "threshold_accepting": json.dumps(tour_ta) if valid_ta else None
            })
        
        times.append({
            "graph": G.name,
            "nn": nn_time,
            "two_opt": twoopt_time,
            "greedy": greedy_time,
            "christofides": christofides_time,
            "simulated_annealing": sa_time,
            "threshold_accepting": ta_time
            })
    
    # Common CSV structure
    fieldnames = [
        "graph",
        "nn",
        "two_opt",
        "greedy", "christofides", "simulated_annealing", "threshold_accepting"
        "christofides",
        "simulated_annealing",
        "threshold_accepting"
    ]

    # Save tours
    tours_path = Path("Datasets/tours_heuristics.csv")
    with open(tours_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(tours)
    print(f"Tours saved to {tours_path}")

    # Save times
    times_path = Path("Datasets/times_heuristics.csv")
    with open(times_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(times)
    print(f"Times saved to {times_path}")

    shutil.rmtree(extract_path)
    

if __name__ == "__main__":
    main()

