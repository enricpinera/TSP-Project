from pathlib import Path
import shutil
import tarfile
import gzip

import tsplib95
import networkx as nx
import torch
from torch_geometric.data import Data


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


def prepare_graph(G):
    """
    Prepare a TSPLIB graph loaded with tsplib95:
    - ensure undirected
    - remove self-loops
    - keep only edge weight
    - keep only node id + initial/current/target
    """

    # Ensure undirected structure
    G = nx.Graph(G)

    # Remove self-loops
    G.remove_edges_from(nx.selfloop_edges(G))

    # Initialize node attributes
    first_node = min(G.nodes)
    for node in G.nodes:
        G.nodes[node].clear()
        G.nodes[node]["initial"] = int(node == first_node)
        G.nodes[node]["current"] = int(node == first_node)
        G.nodes[node]["target"] = 0

    # Keep only edge weight
    for u, v, attrs in G.edges(data=True):
        w = attrs.get("weight", None)
        attrs.clear()
        attrs["weight"] = w

    return G


def nx_to_pyg(G):
    """
    Convert a prepared NetworkX TSP graph into a PyTorch Geometric Data object.
    Keeps:
      - x: [initial, current]
      - edge_index (bidirectional)
      - edge_attr (weight)
      - node_id (original TSPLIB ids)
      - y: index of target node (0-based)
    """

    # Sorted node list for consistent indexing
    nodes = sorted(G.nodes())
    mapping = {node: i for i, node in enumerate(nodes)}

    # Node features
    x = torch.tensor(
        [
            [
                G.nodes[node]["initial"],
                G.nodes[node]["current"]
            ]
            for node in nodes
        ],
        dtype=torch.float
    )

    # Original TSPLIB node IDs
    node_id = torch.tensor(nodes, dtype=torch.long)

    # Target node (converted to PyTorch index)
    target_node = next((node for node in nodes if G.nodes[node]["target"] == 1), None)
    y = torch.tensor(
        mapping[target_node] if target_node is not None else -1,
        dtype=torch.long
    )

    # Edges (bidirectional)
    edge_index_list = []
    edge_attr_list = []

    for u, v, attrs in G.edges(data=True):
        i, j = mapping[u], mapping[v]
        w = attrs["weight"]

        edge_index_list.append([i, j])
        edge_attr_list.append([w])

        edge_index_list.append([j, i])
        edge_attr_list.append([w])

    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)

    # Build Data object
    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        node_id=node_id,
        y=y
    )


def generate_training_graphs(G, tour):
    """
    Given a prepared graph G and a normalized tour,
    generate one graph per decision (num_nodes - 2).
    """

    graphs = []

    # Copy nodes to track which remain
    remaining = list(tour)

    initial = tour[0]

    for step in range(len(tour) - 2):
        current = tour[step]
        target = tour[step + 1]

        # Build a fresh copy of the graph
        H = G.copy()

        # Remove visited nodes except initial and current
        visited = tour[:step]
        for v in visited:
            if v != initial:
                if v in H:
                    H.remove_node(v)

        # Reset attributes
        for node in H.nodes:
            H.nodes[node]["initial"] = int(node == initial)
            H.nodes[node]["current"] = int(node == current)
            H.nodes[node]["target"] = int(node == target)

        graphs.append(H)

    return graphs


def load_opt_tour(tour_path: Path):
    """
    Load a TSPLIB .opt.tour file and return the tour as a list of node IDs.
    Handles:
      - one node per line
      - multiple nodes per line
      - -1 or EOF termination
    """
    tour = []
    reading = False

    with open(tour_path, "r") as f:
        for line in f:
            line = line.strip()

            if line == "TOUR_SECTION":
                reading = True
                continue

            if not reading:
                continue

            if line == "-1" or line == "EOF":
                break

            # Split line into tokens (handles multiple numbers per line)
            parts = line.split()
            for p in parts:
                tour.append(int(p))

    # Remove possible duplicated last node
    if len(tour) > 1 and tour[0] == tour[-1]:
        tour = tour[:-1]

    return tour


def save_graph(data: Data, pt_path: Path, count: int):
    """
    Save a PyTorch Geometric Data object to disk with a sequential filename.
    """
    file_path = pt_path / f"{count:05d}.pt"
    torch.save(data, file_path)
    return count + 1


def main():
    """
    Main function
    """

    # Prepare output directory
    pt_path = Path("Datasets/train_pyg")
    if pt_path.exists():
        shutil.rmtree(pt_path)
    pt_path.mkdir(parents=True)

    count = 0

    # Extract archive
    tar_path = Path("Datasets/ALL_tsp.tar")
    extract_path = Path("Datasets/ALL_tsp")
    tsp_files = extract_tsp_archive(tar_path, extract_path)

    # Process each TSP instance
    valid_i = 0
    for tsp_file in tsp_files:
        name = tsp_file.stem
        tour_path = extract_path / f"{name}.opt.tour"

        # Skip if no optimal tour
        if not tour_path.exists():
            continue

        problem = tsplib95.load(tsp_file)
        print(f"\n---Graph {valid_i}: {problem.name}---")

        # Process only symmetric TSP instances
        if problem.type != "TSP":
            print(f"⚠️ Skipped (TYPE: {problem.type})")
            continue

        # Skip large instances
        if problem.dimension > 1000:
            print(f"⚠️ Skipped (DIMENSION: {problem.dimension})")
            continue

        # Load graph
        print("  Loading graph...")
        G = problem.get_graph()

        # Clean graph
        print("  Preparing graph...")
        G = prepare_graph(G)

        # Load tour
        print("  Loading tour...")
        tour = load_opt_tour(tour_path)

        # Generate train graphs
        print("  Generating training graphs...")
        graphs = generate_training_graphs(G, tour)

        # Convert to PyTorch Geometric and save Data Object
        print("  Converting to PyTorch Geometric format and saving...")
        for H in graphs:
            data = nx_to_pyg(H)
            count = save_graph(data, pt_path, count)

        print("✅ Success!")
        valid_i += 1

    print(f"\nFinished. Saved {count} graphs to {pt_path}")
    shutil.rmtree(extract_path)


if __name__ == "__main__":
    main()

