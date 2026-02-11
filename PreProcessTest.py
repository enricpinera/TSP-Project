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
    - convert to 1-based indexing if needed
    - keep only edge weight
    - initialize node attributes (initial/current/target)
    """

    # Ensure undirected structure
    G = nx.Graph(G)

    # Remove self-loops
    G.remove_edges_from(nx.selfloop_edges(G))

    # Detect and convert 0-based graphs to 1-based ---
    nodes = sorted(G.nodes())
    if nodes[0] == 0:
        # Build mapping: 0→1, 1→2, ..., n-1→n
        mapping = {old: old + 1 for old in nodes}
        G = nx.relabel_nodes(G, mapping, copy=True)

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
    Convert a prepared 1-based NetworkX TSP graph into a PyTorch Geometric Data object.
    Used for TEST graphs (no target).
    Keeps:
      - x: [initial, current]
      - edge_index (bidirectional, 0-based)
      - edge_attr (weight)
    """

    # Sorted node list (1-based)
    nodes = sorted(G.nodes())

    # Convert to 0-based indexing for PyTorch
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

    # Build Data object (no y)
    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr
    )


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
    pt_path = Path("Datasets/test_pyg")
    if pt_path.exists():
        shutil.rmtree(pt_path)
    pt_path.mkdir(parents=True)

    count = 0

    # Extract archive
    tar_path = Path("Datasets/ALL_tsp.tar")
    extract_path = Path("Datasets/ALL_tsp")
    tsp_files = extract_tsp_archive(tar_path, extract_path)

    # Process each TSP instance
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

        # Load graph
        print("  Loading graph...")
        G = problem.get_graph()

        # Clean graph
        print("  Preparing graph...")
        G = prepare_graph(G)

        # Convert to PyTorch Geometric
        print("  Converting to PyTorch Geometric format...")
        data = nx_to_pyg(G)

        # Save Data object
        print("  Saving graph...")
        count = save_graph(data, pt_path, count)

        print("✅ Success!")

    print(f"\nFinished. Saved {count} graphs to {pt_path}")

    shutil.rmtree(extract_path)


if __name__ == "__main__":
    main()