"""
Solution #1: Graph Sampling for Scalability
============================================

Samples k-hop neighborhoods around important nodes to create
uniform-sized subgraphs (~200 nodes max).

This is the QUICK FIX to get training working on full SWE-bench dataset.

Expected impact: 60.5% → 65-70% accuracy
Implementation time: 2-3 days
"""
import torch
import numpy as np
from torch_geometric.data import Data
from typing import List, Set
import networkx as nx


def sample_k_hop_subgraph(
    data: Data,
    seed_nodes: List[int],
    k_hops: int = 3,
    max_nodes: int = 200
) -> Data:
    """
    Sample k-hop neighborhood around seed nodes.

    Args:
        data: Original graph (potentially huge)
        seed_nodes: Important nodes (e.g., changed lines, function roots)
        k_hops: How many hops to expand
        max_nodes: Maximum nodes to keep

    Returns:
        Sampled subgraph with ≤max_nodes nodes
    """
    # Convert to NetworkX for easier neighborhood computation
    edge_index_np = data.edge_index.numpy()
    G = nx.Graph()
    G.add_edges_from(edge_index_np.T)

    # Keep original edge_index as tensor for later
    edge_index = data.edge_index

    # Expand k-hop neighborhood from seed nodes
    subgraph_nodes: Set[int] = set(seed_nodes)

    for hop in range(k_hops):
        new_neighbors = set()
        for node in list(subgraph_nodes):
            if node in G:
                neighbors = set(G.neighbors(node))
                new_neighbors.update(neighbors)

        subgraph_nodes.update(new_neighbors)

        # Stop if we've exceeded max_nodes
        if len(subgraph_nodes) >= max_nodes:
            break

    # Limit to max_nodes (prioritize seed nodes)
    if len(subgraph_nodes) > max_nodes:
        # Keep all seed nodes + closest neighbors
        subgraph_nodes = set(seed_nodes)
        remaining = max_nodes - len(seed_nodes)

        # BFS to fill remaining slots
        queue = list(seed_nodes)
        visited = set(seed_nodes)

        while queue and len(subgraph_nodes) < max_nodes:
            node = queue.pop(0)
            if node in G:
                for neighbor in G.neighbors(node):
                    if neighbor not in visited:
                        subgraph_nodes.add(neighbor)
                        visited.add(neighbor)
                        queue.append(neighbor)
                        if len(subgraph_nodes) >= max_nodes:
                            break

    # Create node mapping (old_id -> new_id)
    subgraph_nodes_list = sorted(list(subgraph_nodes))
    node_mapping = {old_id: new_id for new_id, old_id in enumerate(subgraph_nodes_list)}

    # Extract subgraph node features
    new_x = data.x[subgraph_nodes_list]

    # Extract edges (only those within subgraph)
    mask = torch.tensor([
        edge_index[0, i] in subgraph_nodes and edge_index[1, i] in subgraph_nodes
        for i in range(edge_index.shape[1])
    ])

    subgraph_edges = edge_index[:, mask]

    # Remap edge indices to new node IDs
    new_edge_index = torch.zeros_like(subgraph_edges)
    for i in range(subgraph_edges.shape[1]):
        src = int(subgraph_edges[0, i])
        dst = int(subgraph_edges[1, i])
        new_edge_index[0, i] = node_mapping[src]
        new_edge_index[1, i] = node_mapping[dst]

    # Extract edge attributes if present
    new_edge_attr = data.edge_attr[mask] if data.edge_attr is not None else None

    # Create sampled graph
    sampled = Data(
        x=new_x,
        edge_index=new_edge_index,
        edge_attr=new_edge_attr,
        y=data.y,  # Keep original label
        sample_meta=data.sample_meta if hasattr(data, 'sample_meta') else {}
    )

    return sampled


def identify_seed_nodes(data: Data) -> List[int]:
    """
    Identify important seed nodes for sampling.

    For code graphs, this could be:
    - Changed lines (for bug fixes)
    - Function definition nodes
    - Complex control flow nodes
    - Nodes with high degree (many edges)

    For now, use simple heuristics:
    1. Root node (usually function def)
    2. High-degree nodes (important connectors)
    """
    seed_nodes = []

    # Heuristic 1: Root node (first node)
    seed_nodes.append(0)

    # Heuristic 2: High-degree nodes (top 5%)
    edge_index = data.edge_index
    degrees = torch.zeros(data.x.shape[0], dtype=torch.long)

    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i], edge_index[1, i]
        degrees[src] += 1
        degrees[dst] += 1

    # Get top 5% by degree
    k = max(1, int(0.05 * data.x.shape[0]))
    top_k_nodes = torch.topk(degrees, k=k).indices.tolist()

    seed_nodes.extend(top_k_nodes)

    return list(set(seed_nodes))  # Remove duplicates


def process_dataset_with_sampling(
    dataset_path: str,
    output_path: str,
    k_hops: int = 3,
    max_nodes: int = 200
):
    """
    Process entire dataset with graph sampling.

    Args:
        dataset_path: Path to original dataset
        output_path: Where to save sampled dataset
        k_hops: Neighborhood size
        max_nodes: Maximum nodes per graph
    """
    print(f"Loading dataset from {dataset_path}...")
    data = torch.load(dataset_path, weights_only=False)
    samples = data['samples']

    print(f"Original dataset: {len(samples)} graphs")

    # Analyze original sizes
    original_nodes = [s.x.shape[0] for s in samples]
    print(f"\nOriginal size distribution:")
    print(f"  Min:    {min(original_nodes):,}")
    print(f"  Max:    {max(original_nodes):,}")
    print(f"  Mean:   {np.mean(original_nodes):.1f}")
    print(f"  Median: {np.median(original_nodes):.1f}")

    # Sample all graphs
    print(f"\nSampling graphs (k_hops={k_hops}, max_nodes={max_nodes})...")
    sampled_graphs = []

    for i, graph in enumerate(samples):
        if i % 100 == 0:
            print(f"  Progress: {i}/{len(samples)}")

        # Identify important nodes
        seed_nodes = identify_seed_nodes(graph)

        # Sample subgraph
        sampled = sample_k_hop_subgraph(
            graph,
            seed_nodes=seed_nodes,
            k_hops=k_hops,
            max_nodes=max_nodes
        )

        sampled_graphs.append(sampled)

    # Analyze sampled sizes
    sampled_nodes = [s.x.shape[0] for s in sampled_graphs]
    print(f"\nSampled size distribution:")
    print(f"  Min:    {min(sampled_nodes):,}")
    print(f"  Max:    {max(sampled_nodes):,}")
    print(f"  Mean:   {np.mean(sampled_nodes):.1f}")
    print(f"  Median: {np.median(sampled_nodes):.1f}")

    # Verify reduction
    reduction_pct = (1 - np.mean(sampled_nodes) / np.mean(original_nodes)) * 100
    print(f"\nSize reduction: {reduction_pct:.1f}%")

    # Save sampled dataset
    print(f"\nSaving sampled dataset to {output_path}...")
    torch.save({'samples': sampled_graphs}, output_path)

    print("✓ Dataset sampling complete!")


if __name__ == "__main__":
    import sys

    # Demo: Sample the problematic SWE-bench dataset
    dataset_path = "experiments/datasets/gnn/proper_with_swe_bench/supervised/20251030T182051Z/dataset.pt"
    output_path = "experiments/datasets/gnn/sampled_swe_bench.pt"

    print("="*70)
    print("SOLUTION #1: GRAPH SAMPLING FOR GNN SCALABILITY")
    print("="*70)
    print()

    process_dataset_with_sampling(
        dataset_path=dataset_path,
        output_path=output_path,
        k_hops=3,
        max_nodes=200
    )

    print()
    print("="*70)
    print("NEXT STEPS:")
    print("="*70)
    print("1. Train on sampled dataset:")
    print(f"   python3 -m nerion_digital_physicist.training.run_training \\")
    print(f"     --dataset {output_path} \\")
    print(f"     --architecture sage \\")
    print(f"     --epochs 50 \\")
    print(f"     --batch-size 32")
    print()
    print("2. Expected result: Accuracy ≥ 60% on full dataset")
    print()
    print("3. If successful, proceed to Solution #2 (Hierarchical Pooling)")
    print("   for 70-78% accuracy target")
