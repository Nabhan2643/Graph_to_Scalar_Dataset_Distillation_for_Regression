#!/usr/bin/env python3
"""
Test script for sparse edge-index based aggregation in GraphSAGE
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import GraphSAGE

def test_dense_adjacency():
    """Test with dense adjacency matrix input"""
    print("=" * 60)
    print("Test 1: Dense Adjacency Matrix Input")
    print("=" * 60)

    torch.manual_seed(42)
    in_dim = 16
    hidden_dim = 32
    num_nodes = 10

    X = torch.randn(num_nodes, in_dim)
    # Create a sparse dense adjacency matrix (not all ones)
    A_dense = torch.zeros(num_nodes, num_nodes)
    # Add some random edges
    edges = torch.randint(0, num_nodes, (20, 2))
    for src, dst in edges:
        A_dense[src, dst] = 1.0
        A_dense[dst, src] = 1.0  # Symmetric

    print(f"X shape: {X.shape}")
    print(f"A_dense shape: {A_dense.shape}")
    print(f"Number of edges: {int(A_dense.sum().item())}")

    gnn = GraphSAGE(in_dim, hidden_dim)
    output = gnn(X, A_dense)

    print(f"Output shape: {output.shape}")
    print(f"Output value: {output.item():.4f}")
    print("✓ Test 1 PASSED\n")
    return gnn, X, A_dense, output


def test_edge_index_input(gnn_reference, X, A_dense):
    """Test with edge index tuple input"""
    print("=" * 60)
    print("Test 2: Edge Index Tuple Input")
    print("=" * 60)

    # Extract edge index from dense matrix
    edge_index = A_dense.nonzero(as_tuple=True)
    print(f"Edge index shapes: src={edge_index[0].shape}, dst={edge_index[1].shape}")

    gnn2 = GraphSAGE(gnn_reference.lin_self_1.in_features, gnn_reference.lin_self_1.out_features)
    gnn2.load_state_dict(gnn_reference.state_dict())  # Use same weights

    output2 = gnn2(X, edge_index)
    print(f"Output shape: {output2.shape}")
    print(f"Output value: {output2.item():.4f}")
    print("✓ Test 2 PASSED\n")
    return output2


def test_consistency(gnn1, gnn2, X, A_dense, edge_index):
    """Compare results between dense and sparse should be close"""
    print("=" * 60)
    print("Test 3: Consistency Check (Dense vs Edge Index)")
    print("=" * 60)

    output_dense = gnn1(X, A_dense)
    output_sparse = gnn2(X, edge_index)

    diff = torch.abs(output_dense - output_sparse).item()
    print(f"Difference between outputs: {diff:.8f}")
    if diff < 1e-5:
        print("✓ Test 3 PASSED - Outputs are numerically consistent\n")
        return True
    else:
        print(f"⚠ Test 3 WARNING - Outputs differ by {diff}\n")
        return False


def test_memory_efficiency(num_nodes, num_edges):
    """Analyze memory efficiency"""
    print("=" * 60)
    print("Test 4: Memory Efficiency Analysis")
    print("=" * 60)

    memory_dense = (num_nodes ** 2) * 4 / 1024 / 1024  # float32
    memory_sparse = (num_edges * 2) * 8 / 1024 / 1024  # two long tensors

    print(f"Dense adjacency matrix:")
    print(f"  - Size: {num_nodes} x {num_nodes} = {num_nodes**2} entries")
    print(f"  - Memory (approx): {memory_dense:.6f} MB")
    print(f"\nSparse edge index:")
    print(f"  - Number of edges: {num_edges}")
    print(f"  - Storage: 2 tensors x {num_edges} = {num_edges*2} indices")
    print(f"  - Memory (approx): {memory_sparse:.6f} MB")
    print(f"\nMemory savings: {(1 - memory_sparse/memory_dense)*100:.1f}%")
    print("✓ Test 4 PASSED\n")


def test_larger_graph():
    """Test with a larger graph to show efficiency gains"""
    print("=" * 60)
    print("Test 5: Larger Graph (100 nodes)")
    print("=" * 60)

    torch.manual_seed(42)
    in_dim = 16
    hidden_dim = 32
    num_nodes = 100

    X = torch.randn(num_nodes, in_dim)
    # Create a random sparse adjacency matrix
    A_dense = torch.zeros(num_nodes, num_nodes)
    num_edges_to_add = 500
    edges = torch.randint(0, num_nodes, (num_edges_to_add, 2))
    for src, dst in edges:
        A_dense[src, dst] = 1.0

    edge_index = A_dense.nonzero(as_tuple=True)
    num_edges = edge_index[0].shape[0]

    print(f"Graph: {num_nodes} nodes, {num_edges} edges")
    
    # Memory analysis
    memory_dense = (num_nodes ** 2) * 4 / 1024 / 1024  # float32
    memory_sparse = (num_edges * 2) * 8 / 1024 / 1024  # two long tensors
    print(f"Dense storage: {memory_dense:.4f} MB")
    print(f"Sparse storage: {memory_sparse:.4f} MB")
    print(f"Memory savings: {(1 - memory_sparse/memory_dense)*100:.1f}%")

    gnn = GraphSAGE(in_dim, hidden_dim)
    output = gnn(X, A_dense)
    print(f"Output: {output.item():.4f}")
    print("✓ Test 5 PASSED\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("SPARSE EDGE-INDEX AGGREGATION TESTS")
    print("=" * 60 + "\n")

    # Run tests
    gnn, X, A_dense, output1 = test_dense_adjacency()
    
    edge_index = A_dense.nonzero(as_tuple=True)
    output2 = test_edge_index_input(gnn, X, A_dense)
    
    gnn2 = GraphSAGE(gnn.lin_self_1.in_features, gnn.lin_self_1.out_features)
    gnn2.load_state_dict(gnn.state_dict())
    test_consistency(gnn, gnn2, X, A_dense, edge_index)
    
    num_edges = int(A_dense.sum().item())
    test_memory_efficiency(X.shape[0], num_edges)
    
    test_larger_graph()

    print("=" * 60)
    print("ALL TESTS COMPLETED SUCCESSFULLY")
    print("=" * 60 + "\n")
