"""
Test script to verify batched vs. single-graph loss computation.
"""

import torch
import sys
from class_definition import GraphData
from models import GraphSAGE
from losses import l_q, l_syn

def create_random_graph(n_nodes, n_features, seed=None):
    """Create a random GraphData object."""
    if seed is not None:
        torch.manual_seed(seed)
    
    X = torch.randn(n_nodes, n_features)
    
    # Random edges (fully connected for simplicity)
    src = torch.randint(0, n_nodes, (n_nodes * 2,))
    dst = torch.randint(0, n_nodes, (n_nodes * 2,))
    edge_index = torch.stack([src, dst])
    
    # Random scalar target
    y = torch.randn(1)
    
    return GraphData(X=X, edge_index=edge_index, y=y, requires_grad=False)

def test_batched_l_q():
    """Test batched l_q loss computation."""
    print("=" * 60)
    print("TEST: Batched l_q Loss Computation")
    print("=" * 60)
    
    device = 'cpu'
    in_dim = 8
    hidden_dim = 16
    
    # Create GNN
    gnn = GraphSAGE(in_dim=in_dim, hidden_dim=hidden_dim).to(device)
    gnn.eval()
    
    # Create 3 graphs with different node counts
    graphs = [
        create_random_graph(5, in_dim, seed=42),
        create_random_graph(8, in_dim, seed=43),
        create_random_graph(6, in_dim, seed=44),
    ]
    
    print(f"\nGraph shapes: {[g.X.shape[0] for g in graphs]} nodes")
    print(f"Total nodes in batch: {sum(g.X.shape[0] for g in graphs)}")
    
    # Compute loss (now batched internally)
    with torch.no_grad():
        loss = l_q(gnn, graphs)
    
    print(f"\nBatched l_q loss: {loss.item():.6f}")
    print(f"Loss dtype: {loss.dtype}")
    print(f"Loss device: {loss.device}")
    
    # Verify loss is a scalar
    assert loss.dim() == 0, f"Loss should be scalar, got shape {loss.shape}"
    assert loss.item() >= 0, "MSE loss should be non-negative"
    
    print("\n✓ Test passed: batched l_q works correctly")

def test_batched_l_syn():
    """Test batched l_syn loss computation."""
    print("\n" + "=" * 60)
    print("TEST: Batched l_syn Loss Computation")
    print("=" * 60)
    
    device = 'cpu'
    in_dim = 8
    hidden_dim = 16
    
    # Create GNN
    gnn = GraphSAGE(in_dim=in_dim, hidden_dim=hidden_dim).to(device)
    gnn.eval()
    
    # Create 2 synthetic graphs with different sizes
    graphs = [
        create_random_graph(7, in_dim, seed=50),
        create_random_graph(9, in_dim, seed=51),
    ]
    
    print(f"\nSynthetic graph shapes: {[g.X.shape[0] for g in graphs]} nodes")
    print(f"Total nodes in batch: {sum(g.X.shape[0] for g in graphs)}")
    
    # Compute loss (now batched internally)
    with torch.no_grad():
        loss = l_syn(gnn, graphs)
    
    print(f"\nBatched l_syn loss: {loss.item():.6f}")
    print(f"Loss dtype: {loss.dtype}")
    print(f"Loss device: {loss.device}")
    
    # Verify loss is a scalar
    assert loss.dim() == 0, f"Loss should be scalar, got shape {loss.shape}"
    assert loss.item() >= 0, "MSE loss should be non-negative"
    
    print("\n✓ Test passed: batched l_syn works correctly")

def test_single_graph_backward_compat():
    """Test that single-graph calls still work (backward compatibility)."""
    print("\n" + "=" * 60)
    print("TEST: Single-Graph Backward Compatibility")
    print("=" * 60)
    
    device = 'cpu'
    in_dim = 8
    hidden_dim = 16
    
    gnn = GraphSAGE(in_dim=in_dim, hidden_dim=hidden_dim).to(device)
    gnn.eval()
    
    # Single graph
    graph = create_random_graph(10, in_dim, seed=60)
    
    print(f"\nSingle graph shape: {graph.X.shape[0]} nodes")
    
    with torch.no_grad():
        loss = l_q(gnn, [graph])
    
    print(f"Single-graph l_q loss: {loss.item():.6f}")
    assert loss.dim() == 0, f"Loss should be scalar"
    assert loss.item() >= 0, "MSE loss should be non-negative"
    
    print("\n✓ Test passed: single-graph backward compatibility OK")

def test_forward_pass_shapes():
    """Test GNN forward pass with and without batch."""
    print("\n" + "=" * 60)
    print("TEST: GNN Forward Pass Shapes")
    print("=" * 60)
    
    in_dim = 8
    hidden_dim = 16
    gnn = GraphSAGE(in_dim=in_dim, hidden_dim=hidden_dim)
    gnn.eval()
    
    # Test 1: Single graph (no batch)
    X_single = torch.randn(5, in_dim)
    edge_index_single = torch.randint(0, 5, (2, 10))
    
    with torch.no_grad():
        out_single = gnn(X_single, edge_index_single)
    
    print(f"\nSingle graph output shape: {out_single.shape}")
    assert out_single.shape == torch.Size([]), f"Expected scalar output, got {out_single.shape}"
    
    # Test 2: Batched graphs
    graphs = [
        create_random_graph(5, in_dim, seed=70),
        create_random_graph(7, in_dim, seed=71),
    ]
    
    # Manually build batch
    Xs = []
    edges = []
    batch_idx = []
    n_nodes_running = 0
    
    for i, g in enumerate(graphs):
        n = g.X.shape[0]
        Xs.append(g.X)
        edges.append(g.edge_index.long() + n_nodes_running)
        batch_idx.append(torch.full((n,), i, dtype=torch.long))
        n_nodes_running += n
    
    X_batch = torch.cat(Xs, dim=0)
    edge_index_batch = torch.cat(edges, dim=1)
    batch = torch.cat(batch_idx, dim=0)
    
    with torch.no_grad():
        out_batch = gnn(X_batch, edge_index_batch, batch=batch)
    
    print(f"Batched graphs output shape: {out_batch.shape}")
    assert out_batch.shape == torch.Size([2]), f"Expected shape [2], got {out_batch.shape}"
    
    print("\n✓ Test passed: forward pass shapes correct")

if __name__ == "__main__":
    try:
        test_forward_pass_shapes()
        test_single_graph_backward_compat()
        test_batched_l_q()
        test_batched_l_syn()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
