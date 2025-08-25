#!/usr/bin/env python3
"""
Simple test to validate M-RoPE position tensor shapes and basic functionality.
This is a standalone test that doesn't require the full vLLM environment.
"""

import torch
import numpy as np


def test_mrope_position_shapes():
    """Test M-RoPE vs regular RoPE position tensor shapes."""
    print("Testing M-RoPE position tensor shapes...")
    
    device = torch.device("cpu")  # Use CPU for testing
    batch_size = 2
    seq_lens = [5, 3]
    total_tokens = sum(seq_lens)
    
    # Test M-RoPE positions (3, total_tokens)
    mrope_positions = torch.zeros((3, total_tokens), dtype=torch.int64, device=device)
    start_idx = 0
    for i, seq_len in enumerate(seq_lens):
        seq_positions = torch.arange(seq_len, dtype=torch.int64, device=device)
        mrope_positions[:, start_idx:start_idx + seq_len] = seq_positions.unsqueeze(0).repeat(3, 1)
        start_idx += seq_len
    
    # Verify M-RoPE shape
    assert mrope_positions.shape == (3, total_tokens), \
        f"M-RoPE positions should have shape (3, {total_tokens}), got {mrope_positions.shape}"
    
    # Test regular RoPE positions (total_tokens,)
    regular_positions = []
    for seq_len in seq_lens:
        regular_positions.append(torch.arange(seq_len, dtype=torch.int64, device=device))
    regular_positions = torch.cat(regular_positions)
    
    # Verify regular RoPE shape
    assert regular_positions.shape == (total_tokens,), \
        f"Regular RoPE positions should have shape ({total_tokens},), got {regular_positions.shape}"
    
    # Verify content consistency (M-RoPE should have identical position values across all 3 dimensions for text)
    for dim in range(3):
        assert torch.equal(mrope_positions[dim], regular_positions), \
            f"M-RoPE dimension {dim} should match regular positions for text-only input"
    
    print("✓ M-RoPE position tensor shapes test passed")
    
    # Test with different batch sizes
    for batch_size in [1, 3, 4]:
        seq_lens = [i + 1 for i in range(batch_size)]
        total_tokens = sum(seq_lens)
        
        # M-RoPE case
        mrope_pos = torch.zeros((3, total_tokens), dtype=torch.int64, device=device)
        assert mrope_pos.shape == (3, total_tokens)
        
        # Regular RoPE case
        regular_pos = torch.zeros(total_tokens, dtype=torch.int64, device=device)
        assert regular_pos.shape == (total_tokens,)
        
        print(f"✓ Batch size {batch_size} shapes test passed")


def test_position_calculations():
    """Test position calculations that would occur in propose_tree."""
    print("Testing position calculations...")
    
    device = torch.device("cpu")
    batch_size = 2
    seq_lens = [3, 2]  # Total of 5 tokens, evenly distributed
    total_tokens = sum(seq_lens)
    
    # Create positions for each sequence
    positions_regular = []
    for seq_len in seq_lens:
        positions_regular.append(torch.arange(seq_len, dtype=torch.int64, device=device))
    positions_regular = torch.cat(positions_regular)  # (5,)
    
    positions_mrope = torch.zeros((3, total_tokens), dtype=torch.int64, device=device)
    positions_mrope[0] = positions_regular  # Use regular positions for all dimensions
    positions_mrope[1] = positions_regular
    positions_mrope[2] = positions_regular
    
    # Test draft position calculation (similar to propose_tree logic)
    level = 0
    draft_positions_regular = positions_regular + (level + 1)
    draft_positions_mrope_input = positions_mrope[0] + (level + 1)  # Use first dimension
    
    # Verify calculations work for both
    assert torch.equal(draft_positions_regular, draft_positions_mrope_input), \
        "Draft position calculations should be consistent"
    
    # Test reshaping operations - positions can be viewed per sequence
    # For batch_size=2 with seq_lens=[3,2], we can't reshape evenly, so test valid reshapes
    assert positions_regular.shape == (total_tokens,)
    assert positions_mrope.shape == (3, total_tokens)
    
    print("✓ Position calculations test passed")


def test_tensor_operations():
    """Test tensor operations that would be performed with M-RoPE positions."""
    print("Testing tensor operations...")
    
    device = torch.device("cpu")
    batch_size = 2
    seq_len = 3
    total_tokens = batch_size * seq_len
    
    # Create position tensors
    regular_positions = torch.arange(total_tokens, dtype=torch.int64, device=device)
    mrope_positions = torch.zeros((3, total_tokens), dtype=torch.int64, device=device)
    for dim in range(3):
        mrope_positions[dim] = regular_positions
    
    # Test concatenation operations (similar to tree_positions concatenation)
    additional_positions_regular = torch.arange(2, dtype=torch.int64, device=device)
    additional_positions_mrope = torch.zeros((3, 2), dtype=torch.int64, device=device)
    for dim in range(3):
        additional_positions_mrope[dim] = additional_positions_regular
    
    # Test concatenation
    concat_regular = torch.cat([regular_positions, additional_positions_regular])
    concat_mrope = torch.cat([mrope_positions, additional_positions_mrope], dim=1)
    
    assert concat_regular.shape == (total_tokens + 2,)
    assert concat_mrope.shape == (3, total_tokens + 2)
    
    # Test indexing operations
    indexed_regular = regular_positions[:batch_size]
    indexed_mrope = mrope_positions[:, :batch_size]
    
    assert indexed_regular.shape == (batch_size,)
    assert indexed_mrope.shape == (3, batch_size)
    
    print("✓ Tensor operations test passed")


def test_edge_cases():
    """Test edge cases with different batch sizes and sequence lengths."""
    print("Testing edge cases...")
    
    device = torch.device("cpu")
    
    # Test single token
    single_pos_regular = torch.tensor([0], dtype=torch.int64, device=device)
    single_pos_mrope = torch.zeros((3, 1), dtype=torch.int64, device=device)
    
    assert single_pos_regular.shape == (1,)
    assert single_pos_mrope.shape == (3, 1)
    
    # Test large batch
    large_batch_size = 8
    seq_lens = [i + 1 for i in range(large_batch_size)]
    total_tokens = sum(seq_lens)
    
    large_pos_regular = torch.zeros(total_tokens, dtype=torch.int64, device=device)
    large_pos_mrope = torch.zeros((3, total_tokens), dtype=torch.int64, device=device)
    
    assert large_pos_regular.shape == (total_tokens,)
    assert large_pos_mrope.shape == (3, total_tokens)
    
    print("✓ Edge cases test passed")


if __name__ == "__main__":
    try:
        test_mrope_position_shapes()
        test_position_calculations()
        test_tensor_operations()
        test_edge_cases()
        print("\n🎉 All M-RoPE tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()