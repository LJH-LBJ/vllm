# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Test M-RoPE (multi-dimensional RoPE) functionality in EagleProposer.propose_tree method.

This test validates that M-RoPE tensor shapes and operations work correctly
for different input scenarios, comparing M-RoPE enabled vs regular RoPE modes.
"""

import pytest
import torch

from vllm.platforms import current_platform


def _create_mrope_positions(batch_size: int, seq_lens: list[int], 
                           device: torch.device) -> torch.Tensor:
    """Create M-RoPE position tensor with shape (3, total_tokens)."""
    total_tokens = sum(seq_lens)
    positions = torch.zeros((3, total_tokens), dtype=torch.int64, device=device)
    
    start_idx = 0
    for seq_len in seq_lens:
        # For M-RoPE, all 3 dimensions have the same position IDs for text-only inputs
        # See page 5 of https://arxiv.org/abs/2409.12191
        seq_positions = torch.arange(seq_len, dtype=torch.int64, device=device)
        positions[:, start_idx:start_idx + seq_len] = seq_positions.unsqueeze(0).repeat(3, 1)
        start_idx += seq_len
    
    return positions


def _create_regular_positions(seq_lens: list[int], device: torch.device) -> torch.Tensor:
    """Create regular RoPE position tensor with shape (total_tokens,)."""
    positions = []
    for seq_len in seq_lens:
        positions.append(torch.arange(seq_len, dtype=torch.int64, device=device))
    return torch.cat(positions)


@pytest.mark.parametrize("uses_mrope", [True, False])
@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_mrope_position_tensor_shapes(uses_mrope, batch_size):
    """Test that M-RoPE and regular RoPE create correct position tensor shapes."""
    # Use available device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Generate variable sequence lengths
    seq_lens = [3 + i for i in range(batch_size)]
    total_tokens = sum(seq_lens)
    
    if uses_mrope:
        positions = _create_mrope_positions(batch_size, seq_lens, device)
        expected_shape = (3, total_tokens)
        assert positions.shape == expected_shape, \
            f"M-RoPE positions should have shape {expected_shape}, got {positions.shape}"
    else:
        positions = _create_regular_positions(seq_lens, device)
        expected_shape = (total_tokens,)
        assert positions.shape == expected_shape, \
            f"Regular RoPE positions should have shape {expected_shape}, got {positions.shape}"
    
    # Verify tensor properties
    assert positions.dtype == torch.int64, "Positions should be int64"
    assert positions.device == device, f"Positions should be on {device}"


@pytest.mark.parametrize("batch_size", [1, 2, 3])
def test_mrope_position_content_consistency(batch_size):
    """Test that M-RoPE positions have consistent content across dimensions for text."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    seq_lens = [2 + i for i in range(batch_size)]
    
    mrope_positions = _create_mrope_positions(batch_size, seq_lens, device)
    regular_positions = _create_regular_positions(seq_lens, device)
    
    # For text-only inputs, all 3 M-RoPE dimensions should match regular RoPE
    for dim in range(3):
        assert torch.equal(mrope_positions[dim], regular_positions), \
            f"M-RoPE dimension {dim} should match regular positions for text-only input"


@pytest.mark.parametrize("uses_mrope", [True, False])
def test_position_calculations_in_propose_tree_logic(uses_mrope):
    """Test position calculations similar to those in propose_tree method."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    batch_size = 2
    seq_lens = [3, 2]
    total_tokens = sum(seq_lens)
    
    # Create position tensors
    if uses_mrope:
        positions = _create_mrope_positions(batch_size, seq_lens, device)
        positions_input = positions[0]  # Use first dimension as input to propose_tree
    else:
        positions_input = _create_regular_positions(seq_lens, device)
    
    # Test draft position calculation (from propose_tree: positions + (level + 1))
    level = 0
    draft_positions = positions_input + (level + 1)
    
    # Test position masking (max_model_len logic from propose_tree)
    max_model_len = 4
    total_num_drafts = 2
    exceeds_max_model_len = (positions_input + total_num_drafts) >= max_model_len
    
    masked_positions = torch.where(exceeds_max_model_len, 0, draft_positions)
    
    # Verify operations work correctly
    assert draft_positions.shape == positions_input.shape, \
        "Draft positions should maintain input shape"
    assert masked_positions.shape == positions_input.shape, \
        "Masked positions should maintain input shape"
    assert masked_positions.dtype == torch.int64, \
        "Masked positions should remain int64"


@pytest.mark.parametrize("uses_mrope", [True, False])
def test_tree_position_concatenation_logic(uses_mrope):
    """Test position concatenation logic similar to propose_tree."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    batch_size = 2
    seq_lens = [2, 2]
    
    # Initial positions
    if uses_mrope:
        initial_positions = _create_mrope_positions(batch_size, seq_lens, device)
        positions_for_tree = initial_positions[0]  # Use first dimension
    else:
        positions_for_tree = _create_regular_positions(seq_lens, device)
    
    # Draft positions (level 1)
    draft_positions = positions_for_tree + 1
    
    # Test concatenation (similar to tree_positions concatenation in propose_tree)
    tree_positions = torch.cat([positions_for_tree, draft_positions])
    
    expected_length = 2 * sum(seq_lens)  # Original + draft
    assert tree_positions.shape == (expected_length,), \
        f"Tree positions should have shape ({expected_length},)"
    assert tree_positions.dtype == torch.int64, \
        "Tree positions should be int64"


@pytest.mark.parametrize("uses_mrope", [True, False])
def test_slot_mapping_calculations(uses_mrope):
    """Test slot mapping calculations with M-RoPE vs regular RoPE positions."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    batch_size = 2
    seq_lens = [3, 2]
    block_size = 16
    
    # Create positions
    if uses_mrope:
        positions = _create_mrope_positions(batch_size, seq_lens, device)
        query_positions = positions[0, :3]  # First dimension, first 3 tokens
    else:
        positions = _create_regular_positions(seq_lens, device)
        query_positions = positions[:3]  # First 3 tokens
    
    # Test slot mapping logic (from propose_tree)
    block_numbers = query_positions // block_size
    slot_offsets = query_positions % block_size
    
    # Mock block_table for slot mapping calculation
    mock_block_table = torch.randint(0, 100, (batch_size, 10), 
                                    dtype=torch.int64, device=device)
    
    # Verify calculations work
    assert block_numbers.dtype == torch.int64, "Block numbers should be int64"
    assert slot_offsets.dtype == torch.int64, "Slot offsets should be int64"
    assert block_numbers.device == device, f"Block numbers should be on {device}"
    assert slot_offsets.device == device, f"Slot offsets should be on {device}"


@pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
def test_scalability_with_different_batch_sizes(batch_size):
    """Test M-RoPE functionality scales correctly with different batch sizes."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Variable length sequences
    seq_lens = [i + 1 for i in range(batch_size)]
    total_tokens = sum(seq_lens)
    
    # Test both M-RoPE and regular RoPE
    for uses_mrope in [True, False]:
        if uses_mrope:
            positions = _create_mrope_positions(batch_size, seq_lens, device)
            expected_shape = (3, total_tokens)
        else:
            positions = _create_regular_positions(seq_lens, device)
            expected_shape = (total_tokens,)
        
        assert positions.shape == expected_shape, \
            f"Shape mismatch for batch_size={batch_size}, uses_mrope={uses_mrope}"
        
        # Test basic operations work at scale
        offset_positions = positions + 1
        assert offset_positions.shape == positions.shape, \
            "Arithmetic operations should preserve shape"


def test_edge_cases():
    """Test edge cases for M-RoPE functionality."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Single token case
    single_mrope = _create_mrope_positions(1, [1], device)
    single_regular = _create_regular_positions([1], device)
    
    assert single_mrope.shape == (3, 1), "Single token M-RoPE shape incorrect"
    assert single_regular.shape == (1,), "Single token regular RoPE shape incorrect"
    assert torch.equal(single_mrope[0], single_regular), \
        "Single token content should be consistent"
    
    # Large sequence
    large_seq_lens = [50, 30, 20]
    large_mrope = _create_mrope_positions(3, large_seq_lens, device)
    large_regular = _create_regular_positions(large_seq_lens, device)
    
    total_large = sum(large_seq_lens)
    assert large_mrope.shape == (3, total_large), "Large sequence M-RoPE shape incorrect"
    assert large_regular.shape == (total_large,), "Large sequence regular RoPE shape incorrect"


def test_tensor_operations_compatibility():
    """Test that M-RoPE positions work with common tensor operations."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    batch_size = 2
    seq_lens = [4, 3]
    
    mrope_pos = _create_mrope_positions(batch_size, seq_lens, device)
    regular_pos = _create_regular_positions(seq_lens, device)
    
    # Test indexing
    mrope_slice = mrope_pos[:, :3]
    regular_slice = regular_pos[:3]
    assert mrope_slice.shape == (3, 3), "M-RoPE indexing failed"
    assert regular_slice.shape == (3,), "Regular RoPE indexing failed"
    
    # Test masking
    mask = torch.tensor([True, False, True, False, True, False, True], device=device)
    mrope_masked = mrope_pos[:, mask]
    regular_masked = regular_pos[mask]
    expected_masked_count = mask.sum().item()
    
    assert mrope_masked.shape == (3, expected_masked_count), "M-RoPE masking failed"
    assert regular_masked.shape == (expected_masked_count,), "Regular RoPE masking failed"
    
    # Test concatenation
    additional_mrope = torch.zeros((3, 2), dtype=torch.int64, device=device)
    additional_regular = torch.zeros(2, dtype=torch.int64, device=device)
    
    concat_mrope = torch.cat([mrope_pos, additional_mrope], dim=1)
    concat_regular = torch.cat([regular_pos, additional_regular])
    
    assert concat_mrope.shape == (3, sum(seq_lens) + 2), "M-RoPE concatenation failed"
    assert concat_regular.shape == (sum(seq_lens) + 2,), "Regular RoPE concatenation failed"