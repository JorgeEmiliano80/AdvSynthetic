import torch
import pytest
from advsynthetic.pipeline.sampler import HardExampleMiner

def test_hard_example_mining():

    scores = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

    miner = HardExampleMiner(select_top_percent=0.3)
    
    indices, values = miner.select_batch(scores)

    print(f"\n[Sampler Stats] Selected Indices: {indices.tolist()}")
    print(f"[Sampler Stats] Selected Values: {values.tolist()}")
    
    assert len(indices) == 3, "Should select exactly 3 samples (30% of 10)"
    
    # Check if it picked the highest values
    expected_values = torch.tensor([10.0, 9.0, 8.0])
    assert torch.allclose(values, expected_values), "Miner failed to pick the highest entropy scores"
    
    # Check indices (PyTorch topk usually returns sorted, but we check set membership)
    assert 9 in indices and 8 in indices and 7 in indices
    
if __name__ == "__main__":
    test_hard_example_mining()