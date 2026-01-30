# tests/test_auditor.py
import torch
import torch.nn as nn
import pytest
from advsynthetic.auditor.uncertainty import MCDropoutAuditor

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.dropout(x)
        return self.fc(x)

def test_mc_dropout_variance():
    """
    Verifies that the Auditor actually captures stochasticity.
    """
    model = DummyModel()
    auditor = MCDropoutAuditor(model, num_mc_samples=20)

    x = torch.randn(4, 10) 
    
    mean_probs, entropy = auditor.estimate(x)

    assert mean_probs.shape == (4, 5), "Output shape mismatch"
    assert entropy.shape == (4,), "Entropy should be a scalar per image"

    assert (entropy > 0).all(), "Entropy should be non-zero for stochastic models"
    
    print(f"\n[Test Passed] Mean Entropy: {entropy.mean().item():.4f}")

if __name__ == "__main__":
    test_mc_dropout_variance()