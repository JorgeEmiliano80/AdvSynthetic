import torch
from abc import ABC, abstractmethod
from typing import List, Tuple 

class AbstractSampler(ABC):
    """
    Interface for active Learning selection strategies.
    Decides WHICH data points are worthy of synthetic augmentation.
    """
    @abstractmethod
    def select_batch(self, uncertainty_scores: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
          uncertainty_scores: Tensor (batch_size,) with entropy values.
        Returns:
          indices: the indices of the selected hard examples.
          values: the actual uncertainty scores of those examples.
        """
        pass
    
class HardExampleMiner(AbstractSampler):
    """
    Mines the 'hardest' examples based on predictive entropy.
    We use a dynamic percentile threshold. We don't care about
    images where the model is confident (low entropy). We focus compute
    on the 'Long Tail' of uncertainty.
    """
    def __init__(self, select_top_percent: float = 0.10):
        # select_top_percent: Fraction of batch to select (e.g, 0.10 = top 10%)
        if not 0 < select_top_percent <= 1.0:
            raise ValueError("Percentage must be between 0 and 1")
        self.select_top_percent = select_top_percent

    def select_batch(self, uncertainty_scores: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = uncertainty_scores.shape[0]

        k = int(batch_size * self.select_top_percent)
        k = max(k, 1)

        values, indices = torch.topk(uncertainty_scores, k)

        return indices, values