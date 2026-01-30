import torch
import torch.nn as nn 
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Tuple, Optional

class UncertaintyEstimator(ABC):
  """
  Abstract Interface for uncertainty quantification mechanisms.
  Designed to be extensible for future methods (Ensembles, Bayesian Layers).
  """
  @abstractmethod
  def estimate(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the predictive mean and uncertainty score.

    Args:
      x: input tensor of shape (batch_size, channels, height, width)

    Returns:
      mean_probs: Averaged probability distribution (batch_size, num_classes)
      uncertainty: Scalar uncertainty score per sample (batch_size)
    """
    pass

class MCDropoutAuditor(UncertaintyEstimator):
  def __init__(self, model: nn.Module, num_mc_samples: int = 25):
    self.model = model
    self.num_mc_samples = num_mc_samples

  def _enamble_dropout_only(self):
    self.model.eval()
    for m in self.model.modules():
      if m.__class__.__name__.startswith('Dropout'):
        m.train()

  @torch.no_grad()
  def estimate(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    self._enamble_dropout_only()

    mc_outputs = []
    for _ in range(self.num_mc_samples):
      logits = self.model(x)
      probs = F.softmax(logits, dim=-1)
      mc_outputs.append(probs)

    mc_probs = torch.stack(mc_outputs)

    mean_probs = mc_probs.mean(dim=0)

    entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-12), dim=-1)

    return mean_probs, entropy