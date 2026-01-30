from abc import ABC, abstractmethod
from typing import List, Optional, Any 
from PIL import Image 

class ISyntheticGenerator(ABC):
  """
  Abstract interface for synthetic image generation engines.

  Design Principles:
    1. Dependency inversion: the pipelines depends on this interface, not on specific libraries (e.g., difussers, torch), allowing
      backend swaps without refactoring.
    2. Resource Efficiency: enforces a lazy loading pattern to prevent unnecessary
      memory allocation during initialization.
  """

  @abstractmethod
  def load_model(self) -> None:
    """
    Loads the model weights into memory (VRAM/RAM)

    implementation note:
    This method must implement lazy loading logic. If the model is already
    loaded, it should function as a no-op. This ensures resources are only
    consumed when explicitly requested.
    """
    pass

  @abstractmethod
  def generate(self, prompts: List[str], num_inference_steps: int = 20, **kwargs: Any) -> List[Image.Image]:
    pass 