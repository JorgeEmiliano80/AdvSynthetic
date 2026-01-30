import logging
from typing import List, Optional, Any
import torch
from PIL import Image

# Lazy import
from diffusers import StableDiffusionPipeline

from advsynthetic.core.generator import ISyntheticGenerator

# Configure structured logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AdvSynthetic.Engine")

class StableDiffusionEngine(ISyntheticGenerator):
    """
    Production-grade wrapper for Stable Diffusion v1.5.
    """

    def __init__(self, model_id: str = "runwayml/stable-diffusion-v1-5", device: Optional[str] = None):
        self.model_id = model_id
        self._pipeline = None 

        if device:
            self.device = device
        elif torch.backends.mps.is_available():
            self.device = "mps" 
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu" 
        
        logger.info(f"Engine initialized. Target hardware: {self.device.upper()}")

    def load_model(self) -> None:
        """
        Loads the model weights into memory only when requested.
        """
        if self._pipeline is not None:
            return

        logger.info(f"Loading weights for {self.model_id} into {self.device}...")
        
        try:
            torch_dtype = torch.float32 if self.device == "cpu" else torch.float16
            
            self._pipeline = StableDiffusionPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch_dtype,
                use_safetensors=True
            )

            self._pipeline.to(self.device)

            if self.device in ["mps", "cpu"]:
                self._pipeline.enable_attention_slicing()

            logger.info("Model loaded successfully.")
            
        except Exception as e:
            logger.error(f"Critical error loading model: {e}")
            raise RuntimeError("Model loading failed") from e

    def generate(self, prompts: List[str], num_inference_steps: int = 20, **kwargs: Any) -> List[Image.Image]:
        if self._pipeline is None:
            self.load_model()
            
        if not prompts:
            return []

        logger.info(f"Generating {len(prompts)} images with {num_inference_steps} steps...")
        
        generated_images = []
        
        for i, prompt in enumerate(prompts):
            logger.debug(f"Processing [{i+1}/{len(prompts)}]: {prompt[:40]}...")
            
            with torch.no_grad():
                output = self._pipeline(
                    prompt, 
                    num_inference_steps=num_inference_steps,
                    **kwargs
                )
            
            image = output.images[0]
            generated_images.append(image)
            
        return generated_images