import pytest
import torch
from advsynthetic.generator.sd_engine import StableDiffusionEngine

# We mark this as an integration test because it hits the network and loads weights
def test_engine_lifecycle_and_generation():
    """
    Integration Test:
    Verifies that the engine can initialize, load weights, and run a minimal inference cycle.
    """
    
    # 1. Initialization (Should be fast - Lazy Loading)
    engine = StableDiffusionEngine()
    print(f"\n[Test Info] Device detected: {engine.device}")
    
    assert engine.device in ["cpu", "mps", "cuda"]

    prompts = ["a photograph of a futuristic city"]
    
    images = engine.generate(prompts, num_inference_steps=1)

    assert len(images) == 1
    assert images[0].size == (512, 512)
    print("[Test Info] Image generated successfully.")

if __name__ == "__main__":
    test_engine_lifecycle_and_generation()