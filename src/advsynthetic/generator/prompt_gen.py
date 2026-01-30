import random
from typing import List

class AdversarialPromptEngine:
    """
    Logic: Deterministic combinatorial engine for semantic perturbations.
    """
    
    WEATHER_CONDITIONS = [
        "in heavy fog", "during a rainstorm", "in snowy weather", 
        "with sandstorm dust", "in overcast lighting"
    ]
    
    LIGHTING_CONDITIONS = [
        "at night with low light", "with harsh lens flare", 
        "in deep shadows", "under flickering neon lights"
    ]
    
    CAMERA_ARTIFACTS = [
        "with motion blur", "out of focus", "with jpeg compression artifacts",
        "captured by a low resolution cctv camera"
    ]

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def generate_adversarial_prompts(self, base_class: str, num_variants: int = 3) -> List[str]:
        prompts = []
        
        all_perturbations = []
        all_perturbations.extend(self.WEATHER_CONDITIONS)
        all_perturbations.extend(self.LIGHTING_CONDITIONS)
        all_perturbations.extend(self.CAMERA_ARTIFACTS)
        
        if num_variants > len(all_perturbations):
            selected = self.rng.choices(all_perturbations, k=num_variants)
        else:
            selected = self.rng.sample(all_perturbations, k=num_variants)
            
        for p in selected:
            prompts.append(f"a photo of a {base_class} {p}, highly detailed, realistic")
            
        return prompts