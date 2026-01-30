import logging
import os
from typing import List, Dict, Any
from PIL import Image 
import torch 

from advsynthetic.auditor.uncertainty import UncertaintyAuditor
from advsynthetic.pipeline.sampler import HardExampleMiner
from advsynthetic.generator.prompt_gen import AdversarialPromptEngine
from advsynthetic.generator.sd_engine import StableDiffusionEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AdvSynthetic.Manager")

class AdvSyntheticPipeline:
    """
     The main controller for the Adversarial Synthetic Data Pipeline
    """

    def __init__(self, config: Dict[str, Any]):
        self.cfg = config 

        logger.info("Initializing Pipeline Components...")

        self.auditor = UncertaintyAuditor()

        self.sampler = HardExampleMiner(
            select_top_percent=self.cfg['selection']['top_k_percent']
        )

        self.prompt_engine = AdversarialPromptEngine()

        self.generator = StableDiffusionEngine(
            model_id=self.cfg['model']['id']
        )

        self.output_dir = self.cfg['pipeline']['output_dir']
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"Pipeline initialized. Output directory: {self.output_dir}")

    def run(self, input_labels: List[str], model_logits: torch.Tensor) -> List[Image.Image]:
        logger.info(">>> STEP 1: AUDIT (Detecting Weaknesses)")
        entropy = self.auditor.calculate_entropy(model_logits)
        logger.info(f"  Mean Batch Entropy: {entropy.mean():.4f}")

        logger.info(">>> STEP 2: MINING (Selecting Hard Examples)")
        hard_indices, hard_scores = self.sampler.select_batch(entropy)

        if len(hard_indices) == 0:
            logger.info("   No hard examples found. Skipping generation")
            return[]
        
        logger.info(f"    Selected {len(hard_indices)} hard examples for augmentation")

        all_synthetic_images = []

        logger.ingo(">>> STEP 3 & 4: ADVERSARIAL GENERATION")
        for idx in hard_indices:
            original_label = input_labels[idx]
            uncertainty_score = entropy[idx].item()

            logger.info(f"    [Target: {original_label} | Entropy: {uncertainty_score:.2}] generating attacks...")

            prompts = self.prompt_engine.generate_adversarial_prompts(
                base_class=original_label,
                num_variants=self.cfg['generation']['variants_per_image']
            )

            images = self.generator.generate(
                prompts,
                num_inference_steps=self.cfg['model']['steps'],
                guidance_scale=self.cfg['model']['guidance_scale']
            )

            for i, img in enumerate(images):
                filename = os.path.join(self.output_dir, f"{original_label}_adv_{idx}_{i}.png")
                img.save(filename)
                all_synthetic_images.append(img)
                logger.debug(f"   Saved artifact: {filename}")

        logger.info(f"Pipeline Finished. Generated {len(all_synthetic_images)} new training assets.")
        return all_synthetic_images