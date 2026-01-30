<div align="center">

# 🧬 AdvSynthetic
### Production-Grade Adversarial Synthetic Data Pipeline

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-ee4c2c.svg?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg?style=for-the-badge)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-macOS-lightgrey?style=for-the-badge&logo=apple&logoColor=black)](https://www.apple.com/macos/)

<p align="center">
  <a href="#-key-features">Key Features</a> •
  <a href="#-installation">Installation</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-architecture">Architecture</a>
</p>

</div>

---

**AdvSynthetic** is an active learning framework designed to improve computer vision model robustness. It automates the discovery of "hard examples" (high-entropy predictions) and generates targeted synthetic training data using Stable Diffusion.

Unlike generic data augmentation, **AdvSynthetic** closes the loop between model inference and data generation, acting as an automated "Red Teamer" that finds your model's blind spots and fixes them.

## 🚀 Key Features

* **🔄 Active Learning Loop**: Automatically audits model uncertainty (Entropy/Margin) to detect confused predictions in production batches.
* **🎨 Adversarial Generation**: Uses Large Language Models (LLM) logic to craft edge-case prompts (e.g., *"a stop sign partially covered by snow in heavy fog"*).
* **⚡ Hardware Optimized (MPS)**: Built with native support for **macOS Metal Performance Shaders**.
    * **Legacy Support**: Optimized for Intel Macs (AMD Radeon).
    * **Next-Gen Ready**: Zero-config acceleration for Apple Silicon (M1/M2/M3/M4).
* **🧠 Lazy Loading Architecture**: Resource-efficient engineering that manages VRAM usage dynamically, loading diffusion weights only during the generation phase.

## 📦 Installation

Clone the repository and install dependencies. We recommend using a virtual environment.

```bash
git clone [https://github.com/JorgeEmiliano80/AdvSynthetic.git](https://github.com/JorgeEmiliano80/AdvSynthetic.git)
cd AdvSynthetic

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install in editable mode
pip install -e .
```

