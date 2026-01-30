from setuptools import setup, find_packages

def get_long_description():
    """Reads the README for PyPI/GitHub rendering."""
    try:
        with open("README.md", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "AdvSynthetix: Adversarial Synthetic Data Factory"

setup(
    name="advsynthetic",
    version="0.1.0",
    author="Jorge Emiliano",
    description="Adversarial Synthetic Data Factory for CV Uncertainty Patching",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/JorgeEmiliano80/AdvSynthetic",

    package_dir={"": "src"},
    packages=find_packages(where="src"),
    
    python_requires=">=3.10", 
    install_requires=[
        "torch>=2.5.0",      
        "torchvision>=0.20.0",
        "transformers>=4.41,<5",
        "accelerate>=0.34,<2",
        "diffusers>=0.31.0",
        "scikit-learn>=1.6.0,<2",
        "hydra-core>=1.3.2",
        "mlflow>=2.7.1",
        "pillow>=10.0.0",
    ],
    extras_require={
        "dev": [
            "pytest~=8.0",
            "mypy~=1.9",
            "ruff",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)