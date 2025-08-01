"""
Setup script for Vertex AI training package
"""

from setuptools import setup, find_packages

with open("requirements-vertex.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="vibenews-bias-trainer",
    version="1.0.0",
    description="Bias classification training for Vertex AI",
    author="VibeNews Team",
    packages=find_packages(),
    install_requires=requirements,
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "train-bias=trainer.task:main",
        ],
    },
)