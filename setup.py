"""Setup script for OLMo3-MIRAS."""

from setuptools import setup, find_packages

setup(
    name="olmo3_miras",
    version="0.1.0",
    description="OLMo3 with MIRAS Neural Long-Term Memory Integration",
    author="AI Research",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0",
        "transformers>=4.40",
        "einops>=0.7",
        "datasets>=2.0",
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-cov",
        ],
    },
)
