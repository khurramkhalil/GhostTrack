"""Setup script for GhostTrack."""

from setuptools import setup, find_packages
from pathlib import Path

# Read long description from README
this_directory = Path(__file__).parent
long_description = (this_directory / "README_PHASE4.md").read_text()

setup(
    name="ghosttrack",
    version="1.0.0",
    author="GhostTrack Team",
    author_email="research@anthropic.com",
    description="Multi-Hypothesis Tracking for Hallucination Detection in LLMs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/anthropics/ghosttrack",
    packages=find_packages(exclude=["tests", "tests.*", "scripts"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "datasets>=2.12.0",
        "scikit-learn>=1.2.0",
        "scipy>=1.10.0",
        "numpy>=1.24.0",
        "tqdm>=4.65.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "pytest-cov>=4.0.0",
            "pytest-mock>=3.10.0",
            "black>=23.3.0",
            "flake8>=6.0.0",
            "mypy>=1.3.0",
            "isort>=5.12.0",
        ],
        "viz": [
            "plotly>=5.14.0",
            "kaleido>=0.2.1",
        ],
        "all": [
            "pytest>=7.3.0",
            "pytest-cov>=4.0.0",
            "pytest-mock>=3.10.0",
            "black>=23.3.0",
            "flake8>=6.0.0",
            "mypy>=1.3.0",
            "isort>=5.12.0",
            "plotly>=5.14.0",
            "kaleido>=0.2.1",
        ],
    },
    entry_points={
        "console_scripts": [
            "ghosttrack=scripts.run_evaluation:main",
        ],
    },
    include_package_data=True,
    package_data={
        "ghosttrack": ["config/*.yaml"],
    },
)
