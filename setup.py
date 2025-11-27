"""
Setup script for ChemDescriptorML
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding='utf-8') if readme_file.exists() else ""

setup(
    name="ChemDescriptorML",
    version="1.0.0",
    author="KAERI_UES",
    author_email="your.email@kaeri.re.kr",
    description="GPU-accelerated molecular descriptor calculation, filtering, and ML training toolkit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hyunwoo47p/descriptor-toolkit",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.11",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "pyarrow>=12.0.0",
        "torch>=2.0.0",
        "rdkit>=2023.3.1",
        "mordred>=1.2.0",
        "tqdm>=4.65.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "ruff>=0.1.0",
        ],
        "gpu": [
            "torch>=2.0.0",  # Install CUDA version separately
        ],
        "ml": [
            "xgboost>=2.0.0",
            "lightgbm>=4.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "cdml=Chem_Descriptor_ML.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
