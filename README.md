# Molecular Descriptor Toolkit v1.0

**Production-ready molecular descriptor filtering pipeline for large-scale chemical databases**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status: Production](https://img.shields.io/badge/status-production-green.svg)](README.md)

---

## Overview

Molecular Descriptor Toolkit은 대규모 화학 데이터베이스를 위한 고성능 molecular descriptor 필터링 파이프라인입니다. PubChem 규모(10억+ 화합물)의 데이터셋을 효율적으로 처리할 수 있도록 설계되었습니다.

### Key Features

- **5-Stage Filtering Pipeline**: Variance → Correlation → VIF → Nonlinearity → Final selection
- **GPU Acceleration**: PyTorch 기반 HSIC/RDC 비선형 분석 (20-35x speedup)
- **Memory Efficient**: Streaming + chunked processing for billion-scale datasets
- **Flexible Configuration**: 60+ parameters with YAML-based settings
- **Production Ready**: Comprehensive error handling, checkpointing, and logging

### Performance

- **Dataset Scale**: 1.2 billion compounds
- **GPU Performance**: 20-35x faster than CPU (NVIDIA RTX 6000 Ada, 48GB VRAM)
- **Memory Footprint**: Optimized for 100K+ descriptors with 500K compounds per chunk
- **Throughput**: ~500K compounds/batch processing

---

## Quick Start

### Installation

```bash
# 1. Clone or extract the toolkit
cd molecular_descriptor_toolkit

# 2. Install dependencies
pip install numpy torch pyarrow scipy igraph leidenalg tqdm statsmodels

# Optional: For preprocessing
pip install rdkit mordred
```

### Basic Usage

```bash
# Set Python path
export PYTHONPATH=$(pwd):$PYTHONPATH

# Run complete pipeline
python -m molecular_descriptor_toolkit.cli run \
    --input /path/to/descriptors.parquet \
    --output ./filtered_results \
    --config config/default_settings.yaml
```

---

## Pipeline Architecture

### 5-Stage Filtering Process

```
Input Descriptors (100K+)
    ↓
Pass 0: Variance Filtering (σ² < threshold)
    ↓ ~50K descriptors
Pass 1: Spearman Correlation Clustering
    ↓ ~10K descriptors
Pass 2: Pearson Correlation GPU Processing
    ↓ ~5K descriptors
Pass 3: VIF (Variance Inflation Factor)
    ↓ ~2K descriptors
Pass 4: Nonlinear Analysis (HSIC + RDC)
    ↓ Final Set
Output: Optimized Descriptor Set
```

---

## Project Structure

```
molecular_descriptor_toolkit/
├── config/
│   ├── settings.py              # Configuration dataclasses
│   └── default_settings.yaml    # Default configuration
├── filtering/
│   ├── pipeline.py              # Main pipeline orchestrator
│   └── passes/
│       ├── pass0_variance.py    # Variance filtering
│       ├── pass1_spearman.py    # Spearman clustering
│       ├── pass2_correlation.py # Correlation filtering (GPU)
│       ├── pass3_vif.py         # VIF filtering
│       └── pass4_nonlinear.py   # Nonlinear analysis
├── preprocessing/
│   ├── xml_parser.py            # PubChem XML parser
│   ├── descriptor_calculator.py # RDKit/Mordred wrapper
│   └── pipeline.py              # Preprocessing pipeline
├── utils/
│   ├── device_utils.py          # GPU management
│   └── memory_utils.py          # Memory optimization
├── cli.py                       # Command-line interface
└── __init__.py
```

---

## Requirements

### Core Dependencies

- numpy>=1.24.0
- torch>=2.0.0
- pyarrow>=12.0.0
- scipy>=1.10.0
- igraph>=0.10.0
- leidenalg>=0.10.0
- tqdm>=4.65.0
- statsmodels>=0.14.0

### Optional Dependencies

- rdkit>=2023.3.1 (for preprocessing)
- mordred>=1.2.0 (for descriptor calculation)

---

## Documentation

- [Quick Start Guide](QUICKSTART.md) - Complete A-Z testing guide
- [Configuration Guide](CONFIG_GUIDE.md) - All 60+ parameters
- [Structure Guide](STRUCTURE_GUIDE.md) - Architecture details

---

## Changelog

### v1.0.0 (2024-11-10)

**Initial Production Release**

- Complete 5-stage filtering pipeline
- GPU acceleration for correlation and nonlinear analysis
- Comprehensive configuration system (60+ parameters)
- Memory-efficient streaming for billion-scale datasets
- Full CLI interface with checkpointing
- Production-ready error handling and logging

**Validated On**:
- Dataset: PubChem Compound Database
- Scale: 1.2 billion compounds
- Descriptors: 100K+ molecular descriptors
- Hardware: NVIDIA RTX 6000 Ada (48GB VRAM)

---

**Version**: 1.0.0  
**Status**: Production Ready  
**Last Updated**: 2024-11-10
