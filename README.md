# ChemDescriptorML v1.0

**GPU-accelerated molecular descriptor calculation, filtering, and ML training toolkit**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status: Production](https://img.shields.io/badge/status-production-green.svg)](README.md)

---

## Overview

ChemDescriptorML은 분자 descriptor 계산, 필터링, 그리고 ML 모델 학습까지 통합된 파이프라인입니다. 대규모 화학 데이터베이스(PubChem 규모)부터 소규모 연구 데이터까지 효율적으로 처리할 수 있습니다.

### Key Features

- **Track 1: Descriptor Extraction & Filtering**
  - 5-Stage Filtering: Variance → Spearman → VIF → HSIC/RDC → Final selection
  - GPU Acceleration: PyTorch 기반 고속 처리
  - Cluster-aware Selection: 상관관계 기반 클러스터링

- **Track 2: ML Model Training**
  - 8개 회귀 모델 자동 학습 (RandomForest, XGBoost, LightGBM 등)
  - K-Fold Cross-Validation + Hold-Out 평가
  - 극단값(Extreme Value) 예측 성능 분석

### Performance

- **Filtering**: GPU 가속으로 20-35x 성능 향상
- **ML Training**: 8 models × N descriptor sizes 자동 실험
- **Best Result**: XGBoost 30D에서 R² = 0.78 달성 (77 샘플 데이터)

---

## Quick Start

### Installation

```bash
# 1. 의존성 설치
pip install -r requirements.txt

# 2. 패키지 설치
pip install -e .

# 3. (선택) ML 부스팅 모델 설치
pip install xgboost lightgbm
```

### 설치 확인

```bash
cdml --version
# 출력: cdml 1.0.0
```

### Basic Usage

```bash
# Track 1: Descriptor 추출 및 필터링
cdml process-all \
    --input molecules.csv \
    --output-dir results/ \
    --smiles-col SMILES

# Track 2: ML 모델 학습
cdml train \
    --input Labeled_descriptors.parquet \
    --target-col pLeach \
    --output-dir ml_output/
```

---

## Pipeline Architecture

### Track 1: Filtering Pipeline

```
Input SMILES/Descriptors
    ↓
Pass 0: Sampling (대규모 데이터 처리)
    ↓
Pass 1: Variance Filtering (σ² < threshold)
    ↓
Pass 2: Spearman Correlation Clustering
    ↓
Pass 3: VIF (Variance Inflation Factor)
    ↓
Pass 4: Nonlinear Analysis (HSIC + RDC)
    ↓
Output: Optimized Descriptor Set + Cluster Info
```

### Track 2: ML Training Pipeline

```
Input: Labeled Descriptors + Cluster Info
    ↓
Descriptor Selection (Sequential/Cluster-based)
    ↓
8 Models × N Descriptor Sizes
    ↓
K-Fold CV + Hold-Out Evaluation
    ↓
Output: Best Model + Performance Reports
```

---

## Project Structure

```
ChemDescriptorML/
├── Chem_Descriptor_ML/          # 메인 패키지
│   ├── cli.py                   # CLI 진입점 (cdml 명령어)
│   ├── config/                  # 설정 관리
│   ├── filtering/               # 필터링 파이프라인
│   │   ├── pipeline.py
│   │   └── passes/              # Pass 0-4 구현
│   ├── preprocessing/           # 전처리 (XML, Descriptor 계산)
│   └── ml/                      # ML 학습 모듈
│       └── ensemble.py          # OptimalMLEnsemble
├── docs/                        # 문서
│   ├── 프로그램_구동방법.md
│   └── 첨부_파일_구성.md
├── reference/                   # 참조 결과
├── setup.py
└── requirements.txt
```

---

## CLI Commands

| 명령어 | 설명 |
|--------|------|
| `cdml process-all` | 통합 파이프라인 (SMILES → Descriptor → Filtering) |
| `cdml run` | 필터링 파이프라인 (Pass 0-4) |
| `cdml filter` | 개별 Pass 실행 |
| `cdml preprocess` | 전처리 (XML 변환, 스키마 생성, Descriptor 계산) |
| `cdml train` | ML 모델 학습 |

```bash
# 도움말 확인
cdml --help
cdml train --help
```

---

## Requirements

### Core Dependencies

- numpy>=1.24.0
- pandas>=2.0.0
- pyarrow>=12.0.0
- torch>=2.0.0
- rdkit>=2023.3.1
- mordred>=1.2.0
- scikit-learn>=1.3.0
- matplotlib>=3.7.0

### Optional Dependencies

- xgboost>=2.0.0 (XGBoost 모델)
- lightgbm>=4.0.0 (LightGBM 모델)

---

## Documentation

- [프로그램 구동방법](docs/프로그램_구동방법.md) - 상세 사용 가이드
- [첨부 파일 구성](docs/첨부_파일_구성.md) - 파일 구조 설명

---

## Changelog

### v1.0.0 (2024-11-27)

**Initial Release**

- Track 1: 5-Stage Descriptor Filtering Pipeline
- Track 2: 8-Model ML Ensemble Training
- GPU acceleration for correlation analysis
- K-Fold CV + Hold-Out evaluation
- Cluster-aware descriptor selection
- Comprehensive CLI interface

---

**Version**: 1.0.0
**Status**: Production Ready
**Last Updated**: 2024-11-27
