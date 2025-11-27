# CLAUDE.md - 프로젝트 컨텍스트 및 작업 이력

이 파일은 Claude Code와의 작업 이력을 기록하여 컨텍스트를 유지합니다.

## 프로젝트 개요

**ChemDescriptorML (CDML)** - GPU 가속 분자 descriptor 계산, 필터링, 그리고 ML 학습 통합 파이프라인

### 주요 기능

**Track 1: Descriptor Filtering Pipeline**
- Pass 0: Sampling
- Pass 1: Statistics + Variance Filtering
- Pass 2: Spearman Correlation Clustering
- Pass 3: VIF Multicollinearity
- Pass 4: HSIC + RDC Nonlinear Detection

**Track 2: ML Training Pipeline**
- 8개 회귀 모델 자동 학습 (RandomForest, XGBoost, LightGBM 등)
- K-Fold Cross-Validation + Hold-Out 평가
- Cluster-aware Descriptor Selection

## 프로젝트 구조

```
descriptor-toolkit/
├── Chem_Descriptor_ML/              # 메인 패키지
│   ├── __init__.py
│   ├── cli.py                       # CLI 진입점 (cdml 명령어)
│   ├── config/                      # 설정 관리
│   │   ├── settings.py              # Config 클래스 정의
│   │   └── loader.py                # YAML 로더
│   ├── filtering/                   # 필터링 파이프라인
│   │   ├── pipeline.py              # 메인 파이프라인
│   │   ├── passes/                  # 각 Pass 구현
│   │   ├── io/                      # Parquet I/O
│   │   └── utils/                   # 유틸리티
│   ├── preprocessing/               # 전처리 모듈
│   ├── ml/                          # ML 학습 모듈
│   │   └── ensemble.py              # OptimalMLEnsemble
│   └── workflows/                   # 워크플로우
├── docs/                            # 문서
│   ├── 프로그램_구동방법.md
│   └── 첨부_파일_구성.md
├── reference/                       # 참조 결과
├── setup.py
├── requirements.txt
└── README.md
```

## 설정 구조 (Config)

계층적 SSOT 설계:
```python
Config(
    device=DeviceConfig(...),      # GPU/CPU 설정
    io=IOConfig(...),              # 입출력 설정
    filtering=FilteringConfig(...), # 필터링 파라미터
    system=SystemConfig(...),      # 시스템 설정
)
```

---

## 작업 이력

### 2025-11-27: 프로젝트 리네이밍 및 ML 모듈 추가

**대대적 리네이밍:**
- 프로젝트명: Molecular Descriptor Toolkit (MDT) → **ChemDescriptorML (CDML)**
- 패키지 폴더: `molecular_descriptor_toolkit/` → `Chem_Descriptor_ML/`
- CLI 명령어: `mdt` → `cdml`
- 모든 문서 업데이트 완료

**Track 2: ML 학습 모듈 추가:**
- 8개 회귀 모델 앙상블 학습 (`cdml train`)
- K-Fold CV + Hold-Out 평가
- Cluster-aware descriptor selection (sequential, representative, random_alternative, mixed)
- 참조 데이터셋 테스트 결과: XGBoost 30D에서 R² = 0.78 (hold-out)

### 2025-11-25: 초기 버그 수정

**피드백 분석 및 수정 완료:**

1. **패키지 구조 문제** (High)
   - 문제: 패키지 디렉토리 없이 import 경로가 해당 패키지를 참조
   - 해결: 최상위에 패키지 디렉토리 생성 후 모듈 이동

2. **CLI Config 생성 오류** (High)
   - 문제: `cli.py`에서 flat 키워드로 Config 생성 시도
   - 해결: 섹션 기반 Config 생성으로 변경
   ```python
   # Before (오류)
   Config(parquet_glob=..., output_dir=..., prefer_gpu=...)

   # After (수정됨)
   Config(
       io=IOConfig(parquet_glob=..., output_dir=...),
       device=DeviceConfig(prefer_gpu=...),
       ...
   )
   ```

3. **잘못된 속성명** (High)
   - 문제: `self.io_cfg.n_metadata_cols` 사용 (존재하지 않음)
   - 해결: `self.io_cfg.n_metadata`로 수정

4. **self.config 참조 오류** (Medium)
   - 문제: `pass1_statistics.py`에서 `self.config` 참조 (존재하지 않음)
   - 해결: `self.filtering_cfg`로 수정

5. **VIF 클래스 생성자 인자 오류** (Medium)
   - 문제: `pipeline.py`에서 VIFFilteringPassGPUWithClusters 호출 시 잘못된 인자 전달
   - 해결: 올바른 인자 순서로 수정

**테스트 결과:** ✅ 전체 파이프라인 정상 동작 확인

---

## 개발 가이드

### CLI 사용법
```bash
# Track 1: 통합 파이프라인 (SMILES → Descriptors → Filtering)
cdml process-all --input molecules.csv --output-dir results/ --smiles-col SMILES --id-col CID

# Track 1: 필터링만 실행 (이미 descriptor가 있는 경우)
cdml run --parquet-glob "data/*.parquet" --output-dir results/

# Track 1: 개별 Pass 실행
cdml filter pass0 --parquet-glob "data/*.parquet" --output-dir results/
cdml filter pass1 --parquet-glob "data/*.parquet" --output-dir results/
cdml filter pass234 --parquet-glob "data/*.parquet" --output-dir results/

# Track 1: 전처리만
cdml preprocess xml-to-parquet --input compounds.xml --output molecules.parquet
cdml preprocess generate-schema --input data/ --output schema.json --quick
cdml preprocess calculate-descriptors --input data.csv --output descriptors.parquet --schema schema.json

# Track 2: ML 학습
cdml train --input Labeled_descriptors.parquet --target-col pLeach --output-dir ml_output/
```

### 설치
```bash
# 의존성 설치
pip install -r requirements.txt

# 패키지 설치
pip install -e .

# (선택) ML 부스팅 모델
pip install xgboost lightgbm
```

### 테스트
```bash
pytest tests/
```

---

## 알려진 이슈

(현재 없음)

---

## 다음 작업 예정

- [ ] 단위 테스트 추가
- [ ] 문서화 개선
