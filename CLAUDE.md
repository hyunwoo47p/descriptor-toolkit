# CLAUDE.md - 프로젝트 컨텍스트 및 작업 이력

이 파일은 Claude Code와의 작업 이력을 기록하여 컨텍스트를 유지합니다.

## 프로젝트 개요

**Molecular Descriptor Toolkit (MDT)** - GPU 가속 분자 descriptor 계산 및 필터링 파이프라인

### 주요 기능
- Pass 0: Sampling
- Pass 1: Statistics + Variance Filtering
- Pass 2: Spearman Correlation
- Pass 3: VIF Multicollinearity
- Pass 4: HSIC + RDC Nonlinear Detection

## 프로젝트 구조

```
descriptor-toolkit/
├── molecular_descriptor_toolkit/    # 메인 패키지
│   ├── __init__.py
│   ├── cli.py                       # CLI 진입점
│   ├── config/                      # 설정 관리
│   │   ├── settings.py              # Config 클래스 정의
│   │   └── loader.py                # YAML 로더
│   ├── filtering/                   # 필터링 파이프라인
│   │   ├── pipeline.py              # 메인 파이프라인
│   │   ├── passes/                  # 각 Pass 구현
│   │   ├── io/                      # Parquet I/O
│   │   └── utils/                   # 유틸리티
│   ├── preprocessing/               # 전처리 모듈
│   └── workflows/                   # 워크플로우
├── tests/
├── setup.py
└── requirements.txt
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

### 2025-11-25: 초기 버그 수정

**피드백 분석 및 수정 완료:**

1. **패키지 구조 문제** (High)
   - 문제: `molecular_descriptor_toolkit/` 디렉토리 없이 import 경로가 해당 패키지를 참조
   - 해결: 최상위에 `molecular_descriptor_toolkit/` 디렉토리 생성 후 모듈 이동

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
   - 해결: `self.io_cfg.n_metadata`로 수정 (`filtering/pipeline.py:628`)

4. **self.config 참조 오류** (Medium)
   - 문제: `pass1_statistics.py`에서 `self.config` 참조 (존재하지 않음)
   - 해결: `self.filtering_cfg`로 수정 (`filtering/passes/pass1_statistics.py:281`)

**테스트 결과:** ✅ 전체 파이프라인 정상 동작 확인 (CPU 모드)

### 2025-11-25: 추가 버그 수정 및 기능 추가

**추가 버그 수정:**

5. **VIF 클래스 생성자 인자 오류** (Medium)
   - 문제: `pipeline.py`에서 VIFFilteringPassGPUWithClusters 호출 시 잘못된 인자 전달
   - 해결: `(self.filtering_cfg, self.io_cfg, self.system_cfg, self.device)`로 수정 (`filtering/pipeline.py:453-457`)

**새 기능: `mdt process-all` 통합 명령어**

XML/CSV/Parquet → Descriptor 계산 → Filtering을 한 번에 처리하는 통합 명령어 추가.
**입력 형식은 파일 확장자로 자동 감지:**

| 확장자 | 처리 방식 |
|--------|-----------|
| `.xml`, `.xml.gz` | PubChem XML 파싱 → SMILES 추출 → Descriptor → Filtering |
| `.csv` | CSV에서 SMILES 읽기 → Descriptor → Filtering |
| `.parquet` | Parquet에서 SMILES 읽기 → Descriptor → Filtering |

```bash
# CSV/Parquet 입력 (3단계)
mdt process-all \
  --input molecules.csv \
  --output-dir results/ \
  --smiles-col SMILES \
  --id-col CID \
  --cpu

# PubChem XML 입력 (4단계: XML 파싱 추가)
mdt process-all \
  --input compounds.xml.gz \
  --output-dir results/ \
  --filter-property "H-Bond Donor Count" \
  --filter-max 5 \
  --cpu
```

**처리 단계:**

CSV/Parquet 입력:
1. 스키마 생성 (또는 제공된 스키마 사용)
2. RDKit + Mordred descriptor 계산
3. 4단계 필터링 파이프라인 실행

XML 입력:
1. **PubChem XML 파싱** (SMILES, InChI, 물성 추출)
2. 스키마 생성
3. Descriptor 계산
4. 필터링 파이프라인

**테스트 결과:** ✅ CSV 입력 100개 분자 테스트 완료
- 1775개 descriptor 계산
- 9개 최종 descriptor로 필터링 (99.5% 감소)

### 2025-11-25: examples.py 및 CLI GPU 모드 수정

**피드백:**
- `examples.py`가 여전히 flat Config 형식 사용
- CLI에서 GPU/CPU 모드가 항상 CPU로 표시됨 (`validate_and_finalize()` 미호출)

**수정 완료:**

6. **examples.py flat Config → 섹션 기반 변경**
   - 모든 example 함수에서 섹션 기반 Config 사용
   - `validate_and_finalize()` 호출 추가

7. **CLI GPU/CPU 모드 표시 오류**
   - 문제: `Config.using_gpu` 속성이 `validate_and_finalize()` 호출 전에는 정확하지 않음
   - 해결: 모든 CLI 함수에서 Config 생성 후 `config.validate_and_finalize()` 호출 추가
   - 수정 위치:
     - `run_full_pipeline()` (cli.py:271)
     - `run_filter()` (cli.py:326)
     - `run_process_all()` (cli.py:558-561)

**수정 후 동작:**
```
🚀 Molecular Descriptor Toolkit v1.0.0
📊 Mode: GPU  # 또는 CPU (GPU 사용 불가 시)
```

---

## 개발 가이드

### CLI 사용법
```bash
# ✨ 통합 파이프라인 (XML → SMILES → Descriptors → Filtering)
mdt process-all --input compounds.xml.gz --output-dir results/ --cpu

# ✨ 통합 파이프라인 (CSV/Parquet → Descriptors → Filtering)
mdt process-all --input molecules.csv --output-dir results/ --smiles-col SMILES --id-col CID --cpu

# 필터링만 실행 (이미 descriptor가 있는 경우)
mdt run --parquet-glob "data/*.parquet" --output-dir results/

# 개별 Pass 실행
mdt filter pass0 --parquet-glob "data/*.parquet" --output-dir results/
mdt filter pass1 --parquet-glob "data/*.parquet" --output-dir results/
mdt filter pass234 --parquet-glob "data/*.parquet" --output-dir results/

# 전처리만
mdt preprocess xml-to-parquet --input compounds.xml --output molecules.parquet
mdt preprocess generate-schema --input data/ --output schema.json --quick
mdt preprocess calculate-descriptors --input data.csv --output descriptors.parquet --schema schema.json
```

### 설치
```bash
pip install -e .
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
