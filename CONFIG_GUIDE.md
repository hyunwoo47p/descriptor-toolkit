# 📐 Configuration System Guide

## ✨ 개요

Molecular Descriptor Toolkit은 **계층형 SSOT (Single Source of Truth)** 설계의 설정 시스템을 사용합니다.

### 핵심 특징

- ✅ **계층 구조**: 섹션별로 관련 설정 그룹화
- ✅ **SSOT**: 모든 설정이 하나의 Config 객체에
- ✅ **관심사 분리**: 각 모듈은 필요한 섹션만 사용
- ✅ **다양한 로딩 방법**: Python API, YAML, JSON, ENV, CLI
- ✅ **하위 호환성**: 기존 flat 접근 지원 (deprecation warning)
- ✅ **타입 안전**: Literal typing과 validation

---

## 🎯 빠른 시작

### 1. Python API

```python
from molecular_descriptor_toolkit.config import Config, load_config

# 기본 설정
config = Config()
config.validate_and_finalize()

# 섹션 접근
print(config.device.prefer_gpu)         # True
print(config.filtering.vif_threshold)  # 10.0
```

### 2. YAML 파일

```yaml
# config.yaml
device:
  prefer_gpu: false
  gpu_id: 1

io:
  parquet_glob: "data/*.parquet"
  output_dir: "results/"

filtering:
  variance_threshold: 0.001
  vif_threshold: 8.0
```

```python
from molecular_descriptor_toolkit.config import load_config

config = load_config("config.yaml")
```

### 3. Override 사용

```python
config = load_config(
    "config.yaml",
    overrides={
        "filtering.vif_threshold": 15.0,
        "device.gpu_id": 2,
    }
)
```

### 4. 환경 변수

```bash
export MDTK_DEVICE_GPU_ID=1
export MDTK_FILTERING_VIF_THRESHOLD=8.0
```

```python
config = load_config()  # 자동으로 환경 변수 읽음
```

---

## 📚 설정 섹션

### DeviceConfig (device)

GPU와 디바이스 설정

```python
device:
  prefer_gpu: bool = True          # GPU 가속 활성화
  gpu_id: int = 0                  # CUDA 디바이스 ID
  device: Optional[str] = None     # 'cuda' or 'cpu' (자동 감지)
```

**예시:**
```python
config.device.prefer_gpu = False  # CPU 강제 사용
```

---

### IOConfig (io)

입출력 및 체크포인트 설정

```python
io:
  parquet_glob: str = ""                          # 입력 Parquet 파일 패턴
  output_dir: str = "output"                      # 출력 디렉토리
  descriptor_columns: Optional[List[str]] = None  # 처리할 컬럼 (None=전체)
  n_metadata: int = 6                             # 메타데이터 컬럼 수
  row_group_size: int = 10000                     # Parquet row group 크기
  part_bytes_target: int = 256MB                  # Part 파일 목표 크기
  atomic_write: bool = True                       # Atomic write 사용
  resume_mode: "off" | "scan_parts" = "scan_parts"  # 체크포인트 재개 모드
```

---

### PreprocessingConfig (preprocessing)

분자 표준화 및 파싱 설정

```python
preprocessing:
  profile: "neutral" | "ligand_only" | "complex_included" = "neutral"
  std_core: bool = True                    # RDKit Cleanup 사용
  use_normalizer: bool = False             # RDKit Normalizer
  use_reionizer: bool = False              # RDKit Reionizer
  use_metal_disconnector: bool = False     # Metal Disconnector
  keep_largest_fragment: bool = False      # 최대 fragment만 유지
  primary_id_col: str = "CID"              # Primary ID 컬럼
  smiles_col: str = "SMILES::Absolute"     # SMILES 컬럼
  inchi_col: str = "InChI::Standard"       # InChI 컬럼
  parse_order: ["smiles", "inchi"]         # 파싱 우선순위
```

---

### DescriptorConfig (descriptor)

Descriptor 계산 설정

```python
descriptor:
  descriptor_set: "rdkit" | "mordred" | "both" = "both"
  descriptor_include: List[str] = []       # 포함할 descriptor (whitelist)
  descriptor_exclude: List[str] = []       # 제외할 descriptor (blacklist)
  per_molecule_timeout_sec: int = 60       # 분자당 타임아웃 (초)
  workers: int = 0                         # 워커 프로세스 수 (0=자동)
```

**워커 수 자동 감지:**
```python
config.descriptor.workers = 0  # 0이면 CPU 코어 수만큼 자동 설정
actual_workers = config.descriptor.get_workers()  # 실제 워커 수 반환
```

---

### FilteringConfig (filtering)

필터링 파이프라인 설정 (Pass 0-4)

#### Pass 0: Sampling
```python
filtering:
  sample_per_file: Optional[int] = None     # 파일당 샘플 수
  file_independent_sampling: bool = False   # 파일별 독립 샘플링
```

#### Pass 1: Statistics & Variance
```python
filtering:
  variance_threshold: float = 0.002         # 최소 normalized variance
  max_missing_ratio: float = 0.5            # 최대 결측치 비율
  min_effective_n: int = 100                # 최소 유효 샘플 수
  range_mode: "MINMAX" | "TRIMMED" | "IQR" = "TRIMMED"
  trim_lower: float = 0.025                 # 하위 백분위 (2.5%)
  trim_upper: float = 0.975                 # 상위 백분위 (97.5%)
  force_recompute: bool = False             # 통계 재계산 강제
```

#### Pass 2: Correlation
```python
filtering:
  spearman_threshold: float = 0.95          # Spearman 상관 임계값
  m: int = 64                               # CountSketch buckets
  r: int = 8                                # CountSketch repetitions
```

#### Pass 3: VIF
```python
filtering:
  vif_threshold: float = 10.0               # VIF 임계값
```

#### Pass 4: Nonlinear Detection
```python
filtering:
  nonlinear_threshold: float = 0.30         # 일반 비선형 임계값
  hsic_threshold: float = 0.30              # HSIC 임계값
  rdc_threshold: float = 0.30               # RDC 임계값
```

#### Clustering
```python
filtering:
  resolution: float = 1.0                   # Leiden 해상도
  n_iterations: int = -1                    # 반복 횟수 (-1=자동)
```

#### Binary Skew Filter
```python
filtering:
  use_binary_skew_filter: bool = False      # 활성화
  binary_skew_threshold: float = 0.40       # 이진 descriptor 임계값
  binary_minority_frac: float = 0.10        # 최소 소수 클래스 비율
```

---

### SystemConfig (system)

시스템 전역 설정

```python
system:
  random_seed: int = 42                                  # 재현성을 위한 시드
  deterministic_ops: bool = True                         # 결정적 연산 강제
  checkpoint: bool = True                                # 체크포인트 활성화
  verbose: bool = True                                   # 상세 로깅
  log_level: "DEBUG" | "INFO" | "WARN" | "ERROR" = "INFO"
  log_json: bool = False                                 # JSON 로그
  log_dir: str = "logs"                                  # 로그 디렉토리
  progress_every_n: int = 1000                           # 진행률 출력 간격
  error_policy: "continue" | "skip_molecule" | "fail_fast" = "continue"
```

---

## 🔧 고급 사용법

### 1. 설정 저장

```python
from molecular_descriptor_toolkit.config import save_config

config = load_config("config.yaml")
config.filtering.vif_threshold = 15.0

# YAML로 저장
save_config(config, "modified_config.yaml")

# JSON으로 저장
save_config(config, "config.json")
```

### 2. 설정 비교 (Diff)

```python
from molecular_descriptor_toolkit.config import config_diff

cfg1 = load_config("exp1.yaml")
cfg2 = load_config("exp2.yaml")

diffs = config_diff(cfg1, cfg2)
for diff in diffs:
    print(diff)
# 출력:
# filtering.vif_threshold: 10.0 → 8.0
# device.gpu_id: 0 → 1
```

### 3. 설정 동결 (Freeze)

```python
from molecular_descriptor_toolkit.config import freeze

config = load_config("config.yaml").validate_and_finalize()
freeze(config)

# 파이프라인 실행
run_pipeline(config)  # config는 이제 수정 불가
```

**주의:** Python dataclass의 한계로 완전한 immutability는 보장되지 않습니다. 프로덕션 환경에서 권장됩니다.

### 4. 우선순위 체인

설정은 다음 우선순위로 로드됩니다:

1. **기본값** (dataclass defaults)
2. **YAML/JSON 파일** (`load_config(path=...)`)
3. **환경 변수** (`MDTK_*` prefix)
4. **명시적 override** (`overrides={...}`)

```python
# 1. 기본값
config = Config()  # 모든 기본값 사용

# 2. YAML 파일
config = load_config("base_config.yaml")  # 파일이 기본값 덮어씀

# 3. 환경 변수가 자동으로 파일 덮어씀
# export MDTK_FILTERING_VIF_THRESHOLD=12.0

# 4. Override가 최종 우선순위
config = load_config(
    "base_config.yaml",
    overrides={"filtering.vif_threshold": 15.0}  # 최종값: 15.0
)
```

### 5. 모듈별 Config 전달

각 모듈은 필요한 섹션만 받습니다:

```python
def run_descriptors(
    descriptor_cfg: DescriptorConfig,
    io_cfg: IOConfig,
    system_cfg: SystemConfig
):
    workers = descriptor_cfg.get_workers()
    output = Path(io_cfg.output_dir) / "descriptors.parquet"
    # ...

# 사용
config = load_config("config.yaml")
run_descriptors(
    config.descriptor,
    config.io,
    config.system
)
```

---

## 🔄 마이그레이션 가이드

### 기존 Flat Config에서 마이그레이션

**Before (Flat)**:
```python
config = Config(
    prefer_gpu=True,
    gpu_id=0,
    parquet_glob="data/*.parquet",
    output_dir="results/",
    variance_threshold=0.001,
    vif_threshold=10.0,
)
```

**After (Hierarchical)**:
```python
config = Config(
    device=DeviceConfig(prefer_gpu=True, gpu_id=0),
    io=IOConfig(parquet_glob="data/*.parquet", output_dir="results/"),
    filtering=FilteringConfig(variance_threshold=0.001, vif_threshold=10.0),
)

# 또는 더 간단하게 (defaults 사용)
config = load_config(overrides={
    "device.prefer_gpu": True,
    "io.parquet_glob": "data/*.parquet",
    "filtering.variance_threshold": 0.001,
})
```

### 하위 호환성

기존 flat 접근도 작동합니다 (deprecation warning):

```python
config = Config()

# ⚠️ Deprecated (하지만 작동함)
prefer_gpu = config.prefer_gpu
variance = config.variance_threshold

# ✅ 권장
prefer_gpu = config.device.prefer_gpu
variance = config.filtering.variance_threshold
```

---

## 📝 체크리스트

### 새 프로젝트 시작 시

- [ ] `config.yaml` 파일 생성
- [ ] 필요한 섹션만 오버라이드
- [ ] `validate_and_finalize()` 호출
- [ ] 실험별로 config diff 저장

### 모듈 개발 시

- [ ] 필요한 섹션만 파라미터로 받기
  ```python
  def my_function(cfg: FilteringConfig, io: IOConfig):
      ...
  ```
- [ ] Config 전체를 받지 않기
- [ ] Docstring에 필요한 섹션 명시

### 실험 관리

- [ ] 실험마다 별도 YAML 저장
- [ ] Config diff를 실험 로그에 기록
- [ ] 재현 가능하도록 YAML + 코드 버전 관리

---

## 🎓 예제

### 예제 1: 필터링만

```python
from molecular_descriptor_toolkit.config import load_config
from molecular_descriptor_toolkit.filtering import DescriptorPipeline

config = load_config(overrides={
    "io.parquet_glob": "preprocessed/*.parquet",
    "io.output_dir": "filtered/",
    "filtering.variance_threshold": 0.001,
    "filtering.vif_threshold": 8.0,
})

pipeline = DescriptorPipeline(config)
pipeline.run()
```

### 예제 2: 전체 파이프라인

```yaml
# pipeline_config.yaml
preprocessing:
  profile: ligand_only
  std_core: true

descriptor:
  descriptor_set: both
  workers: 8

filtering:
  variance_threshold: 0.001
  vif_threshold: 8.0

system:
  random_seed: 42
  verbose: true
```

```python
config = load_config("pipeline_config.yaml")

# Preprocessing
preprocess(config.preprocessing, config.io, config.system)

# Descriptor calculation
calculate_descriptors(config.descriptor, config.io, config.system)

# Filtering
filter_pipeline(config.filtering, config.io, config.system)
```

### 예제 3: 실험 비교

```python
from molecular_descriptor_toolkit.config import load_config, config_diff

# 두 실험 설정 로드
exp1 = load_config("experiments/exp1.yaml")
exp2 = load_config("experiments/exp2.yaml")

# 차이점 출력
print("Experiment differences:")
for diff in config_diff(exp1, exp2):
    print(f"  - {diff}")

# 실험 실행
run_experiment(exp1, name="exp1")
run_experiment(exp2, name="exp2")
```

---

## 📖 참고

- **타입 안전성**: IDE에서 자동완성 지원
- **Validation**: `validate_and_finalize()` 호출 시 자동 검증
- **문서화**: 각 필드에 docstring 포함
- **확장성**: 새 섹션 추가 용이

---

**버전**: 1.0.0  
**마지막 업데이트**: 2024-11-10
