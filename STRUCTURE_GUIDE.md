# 📁 Python 패키지 구조 가이드

## 목차
1. [__init__.py의 역할](#__init__py의-역할)
2. [config/ 디렉토리](#config-디렉토리)
3. [각 모듈별 표준 구조](#각-모듈별-표준-구조)
4. [파일 명명 규칙](#파일-명명-규칙)
5. [Import 패턴](#import-패턴)

---

## __init__.py의 역할

### 1. 최상위 __init__.py (패키지 루트)

**위치**: `molecular_descriptor_toolkit/__init__.py`

**목적**:
- 패키지의 "얼굴" - 사용자가 가장 먼저 보는 것
- 가장 자주 쓰는 클래스/함수만 export
- 패키지 메타데이터 정의

**내용**:
```python
"""
패키지 docstring
- 패키지가 무엇인지
- 주요 기능
- 간단한 사용 예제
"""

# 버전 정보
__version__ = "1.0.0"
__author__ = "KAERI_UES"

# 핵심 클래스만 export (가장 자주 쓰이는 것)
from molecular_descriptor_toolkit.config import Config

# __all__로 명시적 export 관리
__all__ = [
    "Config",
]
```

**원칙**:
- ✅ 가장 자주 쓰는 1-3개 클래스만
- ✅ 짧고 간결하게
- ❌ 모든 submodule import 하지 않기
- ❌ 복잡한 로직 넣지 않기

---

### 2. 서브모듈 __init__.py

#### 2-1. config/__init__.py (설정 모듈)

**목적**: Config 관련 모든 것을 한 곳에서

**내용**:
```python
"""Configuration module for the toolkit"""

# 주요 클래스 import
from molecular_descriptor_toolkit.config.settings import (
    Config,
    RangeMode,
)

# 하위 호환성을 위한 별칭
PipelineConfig = Config

# Export 목록
__all__ = [
    "Config",
    "RangeMode",
    "PipelineConfig",  # 별칭도 포함
]
```

**원칙**:
- ✅ 모듈의 모든 public 클래스 export
- ✅ 별칭(alias) 제공 시 여기서 정의
- ✅ 짧고 명확한 docstring

---

#### 2-2. filtering/__init__.py (큰 모듈)

**목적**: 
- 핵심 클래스만 노출
- 내부 구조 숨기기

**내용 (방법 1: 직접 import)**:
```python
"""Filtering pipeline module"""

from molecular_descriptor_toolkit.filtering.pipeline import DescriptorPipeline

__all__ = ["DescriptorPipeline"]
```

**내용 (방법 2: Lazy import - 의존성 문제 있을 때)**:
```python
"""Filtering pipeline module - Lazy imports for optional dependencies"""

__all__ = ["DescriptorPipeline"]

def __getattr__(name):
    """Lazy import to avoid dependency issues"""
    if name == "DescriptorPipeline":
        from molecular_descriptor_toolkit.filtering.pipeline import DescriptorPipeline
        return DescriptorPipeline
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
```

**선택 기준**:
- 의존성 없거나 필수 → 방법 1 (직접 import)
- 선택적 의존성(torch, pyarrow 등) → 방법 2 (Lazy import)

---

#### 2-3. filtering/passes/__init__.py (여러 Pass 클래스)

**목적**: 
- 모든 Pass 클래스를 한 곳에서 import
- 내부 파일명 숨기기

**내용**:
```python
"""Filtering passes - Individual pass implementations"""

__all__ = [
    "SamplingPass",
    "StatisticsAndVarianceFilter",
    "SpearmanComputerGPU",
    "VIFFilteringPassGPUWithClusters",
    "NonlinearDetectionPassGPU",
    # Helper classes
    "GraphBuilder",
    "SeedManager",
]

# Lazy imports (의존성 있을 때)
def __getattr__(name):
    if name == "SamplingPass":
        from molecular_descriptor_toolkit.filtering.passes.pass0_sampling import SamplingPass
        return SamplingPass
    elif name == "StatisticsAndVarianceFilter":
        from molecular_descriptor_toolkit.filtering.passes.pass1_statistics import (
            StatisticsAndVarianceFilter,
        )
        return StatisticsAndVarianceFilter
    # ... 나머지 클래스들
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
```

**원칙**:
- ✅ 파일명(`pass0_sampling.py`)은 내부 구현
- ✅ 클래스명(`SamplingPass`)만 외부 노출
- ✅ 사용자는 `from filtering.passes import SamplingPass` 만 알면 됨

---

#### 2-4. filtering/utils/__init__.py (유틸리티)

**내용**:
```python
"""Utility functions for filtering"""

# 의존성 없는 것은 직접 import
from molecular_descriptor_toolkit.filtering.utils.logging import log

__all__ = [
    "log",
    "get_optimal_device",  # Lazy import
]

# 의존성 있는 것은 lazy import
def __getattr__(name):
    if name == "get_optimal_device":
        from molecular_descriptor_toolkit.filtering.utils.gpu import get_optimal_device
        return get_optimal_device
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
```

**원칙**:
- ✅ 의존성 없으면 직접 import
- ✅ 의존성 있으면 lazy import
- ✅ __all__에는 모두 나열

---

#### 2-5. 빈 __init__.py (예: tests/, workflows/)

**내용**:
```python
"""Tests module"""
# 비워두거나 간단한 docstring만
```

**원칙**:
- ✅ 최소한 docstring은 넣기
- ✅ 외부에 노출할 필요 없으면 비워두기

---

## config/ 디렉토리

### 구조
```
config/
├── __init__.py          # Config, RangeMode export
└── settings.py          # 실제 설정 클래스 정의
```

### settings.py 표준 구조

```python
"""
Configuration settings for the toolkit

이 파일의 역할:
1. 모든 설정을 한 곳에 모음
2. Type hints로 타입 명확화
3. Default 값 제공
4. Validation 로직
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List
from pathlib import Path


# ========== Enums (설정 옵션) ==========
class RangeMode(Enum):
    """Range calculation modes"""
    MINMAX = "minmax"
    TRIMMED = "trimmed"
    IQR = "iqr"


# ========== Helper Functions (설정 초기화용) ==========
def _auto_detect_device() -> str:
    """Auto-detect best available device"""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


# ========== Main Config Class ==========
@dataclass
class Config:
    """
    Unified configuration for molecular descriptor toolkit
    
    Usage:
        config = Config(
            parquet_glob="data/*.parquet",
            output_dir="results/",
            prefer_gpu=True,
        )
    
    Attributes:
        - Device settings
        - Input/Output paths
        - Pass-specific parameters
        - System settings
    """
    
    # ===== Device Settings =====
    prefer_gpu: bool = True
    gpu_id: int = 0
    device: Optional[str] = None  # Auto-detected
    
    # ===== Input/Output =====
    parquet_glob: str = ""
    output_dir: str = "output"
    
    # ===== Pass 0: Sampling =====
    sample_per_file: Optional[int] = None
    
    # ===== Pass 1: Statistics =====
    variance_threshold: float = 0.002
    max_missing_ratio: float = 0.5
    
    # ===== Pass 2: Correlation =====
    spearman_threshold: float = 0.95
    
    # ===== System =====
    random_seed: int = 42
    checkpoint: bool = True
    verbose: bool = True
    
    def __post_init__(self):
        """Validation and initialization after dataclass creation"""
        # Device detection
        if self.device is None:
            self.device = _auto_detect_device() if self.prefer_gpu else "cpu"
        
        # Validation
        if self.variance_threshold < 0:
            raise ValueError("variance_threshold must be >= 0")
        
        # Path conversion
        self.output_dir = Path(self.output_dir)
    
    # ===== Properties (computed attributes) =====
    @property
    def using_gpu(self) -> bool:
        """Check if GPU is being used"""
        return self.device == "cuda"
    
    # ===== Methods =====
    def get_device_info(self) -> str:
        """Get human-readable device information"""
        if self.using_gpu:
            try:
                import torch
                gpu_name = torch.cuda.get_device_name(self.gpu_id)
                gpu_mem = torch.cuda.get_device_properties(self.gpu_id).total_memory / 1e9
                return f"GPU: {gpu_name} ({gpu_mem:.1f} GB)"
            except:
                return "GPU: Unknown"
        return "Device: CPU"


# ===== Backward Compatibility =====
PipelineConfig = Config  # Alias
```

**원칙**:
- ✅ **하나의 큰 Config 클래스** - 모든 설정을 한 곳에
- ✅ **Dataclass 사용** - 간결하고 타입 안전
- ✅ **섹션 주석으로 구분** (`# ===== Pass 1 =====`)
- ✅ **__post_init__으로 validation**
- ✅ **Property로 computed 값**
- ✅ **별칭은 파일 끝에**

---

## 각 모듈별 표준 구조

### 1. filtering/passes/passX_*.py (Pass 구현)

```python
"""
Pass X: 기능 설명

설명:
- 이 Pass가 하는 일
- 입력/출력
- 알고리즘 설명
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path

from molecular_descriptor_toolkit.config import Config
from molecular_descriptor_toolkit.filtering.utils import log


class PassXName:
    """
    Pass X 클래스
    
    Attributes:
        config: Configuration object
        device: torch.device
        verbose: Verbose logging flag
    
    Methods:
        run: Main execution method
        _helper_method: Private helper
    """
    
    def __init__(self, config: Config, verbose: bool = True):
        """
        Initialize Pass X
        
        Args:
            config: Configuration object
            verbose: Enable verbose logging
        """
        self.config = config
        self.verbose = verbose
    
    def run(self, input_data, **kwargs):
        """
        Main execution method
        
        Args:
            input_data: Input data
            **kwargs: Additional arguments
        
        Returns:
            Processed result
        
        Raises:
            ValueError: If input is invalid
        """
        self._log("Starting Pass X")
        
        # Implementation
        result = self._process(input_data)
        
        self._log("Pass X completed")
        return result
    
    def _process(self, data):
        """Private helper method"""
        # Implementation
        pass
    
    def _log(self, msg: str):
        """Helper for logging"""
        log(msg, self.verbose)
```

**원칙**:
- ✅ 클래스 기반
- ✅ `__init__`에서 config 받기
- ✅ `run()` 메서드가 main entry point
- ✅ Private 메서드는 `_` prefix
- ✅ 명확한 docstring

---

### 2. filtering/utils/*.py (유틸리티)

```python
"""
Utility: 기능 설명

이 모듈이 제공하는 것:
- 함수 1
- 함수 2
"""

from typing import Optional


def utility_function(arg1: str, arg2: int = 0) -> str:
    """
    함수 설명
    
    Args:
        arg1: 설명
        arg2: 설명 (default: 0)
    
    Returns:
        결과 설명
    
    Examples:
        >>> utility_function("test", 5)
        'test_5'
    """
    return f"{arg1}_{arg2}"


class UtilityClass:
    """간단한 유틸리티는 클래스로도 가능"""
    pass
```

**원칙**:
- ✅ 함수 위주 (stateless)
- ✅ Type hints 필수
- ✅ Examples in docstring
- ✅ 짧고 독립적

---

### 3. filtering/io/*.py (I/O 처리)

```python
"""
I/O: Parquet file handling

Functions:
- iter_batches: Stream data in batches
- save_parquet: Save to parquet file
"""

import pyarrow as pa
import pyarrow.parquet as pq
from typing import Iterator, List, Tuple
from pathlib import Path


def iter_batches(
    parquet_paths: List[str],
    columns: List[str],
    batch_rows: int = 10000,
) -> Iterator[Tuple[np.ndarray, int]]:
    """
    Iterate over parquet files in batches
    
    Args:
        parquet_paths: List of parquet file paths
        columns: Columns to read
        batch_rows: Rows per batch
    
    Yields:
        Tuple of (data_array, offset)
    
    Examples:
        >>> for data, offset in iter_batches(paths, cols):
        ...     process(data)
    """
    # Implementation
    pass
```

**원칙**:
- ✅ 함수 위주
- ✅ Generator 사용 (메모리 효율)
- ✅ Type hints with typing module
- ✅ 명확한 yields/returns

---

### 4. filtering/pipeline.py (메인 파이프라인)

```python
"""
Main Pipeline - Descriptor filtering pipeline

Architecture:
- Pass 0: Sampling
- Pass 1: Statistics
- Pass 2: Correlation
- Pass 3: VIF
- Pass 4: Nonlinear
"""

from typing import Dict, List, Optional, Any
from pathlib import Path

from molecular_descriptor_toolkit.config import Config
from molecular_descriptor_toolkit.filtering.passes import (
    SamplingPass,
    StatisticsAndVarianceFilter,
    # ... others
)


class DescriptorPipeline:
    """
    Main descriptor filtering pipeline
    
    Attributes:
        config: Configuration
        device: Computation device
        passes: Dictionary of pass instances
    
    Methods:
        run: Execute full pipeline
        run_pass0: Execute Pass 0 only
        run_pass1: Execute Pass 1 only
    """
    
    def __init__(self, config: Config):
        """Initialize pipeline with configuration"""
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize passes
        self._init_passes()
    
    def _init_passes(self):
        """Initialize all pass instances"""
        self.pass0 = SamplingPass(self.config)
        self.pass1 = StatisticsAndVarianceFilter(self.config)
        # ...
    
    def run(self) -> Dict[str, Any]:
        """
        Execute full pipeline (Pass 0-4)
        
        Returns:
            Dictionary with results:
            {
                'final_columns': List[str],
                'stats': Dict,
                'removed_count': int,
            }
        """
        # Implementation
        pass
    
    def run_pass0(self):
        """Execute Pass 0: Sampling only"""
        pass
    
    def run_pass1(self):
        """Execute Pass 1: Statistics only"""
        pass
```

**원칙**:
- ✅ 큰 클래스 (orchestration 담당)
- ✅ `__init__`에서 모든 pass 초기화
- ✅ `run()` = 전체 실행
- ✅ `run_passX()` = 개별 실행
- ✅ 결과는 Dict 반환

---

## 파일 명명 규칙

### 1. Python 파일명

**패턴**: `lowercase_with_underscores.py`

```
✅ pass0_sampling.py
✅ parquet_reader.py
✅ descriptor_calculator.py

❌ Pass0Sampling.py      # 클래스명 아님
❌ parquetReader.py      # camelCase 사용 안함
❌ descriptor-calculator.py  # 하이픈 사용 안함
```

**규칙**:
- 파일명은 모두 소문자
- 단어 구분은 `_`
- 파일 하나 = 클래스 하나가 원칙

---

### 2. 클래스명

**패턴**: `PascalCase` (각 단어 첫 글자 대문자)

```python
✅ class DescriptorPipeline:
✅ class SamplingPass:
✅ class VIFFilteringPassGPU:

❌ class descriptor_pipeline:   # snake_case 아님
❌ class samplingPass:          # camelCase 아님
```

---

### 3. 함수/메서드명

**패턴**: `lowercase_with_underscores`

```python
✅ def run_pipeline():
✅ def get_device_info():
✅ def _private_helper():

❌ def runPipeline():      # camelCase 아님
❌ def GetDeviceInfo():    # PascalCase 아님
```

---

### 4. 상수명

**패턴**: `UPPERCASE_WITH_UNDERSCORES`

```python
✅ DEFAULT_BATCH_SIZE = 10000
✅ MAX_ITERATIONS = 100

❌ default_batch_size = 10000  # 소문자 아님
❌ DefaultBatchSize = 10000    # PascalCase 아님
```

---

### 5. 변수명

**패턴**: `lowercase_with_underscores`

```python
✅ sample_count = 100
✅ parquet_paths = ["a.parquet", "b.parquet"]

❌ sampleCount = 100       # camelCase 아님
❌ SampleCount = 100       # PascalCase 아님
```

---

## Import 패턴

### 1. Import 순서 (PEP 8)

```python
"""Module docstring"""

# 1. Standard library imports
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional

# 2. Third-party imports
import numpy as np
import pandas as pd
import torch

# 3. Local application imports
from molecular_descriptor_toolkit.config import Config
from molecular_descriptor_toolkit.filtering.utils import log
```

**규칙**:
- 3개 그룹으로 나누기
- 그룹 사이 빈 줄
- 각 그룹 내에서 알파벳 순

---

### 2. Import 스타일

```python
# ✅ 좋은 예
from molecular_descriptor_toolkit.config import Config, RangeMode
from molecular_descriptor_toolkit.filtering import DescriptorPipeline

# ✅ 긴 경우 괄호 사용
from molecular_descriptor_toolkit.filtering.passes import (
    SamplingPass,
    StatisticsAndVarianceFilter,
    SpearmanComputerGPU,
)

# ❌ 나쁜 예
from molecular_descriptor_toolkit.config import *  # wildcard import 금지
import molecular_descriptor_toolkit.filtering.passes.pass0_sampling  # 너무 김
```

---

### 3. Relative vs Absolute Import

```python
# 현재 위치: molecular_descriptor_toolkit/filtering/passes/pass0_sampling.py

# ✅ Absolute import (권장)
from molecular_descriptor_toolkit.config import Config
from molecular_descriptor_toolkit.filtering.utils import log

# ⚠️ Relative import (패키지 내부에서만)
from ...config import Config
from ..utils import log

# 선택: Absolute가 더 명확함
```

---

## 요약 체크리스트

### __init__.py
- [ ] Docstring 있음
- [ ] 주요 클래스만 export
- [ ] `__all__` 정의
- [ ] 의존성 있으면 lazy import

### config/settings.py
- [ ] Dataclass 사용
- [ ] 모든 설정 한 곳에
- [ ] Type hints 완전
- [ ] `__post_init__` validation
- [ ] Property로 computed values

### 각 Pass 파일
- [ ] 클래스 기반
- [ ] `__init__`에서 config
- [ ] `run()` 메서드
- [ ] Private 메서드 `_` prefix
- [ ] 명확한 docstring

### 명명 규칙
- [ ] 파일명: `lowercase_with_underscores.py`
- [ ] 클래스: `PascalCase`
- [ ] 함수/메서드: `lowercase_with_underscores`
- [ ] 상수: `UPPERCASE_WITH_UNDERSCORES`

### Import
- [ ] 3개 그룹 (stdlib, third-party, local)
- [ ] 그룹 사이 빈 줄
- [ ] Absolute import 사용
- [ ] `from X import Y` 형식
