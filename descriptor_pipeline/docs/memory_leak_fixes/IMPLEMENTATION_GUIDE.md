# 실전 적용 가이드: 메모리 누수 수정

## 🎯 즉시 적용할 수정사항 (우선순위별)

### Priority 1: Critical Fixes (즉시 수정 필수)

#### 1. `descriptor_pipeline/io/parquet_reader_duckdb.py`

**문제 1: 중복 함수 정의**
```python
# 라인 226-407 삭제
# iter_batches_duckdb 함수가 두 번 정의되어 있음
```

**문제 2: DataFrame view 참조**
```python
# 기존 (라인 103, 156, 193, 356, 393)
X = df_batch[columns].values.astype(np.float64)

# 수정 후
X = df_batch[columns].values.copy().astype(np.float64)
```

**전체 수정 위치**:
- 라인 103: 첫 번째 사용
- 라인 156: 샘플링 모드 - 전체 사용
- 라인 193: 샘플링 모드 - 샘플링
- 라인 226-407: 전체 삭제 (중복 정의)

---

#### 2. `descriptor_pipeline/core/pipeline.py`

**문제 1: 함수 호출 인자 불일치 (라인 123-125)**
```python
# 기존 (오류 발생)
columns_p2, spearman_info, indices_p2 = spearman_pass.process(
    data, columns_p1, G_spearman, self.graph_builder, self.leiden
)

# 수정 후
columns_p2, spearman_info, indices_p2 = spearman_pass.process(
    data, columns_p1, G_spearman, stats_p1
)
```

**문제 2: NumPy view 참조 (라인 142-143)**
```python
# 기존
data_p2 = data[:, indices_p2]
G_spearman_p2 = G_spearman[indices_p2][:, indices_p2]

# 수정 후
data_p2 = data[:, indices_p2].copy()
G_spearman_p2 = G_spearman[indices_p2][:, indices_p2].copy()

# 원본 삭제 추가
del data, G_spearman
gc.collect()
```

**문제 3: Pass 3 데이터 준비 (라인 169)**
```python
# 기존
data_p3 = data[:, indices_p3]

# 수정 후
data_p3 = data_p2[:, indices_p3_sub].copy()

# 이전 데이터 삭제
del data_p2
gc.collect()
```

**전체 수정 위치**:
- 라인 22: `import gc` 추가
- 라인 123-125: 함수 호출 수정
- 라인 142-143: .copy() 추가 + 원본 삭제
- 라인 169: .copy() 추가 + 원본 삭제
- 라인 96, 108, 134, 161, 204 등: `gc.collect()` 추가

---

#### 3. `descriptor_pipeline/core/similarity_gpu.py`

**문제: GPU 텐서 변환 시 참조 유지**

**3개 위치 수정 필요**:

**위치 1: 라인 158 (Spearman)**
```python
# 기존
G_cpu = G.cpu().numpy()

# 수정 후
G_cpu = G.detach().cpu().numpy().copy()
```

**위치 2: 라인 390 (HSIC)**
```python
# 기존
H_cpu = H.cpu().numpy()

# 수정 후
H_cpu = H.detach().cpu().numpy().copy()
```

**위치 3: 라인 634 (RDC)**
```python
# 기존
R_cpu = R.cpu().numpy()

# 수정 후
R_cpu = R.detach().cpu().numpy().copy()
```

---

### Priority 2: Important Memory Management

#### 4. `pipeline.py` - 명시적 메모리 정리 메서드 추가

**추가 위치: 클래스 내부 (라인 60 이후)**
```python
def _cleanup_memory(self):
    """명시적 메모리 정리"""
    import gc
    gc.collect()
    if self.using_gpu:
        import torch
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
```

**사용 위치 (각 Pass 후)**:
```python
# Pass 0 후
self._cleanup_memory()

# Pass 1 후  
self._cleanup_memory()

# Pass 2 후
self._cleanup_memory()

# Pass 3 후
self._cleanup_memory()

# Pass 4 후
self._cleanup_memory()
```

---

#### 5. `pipeline.py` - _filter_stats_by_indices 수정 (라인 253-264)

```python
def _filter_stats_by_indices(self, stats: Dict, indices: np.ndarray) -> Dict:
    """Filter statistics by indices with explicit copy"""
    stats_filtered = {}
    
    for key, value in stats.items():
        if isinstance(value, np.ndarray) and value.ndim >= 1:
            try:
                # FIXED: .copy() 추가
                stats_filtered[key] = value[indices].copy()
            except:
                stats_filtered[key] = value
        elif isinstance(value, list):
            # 리스트도 필터링
            try:
                stats_filtered[key] = [value[i] for i in indices]
            except:
                stats_filtered[key] = value
        else:
            stats_filtered[key] = value
    
    return stats_filtered
```

---

#### 6. `pipeline.py` - _load_data 개선 (라인 246-251)

```python
def _load_data(self, parquet_paths: List[str], columns: List[str]) -> np.ndarray:
    """Load data into memory with explicit cleanup"""
    batches = []
    batch_generator = None
    
    try:
        batch_generator = iter_batches(parquet_paths, columns, self.config.batch_rows)
        
        for batch_data, offset in batch_generator:
            # 명시적 복사로 generator 참조 제거
            batches.append(batch_data.copy())
            del batch_data
    
    finally:
        # Generator cleanup
        if batch_generator is not None:
            try:
                batch_generator.close()
            except:
                pass
        
        gc.collect()
    
    result = np.vstack(batches)
    
    # 중간 리스트 삭제
    del batches
    gc.collect()
    
    return result
```

---

### Priority 3: Optional Enhancements

#### 7. Context Manager for GPU Operations

**새 파일 생성: `descriptor_pipeline/utils/memory.py`**
```python
"""
Memory management utilities
"""

import gc
import torch
from typing import Optional


class GPUMemoryContext:
    """GPU 메모리 안전 컨텍스트"""
    
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device
    
    def __enter__(self):
        if self.device is not None and self.device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.device is not None and self.device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
        return False


def cleanup_memory(device: Optional[torch.device] = None):
    """명시적 메모리 정리"""
    gc.collect()
    if device is not None and device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
```

**사용 예시 (pipeline.py)**:
```python
from descriptor_pipeline.utils.memory import GPUMemoryContext

# GPU 연산 전후 자동 정리
with GPUMemoryContext(self.device):
    G_spearman = spearman_gpu.compute(parquet_paths, columns_p1, stats_p1)
```

---

## 🔧 수정 순서 (실전 적용 시)

### Step 1: 백업
```bash
# 현재 코드 백업
cp -r descriptor_pipeline descriptor_pipeline_backup_$(date +%Y%m%d_%H%M%S)
```

### Step 2: Critical Fixes (순서대로)
1. `parquet_reader_duckdb.py` 수정
2. `pipeline.py` 함수 호출 인자 수정
3. `pipeline.py` NumPy view 수정
4. `similarity_gpu.py` GPU 텐서 변환 수정

### Step 3: 테스트
```bash
# 작은 데이터셋으로 테스트
python -m descriptor_pipeline.cli.run_pipeline \
    --parquet-glob "data/test_*.parquet" \
    --output-dir "output/test" \
    --checkpoint
```

### Step 4: Memory Profiling
```python
import tracemalloc
tracemalloc.start()

# 파이프라인 실행
pipeline.run()

# 메모리 스냅샷
snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')

for stat in top_stats[:10]:
    print(stat)
```

### Step 5: Important Memory Management 적용
- `_cleanup_memory()` 메서드 추가
- `_filter_stats_by_indices()` 수정
- `_load_data()` 개선

### Step 6: 전체 데이터셋 테스트
```bash
# 메모리 모니터링하면서 실행
watch -n 1 'nvidia-smi && free -h'

# 실제 파이프라인 실행
python -m descriptor_pipeline.cli.run_pipeline \
    --parquet-glob "data/full_*.parquet" \
    --output-dir "output/full" \
    --prefer-gpu \
    --checkpoint \
    --verbose
```

---

## 📊 수정 전후 예상 효과

### 메모리 사용량 개선 예상
- **Before**: iteration마다 1-2GB 증가
- **After**: iteration마다 최대 100-200MB 증가 (정상 범위)

### 주요 개선사항
1. **NumPy view 제거**: 가장 큰 누수 원인 해결
2. **GPU 텐서 참조 제거**: GPU 메모리 안정화
3. **DataFrame 참조 제거**: pandas 메모리 누수 방지
4. **명시적 cleanup**: 각 pass 후 즉시 메모리 해제

---

## 🐛 디버깅 Tips

### 메모리 누수 확인
```python
import psutil
import os

process = psutil.Process(os.getpid())

# iteration 전
mem_before = process.memory_info().rss / 1024 / 1024  # MB
print(f"Memory before: {mem_before:.1f} MB")

# iteration 실행
# ...

# iteration 후
mem_after = process.memory_info().rss / 1024 / 1024  # MB
print(f"Memory after: {mem_after:.1f} MB")
print(f"Leaked: {mem_after - mem_before:.1f} MB")
```

### GPU 메모리 확인
```python
if torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated() / 1024 / 1024  # MB
    reserved = torch.cuda.memory_reserved() / 1024 / 1024  # MB
    print(f"GPU allocated: {allocated:.1f} MB")
    print(f"GPU reserved: {reserved:.1f} MB")
```

---

## ✅ 체크리스트

적용 전 확인:
- [ ] 코드 백업 완료
- [ ] 가상환경 활성화 확인
- [ ] 테스트 데이터 준비

Priority 1 (Critical):
- [ ] parquet_reader_duckdb.py - 중복 함수 삭제
- [ ] parquet_reader_duckdb.py - .copy() 추가 (4곳)
- [ ] pipeline.py - 함수 호출 인자 수정
- [ ] pipeline.py - NumPy view → copy (2곳)
- [ ] similarity_gpu.py - .detach().cpu().numpy().copy() (3곳)

Priority 2 (Important):
- [ ] pipeline.py - _cleanup_memory() 추가
- [ ] pipeline.py - Pass 후 cleanup 호출 (5곳)
- [ ] pipeline.py - _filter_stats_by_indices 수정
- [ ] pipeline.py - _load_data 개선

Priority 3 (Optional):
- [ ] utils/memory.py 생성
- [ ] GPUMemoryContext 적용

테스트:
- [ ] 작은 데이터셋 테스트 통과
- [ ] 메모리 프로파일링 확인
- [ ] 전체 데이터셋 테스트 통과
- [ ] 메모리 누수 해결 확인

---

## 🚨 주의사항

1. **.copy() 남용 주의**: 
   - 너무 자주 사용하면 성능 저하
   - 꼭 필요한 곳(view 참조 제거)에만 사용

2. **gc.collect() 남용 주의**:
   - Pass 단위로만 호출 (배치마다는 비효율)
   - GPU 작업 후에는 필수

3. **del 문 순서**:
   - 참조 관계 고려하여 역순으로 삭제
   - del 후 즉시 gc.collect() 호출

4. **테스트 필수**:
   - 수정 후 반드시 작은 데이터로 먼저 테스트
   - 메모리 모니터링 도구 활용

---

이 가이드를 따라 단계적으로 수정하면 메모리 누수 문제가 해결될 것입니다!
