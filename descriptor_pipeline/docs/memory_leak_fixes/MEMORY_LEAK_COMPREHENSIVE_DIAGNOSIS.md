# 메모리 누수 종합 진단 및 해결 방안

## 🔍 발견된 문제들

### 1. **중복 함수 정의 (parquet_reader_duckdb.py)**

**위치**: `descriptor_pipeline/io/parquet_reader_duckdb.py`

**문제**:
```python
# 라인 26-208: 첫 번째 iter_batches_duckdb 정의
def iter_batches_duckdb(...):
    ...

# 라인 226-407: 두 번째 iter_batches_duckdb 정의 (완전히 동일)
def iter_batches_duckdb(...):
    ...
```

**영향**: 
- Python은 마지막 정의만 사용하므로 첫 번째 정의는 무시됨
- 코드 혼란과 유지보수 문제
- 직접적인 메모리 누수 원인은 아니지만 코드 품질 저하

**해결**:
```python
# 라인 226-407의 중복 정의 삭제 필요
```

---

### 2. **파이프라인 함수 호출 인자 불일치 (pipeline.py)**

**위치**: `descriptor_pipeline/core/pipeline.py` 라인 121-125

**문제**:
```python
# pipeline.py에서
spearman_pass.process(
    data, columns_p1, G_spearman, self.graph_builder, self.leiden  # 5개 인자
)

# advanced_filtering.py에서
def process(self, data: np.ndarray, columns: List[str],
           G_spearman: np.ndarray, stats: Dict):  # 4개 인자만 받음
```

**영향**:
- TypeError 발생 예상: `process() takes 5 positional arguments but 6 were given`
- 파이프라인 실행 실패

**해결**:
```python
# pipeline.py 라인 123-124 수정
columns_p2, spearman_info, indices_p2 = spearman_pass.process(
    data, columns_p1, G_spearman, stats_p1  # stats_p1으로 수정
)
```

---

### 3. **DuckDB 연결 재사용 누수**

**위치**: `descriptor_pipeline/io/parquet_reader_duckdb.py` 라인 54

**문제**:
```python
def iter_batches_duckdb(...):
    global_offset = 0
    
    # 매번 새 연결 생성
    conn = duckdb.connect(':memory:')
```

**메모리 누수 시나리오**:
- 이 함수가 generator로서 yield 중간에 중단되면 `finally` 블록이 실행되지 않을 수 있음
- DuckDB 연결이 제대로 닫히지 않으면 메모리에 누적
- 반복 호출 시 연결 객체가 쌓임

**해결**:
```python
def iter_batches_duckdb(...):
    conn = None
    try:
        conn = duckdb.connect(':memory:')
        # ... 기존 코드
        
        # 각 yield 후 명시적 캐시 정리
        for ...:
            yield X, global_offset
            
            # 즉시 정리
            del X
            gc.collect()
            
    finally:
        if conn is not None:
            try:
                conn.close()
            except:
                pass
        gc.collect()
```

---

### 4. **DataFrame 변환 시 메모리 누수**

**위치**: `parquet_reader_duckdb.py` 여러 위치

**문제**:
```python
df_batch = conn.execute(batch_query).fetch_df()  # pandas DataFrame 생성

# NumPy로 변환
X = df_batch[columns].values.astype(np.float64)
del df_batch  # 삭제하지만 pandas의 내부 캐시는 남을 수 있음
```

**메모리 누수 원인**:
- pandas DataFrame은 내부적으로 BlockManager를 사용
- `.values`는 view를 반환할 수 있어 원본 참조 유지
- `del df_batch`로 삭제해도 NumPy array가 pandas 메모리를 참조할 수 있음

**해결**:
```python
df_batch = conn.execute(batch_query).fetch_df()

# 명시적 복사로 독립성 보장
X = df_batch[columns].values.copy().astype(np.float64)

# pandas 객체 완전 삭제
del df_batch
gc.collect()
```

---

### 5. **GPU 텐서 누수**

**위치**: `descriptor_pipeline/core/similarity_gpu.py`

**문제**:
```python
@torch.no_grad()
def _compute_correlation_matrix_gpu(self, X: np.ndarray):
    X_gpu = torch.from_numpy(X).to(self.device, dtype=torch.float32)
    
    # ... 계산
    
    G_cpu = G.cpu().numpy()
    del X_gpu, G
    torch.cuda.empty_cache()
```

**메모리 누수 원인**:
- `torch.no_grad()` 컨텍스트만으로는 불충분
- 중간 텐서들이 계산 그래프에 남을 수 있음
- `.cpu().numpy()` 변환 시 GPU 메모리 참조 유지 가능

**해결**:
```python
@torch.no_grad()
def _compute_correlation_matrix_gpu(self, X: np.ndarray):
    try:
        X_gpu = torch.from_numpy(X).to(self.device, dtype=torch.float32)
        
        # 계산 (중간 변수 즉시 삭제)
        # ...
        
        # GPU -> CPU 변환 (명시적 복사)
        G_cpu = G.detach().cpu().numpy().copy()
        
    finally:
        # 모든 GPU 텐서 삭제
        if 'X_gpu' in locals():
            del X_gpu
        if 'G' in locals():
            del G
        
        # GPU 메모리 강제 해제
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, 'reset_peak_memory_stats'):
            torch.cuda.reset_peak_memory_stats()
```

---

### 6. **NumPy 배열 view 참조 누수**

**위치**: 여러 파일에서 배열 슬라이싱

**문제**:
```python
# pipeline.py
data_p2 = data[:, indices_p2]  # view 생성, 원본 data 참조 유지
data_p3 = data[:, indices_p3]  # 또 다른 view

# 원본 data가 해제되지 않음
```

**메모리 누수 원인**:
- NumPy slicing은 기본적으로 view 반환
- View는 원본 배열 전체를 메모리에 유지
- Pass마다 새로운 view 생성 → 원본 메모리 계속 점유

**해결**:
```python
# 명시적 복사로 독립성 확보
data_p2 = data[:, indices_p2].copy()

# 원본 명시적 삭제
del data
gc.collect()

# 다음 pass
data_p3 = data_p2[:, indices_p3_sub].copy()
del data_p2
gc.collect()
```

---

### 7. **통계 정보 딕셔너리 누적**

**위치**: `pass1_statistics.py` 및 pipeline에서 stats 전달

**문제**:
```python
# stats에 대량의 데이터 저장
stats = {
    'means': np.array(...),      # (p,)
    'stds': np.array(...),        # (p,)
    'cdf_lookups': [...],         # p개의 리스트
    'missing_rates': ...,
    # ... 계속 추가
}

# 매 pass마다 복사/전달되지만 이전 stats는 삭제 안됨
```

**메모리 누수 원인**:
- CDF lookup 테이블이 각 descriptor마다 저장 (메모리 집약적)
- Pass마다 stats를 필터링하지만 이전 버전이 남아있을 수 있음
- 딕셔너리 내부의 NumPy 배열들이 참조 유지

**해결**:
```python
def _filter_stats_by_indices(self, stats: Dict, indices: np.ndarray) -> Dict:
    """Filter statistics by indices with explicit memory management"""
    stats_filtered = {}
    
    for key, value in stats.items():
        if isinstance(value, np.ndarray) and value.ndim >= 1:
            try:
                # 명시적 복사
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
    
    # 이전 stats 명시적 삭제
    return stats_filtered
```

---

### 8. **iter_batches 순환 참조**

**위치**: `descriptor_pipeline/io/__init__.py` 및 사용처

**문제**:
```python
# iter_batches가 generator로 구현되어 있으면
for batch_data, offset in iter_batches(...):
    batches.append(batch_data)
    # batch_data가 리스트에 누적되면서
    # generator 내부 상태도 유지됨
```

**메모리 누수 원인**:
- Generator가 완전히 소진되지 않으면 내부 상태 유지
- Generator 내부의 DuckDB 연결이나 중간 변수들이 메모리에 남음
- Exception 발생 시 generator cleanup 안됨

**해결**:
```python
# iter_batches 사용 시 명시적 cleanup
def _load_data(self, parquet_paths: List[str], columns: List[str]) -> np.ndarray:
    """Load data into memory with explicit cleanup"""
    batches = []
    batch_generator = None
    
    try:
        batch_generator = iter_batches(parquet_paths, columns, self.config.batch_rows)
        
        for batch_data, offset in batch_generator:
            # 명시적 복사
            batches.append(batch_data.copy())
            del batch_data
            
    finally:
        # Generator 명시적 종료
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

## 🔧 추가 메모리 관리 권장사항

### 9. **Pass 간 명시적 메모리 정리**

**위치**: `pipeline.py`의 각 pass 사이

**추가 필요**:
```python
# Pass 1 완료 후
columns_p1, stats_p1, indices_p1 = self.pass1.compute(...)

# 명시적 정리
import gc
gc.collect()

if self.using_gpu:
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

# Pass 2 시작
```

---

### 10. **Context Manager 사용**

**위치**: GPU 연산이 있는 모든 함수

**추가 필요**:
```python
class GPUContext:
    """GPU 메모리 안전 컨텍스트"""
    def __init__(self, device):
        self.device = device
    
    def __enter__(self):
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()

# 사용
with GPUContext(self.device):
    G_spearman = spearman_gpu.compute(...)
```

---

## 📋 Import 및 변수 호출 문제 체크리스트

### ✅ 확인된 Import 문제:

1. **pipeline.py 라인 22**: 
```python
from descriptor_pipeline.io import iter_batches
```
→ `parquet_reader_duckdb.py`의 `iter_batches_duckdb` 사용해야 함
→ 또는 `__init__.py`에서 제대로 export 되는지 확인 필요

2. **advanced_filtering.py**:
```python
# 현재는 문제 없음 - 모든 import 정상
```

3. **similarity_gpu.py**:
```python
# 체크 필요: memory_leak_patch가 실제로 import되는지
# pipeline.py나 __main__.py에서 명시적 import 필요
```

---

## 🚀 즉시 적용 가능한 수정사항

### 우선순위 1 (Critical - 즉시 수정):
1. `pipeline.py` 라인 123-124: 함수 호출 인자 수정
2. `parquet_reader_duckdb.py`: 중복 함수 정의 제거 (라인 226-407 삭제)
3. `parquet_reader_duckdb.py`: `.copy()` 추가하여 view 참조 방지

### 우선순위 2 (Important - 빠른 시일 내 수정):
4. `pipeline.py`: NumPy view → copy로 변경
5. `similarity_gpu.py`: `.detach().cpu().numpy().copy()` 패턴 적용
6. Pass 간 명시적 `gc.collect()` 추가

### 우선순위 3 (Recommended - 점진적 개선):
7. Context Manager 추가
8. Generator cleanup 개선
9. 통계 정보 필터링 개선

---

## 🧪 메모리 누수 검증 방법

```python
import tracemalloc
import gc

# 테스트 시작
tracemalloc.start()
gc.collect()

snapshot1 = tracemalloc.take_snapshot()

# 파이프라인 실행
pipeline.run()

gc.collect()
snapshot2 = tracemalloc.take_snapshot()

# 메모리 증가 분석
top_stats = snapshot2.compare_to(snapshot1, 'lineno')

print("\n[ Top 10 Memory Increases ]")
for stat in top_stats[:10]:
    print(stat)
```

---

## 결론

**주요 메모리 누수 원인 요약**:
1. NumPy array view 참조 (가장 큰 원인)
2. DuckDB DataFrame → NumPy 변환 시 참조 유지
3. GPU 텐서 불완전한 해제
4. Generator 내부 상태 유지
5. 통계 정보 딕셔너리 누적

**해결 핵심**:
- 모든 배열 변환에 `.copy()` 추가
- 각 pass 후 명시적 `del` + `gc.collect()`
- GPU 사용 후 `.detach()` 추가
- Context Manager로 자동 정리
