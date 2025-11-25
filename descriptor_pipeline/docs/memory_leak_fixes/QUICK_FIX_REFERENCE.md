# 빠른 수정 참조 시트 (Quick Fix Reference)

## 파일별 수정사항 한눈에 보기

### 📄 1. `descriptor_pipeline/io/parquet_reader_duckdb.py`

#### 삭제할 부분:
```
라인 226-407: 전체 삭제 (중복 함수 정의)
```

#### 수정할 부분 (총 5곳):
```python
# 라인 103
X = df_batch[columns].values.astype(np.float64)
# →
X = df_batch[columns].values.copy().astype(np.float64)

# 라인 156
X = df_batch[columns].values.astype(np.float64)
# →
X = df_batch[columns].values.copy().astype(np.float64)

# 라인 193
X = df_batch[columns].values.astype(np.float64)
# →
X = df_batch[columns].values.copy().astype(np.float64)

# 라인 356 (중복 함수 내 - 삭제 예정)
# 라인 393 (중복 함수 내 - 삭제 예정)
```

---

### 📄 2. `descriptor_pipeline/core/pipeline.py`

#### 추가할 import:
```python
# 라인 6 근처에 추가
import gc
```

#### 추가할 메서드:
```python
# 라인 62 근처에 추가 (run() 메서드 전)
def _cleanup_memory(self):
    """명시적 메모리 정리"""
    gc.collect()
    if self.using_gpu:
        import torch
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
```

#### 수정할 부분 1: 라인 123-125
```python
# Before
columns_p2, spearman_info, indices_p2 = spearman_pass.process(
    data, columns_p1, G_spearman, self.graph_builder, self.leiden
)

# After
columns_p2, spearman_info, indices_p2 = spearman_pass.process(
    data, columns_p1, G_spearman, stats_p1
)
```

#### 수정할 부분 2: 라인 142-143
```python
# Before
data_p2 = data[:, indices_p2]
G_spearman_p2 = G_spearman[indices_p2][:, indices_p2]

# After
data_p2 = data[:, indices_p2].copy()
G_spearman_p2 = G_spearman[indices_p2][:, indices_p2].copy()
stats_p2 = self._filter_stats_by_indices(stats_p1, indices_p2)

# 추가: 원본 삭제
del data, G_spearman
self._cleanup_memory()
```

#### 수정할 부분 3: 라인 169
```python
# Before
data_p3 = data[:, indices_p3]

# After  
data_p3 = data_p2[:, indices_p3_sub].copy()
stats_p3 = self._filter_stats_by_indices(stats_p1, indices_p3)

# 추가: 이전 데이터 삭제
del data_p2, stats_p1, stats_p2
self._cleanup_memory()
```

#### 수정할 부분 4: 라인 253-264 (_filter_stats_by_indices)
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
            # 추가: 리스트도 필터링
            try:
                stats_filtered[key] = [value[i] for i in indices]
            except:
                stats_filtered[key] = value
        else:
            stats_filtered[key] = value
    
    return stats_filtered
```

#### 수정할 부분 5: 라인 246-251 (_load_data)
```python
def _load_data(self, parquet_paths: List[str], columns: List[str]) -> np.ndarray:
    """Load data into memory with explicit cleanup"""
    batches = []
    batch_generator = None
    
    try:
        batch_generator = iter_batches(parquet_paths, columns, self.config.batch_rows)
        
        for batch_data, offset in batch_generator:
            batches.append(batch_data.copy())  # .copy() 추가
            del batch_data
    
    finally:
        if batch_generator is not None:
            try:
                batch_generator.close()
            except:
                pass
        gc.collect()
    
    result = np.vstack(batches)
    del batches
    gc.collect()
    
    return result
```

#### 추가할 cleanup 호출 (Pass 후):
```python
# 라인 96 근처 (Pass 0 후)
self._cleanup_memory()

# 라인 108 근처 (Pass 1 후)
self._cleanup_memory()

# 라인 134 근처 (Pass 2 후)
self._cleanup_memory()

# 라인 161 근처 (Pass 3 후)
self._cleanup_memory()

# 라인 204 근처 (Pass 4 후)
self._cleanup_memory()
```

---

### 📄 3. `descriptor_pipeline/core/similarity_gpu.py`

#### 수정할 부분 (총 3곳):

```python
# 라인 158 근처
# Before
G_cpu = G.cpu().numpy()
# After
G_cpu = G.detach().cpu().numpy().copy()

# 라인 390 근처
# Before
H_cpu = H.cpu().numpy()
# After
H_cpu = H.detach().cpu().numpy().copy()

# 라인 634 근처
# Before
R_cpu = R.cpu().numpy()
# After
R_cpu = R.detach().cpu().numpy().copy()
```

---

## 🔍 수정 확인 방법

### 1. 라인 번호로 찾기
```bash
# parquet_reader_duckdb.py
sed -n '103p' descriptor_pipeline/io/parquet_reader_duckdb.py
sed -n '156p' descriptor_pipeline/io/parquet_reader_duckdb.py
sed -n '226,407p' descriptor_pipeline/io/parquet_reader_duckdb.py  # 삭제할 부분

# pipeline.py
sed -n '123,125p' descriptor_pipeline/core/pipeline.py
sed -n '142,143p' descriptor_pipeline/core/pipeline.py
sed -n '169p' descriptor_pipeline/core/pipeline.py

# similarity_gpu.py
sed -n '158p' descriptor_pipeline/core/similarity_gpu.py
sed -n '390p' descriptor_pipeline/core/similarity_gpu.py
sed -n '634p' descriptor_pipeline/core/similarity_gpu.py
```

### 2. 패턴으로 찾기
```bash
# .values.astype 패턴 찾기
grep -n "\.values\.astype" descriptor_pipeline/io/parquet_reader_duckdb.py

# process 함수 호출 찾기
grep -n "spearman_pass.process" descriptor_pipeline/core/pipeline.py

# GPU 텐서 변환 찾기
grep -n "\.cpu()\.numpy()" descriptor_pipeline/core/similarity_gpu.py
```

---

## 📝 수정 스크립트 (자동 패치)

### Option 1: sed를 이용한 자동 수정

```bash
#!/bin/bash
# fix_memory_leaks.sh

# 백업 생성
cp descriptor_pipeline/io/parquet_reader_duckdb.py descriptor_pipeline/io/parquet_reader_duckdb.py.backup
cp descriptor_pipeline/core/pipeline.py descriptor_pipeline/core/pipeline.py.backup
cp descriptor_pipeline/core/similarity_gpu.py descriptor_pipeline/core/similarity_gpu.py.backup

# 1. parquet_reader_duckdb.py 수정
sed -i 's/\.values\.astype(np\.float64)/.values.copy().astype(np.float64)/g' \
    descriptor_pipeline/io/parquet_reader_duckdb.py

# 2. similarity_gpu.py 수정
sed -i 's/\.cpu()\.numpy()/.detach().cpu().numpy().copy()/g' \
    descriptor_pipeline/core/similarity_gpu.py

# 3. pipeline.py는 수동 수정 권장 (복잡한 로직)
echo "pipeline.py는 수동 수정이 필요합니다."
echo "IMPLEMENTATION_GUIDE.md를 참조하세요."
```

### Option 2: Python 스크립트로 수정

```python
# fix_memory_leaks.py
import re
from pathlib import Path

def fix_parquet_reader():
    """parquet_reader_duckdb.py 수정"""
    file_path = Path("descriptor_pipeline/io/parquet_reader_duckdb.py")
    content = file_path.read_text()
    
    # .copy() 추가
    content = content.replace(
        ".values.astype(np.float64)",
        ".values.copy().astype(np.float64)"
    )
    
    # 중복 함수 제거 (라인 226-407)
    lines = content.split('\n')
    # 라인 226 찾기
    for i, line in enumerate(lines):
        if i >= 225 and 'def iter_batches_duckdb' in line:
            # 두 번째 정의 찾음
            lines = lines[:i]  # 이후 제거
            break
    
    content = '\n'.join(lines)
    file_path.write_text(content)
    print(f"✓ Fixed: {file_path}")

def fix_similarity_gpu():
    """similarity_gpu.py 수정"""
    file_path = Path("descriptor_pipeline/core/similarity_gpu.py")
    content = file_path.read_text()
    
    # GPU 텐서 변환 수정
    content = content.replace(
        ".cpu().numpy()",
        ".detach().cpu().numpy().copy()"
    )
    
    file_path.write_text(content)
    print(f"✓ Fixed: {file_path}")

if __name__ == "__main__":
    print("메모리 누수 자동 수정 시작...")
    
    # 백업
    import shutil
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    shutil.copytree(
        "descriptor_pipeline",
        f"descriptor_pipeline_backup_{timestamp}",
        dirs_exist_ok=True
    )
    print(f"✓ Backup created: descriptor_pipeline_backup_{timestamp}")
    
    # 수정 적용
    fix_parquet_reader()
    fix_similarity_gpu()
    
    print("\n⚠️  pipeline.py는 수동 수정이 필요합니다.")
    print("IMPLEMENTATION_GUIDE.md를 참조하세요.")
```

---

## 🧪 수정 후 테스트

```python
# test_memory_fix.py
import tracemalloc
import gc
from descriptor_pipeline.core.pipeline import DescriptorPipeline
from descriptor_pipeline.config.settings import PipelineConfig

def test_memory():
    """메모리 누수 테스트"""
    tracemalloc.start()
    
    config = PipelineConfig(
        parquet_glob="data/test_*.parquet",
        output_dir="output/test",
        checkpoint=True,
        verbose=True
    )
    
    pipeline = DescriptorPipeline(config)
    
    # 실행 전 스냅샷
    gc.collect()
    snapshot1 = tracemalloc.take_snapshot()
    
    # 파이프라인 실행
    results = pipeline.run()
    
    # 실행 후 스냅샷
    gc.collect()
    snapshot2 = tracemalloc.take_snapshot()
    
    # 메모리 증가 분석
    top_stats = snapshot2.compare_to(snapshot1, 'lineno')
    
    print("\n" + "="*70)
    print("Top 10 Memory Increases:")
    print("="*70)
    for stat in top_stats[:10]:
        print(stat)
    
    tracemalloc.stop()
    
    return results

if __name__ == "__main__":
    results = test_memory()
    print(f"\n✅ Pipeline completed: {results['final_count']} descriptors")
```

---

## 📊 수정 완료 체크리스트

### 파일별 확인
- [ ] `parquet_reader_duckdb.py`
  - [ ] 라인 103: .copy() 추가
  - [ ] 라인 156: .copy() 추가
  - [ ] 라인 193: .copy() 추가
  - [ ] 라인 226-407: 삭제
  
- [ ] `pipeline.py`
  - [ ] import gc 추가
  - [ ] _cleanup_memory() 메서드 추가
  - [ ] 라인 123-125: 함수 호출 수정
  - [ ] 라인 142-143: .copy() 추가 + 원본 삭제
  - [ ] 라인 169: .copy() 추가 + 원본 삭제
  - [ ] _filter_stats_by_indices: .copy() 추가
  - [ ] _load_data: 개선
  - [ ] 5곳에 _cleanup_memory() 호출 추가
  
- [ ] `similarity_gpu.py`
  - [ ] 라인 158: .detach().cpu().numpy().copy()
  - [ ] 라인 390: .detach().cpu().numpy().copy()
  - [ ] 라인 634: .detach().cpu().numpy().copy()

### 테스트 확인
- [ ] 코드 백업 완료
- [ ] 수정 적용 완료
- [ ] 작은 데이터셋 테스트 통과
- [ ] 메모리 프로파일링 정상
- [ ] 전체 데이터셋 테스트 통과

---

## 🔗 관련 문서
- `MEMORY_LEAK_COMPREHENSIVE_DIAGNOSIS.md`: 상세 진단
- `IMPLEMENTATION_GUIDE.md`: 단계별 가이드
- `parquet_reader_duckdb_FIXED.py`: 수정된 파일 예시
- `pipeline_FIXED.py`: 수정된 파일 예시
