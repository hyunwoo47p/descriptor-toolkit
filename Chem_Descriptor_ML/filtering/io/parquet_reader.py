"""
Parquet 파일 스트리밍 I/O
"""

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from typing import List, Iterator, Tuple, Dict, Optional
from pathlib import Path
from Chem_Descriptor_ML.filtering.utils.logging import log


# Parquet writer 설정
ROW_GROUP_SIZE = 5000  # 메모리 안정성을 위한 row group 크기


def generate_sample_indices_per_file(
    parquet_paths: List[str],
    sample_per_file: int,
    random_seed: int,
    output_path: Optional[Path] = None,
    verbose: bool = False,
    file_independent: bool = False
) -> Dict[str, np.ndarray]:
    """
    각 파일마다 랜덤 샘플 인덱스 생성 및 저장
    
    Args:
        parquet_paths: Parquet 파일 경로 리스트
        sample_per_file: 파일당 샘플링할 행 수
        random_seed: 랜덤 시드
        output_path: 인덱스를 저장할 경로 (None이면 저장 안 함)
        verbose: 로깅 출력 여부
        file_independent: True면 파일별 독립 RNG (증분 업데이트 시 기존 파일 샘플 유지)
        
    Returns:
        Dict[str, np.ndarray]: {파일경로: 샘플 인덱스 배열}
    
    Note:
        file_independent=True:
            - 파일 추가/제거 시 기존 파일의 샘플 불변
            - NumPy SeedSequence.spawn 사용 (권장 방식)
        file_independent=False (기본):
            - 기존 동작 유지 (빠르지만 파일 순서 의존)
    """
    sample_indices = {}
    
    total_sampled = 0
    total_available = 0
    files_fully_used = 0
    files_sampled = 0
    
    # 정렬된 파일 목록
    sorted_paths = sorted(parquet_paths)
    
    if verbose:
        sampling_mode = "file-independent" if file_independent else "sequential"
        log(f"  Generating sample indices (seed={random_seed}, mode={sampling_mode})...")
    
    # RNG 초기화 방식 선택
    if file_independent:
        # 파일별 독립 RNG (NumPy 권장 방식)
        from numpy.random import SeedSequence, default_rng
        
        base_seq = SeedSequence(random_seed)
        file_seeds = base_seq.spawn(len(sorted_paths))
        rngs = [default_rng(seed) for seed in file_seeds]
    else:
        # 기존 방식: 하나의 RNG (빠르지만 파일 순서 의존)
        base_rng = np.random.RandomState(random_seed)
        rngs = [base_rng] * len(sorted_paths)
    
    for file_idx, (path, rng) in enumerate(zip(sorted_paths, rngs)):
        try:
            parquet_file = pq.ParquetFile(path)
            file_rows = parquet_file.metadata.num_rows
        except:
            file_dataset = ds.dataset(path, format="parquet")
            file_rows = file_dataset.count_rows()
        
        total_available += file_rows
        
        if file_rows <= sample_per_file:
            # 파일이 작으면 전체 사용
            indices = np.arange(file_rows, dtype=np.int64)
            total_sampled += file_rows
            files_fully_used += 1
        else:
            # 랜덤 샘플링
            indices = rng.choice(file_rows, size=sample_per_file, replace=False)
            indices = np.sort(indices)  # 정렬해서 효율적 접근
            indices = np.unique(indices)  # 중복 제거 (안전장치)
            total_sampled += len(indices)
            files_sampled += 1
        
        sample_indices[path] = indices
        
        if verbose and (file_idx + 1) % 50 == 0:
            log(f"    Processed {file_idx + 1}/{len(sorted_paths)} files")
    
    # 통계 출력
    if verbose:
        log(f"  Sample indices generation complete:")
        log(f"    Total available rows: {total_available:,}")
        log(f"    Total selected samples: {total_sampled:,}")
        log(f"    Average per file: {total_sampled / len(sorted_paths):.0f}")
        log(f"    Sampling ratio: {100.0 * total_sampled / max(total_available, 1):.2f}%")
        log(f"    Files fully used: {files_fully_used}")
        log(f"    Files sampled: {files_sampled}")
    
    # 저장
    if output_path is not None:
        np.save(output_path, sample_indices, allow_pickle=True)
        if verbose:
            log(f"  Saved sample indices to: {output_path}")
    
    return sample_indices


def load_sample_indices(index_path: Path, verbose: bool = False) -> Dict[str, np.ndarray]:
    """
    저장된 샘플 인덱스 로드
    
    Args:
        index_path: 인덱스 파일 경로
        verbose: 로깅 출력 여부
        
    Returns:
        Dict[str, np.ndarray]: {파일경로: 샘플 인덱스 배열}
    """
    if verbose:
        log(f"  Loading sample indices from: {index_path}")
    
    sample_indices = np.load(index_path, allow_pickle=True).item()
    
    if verbose:
        total_rows = sum(len(v) for v in sample_indices.values())
        log(f"  Loaded {len(sample_indices)} files, {total_rows:,} total rows")
    
    return sample_indices


def get_total_rows(parquet_paths: List[str], sample_per_file: int = None) -> int:
    """
    Parquet 파일들의 총 행 수를 계산 (샘플링 고려)
    
    Args:
        parquet_paths: Parquet 파일 경로 리스트
        sample_per_file: 파일당 샘플링할 행 수 (None이면 전체)
        
    Returns:
        총 행 수 (샘플링을 고려한)
    """
    total = 0
    
    if sample_per_file is None:
        # 샘플링 없음: 전체 행 수
        try:
            # 통합 데이터셋으로 시도
            dataset = ds.dataset(parquet_paths, format="parquet")
            total = dataset.count_rows()
        except:
            # 실패 시 파일별로 계산
            for path in parquet_paths:
                try:
                    parquet_file = pq.ParquetFile(path)
                    total += parquet_file.metadata.num_rows
                except:
                    # 메타데이터 접근 실패 시 스캔
                    file_dataset = ds.dataset(path, format="parquet")
                    total += file_dataset.count_rows()
    else:
        # 샘플링 있음: 각 파일에서 min(파일크기, sample_per_file)
        for path in parquet_paths:
            try:
                parquet_file = pq.ParquetFile(path)
                file_rows = parquet_file.metadata.num_rows
            except:
                file_dataset = ds.dataset(path, format="parquet")
                file_rows = file_dataset.count_rows()
            
            total += min(file_rows, sample_per_file)
    
    return total


def iter_batches(parquet_paths: List[str], 
                columns: List[str], 
                batch_rows: int, 
                verbose: bool = False,
                sample_per_file: int = None,
                random_seed: int = None) -> Iterator[Tuple[np.ndarray, int]]:
    """
    Memory-efficient streaming with schema unification and optional random sampling.
    
    Args:
        parquet_paths: Parquet 파일 경로 리스트
        columns: 읽을 컬럼 이름 리스트
        batch_rows: 배치 크기
        verbose: 로깅 출력 여부
        sample_per_file: 각 파일당 샘플링할 행 수 (None이면 전체, 파일 크기보다 크면 전체)
        random_seed: 랜덤 샘플링 시드 (재현성 보장)
        
    Yields:
        (X, offset): 데이터 배치와 글로벌 오프셋
        
    Note:
        - sample_per_file이 지정되면 각 파일에서 랜덤 샘플링
        - 파일 크기 < sample_per_file인 경우 전체 사용
        - 샘플링 시 file-by-file 방식 강제 사용 (통합 Dataset 불가능)
    """
    global_offset = 0
    
    # 샘플링이 지정되면 file-by-file 방식 강제
    use_unified = (sample_per_file is None)
    
    # 단일 Dataset 우선 시도 (샘플링 없을 때만)
    if use_unified:
        try:
            dataset = ds.dataset(parquet_paths, format="parquet")
            
            if verbose:
                log(f"  Unified dataset schema detected ({len(dataset.schema.names)} total columns), selecting {len(columns)} for processing")
            
            scanner = dataset.scanner(
                columns=columns,
                batch_size=batch_rows
            )
            
            for batch in scanner.to_batches():
                cols_np = [np.array(batch.column(i), dtype=np.float64) 
                          for i in range(len(columns))]
                X = np.column_stack(cols_np)
                
                yield X, global_offset
                global_offset += len(X)
            
            return  # 성공 시 여기서 종료
            
        except Exception as e:
            # Fallback으로 계속 진행
            log(f"⚠️ Warning: Could not create unified dataset: {e}", verbose)
            log("  Falling back to file-by-file scanning", verbose)
    
    # File-by-file 스캔 (샘플링 있거나 통합 Dataset 실패 시)
    if sample_per_file is not None and verbose:
        log(f"  Random sampling mode: {sample_per_file:,} rows per file (seed={random_seed})")
    
    # Random seed 설정
    if random_seed is not None:
        rng = np.random.RandomState(random_seed)
    else:
        rng = np.random.RandomState()
    
    total_sampled = 0
    total_available = 0
    files_fully_used = 0
    files_sampled = 0
    
    for file_idx, path in enumerate(sorted(parquet_paths)):
        if verbose and (file_idx + 1) % 50 == 0:
            log(f"  Processing file {file_idx + 1}/{len(parquet_paths)}")
        
        file_dataset = ds.dataset(path, format="parquet")
        available_cols = set(file_dataset.schema.names)
        missing_cols = set(columns) - available_cols
        
        if missing_cols and verbose:
            log(f"  ⚠️ File {file_idx}: missing {len(missing_cols)} columns")
        
        cols_to_scan = [c for c in columns if c in available_cols]
        
        # 파일 크기 확인
        try:
            file_rows = file_dataset.count_rows()
        except:
            # 메타데이터 접근 실패 시 전체 읽기
            file_rows = None
        
        # 샘플링 결정
        if sample_per_file is not None and file_rows is not None:
            total_available += file_rows
            
            if file_rows <= sample_per_file:
                # 파일이 작으면 전체 사용
                sample_indices = None
                total_sampled += file_rows
                files_fully_used += 1
            else:
                # 랜덤 샘플링
                sample_indices = rng.choice(file_rows, size=sample_per_file, replace=False)
                sample_indices = np.sort(sample_indices)  # 정렬해서 효율적 접근
                total_sampled += sample_per_file
                files_sampled += 1
        else:
            sample_indices = None
            if file_rows is not None:
                total_available += file_rows
                total_sampled += file_rows
                files_fully_used += 1
        
        # 데이터 읽기
        if sample_indices is None:
            # 전체 읽기 (배치 단위)
            scanner = file_dataset.scanner(
                columns=cols_to_scan,
                batch_size=batch_rows
            )
            
            for batch in scanner.to_batches():
                # 누락 컬럼은 NaN으로 채움
                X = np.full((len(batch), len(columns)), np.nan, dtype=np.float64)
                
                for i, col_name in enumerate(columns):
                    if col_name in cols_to_scan:
                        col_idx = cols_to_scan.index(col_name)
                        X[:, i] = np.array(batch.column(col_idx), dtype=np.float64)
                
                yield X, global_offset
                global_offset += len(X)
        else:
            # 샘플링된 인덱스만 읽기
            # PyArrow take를 사용하여 효율적으로 샘플링
            table = file_dataset.to_table(columns=cols_to_scan)
            sampled_table = table.take(sample_indices)
            
            # 배치 단위로 나눠서 yield
            n_rows = len(sampled_table)
            for start_idx in range(0, n_rows, batch_rows):
                end_idx = min(start_idx + batch_rows, n_rows)
                batch_slice = sampled_table.slice(start_idx, end_idx - start_idx)
                
                # 누락 컬럼은 NaN으로 채움
                X = np.full((len(batch_slice), len(columns)), np.nan, dtype=np.float64)
                
                for i, col_name in enumerate(columns):
                    if col_name in cols_to_scan:
                        col_idx = cols_to_scan.index(col_name)
                        X[:, i] = np.array(batch_slice.column(col_idx), dtype=np.float64)
                
                yield X, global_offset
                global_offset += len(X)
    
    # 샘플링 통계 출력
    if sample_per_file is not None and verbose:
        log(f"  Sampling statistics:")
        log(f"    Total available rows: {total_available:,}")
        log(f"    Total selected samples: {total_sampled:,}")
        log(f"    Average per file: {total_sampled / len(parquet_paths):.0f}")
        log(f"    Sampling ratio: {100.0 * total_sampled / max(total_available, 1):.2f}%")
        log(f"    Files fully used: {files_fully_used}")
        log(f"    Files sampled: {files_sampled}")


def create_sampled_parquet_file(
    parquet_paths: List[str],
    columns: List[str],
    sample_per_file: int,
    output_path: Path,
    random_seed: int,
    verbose: bool = False,
    file_independent: bool = False,
    batch_flush_size: int = 50
) -> str:
    """
    메모리 효율적인 샘플 파일 생성 (스트리밍 방식)
    
    Args:
        parquet_paths: 원본 Parquet 파일 경로 리스트
        columns: 저장할 컬럼 리스트
        sample_per_file: 파일당 샘플링할 행 수
        output_path: 출력 파일 경로
        random_seed: 랜덤 시드
        verbose: 로깅 출력 여부
        file_independent: True면 파일별 독립 RNG
        batch_flush_size: 메모리 절약을 위한 배치 크기 (기본: 50)
        
    Returns:
        str: 생성된 파일 경로
        
    Memory Optimization:
        - 스트리밍 방식: 테이블을 메모리에 누적하지 않고 즉시 쓰기
        - Row group 기반 샘플링: 전체 파일을 읽지 않음
        - 배치 플러시: 일정 개수마다 메모리 해제
    """
    import time
    import gc
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        sampling_mode = "file-independent" if file_independent else "sequential"
        log(f"  Creating sampled parquet (memory-efficient)")
        log(f"  Seed: {random_seed}, Mode: {sampling_mode}, Batch: {batch_flush_size}")
        log(f"  Output: {output_path}")
    
    # 정렬된 파일 목록
    sorted_paths = sorted(parquet_paths)
    
    # RNG 초기화
    if file_independent:
        from numpy.random import SeedSequence, default_rng
        base_seq = SeedSequence(random_seed)
        file_seeds = base_seq.spawn(len(sorted_paths))
        rngs = [default_rng(seed) for seed in file_seeds]
    else:
        base_rng = np.random.RandomState(random_seed)
        rngs = [base_rng] * len(sorted_paths)
    
    start_time = time.time()
    total_sampled = 0
    total_available = 0
    files_processed = 0
    
    # 첫 번째 유효한 파일에서 스키마 추출
    schema = None
    for path in sorted_paths[:10]:
        try:
            pf = pq.ParquetFile(path, memory_map=True)
            if columns:
                schema = pf.schema_arrow.select(columns)
            else:
                schema = pf.schema_arrow
            break
        except:
            continue
    
    if schema is None:
        raise ValueError("Could not extract schema from any input file")
    
    if verbose:
        log(f"    Starting sampling from {len(sorted_paths)} files...")
        log(f"    Target: {sample_per_file} rows per file")
    
    # ParquetWriter로 스트리밍 쓰기
    writer = pq.ParquetWriter(output_path, schema, compression='snappy')
    
    # 배치 버퍼 (메모리 제어용)
    batch_buffer = []
    
    try:
        for file_idx, (path, rng) in enumerate(zip(sorted_paths, rngs)):
            try:
                parquet_file = pq.ParquetFile(path, memory_map=True)
                file_rows = parquet_file.metadata.num_rows
                total_available += file_rows
                
                if file_rows <= sample_per_file:
                    # 작은 파일은 전체 사용
                    if columns:
                        table = parquet_file.read(columns=columns)
                    else:
                        table = parquet_file.read()
                    batch_buffer.append(table)
                    total_sampled += file_rows
                else:
                    # Row group 기반 효율적 샘플링
                    indices = rng.choice(file_rows, size=sample_per_file, replace=False)
                    indices = np.sort(indices)
                    
                    sampled_table = _sample_from_row_groups(
                        parquet_file, indices, columns
                    )
                    batch_buffer.append(sampled_table)
                    total_sampled += len(sampled_table)
                
                files_processed += 1
                
                # 첫 파일 처리 로그
                if verbose and file_idx == 0:
                    sampled_rows = min(file_rows, sample_per_file)
                    log(f"    [1/{len(sorted_paths)}] First file processed: {Path(path).name}")
                    log(f"      File has {file_rows:,} rows, sampled {sampled_rows:,} rows")
                
                # 배치 플러시 (메모리 절약)
                if len(batch_buffer) >= batch_flush_size:
                    for table in batch_buffer:
                        writer.write_table(table, row_group_size=ROW_GROUP_SIZE)
                    batch_buffer = []
                    gc.collect()
                
                # 진행 상황 로그 (20개마다 또는 마지막)
                if verbose and ((file_idx + 1) % 20 == 0 or file_idx == len(sorted_paths) - 1):
                    elapsed = time.time() - start_time
                    rate = files_processed / elapsed if elapsed > 0 else 0
                    eta = (len(sorted_paths) - file_idx - 1) / rate if rate > 0 else 0
                    log(f"    [{file_idx + 1}/{len(sorted_paths)}] "
                        f"{total_sampled:,} rows | {rate:.1f} files/s | ETA: {eta/60:.1f}min")
            
            except Exception as e:
                if verbose:
                    log(f"  ⚠️ Skip {path}: {str(e)[:50]}")
                continue
        
        # 남은 배치 쓰기
        if batch_buffer:
            for table in batch_buffer:
                writer.write_table(table, row_group_size=ROW_GROUP_SIZE)
            batch_buffer = []
    
    finally:
        writer.close()
        gc.collect()
    
    if verbose:
        total_elapsed = time.time() - start_time
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        avg_per_file = total_sampled / files_processed if files_processed > 0 else 0
        log(f"\n  ✓ Sampling completed:")
        log(f"    Total files processed: {files_processed}/{len(sorted_paths)}")
        log(f"    Total rows sampled: {total_sampled:,}")
        log(f"    Average per file: {avg_per_file:.0f} rows")
        log(f"    Output size: {file_size_mb:.1f} MB")
        log(f"    Time elapsed: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")
        log(f"    Processing rate: {files_processed/total_elapsed:.1f} files/s")
    
    return str(output_path)


def _sample_from_row_groups(
    parquet_file: pq.ParquetFile,
    indices: np.ndarray,
    columns: Optional[List[str]] = None
) -> pa.Table:
    """
    Row group 단위로 효율적 샘플링
    
    전체 파일을 읽지 않고 필요한 row group만 읽어 메모리 효율성 향상
    """
    num_row_groups = parquet_file.num_row_groups
    
    # 각 row group의 offset 계산
    rg_offsets = [0]
    for i in range(num_row_groups):
        rg_offsets.append(rg_offsets[-1] + parquet_file.metadata.row_group(i).num_rows)
    
    sampled_batches = []
    
    for rg_idx in range(num_row_groups):
        rg_start = rg_offsets[rg_idx]
        rg_end = rg_offsets[rg_idx + 1]
        
        # 이 row group에 속하는 인덱스
        mask = (indices >= rg_start) & (indices < rg_end)
        if not mask.any():
            continue
        
        # Row group 읽기
        if columns:
            rg_table = parquet_file.read_row_group(rg_idx, columns=columns)
        else:
            rg_table = parquet_file.read_row_group(rg_idx)
        
        # 상대 인덱스로 변환
        relative_indices = indices[mask] - rg_start
        sampled = rg_table.take(relative_indices)
        sampled_batches.append(sampled)
    
    if len(sampled_batches) == 0:
        # 빈 테이블
        if columns:
            schema = parquet_file.schema_arrow.select(columns)
        else:
            schema = parquet_file.schema_arrow
        return pa.table({col.name: [] for col in schema})
    
    return pa.concat_tables(sampled_batches)



def iter_sampled_batches(
    parquet_paths: List[str],
    columns: List[str],
    sample_indices: Dict[str, np.ndarray],
    batch_rows: int,
    verbose: bool = False
) -> Iterator[Tuple[np.ndarray, int]]:
    """
    저장된 샘플 인덱스에 해당하는 행만 읽어서 배치로 반환
    
    최적화 전략:
    - PyArrow의 read_table + take 조합 (컬럼 스토리지 활용)
    - 필요한 컬럼만 읽기
    - 배치 단위로 yield
    
    Args:
        parquet_paths: Parquet 파일 경로 리스트
        columns: 읽을 컬럼 이름 리스트
        sample_indices: {파일경로: 샘플 인덱스 배열} 딕셔너리
        batch_rows: 배치 크기
        verbose: 로깅 출력 여부
        
    Yields:
        (X, offset): 데이터 배치와 글로벌 오프셋
    """
    import time
    
    global_offset = 0
    
    if verbose:
        total_samples = sum(len(indices) for indices in sample_indices.values())
        log(f"  Reading sampled data: {total_samples:,} total rows from {len(sample_indices)} files")
    
    start_time = time.time()
    files_processed = 0
    
    for file_idx, path in enumerate(sorted(parquet_paths)):
        # 이 파일의 샘플 인덱스 가져오기
        indices = sample_indices.get(path)
        if indices is None or len(indices) == 0:
            continue
        
        file_start = time.time()
        
        try:
            # PyArrow ParquetFile 사용 (memory_map으로 효율적 읽기)
            # memory_map=True: 전체를 메모리에 로드하지 않고 필요한 부분만 읽음
            parquet_file = pq.ParquetFile(path, memory_map=True)
            
            # 스키마 확인
            available_cols = set(parquet_file.schema.names)
            missing_cols = set(columns) - available_cols
            
            if missing_cols and verbose and file_idx == 0:
                log(f"  ⚠️ Some files missing {len(missing_cols)} columns (will be filled with NaN)")
            
            cols_to_scan = [c for c in columns if c in available_cols]
            
            # 필요한 컬럼만 읽고 샘플 추출
            # Note: read()는 내부적으로 row group을 효율적으로 처리
            table = parquet_file.read(columns=cols_to_scan)
            sampled_table = table.take(indices)
            
            # 배치 단위로 나눠서 yield
            n_rows = len(sampled_table)
            for start_idx in range(0, n_rows, batch_rows):
                end_idx = min(start_idx + batch_rows, n_rows)
                batch_slice = sampled_table.slice(start_idx, end_idx - start_idx)
                
                # NumPy로 변환 (누락 컬럼은 NaN으로 채움)
                X = np.full((len(batch_slice), len(columns)), np.nan, dtype=np.float64)
                
                for i, col_name in enumerate(columns):
                    if col_name in cols_to_scan:
                        col_idx = cols_to_scan.index(col_name)
                        X[:, i] = np.array(batch_slice.column(col_idx), dtype=np.float64)
                
                yield X, global_offset
                global_offset += len(X)
            
            files_processed += 1
            file_elapsed = time.time() - file_start
            
            # 진행 로그 (10개마다 또는 처음/마지막)
            if verbose and (files_processed % 10 == 0 or files_processed == 1 or file_idx == len(parquet_paths) - 1):
                elapsed = time.time() - start_time
                rate = files_processed / elapsed if elapsed > 0 else 0
                eta = (len(sample_indices) - files_processed) / rate if rate > 0 else 0
                log(f"    [{files_processed}/{len(sample_indices)} files] "
                    f"{global_offset:,} rows processed | "
                    f"{rate:.1f} files/s | ETA: {eta/60:.1f}min")
        
        except Exception as e:
            if verbose:
                log(f"  ⚠️ Warning: Could not read file {path}: {e}")
            continue
    
    if verbose:
        total_elapsed = time.time() - start_time
        log(f"  Sampled data reading complete: {global_offset:,} rows in {total_elapsed:.1f}s "
            f"({global_offset/total_elapsed:.0f} rows/s)")

