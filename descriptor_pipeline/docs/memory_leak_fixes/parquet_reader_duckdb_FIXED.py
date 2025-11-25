"""
DuckDB 기반 Parquet 파일 스트리밍 I/O (메모리 안정화 v2)

PyArrow의 scanner.to_batches()는 내부적으로 메모리를 유지하는 문제가 있어
DuckDB로 완전히 교체하여 진정한 스트리밍 처리를 구현합니다.

메모리 누수 수정사항:
- DataFrame → NumPy 변환 시 .copy() 추가하여 view 참조 방지
- DuckDB 연결 완전 종료 보장
- 중복 함수 정의 제거
- 각 배치마다 명시적 메모리 정리

Note:
    - iter_batches만 DuckDB로 교체
    - 나머지 유틸리티 함수들은 기존 parquet_reader.py 재사용
"""

import gc
import numpy as np
import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
from typing import List, Iterator, Tuple, Dict, Optional
from pathlib import Path
from descriptor_pipeline.utils.logging import log


# Parquet writer 설정
ROW_GROUP_SIZE = 5000  # 메모리 안정성을 위한 row group 크기


def iter_batches_duckdb(
    parquet_paths: List[str],
    columns: List[str],
    batch_rows: int = 1_000_000,
    verbose: bool = False,
    sample_per_file: Optional[int] = None,
    random_seed: Optional[int] = None
) -> Iterator[Tuple[np.ndarray, int]]:
    """
    DuckDB를 사용한 메모리 안정적 배치 스트리밍
    
    PyArrow의 메모리 누수 문제를 완전히 해결하기 위해 DuckDB로 재작성
    DuckDB는 진정한 스트리밍 방식으로 메모리를 관리합니다.
    
    메모리 누수 수정:
    - DataFrame.values → .copy()로 독립 배열 생성
    - 각 배치 후 명시적 del + gc.collect()
    - finally 블록에서 확실한 연결 종료
    
    Args:
        parquet_paths: Parquet 파일 경로 리스트
        columns: 읽을 컬럼 이름 리스트
        batch_rows: 배치 크기
        verbose: 로깅 출력 여부
        sample_per_file: 각 파일당 샘플링할 행 수 (None이면 전체)
        random_seed: 랜덤 샘플링 시드
        
    Yields:
        (X, offset): 데이터 배치와 글로벌 오프셋
    """
    global_offset = 0
    conn = None
    
    try:
        # DuckDB 연결 생성 (메모리 제한 설정)
        conn = duckdb.connect(':memory:')
        
        # 메모리 제한 설정 (중요!)
        conn.execute("SET memory_limit='10GB'")  # DuckDB가 사용할 최대 메모리
        conn.execute("SET threads=4")  # 스레드 제한
        
        if verbose:
            log(f"  DuckDB-based streaming: {len(parquet_paths)} files, {len(columns)} columns")
            log(f"  Batch size: {batch_rows:,} rows")
        
        # 샘플링 없이 전체 읽기
        if sample_per_file is None:
            # 파일별로 읽기 (메모리 안전 최우선)
            for file_idx, path in enumerate(sorted(parquet_paths)):
                if verbose and (file_idx + 1) % 50 == 0:
                    log(f"  Processing file {file_idx + 1}/{len(parquet_paths)}")
                
                # 컬럼 선택
                column_list = ", ".join([f'"{col}"' for col in columns])
                
                # 파일 크기 확인
                count_query = f'SELECT COUNT(*) as cnt FROM read_parquet("{path}")'
                try:
                    file_rows = conn.execute(count_query).fetchone()[0]
                except:
                    if verbose:
                        log(f"  ⚠️ Skipping file {path} (read error)")
                    continue
                
                if file_rows == 0:
                    continue
                
                # 파일 내에서 배치 단위로 읽기
                file_offset = 0
                while file_offset < file_rows:
                    # LIMIT/OFFSET로 배치 읽기
                    batch_query = f'''
                        SELECT {column_list}
                        FROM read_parquet("{path}")
                        LIMIT {batch_rows} OFFSET {file_offset}
                    '''
                    
                    df_batch = conn.execute(batch_query).fetch_df()
                    
                    if df_batch is None or len(df_batch) == 0:
                        break
                    
                    # NumPy로 변환 - CRITICAL: .copy()로 독립 배열 생성
                    # pandas DataFrame 참조를 끊어서 메모리 누수 방지
                    X = df_batch[columns].values.copy().astype(np.float64)
                    
                    # DataFrame 즉시 삭제
                    del df_batch
                    
                    yield X, global_offset
                    global_offset += len(X)
                    file_offset += len(X)
                    
                    # 배치마다 메모리 정리
                    gc.collect()
                    
                    # 마지막 배치 확인
                    if len(X) < batch_rows:
                        break
        
        else:
            # 샘플링 모드
            if verbose:
                log(f"  Sampling mode: {sample_per_file:,} rows per file (seed={random_seed})")
            
            if random_seed is not None:
                np.random.seed(random_seed)
            
            for file_idx, path in enumerate(sorted(parquet_paths)):
                if verbose and (file_idx + 1) % 50 == 0:
                    log(f"  Processing file {file_idx + 1}/{len(parquet_paths)}")
                
                # 파일 크기 확인
                column_list = ", ".join([f'"{col}"' for col in columns])
                count_query = f'SELECT COUNT(*) as cnt FROM read_parquet("{path}")'
                try:
                    file_rows = conn.execute(count_query).fetchone()[0]
                except:
                    if verbose:
                        log(f"  ⚠️ Skipping file {path}")
                    continue
                
                if file_rows == 0:
                    continue
                
                if file_rows <= sample_per_file:
                    # 전체 사용 (배치로 나누기)
                    file_offset = 0
                    while file_offset < file_rows:
                        batch_query = f'''
                            SELECT {column_list}
                            FROM read_parquet("{path}")
                            LIMIT {batch_rows} OFFSET {file_offset}
                        '''
                        df_batch = conn.execute(batch_query).fetch_df()
                        
                        if df_batch is None or len(df_batch) == 0:
                            break
                        
                        # CRITICAL: .copy() 추가
                        X = df_batch[columns].values.copy().astype(np.float64)
                        del df_batch
                        
                        yield X, global_offset
                        global_offset += len(X)
                        file_offset += len(X)
                        gc.collect()
                        
                        if len(X) < batch_rows:
                            break
                else:
                    # 샘플링
                    sample_indices = np.random.choice(file_rows, size=sample_per_file, replace=False)
                    sample_indices = np.sort(sample_indices)
                    
                    # 샘플링된 행들을 배치로 읽기
                    for start_idx in range(0, len(sample_indices), batch_rows):
                        end_idx = min(start_idx + batch_rows, len(sample_indices))
                        batch_indices = sample_indices[start_idx:end_idx]
                        
                        # WHERE 절로 특정 행만 선택
                        # DuckDB는 row_number()를 지원
                        indices_str = ','.join(map(str, batch_indices))
                        batch_query = f'''
                            SELECT {column_list}
                            FROM (
                                SELECT *, ROW_NUMBER() OVER () - 1 as row_num
                                FROM read_parquet("{path}")
                            )
                            WHERE row_num IN ({indices_str})
                        '''
                        
                        df_batch = conn.execute(batch_query).fetch_df()
                        
                        if df_batch is None or len(df_batch) == 0:
                            break
                        
                        # CRITICAL: .copy() 추가
                        X = df_batch[columns].values.copy().astype(np.float64)
                        del df_batch
                        
                        yield X, global_offset
                        global_offset += len(X)
                        gc.collect()
    
    finally:
        # 연결 명시적 종료 - 확실하게!
        if conn is not None:
            try:
                conn.close()
            except Exception as e:
                if verbose:
                    log(f"  Warning: Error closing DuckDB connection: {e}")
        
        del conn
        gc.collect()
        
        if verbose:
            log(f"  DuckDB streaming completed: {global_offset:,} total rows")


# ============================================================================
# 기존 parquet_reader.py의 유틸리티 함수들 재사용
# (create_sampled_parquet_file, load_sample_indices, get_total_rows 등)
# ============================================================================

# 이 함수들은 기존 parquet_reader.py에서 import하여 사용
# from descriptor_pipeline.io.parquet_reader import (
#     generate_sample_indices_per_file,
#     load_sample_indices,
#     get_total_rows,
#     create_sampled_parquet_file,
#     write_parquet_batches
# )
