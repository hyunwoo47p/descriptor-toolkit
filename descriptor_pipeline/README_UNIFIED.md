# Unified Descriptor Pipeline

## 개요
`cluster_gpu_parallel`과 `global_processor`의 모든 CLI 인자를 통합한 단일 파이프라인 인터페이스입니다.

## 주요 변경사항

### 수정된 파일
1. **`cli/run_pipeline.py`** (92줄 → 594줄)
   - 모든 파라미터를 지원하는 통합 CLI
   - 10개 이상의 인자 그룹으로 구조화
   - GPU 자동 감지 및 검증

2. **`config/settings.py`** (81줄 → 154줄)
   - 70개 이상의 설정 파라미터 지원
   - 모든 원본 소스의 옵션 포함
   - 하위 호환성 유지

## 빠른 시작

### 기본 실행
```bash
python -m descriptor_pipeline.cli.run_pipeline \
    --parquet-glob "data/*.parquet" \
    --output-dir ./results \
    --gpu
```

### 병렬 처리
```bash
python -m descriptor_pipeline.cli.run_pipeline \
    --parquet-glob "data/*.parquet" \
    --output-dir ./results \
    --gpu \
    --parallel --workers 8
```

### 고급 필터링
```bash
python -m descriptor_pipeline.cli.run_pipeline \
    --parquet-glob "data/*.parquet" \
    --output-dir ./results \
    --gpu \
    --variance-threshold 0.002 \
    --spearman-threshold 0.95 \
    --vif-threshold 10.0 \
    --nonlinear-threshold 0.3
```

## 새로운 기능

### 1. GPU 설정
- `--gpu`: GPU 가속 활성화
- `--no-gpu`: CPU 모드 강제
- `--gpu-id N`: 특정 GPU 선택

### 2. 병렬 처리
- `--parallel`: 병렬 처리 활성화
- `--workers N`: 워커 프로세스 수
- `--file-batch-size N`: 파일 배치 크기
- `--column-batch-size N`: 컬럼 배치 크기

### 3. 샤딩
- `--shards N`: 총 샤드 수
- `--shard-index I`: 현재 샤드 인덱스

### 4. 2-패스 모드
- `--decide-only`: 필터 결정만 계산
- `--apply-only`: 결정 적용만
- `--decisions-out FILE`: 결정 저장
- `--decisions-in FILE`: 결정 로드

### 5. 고급 필터
- `--min-effective-n N`: 최소 유효 샘플
- `--binary-filter`: 이진 필터 활성화
- `--trim-lower P`: trimmed 범위 하한
- `--trim-upper P`: trimmed 범위 상한

### 6. DuckDB 설정
- `--memory SIZE`: 메모리 제한 (예: 4GB)
- `--duckdb-threads N`: 스레드 수
- `--pattern GLOB`: 파일 패턴

## 파라미터 그룹

| 그룹 | 파라미터 수 | 주요 기능 |
|------|------------|----------|
| I/O | 4 | 입출력 설정 |
| GPU | 3 | GPU 설정 |
| Sampling | 3 | 샘플링 설정 |
| Statistics | 8 | 통계 및 분산 |
| Advanced Filter | 6 | 고급 필터 |
| Spearman | 3 | 상관관계 |
| VIF | 1 | 다중공선성 |
| Nonlinear | 6 | HSIC+RDC |
| Processing | 7 | 처리 옵션 |
| DuckDB | 3 | DuckDB 설정 |
| Parallel | 4 | 병렬 처리 |
| Sharding | 2 | 샤딩 |
| Two-Pass | 4 | 2-패스 모드 |
| Output | 2 | 출력 설정 |

## 도움말

모든 파라미터 확인:
```bash
python -m descriptor_pipeline.cli.run_pipeline --help
```

## 참고 문서
- `INTEGRATION_SUMMARY.md`: 상세한 통합 가이드
- `ARGUMENTS.md`: 모든 인자 설명 (준비 중)
- `EXAMPLES.md`: 사용 예제 (준비 중)
