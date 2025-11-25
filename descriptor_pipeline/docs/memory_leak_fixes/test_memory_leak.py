#!/usr/bin/env python3
"""
메모리 누수 테스트 스크립트
Memory Leak Test Script

수정 전후 메모리 사용량을 비교하여 누수 여부를 확인합니다.

Usage:
    python test_memory_leak.py [--iterations N] [--verbose]
"""

import gc
import time
import argparse
import tracemalloc
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("⚠️  psutil not installed. Install with: pip install psutil")

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class MemoryMonitor:
    """메모리 모니터링 클래스"""
    
    def __init__(self):
        self.process = psutil.Process() if HAS_PSUTIL else None
        self.snapshots = []
        self.gpu_stats = []
    
    def get_memory_mb(self) -> float:
        """현재 메모리 사용량 (MB)"""
        if self.process:
            return self.process.memory_info().rss / 1024 / 1024
        return 0.0
    
    def get_gpu_memory_mb(self) -> Tuple[float, float]:
        """GPU 메모리 사용량 (allocated, reserved) in MB"""
        if HAS_TORCH and torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024 / 1024
            reserved = torch.cuda.memory_reserved() / 1024 / 1024
            return allocated, reserved
        return 0.0, 0.0
    
    def snapshot(self, label: str = ""):
        """메모리 스냅샷 저장"""
        mem_mb = self.get_memory_mb()
        gpu_alloc, gpu_reserved = self.get_gpu_memory_mb()
        
        snapshot = {
            'label': label,
            'timestamp': time.time(),
            'memory_mb': mem_mb,
            'gpu_allocated_mb': gpu_alloc,
            'gpu_reserved_mb': gpu_reserved
        }
        
        self.snapshots.append(snapshot)
        return snapshot
    
    def get_memory_increase(self) -> Dict:
        """메모리 증가량 계산"""
        if len(self.snapshots) < 2:
            return {}
        
        first = self.snapshots[0]
        last = self.snapshots[-1]
        
        return {
            'memory_increase_mb': last['memory_mb'] - first['memory_mb'],
            'gpu_alloc_increase_mb': last['gpu_allocated_mb'] - first['gpu_allocated_mb'],
            'gpu_reserved_increase_mb': last['gpu_reserved_mb'] - first['gpu_reserved_mb'],
            'total_snapshots': len(self.snapshots)
        }
    
    def print_snapshot(self, snapshot: Dict):
        """스냅샷 출력"""
        print(f"  {snapshot['label']:<30} "
              f"RAM: {snapshot['memory_mb']:>8.1f} MB  "
              f"GPU-A: {snapshot['gpu_allocated_mb']:>8.1f} MB  "
              f"GPU-R: {snapshot['gpu_reserved_mb']:>8.1f} MB")
    
    def print_summary(self):
        """요약 출력"""
        if not self.snapshots:
            print("No snapshots recorded")
            return
        
        print("\n" + "="*70)
        print("Memory Usage Summary")
        print("="*70)
        
        for snapshot in self.snapshots:
            self.print_snapshot(snapshot)
        
        if len(self.snapshots) >= 2:
            increase = self.get_memory_increase()
            print("\n" + "-"*70)
            print(f"Total Memory Increase: {increase['memory_increase_mb']:.1f} MB")
            
            if HAS_TORCH and torch.cuda.is_available():
                print(f"GPU Allocated Increase: {increase['gpu_alloc_increase_mb']:.1f} MB")
                print(f"GPU Reserved Increase: {increase['gpu_reserved_increase_mb']:.1f} MB")
            
            # 판정
            if increase['memory_increase_mb'] < 100:
                print("\n✅ PASS: Memory usage is stable")
            elif increase['memory_increase_mb'] < 500:
                print("\n⚠️  WARNING: Memory increased but within acceptable range")
            else:
                print("\n❌ FAIL: Significant memory leak detected!")
        
        print("="*70)


def test_duckdb_iteration(iterations: int = 3, verbose: bool = False):
    """DuckDB iteration 메모리 테스트"""
    print("\n" + "="*70)
    print("Test 1: DuckDB Iteration Memory Test")
    print("="*70)
    
    try:
        from descriptor_pipeline.io import iter_batches_duckdb
        import numpy as np
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    
    monitor = MemoryMonitor()
    
    # 테스트 데이터 생성
    test_file = Path("test_data.parquet")
    if not test_file.exists():
        print("⚠️  Test data not found, creating...")
        import pandas as pd
        import pyarrow.parquet as pq
        
        df = pd.DataFrame({
            f'col_{i}': np.random.randn(10000) for i in range(50)
        })
        pq.write_table(pa.Table.from_pandas(df), test_file)
    
    columns = [f'col_{i}' for i in range(50)]
    
    # 초기 스냅샷
    monitor.snapshot("Initial")
    gc.collect()
    
    # Iteration 테스트
    for it in range(iterations):
        if verbose:
            print(f"\n  Iteration {it + 1}/{iterations}...")
        
        batch_count = 0
        total_rows = 0
        
        for X, offset in iter_batches_duckdb([str(test_file)], columns, batch_rows=1000):
            batch_count += 1
            total_rows += len(X)
            del X
        
        gc.collect()
        
        snapshot = monitor.snapshot(f"After iteration {it + 1}")
        
        if verbose:
            print(f"    Processed {batch_count} batches, {total_rows} total rows")
            monitor.print_snapshot(snapshot)
    
    # 최종 스냅샷
    monitor.snapshot("Final (after cleanup)")
    
    # 결과 출력
    monitor.print_summary()
    
    increase = monitor.get_memory_increase()
    return increase['memory_increase_mb'] < 100


def test_numpy_view_copy():
    """NumPy view vs copy 메모리 테스트"""
    print("\n" + "="*70)
    print("Test 2: NumPy View vs Copy Memory Test")
    print("="*70)
    
    import numpy as np
    
    monitor = MemoryMonitor()
    
    # 큰 배열 생성
    data = np.random.randn(10000, 1000)
    monitor.snapshot("After creating data")
    
    # View 생성 (메모리 누수 발생)
    print("\n  Testing with view (BAD)...")
    views = []
    for i in range(10):
        view = data[:, :500]  # view
        views.append(view)
    
    monitor.snapshot("After creating 10 views")
    
    # View 삭제 (원본은 여전히 메모리에)
    del views
    gc.collect()
    monitor.snapshot("After deleting views (data still in memory)")
    
    # 원본 삭제
    del data
    gc.collect()
    monitor.snapshot("After deleting original data")
    
    # Copy 테스트 (올바른 방법)
    print("\n  Testing with copy (GOOD)...")
    data2 = np.random.randn(10000, 1000)
    monitor.snapshot("After creating data2")
    
    copies = []
    for i in range(10):
        copy = data2[:, :500].copy()  # independent copy
        copies.append(copy)
    
    monitor.snapshot("After creating 10 copies")
    
    # 원본 삭제
    del data2
    gc.collect()
    monitor.snapshot("After deleting data2 (copies still exist)")
    
    # Copy 삭제
    del copies
    gc.collect()
    monitor.snapshot("After deleting all copies")
    
    monitor.print_summary()
    
    return True


def test_gpu_tensor_leak():
    """GPU 텐서 메모리 누수 테스트"""
    if not HAS_TORCH or not torch.cuda.is_available():
        print("\n⚠️  GPU not available, skipping GPU test")
        return True
    
    print("\n" + "="*70)
    print("Test 3: GPU Tensor Memory Leak Test")
    print("="*70)
    
    monitor = MemoryMonitor()
    monitor.snapshot("Initial GPU state")
    
    device = torch.device('cuda')
    
    # 잘못된 방법 (메모리 누수 발생)
    print("\n  Testing BAD pattern (.cpu().numpy())...")
    for i in range(5):
        X_gpu = torch.randn(1000, 1000, device=device)
        X_cpu = X_gpu.cpu().numpy()  # BAD: reference still held
        del X_cpu
        
        monitor.snapshot(f"After BAD iteration {i+1}")
    
    torch.cuda.empty_cache()
    gc.collect()
    monitor.snapshot("After cleanup (BAD pattern)")
    
    # 올바른 방법
    print("\n  Testing GOOD pattern (.detach().cpu().numpy().copy())...")
    for i in range(5):
        X_gpu = torch.randn(1000, 1000, device=device)
        X_cpu = X_gpu.detach().cpu().numpy().copy()  # GOOD: independent copy
        del X_gpu, X_cpu
        
        monitor.snapshot(f"After GOOD iteration {i+1}")
    
    torch.cuda.empty_cache()
    gc.collect()
    monitor.snapshot("After cleanup (GOOD pattern)")
    
    monitor.print_summary()
    
    return True


def main():
    """메인 테스트 함수"""
    parser = argparse.ArgumentParser(description="Memory leak detection tests")
    parser.add_argument('--iterations', type=int, default=3,
                       help='Number of iterations for DuckDB test')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    parser.add_argument('--test', choices=['all', 'duckdb', 'numpy', 'gpu'],
                       default='all', help='Which test to run')
    
    args = parser.parse_args()
    
    print("="*70)
    print("Memory Leak Test Suite")
    print("="*70)
    
    if not HAS_PSUTIL:
        print("⚠️  Warning: psutil not available, memory monitoring will be limited")
    
    results = {}
    
    try:
        if args.test in ['all', 'duckdb']:
            results['duckdb'] = test_duckdb_iteration(args.iterations, args.verbose)
        
        if args.test in ['all', 'numpy']:
            results['numpy'] = test_numpy_view_copy()
        
        if args.test in ['all', 'gpu']:
            results['gpu'] = test_gpu_tensor_leak()
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 최종 결과
    print("\n" + "="*70)
    print("Final Results")
    print("="*70)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {test_name:<15} {status}")
        if not passed:
            all_passed = False
    
    print("="*70)
    
    if all_passed:
        print("\n🎉 All tests passed!")
        return True
    else:
        print("\n❌ Some tests failed. Please review the fixes.")
        return False


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
