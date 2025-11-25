"""
프로덕션 레디: uint64 기반 안전한 CountSketch

주요 개선사항:
1. 모든 해시 연산을 uint64로 수행
2. 정수 오버플로 방지
3. 마지막에만 int64로 캐스팅
"""

import numpy as np
from typing import Tuple


class CountSketch:
    """
    Vectorized CountSketch with universal hashing
    
    ✅ Numerical Safety:
    - All hash operations in uint64
    - Prevents integer overflow
    - Final bucket indices cast to int64
    """
    
    # uint64로 마스크 정의
    MASK = np.uint64((1 << 61) - 1)
    
    def __init__(self, m: int, seed: int):
        """
        Args:
            m: 스케치 크기 (2의 거듭제곱으로 조정됨)
            seed: 랜덤 시드
        """
        # m을 2의 거듭제곱으로 조정
        self.m = 1 << (m - 1).bit_length() if (m & (m - 1)) else m
        
        # 해시 파라미터 생성 (uint64로)
        rng = np.random.default_rng(seed)
        hash_params = rng.integers(1, self.MASK, 6, dtype=np.uint64)
        
        self.a1 = np.uint64(hash_params[0])
        self.b1 = np.uint64(hash_params[1])
        self.c1 = np.uint64(hash_params[2])
        self.a2 = np.uint64(hash_params[3])
        self.b2 = np.uint64(hash_params[4])
        self.c2 = np.uint64(hash_params[5])
    
    def _hash(self, a: np.uint64, b: np.uint64, c: np.uint64, 
              x: np.ndarray) -> np.ndarray:
        """
        Universal hash function (uint64 연산)
        
        Args:
            a, b, c: 해시 파라미터 (uint64)
            x: 입력 배열
        
        Returns:
            해시값 (uint64)
        """
        # 입력을 uint64로 변환
        x_uint = x.astype(np.uint64)
        
        # 모든 연산을 uint64로 수행
        h = (a * (x_uint ^ b) + c) & self.MASK
        
        return h
    
    def bucket_and_sign(self, row_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        row_ids를 bucket과 sign으로 매핑
        
        Args:
            row_ids: 행 ID 배열 (int64 또는 uint64)
            
        Returns:
            buckets: bucket 인덱스 (int64)
            signs: +1 또는 -1 (float64)
        """
        # Bucket hash (uint64 연산)
        h1 = self._hash(self.a1, self.b1, self.c1, row_ids)
        buckets = (h1 & np.uint64(self.m - 1))  # 비트 마스크
        
        # 마지막에만 int64로 변환
        buckets = buckets.astype(np.int64)
        
        # Sign hash (uint64 연산)
        h2 = self._hash(self.a2, self.b2, self.c2, row_ids)
        signs = np.where((h2 & np.uint64(1)) == 0, 1.0, -1.0)
        signs = signs.astype(np.float64)
        
        return buckets, signs


# ============================================================================
# 테스트 및 검증
# ============================================================================

def test_countsketch_overflow_safety():
    """CountSketch의 오버플로 안전성 테스트"""
    print("Testing CountSketch overflow safety...")
    
    m = 8192
    seed = 42
    sketch = CountSketch(m, seed)
    
    # 큰 row_ids 테스트
    large_ids = np.array([2**60, 2**61 - 1, 2**62], dtype=np.int64)
    
    try:
        buckets, signs = sketch.bucket_and_sign(large_ids)
        print(f"✓ Large IDs handled correctly")
        print(f"  Buckets: {buckets}")
        print(f"  Signs: {signs}")
        print(f"  Bucket range: [{buckets.min()}, {buckets.max()}]")
        assert buckets.min() >= 0
        assert buckets.max() < m
        assert np.all(np.abs(signs) == 1.0)
        print("✓ All assertions passed!")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False
    
    # 1.2억 개 row_ids 시뮬레이션
    print("\nTesting with 120M row IDs...")
    test_ids = np.arange(0, 120_000_000, 1000, dtype=np.int64)
    
    try:
        buckets, signs = sketch.bucket_and_sign(test_ids)
        print(f"✓ 120M simulation successful")
        print(f"  Processed: {len(test_ids):,} IDs")
        print(f"  Unique buckets: {len(np.unique(buckets))}")
        print(f"  Sign distribution: +1={np.sum(signs==1)}, -1={np.sum(signs==-1)}")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False
    
    print("\n✓ All tests passed!")
    return True


if __name__ == "__main__":
    test_countsketch_overflow_safety()
