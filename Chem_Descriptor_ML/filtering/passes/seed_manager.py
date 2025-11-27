"""
Seed 관리 - SeedSequence.spawn을 사용하여 독립적이고 재현 가능한 시드 생성
"""

import numpy as np
from typing import List, Dict


class SeedManager:
    """
    독립적인 seed 공간 관리
    
    SeedSequence.spawn을 사용하여 각 track마다 독립적인 시드 스트림 생성
    - 파일 추가/제거 시에도 재현성 보장
    - 각 track은 완전히 독립적인 난수 시퀀스 사용
    """
    SPACES = {
        'statistics': 0,
        'spearman': 1,
        'hsic': 2,
        'rdc': 3
    }
    
    def __init__(self, base_seed: int):
        """
        Args:
            base_seed: 기본 랜덤 시드
        """
        self.base = base_seed
        # 메인 SeedSequence 생성
        self.main_sequence = np.random.SeedSequence(base_seed)
        # Track별 독립적인 시드 생성
        self.track_sequences: Dict[str, np.random.SeedSequence] = {}
        
        # 각 track에 대해 spawn
        track_seeds = self.main_sequence.spawn(len(self.SPACES))
        for track_name, seed_seq in zip(self.SPACES.keys(), track_seeds):
            self.track_sequences[track_name] = seed_seq
    
    def get_seeds(self, track: str, n: int) -> List[int]:
        """
        특정 track의 seed 리스트 반환
        
        Args:
            track: seed track 이름 ('statistics', 'spearman', 'hsic', 'rdc')
            n: 필요한 seed 개수
            
        Returns:
            독립적인 seed 리스트
        """
        if track not in self.track_sequences:
            raise ValueError(f"Unknown track: {track}")
        
        # 이 track에서 n개의 독립적인 시드 생성
        spawned = self.track_sequences[track].spawn(n)
        # entropy를 정수로 변환, 32-bit signed int 범위로 제한 (leidenalg 호환)
        seeds = []
        for seq in spawned:
            # generate_state returns uint64, convert to int32 range
            seed_val = int(seq.generate_state(1)[0] % 2**31)
            seeds.append(seed_val)
        return seeds
    
    def get_rng(self, track: str, idx: int = 0):
        """
        특정 track의 numpy random generator 반환
        
        Args:
            track: seed track 이름
            idx: generator 인덱스
            
        Returns:
            독립적인 numpy Generator
        """
        if track not in self.track_sequences:
            raise ValueError(f"Unknown track: {track}")
        
        # idx번째 독립적인 generator 생성
        child_seq = self.track_sequences[track].spawn(idx + 1)[idx]
        return np.random.default_rng(child_seq)
