"""
Cluster Backtracking Module - surviving_descriptors_clusters.json 생성

Pass 4 → 3 → 2 → 1 순서로 역추적하면서 각 surviving descriptor의
모든 클러스터 멤버들을 재귀적으로 찾아냅니다.

Author: Memory-Safe Cluster Tracker
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Set, Optional
from collections import defaultdict


class ClusterBacktracker:
    """
    역방향 클러스터 추적기
    
    최종 surviving descriptors에서 시작하여 Pass 4 → 3 → 2 → 1 순서로
    거슬러 올라가며 각 descriptor와 클러스터로 묶였던 모든 멤버들을 찾습니다.
    
    Example:
        - A가 최종 생존
        - Pass 3: A가 대표, B가 제거 (A-B 클러스터)
        - Pass 2: B가 대표, C가 제거 (B-C 클러스터)
        → A의 all_cluster_members = {A, B, C}
    """
    
    def __init__(self, output_dir: Path, verbose: bool = True):
        """
        Args:
            output_dir: checkpoint 파일들이 있는 디렉토리
            verbose: 로깅 출력 여부
        """
        self.output_dir = Path(output_dir)
        self.verbose = verbose
        
        # Checkpoint 파일 경로
        self.checkpoint_files = {
            'pass1': self.output_dir / 'pass1_variance_filtering.json',
            'pass2': self.output_dir / 'pass2_spearman.json',
            'pass3': self.output_dir / 'pass3_vif.json',
            'pass4': self.output_dir / 'pass4_nonlinear.json',
        }
        
        # 로드된 checkpoint 데이터
        self.checkpoints = {}
        
    def _log(self, msg: str):
        """로깅"""
        if self.verbose:
            print(msg)
    
    def load_checkpoints(self):
        """모든 checkpoint 파일 로드"""
        self._log("\n📂 Loading checkpoint files...")
        
        for pass_name, file_path in self.checkpoint_files.items():
            if file_path.exists():
                with open(file_path, 'r') as f:
                    self.checkpoints[pass_name] = json.load(f)
                self._log(f"  ✓ {pass_name}: {file_path.name}")
            else:
                self._log(f"  ⚠ {pass_name}: Not found - {file_path.name}")
                self.checkpoints[pass_name] = None
    
    def _extract_cluster_info(self, pass_name: str) -> Dict[str, Set[str]]:
        """
        특정 Pass의 클러스터 정보 추출
        
        Args:
            pass_name: 'pass2', 'pass3', 'pass4'
        
        Returns:
            Dict[representative, Set[all_members]]
            예: {'A': {'A', 'B', 'C'}, 'D': {'D', 'E'}}
        """
        checkpoint = self.checkpoints.get(pass_name)
        if checkpoint is None:
            return {}
        
        cluster_map = {}
        
        # Pass 1은 클러스터 정보가 없음 (variance filtering만)
        if pass_name == 'pass1':
            return {}
        
        # Pass 2, 3, 4는 클러스터 정보 있음
        if 'clusters' in checkpoint:
            for cluster in checkpoint['clusters']:
                representative = cluster.get('representative')
                members = set(cluster.get('members', []))
                
                if representative and members:
                    cluster_map[representative] = members
        
        # representatives 딕셔너리에서도 정보 추출
        if 'representatives' in checkpoint:
            for rep, cluster_info in checkpoint['representatives'].items():
                members = set(cluster_info.get('members', []))
                if members:
                    cluster_map[rep] = members
        
        return cluster_map
    
    def backtrack_clusters(self, surviving_descriptors: List[str]) -> Dict[str, Dict]:
        """
        각 surviving descriptor의 전체 클러스터 멤버 역추적
        
        Args:
            surviving_descriptors: 최종 생존한 descriptors 리스트
        
        Returns:
            Dict[descriptor, cluster_info]
            cluster_info = {
                'all_cluster_members': Set[str],
                'alternative_descriptors': Set[str],
                'removal_history': Dict[pass, List[removed]]
            }
        """
        self._log(f"\n🔍 Backtracking clusters for {len(surviving_descriptors)} descriptors...")
        
        results = {}
        
        for descriptor in surviving_descriptors:
            cluster_info = self._backtrack_single_descriptor(descriptor)
            results[descriptor] = cluster_info
        
        return results
    
    def _backtrack_single_descriptor(self, descriptor: str) -> Dict:
        """
        단일 descriptor의 클러스터 역추적
        
        재귀적으로 Pass 4 → 3 → 2 → 1 순서로 추적
        """
        # 추적할 descriptors (BFS 방식)
        tracked = {descriptor}
        all_members = {descriptor}
        removal_history = {}
        
        # Pass 순서 (역순)
        passes = ['pass4', 'pass3', 'pass2', 'pass1']
        
        for pass_name in passes:
            # 이번 pass의 클러스터 맵
            cluster_map = self._extract_cluster_info(pass_name)
            
            if not cluster_map:
                continue
            
            # 현재 tracked descriptors가 대표인 클러스터 찾기
            new_members = set()
            removed_in_this_pass = []
            
            for tracked_desc in list(tracked):
                if tracked_desc in cluster_map:
                    # 이 descriptor가 대표인 클러스터의 모든 멤버
                    cluster_members = cluster_map[tracked_desc]
                    new_members.update(cluster_members)
                    
                    # 제거된 멤버 (대표 제외)
                    removed = cluster_members - {tracked_desc}
                    removed_in_this_pass.extend(removed)
            
            # 새로운 멤버들을 다음 pass에서도 추적
            if new_members:
                all_members.update(new_members)
                tracked.update(new_members)
            
            # 이번 pass에서 제거된 멤버 기록
            if removed_in_this_pass:
                removal_history[pass_name] = sorted(removed_in_this_pass)
        
        # Alternative descriptors (본인 제외)
        alternative = all_members - {descriptor}
        
        return {
            'all_cluster_members': sorted(all_members),
            'alternative_descriptors': sorted(alternative),
            'removal_history': removal_history,
            'total_alternatives': len(alternative)
        }
    
    def build_cluster_structure(self, final_descriptors_file: Optional[str] = None) -> Dict:
        """
        surviving_descriptors_clusters.json 구조 생성
        
        Args:
            final_descriptors_file: final_descriptors.txt 파일 경로
                                   (None이면 output_dir/final_descriptors.txt)
        
        Returns:
            전체 클러스터 구조 딕셔너리
        """
        self._log("\n" + "="*70)
        self._log("Building Surviving Descriptors Cluster Structure")
        self._log("="*70)
        
        # Checkpoint 파일 로드
        self.load_checkpoints()
        
        # 최종 descriptors 로드
        if final_descriptors_file is None:
            final_descriptors_file = self.output_dir / 'final_descriptors.txt'
        else:
            final_descriptors_file = Path(final_descriptors_file)
        
        if not final_descriptors_file.exists():
            raise FileNotFoundError(f"Final descriptors file not found: {final_descriptors_file}")
        
        with open(final_descriptors_file, 'r') as f:
            surviving_descriptors = [line.strip() for line in f if line.strip()]
        
        self._log(f"\n📊 Total surviving descriptors: {len(surviving_descriptors)}")
        
        # 클러스터 역추적
        cluster_results = self.backtrack_clusters(surviving_descriptors)
        
        # 통계 계산
        stats = self._calculate_statistics(cluster_results)
        
        # 최종 구조 생성
        structure = {
            'metadata': {
                'description': f'Cluster structure for {len(surviving_descriptors)} surviving descriptors with full backtracking',
                'total_descriptors': len(surviving_descriptors),
                'descriptors_with_alternatives': sum(1 for info in cluster_results.values() if info['total_alternatives'] > 0),
                'standalone_descriptors': sum(1 for info in cluster_results.values() if info['total_alternatives'] == 0),
                'total_alternative_descriptors': sum(info['total_alternatives'] for info in cluster_results.values()),
            },
            'statistics': stats,
            'descriptors': {}
        }
        
        # 각 descriptor 정보 추가
        for descriptor, info in cluster_results.items():
            structure['descriptors'][descriptor] = {
                'cluster_size': len(info['all_cluster_members']),
                'is_representative': True,  # 최종 생존자는 모두 대표
                'alternative_descriptors': info['alternative_descriptors'],
                'all_cluster_members': info['all_cluster_members'],
                'removal_history': info['removal_history'],
                'total_alternatives': info['total_alternatives']
            }
        
        self._log("\n" + "="*70)
        self._log("Cluster Structure Built Successfully!")
        self._log("="*70)
        self._log(f"Total descriptors: {structure['metadata']['total_descriptors']}")
        self._log(f"With alternatives: {structure['metadata']['descriptors_with_alternatives']}")
        self._log(f"Standalone: {structure['metadata']['standalone_descriptors']}")
        self._log(f"Total alternatives: {structure['metadata']['total_alternative_descriptors']}")
        
        return structure
    
    def _calculate_statistics(self, cluster_results: Dict) -> Dict:
        """클러스터 통계 계산"""
        cluster_sizes = [len(info['all_cluster_members']) for info in cluster_results.values()]
        
        # 크기별 분포
        size_distribution = defaultdict(int)
        for size in cluster_sizes:
            size_distribution[size] += 1
        
        return {
            'cluster_size_mean': float(np.mean(cluster_sizes)) if cluster_sizes else 0,
            'cluster_size_median': float(np.median(cluster_sizes)) if cluster_sizes else 0,
            'cluster_size_min': int(np.min(cluster_sizes)) if cluster_sizes else 0,
            'cluster_size_max': int(np.max(cluster_sizes)) if cluster_sizes else 0,
            'cluster_size_std': float(np.std(cluster_sizes)) if cluster_sizes else 0,
            'size_distribution': {str(k): v for k, v in sorted(size_distribution.items())}
        }
    
    def save_to_json(self, structure: Dict, output_file: Optional[str] = None):
        """JSON 파일로 저장"""
        if output_file is None:
            output_file = self.output_dir / 'surviving_descriptors_clusters.json'
        else:
            output_file = Path(output_file)
        
        with open(output_file, 'w') as f:
            json.dump(structure, f, indent=2)
        
        self._log(f"\n💾 Saved to: {output_file}")
        self._log(f"   File size: {output_file.stat().st_size / 1024:.1f} KB")


def create_cluster_structure(output_dir: str, 
                            final_descriptors_file: Optional[str] = None,
                            output_file: Optional[str] = None,
                            verbose: bool = True) -> Dict:
    """
    surviving_descriptors_clusters.json 생성 (편의 함수)
    
    Args:
        output_dir: checkpoint 파일들이 있는 디렉토리
        final_descriptors_file: final_descriptors.txt 파일 경로
        output_file: 출력 JSON 파일 경로
        verbose: 로깅 출력 여부
    
    Returns:
        클러스터 구조 딕셔너리
    
    Example:
        >>> structure = create_cluster_structure(
        ...     output_dir='output/results',
        ...     verbose=True
        ... )
    """
    backtracker = ClusterBacktracker(output_dir, verbose)
    structure = backtracker.build_cluster_structure(final_descriptors_file)
    backtracker.save_to_json(structure, output_file)
    
    return structure


# ============================================================================
# CLI Interface
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate surviving_descriptors_clusters.json with full backtracking"
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Directory containing checkpoint files'
    )
    parser.add_argument(
        '--final-descriptors',
        type=str,
        default=None,
        help='Path to final_descriptors.txt (default: output-dir/final_descriptors.txt)'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        default=None,
        help='Output JSON file path (default: output-dir/surviving_descriptors_clusters.json)'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress verbose output'
    )
    
    args = parser.parse_args()
    
    structure = create_cluster_structure(
        output_dir=args.output_dir,
        final_descriptors_file=args.final_descriptors,
        output_file=args.output_file,
        verbose=not args.quiet
    )
    
    print(f"\n✅ Done! Generated {len(structure['descriptors'])} descriptor clusters")
