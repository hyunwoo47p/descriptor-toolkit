"""
Memory-efficient Iterative VIF Filtering with Pass2 Cluster Tracking

메모리 최적화:
1. VIF 계산 시 불필요한 복사본 제거
2. 명시적 메모리 해제 (del, gc.collect)
3. VIF history 최소화 (마지막 값만 저장)
4. 임시 배열들 즉시 삭제
"""

import numpy as np
import gc
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import json


class IterativeVIFFiltering:
    """
    Memory-efficient Iterative VIF 기반 다중공선성 제거
    """
    
    def __init__(
        self, 
        vif_threshold: float = 10.0,
        correlation_threshold: float = 0.7,
        verbose: bool = True
    ):
        self.vif_threshold = vif_threshold
        self.correlation_threshold = correlation_threshold
        self.verbose = verbose
    
    def _log(self, msg: str):
        if self.verbose:
            print(msg)
    
    def _compute_vif_values(
        self, 
        data: np.ndarray, 
        columns: List[str]
    ) -> Dict[str, float]:
        """
        VIF 계산 (메모리 효율적)
        
        Returns:
            {descriptor_name: vif_value}
        """
        n_samples, p = data.shape
        vif_dict = {}
        
        # NaN 체크 - view만 사용
        nan_mask = np.isnan(data).any(axis=1)
        valid_rows = ~nan_mask
        n_valid = valid_rows.sum()
        
        if n_valid < p + 2:
            self._log(f"Warning: Not enough samples ({n_valid}) for VIF calculation")
            return {col: np.nan for col in columns}
        
        # Clean data - 복사본 한 번만
        clean_data = data[valid_rows].copy()
        
        try:
            # 상수항 추가 - 메모리 효율적으로
            data_with_const = add_constant(clean_data, has_constant='skip')
            
            # VIF 계산
            for i, col in enumerate(columns):
                try:
                    vif = variance_inflation_factor(data_with_const, i + 1)
                    vif_dict[col] = float(vif)
                except Exception as e:
                    if self.verbose:
                        self._log(f"Warning: VIF failed for {col}: {e}")
                    vif_dict[col] = np.nan
            
            # 명시적 메모리 해제
            del data_with_const
            del clean_data
            
        except Exception as e:
            self._log(f"Error in VIF calculation: {e}")
            return {col: np.nan for col in columns}
        
        return vif_dict
    
    def _find_best_surviving_representative(
        self,
        removed_desc: str,
        removed_idx: int,
        surviving_descriptors: List[str],
        correlation_matrix: Optional[np.ndarray],
        all_columns: List[str],
        pass2_clusters: Dict[str, Dict]
    ) -> Optional[Tuple[str, float]]:
        """
        Best surviving descriptor 찾기
        
        Returns:
            (best_descriptor, correlation_value) or (None, 0.0)
        """
        if correlation_matrix is None or len(surviving_descriptors) == 0:
            return None, 0.0
        
        # Surviving descriptors의 인덱스
        surviving_indices = []
        for desc in surviving_descriptors:
            try:
                idx = all_columns.index(desc)
                surviving_indices.append(idx)
            except ValueError:
                continue
        
        if len(surviving_indices) == 0:
            return None, 0.0
        
        # Method 1: 직접 correlation
        best_desc = None
        best_corr = 0.0
        
        for surv_idx in surviving_indices:
            surv_name = all_columns[surv_idx]
            corr = abs(correlation_matrix[removed_idx, surv_idx])
            
            if corr > best_corr:
                best_corr = corr
                best_desc = surv_name
        
        # Method 2: Pass2 클러스터 멤버들과의 평균 (더 robust)
        if removed_desc in pass2_clusters:
            cluster_members = pass2_clusters[removed_desc].get('members', [])
            
            if cluster_members and len(cluster_members) > 0:
                for surv_name in surviving_descriptors:
                    try:
                        surv_idx = all_columns.index(surv_name)
                    except ValueError:
                        continue
                    
                    # 클러스터 멤버들과의 평균 correlation
                    member_corrs = []
                    for member in cluster_members[:50]:  # 최대 50개만 (메모리 절약)
                        try:
                            member_idx = all_columns.index(member)
                            member_corrs.append(abs(correlation_matrix[surv_idx, member_idx]))
                        except (ValueError, IndexError):
                            continue
                    
                    if member_corrs:
                        avg_corr = np.mean(member_corrs)
                        # 가중 평균
                        direct_corr = abs(correlation_matrix[removed_idx, surv_idx])
                        combined_corr = 0.7 * direct_corr + 0.3 * avg_corr
                        
                        if combined_corr > best_corr:
                            best_corr = combined_corr
                            best_desc = surv_name
        
        # Threshold 체크
        if best_corr >= self.correlation_threshold:
            return best_desc, float(best_corr)
        
        return None, 0.0
    
    def process(
        self,
        data: np.ndarray,
        columns: List[str],
        pass2_info: Optional[Dict] = None,
        correlation_matrix: Optional[np.ndarray] = None,
        output_dir: Optional[Path] = None
    ) -> Tuple[List[str], Dict]:
        """
        Iterative VIF 필터링 실행 (메모리 효율적)
        
        Returns:
            (remaining_columns, checkpoint_dict)
        """
        self._log("\n" + "="*80)
        self._log("Pass3: Memory-Efficient Iterative VIF Filtering")
        self._log("="*80)
        self._log(f"VIF Threshold: {self.vif_threshold}")
        self._log(f"Correlation Threshold: {self.correlation_threshold}")
        
        # 초기화
        n_samples, p = data.shape
        all_columns = columns.copy()
        current_columns = columns.copy()
        current_indices = list(range(p))
        
        self._log(f"\nInput: {n_samples} samples x {p} descriptors")
        
        # Pass2 클러스터 정보
        pass2_clusters = {}
        if pass2_info and 'representatives' in pass2_info:
            pass2_clusters = pass2_info['representatives']
            self._log(f"Pass2 clusters: {len(pass2_clusters)}")
        
        # 데이터 표준화 (in-place로 메모리 절약)
        self._log("\nStandardizing data...")
        standardized_data = np.zeros_like(data, dtype=np.float32)  # float32로 메모리 절약
        
        for i in range(p):
            col_data = data[:, i]
            valid_mask = ~np.isnan(col_data)
            if valid_mask.sum() > 0:
                mean = np.mean(col_data[valid_mask])
                std = np.std(col_data[valid_mask])
                standardized_data[:, i] = (col_data - mean) / (std + 1e-10)
        
        # 원본 데이터 삭제
        del data
        gc.collect()
        
        # Tracking (최소한만)
        removed_details = []
        clusters = {}
        iteration = 0
        
        self._log(f"\nStarting iterative VIF removal...")
        
        # Main loop
        while True:
            iteration += 1
            
            # 현재 데이터 선택 (view 사용)
            current_data = standardized_data[:, current_indices]
            
            # VIF 계산
            vif_dict = self._compute_vif_values(current_data, current_columns)
            
            # 유효한 VIF만
            valid_vifs = {k: v for k, v in vif_dict.items() if not np.isnan(v)}
            
            if not valid_vifs:
                self._log(f"\nIteration {iteration}: No valid VIF values, stopping")
                break
            
            # 최대 VIF
            max_desc = max(valid_vifs, key=valid_vifs.get)
            max_vif = valid_vifs[max_desc]
            
            # Threshold 체크
            if max_vif < self.vif_threshold:
                self._log(f"\nIteration {iteration}: Max VIF = {max_vif:.2f} < {self.vif_threshold}, done!")
                break
            
            # 제거할 descriptor
            removed_idx_in_current = current_columns.index(max_desc)
            removed_idx_in_original = current_indices[removed_idx_in_current]
            
            if iteration % 10 == 1:  # 1, 11, 21, ...
                self._log(f"\nIteration {iteration}:")
                self._log(f"  Removing: {max_desc} (VIF = {max_vif:.2f})")
                self._log(f"  Remaining: {len(current_columns) - 1}")
            
            # Best survivor 찾기
            surviving_descs = [d for d in current_columns if d != max_desc]
            best_survivor, corr_value = self._find_best_surviving_representative(
                max_desc,
                removed_idx_in_original,
                surviving_descs,
                correlation_matrix,
                all_columns,
                pass2_clusters
            )
            
            if iteration % 10 == 1 and best_survivor:
                self._log(f"  → Assigned to: {best_survivor} (corr={corr_value:.3f})")
            
            # Pass2 클러스터 정보
            pass2_members = []
            if max_desc in pass2_clusters:
                pass2_members = pass2_clusters[max_desc].get('members', [])
            
            # 제거 정보 저장 (필수 정보만)
            detail = {
                'descriptor': max_desc,
                'descriptor_idx': int(removed_idx_in_original),
                'vif_value': float(max_vif),
                'removed_order': iteration,
                'assigned_to': best_survivor,
                'pass2_cluster_size': len(pass2_members)
            }
            
            if best_survivor and corr_value > 0:
                detail['correlation_with_survivor'] = float(corr_value)
            
            # 중요한 경우만 멤버 저장 (메모리 절약)
            if len(pass2_members) > 0 and len(pass2_members) < 1000:
                detail['pass2_cluster_members'] = pass2_members
            
            removed_details.append(detail)
            
            # 클러스터 정보
            clusters[max_desc] = {
                'survivor': best_survivor,
                'vif_value': float(max_vif),
                'removed_order': iteration
            }
            
            # 리스트에서 제거
            current_columns.remove(max_desc)
            current_indices.pop(removed_idx_in_current)
            
            # 주기적 메모리 정리
            if iteration % 50 == 0:
                gc.collect()
                self._log(f"  [Memory cleanup at iteration {iteration}]")
        
        # 최종 결과
        self._log(f"\n" + "="*80)
        self._log(f"Iterative VIF Complete")
        self._log(f"="*80)
        self._log(f"Total iterations: {iteration}")
        self._log(f"Removed: {len(removed_details)} descriptors")
        self._log(f"Remaining: {len(current_columns)} descriptors")
        
        # Checkpoint 구성
        checkpoint = {
            'removed': [d['descriptor'] for d in removed_details],
            'removed_count': len(removed_details),
            'remaining_count': len(current_columns),
            'iterations': iteration,
            'removed_details': removed_details,
            'clusters': clusters,
            'representatives': {},
            'cluster_statistics': self._calculate_cluster_stats(clusters)
        }
        
        # Representatives (역방향 매핑)
        for removed_desc, cluster_info in clusters.items():
            survivor = cluster_info.get('survivor')
            if survivor:
                if survivor not in checkpoint['representatives']:
                    checkpoint['representatives'][survivor] = {
                        'absorbed_descriptors': [],
                        'total_absorbed': 0
                    }
                checkpoint['representatives'][survivor]['absorbed_descriptors'].append(removed_desc)
        
        for surv, info in checkpoint['representatives'].items():
            info['total_absorbed'] = len(info['absorbed_descriptors'])
        
        # Checkpoint 저장
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            checkpoint_file = output_dir / 'pass3_vif.json'
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint, f, indent=2)
            self._log(f"\nCheckpoint saved: {checkpoint_file}")
        
        # 메모리 정리
        del standardized_data
        gc.collect()
        
        return current_columns, checkpoint
    
    def _calculate_cluster_stats(self, clusters: Dict) -> Dict:
        """클러스터 통계"""
        if not clusters:
            return {'mean': 0, 'median': 0, 'min': 0, 'max': 0, 'std': 0}
        
        # 간단하게 removed order만 통계
        return {
            'total_clusters': len(clusters),
            'total_removed': len(clusters)
        }


def run_iterative_vif(
    data: np.ndarray,
    columns: List[str],
    pass2_info: Optional[Dict] = None,
    correlation_matrix: Optional[np.ndarray] = None,
    output_dir: Optional[str] = None,
    vif_threshold: float = 10.0,
    correlation_threshold: float = 0.7,
    verbose: bool = True
) -> Tuple[List[str], Dict]:
    """
    Convenience function
    """
    vif_filter = IterativeVIFFiltering(
        vif_threshold=vif_threshold,
        correlation_threshold=correlation_threshold,
        verbose=verbose
    )
    
    return vif_filter.process(
        data=data,
        columns=columns,
        pass2_info=pass2_info,
        correlation_matrix=correlation_matrix,
        output_dir=Path(output_dir) if output_dir else None
    )


if __name__ == "__main__":
    print("Memory-Efficient Iterative VIF Filtering Module")
