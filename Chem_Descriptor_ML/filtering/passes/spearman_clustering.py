"""
Advanced filtering modules for Pass2-4
- Pass2: Spearman-based linear correlation clustering with transitive closure
- Pass3: VIF-based multicollinearity removal
- Pass4: Normalized HSIC + RDC nonlinear relationship detection

Author: Refactored Pipeline v3.0
"""

import numpy as np
# import pandas as pd  # Unused - removed
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from scipy.stats import spearmanr
import json

from Chem_Descriptor_ML.config.settings import PipelineConfig
from Chem_Descriptor_ML.filtering.passes.seed_manager import SeedManager
from Chem_Descriptor_ML.filtering.utils.logging import log


# ============================================================================
# DisjointSet (Union-Find) 직접 구현
# ============================================================================

class DisjointSet:
    """
    Union-Find (Disjoint Set Union) 자료구조
    
    경로 압축(path compression)과 랭크 기반 합치기(union by rank)를 사용하여
    거의 상수 시간에 집합 연산을 수행합니다.
    
    Usage:
        ds = DisjointSet([0, 1, 2, 3, 4])
        ds.merge(0, 1)  # 0과 1을 같은 집합으로
        ds.merge(2, 3)  # 2와 3을 같은 집합으로
        root_0 = ds[0]  # 0의 대표 원소
        
    Performance:
        - find: O(α(n)) - 거의 O(1)
        - merge: O(α(n)) - 거의 O(1)
        여기서 α(n)은 역 아커만 함수 (실용적으로 상수)
    """
    
    def __init__(self, elements):
        """
        Args:
            elements: 초기 원소들 (iterable)
        """
        self.parent = {x: x for x in elements}
        self.rank = {x: 0 for x in elements}
    
    def find(self, x):
        """
        원소 x가 속한 집합의 대표 원소(root) 찾기
        경로 압축 적용으로 최적화
        
        Args:
            x: 찾을 원소
            
        Returns:
            x가 속한 집합의 대표 원소
        """
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # 경로 압축
        return self.parent[x]
    
    def merge(self, x, y):
        """
        x가 속한 집합과 y가 속한 집합을 합치기
        랭크 기반 합치기로 최적화
        
        Args:
            x, y: 합칠 두 원소
            
        Returns:
            bool: 합쳐졌으면 True, 이미 같은 집합이면 False
        """
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x == root_y:
            return False  # 이미 같은 집합
        
        # 랭크 기반 합치기
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1
        
        return True
    
    def __getitem__(self, x):
        """
        편의 문법: ds[x]로 find(x) 호출
        
        Args:
            x: 원소
            
        Returns:
            x가 속한 집합의 대표 원소
        """
        return self.find(x)


# ============================================================================
# Pass2: Spearman Correlation Clustering with Transitive Closure
# ============================================================================

class SpearmanClusteringPass:
    """
    Pass2: Spearman 상관계수 기반 선형 상관관계 클러스터링
    
    특징:
    - |ρ| ≥ threshold (예: 0.95)인 쌍들을 찾아 군집화
    - Transitive closure를 통해 간접 상관도 그룹화
    - 각 클러스터에서 품질이 가장 좋은 대표 descriptor 1개만 유지
    - 품질 기준: 결측률↓ → 분산↑ → 비이진성↑ → 도메인의미↑ → 중앙성↑
    """
    
    def __init__(self, config: PipelineConfig, verbose: bool = True):
        self.config = config
        self.verbose = verbose
        self.spearman_threshold = getattr(config, 'spearman_threshold', 0.95)
    
    def _log(self, msg: str):
        if self.verbose:
            log(msg, self.verbose)
    
    def _compute_quality_score(self, 
                               idx: int,
                               missing_rates: np.ndarray,
                               variances: np.ndarray,
                               centrality: Optional[np.ndarray] = None) -> float:
        """
        Descriptor 품질 점수 계산
        
        기준: 결측률↓ → 분산↑ → 중앙성↑
        
        Note: 비이진성 체크는 메모리 효율성을 위해 제거됨 (data 로드 불필요)
        """
        # 결측률 (낮을수록 좋음)
        missing_score = 1.0 - missing_rates[idx]  # 0~1, 높을수록 좋음
        
        # 분산 (높을수록 좋음, 정규화)
        variance_score = variances[idx] / (np.max(variances) + 1e-10)
        
        # 중앙성 (높을수록 좋음, centrality가 제공되면 사용)
        if centrality is not None:
            centrality_score = centrality[idx] / (np.max(centrality) + 1e-10)
        else:
            centrality_score = 0.5  # 기본값
        
        # 가중 조합 (결측률과 분산에 높은 가중치)
        quality = (
            5.0 * missing_score +     # 결측률 가장 중요
            3.0 * variance_score +    # 분산 중요
            1.0 * centrality_score    # 중앙성
        )
        
        return quality
    
    def _find_transitive_clusters(self, 
                                   corr_matrix: np.ndarray,
                                   threshold: float) -> List[Set[int]]:
        """
        Transitive closure를 이용한 군집 찾기
        
        X₁↔X₂ (0.98), X₂↔X₃ (0.97), X₁↔X₃ (0.8) → {X₁, X₂, X₃} 하나의 클러스터
        
        Args:
            corr_matrix: Spearman 상관계수 행렬 (p x p)
            threshold: 상관계수 임계값
            
        Returns:
            List of clusters (each cluster is a set of indices)
        """
        p = corr_matrix.shape[0]
        
        # Disjoint Set (Union-Find) 자료구조 사용
        ds = DisjointSet(range(p))
        
        # |ρ| ≥ threshold인 모든 쌍을 union
        for i in range(p):
            for j in range(i+1, p):
                if abs(corr_matrix[i, j]) >= threshold:
                    ds.merge(i, j)
        
        # 클러스터별로 그룹화
        cluster_dict = {}
        for i in range(p):
            root = ds[i]
            if root not in cluster_dict:
                cluster_dict[root] = set()
            cluster_dict[root].add(i)
        
        # 크기 2 이상인 클러스터만 반환 (singleton 제외)
        clusters = [cluster for cluster in cluster_dict.values() if len(cluster) >= 2]
        
        return clusters
    
    def process(self, 
                data: Optional[np.ndarray],
                columns: List[str],
                G_spearman: np.ndarray,
                stats: Dict) -> Tuple[List[str], Dict, np.ndarray]:
        """
        Pass2 실행: Spearman 기반 군집화 및 대표 선택
        
        Args:
            data: DEPRECATED - kept for compatibility but not used (pass None)
            columns: Descriptor 이름 리스트
            G_spearman: Spearman 유사도 행렬 (p x p)
            stats: Pass1의 통계 정보 (missing_ratio, variance 포함)
            
        Returns:
            - remaining_columns: 남은 descriptor 이름 리스트
            - clustering_info: 클러스터링 정보 (저장용)
            - remaining_indices: 남은 descriptor의 인덱스
        """
        self._log("\n" + "="*60)
        self._log("Pass2: Spearman Correlation Clustering")
        self._log("="*60)
        self._log(f"Threshold: |ρ| ≥ {self.spearman_threshold}")
        
        p = len(columns)
        
        # Spearman 상관계수 행렬 (G_spearman은 유사도이므로 그대로 사용)
        corr_matrix = G_spearman.copy()
        
        # Transitive closure를 이용한 군집 찾기
        clusters = self._find_transitive_clusters(corr_matrix, self.spearman_threshold)
        
        self._log(f"Found {len(clusters)} high-correlation clusters")
        
        # Pass1의 stats에서 결측률 및 분산 가져오기 (메모리 효율적)
        missing_rates = stats['missing_ratio']
        variances = stats['variance']
        
        # 클러스터별로 대표 선택
        removed_indices = set()
        clustering_info = {
            'clusters': [],
            'representatives': {},
            'removed': {}
        }
        
        for cluster_idx, cluster in enumerate(clusters):
            cluster_list = sorted(list(cluster))
            
            # 품질 점수 계산 (data 불필요)
            quality_scores = {
                idx: self._compute_quality_score(idx, missing_rates, variances)
                for idx in cluster_list
            }
            
            # 가장 품질이 높은 descriptor를 대표로 선택
            representative = max(quality_scores, key=quality_scores.get)
            
            # 클러스터 정보 저장
            cluster_info = {
                'cluster_id': cluster_idx,
                'size': len(cluster_list),
                'representative': columns[representative],
                'representative_idx': int(representative),
                'members': [columns[idx] for idx in cluster_list],
                'quality_scores': {columns[idx]: float(quality_scores[idx]) 
                                  for idx in cluster_list},
                'correlation_matrix': corr_matrix[np.ix_(cluster_list, cluster_list)].tolist()
            }
            clustering_info['clusters'].append(cluster_info)
            clustering_info['representatives'][columns[representative]] = cluster_info
            
            # 대표가 아닌 나머지는 제거
            for idx in cluster_list:
                if idx != representative:
                    removed_indices.add(idx)
                    clustering_info['removed'][columns[idx]] = columns[representative]
            
            self._log(f"  Cluster {cluster_idx}: {len(cluster_list)} members, "
                     f"representative = {columns[representative]}")
        
        # 남은 descriptor 선택
        remaining_indices = np.array([i for i in range(p) if i not in removed_indices])
        remaining_columns = [columns[i] for i in remaining_indices]
        
        self._log(f"\nTotal descriptors: {p}")
        self._log(f"Removed (redundant): {len(removed_indices)}")
        self._log(f"Remaining: {len(remaining_columns)}")
        
        return remaining_columns, clustering_info, remaining_indices


# ============================================================================
# Pass3: VIF-based Multicollinearity Removal
# ============================================================================

class VIFFilteringPass:
    """
    Pass3: VIF 기반 다중공선성 제거
    
    특징:
    - VIF ≥ threshold (예: 10)인 변수들을 클러스터링
    - X₁↔X₂↔X₃ 같은 multi-way 선형 종속 감지
    - 각 클러스터에서 VIF 최소이면서 분산 최대인 변수 1개만 유지
    """
    
    def __init__(self, config: PipelineConfig, verbose: bool = True):
        self.config = config
        self.verbose = verbose
        self.vif_threshold = getattr(config, 'vif_threshold', 10.0)
    
    def _log(self, msg: str):
        if self.verbose:
            log(msg, self.verbose)
    
    def _compute_vif_values(self, data: np.ndarray) -> np.ndarray:
        """
        각 변수의 VIF 값 계산
        
        VIF 계산 시 상수항을 추가하여 정확도 향상
        VIF = 1 / (1 - R²)
        여기서 R²는 해당 변수를 다른 변수들로 회귀했을 때의 결정계수
        
        Args:
            data: 표준화된 데이터 (n_samples x p)
            
        Returns:
            VIF 값 배열 (p,)
        """
        n_samples, p = data.shape
        vif_values = np.zeros(p)
        
        for i in range(p):
            try:
                # NaN 처리: 유효한 행만 사용
                mask = ~np.isnan(data).any(axis=1)
                clean_data = data[mask]
                
                if clean_data.shape[0] < p + 2:  # 상수항 추가로 p+2 필요
                    vif_values[i] = np.nan
                    continue
                
                # 상수항 추가 (VIF 계산의 정확도 향상)
                data_with_const = add_constant(clean_data)
                vif_values[i] = variance_inflation_factor(data_with_const, i + 1)  # i+1: 상수항 제외
            except Exception as e:
                self._log(f"    Warning: VIF calculation failed for variable {i}: {e}")
                vif_values[i] = np.nan
        
        return vif_values
    
    def _find_vif_clusters(self, data: np.ndarray, vif_values: np.ndarray) -> List[Set[int]]:
        """
        VIF 값이 높은 변수들을 클러스터링
        
        Strategy:
        - VIF ≥ threshold인 변수들을 선택
        - 이들 간의 상관관계를 분석하여 연결된 클러스터 형성
        - Transitive closure 사용
        
        Args:
            data: 표준화된 데이터
            vif_values: VIF 값 배열
            
        Returns:
            List of clusters
        """
        # VIF가 threshold 이상인 변수들
        high_vif_indices = np.where(vif_values >= self.vif_threshold)[0]
        
        if len(high_vif_indices) == 0:
            return []
        
        # 이들 간의 상관계수 행렬 계산
        high_vif_data = data[:, high_vif_indices]
        mask = ~np.isnan(high_vif_data).any(axis=1)
        clean_data = high_vif_data[mask]
        
        if clean_data.shape[0] < 2:
            return []
        
        # Spearman 상관계수 계산
        corr_matrix = np.zeros((len(high_vif_indices), len(high_vif_indices)))
        for i in range(len(high_vif_indices)):
            for j in range(i, len(high_vif_indices)):
                if i == j:
                    corr_matrix[i, j] = 1.0
                else:
                    try:
                        rho, _ = spearmanr(clean_data[:, i], clean_data[:, j])
                        corr_matrix[i, j] = corr_matrix[j, i] = abs(rho)
                    except:
                        corr_matrix[i, j] = corr_matrix[j, i] = 0.0
        
        # Disjoint Set으로 클러스터링 (상관계수 0.7 이상인 쌍들 연결)
        ds = DisjointSet(range(len(high_vif_indices)))
        correlation_threshold = 0.7
        
        for i in range(len(high_vif_indices)):
            for j in range(i+1, len(high_vif_indices)):
                if corr_matrix[i, j] >= correlation_threshold:
                    ds.merge(i, j)
        
        # 클러스터 그룹화 (원래 인덱스로 변환)
        cluster_dict = {}
        for i in range(len(high_vif_indices)):
            root = ds[i]
            if root not in cluster_dict:
                cluster_dict[root] = set()
            cluster_dict[root].add(high_vif_indices[i])
        
        clusters = [cluster for cluster in cluster_dict.values() if len(cluster) >= 2]
        
        return clusters
    
    def process(self,
                data: np.ndarray,
                columns: List[str],
                stats: Dict) -> Tuple[List[str], Dict, np.ndarray]:
        """
        Pass3 실행: VIF 기반 다중공선성 제거
        
        Args:
            data: 데이터 (n_samples x p)
            columns: Descriptor 이름 리스트
            stats: 통계 정보
            
        Returns:
            - remaining_columns: 남은 descriptor 이름
            - vif_info: VIF 정보
            - remaining_indices: 남은 인덱스
        """
        self._log("\n" + "="*60)
        self._log("Pass3: VIF-based Multicollinearity Removal")
        self._log("="*60)
        self._log(f"Threshold: VIF ≥ {self.vif_threshold}")
        
        p = len(columns)
        
        # 데이터 표준화 (VIF 계산을 위해)
        standardized_data = np.zeros_like(data)
        for i in range(p):
            col_data = data[:, i]
            valid_mask = ~np.isnan(col_data)
            if valid_mask.sum() > 0:
                mean = np.mean(col_data[valid_mask])
                std = np.std(col_data[valid_mask])
                standardized_data[:, i] = (col_data - mean) / (std + 1e-10)
        
        # VIF 값 계산
        self._log("Computing VIF values...")
        vif_values = self._compute_vif_values(standardized_data)
        
        # VIF 클러스터 찾기
        clusters = self._find_vif_clusters(standardized_data, vif_values)
        
        self._log(f"Found {len(clusters)} high-VIF clusters")
        
        # 각 descriptor의 분산 계산
        variances = np.zeros(p)
        for i in range(p):
            col_data = data[:, i]
            valid_data = col_data[~np.isnan(col_data)]
            if len(valid_data) > 0:
                variances[i] = np.var(valid_data)
        
        # 클러스터별로 대표 선택
        removed_indices = set()
        vif_info = {
            'clusters': [],
            'representatives': {},
            'removed': {},
            'all_vif_values': {columns[i]: float(vif_values[i]) for i in range(p)}
        }
        
        for cluster_idx, cluster in enumerate(clusters):
            cluster_list = sorted(list(cluster))
            
            # VIF 최소이면서 분산 최대인 변수 선택
            # 정규화된 점수 사용
            scores = {}
            for idx in cluster_list:
                if not np.isnan(vif_values[idx]):
                    # VIF는 낮을수록 좋음 (역수 사용)
                    vif_score = 1.0 / (vif_values[idx] + 1.0)
                    # 분산은 높을수록 좋음
                    var_score = variances[idx] / (np.max(variances[cluster_list]) + 1e-10)
                    # 가중 조합
                    scores[idx] = 2.0 * vif_score + 1.0 * var_score
                else:
                    scores[idx] = -np.inf
            
            representative = max(scores, key=scores.get)
            
            # 클러스터 정보 저장
            cluster_info = {
                'cluster_id': cluster_idx,
                'size': len(cluster_list),
                'representative': columns[representative],
                'representative_idx': int(representative),
                'members': [columns[idx] for idx in cluster_list],
                'vif_values': {columns[idx]: float(vif_values[idx]) for idx in cluster_list},
                'variances': {columns[idx]: float(variances[idx]) for idx in cluster_list}
            }
            vif_info['clusters'].append(cluster_info)
            vif_info['representatives'][columns[representative]] = cluster_info
            
            # 대표가 아닌 나머지 제거
            for idx in cluster_list:
                if idx != representative:
                    removed_indices.add(idx)
                    vif_info['removed'][columns[idx]] = columns[representative]
            
            self._log(f"  Cluster {cluster_idx}: {len(cluster_list)} members, "
                     f"representative = {columns[representative]} "
                     f"(VIF={vif_values[representative]:.2f})")
        
        # 남은 descriptor 선택
        remaining_indices = np.array([i for i in range(p) if i not in removed_indices])
        remaining_columns = [columns[i] for i in remaining_indices]
        
        self._log(f"\nTotal descriptors: {p}")
        self._log(f"Removed (multicollinear): {len(removed_indices)}")
        self._log(f"Remaining: {len(remaining_columns)}")
        
        return remaining_columns, vif_info, remaining_indices


# ============================================================================
# Pass4: Normalized HSIC + RDC Nonlinear Detection
# ============================================================================

class NonlinearDetectionPass:
    """
    Pass4: Normalized HSIC + RDC 기반 비선형 관계 탐지
    
    특징:
    - Normalized HSIC (0~1 범위로 스케일 맞춤)
    - RDC와 가중 조합: M_ij = w_HSIC * nHSIC_ij + w_RDC * RDC_ij
    - M_ij ≥ threshold (예: 0.75)인 쌍들을 Leiden 군집화
    - 각 클러스터에서 centrality + variance 기준으로 대표 선택
    """
    
    def __init__(self, config: PipelineConfig, verbose: bool = True):
        self.config = config
        self.verbose = verbose
        self.nonlinear_threshold = getattr(config, 'nonlinear_threshold', 0.75)
        
        # 가중치 (v3.0: HSIC, RDC 2개만)
        # config.weights는 (w_hsic, w_rdc) 형태
        weights = config.weights
        total_weight = weights[0] + weights[1]
        
        if total_weight > 0:
            self.w_hsic = weights[0] / total_weight
            self.w_rdc = weights[1] / total_weight
        else:
            # 둘 다 0이면 기본값
            self.w_hsic = 0.5
            self.w_rdc = 0.5
    
    def _log(self, msg: str):
        if self.verbose:
            log(msg, self.verbose)
    
    def _normalize_hsic(self, G_hsic: np.ndarray) -> np.ndarray:
        """
        HSIC를 0~1 범위로 정규화
        
        HSIC는 kernel-based measure이므로 값의 범위가 다양함
        Min-max normalization 또는 diagonal normalization 사용
        """
        # Diagonal normalization: nHSIC_ij = HSIC_ij / sqrt(HSIC_ii * HSIC_jj)
        # 이는 RDC와 유사한 정규화 방식
        
        p = G_hsic.shape[0]
        normalized = np.zeros_like(G_hsic)
        
        for i in range(p):
            for j in range(p):
                if i == j:
                    normalized[i, j] = 1.0
                else:
                    denom = np.sqrt(max(G_hsic[i, i], 1e-10) * max(G_hsic[j, j], 1e-10))
                    normalized[i, j] = G_hsic[i, j] / denom
        
        # 0~1 범위로 클리핑
        normalized = np.clip(normalized, 0.0, 1.0)
        
        return normalized
    
    def _compute_centrality(self, similarity_matrix: np.ndarray) -> np.ndarray:
        """
        각 노드의 중앙성 계산 (degree centrality)
        
        Args:
            similarity_matrix: 유사도 행렬 (p x p)
            
        Returns:
            Centrality 점수 (p,)
        """
        # 각 노드의 평균 유사도를 중앙성으로 사용
        centrality = np.mean(similarity_matrix, axis=1)
        return centrality
    
    def process(self,
                data: np.ndarray,
                columns: List[str],
                G_hsic: np.ndarray,
                G_rdc: np.ndarray,
                stats: Dict,
                graph_builder,
                leiden_clustering) -> Tuple[List[str], Dict, np.ndarray]:
        """
        Pass4 실행: 비선형 관계 탐지 및 군집화
        
        Args:
            data: 데이터 (n_samples x p)
            columns: Descriptor 이름
            G_hsic: HSIC 유사도 행렬
            G_rdc: RDC 유사도 행렬
            stats: 통계 정보
            graph_builder: GraphBuilder 인스턴스
            leiden_clustering: LeidenClustering 인스턴스
            
        Returns:
            - remaining_columns: 남은 descriptor 이름
            - nonlinear_info: 비선형 관계 정보
            - remaining_indices: 남은 인덱스
        """
        self._log("\n" + "="*60)
        self._log("Pass4: Nonlinear Relationship Detection")
        self._log("="*60)
        self._log(f"Weights: HSIC={self.w_hsic:.2f}, RDC={self.w_rdc:.2f}")
        self._log(f"Threshold: M_ij ≥ {self.nonlinear_threshold}")
        
        p = len(columns)
        
        # HSIC 정규화
        self._log("Normalizing HSIC to [0, 1] range...")
        G_hsic_normalized = self._normalize_hsic(G_hsic)
        
        # 가중 조합
        G_combined = self.w_hsic * G_hsic_normalized + self.w_rdc * G_rdc
        
        # 임계값 이상인 쌍만 유지 (sparse)
        G_filtered = G_combined.copy()
        G_filtered[G_filtered < self.nonlinear_threshold] = 0.0
        
        # k-NN 그래프 구성 및 Leiden 군집화
        self._log("Building k-NN graph and clustering...")
        graph = graph_builder.build_knn_graph(G_filtered)
        clusters, consensus_matrix = leiden_clustering.cluster(graph)
        
        # 클러스터별로 그룹화
        cluster_dict = {}
        for i, cluster_id in enumerate(clusters):
            if cluster_id not in cluster_dict:
                cluster_dict[cluster_id] = []
            cluster_dict[cluster_id].append(i)
        
        # 크기 2 이상인 클러스터만 처리
        large_clusters = [indices for indices in cluster_dict.values() if len(indices) >= 2]
        
        self._log(f"Found {len(large_clusters)} nonlinear relationship clusters")
        
        # 각 descriptor의 분산 및 중앙성 계산
        variances = np.zeros(p)
        for i in range(p):
            col_data = data[:, i]
            valid_data = col_data[~np.isnan(col_data)]
            if len(valid_data) > 0:
                variances[i] = np.var(valid_data)
        
        centrality = self._compute_centrality(G_combined)
        
        # 클러스터별로 대표 선택
        removed_indices = set()
        nonlinear_info = {
            'clusters': [],
            'representatives': {},
            'removed': {},
            'combined_similarity': G_combined.tolist()
        }
        
        for cluster_idx, cluster_indices in enumerate(large_clusters):
            # Centrality + variance 기준으로 점수 계산
            scores = {}
            for idx in cluster_indices:
                # 정규화된 점수
                cent_score = centrality[idx] / (np.max(centrality[cluster_indices]) + 1e-10)
                var_score = variances[idx] / (np.max(variances[cluster_indices]) + 1e-10)
                scores[idx] = 1.5 * cent_score + 1.0 * var_score
            
            representative = max(scores, key=scores.get)
            
            # 클러스터 정보 저장
            cluster_info = {
                'cluster_id': cluster_idx,
                'size': len(cluster_indices),
                'representative': columns[representative],
                'representative_idx': int(representative),
                'members': [columns[idx] for idx in cluster_indices],
                'centrality': {columns[idx]: float(centrality[idx]) for idx in cluster_indices},
                'variances': {columns[idx]: float(variances[idx]) for idx in cluster_indices},
                'similarity_matrix': G_combined[np.ix_(cluster_indices, cluster_indices)].tolist()
            }
            nonlinear_info['clusters'].append(cluster_info)
            nonlinear_info['representatives'][columns[representative]] = cluster_info
            
            # 대표가 아닌 나머지 제거
            for idx in cluster_indices:
                if idx != representative:
                    removed_indices.add(idx)
                    nonlinear_info['removed'][columns[idx]] = columns[representative]
            
            self._log(f"  Cluster {cluster_idx}: {len(cluster_indices)} members, "
                     f"representative = {columns[representative]}")
        
        # 남은 descriptor 선택
        remaining_indices = np.array([i for i in range(p) if i not in removed_indices])
        remaining_columns = [columns[i] for i in remaining_indices]
        
        self._log(f"\nTotal descriptors: {p}")
        self._log(f"Removed (nonlinear redundant): {len(removed_indices)}")
        self._log(f"Remaining: {len(remaining_columns)}")
        
        return remaining_columns, nonlinear_info, remaining_indices


# ============================================================================
# Canonical pass aliases (v3.0)
# ============================================================================
Pass2SpearmanClustering = SpearmanClusteringPass
Pass3VIFFilter = VIFFilteringPass
Pass4NonlinearDetection = NonlinearDetectionPass
