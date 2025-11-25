"""
GPU-accelerated VIF filtering (Memory-Optimized)

Key insight: VIF can be computed from correlation matrix alone,
no need to load full data into GPU memory.

VIF_j = 1 / (1 - R²_j) where R²_j is from regressing X_j on other X_(-j)

Author: Memory Optimization v2.0
"""

import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import json

from molecular_descriptor_toolkit.config.settings import FilteringConfig, IOConfig, SystemConfig
from molecular_descriptor_toolkit.filtering.utils.logging import log

# CPU 버전의 DisjointSet 재사용
from molecular_descriptor_toolkit.filtering.passes.spearman_clustering import DisjointSet


class VIFFilteringPassGPU:
    """
    GPU 가속 VIF (메모리 최적화)
    
    상관행렬만 사용하므로 전체 데이터 불필요
    - 입력: Spearman 상관행렬 G (p × p)
    - 출력: VIF < threshold인 descriptors
    
    VIF_j = 1 / (1 - R²_j)
    R²_j = corr(X_j, X_(-j)) @ inv(corr(X_(-j), X_(-j))) @ corr(X_(-j), X_j)
    """
    
    def __init__(self, filtering_cfg: FilteringConfig, io_cfg: IOConfig, system_cfg: SystemConfig, device: torch.device):
        self.filtering_cfg = filtering_cfg
        self.io_cfg = io_cfg
        self.system_cfg = system_cfg
        self.verbose = system_cfg.verbose
        # vif_threshold not used in Pass 4
        
        self.device = device
        
        if self.verbose and self.device.type == 'cuda':
            gpu_name = torch.cuda.get_device_name(0)
            log(f"✓ VIF using GPU: {gpu_name}", self.verbose)
    
    def _log(self, msg: str):
        if self.verbose:
            log(msg, self.verbose)
    
    def process_from_correlation(self, G: np.ndarray, columns: List[str],
                                  stats: Dict) -> Tuple[List[str], Dict, np.ndarray]:
        """
        상관행렬로부터 VIF 계산 (메모리 효율적)
        
        Args:
            G: (p, p) Spearman 상관행렬
            columns: descriptor 이름들
            stats: 통계 정보
        
        Returns:
            filtered_columns: VIF < threshold인 컬럼들
            info: 제거 정보
            indices: 남은 컬럼의 인덱스
        """
        self._log(f"\n  → Computing VIF from correlation matrix (threshold={self.vif_threshold})...")
        
        p_init = len(columns)
        
        # GPU로 상관행렬 전송
        G_tensor = torch.from_numpy(G).to(self.device, dtype=torch.float32)
        
        # VIF 계산 (반복적 제거)
        remaining_indices = torch.arange(p_init, device=self.device)
        removed_indices = []
        removed_columns = []
        vif_values = {}
        
        iteration = 0
        max_iterations = p_init  # 안전장치
        
        while len(remaining_indices) > 0 and iteration < max_iterations:
            iteration += 1
            
            # 현재 남은 descriptors의 상관행렬
            G_current = G_tensor[remaining_indices][:, remaining_indices]
            p_current = len(remaining_indices)
            
            if p_current == 0:
                break
            
            # 모든 descriptors의 VIF 계산
            vifs = self._compute_vif_batch(G_current)
            
            # 최대 VIF 찾기
            max_vif, max_idx_local = vifs.max(dim=0)
            max_vif_value = max_vif.item()
            
            if max_vif_value < self.vif_threshold:
                # 모든 VIF가 threshold 미만
                break
            
            # 최대 VIF를 가진 descriptor 제거
            max_idx_global = remaining_indices[max_idx_local].item()
            col_name = columns[max_idx_global]
            
            vif_values[col_name] = max_vif_value
            removed_indices.append(max_idx_global)
            removed_columns.append(col_name)
            
            # remaining_indices에서 제거
            mask = torch.ones(len(remaining_indices), dtype=torch.bool, device=self.device)
            mask[max_idx_local] = False
            remaining_indices = remaining_indices[mask]
            
            if self.verbose and iteration % 10 == 0:
                self._log(f"    Iteration {iteration}: removed {len(removed_indices)}, remaining {len(remaining_indices)}")
        
        # 결과 정리
        remaining_indices_cpu = remaining_indices.cpu().numpy()
        filtered_columns = [columns[i] for i in remaining_indices_cpu]
        
        info = {
            'removed': removed_columns,
            'removed_count': len(removed_columns),
            'remaining_count': len(filtered_columns),
            'vif_values': vif_values,
            'iterations': iteration
        }
        
        self._log(f"    Removed {len(removed_columns)} descriptors with VIF ≥ {self.vif_threshold}")
        self._log(f"    Remaining: {len(filtered_columns)} descriptors")
        
        return filtered_columns, info, remaining_indices_cpu
    
    def _compute_vif_batch(self, G: torch.Tensor) -> torch.Tensor:
        """
        배치로 모든 VIF 계산
        
        Args:
            G: (p, p) 상관행렬 (torch.Tensor on GPU)
        
        Returns:
            vifs: (p,) VIF 값들
        """
        p = G.shape[0]
        
        if p == 1:
            return torch.ones(1, device=self.device, dtype=torch.float32)
        
        vifs = torch.zeros(p, device=self.device, dtype=torch.float32)
        
        for j in range(p):
            # j번째 descriptor를 나머지로 회귀
            # R²_j = r_j^T @ G_(-j,-j)^(-1) @ r_j
            # where r_j = corr(X_j, X_(-j))
            
            # 인덱스 생성
            idx_other = torch.cat([torch.arange(j, device=self.device),
                                   torch.arange(j+1, p, device=self.device)])
            
            if len(idx_other) == 0:
                vifs[j] = 1.0
                continue
            
            # r_j: (p-1,)
            r_j = G[j, idx_other]
            
            # G_other: (p-1, p-1)
            G_other = G[idx_other][:, idx_other]
            
            try:
                # G_other^(-1) @ r_j
                # Cholesky 분해로 안정적 계산
                L = torch.linalg.cholesky(G_other + 1e-6 * torch.eye(len(idx_other), 
                                                                      device=self.device,
                                                                      dtype=G.dtype))
                # L @ L^T @ x = r_j
                # L^T @ x = L^(-1) @ r_j
                y = torch.linalg.solve_triangular(L, r_j.unsqueeze(1), upper=False)
                x = torch.triangular_solve(y, L.t(), upper=True).solution
                
                # R² = r_j^T @ x
                r_squared = (r_j @ x).item()
                r_squared = max(0.0, min(r_squared, 0.9999))  # 클리핑
                
                # VIF = 1 / (1 - R²)
                vif = 1.0 / (1.0 - r_squared)
                vifs[j] = vif
                
            except Exception as e:
                # 수치적 불안정 시 높은 VIF 할당
                vifs[j] = 1e6
        
        return vifs


class NonlinearDetectionPassGPU:
    """
    GPU 가속 비선형 필터링 (메모리 최적화)
    
    HSIC + RDC 행렬은 이미 청크 방식으로 계산되므로,
    여기서는 필터링 로직만 GPU에서 수행
    
    Note: 이 클래스는 기존 인터페이스와 호환성 유지
    """
    
    def __init__(self, filtering_cfg: FilteringConfig, io_cfg: IOConfig, system_cfg: SystemConfig, device: torch.device):
        self.filtering_cfg = filtering_cfg
        self.io_cfg = io_cfg
        self.system_cfg = system_cfg
        self.verbose = system_cfg.verbose
        
        # Use filtering_cfg attributes
        self.threshold = filtering_cfg.nonlinear_threshold
        self.w_hsic = filtering_cfg.w_hsic
        self.w_rdc = filtering_cfg.w_rdc
        
        self.device = device
        
        if self.verbose and self.device.type == 'cuda':
            gpu_name = torch.cuda.get_device_name(0)
            log(f"✓ Nonlinear filtering using GPU: {gpu_name}", self.verbose)
    
    def _log(self, msg: str):
        if self.verbose:
            log(msg, self.verbose)
    
    def process(self, data: np.ndarray, columns: List[str],
                G_hsic: np.ndarray, G_rdc: np.ndarray, stats: Dict,
                graph_builder: Any = None, leiden_clustering: Any = None) -> Tuple[List[str], Dict, np.ndarray]:
        """
        HSIC + RDC 기반 비선형 클러스터링 (GPU 가속)
        
        Args:
            data: (n, p) 데이터 행렬 (Pass4에서는 사용 안 함, 호환성 유지)
            columns: descriptor 이름들
            G_hsic: (p, p) HSIC 행렬
            G_rdc: (p, p) RDC 행렬
            stats: 통계 정보
            graph_builder: 그래프 빌더 (사용 안 함, 호환성 유지)
            leiden_clustering: Leiden 클러스터링 (사용 안 함, 호환성 유지)
        
        Returns:
            filtered_columns: 대표 descriptors
            info: 제거 정보
            indices: 남은 컬럼의 인덱스
        """
        self._log(f"\n  → Nonlinear filtering (threshold={self.threshold}, w_hsic={self.w_hsic}, w_rdc={self.w_rdc})...")
        
        p = len(columns)
        
        # GPU로 전송
        H_tensor = torch.from_numpy(G_hsic).to(self.device, dtype=torch.float32)
        R_tensor = torch.from_numpy(G_rdc).to(self.device, dtype=torch.float32)
        
        # 가중 결합
        S = self.w_hsic * H_tensor + self.w_rdc * R_tensor
        S = torch.clamp(S, -1.0, 1.0)
        torch.diagonal(S).fill_(1.0)
        
        # Threshold로 엣지 생성
        edges = (torch.abs(S) >= self.threshold).cpu().numpy()
        np.fill_diagonal(edges, False)
        
        # CPU에서 클러스터링 (그래프 알고리즘)
        clusters = self._find_clusters_cpu(edges, p)
        
        # 각 클러스터에서 대표 선택
        representative_indices = []
        removed_columns = []
        
        for cluster in clusters:
            cluster_list = list(cluster)
            if len(cluster_list) == 1:
                representative_indices.append(cluster_list[0])
            else:
                # 클러스터 내에서 평균 연결도가 가장 높은 descriptor 선택
                cluster_S = S[cluster_list][:, cluster_list]
                avg_similarity = cluster_S.abs().sum(dim=1) / (len(cluster) - 1)
                rep_idx_local = avg_similarity.argmax().item()
                rep_idx_global = cluster_list[rep_idx_local]
                representative_indices.append(rep_idx_global)
                
                # 나머지는 제거
                for idx in cluster:
                    if idx != rep_idx_global:
                        removed_columns.append(columns[idx])
        
        representative_indices = sorted(representative_indices)
        filtered_columns = [columns[i] for i in representative_indices]
        
        # 클러스터 정보 생성 (기존 인터페이스 호환)
        clusters_info = []
        for cluster in clusters:
            cluster_list = sorted(list(cluster))
            clusters_info.append({
                'members': [columns[i] for i in cluster_list],
                'size': len(cluster_list)
            })
        
        representatives_info = [columns[i] for i in representative_indices]
        
        info = {
            'removed': removed_columns,
            'removed_count': len(removed_columns),
            'remaining_count': len(filtered_columns),
            'n_clusters': len(clusters),
            'threshold': self.threshold,
            'clusters': clusters_info,
            'representatives': representatives_info
        }
        
        self._log(f"    Found {len(clusters)} clusters")
        self._log(f"    Removed {len(removed_columns)} descriptors")
        self._log(f"    Remaining: {len(filtered_columns)} descriptors")
        
        return filtered_columns, info, np.array(representative_indices)
    
    def _find_clusters_cpu(self, edges: np.ndarray, p: int) -> List[set]:
        """
        Union-Find로 클러스터 찾기 (CPU)
        
        Args:
            edges: (p, p) boolean adjacency matrix
            p: descriptor 수
        
        Returns:
            clusters: List of sets
        """
        ds = DisjointSet(range(p))
        
        # 엣지 연결
        for i in range(p):
            for j in range(i+1, p):
                if edges[i, j]:
                    ds.union(i, j)
        
        # 클러스터 추출
        clusters_dict = {}
        for i in range(p):
            root = ds.find(i)
            if root not in clusters_dict:
                clusters_dict[root] = set()
            clusters_dict[root].add(i)
        
        clusters = list(clusters_dict.values())
        
        return clusters
