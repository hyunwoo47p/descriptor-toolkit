"""
k-NN 그래프 구성 및 Leiden 클러스터링
"""

import numpy as np
import igraph as ig
import leidenalg as la
from typing import Tuple
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster

from descriptor_pipeline.config.settings import PipelineConfig
from descriptor_pipeline.core.seed_manager import SeedManager
from descriptor_pipeline.utils.logging import log


class GraphBuilder:
    """k-NN 그래프 구성"""
    
    def __init__(self, config: PipelineConfig, verbose: bool = True):
        self.config = config
        self.verbose = verbose
    
    def _log(self, msg: str):
        log(msg, self.verbose)
    
    def build_knn_graph(self, S: np.ndarray) -> ig.Graph:
        """
        k-NN graph with correct edge-weight ordering
        
        Args:
            S: 유사도 행렬 (p x p)
            
        Returns:
            igraph.Graph: k-NN 그래프
        """
        self._log(f"Building k-NN graph (k={self.config.topk})...")
        
        p = S.shape[0]
        k = self.config.topk
        edge_list = []
        weight_list = []
        seen = set()
        
        for i in range(p):
            idx = np.argpartition(S[i], -(k + 1))[-(k + 1):]
            idx = idx[idx != i]
            idx = idx[np.argsort(S[i, idx])[::-1]][:k]
            
            for j in idx:
                a, b = (i, int(j)) if i < j else (int(j), i)
                key = (a, b)
                
                if key not in seen:
                    seen.add(key)
                    edge_list.append(key)
                    weight_list.append(float(S[a, b]))
        
        g = ig.Graph()
        g.add_vertices(p)
        
        if edge_list:
            g.add_edges(edge_list)
            g.es['weight'] = weight_list
        
        self._log(f"  Edges: {len(edge_list):,}")
        
        return g
    
    def fuse_similarities(self, G_hsic: np.ndarray, G_rdc: np.ndarray) -> np.ndarray:
        """
        유사도 행렬 융합 (v3.0: HSIC + RDC only, 강제 정규화)
        
        Note: v3.0에서는 Spearman이 Pass2에서 필터링으로만 사용되고
        Pass4 fusion에는 HSIC와 RDC만 사용됩니다.
        
        Args:
            G_hsic: HSIC 유사도 행렬
            G_rdc: RDC 유사도 행렬
            
        Returns:
            S: 융합된 유사도 행렬
        """
        self._log("Fusing similarities (v3.0: HSIC + RDC, forced normalization)...")
        
        w_hsic, w_rdc = self.config.weights
        
        # 정규화
        sum_w = w_hsic + w_rdc
        if abs(sum_w - 1.0) > 1e-6:
            self._log(f"  ⚠️ Weights sum = {sum_w:.6f}, normalizing to 1.0")
            w_hsic, w_rdc = w_hsic/sum_w, w_rdc/sum_w
        
        S = w_hsic * G_hsic + w_rdc * G_rdc
        np.fill_diagonal(S, 1.0)
        
        self._log(f"  Normalized weights: HSIC={w_hsic:.3f}, RDC={w_rdc:.3f}")
        self._log(f"  Consensus range: [{S.min():.3f}, {S.max():.3f}]")
        self._log(f"  Mean: {S[~np.eye(len(S), dtype=bool)].mean():.3f}")
        
        return S


class LeidenClustering:
    """Leiden 클러스터링 with consensus"""
    
    def __init__(self, config: PipelineConfig, seed_mgr: SeedManager, 
                 verbose: bool = True):
        self.config = config
        self.seed_mgr = seed_mgr
        self.verbose = verbose
    
    def _log(self, msg: str):
        log(msg, self.verbose)
    
    def cluster(self, g: ig.Graph) -> Tuple[np.ndarray, np.ndarray]:
        """
        Leiden clustering with consensus
        
        Args:
            g: igraph.Graph
            
        Returns:
            (labels, consensus_matrix): 클러스터 레이블과 consensus 행렬
        """
        self._log(f"Leiden clustering (consensus={self.config.n_consensus})...")
        
        if g.ecount() == 0:
            return np.arange(g.vcount()), np.eye(g.vcount())
        
        p = g.vcount()
        
        # 메모리 경고
        est_memory_gb = p * p * 8 / (1024**3)
        if p > self.config.consensus_memory_limit_p:
            self._log(f"  ⚠️ Warning: p={p} > {self.config.consensus_memory_limit_p}, "
                     f"consensus = {est_memory_gb:.2f} GB")
            self._log(f"  Consider reducing n_consensus or using sparse consensus")
        
        consensus = np.zeros((p, p))
        
        leiden_seeds = self.seed_mgr.get_seeds('statistics', self.config.n_consensus)
        
        for iteration in range(self.config.n_consensus):
            part = la.find_partition(
                g, 
                la.CPMVertexPartition,
                resolution_parameter=self.config.leiden_resolution,
                weights=g.es['weight'],
                seed=leiden_seeds[iteration]
            )
            labels = np.array(part.membership)
            
            for i in range(p):
                for j in range(i+1, p):
                    if labels[i] == labels[j]:
                        consensus[i, j] += 1
                        consensus[j, i] += 1
            
            if (iteration + 1) % 3 == 0 and self.verbose:
                self._log(f"  Iteration {iteration+1}/{self.config.n_consensus}, "
                         f"clusters: {len(set(labels))}")
        
        # Final clustering
        consensus_dist = 1 - (consensus / self.config.n_consensus)
        condensed = squareform(consensus_dist, checks=False)
        Z = linkage(condensed, method='average')
        final_labels = fcluster(Z, t=0.5, criterion='distance')
        
        n_clusters = len(set(final_labels))
        unique, counts = np.unique(final_labels, return_counts=True)
        
        self._log(f"  Final clusters: {n_clusters}")
        self._log(f"  Cluster sizes: min={counts.min()}, max={counts.max()}, "
                 f"median={int(np.median(counts))}")
        
        return final_labels, consensus


# ============================================================================
# Canonical pass aliases (v3.0)
# ============================================================================
Pass4GraphBuilder = GraphBuilder
Pass4LeidenClustering = LeidenClustering
