"""VIF Filtering with Cluster Information v2.0"""
import numpy as np
import torch
from typing import List, Dict, Tuple
from descriptor_pipeline.config.settings import PipelineConfig
from descriptor_pipeline.utils.logging import log


class VIFFilteringPassGPUWithClusters:
    def __init__(self, config: PipelineConfig, verbose: bool = True, device: torch.device = None):
        self.config = config
        self.verbose = verbose
        self.vif_threshold = getattr(config, 'vif_threshold', 10.0)
        self.correlation_threshold = getattr(config, 'vif_cluster_corr_threshold', 0.7)
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        if self.verbose and self.device.type == 'cuda':
            gpu_name = torch.cuda.get_device_name(0)
            log(f"✓ VIF using GPU: {gpu_name}", self.verbose)
    
    def _log(self, msg: str):
        if self.verbose:
            log(msg, self.verbose)
    
    def process_from_correlation(self, G: np.ndarray, columns: List[str], stats: Dict) -> Tuple[List[str], Dict, np.ndarray]:
        self._log(f"\n  → Computing VIF (threshold={self.vif_threshold})...")
        p_init = len(columns)
        G_tensor = torch.from_numpy(G).to(self.device, dtype=torch.float32)
        remaining_indices = torch.arange(p_init, device=self.device)
        removed_info_list = []
        vif_values = {}
        clusters_dict = {}
        iteration = 0
        max_iterations = p_init
        
        while len(remaining_indices) > 0 and iteration < max_iterations:
            iteration += 1
            G_current = G_tensor[remaining_indices][:, remaining_indices]
            p_current = len(remaining_indices)
            if p_current == 0:
                break
            vifs = self._compute_vif_batch(G_current)
            max_vif, max_idx_local = vifs.max(dim=0)
            max_vif_value = max_vif.item()
            if max_vif_value < self.vif_threshold:
                break
            max_idx_global = remaining_indices[max_idx_local].item()
            col_name = columns[max_idx_global]
            
            try:
                eps = 1e-6
                G_inv = torch.linalg.inv(G_current + eps * torch.eye(p_current, device=self.device, dtype=torch.float32))
                diag_inv = torch.diag(G_inv)
                denom = torch.sqrt(diag_inv[max_idx_local] * diag_inv)
                partial_corr = -G_inv[max_idx_local] / denom
                partial_corr[max_idx_local] = 0.0
                marginal_corr = G_current[max_idx_local].abs()
                marginal_corr[max_idx_local] = 0.0
                combined_score = 0.7 * partial_corr.abs() + 0.3 * marginal_corr
                high_corr_mask = combined_score >= self.correlation_threshold
                high_corr_indices_local = torch.where(high_corr_mask)[0]
            except RuntimeError:
                marginal_corr = G_current[max_idx_local].abs()
                marginal_corr[max_idx_local] = 0.0
                high_corr_mask = marginal_corr >= self.correlation_threshold
                high_corr_indices_local = torch.where(high_corr_mask)[0]
            
            high_corr_indices_global = remaining_indices[high_corr_indices_local].cpu().numpy()
            high_corr_indices_global = [int(idx) for idx in high_corr_indices_global if int(idx) != max_idx_global]
            correlated_descriptors = [columns[idx] for idx in high_corr_indices_global]
            corr_values = {columns[idx]: float(G_tensor[max_idx_global, idx].item()) for idx in high_corr_indices_global}
            
            removed_info = {
                'descriptor': col_name,
                'descriptor_idx': int(max_idx_global),
                'vif_value': float(max_vif_value),
                'removed_order': iteration,
                'correlated_descriptors': correlated_descriptors,
                'correlation_values': corr_values,
                'cluster_size': len(correlated_descriptors) + 1
            }
            removed_info_list.append(removed_info)
            clusters_dict[col_name] = {
                'members': [col_name] + correlated_descriptors,
                'vif_value': float(max_vif_value),
                'removed_order': iteration,
                'correlation_values': corr_values
            }
            vif_values[col_name] = float(max_vif_value)
            mask = torch.ones(len(remaining_indices), dtype=torch.bool, device=self.device)
            mask[max_idx_local] = False
            remaining_indices = remaining_indices[mask]
            if self.verbose and iteration % 10 == 0:
                avg_cluster_size = np.mean([info['cluster_size'] for info in removed_info_list])
                self._log(f"    Iteration {iteration}: removed {len(removed_info_list)}, remaining {len(remaining_indices)}, avg {avg_cluster_size:.1f}")
        
        remaining_indices_cpu = remaining_indices.cpu().numpy()
        filtered_columns = [columns[int(i)] for i in remaining_indices_cpu]
        removed_columns = [info['descriptor'] for info in removed_info_list]
        if removed_info_list:
            cluster_sizes = [info['cluster_size'] for info in removed_info_list]
            cluster_stats = {'mean': float(np.mean(cluster_sizes)), 'median': float(np.median(cluster_sizes)), 'min': int(np.min(cluster_sizes)), 'max': int(np.max(cluster_sizes)), 'std': float(np.std(cluster_sizes))}
        else:
            cluster_stats = {'mean': 0, 'median': 0, 'min': 0, 'max': 0, 'std': 0}
        
        info = {
            'removed': removed_columns,
            'removed_count': len(removed_columns),
            'remaining_count': len(filtered_columns),
            'vif_values': vif_values,
            'iterations': iteration,
            'removed_details': removed_info_list,
            'clusters': clusters_dict,
            'representatives': clusters_dict,
            'cluster_statistics': cluster_stats
        }
        self._log(f"    Removed {len(removed_columns)}, Remaining {len(filtered_columns)}")
        return filtered_columns, info, remaining_indices_cpu
    
    def _compute_vif_batch(self, G: torch.Tensor) -> torch.Tensor:
        p = G.shape[0]
        if p == 1:
            return torch.ones(1, device=self.device, dtype=torch.float32)
        vifs = torch.zeros(p, device=self.device, dtype=torch.float32)
        for j in range(p):
            idx_other = torch.cat([torch.arange(j, device=self.device), torch.arange(j+1, p, device=self.device)])
            if len(idx_other) == 0:
                vifs[j] = 1.0
                continue
            r_j = G[j, idx_other]
            G_other = G[idx_other][:, idx_other]
            try:
                L = torch.linalg.cholesky(G_other + 1e-6 * torch.eye(len(idx_other), device=self.device, dtype=torch.float32))
                v = torch.cholesky_solve(r_j.unsqueeze(1), L).squeeze(1)
                R2_j = torch.dot(r_j, v)
                R2_j = torch.clamp(R2_j, 0.0, 0.9999)
                vif_j = 1.0 / (1.0 - R2_j)
                vifs[j] = vif_j
            except RuntimeError:
                vifs[j] = float('inf')
        return vifs
