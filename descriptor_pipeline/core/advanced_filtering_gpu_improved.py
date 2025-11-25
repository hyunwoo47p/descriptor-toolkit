"""VIF Filtering with Cluster Information v2.0

Survivor Selection Method:
--------------------------
When a descriptor D is removed due to high VIF, we find the "survivor" - the descriptor
that most depends on D - by calculating R² changes:

1. Calculate VIF for all descriptors (with D included)
2. Calculate VIF for all descriptors (after removing D)
3. For each remaining descriptor X:
   R²_before = 1 - 1/VIF_before(X)
   R²_after = 1 - 1/VIF_after(X)
   ΔR² = R²_before - R²_after
   
4. The descriptor with maximum ΔR² is the survivor (as long as ΔR² > 0)

Intuition: If removing D causes X's VIF to drop (ΔR² > 0), it means X was dependent
on D. The descriptor with largest ΔR² is most dependent on D, thus inherits D's cluster.

Why no threshold?
- Even ΔR² = 0.0001 indicates relative dependency (if it's the maximum)
- D was removed for high VIF (multicollinearity exists by definition)
- Preserves cluster information: every removed descriptor has a survivor
- Represents "relative dependency" in current descriptor space

This is theoretically sound because:
- VIF = 1/(1-R²) where R² is from regressing X on all other descriptors
- Any ΔR² > 0 means D had some explanatory power for X
- Captures multivariate dependencies (not just pairwise correlation)
- "Most dependent" is always well-defined (argmax exists)
"""
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
        # Note: No threshold for R² drop - we always assign survivor to max(ΔR²)
        # Rationale: Even small ΔR² indicates relative dependency, preserves cluster info
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
    
    def process_from_correlation(self, G: np.ndarray, columns: List[str], stats: Dict, pass2_info: Dict = None) -> Tuple[List[str], Dict, np.ndarray]:
        self._log(f"\n  → Computing VIF (threshold={self.vif_threshold})...")
        p_init = len(columns)
        G_tensor = torch.from_numpy(G).to(self.device, dtype=torch.float32)
        remaining_indices = torch.arange(p_init, device=self.device)
        removed_info_list = []
        vif_values = {}
        clusters_dict = {}
        iteration = 0
        max_iterations = p_init
        
        # Extract Pass2 cluster information
        pass2_clusters = {}
        if pass2_info and 'representatives' in pass2_info:
            for rep, info in pass2_info['representatives'].items():
                if rep in columns:
                    members = info.get('members', [])
                    pass2_clusters[rep] = set(members) - {rep}  # Exclude representative
        
        if self.verbose and pass2_clusters:
            total_pass2_members = sum(len(m) for m in pass2_clusters.values())
            self._log(f"  Pass2 clusters loaded: {len(pass2_clusters)} representatives, {total_pass2_members} members")
        
        # Track cluster inheritance: descriptor -> set of inherited cluster members
        inherited_clusters = {}  # {descriptor: set(members inherited from removed descriptors)}
        
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
            
            # Select survivor based on R² change (VIF drop method)
            # Theory: Descriptor with largest R² drop is most dependent on removed descriptor
            survivor = None
            survivor_r2_drop = 0.0
            survivor_corr = 0.0
            
            # Calculate VIF before removal (already computed)
            vif_before = vifs.cpu().numpy()
            
            # Calculate VIF after removal to find R² changes
            mask_without_removed = torch.ones(len(remaining_indices), dtype=torch.bool, device=self.device)
            mask_without_removed[max_idx_local] = False
            remaining_without_removed = remaining_indices[mask_without_removed]
            
            if len(remaining_without_removed) > 0:
                G_after_removal = G_tensor[remaining_without_removed][:, remaining_without_removed]
                vif_after = self._compute_vif_batch(G_after_removal).cpu().numpy()
                
                # Calculate R² change for each remaining descriptor
                # ΔR² = R²_before - R²_after = (1 - 1/VIF_before) - (1 - 1/VIF_after)
                #     = 1/VIF_after - 1/VIF_before
                r2_drops = {}
                local_idx_mapping = {}  # Map from original local idx to new local idx after removal
                new_idx = 0
                for i in range(len(remaining_indices)):
                    if i == max_idx_local:
                        continue
                    local_idx_mapping[i] = new_idx
                    new_idx += 1
                
                for local_idx in range(len(remaining_indices)):
                    if local_idx == max_idx_local:
                        continue
                    
                    global_idx = remaining_indices[local_idx].item()
                    desc_name = columns[global_idx]
                    
                    vif_b = vif_before[local_idx]
                    vif_a = vif_after[local_idx_mapping[local_idx]]
                    
                    # Avoid division by zero or invalid VIFs
                    if vif_b > 0 and vif_a > 0 and not np.isinf(vif_b) and not np.isinf(vif_a):
                        r2_before = 1.0 - 1.0/vif_b
                        r2_after = 1.0 - 1.0/vif_a
                        delta_r2 = r2_before - r2_after  # Positive means VIF dropped
                        r2_drops[desc_name] = delta_r2
                
                # Find descriptor with maximum R² drop
                if r2_drops:
                    max_drop_desc = max(r2_drops, key=r2_drops.get)
                    max_drop_value = r2_drops[max_drop_desc]
                    
                    # As long as there's any positive ΔR², assign survivor
                    # Even small ΔR² means relative dependency exists
                    if max_drop_value > 0:
                        survivor = max_drop_desc
                        survivor_r2_drop = max_drop_value
                        # Also store correlation for reference
                        if survivor in corr_values:
                            survivor_corr = corr_values[survivor]
            
            # Collect all cluster members for this descriptor
            # 1. Pass2 cluster members (if this was a Pass2 representative)
            pass2_members = []
            if col_name in pass2_clusters:
                pass2_members = sorted(pass2_clusters[col_name])
            
            # 2. Inherited clusters from previously removed descriptors
            inherited_members = []
            if col_name in inherited_clusters:
                inherited_members = sorted(inherited_clusters[col_name])
            
            # 3. All members = self + Pass2 + inherited + Pass3 correlated
            all_members = [col_name] + pass2_members + inherited_members + correlated_descriptors
            all_members = sorted(set(all_members))  # Remove duplicates
            
            # Transfer all clusters to survivor
            if survivor:
                # Survivor inherits: Pass2 members + inherited members + this descriptor itself
                if survivor not in inherited_clusters:
                    inherited_clusters[survivor] = set()
                inherited_clusters[survivor].add(col_name)  # Add removed descriptor
                inherited_clusters[survivor].update(pass2_members)  # Add Pass2 members
                inherited_clusters[survivor].update(inherited_members)  # Propagate inherited
                
            removed_info = {
                'descriptor': col_name,
                'descriptor_idx': int(max_idx_global),
                'vif_value': float(max_vif_value),
                'removed_order': iteration,
                'correlated_descriptors': correlated_descriptors,
                'correlation_values': corr_values,
                'cluster_size': len(all_members),
                'pass2_members': pass2_members,
                'inherited_members': inherited_members,
                'survivor': survivor,  # Who inherits this cluster
                'survivor_correlation': float(survivor_corr) if survivor else 0.0,
                'survivor_r2_drop': float(survivor_r2_drop) if survivor else 0.0  # R² change in survivor
            }
            removed_info_list.append(removed_info)
            clusters_dict[col_name] = {
                'members': all_members,  # All cluster members
                'pass3_correlated': correlated_descriptors,  # Pass3 상관관계
                'pass2_members': pass2_members,  # Pass2 클러스터
                'inherited_members': inherited_members,  # Previously inherited
                'survivor': survivor,  # Who takes over this cluster
                'survivor_correlation': float(survivor_corr) if survivor else 0.0,
                'survivor_r2_drop': float(survivor_r2_drop) if survivor else 0.0,
                'vif_value': float(max_vif_value),
                'removed_order': iteration,
                'correlation_values': corr_values
            }
            vif_values[col_name] = float(max_vif_value)
            mask = torch.ones(len(remaining_indices), dtype=torch.bool, device=self.device)
            mask[max_idx_local] = False
            remaining_indices = remaining_indices[mask]
            
            # Clean up: remove this descriptor from inherited_clusters tracking
            if col_name in inherited_clusters:
                del inherited_clusters[col_name]
            
            if self.verbose and iteration % 10 == 0:
                avg_cluster_size = np.mean([info['cluster_size'] for info in removed_info_list])
                self._log(f"    Iteration {iteration}: removed {len(removed_info_list)}, remaining {len(remaining_indices)}, avg {avg_cluster_size:.1f}")
        
        remaining_indices_cpu = remaining_indices.cpu().numpy()
        filtered_columns = [columns[int(i)] for i in remaining_indices_cpu]
        removed_columns = [info['descriptor'] for info in removed_info_list]
        
        # Assign inherited clusters to final surviving descriptors
        survivors_with_inheritance = {}
        for survivor, inherited_set in inherited_clusters.items():
            survivors_with_inheritance[survivor] = {
                'inherited_members': sorted(inherited_set),
                'inherited_count': len(inherited_set)
            }
        
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
            'cluster_statistics': cluster_stats,
            'survivors_with_inheritance': survivors_with_inheritance  # Final survivors and their inherited clusters
        }
        self._log(f"    Removed {len(removed_columns)}, Remaining {len(filtered_columns)}")
        
        # Log inheritance summary
        if survivors_with_inheritance:
            total_inherited = sum(info['inherited_count'] for info in survivors_with_inheritance.values())
            self._log(f"    Cluster inheritance: {len(survivors_with_inheritance)} survivors inherited {total_inherited} members")
        
        # Log survivor selection summary
        if removed_info_list:
            survivors_found = sum(1 for info in removed_info_list if info['survivor'] is not None)
            if survivors_found > 0:
                avg_r2_drop = np.mean([info['survivor_r2_drop'] for info in removed_info_list if info['survivor'] is not None])
                self._log(f"    Survivor selection: {survivors_found}/{len(removed_info_list)} found (avg ΔR² = {avg_r2_drop:.4f})")
        
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
