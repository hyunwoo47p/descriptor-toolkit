"""
GPU-accelerated similarity computation using PyTorch
MEMORY-OPTIMIZED VERSION with chunk-based accumulation

Key improvements:
1. 2-pass algorithm: 1st pass for statistics, 2nd pass for matrix accumulation
2. In-place operations to minimize memory allocation
3. Row-chunk processing instead of loading all data
4. Float32 option for 2x memory reduction

Author: Memory Optimization v2.0
"""

import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from scipy.special import ndtri

import torch
import torch.nn.functional as F
from tqdm import tqdm

from descriptor_pipeline.config.settings import PipelineConfig
from descriptor_pipeline.core.seed_manager import SeedManager
from descriptor_pipeline.io.parquet_reader import iter_batches
from descriptor_pipeline.utils.logging import log


# ============================================================================
# GPU 유틸리티
# ============================================================================

def get_optimal_device(prefer_gpu: bool = True) -> torch.device:
    """최적 디바이스 선택"""
    if prefer_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✓ Using GPU: {gpu_name} ({gpu_memory:.1f} GB)")
        return device
    else:
        print("⚠ GPU not available, using CPU")
        return torch.device('cpu')


class GPUMemoryManager:
    """GPU 메모리 효율적 관리"""
    
    def __init__(self, device: torch.device, max_gpu_memory_gb: float = 40.0):
        self.device = device
        self.max_gpu_memory = max_gpu_memory_gb * 1e9
    
    def estimate_chunk_size(self, n_features: int, dtype_size: int = 4) -> int:
        """
        안전한 청크 크기 계산
        
        Args:
            n_features: 피처 수 (p)
            dtype_size: 바이트 수 (4=float32, 8=float64)
        
        Returns:
            청크 행 수
        """
        if self.device.type == 'cpu':
            return 500000  # CPU는 메모리 여유
        
        # GPU 메모리의 80%만 사용 (안전 마진)
        usable_memory = self.max_gpu_memory * 0.8
        
        # 각 청크에서 필요한 메모리:
        # - X_chunk: chunk_rows × p × dtype_size
        # - 중간 버퍼: ~2배 (연산 중 임시 텐서)
        bytes_per_row = n_features * dtype_size * 3
        
        chunk_rows = int(usable_memory / bytes_per_row)
        
        # 최소/최대 제한
        chunk_rows = max(10000, min(chunk_rows, 1000000))
        
        return chunk_rows


# ============================================================================
# Pass2: GPU Spearman Computer (MEMORY-OPTIMIZED)
# ============================================================================

class SpearmanComputerGPU:
    """
    GPU 가속 Spearman 상관계수 계산 (메모리 최적화)
    
    2-pass chunk accumulation:
    1) 1st pass: Compute global mean/std from data chunks
    2) 2nd pass: Accumulate G = sum(X_chunk^T @ X_chunk) / n
    
    This avoids loading entire dataset into GPU memory at once.
    """
    
    def __init__(self, config: PipelineConfig, seed_mgr: SeedManager,
                 verbose: bool = True, device: Optional[torch.device] = None,
                 use_float32: bool = True):
        self.config = config
        self.seed_mgr = seed_mgr
        self.verbose = verbose
        self.device = device or get_optimal_device()
        self.use_float32 = use_float32
        # config에서 max_gpu_memory_gb 가져오기 (없으면 40.0 기본값)
        max_gpu_memory = getattr(config, 'max_gpu_memory_gb', 40.0)
        self.memory_manager = GPUMemoryManager(self.device, max_gpu_memory_gb=max_gpu_memory)
    
    def _log(self, msg: str):
        if self.verbose:
            log(msg, self.verbose)
    
    def compute(self, parquet_paths: List[str], columns: List[str],
                stats: Dict, sample_indices_path: Optional[str] = None) -> np.ndarray:
        """
        GPU 가속 Spearman 상관계수 행렬 계산 (청크 누적 방식)
        
        Args:
            parquet_paths: Parquet 파일 경로들
            columns: 컬럼 이름들
            stats: Pass1에서 계산된 통계
            sample_indices_path: 샘플링 인덱스 (optional)
        
        Returns:
            G: Spearman 상관계수 행렬 (p x p)
        """
        start_time = time.time()
        p = len(columns)
        
        self._log("\n" + "="*70)
        self._log("GPU-Accelerated Spearman (Memory-Optimized Chunk Accumulation)")
        self._log(f"Device: {self.device}")
        self._log(f"Precision: {'float32' if self.use_float32 else 'float64'}")
        self._log("="*70)
        
        # 청크 크기 결정
        dtype_size = 4 if self.use_float32 else 8
        chunk_rows = self.memory_manager.estimate_chunk_size(p, dtype_size)
        self._log(f"  Chunk size: {chunk_rows:,} rows")
        
        # 1단계: 1-pass로 mean/std 계산
        self._log("\n  → Pass 1: Computing global mean/std...")
        mean, std, total_samples = self._compute_statistics_1pass(
            parquet_paths, columns, stats, chunk_rows
        )
        self._log(f"  Total samples: {total_samples:,}")
        
        # 2단계: 2-pass로 상관행렬 누적
        self._log("\n  → Pass 2: Accumulating correlation matrix...")
        G = self._accumulate_correlation_2pass(
            parquet_paths, columns, stats, mean, std, chunk_rows, total_samples
        )
        
        # 3단계: CPU로 결과 반환
        G_cpu = G.cpu().numpy()
        np.fill_diagonal(G_cpu, 1.0)
        
        elapsed = time.time() - start_time
        self._log(f"\n✅ Completed: {elapsed:.2f}s")
        self._log(f"  Range: [{G_cpu.min():.3f}, {G_cpu.max():.3f}]")
        self._log("="*70 + "\n")
        
        return G_cpu
    
    def _compute_statistics_1pass(self, parquet_paths: List[str],
                                   columns: List[str], stats: Dict,
                                   chunk_rows: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        1-pass: Welford 온라인 알고리즘으로 global mean/std 계산
        
        Returns:
            mean: (p,) tensor on GPU
            std: (p,) tensor on GPU
            total_samples: int
        """
        import gc
        
        p = len(columns)
        torch_dtype = torch.float32 if self.use_float32 else torch.float64
        
        # Welford 누적 변수 (GPU)
        n_total = 0
        mean_acc = torch.zeros(p, device=self.device, dtype=torch_dtype)
        M2_acc = torch.zeros(p, device=self.device, dtype=torch_dtype)
        
        batch_iterator = iter_batches(parquet_paths, columns, chunk_rows)
        
        for chunk_idx, (X_cpu, offset) in enumerate(tqdm(batch_iterator, desc="      Pass 1 (statistics)",
                                  disable=not self.verbose, leave=False)):
            n_chunk = len(X_cpu)
            
            # Clip (in-place 연산 사용)
            np.clip(X_cpu, stats['clip_lower'], stats['clip_upper'], out=X_cpu)
            
            # Copula transform (CPU에서)
            X_copula = self._copula_transform_cpu(X_cpu, columns, stats)
            
            # X_cpu는 더 이상 필요 없음 - 즉시 삭제
            del X_cpu
            
            # GPU로 전송
            X_chunk = torch.from_numpy(X_copula).to(self.device, dtype=torch_dtype)
            
            # X_copula도 더 이상 필요 없음 - 즉시 삭제
            del X_copula
            
            # NaN/Inf 처리
            X_chunk = torch.nan_to_num(X_chunk, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Welford 업데이트
            n_total += n_chunk
            delta = X_chunk.mean(dim=0) - mean_acc
            mean_acc += delta * (n_chunk / n_total)
            delta2 = X_chunk.mean(dim=0) - mean_acc
            M2_acc += (X_chunk - mean_acc).pow(2).sum(dim=0)
            
            # 명시적 메모리 정리
            del X_chunk, delta, delta2
            
            # GPU 메모리 정리
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            
            # CPU 메모리 정리 (10개 청크마다)
            if (chunk_idx + 1) % 10 == 0:
                gc.collect()
        
        # 최종 통계
        std_acc = torch.sqrt(M2_acc / (n_total - 1))
        std_acc = torch.where(std_acc < 1e-10, torch.ones_like(std_acc), std_acc)
        
        return mean_acc, std_acc, n_total
    
    def _accumulate_correlation_2pass(self, parquet_paths: List[str],
                                       columns: List[str], stats: Dict,
                                       mean: torch.Tensor, std: torch.Tensor,
                                       chunk_rows: int, total_samples: int) -> torch.Tensor:
        """
        2-pass: 청크 단위로 G = sum(X_chunk^T @ X_chunk) / n 누적
        
        Args:
            mean, std: 1-pass에서 계산된 전역 통계 (GPU tensor)
        
        Returns:
            G: (p, p) correlation matrix on GPU
        """
        import gc
        
        p = len(columns)
        torch_dtype = torch.float32 if self.use_float32 else torch.float64
        
        # 누적용 G 행렬 (GPU)
        G = torch.zeros(p, p, device=self.device, dtype=torch_dtype)
        
        batch_iterator = iter_batches(parquet_paths, columns, chunk_rows)
        
        for chunk_idx, (X_cpu, offset) in enumerate(tqdm(batch_iterator, desc="      Pass 2 (accumulation)",
                                  disable=not self.verbose, leave=False)):
            # Clip (in-place)
            np.clip(X_cpu, stats['clip_lower'], stats['clip_upper'], out=X_cpu)
            
            # Copula transform
            X_copula = self._copula_transform_cpu(X_cpu, columns, stats)
            
            # X_cpu 즉시 삭제
            del X_cpu
            
            # GPU로 전송
            X_chunk = torch.from_numpy(X_copula).to(self.device, dtype=torch_dtype)
            
            # X_copula 즉시 삭제
            del X_copula
            
            X_chunk = torch.nan_to_num(X_chunk, nan=0.0, posinf=0.0, neginf=0.0)
            
            # 인플레이스 표준화 (메모리 절약)
            # X = (X - mean) / std
            X_chunk.sub_(mean)  # X -= mean (in-place)
            X_chunk.div_(std)   # X /= std (in-place)
            
            # G += X^T @ X (누적)
            # addmm_: G = beta*G + alpha*(X.T @ X)
            G.addmm_(X_chunk.t(), X_chunk, beta=1.0, alpha=1.0)
            
            # 명시적 메모리 정리
            del X_chunk
            
            # GPU 메모리 정리
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            
            # CPU 메모리 정리 (10개 청크마다)
            if (chunk_idx + 1) % 10 == 0:
                gc.collect()
        
        # 정규화: G /= n
        G.div_(total_samples)
        
        # 수치 안정성
        G.clamp_(-1.0, 1.0)
        torch.diagonal(G).fill_(1.0)
        
        # 대칭 보장
        G = (G + G.t()) / 2
        
        return G
    
    def _copula_transform_cpu(self, X: np.ndarray, columns: List[str],
                               stats: Dict) -> np.ndarray:
        """
        Copula 변환 (CPU에서 수행)
        
        scipy의 ndtri는 GPU 미지원이므로 CPU에서 처리
        """
        X_copula = np.zeros_like(X, dtype=np.float64)
        
        for j in range(len(columns)):
            col = X[:, j]
            valid = np.isfinite(col)
            if valid.any():
                vals = col[valid]
                lookup_vals, lookup_cdfs = stats['cdf_lookups'][j]
                u = np.interp(vals, lookup_vals, lookup_cdfs, left=0.0, right=1.0)
                u = np.clip(u, 1e-12, 1.0 - 1e-12)
                X_copula[valid, j] = ndtri(u)
        
        return X_copula


# ============================================================================
# Pass4: GPU HSIC Computer (MEMORY-OPTIMIZED)
# ============================================================================

class HSICComputerGPU:
    """
    GPU 가속 HSIC (메모리 최적화)
    
    청크 단위로 random features 계산 후 누적
    """
    
    def __init__(self, config: PipelineConfig, seed_mgr: SeedManager,
                 verbose: bool = True, device: Optional[torch.device] = None,
                 use_float32: bool = True):
        self.config = config
        self.seed_mgr = seed_mgr
        self.verbose = verbose
        self.device = device or get_optimal_device()
        self.use_float32 = use_float32
        self.memory_manager = GPUMemoryManager(self.device)
    
    def _log(self, msg: str):
        if self.verbose:
            log(msg, self.verbose)
    
    def compute(self, parquet_paths: List[str], columns: List[str],
                stats: Dict, sample_indices_path: Optional[str] = None) -> np.ndarray:
        """GPU 가속 HSIC 행렬 계산 (청크 누적)"""
        start_time = time.time()
        p = len(columns)
        d = self.config.hsic_D
        r = self.config.r
        
        self._log("\n" + "="*70)
        self._log("GPU-Accelerated HSIC (Memory-Optimized)")
        self._log(f"Device: {self.device}")
        self._log(f"Random features: {d}, Repetitions: {r}")
        self._log("="*70)
        
        # 청크 크기
        dtype_size = 4 if self.use_float32 else 8
        chunk_rows = self.memory_manager.estimate_chunk_size(p, dtype_size)
        self._log(f"  Chunk size: {chunk_rows:,} rows")
        
        # 1단계: Random projections 생성
        self._log("\n  → Generating random projections...")
        W_list = self._generate_random_projections(p, d, r)
        
        # 2단계: 청크 단위로 HSIC 행렬 누적
        self._log("\n  → Computing HSIC matrix (chunk accumulation)...")
        H = self._accumulate_hsic_matrix(parquet_paths, columns, stats, W_list, chunk_rows)
        
        # 3단계: CPU로 결과 반환
        H_cpu = H.cpu().numpy()
        np.fill_diagonal(H_cpu, 1.0)
        
        elapsed = time.time() - start_time
        self._log(f"\n✅ Completed: {elapsed:.2f}s")
        self._log(f"  Range: [{H_cpu.min():.3f}, {H_cpu.max():.3f}]")
        self._log("="*70 + "\n")
        
        return H_cpu
    
    def _generate_random_projections(self, p: int, d: int, r: int) -> List[torch.Tensor]:
        """Random projections 생성"""
        torch_dtype = torch.float32 if self.use_float32 else torch.float64
        W_list = []
        
        for rep in range(r):
            rng = self.seed_mgr.get_rng('hsic', rep)
            W = rng.standard_normal((p, d)).astype(np.float64)
            W_tensor = torch.from_numpy(W).to(self.device, dtype=torch_dtype)
            W_list.append(W_tensor)
        
        return W_list
    
    def _accumulate_hsic_matrix(self, parquet_paths: List[str], columns: List[str],
                                 stats: Dict, W_list: List[torch.Tensor],
                                 chunk_rows: int) -> torch.Tensor:
        """
        청크 단위로 HSIC 행렬 누적
        
        H = (1/n) * sum_chunks(Z_chunk^T @ Z_chunk)
        where Z = sin(X @ W) normalized
        """
        p = len(columns)
        r = len(W_list)
        d = W_list[0].shape[1]
        torch_dtype = torch.float32 if self.use_float32 else torch.float64
        
        H_list = []
        
        for rep, W in enumerate(W_list):
            # 각 repetition마다 독립적으로 누적
            # Z의 통계를 먼저 계산해야 하므로, 2-pass 필요
            
            # 1-pass: Z의 mean/std 계산
            Z_mean, Z_std, total_samples = self._compute_z_statistics(
                parquet_paths, columns, stats, W, chunk_rows
            )
            
            # 2-pass: Z 표준화 후 H 누적
            H_rep = self._accumulate_z_correlation(
                parquet_paths, columns, stats, W, Z_mean, Z_std, 
                chunk_rows, total_samples
            )
            
            H_rep.clamp_(-1.0, 1.0)
            H_list.append(H_rep)
        
        # Median over repetitions
        H_stacked = torch.stack(H_list, dim=0)
        H = torch.median(H_stacked, dim=0).values
        
        # 대칭화
        H = (H + H.t()) / 2
        torch.diagonal(H).fill_(1.0)
        
        return H
    
    def _compute_z_statistics(self, parquet_paths: List[str], columns: List[str],
                               stats: Dict, W: torch.Tensor,
                               chunk_rows: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Z = sin(X @ W)의 전역 mean/std 계산"""
        import gc
        
        p = len(columns)
        d = W.shape[1]
        torch_dtype = torch.float32 if self.use_float32 else torch.float64
        
        # Z는 (p, n, d) 형태이므로, column별로 통계 계산
        n_total = 0
        Z_mean_acc = torch.zeros(p, d, device=self.device, dtype=torch_dtype)
        Z_M2_acc = torch.zeros(p, d, device=self.device, dtype=torch_dtype)
        
        batch_iterator = iter_batches(parquet_paths, columns, chunk_rows)
        
        for chunk_idx, (X_cpu, offset) in enumerate(batch_iterator):
            n_chunk = len(X_cpu)
            
            # Clip (in-place)
            np.clip(X_cpu, stats['clip_lower'], stats['clip_upper'], out=X_cpu)
            
            X_copula = self._copula_transform_cpu(X_cpu, columns, stats)
            
            # X_cpu 즉시 삭제
            del X_cpu
            
            X_chunk = torch.from_numpy(X_copula).to(self.device, dtype=torch_dtype)
            
            # X_copula 즉시 삭제
            del X_copula
            
            X_chunk = torch.nan_to_num(X_chunk, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Z_chunk: (p, n_chunk, d)
            Z_chunk = []
            for i in range(p):
                x_i = X_chunk[:, i:i+1]  # (n_chunk, 1)
                w_i = W[i:i+1, :]  # (1, d)
                z_i = torch.sin(x_i @ w_i)  # (n_chunk, d)
                Z_chunk.append(z_i)
            Z_chunk = torch.stack(Z_chunk, dim=0)  # (p, n_chunk, d)
            
            # Welford update (per column)
            n_total += n_chunk
            chunk_mean = Z_chunk.mean(dim=1)  # (p, d)
            delta = chunk_mean - Z_mean_acc
            Z_mean_acc += delta * (n_chunk / n_total)
            
            # M2 업데이트 (정확한 Welford)
            for i in range(p):
                z_col = Z_chunk[i]  # (n_chunk, d)
                Z_M2_acc[i] += ((z_col - Z_mean_acc[i:i+1]) ** 2).sum(dim=0)
            
            del X_chunk, Z_chunk, chunk_mean, delta
            
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            
            # CPU 메모리 정리 (10개 청크마다)
            if (chunk_idx + 1) % 10 == 0:
                gc.collect()
        
        Z_std_acc = torch.sqrt(Z_M2_acc / (n_total - 1))
        Z_std_acc = torch.where(Z_std_acc < 1e-10, torch.ones_like(Z_std_acc), Z_std_acc)
        
        return Z_mean_acc, Z_std_acc, n_total
    
    def _accumulate_z_correlation(self, parquet_paths: List[str], columns: List[str],
                                   stats: Dict, W: torch.Tensor,
                                   Z_mean: torch.Tensor, Z_std: torch.Tensor,
                                   chunk_rows: int, total_samples: int) -> torch.Tensor:
        """Z 표준화 후 상관행렬 누적"""
        import gc
        
        p = len(columns)
        d = W.shape[1]
        torch_dtype = torch.float32 if self.use_float32 else torch.float64
        
        # 누적 행렬
        H = torch.zeros(p, p, device=self.device, dtype=torch_dtype)
        
        batch_iterator = iter_batches(parquet_paths, columns, chunk_rows)
        
        for chunk_idx, (X_cpu, offset) in enumerate(batch_iterator):
            # Clip (in-place)
            np.clip(X_cpu, stats['clip_lower'], stats['clip_upper'], out=X_cpu)
            
            X_copula = self._copula_transform_cpu(X_cpu, columns, stats)
            
            # X_cpu 즉시 삭제
            del X_cpu
            
            X_chunk = torch.from_numpy(X_copula).to(self.device, dtype=torch_dtype)
            
            # X_copula 즉시 삭제
            del X_copula
            
            X_chunk = torch.nan_to_num(X_chunk, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Z_chunk 계산
            Z_chunk = []
            for i in range(p):
                x_i = X_chunk[:, i:i+1]
                w_i = W[i:i+1, :]
                z_i = torch.sin(x_i @ w_i)
                Z_chunk.append(z_i)
            Z_chunk = torch.stack(Z_chunk, dim=0)  # (p, n_chunk, d)
            
            # 인플레이스 표준화
            Z_chunk -= Z_mean.unsqueeze(1)  # (p, 1, d) broadcast
            Z_chunk /= Z_std.unsqueeze(1)
            
            # Flatten: (p, n_chunk*d)
            Z_flat = Z_chunk.reshape(p, -1)
            
            # H += Z_flat @ Z_flat.T
            H.addmm_(Z_flat, Z_flat.t(), beta=1.0, alpha=1.0)
            
            del X_chunk, Z_chunk, Z_flat
            
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            
            # CPU 메모리 정리 (10개 청크마다)
            if (chunk_idx + 1) % 10 == 0:
                gc.collect()
        
        # 정규화
        H /= (total_samples * d)
        
        return H
    
    def _copula_transform_cpu(self, X: np.ndarray, columns: List[str],
                               stats: Dict) -> np.ndarray:
        """Copula 변환 (CPU)"""
        X_copula = np.zeros_like(X, dtype=np.float64)
        for j in range(len(columns)):
            col = X[:, j]
            valid = np.isfinite(col)
            if valid.any():
                vals = col[valid]
                lookup_vals, lookup_cdfs = stats['cdf_lookups'][j]
                u = np.interp(vals, lookup_vals, lookup_cdfs, left=0.0, right=1.0)
                u = np.clip(u, 1e-12, 1.0 - 1e-12)
                X_copula[valid, j] = ndtri(u)
        return X_copula


# ============================================================================
# Pass4: GPU RDC Computer (MEMORY-OPTIMIZED)
# ============================================================================

class RDCComputerGPU:
    """
    GPU 가속 RDC (메모리 최적화)
    
    HSIC와 동일한 청크 누적 방식 적용
    """
    
    def __init__(self, config: PipelineConfig, seed_mgr: SeedManager,
                 verbose: bool = True, device: Optional[torch.device] = None,
                 use_float32: bool = True):
        self.config = config
        self.seed_mgr = seed_mgr
        self.verbose = verbose
        self.device = device or get_optimal_device()
        self.use_float32 = use_float32
        self.memory_manager = GPUMemoryManager(self.device)
    
    def _log(self, msg: str):
        if self.verbose:
            log(msg, self.verbose)
    
    def compute(self, parquet_paths: List[str], columns: List[str],
                stats: Dict, sample_indices_path: Optional[str] = None) -> np.ndarray:
        """GPU 가속 RDC 행렬 계산 (청크 누적)"""
        start_time = time.time()
        p = len(columns)
        d = self.config.rdc_d
        r = self.config.r
        n_rdc = self.config.rdc_seeds
        
        self._log("\n" + "="*70)
        self._log("GPU-Accelerated RDC (Memory-Optimized)")
        self._log(f"Device: {self.device}")
        self._log(f"Random projections: {d}, Repetitions: {r}, RDC seeds: {n_rdc}")
        self._log("="*70)
        
        # 청크 크기
        dtype_size = 4 if self.use_float32 else 8
        chunk_rows = self.memory_manager.estimate_chunk_size(p, dtype_size)
        self._log(f"  Chunk size: {chunk_rows:,} rows")
        
        # Random projections 생성
        self._log("\n  → Generating random projections...")
        W_list = self._generate_random_projections(p, d, r, n_rdc)
        
        # RDC 행렬 누적
        self._log("\n  → Computing RDC matrix (chunk accumulation)...")
        R = self._accumulate_rdc_matrix(parquet_paths, columns, stats, W_list, chunk_rows)
        
        # CPU로 결과 반환
        R_cpu = R.cpu().numpy()
        np.fill_diagonal(R_cpu, 1.0)
        
        elapsed = time.time() - start_time
        self._log(f"\n✅ Completed: {elapsed:.2f}s")
        self._log(f"  Range: [{R_cpu.min():.3f}, {R_cpu.max():.3f}]")
        self._log("="*70 + "\n")
        
        return R_cpu
    
    def _generate_random_projections(self, p: int, d: int, r: int, n_rdc: int):
        """Random projections 생성"""
        torch_dtype = torch.float32 if self.use_float32 else torch.float64
        W_list = []
        
        for rep in range(r):
            W_rep = []
            for rdc_idx in range(n_rdc):
                rng = self.seed_mgr.get_rng('rdc', rep * n_rdc + rdc_idx)
                W = rng.standard_normal((p, d)).astype(np.float64)
                W_tensor = torch.from_numpy(W).to(self.device, dtype=torch_dtype)
                W_rep.append(W_tensor)
            W_list.append(W_rep)
        
        return W_list
    
    def _accumulate_rdc_matrix(self, parquet_paths: List[str], columns: List[str],
                                stats: Dict, W_list: List[List[torch.Tensor]],
                                chunk_rows: int) -> torch.Tensor:
        """청크 단위로 RDC 행렬 누적"""
        p = len(columns)
        r = len(W_list)
        n_rdc = len(W_list[0])
        
        R_list = []
        
        for rep, W_rep in enumerate(W_list):
            R_rep_accum = None
            
            for W in W_rep:
                # 각 W에 대해 HSIC와 동일한 방식으로 처리
                Z_mean, Z_std, total_samples = self._compute_z_statistics(
                    parquet_paths, columns, stats, W, chunk_rows
                )
                
                R_block = self._accumulate_z_correlation(
                    parquet_paths, columns, stats, W, Z_mean, Z_std,
                    chunk_rows, total_samples
                )
                
                R_block.clamp_(-1.0, 1.0)
                
                if R_rep_accum is None:
                    R_rep_accum = R_block
                else:
                    R_rep_accum += R_block
            
            # Average over n_rdc
            R_rep = R_rep_accum / n_rdc
            R_list.append(R_rep)
        
        # Median over repetitions
        R_stacked = torch.stack(R_list, dim=0)
        R = torch.median(R_stacked, dim=0).values
        
        # 대칭화
        R = (R + R.t()) / 2
        torch.diagonal(R).fill_(1.0)
        
        return R
    
    def _compute_z_statistics(self, parquet_paths: List[str], columns: List[str],
                               stats: Dict, W: torch.Tensor, chunk_rows: int):
        """HSIC와 동일"""
        import gc
        
        p = len(columns)
        d = W.shape[1]
        torch_dtype = torch.float32 if self.use_float32 else torch.float64
        
        n_total = 0
        Z_mean_acc = torch.zeros(p, d, device=self.device, dtype=torch_dtype)
        Z_M2_acc = torch.zeros(p, d, device=self.device, dtype=torch_dtype)
        
        batch_iterator = iter_batches(parquet_paths, columns, chunk_rows)
        
        for chunk_idx, (X_cpu, offset) in enumerate(batch_iterator):
            n_chunk = len(X_cpu)
            
            # Clip (in-place)
            np.clip(X_cpu, stats['clip_lower'], stats['clip_upper'], out=X_cpu)
            
            X_copula = self._copula_transform_cpu(X_cpu, columns, stats)
            
            # X_cpu 즉시 삭제
            del X_cpu
            
            X_chunk = torch.from_numpy(X_copula).to(self.device, dtype=torch_dtype)
            
            # X_copula 즉시 삭제
            del X_copula
            
            X_chunk = torch.nan_to_num(X_chunk, nan=0.0, posinf=0.0, neginf=0.0)
            
            Z_chunk = []
            for i in range(p):
                x_i = X_chunk[:, i:i+1]
                w_i = W[i:i+1, :]
                z_i = torch.sin(x_i @ w_i)
                Z_chunk.append(z_i)
            Z_chunk = torch.stack(Z_chunk, dim=0)
            
            n_total += n_chunk
            chunk_mean = Z_chunk.mean(dim=1)
            delta = chunk_mean - Z_mean_acc
            Z_mean_acc += delta * (n_chunk / n_total)
            
            for i in range(p):
                z_col = Z_chunk[i]
                Z_M2_acc[i] += ((z_col - Z_mean_acc[i:i+1]) ** 2).sum(dim=0)
            
            del X_chunk, Z_chunk, chunk_mean, delta
            
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            
            # CPU 메모리 정리 (10개 청크마다)
            if (chunk_idx + 1) % 10 == 0:
                gc.collect()
        
        Z_std_acc = torch.sqrt(Z_M2_acc / (n_total - 1))
        Z_std_acc = torch.where(Z_std_acc < 1e-10, torch.ones_like(Z_std_acc), Z_std_acc)
        
        return Z_mean_acc, Z_std_acc, n_total
    
    def _accumulate_z_correlation(self, parquet_paths: List[str], columns: List[str],
                                   stats: Dict, W: torch.Tensor,
                                   Z_mean: torch.Tensor, Z_std: torch.Tensor,
                                   chunk_rows: int, total_samples: int):
        """HSIC와 동일"""
        import gc
        
        p = len(columns)
        d = W.shape[1]
        torch_dtype = torch.float32 if self.use_float32 else torch.float64
        
        R = torch.zeros(p, p, device=self.device, dtype=torch_dtype)
        
        batch_iterator = iter_batches(parquet_paths, columns, chunk_rows)
        
        for chunk_idx, (X_cpu, offset) in enumerate(batch_iterator):
            # Clip (in-place)
            np.clip(X_cpu, stats['clip_lower'], stats['clip_upper'], out=X_cpu)
            
            X_copula = self._copula_transform_cpu(X_cpu, columns, stats)
            
            # X_cpu 즉시 삭제
            del X_cpu
            
            X_chunk = torch.from_numpy(X_copula).to(self.device, dtype=torch_dtype)
            
            # X_copula 즉시 삭제
            del X_copula
            
            X_chunk = torch.nan_to_num(X_chunk, nan=0.0, posinf=0.0, neginf=0.0)
            
            Z_chunk = []
            for i in range(p):
                x_i = X_chunk[:, i:i+1]
                w_i = W[i:i+1, :]
                z_i = torch.sin(x_i @ w_i)
                Z_chunk.append(z_i)
            Z_chunk = torch.stack(Z_chunk, dim=0)
            
            Z_chunk -= Z_mean.unsqueeze(1)
            Z_chunk /= Z_std.unsqueeze(1)
            
            Z_flat = Z_chunk.reshape(p, -1)
            
            R.addmm_(Z_flat, Z_flat.t(), beta=1.0, alpha=1.0)
            
            del X_chunk, Z_chunk, Z_flat
            
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            
            # CPU 메모리 정리 (10개 청크마다)
            if (chunk_idx + 1) % 10 == 0:
                gc.collect()
        
        R /= (total_samples * d)
        
        return R
    
    def _copula_transform_cpu(self, X: np.ndarray, columns: List[str],
                               stats: Dict) -> np.ndarray:
        """Copula 변환 (CPU)"""
        X_copula = np.zeros_like(X, dtype=np.float64)
        for j in range(len(columns)):
            col = X[:, j]
            valid = np.isfinite(col)
            if valid.any():
                vals = col[valid]
                lookup_vals, lookup_cdfs = stats['cdf_lookups'][j]
                u = np.interp(vals, lookup_vals, lookup_cdfs, left=0.0, right=1.0)
                u = np.clip(u, 1e-12, 1.0 - 1e-12)
                X_copula[valid, j] = ndtri(u)
        return X_copula
