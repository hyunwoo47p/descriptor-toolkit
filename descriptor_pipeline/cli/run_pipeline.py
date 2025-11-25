"""
Command-line interface for GPU-accelerated descriptor pipeline
Simplified for GPU mode execution
"""

import argparse
import sys
from pathlib import Path
from descriptor_pipeline.config.settings import PipelineConfig, RangeMode
from descriptor_pipeline.core.pipeline import DescriptorPipeline


def create_parser() -> argparse.ArgumentParser:
    """GPU 모드 중심 CLI 파서"""
    parser = argparse.ArgumentParser(
        description="GPU-Accelerated Descriptor Filtering Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input/Output (Required)
    io_group = parser.add_argument_group('Input/Output')
    io_group.add_argument(
        '--parquet-glob', type=str, required=True,
        help='Glob pattern for parquet files (e.g., "data/*.parquet")'
    )
    io_group.add_argument(
        '--output-dir', type=str, required=True,
        help='Output directory for results'
    )
    io_group.add_argument(
        '--descriptor-columns', type=str, nargs='+', default=None,
        help='Specific descriptor columns to use (default: all columns)'
    )
    io_group.add_argument(
        '--n-metadata', type=int, default=6,
        help='Number of metadata columns to exclude from variance filtering'
    )
    
    # GPU Settings
    gpu_group = parser.add_argument_group('GPU Settings')
    gpu_group.add_argument('--gpu', action='store_true', help='Enable GPU acceleration')
    gpu_group.add_argument('--no-gpu', action='store_true', help='Force CPU mode')
    gpu_group.add_argument('--gpu-id', type=int, default=0, help='GPU device ID')
    
    # Sampling
    sampling_group = parser.add_argument_group('Pass 0: Sampling')
    sampling_group.add_argument(
        '--sample-per-file', type=int, default=None,
        help='Sample N rows per file (default: no sampling)'
    )
    sampling_group.add_argument(
        '--file-independent-sampling', action='store_true',
        help='Use file-independent sampling'
    )
    sampling_group.add_argument(
        '--random-seed', type=int, default=42, help='Random seed'
    )
    
    # Statistics & Variance
    stats_group = parser.add_argument_group('Pass 1: Statistics & Variance')
    stats_group.add_argument(
        '--force-recompute', action='store_true',
        help='Force recompute statistics even if cache exists'
    )
    stats_group.add_argument(
        '--variance-threshold', type=float, default=0.002,
        help='Minimum normalized variance threshold'
    )
    stats_group.add_argument(
        '--max-missing-ratio', type=float, default=0.5,
        help='Maximum missing data ratio (0.0-1.0)'
    )
    stats_group.add_argument(
        '--min-effective-n', type=int, default=100,
        help='Minimum number of valid samples required'
    )
    stats_group.add_argument(
        '--range-mode', type=str, choices=['minmax', 'trimmed', 'iqr'],
        default='trimmed', help='Range calculation mode'
    )
    stats_group.add_argument('--trim-lower', type=float, default=2.5, help='Lower percentile')
    stats_group.add_argument('--trim-upper', type=float, default=97.5, help='Upper percentile')
    
    # Spearman
    spearman_group = parser.add_argument_group('Pass 2: Spearman Correlation')
    spearman_group.add_argument(
        '--spearman-threshold', type=float, default=0.95,
        help='Spearman correlation threshold'
    )
    spearman_group.add_argument('--m', type=int, default=64, help='CountSketch buckets')
    spearman_group.add_argument('--r', type=int, default=8, help='CountSketch repetitions')
    
    # VIF
    vif_group = parser.add_argument_group('Pass 3: VIF Multicollinearity')
    vif_group.add_argument('--vif-threshold', type=float, default=10.0, help='VIF threshold')
    
    # Nonlinear
    nonlinear_group = parser.add_argument_group('Pass 4: HSIC + RDC Nonlinear')
    nonlinear_group.add_argument(
        '--nonlinear-threshold', type=float, default=0.3,
        help='Nonlinear similarity threshold'
    )
    nonlinear_group.add_argument('--w-hsic', type=float, default=0.5, help='HSIC weight')
    nonlinear_group.add_argument('--w-rdc', type=float, default=0.5, help='RDC weight')
    nonlinear_group.add_argument('--hsic-D', type=int, default=50, help='HSIC dimension')
    nonlinear_group.add_argument('--rdc-d', type=int, default=20, help='RDC dimension')
    nonlinear_group.add_argument('--rdc-seeds', type=int, default=3, help='RDC random seeds')
    
    # Clustering
    cluster_group = parser.add_argument_group('Graph & Clustering')
    cluster_group.add_argument('--topk', type=int, default=40, help='k-NN graph k value')
    cluster_group.add_argument(
        '--leiden-resolution', type=float, default=1.0, help='Leiden resolution'
    )
    cluster_group.add_argument(
        '--n-consensus', type=int, default=10, help='Consensus iterations'
    )
    
    # Processing
    processing_group = parser.add_argument_group('Processing Options')
    processing_group.add_argument('--batch-rows', type=int, default=10000, help='Batch size')
    processing_group.add_argument(
        '--checkpoint', action='store_true', default=True, help='Enable checkpointing'
    )
    processing_group.add_argument(
        '--no-checkpoint', action='store_true', help='Disable checkpointing'
    )
    
    # Output
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument('-q', '--quiet', action='store_true', help='Quiet mode')
    output_group.add_argument('-v', '--verbose', action='store_true', help='Verbose mode')
    
    return parser


def check_gpu_availability(use_gpu: bool, gpu_id: int):
    """GPU 사용 가능 여부 확인"""
    if not use_gpu:
        return False, 0
    
    try:
        import torch
    except ImportError:
        print("⚠️  PyTorch not installed. Falling back to CPU mode...")
        return False, 0
    
    if not torch.cuda.is_available():
        print("⚠️  CUDA is not available! Falling back to CPU mode...")
        return False, 0
    
    # GPU 정보
    print("="*60)
    print("✓ GPU detected!")
    print("="*60)
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        total_mem = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"  Device {i}: {name} ({total_mem:.1f} GB)")
    print("="*60 + "\n")
    
    if gpu_id >= torch.cuda.device_count():
        print(f"⚠️  GPU {gpu_id} not available. Using GPU 0 instead.\n")
        return True, 0
    
    return True, gpu_id


def main():
    """메인 실행 함수"""
    parser = create_parser()
    args = parser.parse_args()
    
    # GPU 설정
    use_gpu = args.gpu and not args.no_gpu
    use_gpu, gpu_id = check_gpu_availability(use_gpu, args.gpu_id)
    
    if use_gpu:
        import torch
        torch.cuda.set_device(gpu_id)
    else:
        print("Using CPU mode\n")
    
    # Verbose 설정
    verbose = not args.quiet or args.verbose
    
    # Checkpoint 설정
    checkpoint = args.checkpoint and not args.no_checkpoint
    
    # 출력 디렉토리 생성
    Path(args.output_dir).mkdir(exist_ok=True, parents=True)
    
    # RangeMode 변환
    range_mode_map = {
        'minmax': RangeMode.MINMAX,
        'trimmed': RangeMode.TRIMMED,
        'iqr': RangeMode.IQR
    }
    range_mode = range_mode_map.get(args.range_mode, RangeMode.TRIMMED)
    
    # Config 생성
    config = PipelineConfig(
        # I/O
        parquet_glob=args.parquet_glob,
        output_dir=args.output_dir,
        descriptor_columns=args.descriptor_columns,
        n_metadata_cols=args.n_metadata,
        
        # GPU
        prefer_gpu=use_gpu,
        gpu_id=gpu_id,
        
        # Sampling
        sample_per_file=args.sample_per_file,
        file_independent_sampling=args.file_independent_sampling,
        random_seed=args.random_seed,
        
        # Pass 1
        force_recompute=args.force_recompute,
        variance_threshold=args.variance_threshold,
        use_robust_variance=True,
        range_mode=range_mode,
        trim_percentiles=(args.trim_lower, args.trim_upper),
        max_missing_ratio=args.max_missing_ratio,
        min_effective_n=args.min_effective_n,
        
        # Pass 2
        spearman_threshold=args.spearman_threshold,
        m=args.m,
        r=args.r,
        
        # Pass 3
        vif_threshold=args.vif_threshold,
        
        # Pass 4
        nonlinear_threshold=args.nonlinear_threshold,
        w_hsic=args.w_hsic,
        w_rdc=args.w_rdc,
        hsic_D=args.hsic_D,
        rdc_d=args.rdc_d,
        rdc_seeds=args.rdc_seeds,
        
        # Clustering
        topk=args.topk,
        leiden_resolution=args.leiden_resolution,
        n_consensus=args.n_consensus,
        
        # Processing
        batch_rows=args.batch_rows,
        checkpoint=checkpoint,
        verbose=verbose
    )
    
    # Pipeline 실행
    try:
        pipeline = DescriptorPipeline(config)
        
        if verbose:
            print("="*80)
            print("GPU-Accelerated Descriptor Pipeline")
            print("="*80)
            print(f"Input: {args.parquet_glob}")
            print(f"Output: {args.output_dir}")
            print(f"Mode: {'GPU' if use_gpu else 'CPU'}")
            if use_gpu:
                print(f"GPU ID: {gpu_id}")
            print(f"Sampling: {args.sample_per_file or 'None'} rows/file")
            print(f"Thresholds: var={args.variance_threshold}, "
                  f"spear={args.spearman_threshold}, "
                  f"vif={args.vif_threshold}, "
                  f"nonlin={args.nonlinear_threshold}")
            print("="*80)
        
        results = pipeline.run()
        
        # 결과 출력
        print("\n" + "="*80)
        print("PIPELINE COMPLETE")
        print("="*80)
        print(f"Initial descriptors:  {results.get('initial_count', 'N/A'):>6}")
        print(f"Final descriptors:    {results.get('final_count', 'N/A'):>6}")
        if 'initial_count' in results and 'final_count' in results:
            reduction = 100 * (1 - results['final_count'] / results['initial_count'])
            print(f"Reduction:            {reduction:>5.1f}%")
        print("="*80)
        print(f"\n✅ Results saved to: {args.output_dir}")
        
        return 0
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user.")
        return 130
    
    except Exception as e:
        print(f"\n❌ Error: {str(e)}", file=sys.stderr)
        if verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
