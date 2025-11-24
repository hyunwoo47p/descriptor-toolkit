#!/usr/bin/env python3
"""
Molecular Descriptor Toolkit (MDT) - Command Line Interface
============================================================

Unified CLI for molecular descriptor processing with GPU acceleration

Usage:
    # Full workflow
    mdt run --input data/ --output results/
    
    # Preprocessing
    mdt preprocess xml-to-parquet --input data.xml --output data.parquet
    mdt preprocess generate-schema --input data/ --output schema.json
    mdt preprocess calculate-descriptors --input data.parquet --schema schema.json
    
    # Filtering (step-by-step)
    mdt filter pass0 --input data.parquet --output results/
    mdt filter pass1 --input data.parquet --output results/
    mdt filter pass234 --input data.parquet --output results/
    mdt filter all --input data.parquet --output results/  # Pass 0-4
"""

import sys
import argparse
from pathlib import Path

from molecular_descriptor_toolkit import __version__
from molecular_descriptor_toolkit.config import Config


def create_parser():
    """Create main argument parser with GPU as default"""
    parser = argparse.ArgumentParser(
        prog='mdt',
        description='Molecular Descriptor Toolkit - GPU-Accelerated Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full workflow (GPU accelerated by default)
  mdt run --parquet-glob "data/*.parquet" --output-dir results/
  
  # Force CPU mode
  mdt run --parquet-glob "data/*.parquet" --output-dir results/ --cpu
  
  # Filtering steps
  mdt filter pass0 --parquet-glob "data/*.parquet" --output-dir results/
  mdt filter pass1 --parquet-glob "data/*.parquet" --output-dir results/
  mdt filter pass234 --parquet-glob "data/*.parquet" --output-dir results/
  mdt filter all --parquet-glob "data/*.parquet" --output-dir results/

For more information, visit: https://github.com/your-repo
        """
    )
    
    parser.add_argument('--version', action='version', version=f'%(prog)s {__version__}')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # ===== Main run command =====
    run_parser = subparsers.add_parser(
        'run',
        help='Run full pipeline (Pass 0-4)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    add_common_args(run_parser)
    add_all_pass_args(run_parser)
    
    # ===== Filter command with subcommands =====
    filter_parser = subparsers.add_parser(
        'filter',
        help='Run filtering pipeline (separate passes)'
    )
    filter_subparsers = filter_parser.add_subparsers(dest='pass_name', help='Filter pass to run')
    
    # Pass0
    pass0_parser = filter_subparsers.add_parser('pass0', help='Pass 0: Sampling')
    add_common_args(pass0_parser)
    add_pass0_args(pass0_parser)
    
    # Pass1
    pass1_parser = filter_subparsers.add_parser('pass1', help='Pass 1: Statistics + Variance')
    add_common_args(pass1_parser)
    add_pass1_args(pass1_parser)
    
    # Pass2+3+4
    pass234_parser = filter_subparsers.add_parser('pass234', help='Pass 2+3+4: Correlation + VIF + Nonlinear')
    add_common_args(pass234_parser)
    add_pass234_args(pass234_parser)
    
    # All passes
    all_parser = filter_subparsers.add_parser('all', help='All passes (0-4)')
    add_common_args(all_parser)
    add_all_pass_args(all_parser)
    
    # ===== Preprocessing command =====
    preprocess_parser = subparsers.add_parser(
        'preprocess',
        help='Data preprocessing operations'
    )
    preprocess_subparsers = preprocess_parser.add_subparsers(
        dest='operation',
        help='Preprocessing operation'
    )
    
    # XML to Parquet
    xml_parser = preprocess_subparsers.add_parser(
        'xml-to-parquet',
        help='Convert PubChem XML to Parquet'
    )
    xml_parser.add_argument('--input', required=True, help='Input XML file or directory')
    xml_parser.add_argument('--output', required=True, help='Output parquet file or directory')
    xml_parser.add_argument('--filter-property', help='Property for filtering (e.g., H-Bond Donor Count)')
    xml_parser.add_argument('--filter-min', type=float, help='Minimum value for filter')
    xml_parser.add_argument('--filter-max', type=float, help='Maximum value for filter')
    
    # Generate schema
    schema_parser = preprocess_subparsers.add_parser(
        'generate-schema',
        help='Generate descriptor schema'
    )
    schema_parser.add_argument('--input', required=True, help='Input directory with sample files')
    schema_parser.add_argument('--output', required=True, help='Output schema JSON file')
    schema_parser.add_argument('--sample', type=int, default=None, help='Number of files to sample')
    schema_parser.add_argument('--quick', action='store_true', help='Quick mode (fewer molecules per file)')
    
    # Calculate descriptors
    calc_parser = preprocess_subparsers.add_parser(
        'calculate-descriptors',
        help='Calculate molecular descriptors'
    )
    calc_parser.add_argument('--input', required=True, help='Input parquet file(s)')
    calc_parser.add_argument('--output', required=True, help='Output directory')
    calc_parser.add_argument('--schema', required=True, help='Schema JSON file')
    calc_parser.add_argument('--timeout', type=int, default=30, help='Per-molecule timeout (seconds)')
    
    return parser


def add_common_args(parser):
    """Add common arguments (GPU mode, input/output)"""
    # Input/Output
    parser.add_argument('--parquet-glob', required=True, help='Glob pattern for parquet files')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    
    # Device (GPU by default)
    device_group = parser.add_mutually_exclusive_group()
    device_group.add_argument('--cpu', action='store_true', help='Force CPU mode (GPU is default)')
    device_group.add_argument('--gpu-id', type=int, default=0, help='GPU device ID (default: 0)')
    
    # System
    parser.add_argument('--no-checkpoint', action='store_true', help='Disable checkpoint/resume')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--random-seed', type=int, default=42, help='Random seed')


def add_pass0_args(parser):
    """Add Pass 0 specific arguments"""
    parser.add_argument('--sample-per-file', type=int, help='Sample N rows per file')
    parser.add_argument('--file-independent-sampling', action='store_true')


def add_pass1_args(parser):
    """Add Pass 1 specific arguments"""
    parser.add_argument('--variance-threshold', type=float, default=0.002)
    parser.add_argument('--max-missing-ratio', type=float, default=0.5)
    parser.add_argument('--min-effective-n', type=int, default=100)
    parser.add_argument('--range-mode', choices=['minmax', 'trimmed', 'iqr'], default='trimmed')


def add_pass234_args(parser):
    """Add Pass 2+3+4 specific arguments"""
    parser.add_argument('--spearman-threshold', type=float, default=0.95)
    parser.add_argument('--vif-threshold', type=float, default=10.0)
    parser.add_argument('--nonlinear-threshold', type=float, default=0.3)


def add_all_pass_args(parser):
    """Add all pass arguments"""
    add_pass0_args(parser)
    add_pass1_args(parser)
    add_pass234_args(parser)


def main():
    """Main entry point"""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Handle commands
    if args.command == 'run':
        return run_full_pipeline(args)
    elif args.command == 'filter':
        return run_filter(args)
    elif args.command == 'preprocess':
        return run_preprocess(args)
    else:
        parser.print_help()
        return 1


def run_full_pipeline(args):
    """Run full pipeline (all passes)"""
    from molecular_descriptor_toolkit.filtering import DescriptorPipeline
    
    # Create config
    config = Config(
        parquet_glob=args.parquet_glob,
        output_dir=args.output_dir,
        prefer_gpu=not args.cpu,
        gpu_id=args.gpu_id,
        checkpoint=not args.no_checkpoint,
        verbose=args.verbose,
        random_seed=args.random_seed,
        sample_per_file=getattr(args, 'sample_per_file', None),
        variance_threshold=getattr(args, 'variance_threshold', 0.002),
        spearman_threshold=getattr(args, 'spearman_threshold', 0.95),
        vif_threshold=getattr(args, 'vif_threshold', 10.0),
        nonlinear_threshold=getattr(args, 'nonlinear_threshold', 0.3),
    )
    
    print(f"🚀 Molecular Descriptor Toolkit v{__version__}")
    print(f"📊 Mode: {'GPU' if config.using_gpu else 'CPU'}")
    print(f"📂 Input: {args.parquet_glob}")
    print(f"📁 Output: {args.output_dir}")
    print("="*70)
    
    # Run pipeline
    pipeline = DescriptorPipeline(config)
    result = pipeline.run()
    
    print("\n" + "="*70)
    print("✅ Pipeline completed successfully!")
    print(f"📊 Final descriptors: {len(result['final_columns'])}")
    print(f"📁 Results saved to: {args.output_dir}")
    
    return 0


def run_filter(args):
    """Run filtering passes"""
    from molecular_descriptor_toolkit.filtering import DescriptorPipeline
    
    if not args.pass_name:
        print("Error: Please specify a pass to run (pass0, pass1, pass234, all)")
        return 1
    
    # Create config
    config = Config(
        parquet_glob=args.parquet_glob,
        output_dir=args.output_dir,
        prefer_gpu=not args.cpu,
        gpu_id=args.gpu_id,
        checkpoint=not args.no_checkpoint,
        verbose=args.verbose,
        random_seed=args.random_seed,
    )
    
    # Set pass-specific parameters
    if hasattr(args, 'sample_per_file'):
        config.sample_per_file = args.sample_per_file
    if hasattr(args, 'variance_threshold'):
        config.variance_threshold = args.variance_threshold
    if hasattr(args, 'spearman_threshold'):
        config.spearman_threshold = args.spearman_threshold
    if hasattr(args, 'vif_threshold'):
        config.vif_threshold = args.vif_threshold
    if hasattr(args, 'nonlinear_threshold'):
        config.nonlinear_threshold = args.nonlinear_threshold
    
    print(f"🚀 Molecular Descriptor Toolkit v{__version__}")
    print(f"📊 Mode: {'GPU' if config.using_gpu else 'CPU'}")
    print(f"🔧 Pass: {args.pass_name}")
    print("="*70)
    
    # Run specific pass
    pipeline = DescriptorPipeline(config)
    
    if args.pass_name == 'pass0':
        result = pipeline.run_pass0()
        print(f"\n✅ Pass 0 completed: {result}")
    elif args.pass_name == 'pass1':
        columns, stats, indices = pipeline.run_pass1()
        print(f"\n✅ Pass 1 completed: {len(columns)} columns remaining")
    elif args.pass_name == 'pass234':
        columns = pipeline.run_pass234()
        print(f"\n✅ Pass 2+3+4 completed: {len(columns)} final descriptors")
    elif args.pass_name == 'all':
        result = pipeline.run()
        print(f"\n✅ All passes completed: {len(result['final_columns'])} final descriptors")
    
    return 0


def run_preprocess(args):
    """Run preprocessing operations"""
    if not args.operation:
        print("Error: Please specify a preprocessing operation")
        return 1
    
    if args.operation == 'xml-to-parquet':
        print("XML to Parquet conversion")
        print(f"Input: {args.input}")
        print(f"Output: {args.output}")
        # Import and run xml_parser
        import subprocess
        cmd = [
            'python', '-m', 'molecular_descriptor_toolkit.preprocessing.xml_parser',
            '--input', args.input,
            '--output', args.output
        ]
        if args.filter_property:
            cmd.extend(['--filter-property', args.filter_property])
        if args.filter_min is not None:
            cmd.extend(['--filter-min', str(args.filter_min)])
        if args.filter_max is not None:
            cmd.extend(['--filter-max', str(args.filter_max)])
        return subprocess.call(cmd)
    
    elif args.operation == 'generate-schema':
        print("Generating descriptor schema")
        print(f"Input: {args.input}")
        print(f"Output: {args.output}")
        # Import and run schema_generator
        import subprocess
        cmd = [
            'python', '-m', 'molecular_descriptor_toolkit.preprocessing.schema_generator',
            '-i', args.input,
            '-o', args.output
        ]
        if args.sample:
            cmd.extend(['--sample', str(args.sample)])
        if args.quick:
            cmd.append('--quick')
        return subprocess.call(cmd)
    
    elif args.operation == 'calculate-descriptors':
        print("Calculating descriptors")
        print(f"Input: {args.input}")
        print(f"Schema: {args.schema}")
        print(f"Output: {args.output}")
        # Import and run descriptor_calculator
        import subprocess
        cmd = [
            'python', '-m', 'molecular_descriptor_toolkit.preprocessing.descriptor_calculator',
            '--input', args.input,
            '--output', args.output,
            '--schema', args.schema,
            '--timeout', str(args.timeout)
        ]
        return subprocess.call(cmd)
    
    return 1


if __name__ == '__main__':
    sys.exit(main())
