#!/usr/bin/env python3
"""
ChemDescriptorML (CDML) - Command Line Interface
============================================================

Unified CLI for molecular descriptor processing with GPU acceleration

Usage:
    # Full workflow
    cdml run --input data/ --output results/
    
    # Preprocessing
    cdml preprocess xml-to-parquet --input data.xml --output data.parquet
    cdml preprocess generate-schema --input data/ --output schema.json
    cdml preprocess calculate-descriptors --input data.parquet --schema schema.json
    
    # Filtering (step-by-step)
    cdml filter pass0 --input data.parquet --output results/
    cdml filter pass1 --input data.parquet --output results/
    cdml filter pass234 --input data.parquet --output results/
    cdml filter all --input data.parquet --output results/  # Pass 0-4
"""

import sys
import argparse
from pathlib import Path

from Chem_Descriptor_ML import __version__
from Chem_Descriptor_ML.config import (
    Config,
    IOConfig,
    DeviceConfig,
    FilteringConfig,
    SystemConfig,
)


def create_parser():
    """Create main argument parser with GPU as default"""
    parser = argparse.ArgumentParser(
        prog='cdml',
        description='ChemDescriptorML - GPU-Accelerated Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full workflow (GPU accelerated by default)
  cdml run --parquet-glob "data/*.parquet" --output-dir results/
  
  # Force CPU mode
  cdml run --parquet-glob "data/*.parquet" --output-dir results/ --cpu
  
  # Filtering steps
  cdml filter pass0 --parquet-glob "data/*.parquet" --output-dir results/
  cdml filter pass1 --parquet-glob "data/*.parquet" --output-dir results/
  cdml filter pass234 --parquet-glob "data/*.parquet" --output-dir results/
  cdml filter all --parquet-glob "data/*.parquet" --output-dir results/

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

    # ===== Process-all command (XML/SMILES → Descriptors → Filtering) =====
    process_all_parser = subparsers.add_parser(
        'process-all',
        help='Run full pipeline: [XML →] SMILES → Descriptor calculation → Filtering',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    process_all_parser.add_argument('--input', required=True,
        help='Input file: XML (.xml/.xml.gz), CSV, or Parquet with SMILES')
    process_all_parser.add_argument('--output-dir', required=True, help='Output directory for all results')
    process_all_parser.add_argument('--smiles-col', default='SMILES::Absolute', help='SMILES column name')
    process_all_parser.add_argument('--id-col', default='CID', help='ID column name')
    process_all_parser.add_argument('--schema', help='Schema JSON (auto-generated if not provided)')
    process_all_parser.add_argument('--mol-timeout', type=int, default=30, help='Per-molecule timeout (seconds)')
    process_all_parser.add_argument('--cpu', action='store_true', help='Force CPU mode')
    process_all_parser.add_argument('--gpu-id', type=int, default=0, help='GPU device ID')
    process_all_parser.add_argument('--verbose', action='store_true', help='Verbose output')
    process_all_parser.add_argument('--no-checkpoint', action='store_true', help='Disable checkpoint/resume')
    process_all_parser.add_argument('--random-seed', type=int, default=42, help='Random seed')
    # XML filtering options
    process_all_parser.add_argument('--filter-property', help='XML: Property for filtering (e.g., "H-Bond Donor Count")')
    process_all_parser.add_argument('--filter-min', type=float, help='XML: Minimum value for filter')
    process_all_parser.add_argument('--filter-max', type=float, help='XML: Maximum value for filter')
    # Filtering parameters
    process_all_parser.add_argument('--variance-threshold', type=float, default=0.002)
    process_all_parser.add_argument('--spearman-threshold', type=float, default=0.95)
    process_all_parser.add_argument('--vif-threshold', type=float, default=10.0)
    process_all_parser.add_argument('--nonlinear-threshold', type=float, default=0.3)

    # ===== Train command (ML Training) =====
    train_parser = subparsers.add_parser(
        'train',
        help='Train ML models using filtered descriptors',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    train_parser.add_argument('--input', required=True,
        help='Input training data file (.parquet or .csv) with descriptors and target')
    train_parser.add_argument('--test-input', default=None,
        help='Optional separate test data file (.parquet or .csv). If provided, --test-size is ignored')
    train_parser.add_argument('--cluster-info',
        help='Path to final_cluster_info.json for cluster-aware descriptor selection')
    train_parser.add_argument('--target-col', default='pLeach',
        help='Target column name for prediction')
    train_parser.add_argument('--output-dir', default='ml_output',
        help='Output directory for ML results')
    train_parser.add_argument('--descriptor-sizes', default='5,10,15,20,30,40,50',
        help='Comma-separated list of descriptor counts to try')
    train_parser.add_argument('--descriptor-mode', default='sequential',
        choices=['sequential', 'representative', 'random_alternative', 'mixed'],
        help='Descriptor selection mode: sequential (original order), representative, random_alternative, mixed')
    train_parser.add_argument('--models', default=None,
        help='Comma-separated list of models (default: all available)')
    train_parser.add_argument('--test-size', type=float, default=0.2,
        help='Hold-out test set ratio (ignored if --test-input is provided)')
    train_parser.add_argument('--cv-folds', type=int, default=5,
        help='K-Fold cross-validation folds')
    train_parser.add_argument('--random-seed', type=int, default=42,
        help='Random seed for reproducibility')
    train_parser.add_argument('--regularization', action='store_true',
        help='Apply strong regularization to models (for small datasets)')
    train_parser.add_argument('--no-plots', action='store_true',
        help='Skip generating plots')

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
    elif args.command == 'process-all':
        return run_process_all(args)
    elif args.command == 'train':
        return run_train(args)
    else:
        parser.print_help()
        return 1


def run_full_pipeline(args):
    """Run full pipeline (all passes)"""
    from Chem_Descriptor_ML.filtering import DescriptorPipeline

    # Create config with section-based structure
    config = Config(
        io=IOConfig(
            parquet_glob=args.parquet_glob,
            output_dir=args.output_dir,
        ),
        device=DeviceConfig(
            prefer_gpu=not args.cpu,
            gpu_id=args.gpu_id,
        ),
        filtering=FilteringConfig(
            sample_per_file=getattr(args, 'sample_per_file', None),
            variance_threshold=getattr(args, 'variance_threshold', 0.002),
            max_missing_ratio=getattr(args, 'max_missing_ratio', 0.5),
            min_effective_n=getattr(args, 'min_effective_n', 100),
            spearman_threshold=getattr(args, 'spearman_threshold', 0.95),
            vif_threshold=getattr(args, 'vif_threshold', 10.0),
            nonlinear_threshold=getattr(args, 'nonlinear_threshold', 0.3),
        ),
        system=SystemConfig(
            checkpoint=not args.no_checkpoint,
            verbose=args.verbose,
            random_seed=args.random_seed,
        ),
    )

    # Validate and finalize (auto-detect device)
    config.validate_and_finalize()

    print(f"🚀 ChemDescriptorML v{__version__}")
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
    from Chem_Descriptor_ML.filtering import DescriptorPipeline

    if not args.pass_name:
        print("Error: Please specify a pass to run (pass0, pass1, pass234, all)")
        return 1

    # Create config with section-based structure
    config = Config(
        io=IOConfig(
            parquet_glob=args.parquet_glob,
            output_dir=args.output_dir,
        ),
        device=DeviceConfig(
            prefer_gpu=not args.cpu,
            gpu_id=args.gpu_id,
        ),
        filtering=FilteringConfig(
            sample_per_file=getattr(args, 'sample_per_file', None),
            variance_threshold=getattr(args, 'variance_threshold', 0.002),
            max_missing_ratio=getattr(args, 'max_missing_ratio', 0.5),
            min_effective_n=getattr(args, 'min_effective_n', 100),
            spearman_threshold=getattr(args, 'spearman_threshold', 0.95),
            vif_threshold=getattr(args, 'vif_threshold', 10.0),
            nonlinear_threshold=getattr(args, 'nonlinear_threshold', 0.3),
        ),
        system=SystemConfig(
            checkpoint=not args.no_checkpoint,
            verbose=args.verbose,
            random_seed=args.random_seed,
        ),
    )

    # Validate and finalize (auto-detect device)
    config.validate_and_finalize()

    print(f"🚀 ChemDescriptorML v{__version__}")
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
            'python', '-m', 'Chem_Descriptor_ML.preprocessing.xml_parser',
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
            'python', '-m', 'Chem_Descriptor_ML.preprocessing.schema_generator',
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
            'python', '-m', 'Chem_Descriptor_ML.preprocessing.descriptor_calculator',
            '--input', args.input,
            '--output', args.output,
            '--schema', args.schema,
            '--timeout', str(args.timeout)
        ]
        return subprocess.call(cmd)
    
    return 1


def run_process_all(args):
    """
    Run full pipeline: [XML →] SMILES → Descriptor calculation → Filtering

    This command combines:
    0. XML parsing (if input is XML)
    1. Schema generation (if not provided)
    2. Descriptor calculation (RDKit + Mordred)
    3. Filtering pipeline (Pass 1-4)

    Input format is auto-detected by file extension:
    - .xml / .xml.gz → PubChem XML (extracts SMILES, InChI, etc.)
    - .csv → CSV with SMILES column
    - .parquet → Parquet with SMILES column
    """
    import subprocess
    from pathlib import Path
    from Chem_Descriptor_ML.filtering import DescriptorPipeline
    import shutil

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = Path(args.input)

    # Create subdirectories
    outputs_dir = output_dir / "outputs"
    tmp_dir = output_dir / "tmp"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Detect input format
    input_lower = str(input_path).lower()
    is_xml = input_lower.endswith('.xml') or input_lower.endswith('.xml.gz')

    print(f"🚀 ChemDescriptorML v{__version__}")
    print("=" * 70)
    if is_xml:
        print("📌 Running full process: XML → SMILES → Descriptors → Filtering")
    else:
        print("📌 Running full process: SMILES → Descriptors → Filtering")
    print("=" * 70)

    # Determine step count
    total_steps = 5 if is_xml else 4  # Added cleanup step
    current_step = 0

    # Step 0 (XML only): Parse XML to extract SMILES/InChI
    if is_xml:
        current_step += 1
        smiles_path = tmp_dir / "extracted_molecules.parquet"
        print(f"\n📄 Step {current_step}/{total_steps}: Parsing PubChem XML...")
        print(f"   Input: {input_path}")

        cmd = [
            'python', '-m', 'Chem_Descriptor_ML.preprocessing.xml_parser',
            '--input', str(input_path),
            '--output', str(smiles_path)
        ]
        # Add filtering options if provided
        if getattr(args, 'filter_property', None):
            cmd.extend(['--filter-property', args.filter_property])
        if getattr(args, 'filter_min', None) is not None:
            cmd.extend(['--filter-min', str(args.filter_min)])
        if getattr(args, 'filter_max', None) is not None:
            cmd.extend(['--filter-max', str(args.filter_max)])

        result = subprocess.call(cmd)
        if result != 0:
            print("❌ XML parsing failed")
            return 1
        print(f"   ✓ Molecules extracted to: {smiles_path}")

        # Update input path for subsequent steps
        descriptor_input = smiles_path
        # Use PubChem's SMILES column name
        smiles_col = 'SMILES::Absolute'
    else:
        descriptor_input = input_path
        smiles_col = args.smiles_col

    # Step 1: Schema generation (if not provided)
    current_step += 1
    if args.schema:
        schema_path = Path(args.schema)
        print(f"\n📋 Step {current_step}/{total_steps}: Using provided schema: {schema_path}")
    else:
        schema_path = tmp_dir / "descriptor_schema.json"
        print(f"\n📋 Step {current_step}/{total_steps}: Generating descriptor schema...")

        cmd = [
            'python', '-m', 'Chem_Descriptor_ML.preprocessing.schema_generator',
            '--input', str(descriptor_input.parent if descriptor_input.is_file() else descriptor_input),
            '--output', str(schema_path),
            '--quick'
        ]
        result = subprocess.call(cmd)
        if result != 0:
            print("❌ Schema generation failed")
            return 1
        print(f"   ✓ Schema saved to: {schema_path}")

    # Step 2: Descriptor calculation
    current_step += 1
    descriptors_path = tmp_dir / "descriptors.parquet"
    print(f"\n⚗️  Step {current_step}/{total_steps}: Calculating molecular descriptors...")

    cmd = [
        'python', '-m', 'Chem_Descriptor_ML.preprocessing.descriptor_calculator',
        '--input', str(descriptor_input),
        '--output', str(descriptors_path),
        '--schema', str(schema_path),
        '--smiles-col', smiles_col,
        '--id-col', args.id_col,
        '--format', 'parquet',
        '--n-jobs', '1'
    ]
    if args.mol_timeout:
        cmd.extend(['--mol-timeout', str(args.mol_timeout)])

    result = subprocess.call(cmd)
    if result != 0:
        print("❌ Descriptor calculation failed")
        return 1
    print(f"   ✓ Descriptors saved to: {descriptors_path}")

    # Step 3 (or 4): Filtering pipeline
    current_step += 1
    filtering_tmp = tmp_dir / "filtering"
    print(f"\n🔬 Step {current_step}/{total_steps}: Running filtering pipeline...")

    config = Config(
        io=IOConfig(
            parquet_glob=str(descriptors_path),
            output_dir=str(filtering_tmp),
        ),
        device=DeviceConfig(
            prefer_gpu=not args.cpu,
            gpu_id=args.gpu_id,
        ),
        filtering=FilteringConfig(
            variance_threshold=args.variance_threshold,
            spearman_threshold=args.spearman_threshold,
            vif_threshold=args.vif_threshold,
            nonlinear_threshold=args.nonlinear_threshold,
        ),
        system=SystemConfig(
            checkpoint=not args.no_checkpoint,
            verbose=args.verbose,
            random_seed=args.random_seed,
        ),
    )

    # Validate and finalize (auto-detect device)
    config.validate_and_finalize()

    print(f"   Mode: {'GPU' if config.using_gpu else 'CPU'}")

    pipeline = DescriptorPipeline(config)
    pipeline_result = pipeline.run()

    # Step 4 (or 5): Organize output files
    current_step += 1
    print(f"\n📁 Step {current_step}/{total_steps}: Organizing output files...")

    # Move main outputs to outputs/ folder
    final_outputs = {
        'descriptor_schema.json': schema_path,
        'descriptors.parquet': descriptors_path,
        'final_cluster_info.json': filtering_tmp / 'final_cluster_info.json',
        'final_descriptors.txt': filtering_tmp / 'final_descriptors.txt',
    }

    for dest_name, src_path in final_outputs.items():
        dest_path = outputs_dir / dest_name
        if src_path.exists():
            shutil.copy2(src_path, dest_path)
            print(f"   ✓ {dest_name}")

    # Clean up descriptors.parts (intermediate chunks no longer needed)
    parts_dir = tmp_dir / "descriptors.parts"
    if parts_dir.exists() and descriptors_path.exists():
        shutil.rmtree(parts_dir)

    print(f"   ✓ Main outputs saved to: {outputs_dir}")
    print(f"   ✓ Intermediate files in: {tmp_dir}")

    # Summary
    print("\n" + "=" * 70)
    print("✅ Full Pipeline Completed!")
    print("=" * 70)
    print(f"📂 Input: {args.input}")
    print(f"📁 Output directory: {output_dir}")
    print()
    print("📦 Main outputs (outputs/):")
    print(f"   • descriptor_schema.json  - Descriptor schema")
    print(f"   • descriptors.parquet     - Calculated descriptors")
    print(f"   • final_cluster_info.json - Cluster structure info")
    print(f"   • final_descriptors.txt   - Final descriptor list ({len(pipeline_result['final_columns'])} descriptors)")
    print()
    print("🗂️  Intermediate files (tmp/):")
    print(f"   • pass1~4 results, correlation matrix, etc.")
    print("=" * 70)

    return 0


def run_train(args):
    """Run ML training with cluster-aware descriptor selection"""
    from Chem_Descriptor_ML.ml import OptimalMLEnsemble

    print("=" * 70)
    print("🤖 ChemDescriptorML - ML Training")
    print("=" * 70)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}")
        return 1

    # Check test input if provided
    test_input_path = None
    if args.test_input:
        test_input_path = Path(args.test_input)
        if not test_input_path.exists():
            print(f"❌ Test input file not found: {test_input_path}")
            return 1

    # Parse descriptor sizes
    descriptor_sizes = [int(x.strip()) for x in args.descriptor_sizes.split(',')]

    # Parse models if specified
    models = None
    if args.models:
        models = [x.strip() for x in args.models.split(',')]

    # Initialize ensemble
    print(f"\n📊 Loading data:")
    print(f"   Train data: {input_path}")
    if test_input_path:
        print(f"   Test data: {test_input_path} (external)")
    else:
        print(f"   Test split: {args.test_size*100:.0f}% (random)")
    print(f"   Target column: {args.target_col}")

    cluster_info_path = args.cluster_info if args.cluster_info else None
    if cluster_info_path:
        print(f"   Cluster info: {cluster_info_path}")

    try:
        ensemble = OptimalMLEnsemble(
            data_path=str(input_path),
            test_data_path=str(test_input_path) if test_input_path else None,
            target_col=args.target_col,
            cluster_info_path=cluster_info_path,
            test_size=args.test_size,
            cv_folds=args.cv_folds,
            random_state=args.random_seed,
            use_regularization=args.regularization
        )
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        return 1

    # Train all models
    print(f"\n🏋️ Training models...")
    print(f"   Descriptor sizes: {descriptor_sizes}")
    print(f"   Descriptor mode: {args.descriptor_mode}")

    try:
        ensemble.train_all_models(
            descriptor_sizes=descriptor_sizes,
            descriptor_mode=args.descriptor_mode,
            models=models
        )
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Find best model
    print("\n🏆 Finding best model...")
    best = ensemble.find_best_model(metric='holdout_r2')

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n💾 Saving results to: {output_dir}")
    ensemble.save_results(output_dir=str(output_dir))

    # Generate plots
    if not args.no_plots:
        print("\n📈 Generating plots...")
        try:
            ensemble.plot_model_comparison(output_dir=str(output_dir))
        except Exception as e:
            print(f"   ⚠️ Plot generation failed: {e}")

    # Summary
    print("\n" + "=" * 70)
    print("✅ ML Training Completed!")
    print("=" * 70)
    print(f"📁 Output directory: {output_dir}")
    print()
    print("📊 Best Model:")
    print(f"   • Model: {best['model_name']}")
    print(f"   • Descriptors: {best['n_descriptors']}")
    print(f"   • K-Fold R²: {best['kfold_r2_mean']:.4f} ± {best['kfold_r2_std']:.4f}")
    print(f"   • Hold-Out R²: {best['holdout_r2']:.4f}")
    print(f"   • Overfitting Gap: {best['overfitting_gap']:.4f}")
    print()
    print("📦 Output files:")
    print(f"   • ml_results.json        - All experiment results")
    print(f"   • best_model.json        - Best model details")
    if not args.no_plots:
        print(f"   • *.png                  - Comparison plots")
    print("=" * 70)

    return 0


if __name__ == '__main__':
    sys.exit(main())
