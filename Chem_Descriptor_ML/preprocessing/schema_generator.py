#!/usr/bin/env python
"""
ChemDescriptorML (CDML) - Schema Generator

Generates a master schema by collecting ALL unique descriptor columns
from a sample of input files. The schema ensures consistent columns
across all descriptor calculation runs.

Usage:
    cdml preprocess generate-schema -i /path/to/files/ -o master_schema.json
    
    # Generate from specific number of sample files
    python generate_master_schema.py -i /path/to/files/ -o master_schema.json --sample 10
    
    # Quick mode: only calculate descriptors for a small subset of molecules per file
    python generate_master_schema.py -i /path/to/files/ -o master_schema.json --quick --molecules-per-file 100
"""
import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import List, Set, Optional
import pandas as pd
import numpy as np
from collections import OrderedDict

from rdkit import Chem
from rdkit.Chem import Descriptors as RDKitDescriptors
from rdkit.Chem import Fragments as RDKitFragments

# numpy compatibility for mordred
if not hasattr(np, "product"):
    np.product = np.prod

from mordred import Calculator, descriptors as mordred_descriptors

# silence RDKit warnings
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.warning")


# -------- SHARED UTILITY: Column name normalization --------
def normalize_column_names(columns: List[str]) -> List[str]:
    """
    CRITICAL SHARED FUNCTION: Normalize column names with consistent duplicate handling
    
    This function MUST be identical in both:
    - generate_master_schema.py (schema generation)
    - make_descriptors_fixed.py (actual calculation)
    
    Rules:
    1. First occurrence: keep as-is
    2. Second occurrence: add "_1" suffix
    3. Third occurrence: add "_2" suffix, etc.
    
    Example: ["ABC", "ABC", "DEF", "ABC"] -> ["ABC", "ABC_1", "DEF", "ABC_2"]
    """
    seen = {}
    normalized = []
    
    for col in columns:
        if col in seen:
            seen[col] += 1
            normalized.append(f"{col}_{seen[col]}")
        else:
            seen[col] = 0
            normalized.append(col)
    
    return normalized


def flatten_multiindex_columns(columns) -> List[str]:
    """
    CRITICAL SHARED FUNCTION: Flatten MultiIndex columns consistently
    
    Must be identical to implementation in make_descriptors_fixed.py
    """
    if isinstance(columns, pd.MultiIndex):
        return ["_".join(map(str, c)).strip() for c in columns.values]
    else:
        return [str(c) for c in columns]


def get_rdkit_descriptor_names(include_fragments: bool = True) -> List[str]:
    """Get all RDKit descriptor names"""
    rdkit_descriptors = list(RDKitDescriptors.descList)
    names = [name for name, _ in rdkit_descriptors]
    
    if include_fragments:
        seen = set(names)
        for k in dir(RDKitDescriptors):
            if k.startswith("fr_") and k not in seen:
                fn = getattr(RDKitDescriptors, k, None)
                if callable(fn):
                    names.append(k)
                    seen.add(k)
        
        for k in dir(RDKitFragments):
            if k.startswith("fr_") and k not in seen:
                fn = getattr(RDKitFragments, k, None)
                if callable(fn):
                    names.append(k)
                    seen.add(k)
    
    return names


def get_mordred_descriptor_names(use_3d: bool = False) -> List[str]:
    """
    Get all Mordred descriptor names by calculating on a sample molecule
    Uses SHARED normalization functions for consistency
    """
    print("Initializing Mordred calculator to discover all descriptor names...")
    
    calc = Calculator(mordred_descriptors, ignore_3D=not use_3d)
    
    # Use a simple molecule to get all possible descriptor names
    sample_mol = Chem.MolFromSmiles("CCO")  # Ethanol
    
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with np.errstate(all="ignore"):
                # Calculate on sample molecule
                df = calc.pandas([sample_mol])
        
        # FIX 1: Use SHARED flatten function
        columns = flatten_multiindex_columns(df.columns)
        
        # FIX 1: Use SHARED normalization function
        unique_columns = normalize_column_names(columns)
        
        print(f"Found {len(unique_columns)} Mordred descriptor names")
        return unique_columns
        
    except Exception as e:
        print(f"Error getting Mordred descriptor names: {e}")
        return []


def remove_overlapping_descriptors(mordred_names: List[str], rdkit_names: List[str]) -> List[str]:
    """Remove Mordred descriptors that overlap with RDKit"""
    rdkit_lower = {col.lower() for col in rdkit_names}
    unique_mordred = []
    removed = []
    
    for col in mordred_names:
        if col.lower() not in rdkit_lower:
            unique_mordred.append(col)
        else:
            removed.append(col)
    
    if removed:
        print(f"Removed {len(removed)} overlapping Mordred descriptors")
        if len(removed) <= 10:
            print(f"Removed: {removed}")
        else:
            print(f"Sample removed: {removed[:5]}...")
    
    return unique_mordred


def collect_descriptors_from_sample(input_files: List[Path], 
                                    sample_size: Optional[int],
                                    molecules_per_file: int,
                                    desc_set: str,
                                    include_fragments: bool,
                                    use_3d: bool,
                                    random_sample: bool = False) -> Set[str]:
    """
    Collect all unique descriptor names from sample files
    
    Strategy: Calculate descriptors on a small sample of molecules from each file
    to discover which descriptors can actually be calculated
    
    Uses SHARED normalization functions for consistency
    
    Args:
        random_sample: If True, randomly select only 1 molecule per file (fast mode)
    """
    print("\n" + "=" * 70)
    print("COLLECTING DESCRIPTOR NAMES FROM SAMPLE DATA")
    if random_sample:
        print("MODE: Random sampling (1 molecule per file)")
    print("=" * 70)
    
    all_descriptor_columns = set()
    
    # Sample files if requested
    if sample_size and sample_size < len(input_files):
        import random
        sample_files = random.sample(input_files, sample_size)
        print(f"Sampling {sample_size} files out of {len(input_files)} total files")
    else:
        sample_files = input_files
        print(f"Processing all {len(input_files)} files")
    
    for i, file_path in enumerate(sample_files, 1):
        print(f"\n[{i}/{len(sample_files)}] Processing {file_path.name}...")
        
        try:
            # Read sample molecules
            if file_path.suffix.lower() in ('.parquet', '.pq'):
                df = pd.read_parquet(file_path)
            else:
                # For random sample mode, read entire file to enable random selection
                if random_sample:
                    df = pd.read_csv(file_path)
                else:
                    df = pd.read_csv(file_path, nrows=molecules_per_file)
            
            # Limit to requested number of molecules (unless random_sample)
            if not random_sample and len(df) > molecules_per_file:
                df = df.head(molecules_per_file)
            
            print(f"  Loaded {len(df)} molecules from file")
            
            # Get SMILES column (try common names)
            smiles_col = None
            for col in df.columns:
                if 'smiles' in col.lower():
                    smiles_col = col
                    break
            
            if not smiles_col:
                print(f"  WARNING: No SMILES column found, skipping")
                continue
            
            # Parse molecules
            if random_sample:
                # Random mode: try to get 1 valid molecule
                import random
                smiles_list = df[smiles_col].astype(str).tolist()
                # Shuffle to randomize selection
                random.shuffle(smiles_list)
                mols = []
                for smiles in smiles_list:
                    try:
                        mol = Chem.MolFromSmiles(smiles)
                        if mol:
                            mols.append(mol)
                            break  # Got 1 valid molecule, stop
                    except:
                        continue
                
                if not mols:
                    print(f"  WARNING: No valid molecules found in file, skipping")
                    continue
                print(f"  Selected 1 random valid molecule")
            else:
                # Normal mode: parse up to molecules_per_file or 50
                smiles_list = df.sample(n=min(molecules_per_file, len(df)))[smiles_col].astype(str).tolist()
                mols = []
                for smiles in smiles_list:
                    try:
                        mol = Chem.MolFromSmiles(smiles)
                        if mol:
                            mols.append(mol)
                        if len(mols) >= 50:  # Limit to 50 valid molecules
                            break
                    except:
                        continue
                
                if not mols:
                    print(f"  WARNING: No valid molecules parsed, skipping")
                    continue
                
                print(f"  Parsed {len(mols)} valid molecules")
            
            # Calculate descriptors to discover column names
            file_columns = set()
            
            if desc_set in ("rdkit", "both"):
                print(f"  Calculating RDKit descriptors...")
                rdkit_names = get_rdkit_descriptor_names(include_fragments)
                file_columns.update(rdkit_names)
                print(f"  Found {len(rdkit_names)} RDKit descriptors")
            
            if desc_set in ("mordred", "both"):
                print(f"  Calculating Mordred descriptors on sample...")
                
                # Initialize Mordred calculator
                calc = Calculator(mordred_descriptors, ignore_3D=not use_3d)
                
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        with np.errstate(all="ignore"):
                            # Calculate on sample molecules
                            df_mordred = calc.pandas(mols[:10], nproc=1)  # Use only 10 molecules
                    
                    # FIX 1: Use SHARED flatten function
                    mordred_cols = flatten_multiindex_columns(df_mordred.columns)
                    
                    # FIX 1: Use SHARED normalization function
                    unique_mordred_cols = normalize_column_names(mordred_cols)
                    
                    # Remove overlaps with RDKit
                    if desc_set == "both":
                        rdkit_names = get_rdkit_descriptor_names(include_fragments)
                        unique_mordred_cols = remove_overlapping_descriptors(unique_mordred_cols, rdkit_names)
                    
                    file_columns.update(unique_mordred_cols)
                    print(f"  Found {len(unique_mordred_cols)} unique Mordred descriptors")
                    
                except Exception as e:
                    print(f"  WARNING: Mordred calculation failed: {e}")
            
            # Add to global set
            new_columns = file_columns - all_descriptor_columns
            if new_columns:
                print(f"  Added {len(new_columns)} new descriptors to master schema")
                all_descriptor_columns.update(new_columns)
            else:
                print(f"  No new descriptors found (already in schema)")
            
            print(f"  Total unique descriptors so far: {len(all_descriptor_columns)}")
            
        except Exception as e:
            print(f"  ERROR processing {file_path.name}: {e}")
            continue
    
    return all_descriptor_columns


def generate_master_schema_theoretical(desc_set: str, include_fragments: bool, use_3d: bool) -> List[str]:
    """
    Generate master schema theoretically without processing files
    Uses all possible descriptor names from RDKit and Mordred
    Uses SHARED normalization functions for consistency
    """
    print("\n" + "=" * 70)
    print("GENERATING THEORETICAL MASTER SCHEMA (ALL POSSIBLE DESCRIPTORS)")
    print("=" * 70)
    
    metadata_columns = [
        'CID', 
        'isomeric_smiles', 
        'standardized_smiles', 
        'parse_source', 
        'standardization_status',
        'source_file'
    ]
    
    descriptor_columns = []
    
    if desc_set in ("rdkit", "both"):
        print("Getting RDKit descriptor names...")
        rdkit_names = get_rdkit_descriptor_names(include_fragments)
        descriptor_columns.extend(rdkit_names)
        print(f"  Added {len(rdkit_names)} RDKit descriptors")
    
    if desc_set in ("mordred", "both"):
        print("Getting Mordred descriptor names...")
        mordred_names = get_mordred_descriptor_names(use_3d)
        
        # Remove overlaps with RDKit
        if desc_set == "both":
            mordred_names = remove_overlapping_descriptors(mordred_names, descriptor_columns)
        
        descriptor_columns.extend(mordred_names)
        print(f"  Added {len(mordred_names)} Mordred descriptors")
    
    # Combine metadata + descriptors
    all_columns = metadata_columns + descriptor_columns
    
    print(f"\nTotal columns in master schema: {len(all_columns)}")
    print(f"  Metadata columns: {len(metadata_columns)}")
    print(f"  Descriptor columns: {len(descriptor_columns)}")
    
    return all_columns


def save_master_schema(schema: List[str], output_path: Path):
    """Save master schema to JSON file"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(schema, f, ensure_ascii=False, indent=2)
    
    print(f"\nMaster schema saved to: {output_path}")
    print(f"Total columns: {len(schema)}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate master schema for consistent descriptor calculations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Theoretical schema (fastest, all possible descriptors)
  python generate_master_schema.py -o master_schema.json --theoretical
  
  # Random sampling from all files (very fast, recommended for large datasets)
  python generate_master_schema.py -i /data/files/ -o master_schema.json --random-sample
  
  # From sample files (recommended for actual data)
  python generate_master_schema.py -i /data/files/ -o master_schema.json --sample 20
  
  # Quick mode with fewer molecules per file
  python generate_master_schema.py -i /data/files/ -o master_schema.json --quick
        """
    )
    
    p.add_argument("-i", "--input", help="Input file or directory (not needed for --theoretical)")
    p.add_argument("-o", "--output", required=True, help="Output master schema JSON file")
    
    # Schema generation mode
    p.add_argument("--theoretical", action="store_true",
                   help="Generate theoretical schema with ALL possible descriptors (fast, recommended)")
    p.add_argument("--sample", type=int, default=None,
                   help="Number of files to sample (default: all files)")
    p.add_argument("--quick", action="store_true",
                   help="Quick mode: only 100 molecules per file")
    p.add_argument("--random-sample", action="store_true",
                   help="Random mode: select only 1 random molecule per file (fastest, processes all files)")
    p.add_argument("--molecules-per-file", type=int, default=1000,
                   help="Number of molecules to process per file (default: 1000, ignored with --random-sample)")
    
    # Descriptor options
    p.add_argument("--desc-set", choices=["rdkit", "mordred", "both"], default="both",
                   help="Descriptor set (default: both)")
    p.add_argument("--include-fragments", action="store_true",
               help="Include RDKit fragment descriptors")
    p.add_argument("--no-fragments", dest="include_fragments",
               action="store_false", help="Exclude RDKit fragments")
    p.set_defaults(include_fragments=True)
    p.add_argument("--use-3d", action="store_true", default=False,
                   help="Include 3D descriptors")
    
    return p.parse_args()


def main():
    args = parse_args()
    
    print("=" * 70)
    print("MASTER SCHEMA GENERATOR FOR MOLECULAR DESCRIPTORS")
    print("=" * 70)
    print(f"Descriptor set: {args.desc_set}")
    print(f"Include fragments: {args.include_fragments}")
    print(f"Use 3D descriptors: {args.use_3d}")
    print("\nCRITICAL: Using SHARED column normalization logic")
    print("Schema will match make_descriptors_fixed.py output exactly")
    
    # Adjust molecules per file for quick mode or random mode
    if args.random_sample:
        molecules_per_file = 1  # Will be overridden by random selection
        print("\nRandom sampling mode: 1 random molecule per file")
    elif args.quick:
        molecules_per_file = 100
    else:
        molecules_per_file = args.molecules_per_file
    
    if args.theoretical:
        # Theoretical mode - generate from descriptor definitions
        print("\nMode: THEORETICAL (all possible descriptors)")
        schema = generate_master_schema_theoretical(
            args.desc_set, 
            args.include_fragments, 
            args.use_3d
        )
    else:
        # Data-driven mode - collect from actual files
        if not args.input:
            print("ERROR: --input is required when not using --theoretical mode")
            return 1
        
        input_path = Path(args.input)
        
        # Find input files
        if input_path.is_file():
            input_files = [input_path]
        elif input_path.is_dir():
            input_files = []
            for ext in ['.csv', '.parquet', '.pq']:
                input_files.extend(input_path.glob(f"*{ext}"))
            input_files.sort()
        else:
            print(f"ERROR: Input path not found: {input_path}")
            return 1
        
        if not input_files:
            print("ERROR: No supported input files found")
            return 1
        
        print(f"\nMode: DATA-DRIVEN (from actual files)")
        print(f"Found {len(input_files)} input files")
        if args.random_sample:
            print(f"Random sampling mode: 1 random molecule per file (all files processed)")
        elif args.quick:
            print(f"Quick mode: {molecules_per_file} molecules per file")
        else:
            print(f"Processing {molecules_per_file} molecules per file")
        
        # Collect descriptors
        descriptor_columns = collect_descriptors_from_sample(
            input_files,
            args.sample,
            molecules_per_file,
            args.desc_set,
            args.include_fragments,
            args.use_3d,
            args.random_sample  # Pass random_sample flag
        )
        
        # Add metadata columns
        metadata_columns = [
            'CID', 
            'isomeric_smiles', 
            'standardized_smiles', 
            'parse_source', 
            'standardization_status',
            'source_file'
        ]
        
        # Combine: metadata first, then sorted descriptors
        schema = metadata_columns + sorted(descriptor_columns)
        
        print("\n" + "=" * 70)
        print(f"FINAL SCHEMA SUMMARY")
        print("=" * 70)
        print(f"Metadata columns: {len(metadata_columns)}")
        print(f"Descriptor columns: {len(descriptor_columns)}")
        print(f"Total columns: {len(schema)}")
    
    # Save schema
    output_path = Path(args.output)
    save_master_schema(schema, output_path)
    
    print("\n" + "=" * 70)
    print("SUCCESS! Master schema generated")
    print("=" * 70)
    print(f"\nNext step: Use this schema with make_descriptors_fixed.py:")
    print(f"  python make_descriptors_fixed.py \\")
    print(f"    -i <input_files> \\")
    print(f"    -o <output_files> \\")
    print(f"    --schema {output_path} \\")
    print(f"    --format parquet")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
