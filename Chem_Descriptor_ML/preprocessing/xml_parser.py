#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ChemDescriptorML (CDML) - PubChem XML Parser

PubChem PC-Compounds XML to Parquet/CSV converter with:
- XML parsing error recovery (continues with malformed XML)
- Property-based filtering (H-bond donors, MW, etc.)
- Progress bar with ETA and speed information
- Memory-efficient streaming processing
- Support for both .xml and .xml.gz files
- Configurable column selection
- Advanced error handling and logging
- Statistics reporting with filtering metrics
- ALWAYS generates error logs
"""

import os
import csv
import io
import gzip
import xml.etree.ElementTree as ET
import logging
import time
import argparse
import re
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Iterable, Dict, Optional, Generator, List, Tuple
from pathlib import Path

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("Warning: tqdm not installed. Install with 'pip install tqdm' for progress bars.")

# PubChem namespace
NS = {"n": "http://www.ncbi.nlm.nih.gov"}

# Default columns to extract
DEFAULT_COLUMNS = [
    "CID",
    "Charge", 
    "InChI::Standard",
    "InChIKey::Standard",
    "SMILES::Absolute",      # This is the standard SMILES in PubChem
    "SMILES::Connectivity", 
    "Molecular Formula",
    "Molecular Weight",
    "Mass::Exact",
    "Weight::MonoIsotopic",
    "Log P::XLogP3-AA",
    "Log P::XLogP3",
    "Topological::Polar Surface Area",
    "H-Bond Donor Count",
    "H-Bond Acceptor Count",
    "Rotatable Bond Count",
    "Heavy Atom Count"
]

# Property label aliases for robust matching
# NOTE: PubChem XML uses "Count::Hydrogen Bond Donor" format
PROPERTY_ALIASES = {
    "H-Bond Donor Count": [
        "H-Bond Donor Count", 
        "Hydrogen Bond Donor Count", 
        "Hydrogen Bond Donor",
        "Count::Hydrogen Bond Donor"  # Actual XML format
    ],
    "H-Bond Acceptor Count": [
        "H-Bond Acceptor Count", 
        "Hydrogen Bond Acceptor Count", 
        "Hydrogen Bond Acceptor",
        "Count::Hydrogen Bond Acceptor"  # Actual XML format
    ],
    "Molecular Weight": [
        "Molecular Weight", 
        "Weight"
    ],
    "Rotatable Bond Count": [
        "Rotatable Bond Count",
        "Count::Rotatable Bond"  # Actual XML format
    ],
}

def setup_logging(verbose: bool = False) -> logging.Logger:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

def get_file_size(path: str) -> int:
    """Get file size in bytes, handling compressed files."""
    return os.path.getsize(path)

def estimate_compound_count(xml_path: str, sample_size: int = 1024*1024) -> Optional[int]:
    """Estimate the number of compounds by sampling the file."""
    try:
        file_size = get_file_size(xml_path)
        
        # For small files (< 10MB), read the entire file to get exact count
        if file_size < 10 * 1024 * 1024:  # 10MB threshold
            try:
                with _open_xml_readable(xml_path) as fh:
                    content = fh.read()
                    exact_count = content.count('<PC-Compound>')
                    if exact_count > 0:
                        return exact_count
            except Exception as e:
                logging.debug(f"Could not read entire small file: {e}")
        
        # For larger files, use sampling
        with _open_xml_readable(xml_path) as fh:
            sample = fh.read(sample_size)
            compound_count = sample.count('<PC-Compound>')
            
            if compound_count == 0:
                fh.seek(0)
                larger_sample = fh.read(sample_size * 2)
                compound_count = larger_sample.count('<PC-Compound>')
                
                if compound_count == 0:
                    return None
            
            if xml_path.lower().endswith('.gz'):
                estimated_total = max(compound_count * (file_size / sample_size) * 2, compound_count)
            else:
                estimated_total = max(compound_count * (file_size / sample_size), compound_count)
            
            return max(int(estimated_total), compound_count)
            
    except Exception as e:
        logging.debug(f"Could not estimate compound count: {e}")
        return None

def _get_text(elem: ET.Element, path: str) -> Optional[str]:
    """Safely extract text from XML element."""
    try:
        x = elem.find(path, NS)
        return x.text.strip() if x is not None and x.text is not None else None
    except Exception:
        return None

def _info_value(info: ET.Element) -> Optional[str]:
    """Extract value from PC-InfoData element."""
    try:
        for tag in (
            "PC-InfoData_value_sval",
            "PC-InfoData_value_fval", 
            "PC-InfoData_value_ival",
            "PC-InfoData_value_binary",
        ):
            v = info.find(f"n:PC-InfoData_value/n:{tag}", NS)
            if v is not None and v.text:
                return v.text.strip()
        return None
    except Exception:
        return None

def _norm(s: Optional[str]) -> str:
    """Normalize string for property matching."""
    if not s:
        return ""
    s = s.lower().strip().replace("-", " ").replace("_", " ")
    return " ".join(s.split())

def _matches_property(label: str, name: Optional[str], target_label: str, 
                      aliases: Optional[List[str]] = None) -> bool:
    """
    Check if a property matches the target label or its aliases.
    
    Handles both "Label" and "Label::Name" formats.
    Example: "Count::Hydrogen Bond Donor" matches "H-Bond Donor Count"
    """
    # Build full key (e.g., "Count::Hydrogen Bond Donor")
    if name:
        full_key = f"{label}::{name}"
    else:
        full_key = label
    
    # 1. Try full key match with target
    if _norm(full_key) == _norm(target_label):
        return True
    
    # 2. Try full key match with aliases
    if aliases:
        for alias in aliases:
            if _norm(full_key) == _norm(alias):
                return True
    
    # 3. Try label-only match (backward compatibility)
    norm_label = _norm(label)
    norm_target = _norm(target_label)
    
    if norm_label == norm_target:
        return True
    
    # 4. Try label-only match with aliases
    if aliases:
        for alias in aliases:
            if norm_label == _norm(alias):
                return True
    
    return False

def _flatten_infos(pc_compound: ET.Element) -> Dict[str, str]:
    """Extract all property information from PC-Compound element."""
    out = {}
    try:
        for info in pc_compound.findall("n:PC-Compound_props/n:PC-InfoData", NS):
            urn = info.find("n:PC-InfoData_urn/n:PC-Urn", NS)
            if urn is None:
                continue
                
            label = _get_text(urn, "n:PC-Urn_label")
            name = _get_text(urn, "n:PC-Urn_name")
            val = _info_value(info)
            
            if not label:
                continue
                
            key = f"{label}::{name}" if name else label
            out[key] = val
    except Exception as e:
        logging.warning(f"Error extracting compound properties: {e}")
    
    return out

def _open_xml_readable(path: str):
    """Open XML file for reading, handling both regular and gzipped files."""
    if path.lower().endswith(".gz"):
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8")
    return open(path, "r", encoding="utf-8")

def _process_single_file(task: Dict) -> Dict:
    """
    Process a single XML file (worker function for parallel processing).
    
    Args:
        task: Dictionary containing:
            - xml_file: Input XML file path
            - csv_path: Output CSV file path
            - columns: Columns to extract
            - encoding: CSV encoding
            - chunk_size: Buffer size
            - property_filter: PropertyFilter instance or None
    
    Returns:
        Dictionary with status and stats
    """
    try:
        # Create converter with minimal logging (avoid conflicts in multiprocessing)
        converter = EnhancedPubChemConverter()
        
        # Set property filter if provided
        if task['property_filter']:
            converter.set_property_filter(task['property_filter'])
        
        # Convert file
        stats = converter.convert_to_csv(
            xml_path=task['xml_file'],
            csv_path=task['csv_path'],
            columns=task['columns'],
            encoding=task['encoding'],
            chunk_size=task['chunk_size']
        )
        
        return {
            'status': 'SUCCESS',
            'stats': stats
        }
    
    except Exception as e:
        return {
            'status': 'FAILED',
            'error': str(e)
        }


class PropertyFilter:
    """Filter compounds based on property values."""
    
    def __init__(self, property_name: str, min_value: Optional[float] = None, 
                 max_value: Optional[float] = None, aliases: Optional[List[str]] = None):
        """
        Initialize property filter.
        
        Args:
            property_name: Name of the property to filter (e.g., "H-Bond Donor Count")
            min_value: Minimum value (inclusive), None means no lower bound
            max_value: Maximum value (inclusive), None means no upper bound
            aliases: Alternative names for the property
        """
        self.property_name = property_name
        self.min_value = min_value
        self.max_value = max_value
        self.aliases = aliases or []
        
        # Statistics
        self.stats = {
            'checked': 0,
            'passed': 0,
            'filtered': 0,
            'missing': 0
        }
    
    def passes_filter(self, properties: Dict[str, str]) -> bool:
        """
        Check if compound passes the filter.
        
        Args:
            properties: Dictionary of compound properties
            
        Returns:
            True if compound passes filter, False otherwise
        """
        self.stats['checked'] += 1
        
        # Find the property value
        value = None
        for key, val in properties.items():
            # Extract label and name from key
            parts = key.split("::")
            label = parts[0]
            name = parts[1] if len(parts) > 1 else None
            
            if _matches_property(label, name, self.property_name, self.aliases):
                value = val
                break
        
        # If property not found
        if value is None:
            self.stats['missing'] += 1
            return False
        
        # Try to parse as numeric value
        try:
            numeric_value = float(value)
        except (ValueError, TypeError):
            self.stats['missing'] += 1
            return False
        
        # Apply filters
        if self.min_value is not None and numeric_value < self.min_value:
            self.stats['filtered'] += 1
            return False
        
        if self.max_value is not None and numeric_value > self.max_value:
            self.stats['filtered'] += 1
            return False
        
        self.stats['passed'] += 1
        return True
    
    def reset_stats(self):
        """Reset filter statistics."""
        self.stats = {
            'checked': 0,
            'passed': 0,
            'filtered': 0,
            'missing': 0
        }
    
    def get_stats(self) -> Dict[str, int]:
        """Get filter statistics."""
        return self.stats.copy()

class EnhancedPubChemConverter:
    """Enhanced converter with error recovery and filtering capabilities."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.stats = {
            'processed': 0,
            'errors': 0,
            'skipped': 0,
            'start_time': 0,
            'end_time': 0,
        }
        self.error_records = []
        self.property_filter = None
    
    def set_property_filter(self, property_filter: PropertyFilter):
        """Set a property filter for compound filtering."""
        self.property_filter = property_filter
    
    def _clean_element(self, elem: ET.Element):
        """Clean element to free memory. ElementTree compatible."""
        elem.clear()
        # Note: getprevious() and getparent() are lxml-specific
        # ElementTree doesn't support advanced memory cleanup
    
    def iter_pubchem_rows_with_recovery(
        self, 
        xml_path: str, 
        columns: List[str]
    ) -> Generator[Dict[str, str], None, None]:
        """
        Iterate through PubChem XML with error recovery and filtering.
        
        Yields dictionaries with requested columns. Continues processing even
        when individual compounds fail to parse.
        """
        ns = "{http://www.ncbi.nlm.nih.gov}"
        compound_count = 0
        
        with _open_xml_readable(xml_path) as fh:
            try:
                for event, elem in ET.iterparse(fh, events=("end",)):
                    if elem.tag != f"{ns}PC-Compound":
                        continue
                    
                    compound_count += 1
                    
                    try:
                        # Extract CID
                        cid = _get_text(elem, ".//n:PC-CompoundType_id_cid")
                        if not cid:
                            self.logger.debug(f"Compound #{compound_count}: No CID found")
                            self.stats['skipped'] += 1
                            self.error_records.append({
                                'compound_num': compound_count,
                                'cid': 'UNKNOWN',
                                'error': 'No CID found',
                                'recoverable': False
                            })
                            self._clean_element(elem)
                            continue
                        
                        # Extract all properties
                        props = _flatten_infos(elem)
                        
                        # Apply property filter if set
                        if self.property_filter:
                            if not self.property_filter.passes_filter(props):
                                # Compound filtered out - not an error, just doesn't match criteria
                                self._clean_element(elem)
                                continue
                        
                        # Build row
                        row = {}
                        has_data = False
                        
                        for col in columns:
                            if col == "CID":
                                row[col] = cid
                                has_data = True
                            else:
                                # Try exact match first
                                val = props.get(col)
                                
                                # If no exact match, try with aliases
                                if val is None:
                                    aliases = PROPERTY_ALIASES.get(col, [])
                                    for key in props:
                                        # Check if key matches column name or any alias
                                        if _norm(key) == _norm(col):
                                            val = props[key]
                                            break
                                        for alias in aliases:
                                            if _norm(key) == _norm(alias):
                                                val = props[key]
                                                break
                                        if val:
                                            break
                                
                                # If still no match, try without name specifier (backward compatibility)
                                if val is None and "::" in col:
                                    base_col = col.split("::")[0]
                                    for key in props:
                                        if key.startswith(base_col):
                                            val = props[key]
                                            break
                                
                                row[col] = val if val else ""
                                if val:
                                    has_data = True
                        
                        if has_data:
                            self.stats['processed'] += 1
                            yield row
                        else:
                            self.stats['skipped'] += 1
                            self.error_records.append({
                                'compound_num': compound_count,
                                'cid': cid,
                                'error': 'No data extracted for any column',
                                'recoverable': False
                            })
                    
                    except ET.ParseError as e:
                        # XML parsing error in this specific compound
                        self.stats['errors'] += 1
                        self.logger.debug(f"Compound #{compound_count}: ParseError - {e}")
                        self.error_records.append({
                            'compound_num': compound_count,
                            'cid': 'UNKNOWN',
                            'error': f'ParseError: {str(e)}',
                            'recoverable': True
                        })
                    
                    except Exception as e:
                        # Other errors - log and continue
                        self.stats['errors'] += 1
                        try:
                            cid = _get_text(elem, ".//n:PC-CompoundType_id_cid") or "UNKNOWN"
                        except:
                            cid = "UNKNOWN"
                        
                        self.logger.debug(f"Compound #{compound_count} (CID: {cid}): {type(e).__name__} - {e}")
                        self.error_records.append({
                            'compound_num': compound_count,
                            'cid': cid,
                            'error': f'{type(e).__name__}: {str(e)}',
                            'recoverable': True
                        })
                    
                    finally:
                        # Always clean up to prevent memory leaks
                        self._clean_element(elem)
            
            except ET.ParseError as e:
                # Fatal parsing error
                self.logger.error(f"Fatal XML parsing error at compound #{compound_count}: {e}")
                self.logger.error("Attempting to continue with recovered data...")
                self.stats['errors'] += 1
                self.error_records.append({
                    'compound_num': compound_count,
                    'cid': 'FATAL',
                    'error': f'Fatal ParseError: {str(e)}',
                    'recoverable': False
                })
    
    def save_error_log(self, xml_path: str, csv_path: str) -> Optional[str]:
        """Save error records to a log file."""
        if not self.error_records:
            return None
        
        error_log_path = str(Path(csv_path).with_suffix('.errors.csv'))
        
        try:
            with open(error_log_path, 'w', newline='', encoding='utf-8') as f:
                fieldnames = ['compound_num', 'cid', 'error', 'recoverable']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.error_records)
            
            return error_log_path
        except Exception as e:
            self.logger.error(f"Failed to write error log: {e}")
            return None
    
    def convert_to_csv(
        self,
        xml_path: str,
        csv_path: str,
        columns: Optional[List[str]] = None,
        encoding: str = "utf-8",
        chunk_size: int = 10000
    ) -> Dict[str, int]:
        """
        Convert PubChem XML to CSV with error recovery and optional filtering.
        
        Args:
            xml_path: Path to input XML file
            csv_path: Path to output CSV file
            columns: List of columns to extract (default: DEFAULT_COLUMNS)
            encoding: CSV file encoding
            chunk_size: Number of rows to buffer before writing
            
        Returns:
            Dictionary with processing statistics
        """
        if columns is None:
            columns = DEFAULT_COLUMNS
        
        # Ensure CID is always included
        if "CID" not in columns:
            columns = ["CID"] + columns
        
        # Reset statistics
        self.stats = {
            'processed': 0,
            'errors': 0,
            'skipped': 0,
            'start_time': time.time(),
            'end_time': 0,
        }
        self.error_records = []
        
        # Reset filter statistics if filter is set
        if self.property_filter:
            self.property_filter.reset_stats()
        
        self.logger.info(f"Starting ENHANCED conversion: {xml_path}")
        self.logger.info(f"Output: {csv_path}")
        self.logger.info(f"Columns: {', '.join(columns)}")
        
        if self.property_filter:
            filter_desc = f"{self.property_filter.property_name}"
            if self.property_filter.min_value is not None:
                filter_desc += f" >= {self.property_filter.min_value}"
            if self.property_filter.max_value is not None:
                filter_desc += f" <= {self.property_filter.max_value}"
            self.logger.info(f"Filter: {filter_desc}")
        
        # Estimate total compounds for progress bar
        estimated_total = estimate_compound_count(xml_path)
        if estimated_total:
            self.logger.info(f"Estimated compounds: ~{estimated_total:,}")
        
        # Create output directory if needed
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Setup progress bar
        pbar = None
        if HAS_TQDM:
            pbar = tqdm(
                total=estimated_total,
                desc="Processing",
                unit="compounds",
                dynamic_ncols=True
            )
        
        try:
            with open(csv_path, "w", newline="", encoding=encoding) as f:
                writer = csv.DictWriter(f, fieldnames=columns)
                writer.writeheader()
                
                buffer = []
                update_counter = 0
                
                for row in self.iter_pubchem_rows_with_recovery(xml_path, columns):
                    buffer.append(row)
                    
                    # Write in chunks for better performance
                    if len(buffer) >= chunk_size:
                        writer.writerows(buffer)
                        buffer = []
                    
                    # Update progress bar less frequently for better performance
                    update_counter += 1
                    if pbar and update_counter % 100 == 0:
                        pbar.update(100)
                        
                        # Update statistics in progress bar every 1000 compounds
                        if update_counter % 1000 == 0:
                            elapsed = time.time() - self.stats['start_time']
                            rate = self.stats['processed'] / elapsed if elapsed > 0 else 0
                            postfix = {
                                'written': self.stats['processed'],
                                'errors': self.stats['errors'],
                                'rate': f"{rate:.0f}/s"
                            }
                            
                            if self.property_filter:
                                filter_stats = self.property_filter.get_stats()
                                postfix['filtered'] = filter_stats['filtered']
                            
                            pbar.set_postfix(postfix)
                
                # Write remaining buffer
                if buffer:
                    writer.writerows(buffer)
                
                # Final progress bar update
                if pbar and update_counter % 100 != 0:
                    pbar.update(update_counter % 100)
        
        finally:
            if pbar:
                pbar.close()
        
        self.stats['end_time'] = time.time()
        
        # Log final statistics
        elapsed = self.stats['end_time'] - self.stats['start_time']
        rate = self.stats['processed'] / elapsed if elapsed > 0 else 0
        total_attempted = self.stats['processed'] + self.stats['skipped']
        
        self.logger.info(f"ENHANCED conversion completed!")
        self.logger.info(f"Successfully processed: {self.stats['processed']:,} compounds")
        
        if self.property_filter:
            filter_stats = self.property_filter.get_stats()
            self.logger.info(f"Filter statistics:")
            self.logger.info(f"  - Checked: {filter_stats['checked']:,}")
            self.logger.info(f"  - Passed filter: {filter_stats['passed']:,}")
            self.logger.info(f"  - Filtered out: {filter_stats['filtered']:,}")
            self.logger.info(f"  - Missing property: {filter_stats['missing']:,}")
        
        if self.stats['errors'] > 0:
            self.logger.info(f"Recovered from errors: {self.stats['errors']:,} compounds")
        if self.stats['skipped'] > 0:
            self.logger.info(f"Skipped (unrecoverable): {self.stats['skipped']:,} compounds")
        
        if total_attempted > 0:
            success_rate = (self.stats['processed'] / total_attempted) * 100
            self.logger.info(f"Success rate: {success_rate:.1f}%")
        
        self.logger.info(f"Time: {elapsed:.1f} seconds")
        self.logger.info(f"Rate: {rate:.1f} compounds/second")
        self.logger.info(f"Output: {csv_path}")
        
        # ALWAYS generate error log if there were any errors or skips
        if self.error_records:
            self.logger.info("Generating error log file...")
            error_log_path = self.save_error_log(xml_path, csv_path)
            if error_log_path:
                self.logger.info(f"Error log saved: {error_log_path}")
        
        # Add filter statistics to return stats
        result_stats = self.stats.copy()
        if self.property_filter:
            result_stats['filter_stats'] = self.property_filter.get_stats()
        
        return result_stats

class PubChemBatchConverter:
    """Batch converter for multiple PubChem XML files with filtering support."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.batch_stats = {
            'total_files': 0,
            'completed_files': 0,
            'failed_files': 0,
            'total_compounds': 0,
            'total_errors': 0,
            'total_skipped': 0,
            'total_filtered': 0,
            'start_time': None,
            'end_time': None,
            'failed_file_list': []
        }
        self.property_filter = None
    
    def set_property_filter(self, property_filter: PropertyFilter):
        """Set a property filter for batch processing."""
        self.property_filter = property_filter
    
    def find_xml_files(self, input_path: str, recursive: bool = False) -> List[str]:
        """Find all XML files in the given path."""
        input_path = Path(input_path)
        
        if input_path.is_file():
            if input_path.suffix.lower() in ['.xml', '.gz']:
                return [str(input_path)]
            else:
                self.logger.error(f"File is not XML: {input_path}")
                return []
        
        if not input_path.is_dir():
            self.logger.error(f"Path does not exist: {input_path}")
            return []
        
        # Find XML files
        patterns = ['*.xml', '*.xml.gz']
        xml_files = []
        
        for pattern in patterns:
            if recursive:
                xml_files.extend(input_path.rglob(pattern))
            else:
                xml_files.extend(input_path.glob(pattern))
        
        return [str(f) for f in sorted(xml_files)]
    
    def convert_batch_sequential(
        self,
        xml_files: List[str],
        output_dir: str,
        columns: Optional[List[str]] = None,
        encoding: str = "utf-8",
        chunk_size: int = 10000
    ) -> Dict:
        """
        Convert multiple XML files sequentially with enhanced error recovery.
        """
        if columns is None:
            columns = DEFAULT_COLUMNS
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        self.batch_stats['total_files'] = len(xml_files)
        self.batch_stats['start_time'] = time.time()
        
        file_results = []
        
        # Process files with progress bar
        file_iterator = xml_files
        if HAS_TQDM:
            file_iterator = tqdm(xml_files, desc="Batch processing", unit="file")
        
        for xml_file in file_iterator:
            xml_path = Path(xml_file)
            csv_name = xml_path.name.replace('.xml.gz', '.csv').replace('.xml', '.csv')
            csv_path = output_path / csv_name
            
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Processing: {xml_path.name}")
            self.logger.info(f"Output: {csv_path}")
            
            try:
                converter = EnhancedPubChemConverter(self.logger)
                
                # Set filter if available
                if self.property_filter:
                    converter.set_property_filter(self.property_filter)
                
                stats = converter.convert_to_csv(
                    xml_path=str(xml_file),
                    csv_path=str(csv_path),
                    columns=columns,
                    encoding=encoding,
                    chunk_size=chunk_size
                )
                
                self.batch_stats['completed_files'] += 1
                self.batch_stats['total_compounds'] += stats['processed']
                self.batch_stats['total_errors'] += stats['errors']
                self.batch_stats['total_skipped'] += stats['skipped']
                
                if 'filter_stats' in stats:
                    self.batch_stats['total_filtered'] += stats['filter_stats']['filtered']
                
                file_results.append({
                    'file': xml_path.name,
                    'status': 'SUCCESS',
                    'stats': stats
                })
                
            except Exception as e:
                self.logger.error(f"Failed to process {xml_path.name}: {e}")
                self.batch_stats['failed_files'] += 1
                self.batch_stats['failed_file_list'].append(xml_path.name)
                
                file_results.append({
                    'file': xml_path.name,
                    'status': 'FAILED',
                    'error': str(e)
                })
        
        self.batch_stats['end_time'] = time.time()
        
        return {
            'batch_stats': self.batch_stats,
            'file_results': file_results
        }
    
    def convert_batch_parallel(
        self,
        xml_files: List[str],
        output_dir: str,
        columns: Optional[List[str]] = None,
        encoding: str = "utf-8",
        chunk_size: int = 10000,
        max_workers: Optional[int] = None
    ) -> Dict:
        """
        Convert multiple XML files in parallel with enhanced error recovery.
        
        Args:
            xml_files: List of XML file paths
            output_dir: Output directory
            columns: Columns to extract
            encoding: CSV encoding
            chunk_size: Buffer size
            max_workers: Maximum number of parallel workers (default: CPU count)
        """
        if columns is None:
            columns = DEFAULT_COLUMNS
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        self.batch_stats['total_files'] = len(xml_files)
        self.batch_stats['start_time'] = time.time()
        
        file_results = []
        
        # Prepare tasks
        tasks = []
        for xml_file in xml_files:
            xml_path = Path(xml_file)
            csv_name = xml_path.name.replace('.xml.gz', '.csv').replace('.xml', '.csv')
            csv_path = output_path / csv_name
            
            tasks.append({
                'xml_file': str(xml_file),
                'csv_path': str(csv_path),
                'columns': columns,
                'encoding': encoding,
                'chunk_size': chunk_size,
                'property_filter': self.property_filter
            })
        
        # Process files in parallel
        if max_workers is None:
            import os
            max_workers = os.cpu_count() or 1
        
        self.logger.info(f"Processing {len(tasks)} files with {max_workers} workers...")
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(_process_single_file, task): task 
                for task in tasks
            }
            
            # Process results with progress bar
            if HAS_TQDM:
                futures = tqdm(as_completed(future_to_task), total=len(tasks), 
                              desc="Parallel processing", unit="file")
            else:
                futures = as_completed(future_to_task)
            
            for future in futures:
                task = future_to_task[future]
                xml_name = Path(task['xml_file']).name
                
                try:
                    result = future.result()
                    
                    if result['status'] == 'SUCCESS':
                        stats = result['stats']
                        self.batch_stats['completed_files'] += 1
                        self.batch_stats['total_compounds'] += stats['processed']
                        self.batch_stats['total_errors'] += stats['errors']
                        self.batch_stats['total_skipped'] += stats['skipped']
                        
                        if 'filter_stats' in stats:
                            self.batch_stats['total_filtered'] += stats['filter_stats']['filtered']
                        
                        file_results.append({
                            'file': xml_name,
                            'status': 'SUCCESS',
                            'stats': stats
                        })
                    else:
                        self.batch_stats['failed_files'] += 1
                        self.batch_stats['failed_file_list'].append(xml_name)
                        
                        file_results.append({
                            'file': xml_name,
                            'status': 'FAILED',
                            'error': result.get('error', 'Unknown error')
                        })
                
                except Exception as e:
                    self.logger.error(f"Failed to process {xml_name}: {e}")
                    self.batch_stats['failed_files'] += 1
                    self.batch_stats['failed_file_list'].append(xml_name)
                    
                    file_results.append({
                        'file': xml_name,
                        'status': 'FAILED',
                        'error': str(e)
                    })
        
        self.batch_stats['end_time'] = time.time()
        
        return {
            'batch_stats': self.batch_stats,
            'file_results': file_results
        }
    
    def print_batch_summary(self, batch_result: Dict):
        """Print summary of batch processing."""
        stats = batch_result['batch_stats']
        elapsed = stats['end_time'] - stats['start_time']
        
        print("\n" + "="*60)
        print("BATCH PROCESSING SUMMARY")
        print("="*60)
        print(f"Total files: {stats['total_files']}")
        print(f"Completed: {stats['completed_files']}")
        print(f"Failed: {stats['failed_files']}")
        print(f"\nTotal compounds processed: {stats['total_compounds']:,}")
        
        if stats['total_filtered'] > 0:
            print(f"Total compounds filtered out: {stats['total_filtered']:,}")
        
        if stats['total_errors'] > 0:
            print(f"Total errors recovered: {stats['total_errors']:,}")
        if stats['total_skipped'] > 0:
            print(f"Total skipped: {stats['total_skipped']:,}")
        
        print(f"\nTotal time: {elapsed:.1f} seconds")
        if stats['total_compounds'] > 0:
            rate = stats['total_compounds'] / elapsed
            print(f"Average rate: {rate:.1f} compounds/second")
        
        if stats['failed_files'] > 0:
            print(f"\nFailed files:")
            for f in stats['failed_file_list']:
                print(f"  - {f}")

def main():
    """Main command line interface with enhanced error recovery and filtering."""
    parser = argparse.ArgumentParser(
        description="Enhanced PubChem PC-Compounds XML to CSV converter with filtering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
ENHANCED CONVERSION FEATURES:
- XML parsing error recovery: Continues processing even with malformed XML
- Property-based filtering: Filter compounds by numeric properties
- Always generates error logs for debugging
- Detailed recovery and filtering statistics

FILTERING EXAMPLES:
  # Extract only compounds with H-bond donors >= 1
  %(prog)s input.xml.gz --min-donors 1
  
  # Extract compounds with H-bond donors between 2 and 5
  %(prog)s input.xml.gz --min-donors 2 --max-donors 5
  
  # Filter by molecular weight
  %(prog)s input.xml.gz --filter-property "Molecular Weight" --min-value 200 --max-value 500
  
  # Batch processing with filtering
  %(prog)s /path/to/xml/folder/ --output-dir ./results/ --min-donors 1 --recursive

BATCH PROCESSING:
  # Process directory sequentially (default)
  %(prog)s /data/pubchem/ --output-dir ./csv/ --recursive --min-donors 1
  
  # Process directory in PARALLEL (faster for multiple files)
  %(prog)s /data/pubchem/ --output-dir ./csv/ --recursive --min-donors 1 --parallel
  
  # Parallel with custom worker count
  %(prog)s /data/pubchem/ --output-dir ./csv/ --recursive --parallel --max-workers 8
        """
    )
    
    parser.add_argument("input", help="Input XML file or directory containing XML files")
    
    # Output options
    parser.add_argument("-o", "--output", help="Output CSV file path (single file mode only)")
    parser.add_argument("--output-dir", help="Output directory for batch processing")
    
    # Filtering options
    filter_group = parser.add_argument_group('Filtering Options')
    filter_group.add_argument("--min-donors", type=float, metavar="N",
                             help="Minimum H-Bond Donor Count (shortcut for common filter)")
    filter_group.add_argument("--max-donors", type=float, metavar="N",
                             help="Maximum H-Bond Donor Count")
    filter_group.add_argument("--min-acceptors", type=float, metavar="N",
                             help="Minimum H-Bond Acceptor Count")
    filter_group.add_argument("--max-acceptors", type=float, metavar="N",
                             help="Maximum H-Bond Acceptor Count")
    filter_group.add_argument("--filter-property", metavar="NAME",
                             help="Custom property name to filter by")
    filter_group.add_argument("--min-value", type=float, metavar="N",
                             help="Minimum value for custom property filter")
    filter_group.add_argument("--max-value", type=float, metavar="N",
                             help="Maximum value for custom property filter")
    
    # Processing options
    parser.add_argument("--encoding", default="utf-8", help="CSV encoding (default: utf-8)")
    parser.add_argument("--columns", nargs="*", help="Specific columns to extract (default: all)")
    parser.add_argument("--chunk-size", type=int, default=10000, 
                       help="Number of rows to buffer before writing (default: 10000)")
    
    # Batch processing options
    parser.add_argument("--recursive", "-r", action="store_true", 
                       help="Search for XML files recursively in subdirectories")
    parser.add_argument("--parallel", "-p", action="store_true",
                       help="Process multiple files in parallel (batch mode only)")
    parser.add_argument("--max-workers", type=int, default=None,
                       help="Maximum number of parallel workers (default: CPU count)")
    
    # General options
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.verbose)
    
    # Validate input
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input path not found: {args.input}")
        return 1
    
    # Determine processing mode
    is_single_file = input_path.is_file()
    is_directory = input_path.is_dir()
    
    if not (is_single_file or is_directory):
        logger.error(f"Input must be a file or directory: {args.input}")
        return 1
    
    # Use specified columns or default
    columns = args.columns if args.columns else DEFAULT_COLUMNS
    
    # Setup property filter
    property_filter = None
    
    # H-Bond Donor filter (common case)
    if args.min_donors is not None or args.max_donors is not None:
        if args.min_donors is not None and args.min_donors < 0:
            logger.error("--min-donors must be >= 0")
            return 1
        if args.max_donors is not None and args.max_donors < 0:
            logger.error("--max-donors must be >= 0")
            return 1
        
        property_filter = PropertyFilter(
            property_name="H-Bond Donor Count",
            min_value=args.min_donors,
            max_value=args.max_donors,
            aliases=PROPERTY_ALIASES.get("H-Bond Donor Count", [])
        )
        logger.info(f"Filter: H-Bond Donor Count >= {args.min_donors if args.min_donors is not None else 'any'}" +
                   (f" and <= {args.max_donors}" if args.max_donors is not None else ""))
    
    # H-Bond Acceptor filter
    elif args.min_acceptors is not None or args.max_acceptors is not None:
        if args.min_acceptors is not None and args.min_acceptors < 0:
            logger.error("--min-acceptors must be >= 0")
            return 1
        if args.max_acceptors is not None and args.max_acceptors < 0:
            logger.error("--max-acceptors must be >= 0")
            return 1
        
        property_filter = PropertyFilter(
            property_name="H-Bond Acceptor Count",
            min_value=args.min_acceptors,
            max_value=args.max_acceptors,
            aliases=PROPERTY_ALIASES.get("H-Bond Acceptor Count", [])
        )
    
    # Custom property filter
    elif args.filter_property:
        if args.min_value is None and args.max_value is None:
            logger.error("--filter-property requires at least --min-value or --max-value")
            return 1
        
        property_filter = PropertyFilter(
            property_name=args.filter_property,
            min_value=args.min_value,
            max_value=args.max_value,
            aliases=PROPERTY_ALIASES.get(args.filter_property, [])
        )
    
    # Ensure filter column is included in output
    if property_filter and property_filter.property_name not in columns:
        # Try to add the property to columns
        for default_col in DEFAULT_COLUMNS:
            if property_filter.property_name in default_col or default_col in property_filter.property_name:
                if default_col not in columns:
                    columns.append(default_col)
                break
    
    try:
        if is_single_file:
            # Single file mode
            if args.output_dir:
                logger.warning("--output-dir ignored in single file mode, use -o instead")
            
            # Determine output path
            if args.output:
                output_path = args.output
            else:
                base_name = input_path.name.replace('.xml.gz', '').replace('.xml', '')
                output_path = input_path.parent / f"{base_name}.csv"
            
            # Convert single file
            converter = EnhancedPubChemConverter(logger)
            
            # Set filter if available
            if property_filter:
                converter.set_property_filter(property_filter)
            
            stats = converter.convert_to_csv(
                xml_path=str(input_path),
                csv_path=str(output_path), 
                columns=columns,
                encoding=args.encoding,
                chunk_size=args.chunk_size
            )
            
            print(f"\nENHANCED conversion completed!")
            print(f"Output: {output_path}")
            print(f"Successfully processed: {stats['processed']:,} compounds")
            
            if 'filter_stats' in stats:
                filter_stats = stats['filter_stats']
                print(f"\nFilter statistics:")
                print(f"  Checked: {filter_stats['checked']:,}")
                print(f"  Passed filter: {filter_stats['passed']:,}")
                print(f"  Filtered out: {filter_stats['filtered']:,}")
                print(f"  Missing property: {filter_stats['missing']:,}")
            
            if stats['errors'] > 0:
                print(f"Recovered from errors: {stats['errors']:,} compounds")
            if stats['skipped'] > 0:
                print(f"Skipped (unrecoverable): {stats['skipped']:,} compounds")
            
            total_attempted = stats['processed'] + stats['skipped']
            if total_attempted > 0:
                success_rate = (stats['processed'] / total_attempted) * 100
                print(f"Success rate: {success_rate:.1f}%")
        
        else:
            # Batch processing mode
            if args.output and not args.output_dir:
                logger.warning("-o ignored in batch mode, use --output-dir instead")
            
            # Determine output directory
            if args.output_dir:
                output_dir = args.output_dir
            else:
                output_dir = input_path / "csv_output"
            
            # Create batch converter
            batch_converter = PubChemBatchConverter(logger)
            
            # Set filter if available
            if property_filter:
                batch_converter.set_property_filter(property_filter)
            
            # Find XML files
            xml_files = batch_converter.find_xml_files(str(input_path), args.recursive)
            
            if not xml_files:
                logger.error("No XML files found in the specified directory")
                return 1
            
            logger.info(f"Found {len(xml_files)} XML files to process")
            logger.info(f"Output directory: {output_dir}")
            
            if property_filter:
                filter_desc = f"{property_filter.property_name}"
                if property_filter.min_value is not None:
                    filter_desc += f" >= {property_filter.min_value}"
                if property_filter.max_value is not None:
                    filter_desc += f" <= {property_filter.max_value}"
                logger.info(f"Filter: {filter_desc}")
            
            # Choose processing mode
            if args.parallel:
                # Parallel processing
                max_workers = args.max_workers
                if max_workers is None:
                    import os
                    max_workers = os.cpu_count() or 1
                
                logger.info(f"Using PARALLEL processing with {max_workers} workers")
                batch_result = batch_converter.convert_batch_parallel(
                    xml_files=xml_files,
                    output_dir=output_dir,
                    columns=columns,
                    encoding=args.encoding,
                    chunk_size=args.chunk_size,
                    max_workers=max_workers
                )
            else:
                # Sequential processing (default)
                logger.info("Using sequential processing with enhanced error recovery")
                batch_result = batch_converter.convert_batch_sequential(
                    xml_files=xml_files,
                    output_dir=output_dir,
                    columns=columns,
                    encoding=args.encoding,
                    chunk_size=args.chunk_size
                )
            
            # Print summary
            batch_converter.print_batch_summary(batch_result)
            
            # Return appropriate exit code
            if batch_result['batch_stats']['failed_files'] > 0:
                return 1 if batch_result['batch_stats']['completed_files'] == 0 else 2
            
        return 0
        
    except KeyboardInterrupt:
        logger.info("Conversion interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return 1

if __name__ == "__main__":
    exit(main())
