#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Process-based per-molecule timeout version + RESUME FROM PARTS:
- Keeps original pipeline (parse → std → descriptors → schema enforce → streaming write)
- Per-molecule timeout using a supervised worker process
- RESUME: If parquet part files already exist (part-*.parquet), skip those chunks and
          continue with the next missing chunk index. Consolidation keeps parts by default.

Base: user-provided make_descriptors_fixed_complete.py (keeps schema/streaming)  # cite
"""
from __future__ import annotations

import argparse, json, logging, sys, traceback, warnings, os, re, time, glob, uuid, gc, tempfile
from pathlib import Path
from typing import Iterable, List, Optional, Dict, Any, Tuple
from collections import defaultdict

import pandas as pd
import numpy as np
if not hasattr(np, "product"):
    np.product = np.prod
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor

# NEW (process-timeout)
import multiprocessing as mp
import queue

# Arrow/Parquet
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    from pyarrow import dataset as ds
    HAS_PYARROW = True
except Exception:
    HAS_PYARROW = False
    print("WARNING: PyArrow not available. Parquet processing will be less efficient.")

# RDKit / Mordred
from rdkit import Chem
from rdkit.Chem import Descriptors as RDKitDescriptors
from rdkit.Chem import Fragments as RDKitFragments
try:
    from rdkit.Chem.MolStandardize import rdMolStandardize
except ImportError:
    from rdkit.Chem import rdMolStandardize

from rdkit import RDLogger
RDLogger.DisableLog("rdApp.warning")

from mordred import Calculator, descriptors as mordred_descriptors

# ---------- Shared util ----------
def normalize_column_names(columns: List[str]) -> List[str]:
    seen = {}
    out = []
    for c in columns:
        if c in seen:
            seen[c] += 1
            out.append(f"{c}_{seen[c]}")
        else:
            seen[c] = 0
            out.append(c)
    return out

def flatten_multiindex_columns(columns) -> List[str]:
    if isinstance(columns, pd.MultiIndex):
        return ["_".join(map(str, c)).strip() for c in columns.values]
    return [str(c) for c in columns]

_GLOBAL_OBJECTS = {
    'normalizer': None,
    'reionizer': None,
    'metal_disconnector': None,
    'fragment_chooser': None,
    'rdkit_funcs': None,
    'rdkit_func_names': None,
    'mordred_calc': None,
}

_SCHEMA_MANAGER = {
    'master_schema': None,
    'schema_loaded': False,
    'metadata_columns': ['CID','isomeric_smiles','standardized_smiles','parse_source','standardization_status','source_file'],
    'schema_warning_shown': False
}

_SUMMARY_STATS = {'file_summaries': []}

def _descriptor_count_from_schema():
    master = _SCHEMA_MANAGER.get('master_schema') or []
    meta = set(_SCHEMA_MANAGER.get('metadata_columns') or [])
    return len([c for c in master if c not in meta])

def init_global_objects_optimized(std_core=True, use_normalizer=False, use_reionizer=False,
                                  metal_disconnector=False, largest_fragment=False,
                                  include_fragments=True, use_3d=False, **kwargs):
    # Standardize helpers
    if use_normalizer and _GLOBAL_OBJECTS['normalizer'] is None:
        _GLOBAL_OBJECTS['normalizer'] = rdMolStandardize.Normalizer()
    if use_reionizer and _GLOBAL_OBJECTS['reionizer'] is None:
        _GLOBAL_OBJECTS['reionizer'] = rdMolStandardize.Reionizer()
    if metal_disconnector and _GLOBAL_OBJECTS['metal_disconnector'] is None:
        _GLOBAL_OBJECTS['metal_disconnector'] = rdMolStandardize.MetalDisconnector()
    if largest_fragment and _GLOBAL_OBJECTS['fragment_chooser'] is None:
        _GLOBAL_OBJECTS['fragment_chooser'] = rdMolStandardize.LargestFragmentChooser()

    # RDKit descriptor funcs
    if _GLOBAL_OBJECTS['rdkit_funcs'] is None:
        rdkit_desc = list(RDKitDescriptors.descList)
        if include_fragments:
            seen = {n for n,_ in rdkit_desc}
            for k in dir(RDKitDescriptors):
                if k.startswith("fr_"):
                    fn = getattr(RDKitDescriptors, k, None)
                    if callable(fn) and k not in seen:
                        rdkit_desc.append((k, fn)); seen.add(k)
            for k in dir(RDKitFragments):
                if k.startswith("fr_"):
                    fn = getattr(RDKitFragments, k, None)
                    if callable(fn) and k not in seen:
                        rdkit_desc.append((k, fn)); seen.add(k)
        _GLOBAL_OBJECTS['rdkit_funcs'] = [f for _,f in rdkit_desc]
        _GLOBAL_OBJECTS['rdkit_func_names'] = [n for n,_ in rdkit_desc]

    # Mordred calc
    if _GLOBAL_OBJECTS['mordred_calc'] is None:
        _GLOBAL_OBJECTS['mordred_calc'] = Calculator(mordred_descriptors, ignore_3D=not use_3d)

def setup_logging_optimized(log_file: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger(__name__)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout); ch.setFormatter(fmt); logger.addHandler(ch)
    if log_file:
        p = Path(log_file)
        if p.is_dir() or str(log_file).endswith(('/', '\\')):
            logger.info(f"Log dir: {log_file} (per-file logs will be created)")
        else:
            p.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(p, mode="a", encoding="utf-8")
            fh.setFormatter(fmt); logger.addHandler(fh)
            logger.info(f"Logging to: {p}")
    return logger

def setup_logging_per_file(base_logger: logging.Logger, log_base_path: Optional[str], input_filename: str) -> logging.Logger:
    if not log_base_path: return base_logger
    for h in list(base_logger.handlers):
        if isinstance(h, logging.FileHandler):
            h.close(); base_logger.removeHandler(h)
    base = Path(log_base_path)
    try:
        if base.is_dir() or str(log_base_path).endswith(('/', '\\')):
            base.mkdir(parents=True, exist_ok=True)
            log_file = base / f"{Path(input_filename).stem}_descriptors.log"
        else:
            log_file = base
            log_file.parent.mkdir(parents=True, exist_ok=True)
        fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        fh.setFormatter(fmt); base_logger.addHandler(fh)
        base_logger.info("="*60)
        base_logger.info(f"PROCESSING STARTED: {input_filename}")
        base_logger.info(f"Log file: {log_file}")
        base_logger.info("="*60)
    except Exception as e:
        base_logger.warning(f"Failed to set per-file log: {e}")
    return base_logger

# --- TSV helpers ---
def _append_tsv_line(path: Path, header: str, line: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(header + "\n", encoding="utf-8")
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")

# ---------- Schema management ----------
def load_master_schema(schema_path: Path, logger: logging.Logger) -> List[str]:
    if not schema_path or not schema_path.exists():
        logger.error(f"Master schema not found: {schema_path}")
        raise FileNotFoundError("--schema is required")
    cols = json.loads(schema_path.read_text(encoding="utf-8"))
    # de-dup while preserving order
    seen = set(); uniq = []
    for c in cols:
        if c not in seen: uniq.append(c); seen.add(c)
    meta = _SCHEMA_MANAGER['metadata_columns']
    for m in reversed(meta):
        if m not in uniq: uniq.insert(0, m)
    _SCHEMA_MANAGER['master_schema'] = uniq
    _SCHEMA_MANAGER['schema_loaded'] = True
    logger.info(f"Master schema loaded: {len(uniq)} columns (forced on output)")
    return uniq

def enforce_master_schema(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    if not _SCHEMA_MANAGER['schema_loaded']:
        raise RuntimeError("Master schema must be loaded first")
    master = _SCHEMA_MANAGER['master_schema']
    extra = [c for c in df.columns if c not in master]
    if extra and not _SCHEMA_MANAGER['schema_warning_shown']:
        logger.warning(f"Dropping {len(extra)} columns NOT in master schema (e.g. {extra[:8]})")
        _SCHEMA_MANAGER['schema_warning_shown'] = True
    missing = [c for c in master if c not in df.columns]
    for c in missing: df[c] = np.nan
    return df.reindex(columns=master)

# ---------- Parsing / Standardization ----------
def parse_molecules_vectorized(smiles_list: List[str], inchi_list: List[Optional[str]], mol_ids: List[str]):
    n = len(smiles_list)
    mols = [None]*n; sources = ["none"]*n; canon = [None]*n
    p = Chem.SmilesParserParams(); p.sanitize = False
    for i, smi in enumerate(smiles_list):
        if isinstance(smi, str) and smi.strip():
            try:
                m = Chem.MolFromSmiles(smi.strip(), p)
                if m is not None:
                    Chem.SanitizeMol(m); mols[i] = m; sources[i] = "smiles"
                    canon[i] = Chem.MolToSmiles(m, isomericSmiles=True); continue
            except: pass
        inc = inchi_list[i] if inchi_list else None
        if isinstance(inc, str) and inc.strip():
            try:
                inc2 = inc.strip(); inc2 = inc2 if inc2.startswith("InChI=") else "InChI="+inc2
                m = Chem.MolFromInchi(inc2)
                if m is not None:
                    Chem.SanitizeMol(m); mols[i] = m; sources[i] = "inchi_fallback"
                    canon[i] = Chem.MolToSmiles(m, isomericSmiles=True)
            except: pass
    return mols, sources, canon

class SimpleErrorTracker:
    __slots__ = ['parse_errors','std_errors','desc_errors','total','std_error_types','error_cids','recovery_stats']
    def __init__(self):
        self.parse_errors = 0; self.std_errors = 0; self.desc_errors = 0; self.total = 0
        self.std_error_types = defaultdict(int); self.error_cids = defaultdict(list)
        self.recovery_stats = {'attempted':0,'recovered':0}
    def log_parse_error(self): self.parse_errors += 1
    def log_std_error_batch(self, etype, cid=None):
        self.std_errors += 1; self.std_error_types[etype] += 1
        if cid: self.error_cids[etype].append(cid)
    def log_desc_error(self): self.desc_errors += 1
    def add_recovery_stats(self,a,r): self.recovery_stats['attempted']+=a; self.recovery_stats['recovered']+=r
    def merge(self, o):
        self.parse_errors += o.parse_errors; self.std_errors += o.std_errors; self.desc_errors += o.desc_errors
        self.total += o.total
        for k,v in o.std_error_types.items(): self.std_error_types[k]+=v
        for k,ids in o.error_cids.items(): self.error_cids[k].extend(ids)
        self.recovery_stats['attempted']+=o.recovery_stats['attempted']; self.recovery_stats['recovered']+=o.recovery_stats['recovered']
    def report_batch_summary(self, logger):
        if self.std_error_types:
            total = sum(self.std_error_types.values())
            parts = []
            for k,c in self.std_error_types.items():
                ids = self.error_cids.get(k,[])
                if ids:
                    s = f"{k}:{c} (CIDs: {', '.join(ids[:3])}{'...' if len(ids)>3 else ''})"
                else:
                    s = f"{k}:{c}"
                parts.append(s)
            logger.warning(f"Standardization/timeout errors in batch - Total: {total} [{'; '.join(parts)}]")

def standardize_molecules_batch(mols: List[Optional[Chem.Mol]],
                                std_core=True, use_normalizer=False, use_reionizer=False,
                                metal_disconnector=False, largest_fragment=False,
                                error_tracker: Optional[SimpleErrorTracker]=None, mol_ids: List[str]=None):
    stded=[]; status=[]
    for i,m in enumerate(mols):
        mid = mol_ids[i] if mol_ids and i<len(mol_ids) else f"mol_{i}"
        if m is None:
            stded.append(None); status.append("failed(parsing)"); continue
        s="success"; errs=[]
        try:
            w = Chem.Mol(m)
            if std_core:
                try: w = rdMolStandardize.Cleanup(w)
                except Exception: errs.append("Cleanup"); error_tracker and error_tracker.log_std_error_batch("Cleanup", mid)
            if use_normalizer and _GLOBAL_OBJECTS['normalizer']:
                try: w = _GLOBAL_OBJECTS['normalizer'].normalize(w)
                except Exception: errs.append("Normalizer"); error_tracker and error_tracker.log_std_error_batch("Normalizer", mid)
            if use_reionizer and _GLOBAL_OBJECTS['reionizer']:
                try: w = _GLOBAL_OBJECTS['reionizer'].reionize(w)
                except Exception: errs.append("Reionizer"); error_tracker and error_tracker.log_std_error_batch("Reionizer", mid)
            if metal_disconnector and _GLOBAL_OBJECTS['metal_disconnector']:
                try: w = _GLOBAL_OBJECTS['metal_disconnector'].Disconnect(w)
                except Exception: errs.append("MetalDisconnector"); error_tracker and error_tracker.log_std_error_batch("MetalDisconnector", mid)
            if largest_fragment and _GLOBAL_OBJECTS['fragment_chooser']:
                try: w = _GLOBAL_OBJECTS['fragment_chooser'].choose(w)
                except Exception: errs.append("LargestFragment"); error_tracker and error_tracker.log_std_error_batch("LargestFragment", mid)
            try:
                Chem.SanitizeMol(w); stded.append(w)
                if errs: s=f"partial_success({','.join(errs)})"
            except Exception:
                errs.append("FinalSanitize"); error_tracker and error_tracker.log_std_error_batch("FinalSanitize", mid)
                stded.append(None); s=f"failed({','.join(errs)})"
        except Exception:
            errs.append("Unexpected"); error_tracker and error_tracker.log_std_error_batch("Unexpected", mid)
            stded.append(None); s=f"failed({','.join(errs)})"
        status.append(s)
    return stded, status

def retry_inchi_after_std_failure(mols, std_status, parse_sources, inchi_list, mol_ids, std_config,
                                  error_tracker: Optional[SimpleErrorTracker]=None):
    stats={'attempted':0,'recovered':0}
    for i,s in enumerate(std_status):
        failed = isinstance(s,str) and s.startswith("failed(")
        if failed and parse_sources[i]=="smiles":
            inc = inchi_list[i] if inchi_list else None
            if isinstance(inc,str) and inc.strip():
                stats['attempted']+=1
                try:
                    inc2 = inc.strip(); inc2 = inc2 if inc2.startswith("InChI=") else "InChI="+inc2
                    m2 = Chem.MolFromInchi(inc2)
                    if m2 is None: continue
                    Chem.SanitizeMol(m2)
                    m2_list,m2_stat = standardize_molecules_batch([m2], **std_config, error_tracker=None, mol_ids=[mol_ids[i] if mol_ids and i<len(mol_ids) else f"mol_{i}"])
                    m2s = m2_list[0]; m2st = m2_stat[0]
                    if m2s is not None and not (isinstance(m2st,str) and m2st.startswith("failed(")):
                        mols[i]=m2s; std_status[i]=f"recovered_via_inchi({m2st})"; parse_sources[i]="inchi_fallback_after_std"; stats['recovered']+=1
                except Exception:
                    pass
    return mols, std_status, parse_sources, stats

def normalize_final_status(std_status, parse_sources, mols):
    out=[]
    for i,s in enumerate(std_status):
        if mols[i] is not None:
            if parse_sources[i].startswith("inchi_fallback"):
                if isinstance(s,str) and s.startswith("recovered_via_inchi("):
                    inner = s[len("recovered_via_inchi("):-1]
                    if inner.startswith("partial_success("): out.append(inner.replace("partial_success(", "partial_success_via_inchi("))
                    else: out.append("success_via_inchi")
                elif isinstance(s,str) and s.startswith("partial_success("):
                    out.append(s.replace("partial_success(", "partial_success_via_inchi("))
                else: out.append("success_via_inchi")
            else:
                out.append(s if (isinstance(s,str) and s.startswith("partial_success(")) else "success")
        else:
            out.append(s if (isinstance(s,str) and s.startswith("failed(")) else "failed(unknown)")
    return out

# ---------- Mordred postprocess ----------
def _maybe_vector_string_to_scalar(x):
    if isinstance(x, str):
        s = x.strip()
        m = re.fullmatch(r"\[\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\]", s)
        if m: return m.group(1)
    return x

def safe_mordred_postprocess(df_mordred: pd.DataFrame, logger, master_schema: Optional[List[str]]=None):
    if df_mordred.empty: return None, []
    logger.info(f"DEBUG: Post-processing {df_mordred.shape[1]} raw Mordred descriptors")
    if master_schema: logger.info("DEBUG: Master schema provided - schema-aware postprocess")
    # bool → int
    bool_cols = df_mordred.select_dtypes(include=['bool']).columns.tolist()
    for c in bool_cols: df_mordred[c] = df_mordred[c].astype(int)
    # object → numeric
    obj_cols = df_mordred.select_dtypes(include=['object']).columns.tolist()
    kept=[]; removed=[]
    for c in obj_cols:
        in_master = master_schema and c in master_schema
        try:
            if in_master: df_mordred[c] = df_mordred[c].map(_maybe_vector_string_to_scalar)
            df_mordred[c] = pd.to_numeric(df_mordred[c], errors='coerce'); kept.append(c)
        except:
            if in_master:
                df_mordred[c] = np.nan; kept.append(c)
            else:
                df_mordred.drop(columns=[c], inplace=True); removed.append(c)
    # numeric only
    num_cols = df_mordred.select_dtypes(include=['number']).columns.tolist()
    df_final = df_mordred[num_cols].copy()
    vals = df_final.values.astype(np.float64)
    vals = np.where(np.isinf(vals), np.nan, vals)
    return vals, df_final.columns.tolist()

# ---------- RDKit (single-mol helper used by worker) ----------
def compute_rdkit_for_mol(mol: Optional[Chem.Mol]) -> Dict[str,float]:
    res={}
    if mol is None: return res
    funcs = _GLOBAL_OBJECTS['rdkit_funcs']; names=_GLOBAL_OBJECTS['rdkit_func_names']
    for name,fn in zip(names,funcs):
        try:
            v = fn(mol)
            if isinstance(v,(tuple,list)) and len(v)==1: v=v[0]
            res[name] = float(v) if isinstance(v,(int,float)) and np.isfinite(v) else np.nan
        except: res[name]=np.nan
    return res

# ---------- NEW (process-timeout): supervised single worker for per-molecule timeout ----------
def _flush_queue(q):
    try:
        while True: q.get_nowait()
    except queue.Empty:
        pass

def _worker_loop(in_q: mp.Queue, out_q: mp.Queue, worker_cfg: dict, master_schema: Optional[List[str]]):
    """
    Runs in a separate process for the whole chunk (restarted on timeout).
    Receives tasks: (cid, standardized_smiles, desc_set)
    Sends results: ('ok', cid, dict) or ('err', cid, errstr)
    """
    # Initialize heavy globals in child
    try:
        init_global_objects_optimized(**worker_cfg)
    except Exception as e:
        out_q.put(('fatal', None, f'init_failed:{e}')); return

    class DummyLog:
        def info(self, *a, **k): pass
        def warning(self, *a, **k): pass

    while True:
        msg = in_q.get()
        if msg is None:
            break
        cid, std_smiles, desc_set = msg
        try:
            # Rebuild mol from standardized smiles
            m = Chem.MolFromSmiles(std_smiles) if std_smiles else None
            if m is not None:
                Chem.SanitizeMol(m)
            # RDKit
            rd = compute_rdkit_for_mol(m) if desc_set in ('rdkit','both') else {}
            # Mordred
            md={}
            if desc_set in ('mordred','both'):
                calc = _GLOBAL_OBJECTS['mordred_calc']
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    df = calc.pandas([m if m is not None else Chem.Mol()], nproc=1)
                cols = normalize_column_names(flatten_multiindex_columns(df.columns))
                df.columns = cols
                vals, mcols = safe_mordred_postprocess(df, DummyLog(), master_schema)
                if vals is not None:
                    md = {c: vals[0, i] for i,c in enumerate(mcols)}
            # Overlap removal (case-insensitive)
            if rd and md and desc_set=='both':
                rd_lower = {k.lower() for k in rd}
                md = {k:v for k,v in md.items() if k.lower() not in rd_lower}
            out = rd; out.update(md)
            out_q.put(('ok', cid, out))
        except Exception as e:
            out_q.put(('err', cid, str(e)))

def _start_worker(worker_cfg: dict, master_schema: Optional[List[str]]):
    ctx = mp.get_context('fork')  # Linux HPC 기본; spawn보다 초기화 가벼움
    in_q = ctx.Queue(maxsize=1)
    out_q = ctx.Queue(maxsize=1)
    p = ctx.Process(target=_worker_loop, args=(in_q, out_q, worker_cfg, master_schema), daemon=False)
    p.start()
    return p, in_q, out_q

def _stop_worker(p, in_q, out_q, graceful=True):
    try:
        if graceful and p.is_alive():
            in_q.put_nowait(None)
    except Exception:
        pass
    try:
        if p.is_alive():
            p.join(timeout=2.0)
    except Exception:
        pass
    if p.is_alive():
        try: p.terminate()
        except Exception: pass
        try: p.join(timeout=2.0)
        except Exception: pass
    # Best effort to drain queues
    _flush_queue(out_q)
    _flush_queue(in_q)

# ---------- Create final DF ----------
def create_final_dataframe(mol_ids, original_smiles, standardized_smiles, parse_sources, standardization_status,
                           rows_of_descriptors: List[Dict[str,float]], source_file: Optional[str]):
    data = {
        'CID': mol_ids,
        'isomeric_smiles': original_smiles,
        'standardized_smiles': standardized_smiles,
        'parse_source': parse_sources,
        'standardization_status': standardization_status,
    }
    if source_file: data['source_file'] = [source_file]*len(mol_ids)
    # Merge descriptor dicts column-wise
    if rows_of_descriptors:
        keys = set()
        for d in rows_of_descriptors:
            keys.update(d.keys())
        keys = list(keys)
        for k in keys:
            data[k] = [row.get(k, np.nan) for row in rows_of_descriptors]
    df = pd.DataFrame(data)
    return df

# ---------- NEW (process-timeout): per-molecule processing inside chunk ----------
def _process_chunk_with_mol_timeout_proc(chunk_data, chunk_idx, config, source_file):
    smiles_list, inchi_list, mol_ids = chunk_data
    tracker = SimpleErrorTracker(); tracker.total = len(smiles_list)
    init_global_objects_optimized(**config)

    # 1) parse
    mols, parse_sources, canon_smiles = parse_molecules_vectorized(smiles_list, inchi_list, mol_ids)
    tracker.parse_errors += sum(1 for s in parse_sources if s == "none")

    # 2) standardize (+ optional retry via InChI)
    mols, std_status_raw = standardize_molecules_batch(
        mols,
        std_core=config['std_core'],
        use_normalizer=config['use_normalizer'],
        use_reionizer=config['use_reionizer'],
        metal_disconnector=config['metal_disconnector'],
        largest_fragment=config['largest_fragment'],
        error_tracker=None,
        mol_ids=mol_ids
    )
    recovery_stats = {'attempted':0,'recovered':0}
    if config.get('enable_inchi_fallback_after_std', True):
        std_cfg = {
            'std_core': config['std_core'],
            'use_normalizer': config['use_normalizer'],
            'use_reionizer': config['use_reionizer'],
            'metal_disconnector': config['metal_disconnector'],
            'largest_fragment': config['largest_fragment'],
        }
        mols, std_status_after, parse_sources, rec = retry_inchi_after_std_failure(
            mols, std_status_raw, parse_sources, inchi_list, mol_ids, std_cfg, tracker
        )
        recovery_stats = rec
    else:
        std_status_after = std_status_raw
    final_status = normalize_final_status(std_status_after, parse_sources, mols)

    # 3) supervised worker (timeout)
    std_smiles=[]
    for m in mols:
        if m is not None:
            try: std_smiles.append(Chem.MolToSmiles(m, isomericSmiles=True))
            except: std_smiles.append(None)
        else:
            std_smiles.append(None)

    log_dir = Path(config['log_dir']) if config.get('log_dir') else None
    timeouts_path = log_dir / f"{Path(source_file).stem}_timeouts.tsv" if log_dir else None
    metrics = {'timeout':0, 'failed':0, 'success':0}

    master_schema = _SCHEMA_MANAGER['master_schema'] if _SCHEMA_MANAGER['schema_loaded'] else None
    rows_desc: List[Dict[str,float]] = []

    worker_cfg = dict(config); worker_cfg.pop('mol_timeout', None); worker_cfg.pop('log_dir', None)
    proc, in_q, out_q = _start_worker(worker_cfg, master_schema)

    for i, cid in enumerate(mol_ids):
        if mols[i] is None or (isinstance(final_status[i], str) and final_status[i].startswith("failed(")):
            tracker.log_desc_error(); metrics['failed'] += 1
            rows_desc.append({})
            continue

        try:
            _flush_queue(out_q)
            task = (cid, std_smiles[i], config['desc_set'])
            in_q.put(task)
            try:
                ok, rcid, payload = out_q.get(timeout=float(config['mol_timeout']))
                if ok == 'ok' and rcid == cid:
                    rows_desc.append(payload); metrics['success'] += 1
                elif ok == 'err':
                    rows_desc.append({}); metrics['failed'] += 1
                    tracker.log_desc_error(); tracker.log_std_error_batch("DescriptorError", cid)
                elif ok == 'fatal':
                    rows_desc.append({}); metrics['failed'] += 1
                    tracker.log_desc_error(); tracker.log_std_error_batch("WorkerInit", cid)
                else:
                    rows_desc.append({}); metrics['failed'] += 1
                    tracker.log_desc_error(); tracker.log_std_error_batch("UnknownWorkerMsg", cid)
            except queue.Empty:
                metrics['timeout'] += 1; metrics['failed'] += 1
                tracker.log_desc_error(); tracker.log_std_error_batch("Timeout", cid)
                if timeouts_path:
                    _append_tsv_line(
                        timeouts_path,
                        header="timestamp\tfile\tchunk_idx\tCID\treason",
                        line=f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t{source_file}\t{chunk_idx}\t{cid}\ttimeout>{config['mol_timeout']}"
                    )
                final_status[i] = "failed(skip)"
                rows_desc.append({})
                _stop_worker(proc, in_q, out_q, graceful=False)
                proc, in_q, out_q = _start_worker(worker_cfg, master_schema)
        except Exception as e:
            rows_desc.append({})
            metrics['failed'] += 1
            tracker.log_desc_error(); tracker.log_std_error_batch("DescriptorError", cid)

    _stop_worker(proc, in_q, out_q, graceful=True)

    # 4) per-chunk metrics TSV
    if log_dir:
        metrics_path = log_dir / f"{Path(source_file).stem}_chunk_metrics.tsv"
        n_total = len(mol_ids); n_success = metrics['success']; n_failed = metrics['failed']
        success_rate = (n_success / n_total * 100.0) if n_total else 0.0
        _append_tsv_line(
            metrics_path,
            header="timestamp\tfile\tchunk_idx\tn_total\tn_success\tn_failed\tn_timeout\tsuccess_rate",
            line=f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t{source_file}\t{chunk_idx}\t{n_total}\t{n_success}\t{n_failed}\t{metrics['timeout']}\t{success_rate:.2f}"
        )

    df = create_final_dataframe(
        mol_ids, smiles_list, std_smiles, parse_sources, final_status, rows_desc, source_file
    )
    return df, tracker, recovery_stats

# ---------- Vectorized (original) chunk path (fallback when no timeout) ----------
def calculate_rdkit_descriptors_vectorized(mols: List[Optional[Chem.Mol]]) -> np.ndarray:
    funcs = _GLOBAL_OBJECTS['rdkit_funcs']
    if not funcs: raise ValueError("RDKit functions not initialized")
    n_mols = len(mols); n_descs = len(funcs)
    data = np.full((n_mols, n_descs), np.nan, dtype=np.float64)
    valid = [i for i,m in enumerate(mols) if m is not None]
    if not valid: return data
    for j,fn in enumerate(funcs):
        for i in valid:
            try:
                v = fn(mols[i])
                if isinstance(v,(tuple,list)) and len(v)==1: v=v[0]
                if isinstance(v,(int,float)) and np.isfinite(v):
                    data[i,j] = float(v)
            except: pass
    return data

def calculate_mordred_descriptors_vectorized(mols: List[Optional[Chem.Mol]], master_schema: Optional[List[str]]=None):
    calc = _GLOBAL_OBJECTS['mordred_calc']
    if not calc: return None
    safe = [m if m is not None else Chem.Mol() for m in mols]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = calc.pandas(safe, nproc=1)
    if df.empty: return None
    cols = normalize_column_names(flatten_multiindex_columns(df.columns)); df.columns = cols
    class DL: 
        def info(self,*a,**k): pass
        def warning(self,*a,**k): pass
    vals, final_cols = safe_mordred_postprocess(df, DL(), master_schema)
    if vals is None: return None
    return vals, final_cols

def remove_overlapping_descriptors(mordred_data, mordred_cols: List[str], rdkit_cols: List[str]):
    if mordred_data is None or not mordred_cols or not rdkit_cols:
        return mordred_data, mordred_cols
    rd_lower = {c.lower() for c in rdkit_cols}
    keep_idx=[]; keep_cols=[]
    for i,c in enumerate(mordred_cols):
        if c.lower() not in rd_lower:
            keep_idx.append(i); keep_cols.append(c)
    if not keep_idx: return np.empty((len(mordred_data),0)), []
    return mordred_data[:, keep_idx], keep_cols

def process_chunk_optimized(args_tuple):
    chunk_data, chunk_idx, config, source_file = args_tuple
    # NEW (process-timeout)
    if config.get('mol_timeout') and float(config['mol_timeout']) > 0:
        print(f"DEBUG: Chunk {chunk_idx} using process-based per-molecule timeout = {config['mol_timeout']}s")
        return _process_chunk_with_mol_timeout_proc(chunk_data, chunk_idx, config, source_file)

    # Otherwise vectorized path
    init_global_objects_optimized(**config)
    smiles_list, inchi_list, mol_ids = chunk_data
    tracker = SimpleErrorTracker(); tracker.total = len(smiles_list)

    mols, parse_sources, canon = parse_molecules_vectorized(smiles_list, inchi_list, mol_ids)
    tracker.parse_errors += sum(1 for s in parse_sources if s=="none")

    mols, std_status_raw = standardize_molecules_batch(
        mols,
        std_core=config['std_core'],
        use_normalizer=config['use_normalizer'],
        use_reionizer=config['use_reionizer'],
        metal_disconnector=config['metal_disconnector'],
        largest_fragment=config['largest_fragment'],
        error_tracker=None, mol_ids=mol_ids
    )
    recovery_stats={'attempted':0,'recovered':0}
    if config.get('enable_inchi_fallback_after_std', True):
        std_cfg = {
            'std_core': config['std_core'],
            'use_normalizer': config['use_normalizer'],
            'use_reionizer': config['use_reionizer'],
            'metal_disconnector': config['metal_disconnector'],
            'largest_fragment': config['largest_fragment'],
        }
        mols, std_status_after, parse_sources, rec = retry_inchi_after_std_failure(
            mols, std_status_raw, parse_sources, inchi_list, mol_ids, std_cfg, tracker
        )
        recovery_stats = rec
    else:
        std_status_after = std_status_raw
    final_status = normalize_final_status(std_status_after, parse_sources, mols)

    std_smiles=[]
    for m in mols:
        if m is not None:
            try: std_smiles.append(Chem.MolToSmiles(m, isomericSmiles=True))
            except: std_smiles.append(None)
        else:
            std_smiles.append(None)

    rdkit_data=None; rdkit_cols=[]; mordred_data=None; mordred_cols=[]
    master_schema = _SCHEMA_MANAGER['master_schema'] if _SCHEMA_MANAGER['schema_loaded'] else None
    if config['desc_set'] in ('rdkit','both'):
        rdkit_data = calculate_rdkit_descriptors_vectorized(mols); rdkit_cols = _GLOBAL_OBJECTS['rdkit_func_names']
    if config['desc_set'] in ('mordred','both'):
        md = calculate_mordred_descriptors_vectorized(mols, master_schema)
        if md:
            mordred_data, mordred_cols = md
            mordred_data, mordred_cols = remove_overlapping_descriptors(mordred_data, mordred_cols, rdkit_cols)

    rows_desc=[]
    if rdkit_data is not None:
        for i in range(len(mol_ids)):
            rows_desc.append({})
        for j, col in enumerate(rdkit_cols):
            col_vals = rdkit_data[:, j]
            for i in range(len(mol_ids)):
                rows_desc[i][col] = float(col_vals[i]) if np.isfinite(col_vals[i]) else np.nan
    if mordred_data is not None:
        for j, col in enumerate(mordred_cols):
            col_vals = mordred_data[:, j]
            for i in range(len(mol_ids)):
                rows_desc[i][col] = float(col_vals[i]) if np.isfinite(col_vals[i]) else np.nan

    df = create_final_dataframe(mol_ids, smiles_list, std_smiles, parse_sources, final_status, rows_desc, source_file)
    return df, tracker, recovery_stats

# ---------- Output ----------
def write_output_optimized(df: pd.DataFrame, output_path: Path, format_type: str):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if format_type == "parquet":
        df.to_parquet(output_path, index=False, engine='auto', compression='snappy')
    else:
        df.to_csv(output_path, index=False, float_format='%.9g')

def write_output_streaming_partitioned(df: pd.DataFrame, output_path: Path, format_type: str, chunk_idx: int=0, mode: str='w', header: bool=True):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if format_type == "parquet":
        parts_dir = output_path.parent / f"{output_path.stem}.parts"
        parts_dir.mkdir(parents=True, exist_ok=True)
        part_path = parts_dir / f"part-{chunk_idx:06d}.parquet"
        df.to_parquet(part_path, index=False, engine='auto', compression='snappy')
        print(f"DEBUG: Written parquet part: {part_path}")
    else:
        df.to_csv(output_path, index=False, float_format='%.9g', mode=mode, header=header)

def consolidate_parquet_parts(parts_dir: Path, output_path: Path, remove_parts: bool=True):
    if not HAS_PYARROW:
        print("WARNING: PyArrow not available. Cannot consolidate parquet parts efficiently."); return False
    try:
        if not parts_dir.exists():
            print(f"Parts directory not found: {parts_dir}"); return False
        part_files = list(parts_dir.glob("part-*.parquet"))
        if not part_files:
            print(f"No parquet parts found in: {parts_dir}"); return False
        print(f"Consolidating {len(part_files)} parquet parts...")
        dset = ds.dataset(str(parts_dir), format="parquet")
        table = dset.to_table()
        pq.write_table(table, str(output_path))
        if remove_parts:
            for f in part_files: f.unlink()
            parts_dir.rmdir()
        print(f"Consolidated parquet saved to: {output_path}")
        return True
    except Exception as e:
        print(f"Failed to consolidate parquet parts: {e}"); return False

# ---------- NEW: Resume helper ----------
def list_existing_part_indices(parts_dir: Path) -> List[int]:
    """Return sorted list of existing parquet part indices in parts_dir."""
    idxs = []
    for p in parts_dir.glob("part-*.parquet"):
        m = re.match(r"part-(\d{6})\.parquet$", p.name)
        if m:
            try:
                idxs.append(int(m.group(1)))
            except:
                pass
    return sorted(idxs)

# ---------- Input ----------
def read_input_smart_optimized(path: Path, chunksize: int) -> Iterable[pd.DataFrame]:
    suffix = path.suffix.lower()
    file_size_mb = path.stat().st_size / (1024*1024)
    if suffix in (".parquet",".pq"):
        if HAS_PYARROW and file_size_mb > 50:
            print(f"Using PyArrow streaming for large parquet: {file_size_mb:.1f} MB")
            try:
                parquet_file = pq.ParquetFile(path)
                for batch in parquet_file.iter_batches(batch_size=chunksize):
                    yield batch.to_pandas()
                return
            except Exception as e:
                print(f"PyArrow streaming failed, falling back to pandas: {e}")
        if file_size_mb < 50:
            yield pd.read_parquet(path)
        else:
            df = pd.read_parquet(path)
            for i in range(0, len(df), chunksize):
                yield df.iloc[i:i+chunksize].copy()
    elif suffix == ".csv":
        for ch in pd.read_csv(path, chunksize=chunksize, low_memory=False):
            yield ch
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

def read_input_smart(path: Path, chunksize: int) -> Iterable[pd.DataFrame]:
    return read_input_smart_optimized(path, chunksize)

def create_secure_temp_file(suffix: str, prefix: str="descriptor_") -> Path:
    tmp = Path(tempfile.gettempdir())
    name = f"{prefix}{uuid.uuid4().hex[:8]}_{int(time.time())}.{suffix}"
    return tmp / name

# ---------- File / driver ----------
def get_optimal_chunksize(file_size_mb: float, n_jobs: int, manual_chunksize: Optional[int]=None):
    if manual_chunksize is not None: return manual_chunksize
    base=1000
    if file_size_mb < 10: return min(base, max(100, int(file_size_mb*100)))
    elif file_size_mb < 100: return base*2
    elif file_size_mb < 1000: return base*5
    else: return base*10

def process_file_optimized_streaming_practical(input_file: Path, output_file: Path, config: dict,
                                              n_jobs: int, logger: logging.Logger, schema_path: Optional[Path]=None):
    file_size_mb = input_file.stat().st_size/(1024*1024)
    optimal_chunksize = get_optimal_chunksize(file_size_mb, n_jobs, config.get('chunksize'))
    logger.info(f"Processing {input_file.name} ({file_size_mb:.1f} MB) with chunksize {optimal_chunksize} [MASTER SCHEMA ENFORCED]")

    total_mols=0; combined_tracker=SimpleErrorTracker(); total_recovery={'attempted':0,'recovered':0}
    start=time.time()
    use_threads = os.environ.get("DESCRIPTOR_INNER_PARALLEL")=="threads"

    # --- RESUME-AWARE OUTPUT HANDLING (Parquet only) ---
    parts_dir = None
    existing_part_indices = set()
    if config['format'] == "parquet":
        parts_dir = output_file.parent / f"{output_file.stem}.parts"
        if parts_dir.exists():
            existed = list_existing_part_indices(parts_dir)
            if existed:
                existing_part_indices = set(existed)
                logger.info(
                    f"Resume mode: found {len(existed)} existing parts in {parts_dir} "
                    f"(max index = {max(existed):06d}). Will skip those chunks."
                )
        # If only consolidated file exists (no parts), remove it to rebuild from scratch on this run.
        if output_file.exists() and not existing_part_indices:
            try:
                output_file.unlink()
                logger.info(f"Removed previous consolidated output: {output_file}")
            except Exception as e:
                logger.warning(f"Could not remove existing output {output_file}: {e}")
    else:
        # CSV 등 Parquet 이 아닌 경우는 기존 동작: 깨끗이 시작
        if output_file.exists():
            try:
                output_file.unlink()
            except Exception as e:
                logger.warning(f"Could not remove existing output {output_file}: {e}")

    chunk_count_new=0  # 이번 실행에서 실제로 처리한 청크 수
    source_name=input_file.name

    if n_jobs<=1: executor=None; logger.info("Processing chunks sequentially")
    elif use_threads:
        executor=ThreadPoolExecutor(max_workers=n_jobs); logger.info(f"Processing chunks with {n_jobs} threads")
    else:
        executor=Pool(n_jobs); logger.info(f"Processing chunks with {n_jobs} processes")

    try:
        for cidx, chunk in enumerate(read_input_smart_optimized(input_file, optimal_chunksize)):
            # --- Skip if this chunk has already been materialized as a parquet part ---
            if config['format']=="parquet":
                would_be_part = (output_file.parent / f"{output_file.stem}.parts" / f"part-{cidx:06d}.parquet")
                if (cidx in existing_part_indices) or (would_be_part.exists()):
                    logger.info(f"Skipping chunk {cidx} ({len(chunk)} rows) - part-{cidx:06d}.parquet already exists")
                    continue

            if config['smiles_col'] not in chunk.columns:
                logger.warning(f"SMILES column '{config['smiles_col']}' not found in chunk {cidx}, skipping"); continue

            smiles = chunk[config['smiles_col']].astype(str).tolist()
            inchi  = (chunk[config['inchi_col']].astype(str).tolist() if config['inchi_col'] in chunk.columns else [None]*len(chunk))
            ids    = (chunk[config['id_col']].astype(str).tolist() if config['id_col'] in chunk.columns else [f"mol_{i}" for i in range(len(chunk))])

            # count only when we actually process this chunk (not skipped)
            total_mols += len(smiles)

            args_tup = ((smiles, inchi, ids), cidx, config, source_name)
            logger.info(f"Processing chunk {cidx+1} with {len(smiles)} molecules...")

            if executor is None:
                df, tracker, rec = process_chunk_optimized(args_tup)
            else:
                if use_threads:
                    fut = executor.submit(process_chunk_optimized, args_tup); df, tracker, rec = fut.result()
                else:
                    df, tracker, rec = executor.apply(process_chunk_optimized, (args_tup,))

            total_recovery['attempted'] += rec.get('attempted',0); total_recovery['recovered'] += rec.get('recovered',0)
            if df.empty:
                logger.warning(f"Chunk {cidx} returned empty DataFrame, skipping"); continue

            combined_tracker.merge(tracker); tracker.report_batch_summary(logger)

            if _SCHEMA_MANAGER['schema_loaded']: df = enforce_master_schema(df, logger)
            else: logger.warning("Master schema not loaded - output columns may vary")

            # parquet: write as part-{cidx:06d}.parquet; csv: stream append
            write_output_streaming_partitioned(df, output_file, config['format'], chunk_idx=cidx, mode=('w' if cidx==0 else 'a'), header=(cidx==0))
            del df; gc.collect()
            chunk_count_new += 1
            logger.info(f"Chunk {cidx+1} completed and written.")

        # --- Consolidate if parts exist (keep parts by default for future resume) ---
        if config['format']=="parquet":
            parts_dir = output_file.parent / f"{output_file.stem}.parts"
            if parts_dir.exists():
                existing_parts = list(parts_dir.glob("part-*.parquet"))
                if existing_parts:
                    logger.info("Consolidating parquet parts...")
                    ok = consolidate_parquet_parts(parts_dir, output_file, remove_parts=False)
                    if not ok:
                        logger.warning("Failed to consolidate parts - leaving as-is")

        if executor is not None:
            if use_threads: executor.shutdown(wait=True)
            else: executor.close(); executor.join()
    except Exception as e:
        if executor is not None:
            if use_threads: executor.shutdown(wait=False)
            else: executor.terminate(); executor.join()
        raise

    if (chunk_count_new==0) and (not existing_part_indices):
        logger.warning(f"No valid data in {input_file.name}")
        return 0, SimpleErrorTracker()

    dur = time.time()-start
    speed = total_mols/dur if dur>0 else 0.0
    if total_recovery['attempted']>0:
        rr = (total_recovery['recovered']/total_recovery['attempted'])*100.0
        logger.info(f"InChI fallback recovery: {total_recovery['recovered']}/{total_recovery['attempted']} ({rr:.1f}%)")

    final_desc_count = _descriptor_count_from_schema() if _SCHEMA_MANAGER['schema_loaded'] else 'N/A'
    total_fail = combined_tracker.parse_errors + combined_tracker.std_errors + combined_tracker.desc_errors
    success_rate = ((combined_tracker.total - total_fail)/max(combined_tracker.total,1))*100.0

    file_summary = {
        'file_name': input_file.name,
        'molecules_processed': total_mols,
        'parse_errors': combined_tracker.parse_errors,
        'std_errors': combined_tracker.std_errors,
        'desc_errors': combined_tracker.desc_errors,
        'recovery_attempted': combined_tracker.recovery_stats['attempted'],
        'recovery_recovered': combined_tracker.recovery_stats['recovered'],
        'success_rate': success_rate,
        'processing_time': dur,
        'molecules_per_second': speed,
        'total_descriptors': final_desc_count
    }
    _SUMMARY_STATS['file_summaries'].append(file_summary)

    logger.info("="*60)
    logger.info(f"FILE PROCESSING COMPLETE: {input_file.name}")
    logger.info(f"Molecules processed (this run): {total_mols}")
    logger.info(f"Parse errors: {combined_tracker.parse_errors}")
    logger.info(f"Standardization errors: {combined_tracker.std_errors}")
    logger.info(f"Descriptor errors: {combined_tracker.desc_errors}")
    logger.info(f"Success rate: {success_rate:.1f}%")
    logger.info(f"Processing time: {dur:.2f}s  speed: {speed:.1f} mol/s")
    logger.info("="*60)
    return total_mols, combined_tracker

def process_file_optimized_streaming(input_file: Path, output_file: Path, config: dict,
                                     n_jobs: int, logger: logging.Logger, schema_path: Optional[Path]=None):
    return process_file_optimized_streaming_practical(input_file, output_file, config, n_jobs, logger, schema_path)

def process_file_optimized(input_file: Path, output_file: Path, config: dict,
                           n_jobs: int, logger: logging.Logger, schema_path: Optional[Path]=None):
    return process_file_optimized_streaming_practical(input_file, output_file, config, n_jobs, logger, schema_path)

def determine_parallel_mode(input_files: List[Path], n_jobs: int, parallel_mode: str):
    if parallel_mode=="chunk": return "chunk", n_jobs, 1
    elif parallel_mode=="file":
        n_files=len(input_files)
        if n_files >= n_jobs: return "file", n_jobs, 1
        else: return "file", n_files, max(1, n_jobs//n_files)
    else:
        n_files=len(input_files)
        if n_files==1: return "chunk", n_jobs, 1
        elif n_files >= n_jobs: return "file", n_jobs, 1
        else: return "file", n_files, max(1, n_jobs//n_files)

def write_summary_csv(output_path: Path):
    if not _SUMMARY_STATS['file_summaries']: return
    summary_path = output_path.parent / "processing_summary.csv"
    try:
        df = pd.DataFrame(_SUMMARY_STATS['file_summaries'])
        df.to_csv(summary_path, index=False)
        print(f"Processing summary written to: {summary_path}")
    except Exception as e:
        print(f"Failed to write summary CSV: {e}")

# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser(description="Descriptor calculator (MASTER SCHEMA enforced) with process-based per-molecule timeout + RESUME from existing parquet parts")
    # I/O
    p.add_argument("-i","--input", required=True)
    p.add_argument("-o","--output", required=True)
    p.add_argument("--smiles-col", default="SMILES::Absolute")
    p.add_argument("--inchi-col",  default="InChI::Standard")
    p.add_argument("--id-col",     default="CID")
    # Descriptor
    p.add_argument("--desc-set", choices=["rdkit","mordred","both"], default="both")
    p.add_argument("--include-fragments", action="store_true", default=True)
    p.add_argument("--no-fragments", dest="include_fragments", action="store_false")
    p.add_argument("--use-3d", action="store_true", default=False)
    # Output options
    p.add_argument("--format", choices=["csv","parquet"], default="csv")
    p.add_argument("--combine-output", action="store_true", default=False)
    p.add_argument("--suffix", default="_descriptors")
    p.add_argument("--schema", required=True)
    # Processing
    p.add_argument("--n-jobs", type=int, default=None)
    p.add_argument("--chunksize", type=int, default=None)
    p.add_argument("--parallel-mode", choices=["chunk","file","auto"], default="auto")
    # Standardization
    p.add_argument("--std-core", action="store_true", default=True)
    p.add_argument("--no-std-core", dest="std_core", action="store_false")
    p.add_argument("--normalizer", action="store_true", default=False)
    p.add_argument("--reionizer", action="store_true", default=False)
    p.add_argument("--metal-disconnector", action="store_true", default=False)
    p.add_argument("--largest-fragment", action="store_true", default=False)
    p.add_argument("--enable-inchi-fallback-after-std", action="store_true", default=True)
    p.add_argument("--no-inchi-fallback-after-std", dest="enable_inchi_fallback_after_std", action="store_false")
    # Logging
    p.add_argument("--verbose", action="store_true", default=False)
    p.add_argument("--log-file")
    # NEW (process-timeout)
    p.add_argument("--mol-timeout", type=float, default=None,
                   help="Per-molecule wall-time limit (seconds). On timeout: worker is killed, molecule marked failed(skip), chunk continues.")
    return p.parse_args()

def main():
    args = parse_args()
    logger = setup_logging_optimized(args.log_file)
    if args.verbose: logger.setLevel(logging.DEBUG)
    np.seterr(all="ignore"); warnings.filterwarnings("ignore", category=RuntimeWarning)

    n_jobs = args.n_jobs or cpu_count()
    logger.info("Starting descriptor calculation with MASTER SCHEMA ENFORCEMENT")
    logger.info(f"Workers: {n_jobs}, Descriptor set: {args.desc_set}, Mol-timeout: {args.mol_timeout or 'off'}")
    logger.info(f"PyArrow available: {HAS_PYARROW}")

    schema_path = Path(args.schema)
    master_schema = load_master_schema(schema_path, logger)

    if args.enable_inchi_fallback_after_std:
        logger.info("InChI fallback after standardization failure: ENABLED")
    else:
        logger.info("InChI fallback after standardization failure: DISABLED")

    # compute log_dir
    log_dir = None
    if args.log_file:
        lp = Path(args.log_file)
        log_dir = (lp if lp.is_dir() or str(args.log_file).endswith(('/', '\\')) else lp.parent)

    config = {
        'std_core': args.std_core,
        'use_normalizer': args.normalizer,
        'use_reionizer': args.reionizer,
        'metal_disconnector': args.metal_disconnector,
        'largest_fragment': args.largest_fragment,
        'include_fragments': args.include_fragments,
        'use_3d': args.use_3d,
        'desc_set': args.desc_set,
        'smiles_col': args.smiles_col,
        'inchi_col': args.inchi_col,
        'id_col': args.id_col,
        'format': args.format,
        'chunksize': args.chunksize,
        'enable_inchi_fallback_after_std': args.enable_inchi_fallback_after_std,
        'mol_timeout': args.mol_timeout,             # NEW (process-timeout)
        'log_dir': str(log_dir) if log_dir else None # for TSV logs
    }

    input_path = Path(args.input)
    if input_path.is_file(): input_files=[input_path]
    elif input_path.is_dir():
        input_files=[]
        for ext in ['.csv','.parquet','.pq']:
            input_files.extend(input_path.glob(f"*{ext}"))
        input_files.sort()
    else:
        logger.error(f"Input path not found: {input_path}"); return 1
    if not input_files:
        logger.error("No supported input files found"); return 1
    logger.info(f"Found {len(input_files)} input files")

    parallel_mode, n_workers, workers_per_file = determine_parallel_mode(input_files, n_jobs, args.parallel_mode)
    if parallel_mode=="file":
        logger.info(f"Using parallelization mode: {parallel_mode} ({n_workers} workers, {workers_per_file} per file)")
    else:
        logger.info(f"Using parallelization mode: {parallel_mode}")

    # File-level parallelization note:
    # this script is typically used with --n-jobs 1 (inner) per file when mol-timeout is enabled.
    if parallel_mode=="file" and len(input_files)>1:
        output_path = Path(args.output)
        config.update({'output_base': str(output_path), 'combine_output': args.combine_output,
                       'all_files': input_files, 'suffix': args.suffix})
        file_args = [(inp, config, workers_per_file, {'log_file': args.log_file}, schema_path) for inp in input_files]
        total_start=time.time()
        os.environ["DESCRIPTOR_INNER_PARALLEL"] = "threads"
        with Pool(n_workers) as pool:
            results = pool.map(process_single_file_wrapper, file_args)
        total_mols=0; overall=SimpleErrorTracker(); temp_files=[]
        for r in results:
            if len(r)==5:
                fname, mols, trk, tmp, summary = r
            else:
                fname, mols, trk, tmp = r; summary=None
            total_mols+=mols; overall.merge(trk)
            if tmp: temp_files.append(tmp)
        if args.combine_output and temp_files:
            combined=[]
            for t in temp_files:
                if t.exists():
                    df = (pd.read_parquet(t) if args.format=="parquet" else pd.read_csv(t))
                    combined.append(df); t.unlink()
            if combined:
                final_df = pd.concat(combined, axis=0, ignore_index=True)
                out = output_path if output_path.suffix else output_path / f"combined_descriptors.{args.format}"
                write_output_optimized(final_df, out, args.format); write_summary_csv(out)
        logger.info("FILE-LEVEL PARALLEL COMPLETE")
        return 0
    else:
        total_start=time.time()
        total_mols=0; overall=SimpleErrorTracker(); temp_files=[]
        output_path = Path(args.output)
        for i, inp in enumerate(input_files):
            if len(input_files)>1 and args.log_file:
                logger = setup_logging_per_file(logger, args.log_file, inp.name)
            logger.info(f"Processing file {i+1}/{len(input_files)}: {inp.name}")
            try:
                if args.combine_output:
                    tmp = create_secure_temp_file(args.format, f"{inp.stem}_temp_"); final_out = tmp; temp_files.append(tmp)
                elif len(input_files)==1:
                    final_out = output_path
                else:
                    if output_path.is_dir() or str(output_path).endswith('/'):
                        output_path.mkdir(parents=True, exist_ok=True)
                        final_out = output_path / f"{inp.stem}{args.suffix}.{args.format}"
                    else:
                        final_out = output_path.parent / f"{inp.stem}{args.suffix}.{args.format}"
                file_mols, trk = process_file_optimized_streaming_practical(inp, final_out, config, n_jobs, logger, schema_path)
                total_mols += file_mols; overall.merge(trk)
            except Exception as e:
                logger.error(f"Error processing {inp.name}: {e}")
                if args.verbose: logger.error(traceback.format_exc())

        if args.combine_output and temp_files:
            combined=[]
            for t in temp_files:
                if t.exists():
                    df = (pd.read_parquet(t) if args.format=="parquet" else pd.read_csv(t))
                    combined.append(df); t.unlink()
            if combined:
                final_df = pd.concat(combined, axis=0, ignore_index=True)
                out = output_path if output_path.suffix else output_path / f"combined_descriptors.{args.format}"
                write_output_optimized(final_df, out, args.format); write_summary_csv(out)

        tot_dur=time.time()-total_start
        logger.info(f"Total molecules processed: {total_mols} in {tot_dur:.2f}s")
        return 0

def process_single_file_wrapper(args_tuple):
    input_file, config, n_jobs_per_file, logger_info, schema_path = args_tuple
    worker_logger = setup_logging_optimized(None)
    if logger_info and 'log_file' in logger_info and logger_info['log_file']:
        worker_logger = setup_logging_per_file(worker_logger, logger_info['log_file'], input_file.name)
    output_base = Path(config['output_base'])
    if config.get('combine_output'):
        temp_out = create_secure_temp_file(config['format'], f"{input_file.stem}_temp_"); final_out = temp_out
    elif len(config['all_files'])==1:
        final_out = output_base
    else:
        if output_base.is_dir() or str(output_base).endswith('/'):
            output_base.mkdir(parents=True, exist_ok=True)
            final_out = output_base / f"{input_file.stem}{config['suffix']}.{config['format']}"
        else:
            final_out = output_base.parent / f"{input_file.stem}{config['suffix']}.{config['format']}"
    try:
        os.environ["DESCRIPTOR_INNER_PARALLEL"] = "threads"  # daemon-safety
        mols, trk = process_file_optimized_streaming_practical(input_file, final_out, config, n_jobs_per_file, worker_logger, schema_path)
        total_fail = trk.parse_errors + trk.std_errors + trk.desc_errors
        success_rate = ((trk.total - total_fail)/max(trk.total,1))*100.0
        summary = {
            'file_name': input_file.name,
            'molecules_processed': mols,
            'parse_errors': trk.parse_errors,
            'std_errors': trk.std_errors,
            'desc_errors': trk.desc_errors,
            'success_rate': success_rate,
            'recovery_attempted': trk.recovery_stats['attempted'],
            'recovery_recovered': trk.recovery_stats['recovered'],
            'total_descriptors': _descriptor_count_from_schema()
        }
        return input_file.name, mols, trk, (final_out if config.get('combine_output') else None), summary
    except Exception as e:
        worker_logger.error(f"Worker failed: {input_file.name} - {e}")
        traceback.print_exc()
        summary={'file_name':input_file.name,'molecules_processed':0,'parse_errors':0,'std_errors':0,'desc_errors':0,'success_rate':0.0,
                 'recovery_attempted':0,'recovery_recovered':0,'total_descriptors':0,'error':str(e)}
        return input_file.name, 0, SimpleErrorTracker(), None, summary

if __name__ == "__main__":
    sys.exit(main())
