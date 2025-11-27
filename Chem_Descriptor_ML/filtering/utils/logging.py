"""ChemDescriptorML (CDML) - Logging Utilities"""
import sys
from datetime import datetime
from typing import Optional

def log(message: str, verbose: bool = True, file: Optional[str] = None):
    if not verbose:
        return
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_msg = f"[{timestamp}] {message}"
    print(formatted_msg, flush=True)
    if file:
        try:
            with open(file, 'a') as f:
                f.write(formatted_msg + '\n')
        except Exception as e:
            print(f"Warning: Could not write to log file: {e}", file=sys.stderr)
