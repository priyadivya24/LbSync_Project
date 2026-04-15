import pandas as pd
import pyarrow.parquet as pq
import os
from pathlib import Path
from datetime import datetime
import pyarrow as pa

def get_doocs_properties(base_path: Path):
    """
    Scans the base_path for folders containing .parquet files and returns a mapping.
    Key: Absolute path string
    Value: Human-readable label (str)
    """
    properties = {}
    if not base_path.exists():
        print(f"Directory not found: {base_path}")
        return properties
        
    for root, dirs, files in os.walk(base_path):
        if any(f.endswith('.parquet') for f in files):
            # Create a label from the relative path
            try:
                rel_path = os.path.relpath(root, base_path)
                # Example: LASER.LOCK.XLO\XHEXP1.SLO1\CTRL0.OUT.MEAN.RD
                label = "XFEL.SYNC/" + rel_path.replace(os.sep, '/')
                properties[str(Path(root).absolute())] = label
            except Exception as e:
                print(f"Error processing {root}: {e}")
                
    return properties

def load_parquet_data(paths: list, start_date: datetime, end_date: datetime):
    """
    Loads parquet data from the given paths within the specified date range.
    Returns: Dict[str, pa.Table] where keys are the path strings.
    """
    results = {}
    for p_val in paths:
        p = Path(p_val)
        # Expected filename format: YYYY-MM.parquet
        all_files = list(p.glob("*.parquet"))
        relevant_files = []
        
        for f in all_files:
            try:
                # Basic filename filtering: 2023-10.parquet
                file_date_str = f.stem # e.g. 2023-10
                file_date = datetime.strptime(file_date_str, "%Y-%m")
                
                # Check if file month overlaps with [start_date, end_date]
                if (file_date.year == start_date.year and file_date.month >= start_date.month) or \
                   (file_date.year == end_date.year and file_date.month <= end_date.month) or \
                   (start_date.year < file_date.year < end_date.year):
                    relevant_files.append(f)
            except ValueError:
                # If filename doesn't match format, just include it to be safe
                relevant_files.append(f)
        
        if not relevant_files:
            continue
            
        # Load and concatenate
        tables = []
        for f in relevant_files:
            try:
                table = pq.read_table(f)
                tables.append(table)
            except Exception as e:
                print(f"Error reading {f}: {e}")
                
        if tables:
            results[str(p.absolute())] = pa.concat_tables(tables)
            
    return results
