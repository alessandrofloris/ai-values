#!/usr/bin/env python3
"""
Adapter script to score and project agent simulation responses onto the Inglehart-Welzel 2D space.

Usage:
  python src/adapt_agents.py path/to/simulation_responses.csv
  python src/adapt_agents.py path/to/simulation_responses.csv --out out/coords.js
  python src/adapt_agents.py path/to/simulation_responses.csv --out out/agents_coords.csv
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow importing local modules
sys.path.insert(0, str(Path(__file__).parent))
from impute import MIN_OBSERVED, map_coords
from score import compute_score

ROOT = Path(__file__).parent.parent
WEIGHTS_FILE = ROOT / 'out' / 'weights.txt'
MEANS_FILE = ROOT / 'out' / 'column_means.txt'
SDS_FILE = ROOT / 'out' / 'column_sds.txt'
CORR_FILE = ROOT / 'out' / 'item_correlations.txt'
DEFAULT_COORDS_JS = ROOT / 'out' / 'coords.js'

VARIABLES = ['F063', 'Y003', 'F120', 'G006', 'E018', 'Y002', 'A008', 'F118', 'E025', 'A165']


def score_cell(var_name: str, value) -> float:
    """Score a single cell answer using compute_score, with fallback to numeric value if already scored."""
    if pd.isna(value) or value is None:
        return np.nan
    val_str = str(value).strip()
    if not val_str:
        return np.nan

    # Try compute_score from score.py
    try:
        score = compute_score(var_name, val_str.casefold())
        if not np.isnan(score):
            return float(score)
    except Exception:
        pass

    # Fallback: if already a numeric score in the CSV
    try:
        return float(val_str)
    except ValueError:
        return np.nan


def find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Find the first matching column name in df (case-insensitive)."""
    cols_lower = {col.lower(): col for col in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None


def update_coords_js(out_file: Path, new_records: list[dict], region_to_replace: str = 'Agents'):
    """Merge new records into window.COORDS in coords.js, replacing points from the same region."""
    existing_records = []
    if out_file.exists():
        text = out_file.read_text(encoding='utf-8')
        m = re.search(r'window\.COORDS\s*=\s*(\[.*?\])\s*;', text, re.DOTALL)
        if m:
            try:
                existing_records = json.loads(m.group(1))
            except json.JSONDecodeError:
                existing_records = []

    # Replace existing points belonging to the same region (to avoid duplicates on re-runs)
    kept_records = [r for r in existing_records if r.get('region') != region_to_replace]
    combined = kept_records + new_records

    # Clean NaNs for JSON serialization
    clean = []
    for r in combined:
        clean_r = {k: v for k, v in r.items() if not (isinstance(v, float) and np.isnan(v))}
        clean.append(clean_r)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text('window.COORDS = ' + json.dumps(clean, indent=2) + ';\n', encoding='utf-8')


def process_agents_csv(input_path: Path, region_override: str = None) -> pd.DataFrame:
    """Read agents CSV, score question cells, and project into 2D Inglehart-Welzel coordinates."""
    df = pd.read_csv(input_path)

    # 1. Match the 10 variables
    var_cols = {}
    for var in VARIABLES:
        col = find_column(df, [var, var.lower()])
        if not col:
            raise ValueError(f"Required variable column '{var}' not found in {input_path}. Found: {list(df.columns)}")
        var_cols[var] = col

    # 2. Match metadata columns
    label_col = find_column(df, ['label', 'country', 'agent_label', 'name'])
    agent_col = find_column(df, ['lineage', 'agent_id', 'agent', 'id'])
    time_col = find_column(df, ['timepoint', 'time', 'release', 'step', 'round'])
    region_col = find_column(df, ['region', 'vendor'])

    # Determine visible label (country on map)
    if label_col:
        labels = df[label_col].astype(str)
    elif agent_col and time_col:
        labels = df[agent_col].astype(str) + ' (' + df[time_col].astype(str) + ')'
    elif agent_col:
        labels = df[agent_col].astype(str)
    else:
        labels = pd.Series([f"Agent {i+1}" for i in range(len(df))])

    # Determine lineage (same agent across time belongs to same lineage)
    if agent_col:
        lineages = df[agent_col].astype(str)
    else:
        lineages = labels

    # Determine release (time order)
    if time_col:
        releases = df[time_col].astype(str)
    else:
        releases = pd.Series(['T0'] * len(df))

    # Determine region (defaults to 'Agents')
    if region_override:
        regions = pd.Series([region_override] * len(df))
    elif region_col:
        regions = df[region_col].astype(str)
    else:
        regions = pd.Series(['Agents'] * len(df))

    # 3. Score all question cells
    scored_data = np.zeros((len(df), len(VARIABLES)), dtype=float)
    for j, var in enumerate(VARIABLES):
        raw_vals = df[var_cols[var]].values
        scored_data[:, j] = [score_cell(var, v) for v in raw_vals]

    # 4. Project using PCA weights and means/sds
    weights = np.loadtxt(WEIGHTS_FILE)
    means = np.loadtxt(MEANS_FILE)
    sds = np.loadtxt(SDS_FILE)
    item_corr = np.loadtxt(CORR_FILE)

    coords = map_coords(scored_data, means, sds, weights, R=item_corr, min_observed=MIN_OBSERVED)

    # 5. Assemble results DataFrame
    out_df = pd.DataFrame({
        'country': labels,
        'region': regions,
        'x': np.round(coords[:, 0], 6),
        'y': np.round(coords[:, 1], 6),
        'lineage': lineages,
        'release': releases,
    })

    # Add scored columns for reference
    for j, var in enumerate(VARIABLES):
        out_df[var] = scored_data[:, j]

    return out_df


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('input', type=Path, help='Path to simulation responses CSV')
    parser.add_argument('--out', '-o', type=Path, default=DEFAULT_COORDS_JS,
                        help=f'Output file (default: {DEFAULT_COORDS_JS})')
    parser.add_argument('--region', type=str, default='Agents',
                        help="Region/group name for the points (default: 'Agents')")
    parser.add_argument('--standalone', action='store_true',
                        help='Write ONLY these points to coords.js without merging with existing map')

    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: input file {args.input} not found.")
        sys.exit(1)

    print(f"Reading simulation responses from: {args.input}")
    res_df = process_agents_csv(args.input, region_override=args.region)

    valid_mask = res_df['x'].notna() & res_df['y'].notna()
    print(f"Successfully scored and projected {valid_mask.sum()}/{len(res_df)} agents.")

    records = res_df[['country', 'region', 'x', 'y', 'lineage', 'release']].dropna(subset=['x', 'y']).to_dict(orient='records')

    out_path = args.out
    if out_path.suffix.lower() == '.js':
        if args.standalone:
            clean = [{k: v for k, v in r.items() if not (isinstance(v, float) and np.isnan(v))} for r in records]
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text('window.COORDS = ' + json.dumps(clean, indent=2) + ';\n', encoding='utf-8')
            print(f"Exported {len(records)} standalone points to: {out_path}")
        else:
            update_coords_js(out_path, records, region_to_replace=args.region)
            print(f"Merged {len(records)} agent points into: {out_path}")
    elif out_path.suffix.lower() == '.json':
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(records, f, indent=2)
        print(f"Saved {len(records)} points to JSON: {out_path}")
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        res_df.to_csv(out_path, index=False)
        print(f"Saved coordinates table to CSV: {out_path}")


if __name__ == '__main__':
    main()

