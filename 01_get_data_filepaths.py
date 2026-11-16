"""
Helper script to collect CSV file paths from a source directory.
The resulting list is consumed by 02_run_data_pipeline.py.

Usage:
    python 01_get_data_filepaths.py --source_dir ../path/to/your/data
"""

import argparse
import os

parser = argparse.ArgumentParser(description='Collect CSV file paths from a source directory.')
parser.add_argument('--source_dir', type=str, required=True, help='Path to the source directory')
parser.add_argument('--output', type=str, default='data/data_source_files.txt',
                    help='File to write paths to (default: data/data_source_files.txt)')
args = parser.parse_args()

source_files = []
for dirpath, _, filenames in os.walk(args.source_dir):
    for filename in filenames:
        if filename.lower().endswith('.csv'):
            source_files.append(os.path.join(dirpath, filename))

source_files.sort()
os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
with open(args.output, 'w') as f:
    f.write('\n'.join(source_files) + ('\n' if source_files else ''))

print(f'Wrote {len(source_files)} CSV paths to {args.output}')
