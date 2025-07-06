#!/usr/bin/env python3
"""
Merge multiple JSONL files by randomly mixing their content lines.

This script takes multiple JSONL files as input and creates a single merged file
with all lines randomly shuffled. This is useful for creating diversified training
data for model fine-tuning.

Usage:
    python merge_training_jsonl_files.py file1.jsonl file2.jsonl [file3.jsonl ...] [-o output.jsonl]
    python merge_training_jsonl_files.py *.jsonl -o merged_training_data.jsonl

Author: Generated for biomed_querygenius project
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import List, Dict, Any


def load_jsonl_file(file_path: Path) -> List[Dict[str, Any]]:
    """
    Load a JSONL file and return a list of JSON objects.
    
    Args:
        file_path: Path to the JSONL file
        
    Returns:
        List of dictionaries representing JSON lines
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If a line contains invalid JSON
    """
    lines = []
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:  # Skip empty lines
                continue
            
            try:
                json_obj = json.loads(line)
                lines.append(json_obj)
            except json.JSONDecodeError as e:
                print(f"Warning: Invalid JSON on line {line_num} in {file_path}: {e}", file=sys.stderr)
                continue
    
    return lines


def save_jsonl_file(data: List[Dict[str, Any]], file_path: Path) -> None:
    """
    Save a list of JSON objects to a JSONL file.
    
    Args:
        data: List of dictionaries to save
        file_path: Path where to save the JSONL file
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')


def merge_jsonl_files(input_files: List[Path], output_file: Path, seed: int = None) -> None:
    """
    Merge multiple JSONL files by randomly mixing their content.
    
    Args:
        input_files: List of input JSONL file paths
        output_file: Path for the output merged file
        seed: Random seed for reproducible shuffling (optional)
    """
    if seed is not None:
        random.seed(seed)
    
    all_lines = []
    file_stats = {}
    
    # Load all files
    for file_path in input_files:
        print(f"Loading {file_path}...")
        try:
            lines = load_jsonl_file(file_path)
            all_lines.extend(lines)
            file_stats[str(file_path)] = len(lines)
            print(f"  Loaded {len(lines)} lines from {file_path}")
        except Exception as e:
            print(f"Error loading {file_path}: {e}", file=sys.stderr)
            continue
    
    if not all_lines:
        print("No valid lines found in any input files!", file=sys.stderr)
        sys.exit(1)
    
    # Randomly shuffle all lines
    print(f"\nShuffling {len(all_lines)} total lines...")
    random.shuffle(all_lines)
    
    # Save merged file
    print(f"Saving merged data to {output_file}...")
    save_jsonl_file(all_lines, output_file)
    
    # Print summary
    print(f"\nMerge completed successfully!")
    print(f"Total lines merged: {len(all_lines)}")
    print(f"Output file: {output_file}")
    print("\nInput file statistics:")
    for file_path, count in file_stats.items():
        percentage = (count / len(all_lines)) * 100
        print(f"  {file_path}: {count} lines ({percentage:.1f}%)")


def main():
    """Main function to handle command line arguments and execute the merge."""
    parser = argparse.ArgumentParser(
        description="Merge multiple JSONL files by randomly mixing their content lines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s file1.jsonl file2.jsonl file3.jsonl
  %(prog)s *.jsonl -o training_data.jsonl
  %(prog)s data/*.jsonl synthetic_data/*.jsonl -o merged.jsonl --seed 42
        """
    )
    
    parser.add_argument(
        'input_files',
        nargs='+',
        type=Path,
        help='Input JSONL files to merge'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=Path,
        default=Path('merged.jsonl'),
        help='Output file name (default: merged.jsonl)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        help='Random seed for reproducible shuffling'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be merged without actually creating the output file'
    )
    
    args = parser.parse_args()
    
    # Validate input files
    valid_files = []
    for file_path in args.input_files:
        if not file_path.exists():
            print(f"Warning: File not found: {file_path}", file=sys.stderr)
            continue
        if not file_path.is_file():
            print(f"Warning: Not a file: {file_path}", file=sys.stderr)
            continue
        valid_files.append(file_path)
    
    if not valid_files:
        print("Error: No valid input files found!", file=sys.stderr)
        sys.exit(1)
    
    # Check if output file already exists
    if args.output.exists() and not args.dry_run:
        response = input(f"Output file {args.output} already exists. Overwrite? (y/N): ")
        if response.lower() not in ['y', 'yes']:
            print("Operation cancelled.")
            sys.exit(0)
    
    if args.dry_run:
        print("DRY RUN - No files will be created")
        print(f"Would merge {len(valid_files)} files:")
        for file_path in valid_files:
            print(f"  {file_path}")
        print(f"Output would be saved to: {args.output}")
        return
    
    # Perform the merge
    try:
        merge_jsonl_files(valid_files, args.output, args.seed)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error during merge: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
