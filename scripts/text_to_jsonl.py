#!/usr/bin/env python3
"""
Convert human-edited text file back to JSONL format.

This script converts a text file (created by jsonl_to_text.py) back to the
original JSONL format. It expects the text file to have the format:
- Line 1: Query text
- Line 2: Completion text  
- Line 3: Blank line (separator)
- Repeat...
"""

import json
import argparse
import sys
from pathlib import Path
from typing import List, Tuple


def parse_text_file(text_file: Path) -> List[Tuple[str, str]]:
    """
    Parse the text file and extract query-completion pairs.
    
    Args:
        text_file: Path to the text file
        
    Returns:
        List of (query, completion) tuples
    """
    pairs = []
    
    try:
        with open(text_file, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        
        i = 0
        entry_num = 1
        
        while i < len(lines):
            # Skip empty lines at the beginning
            while i < len(lines) and lines[i].strip() == '':
                i += 1
            
            if i >= len(lines):
                break
            
            # Read query (first line)
            if i < len(lines):
                query = lines[i].strip()
                i += 1
            else:
                break
            
            # Read completion (second line)
            if i < len(lines):
                completion = lines[i].strip()
                i += 1
            else:
                print(f"Warning: Entry {entry_num} has query but no completion, skipping")
                break
            
            # Skip the blank line separator (third line)
            if i < len(lines) and lines[i].strip() == '':
                i += 1
            
            # Validate that we have both query and completion
            if query and completion:
                pairs.append((query, completion))
                entry_num += 1
            else:
                print(f"Warning: Entry {entry_num} has empty query or completion, skipping")
                entry_num += 1
        
        return pairs
        
    except FileNotFoundError:
        print(f"Error: Text file '{text_file}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading text file: {e}")
        sys.exit(1)


def convert_text_to_jsonl(text_file: Path, output_file: Path, prompt_prefix: str = None) -> None:
    """
    Convert text file back to JSONL format.
    
    Args:
        text_file: Path to the input text file
        output_file: Path to the output JSONL file
        prompt_prefix: Prefix to add before the query in the prompt field
    """
    # Default prompt prefix based on the original format
    if prompt_prefix is None:
        prompt_prefix = "Convert this medical query to a PostgreSQL tsquery expression:"
    
    pairs = parse_text_file(text_file)
    
    if not pairs:
        print("No valid query-completion pairs found in the text file")
        sys.exit(1)
    
    try:
        with open(output_file, 'w', encoding='utf-8') as outfile:
            for query, completion in pairs:
                # Reconstruct the original prompt format
                full_prompt = f"{prompt_prefix}\n{query}"
                
                # Create the JSON object
                json_obj = {
                    "prompt": full_prompt,
                    "completion": completion
                }
                
                # Write as JSONL (one JSON object per line)
                outfile.write(json.dumps(json_obj, ensure_ascii=False) + '\n')
        
        print(f"Successfully converted {len(pairs)} entries from {text_file} to {output_file}")
        
    except Exception as e:
        print(f"Error writing JSONL file: {e}")
        sys.exit(1)


def main():
    """Main function to handle command line arguments and execute conversion."""
    parser = argparse.ArgumentParser(
        description="Convert human-edited text file back to JSONL format"
    )
    parser.add_argument(
        'input_file',
        type=Path,
        help='Input text file path'
    )
    parser.add_argument(
        'output_file',
        type=Path,
        help='Output JSONL file path'
    )
    parser.add_argument(
        '--prompt-prefix',
        type=str,
        default="Convert this medical query to a PostgreSQL tsquery expression:",
        help='Prefix to add before each query in the prompt field (default: "Convert this medical query to a PostgreSQL tsquery expression:")'
    )
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not args.input_file.exists():
        print(f"Error: Input file '{args.input_file}' does not exist")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    
    convert_text_to_jsonl(args.input_file, args.output_file, args.prompt_prefix)


if __name__ == "__main__":
    main()
