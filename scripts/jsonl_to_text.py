#!/usr/bin/env python3
"""
Convert JSONL file to human-readable text format for editing.

This script converts a JSONL file where each line contains a JSON object with
"prompt" and "completion" fields into a text format that's easier for humans
to edit. The prompt text after the '\n' character is saved on the first line,
the completion text on the second line, followed by a blank line.
"""

import json
import argparse
import sys
from pathlib import Path
from typing import Dict, Any


def extract_prompt_query(prompt: str) -> str:
    """
    Extract the actual query from the prompt by taking everything after the first '\n'.
    
    Args:
        prompt: The full prompt string
        
    Returns:
        The query part of the prompt (everything after the first newline)
    """
    if '\n' in prompt:
        return prompt.split('\n', 1)[1]
    return prompt


def convert_jsonl_to_text(input_file: Path, output_file: Path) -> None:
    """
    Convert JSONL file to human-readable text format.
    
    Args:
        input_file: Path to the input JSONL file
        output_file: Path to the output text file
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as infile, \
             open(output_file, 'w', encoding='utf-8') as outfile:
            
            line_count = 0
            for line_num, line in enumerate(infile, 1):
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    data = json.loads(line)
                    
                    # Validate required fields
                    if 'prompt' not in data or 'completion' not in data:
                        print(f"Warning: Line {line_num} missing required fields, skipping")
                        continue
                    
                    # Extract the query part from the prompt
                    query = extract_prompt_query(data['prompt'])
                    completion = data['completion']
                    
                    # Write in the format: query, completion, blank line
                    outfile.write(f"{query}\n")
                    outfile.write(f"{completion}\n")
                    outfile.write("\n")  # Blank line separator
                    
                    line_count += 1
                    
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON on line {line_num}: {e}")
                    continue
            
            print(f"Successfully converted {line_count} entries from {input_file} to {output_file}")
            
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def main():
    """Main function to handle command line arguments and execute conversion."""
    parser = argparse.ArgumentParser(
        description="Convert JSONL file to human-readable text format for editing"
    )
    parser.add_argument(
        'input_file',
        type=Path,
        help='Input JSONL file path'
    )
    parser.add_argument(
        'output_file',
        type=Path,
        help='Output text file path'
    )
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not args.input_file.exists():
        print(f"Error: Input file '{args.input_file}' does not exist")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    
    convert_jsonl_to_text(args.input_file, args.output_file)


if __name__ == "__main__":
    main()
