#!/usr/bin/env python3
"""
Convert JSONL files from prompt/completion format to phi3 MLX format.

This script converts existing JSONL files with the format:
{"prompt": "Convert this medical query...", "completion": "answer"}

To the phi3 MLX format:
{"text": "<|user|>\nIs 91 a prime number? <|end|>\n<|assistant|> \nNo, 91 is not a prime number <|end|>"}

The script extracts the actual query from the prompt (removing the instruction prefix)
and formats it according to phi3's chat template.
"""

import json
import argparse
import sys
import re
from pathlib import Path
from typing import List, Dict


def extract_query_from_prompt(prompt: str) -> str:
    """
    Extract the actual query from the prompt by removing the instruction prefix.
    
    Args:
        prompt: The full prompt string
        
    Returns:
        The extracted query text
    """
    # Common instruction prefixes to remove
    prefixes = [
        "Convert this medical query to a PostgreSQL tsquery expression:",
        "Convert this medical query to a PostgreSQL tsquery expression:\n",
        "Convert this query to a PostgreSQL tsquery expression:",
        "Convert this query to a PostgreSQL tsquery expression:\n",
    ]
    
    query = prompt
    for prefix in prefixes:
        if query.startswith(prefix):
            query = query[len(prefix):].strip()
            break
    
    return query


def convert_to_phi3_format(prompt: str, completion: str) -> str:
    """
    Convert prompt/completion pair to phi3 MLX chat format.

    Args:
        prompt: The original prompt (full instruction + query)
        completion: The completion text

    Returns:
        Formatted text string for phi3 MLX training
    """
    # Use the full prompt as the user message (preserving the instruction)
    # Format according to phi3 chat template
    formatted_text = f"<|user|>\n{prompt} <|end|>\n<|assistant|> \n{completion} <|end|>"

    return formatted_text


def convert_jsonl_file(input_file: Path, output_file: Path) -> None:
    """
    Convert a JSONL file from prompt/completion format to phi3 MLX format.
    
    Args:
        input_file: Path to the input JSONL file
        output_file: Path to the output JSONL file
    """
    converted_count = 0
    skipped_count = 0
    
    try:
        with open(input_file, 'r', encoding='utf-8') as infile, \
             open(output_file, 'w', encoding='utf-8') as outfile:
            
            for line_num, line in enumerate(infile, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    # Parse the JSON line
                    item = json.loads(line)
                    
                    # Check if it has the expected format
                    if 'prompt' not in item or 'completion' not in item:
                        print(f"Warning: Line {line_num} missing 'prompt' or 'completion' keys, skipping")
                        skipped_count += 1
                        continue
                    
                    # Convert to phi3 format
                    phi3_text = convert_to_phi3_format(item['prompt'], item['completion'])
                    
                    # Create the new JSON object
                    phi3_item = {"text": phi3_text}
                    
                    # Write to output file
                    outfile.write(json.dumps(phi3_item, ensure_ascii=False) + '\n')
                    converted_count += 1
                    
                except json.JSONDecodeError as e:
                    print(f"Warning: Line {line_num} contains invalid JSON - {e}, skipping")
                    skipped_count += 1
                    continue
                except Exception as e:
                    print(f"Warning: Error processing line {line_num} - {e}, skipping")
                    skipped_count += 1
                    continue
        
        print(f"Conversion complete!")
        print(f"  Converted: {converted_count} entries")
        print(f"  Skipped: {skipped_count} entries")
        print(f"  Output file: {output_file}")
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error during conversion: {e}")
        sys.exit(1)


def preview_conversion(input_file: Path, num_examples: int = 3) -> None:
    """
    Preview the conversion by showing a few examples.
    
    Args:
        input_file: Path to the input JSONL file
        num_examples: Number of examples to show
    """
    print(f"Preview of conversion from {input_file}:")
    print("=" * 60)
    
    try:
        with open(input_file, 'r', encoding='utf-8') as infile:
            for line_num, line in enumerate(infile, 1):
                if line_num > num_examples:
                    break
                
                line = line.strip()
                if not line:
                    continue
                
                try:
                    item = json.loads(line)
                    
                    if 'prompt' not in item or 'completion' not in item:
                        continue
                    
                    print(f"\nExample {line_num}:")
                    print(f"Original prompt: {item['prompt'][:100]}...")
                    print(f"Original completion: {item['completion'][:100]}...")
                    
                    phi3_text = convert_to_phi3_format(item['prompt'], item['completion'])
                    print(f"Phi3 format: {phi3_text[:150]}...")
                    print("-" * 40)
                    
                except json.JSONDecodeError:
                    continue
                    
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found")
        sys.exit(1)


def main():
    """Main function to handle command line arguments and execute conversion."""
    parser = argparse.ArgumentParser(
        description="Convert JSONL files from prompt/completion format to phi3 MLX format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert a JSONL file
  python convert2phi3mlxformat.py input.jsonl output.jsonl
  
  # Preview conversion without creating output file
  python convert2phi3mlxformat.py input.jsonl --preview
  
  # Preview with more examples
  python convert2phi3mlxformat.py input.jsonl --preview --examples 5
        """
    )
    
    parser.add_argument(
        'input_file',
        type=Path,
        help='Input JSONL file with prompt/completion format'
    )
    parser.add_argument(
        'output_file',
        type=Path,
        nargs='?',
        help='Output JSONL file in phi3 MLX format (required unless --preview is used)'
    )
    parser.add_argument(
        '--preview',
        action='store_true',
        help='Preview the conversion without creating output file'
    )
    parser.add_argument(
        '--examples',
        type=int,
        default=3,
        help='Number of examples to show in preview mode (default: 3)'
    )
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not args.input_file.exists():
        print(f"Error: Input file '{args.input_file}' does not exist")
        sys.exit(1)
    
    if args.preview:
        # Preview mode
        preview_conversion(args.input_file, args.examples)
    else:
        # Conversion mode
        if not args.output_file:
            print("Error: Output file is required unless --preview is used")
            parser.print_help()
            sys.exit(1)
        
        # Create output directory if it doesn't exist
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        
        convert_jsonl_file(args.input_file, args.output_file)


if __name__ == "__main__":
    main()
