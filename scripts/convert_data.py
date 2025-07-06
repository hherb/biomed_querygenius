#!/usr/bin/env python3
"""
Convenience script for converting between JSONL and text formats.

This script provides a simple interface to convert JSONL files to human-readable
text format for editing, and convert them back to JSONL format.
"""

import argparse
import sys
from pathlib import Path
import subprocess


def run_conversion(script_name: str, input_file: Path, output_file: Path, extra_args: list = None) -> None:
    """
    Run a conversion script with the given arguments.
    
    Args:
        script_name: Name of the script to run
        input_file: Input file path
        output_file: Output file path
        extra_args: Additional arguments to pass to the script
    """
    script_path = Path(__file__).parent / script_name
    
    cmd = [sys.executable, str(script_path), str(input_file), str(output_file)]
    if extra_args:
        cmd.extend(extra_args)
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}: {e}")
        if e.stdout:
            print(e.stdout)
        if e.stderr:
            print(e.stderr, file=sys.stderr)
        sys.exit(1)


def main():
    """Main function to handle command line arguments and execute conversions."""
    parser = argparse.ArgumentParser(
        description="Convert between JSONL and human-readable text formats",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert JSONL to text for editing
  python convert_data.py to-text synthetic_data/synthetic_deepseekR1.jsonl data_for_editing.txt
  
  # Convert edited text back to JSONL
  python convert_data.py to-jsonl data_for_editing.txt synthetic_data/synthetic_deepseekR1_edited.jsonl
  
  # Convert back to JSONL with custom prompt prefix
  python convert_data.py to-jsonl data_for_editing.txt output.jsonl --prompt-prefix "Custom prefix:"
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Conversion direction')
    
    # Subcommand for JSONL to text
    to_text_parser = subparsers.add_parser(
        'to-text',
        help='Convert JSONL file to human-readable text format'
    )
    to_text_parser.add_argument('input_jsonl', type=Path, help='Input JSONL file')
    to_text_parser.add_argument('output_text', type=Path, help='Output text file')
    
    # Subcommand for text to JSONL
    to_jsonl_parser = subparsers.add_parser(
        'to-jsonl',
        help='Convert text file back to JSONL format'
    )
    to_jsonl_parser.add_argument('input_text', type=Path, help='Input text file')
    to_jsonl_parser.add_argument('output_jsonl', type=Path, help='Output JSONL file')
    to_jsonl_parser.add_argument(
        '--prompt-prefix',
        type=str,
        default="Convert this medical query to a PostgreSQL tsquery expression:",
        help='Prefix to add before each query in the prompt field'
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    if args.command == 'to-text':
        run_conversion('jsonl_to_text.py', args.input_jsonl, args.output_text)
    elif args.command == 'to-jsonl':
        extra_args = ['--prompt-prefix', args.prompt_prefix] if args.prompt_prefix else []
        run_conversion('text_to_jsonl.py', args.input_text, args.output_jsonl, extra_args)


if __name__ == "__main__":
    main()
