#!/usr/bin/env python3
"""
Unit tests for convert2phi3mlxformat.py

This module tests the conversion functionality to ensure it works correctly
with various input formats and edge cases.
"""

import unittest
import tempfile
import json
from pathlib import Path
import sys

# Add the scripts directory to the path so we can import the module
sys.path.insert(0, str(Path(__file__).parent))

from convert2phi3mlxformat import extract_query_from_prompt, convert_to_phi3_format, convert_jsonl_file


class TestConvert2Phi3MLXFormat(unittest.TestCase):
    """Test cases for the phi3 MLX format conversion functions."""

    def test_extract_query_from_prompt(self):
        """Test query extraction from various prompt formats."""
        # Standard medical query format
        prompt1 = "Convert this medical query to a PostgreSQL tsquery expression:\nEffect of aspirin"
        expected1 = "Effect of aspirin"
        self.assertEqual(extract_query_from_prompt(prompt1), expected1)
        
        # Alternative format
        prompt2 = "Convert this query to a PostgreSQL tsquery expression:\nMRI vs CT"
        expected2 = "MRI vs CT"
        self.assertEqual(extract_query_from_prompt(prompt2), expected2)
        
        # Format with newline
        prompt3 = "Convert this medical query to a PostgreSQL tsquery expression:\n\nMultiline query"
        expected3 = "Multiline query"
        self.assertEqual(extract_query_from_prompt(prompt3), expected3)
        
        # No matching prefix
        prompt4 = "Different prefix: Some query"
        expected4 = "Different prefix: Some query"
        self.assertEqual(extract_query_from_prompt(prompt4), expected4)
        
        # Empty prompt
        prompt5 = ""
        expected5 = ""
        self.assertEqual(extract_query_from_prompt(prompt5), expected5)

    def test_convert_to_phi3_format(self):
        """Test conversion to phi3 MLX format."""
        prompt = "Convert this medical query to a PostgreSQL tsquery expression:\nTest query"
        completion = "test & query"

        result = convert_to_phi3_format(prompt, completion)
        expected = "<|user|>\nConvert this medical query to a PostgreSQL tsquery expression:\nTest query <|end|>\n<|assistant|> \ntest & query <|end|>"

        self.assertEqual(result, expected)

    def test_convert_to_phi3_format_edge_cases(self):
        """Test conversion with edge cases."""
        # Empty query
        prompt1 = "Convert this medical query to a PostgreSQL tsquery expression:\n"
        completion1 = "empty & query"
        result1 = convert_to_phi3_format(prompt1, completion1)
        expected1 = "<|user|>\nConvert this medical query to a PostgreSQL tsquery expression:\n <|end|>\n<|assistant|> \nempty & query <|end|>"
        self.assertEqual(result1, expected1)

        # Empty completion
        prompt2 = "Convert this medical query to a PostgreSQL tsquery expression:\nSome query"
        completion2 = ""
        result2 = convert_to_phi3_format(prompt2, completion2)
        expected2 = "<|user|>\nConvert this medical query to a PostgreSQL tsquery expression:\nSome query <|end|>\n<|assistant|> \n <|end|>"
        self.assertEqual(result2, expected2)

    def test_convert_jsonl_file(self):
        """Test full JSONL file conversion."""
        # Create test data
        test_data = [
            {"prompt": "Convert this medical query to a PostgreSQL tsquery expression:\nTest 1", "completion": "test1"},
            {"prompt": "Convert this query to a PostgreSQL tsquery expression:\nTest 2", "completion": "test2"},
            {"invalid": "data"},  # This should be skipped
            {"prompt": "Test 3", "completion": "test3"}
        ]
        
        # Create temporary input file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as input_file:
            for item in test_data:
                input_file.write(json.dumps(item) + '\n')
            input_file_path = Path(input_file.name)
        
        # Create temporary output file
        with tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False) as output_file:
            output_file_path = Path(output_file.name)
        
        try:
            # Convert the file
            convert_jsonl_file(input_file_path, output_file_path)
            
            # Read and verify the output
            with open(output_file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Should have 3 valid conversions (skipping the invalid one)
            self.assertEqual(len(lines), 3)
            
            # Check first conversion
            first_item = json.loads(lines[0])
            self.assertIn("text", first_item)
            self.assertIn("<|user|>\nConvert this medical query to a PostgreSQL tsquery expression:\nTest 1 <|end|>", first_item["text"])
            self.assertIn("<|assistant|> \ntest1 <|end|>", first_item["text"])

            # Check second conversion
            second_item = json.loads(lines[1])
            self.assertIn("<|user|>\nConvert this query to a PostgreSQL tsquery expression:\nTest 2 <|end|>", second_item["text"])

            # Check third conversion (no prefix to remove, full prompt preserved)
            third_item = json.loads(lines[2])
            self.assertIn("<|user|>\nTest 3 <|end|>", third_item["text"])
            
        finally:
            # Clean up temporary files
            input_file_path.unlink()
            output_file_path.unlink()


if __name__ == '__main__':
    unittest.main()
