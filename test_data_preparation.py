#!/usr/bin/env python3
"""
Test script to verify data preparation logic without MLX dependencies.
"""

import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_jsonl_data(jsonl_file: str) -> list:
    """Load data from JSONL file with prompt/completion format."""
    data = []
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                data.append(item)
            except json.JSONDecodeError as e:
                logger.warning(f"Line {line_num}: Invalid JSON - {e}")
                continue
    
    logger.info(f"Loaded {len(data)} examples from {jsonl_file}")
    return data

def extract_query_from_prompt(prompt: str) -> str:
    """Extract the natural language query from a training prompt."""
    # Look for the pattern after the instruction
    query_start = prompt.find("Convert this medical query to a PostgreSQL tsquery expression:\n")
    if query_start != -1:
        query_start += len("Convert this medical query to a PostgreSQL tsquery expression:\n")
        # Extract everything after this point, removing any trailing assistant tokens
        query = prompt[query_start:].strip()
        
        # Remove common trailing tokens
        for token in ["<|assistant|>", "<|end|>", "<|endoftext|>"]:
            if query.endswith(token):
                query = query[:-len(token)].strip()
        
        return query
    else:
        # Fallback: assume the whole prompt is the query
        return prompt.strip()

def prepare_dataset_test(jsonl_file: str = None, output_dir: str = "./data"):
    """Test the dataset preparation logic."""
    Path(output_dir).mkdir(exist_ok=True)
    
    # Determine input file
    if jsonl_file:
        logger.info(f"Loading data from JSONL file: {jsonl_file}")
        raw_data = load_jsonl_data(jsonl_file)
    else:
        # Try default locations
        default_jsonl = "./human_verified_training_data/training_data.jsonl"
        
        if Path(default_jsonl).exists():
            logger.info(f"Using default JSONL file: {default_jsonl}")
            raw_data = load_jsonl_data(default_jsonl)
        else:
            raise FileNotFoundError(f"No training data found at {default_jsonl}")
    
    # Process data into MLX-LM format
    processed_data = []
    for item in raw_data:
        # Handle multiple formats: phi3 text format, prompt/completion format, and input/output format
        if 'text' in item:
            # Phi3 format with full conversation in 'text' key - use directly
            processed_data.append(item)
        elif 'prompt' in item and 'completion' in item:
            # JSONL format - convert to phi3 format
            natural_query = extract_query_from_prompt(item['prompt'])
            tsquery = item['completion']
            # Create phi3 format
            phi3_text = f"<|user|>\nConvert this medical query to a PostgreSQL tsquery expression:\n{natural_query} <|end|>\n<|assistant|> \n{tsquery} <|end|>"
            processed_data.append({"text": phi3_text})
        elif 'input' in item and 'output' in item:
            # JSON format - convert to phi3 format
            natural_query = item['input']
            tsquery = item['output']
            # Create phi3 format
            phi3_text = f"<|user|>\nConvert this medical query to a PostgreSQL tsquery expression:\n{natural_query} <|end|>\n<|assistant|> \n{tsquery} <|end|>"
            processed_data.append({"text": phi3_text})
        else:
            logger.warning(f"Skipping item with unknown format: {list(item.keys())}")
            continue
    
    # Split dataset (80/20 train/val)
    split_idx = int(0.8 * len(processed_data))
    train_data = processed_data[:split_idx]
    val_data = processed_data[split_idx:]
    
    # Save splits
    train_file = f"{output_dir}/train.jsonl"
    val_file = f"{output_dir}/valid.jsonl"
    
    with open(train_file, 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item) + '\n')
    
    with open(val_file, 'w', encoding='utf-8') as f:
        for item in val_data:
            f.write(json.dumps(item) + '\n')
    
    logger.info(f"Training data: {len(train_data)} examples -> {train_file}")
    logger.info(f"Validation data: {len(val_data)} examples -> {val_file}")
    
    # Show sample of processed data
    logger.info("\nSample of processed training data:")
    for i, item in enumerate(train_data[:2]):
        logger.info(f"Example {i+1}:")
        if 'text' in item:
            logger.info(f"  Text: {item['text'][:150]}...")
        else:
            logger.info(f"  Keys: {list(item.keys())}")
    
    return train_file, val_file

if __name__ == "__main__":
    try:
        train_file, val_file = prepare_dataset_test()
        logger.info("Dataset preparation test completed successfully!")
        
        # Verify the files are not empty
        with open(train_file, 'r') as f:
            train_lines = len(f.readlines())
        with open(val_file, 'r') as f:
            val_lines = len(f.readlines())
            
        logger.info(f"Final verification:")
        logger.info(f"  Train file: {train_lines} lines")
        logger.info(f"  Validation file: {val_lines} lines")
        
        if train_lines > 0 and val_lines > 0:
            logger.info("✅ SUCCESS: Both files contain data!")
        else:
            logger.error("❌ FAILED: One or both files are empty!")
            
    except Exception as e:
        logger.error(f"Dataset preparation failed: {e}")
        raise
