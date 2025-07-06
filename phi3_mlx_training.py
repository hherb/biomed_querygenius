#!/usr/bin/env python3
"""
Phi-3-mini Fine-tuning Pipeline for PostgreSQL tsquery Generation
Optimized for Apple Silicon using MLX-LM

This script fine-tunes Phi-3-mini to convert natural language medical queries
into PostgreSQL full-text search expressions using the MLX-LM library.
"""

import json
import logging
import time
import subprocess
import sys
from pathlib import Path
from typing import List, Dict
import argparse

import mlx.core as mx
from mlx_lm import load, generate

# Configure logging with forced flushing
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('phi3_training.log'),
        logging.StreamHandler(sys.stdout)  # Explicitly use stdout
    ],
    force=True  # Force reconfiguration if already configured
)
logger = logging.getLogger(__name__)

# Force stdout to be unbuffered for real-time output
import os
import re
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Dict, Tuple
os.environ['PYTHONUNBUFFERED'] = '1'

def flush_print(*args, **kwargs):
    """Print with forced flush for immediate output."""
    print(*args, **kwargs, flush=True)

def parse_training_metrics(output_lines: List[str]) -> Dict[str, List[float]]:
    """Parse training output to extract metrics."""
    metrics = {
        'iteration': [],
        'train_loss': [],
        'val_loss': [],
        'learning_rate': [],
        'tokens_per_sec': [],
        'peak_memory_gb': []
    }

    # Regex patterns for different metrics
    iter_pattern = r'Iter (\d+):'
    train_loss_pattern = r'Train loss ([\d.]+)'
    val_loss_pattern = r'Val loss ([\d.]+)'
    lr_pattern = r'Learning Rate ([\d.e-]+)'
    tokens_sec_pattern = r'Tokens/sec ([\d.]+)'
    memory_pattern = r'Peak mem ([\d.]+) GB'

    for line in output_lines:
        # Look for iteration lines that contain multiple metrics
        iter_match = re.search(iter_pattern, line)
        if iter_match:
            iteration = int(iter_match.group(1))

            # Extract all metrics from this line
            train_loss_match = re.search(train_loss_pattern, line)
            val_loss_match = re.search(val_loss_pattern, line)
            lr_match = re.search(lr_pattern, line)
            tokens_sec_match = re.search(tokens_sec_pattern, line)
            memory_match = re.search(memory_pattern, line)

            # Only add if we have at least train or val loss
            if train_loss_match or val_loss_match:
                metrics['iteration'].append(iteration)
                metrics['train_loss'].append(float(train_loss_match.group(1)) if train_loss_match else None)
                metrics['val_loss'].append(float(val_loss_match.group(1)) if val_loss_match else None)
                metrics['learning_rate'].append(float(lr_match.group(1)) if lr_match else None)
                metrics['tokens_per_sec'].append(float(tokens_sec_match.group(1)) if tokens_sec_match else None)
                metrics['peak_memory_gb'].append(float(memory_match.group(1)) if memory_match else None)

        # Also look for standalone validation loss lines
        elif 'Val loss' in line and 'Iter' in line:
            iter_val_match = re.search(r'Iter (\d+): Val loss ([\d.]+)', line)
            if iter_val_match:
                iteration = int(iter_val_match.group(1))
                val_loss = float(iter_val_match.group(2))

                # Check if this iteration already exists
                if iteration not in metrics['iteration']:
                    metrics['iteration'].append(iteration)
                    metrics['train_loss'].append(None)
                    metrics['val_loss'].append(val_loss)
                    metrics['learning_rate'].append(None)
                    metrics['tokens_per_sec'].append(None)
                    metrics['peak_memory_gb'].append(None)
                else:
                    # Update existing entry
                    idx = metrics['iteration'].index(iteration)
                    if metrics['val_loss'][idx] is None:
                        metrics['val_loss'][idx] = val_loss

    return metrics

def create_training_visualization(metrics: Dict[str, List[float]], output_dir: str = "./lora_adapters"):
    """Create training visualization with loss plots and metrics table."""
    if not metrics['iteration']:
        logger.warning("No training metrics found to visualize")
        return

    # Create DataFrame for easier handling
    df = pd.DataFrame(metrics)
    df = df.dropna(subset=['iteration'])  # Remove any rows without iteration

    if df.empty:
        logger.warning("No valid training data to visualize")
        return

    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Progress Visualization', fontsize=16, fontweight='bold')

    # Plot 1: Training and Validation Loss
    ax1.set_title('Training & Validation Loss', fontweight='bold')

    # Filter out None values for plotting
    train_data = df[df['train_loss'].notna()]
    val_data = df[df['val_loss'].notna()]

    if not train_data.empty:
        ax1.plot(train_data['iteration'], train_data['train_loss'],
                'b-o', label='Training Loss', linewidth=2, markersize=6)
    if not val_data.empty:
        ax1.plot(val_data['iteration'], val_data['val_loss'],
                'r-s', label='Validation Loss', linewidth=2, markersize=6)

    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Learning Rate
    lr_data = df[df['learning_rate'].notna()]
    if not lr_data.empty:
        ax2.set_title('Learning Rate', fontweight='bold')
        ax2.plot(lr_data['iteration'], lr_data['learning_rate'],
                'g-^', linewidth=2, markersize=6)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Learning Rate')
        ax2.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No Learning Rate Data', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Learning Rate', fontweight='bold')

    # Plot 3: Tokens per Second
    tokens_data = df[df['tokens_per_sec'].notna()]
    if not tokens_data.empty:
        ax3.set_title('Training Speed (Tokens/sec)', fontweight='bold')
        ax3.plot(tokens_data['iteration'], tokens_data['tokens_per_sec'],
                'purple', marker='d', linewidth=2, markersize=6)
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Tokens/sec')
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'No Speed Data', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Training Speed (Tokens/sec)', fontweight='bold')

    # Plot 4: Memory Usage
    memory_data = df[df['peak_memory_gb'].notna()]
    if not memory_data.empty:
        ax4.set_title('Peak Memory Usage (GB)', fontweight='bold')
        ax4.plot(memory_data['iteration'], memory_data['peak_memory_gb'],
                'orange', marker='h', linewidth=2, markersize=6)
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('Memory (GB)')
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'No Memory Data', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Peak Memory Usage (GB)', fontweight='bold')

    plt.tight_layout()

    # Save the plot
    plot_path = Path(output_dir) / "training_progress.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"📊 Training visualization saved to: {plot_path}")

    # Show the plot
    plt.show()

    return df

def print_training_summary_table(df: pd.DataFrame):
    """Print a nice summary table of training metrics."""
    if df.empty:
        logger.warning("No data available for summary table")
        return

    logger.info("\n" + "="*80)
    logger.info("📈 TRAINING SUMMARY TABLE")
    logger.info("="*80)

    # Create summary table
    summary_data = []
    for _, row in df.iterrows():
        summary_row = {
            'Iteration': int(row['iteration']),
            'Train Loss': f"{row['train_loss']:.3f}" if pd.notna(row['train_loss']) else "N/A",
            'Val Loss': f"{row['val_loss']:.3f}" if pd.notna(row['val_loss']) else "N/A",
            'Learning Rate': f"{row['learning_rate']:.2e}" if pd.notna(row['learning_rate']) else "N/A",
            'Tokens/sec': f"{row['tokens_per_sec']:.1f}" if pd.notna(row['tokens_per_sec']) else "N/A",
            'Memory (GB)': f"{row['peak_memory_gb']:.2f}" if pd.notna(row['peak_memory_gb']) else "N/A"
        }
        summary_data.append(summary_row)

    # Print table header
    headers = ['Iteration', 'Train Loss', 'Val Loss', 'Learning Rate', 'Tokens/sec', 'Memory (GB)']
    col_widths = [10, 12, 12, 15, 12, 12]

    # Print header
    header_line = "| " + " | ".join(f"{h:^{w}}" for h, w in zip(headers, col_widths)) + " |"
    separator_line = "|" + "|".join("-" * (w + 2) for w in col_widths) + "|"

    logger.info(header_line)
    logger.info(separator_line)

    # Print data rows
    for row in summary_data:
        data_line = "| " + " | ".join(f"{str(row[h]):^{w}}" for h, w in zip(headers, col_widths)) + " |"
        logger.info(data_line)

    logger.info(separator_line)

    # Print summary statistics
    train_losses = df['train_loss'].dropna()
    val_losses = df['val_loss'].dropna()

    logger.info("\n📊 TRAINING STATISTICS:")
    if not train_losses.empty:
        logger.info(f"   Training Loss:   Initial: {train_losses.iloc[0]:.3f} → Final: {train_losses.iloc[-1]:.3f} (Δ: {train_losses.iloc[-1] - train_losses.iloc[0]:+.3f})")
    if not val_losses.empty:
        logger.info(f"   Validation Loss: Initial: {val_losses.iloc[0]:.3f} → Final: {val_losses.iloc[-1]:.3f} (Δ: {val_losses.iloc[-1] - val_losses.iloc[0]:+.3f})")

    if not train_losses.empty and not val_losses.empty:
        final_gap = abs(train_losses.iloc[-1] - val_losses.iloc[-1])
        logger.info(f"   Final Loss Gap:  {final_gap:.3f} ({'Overfitting' if train_losses.iloc[-1] < val_losses.iloc[-1] else 'Underfitting' if final_gap > 0.5 else 'Good fit'})")

    logger.info("="*80)

class PostgresQueryDataset:
    """Dataset class for loading and processing the tsquery training data."""
    
    def __init__(self, json_file: str):
        self.data = self._load_and_process_data(json_file)
        
    def _load_and_process_data(self, json_file: str) -> List[Dict]:
        """Load and process the training data into the format expected by MLX-LM."""
        with open(json_file, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        processed_data = []
        for item in raw_data:
            # Create instruction-following format
            prompt = self._create_prompt(item['input'])
            completion = item['output']
            
            # Create the chat format expected by MLX-LM
            processed_data.append({
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a medical database search expert. Convert natural language medical queries into PostgreSQL full-text search expressions using tsquery syntax.\n\nRules:\n- Use & for AND operations\n- Use | for OR operations\n- Use quotes for exact phrases\n- Favor broader terms to avoid false negatives\n- Don't include subset terms (e.g., use 'thoracotomy' not 'emergency thoracotomy')\n- Let PostgreSQL stemming handle plurals"
                    },
                    {
                        "role": "user", 
                        "content": f"Convert this medical query to a PostgreSQL tsquery expression:\n{item['input']}"
                    },
                    {
                        "role": "assistant",
                        "content": completion
                    }
                ]
            })
        
        logger.info(f"Processed {len(processed_data)} training examples")
        return processed_data
    
    def _create_prompt(self, query: str) -> str:
        """Create a structured prompt for the model."""
        return f"""<|system|>
You are a medical database search expert. Convert natural language medical queries into PostgreSQL full-text search expressions using tsquery syntax.

Rules:
- Use & for AND operations
- Use | for OR operations  
- Use quotes for exact phrases
- Favor broader terms to avoid false negatives
- Don't include subset terms (e.g., use 'thoracotomy' not 'emergency thoracotomy')
- Let PostgreSQL stemming handle plurals

<|user|>
Convert this medical query to a PostgreSQL tsquery expression:
{query}

<|assistant|>
"""

    def save_to_jsonl(self, output_file: str):
        """Save the dataset in JSONL format for MLX-LM."""
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in self.data:
                f.write(json.dumps(item) + '\n')
        logger.info(f"Dataset saved to {output_file}")

def prepare_dataset(jsonl_file: str = None, json_file: str = None, output_dir: str = "./data"):
    """Prepare the dataset for MLX-LM training from JSONL or JSON input."""
    Path(output_dir).mkdir(exist_ok=True)
    
    # Determine input file
    if jsonl_file:
        logger.info(f"Loading data from JSONL file: {jsonl_file}")
        raw_data = load_jsonl_data(jsonl_file)
    elif json_file:
        logger.info(f"Loading data from JSON file: {json_file}")
        raw_data = load_json_data(json_file)
    else:
        # Try default locations
        default_jsonl = "./human_verified_training_data/training_data.jsonl"
        default_json = "postgres_training_data.json"
        
        if Path(default_jsonl).exists():
            logger.info(f"Using default JSONL file: {default_jsonl}")
            raw_data = load_jsonl_data(default_jsonl)
        elif Path(default_json).exists():
            logger.info(f"Using default JSON file: {default_json}")
            raw_data = load_json_data(default_json)
        else:
            raise FileNotFoundError(f"No training data found. Please provide --data-file or place data in {default_jsonl} or {default_json}")
    
    # Process data into MLX-LM format
    processed_data = []
    for item in raw_data:
        # Handle both JSONL (prompt/completion) and JSON (input/output) formats
        if 'prompt' in item and 'completion' in item:
            # JSONL format
            natural_query = extract_query_from_prompt(item['prompt'])
            tsquery = item['completion']
        elif 'input' in item and 'output' in item:
            # JSON format  
            natural_query = item['input']
            tsquery = item['output']
        else:
            logger.warning(f"Skipping item with unknown format: {list(item.keys())}")
            continue
            
        # Create MLX-LM training format
        processed_data.append({
            "prompt": f"Convert this medical query to a PostgreSQL tsquery expression:\n{natural_query}",
            "completion": tsquery
        })
    
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
    
    return train_file, val_file

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

def load_json_data(json_file: str) -> list:
    """Load data from JSON file with input/output format."""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    logger.info(f"Loaded {len(data)} examples from {json_file}")
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

def train_model_with_mlx_lm(
    model_path: str = "microsoft/Phi-3-mini-4k-instruct",
    train_file: str = "./data/train.jsonl",
    val_file: str = "./data/valid.jsonl",
    output_dir: str = "./lora_adapters",
    num_iters: int = 1000,
    learning_rate: float = 1e-5,
    batch_size: int = 4,
    num_layers: int = 16,
):
    """Train the model using MLX-LM command line interface."""

    # MLX-LM expects a directory containing train.jsonl, valid.jsonl, test.jsonl
    # Extract the directory from the train_file path
    data_dir = str(Path(train_file).parent)

    # Prepare the command - using correct MLX-LM syntax
    cmd = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--model", model_path,
        "--train",
        "--data", data_dir,  # Pass directory, not file
        "--iters", str(num_iters),
        "--learning-rate", str(learning_rate),
        "--batch-size", str(batch_size),
        "--num-layers", str(num_layers),
        "--adapter-path", output_dir,
        "--fine-tune-type", "lora",
        "--steps-per-report", "10",
        "--steps-per-eval", "100",
        "--save-every", "200",
    ]
    
    logger.info("Starting training with MLX-LM...")
    logger.info(f"Command: {' '.join(cmd)}")
    logger.info(f"Model: {model_path}")
    logger.info(f"Training file: {train_file}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Training iterations: {num_iters}")
    logger.info(f"Learning rate: {learning_rate}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"LoRA layers: {num_layers}")
    flush_print("\n🚀 Training starting... Progress will be shown below:")
    flush_print("=" * 80)
    
    try:
        # Run the training command with real-time output
        logger.info("Starting training process...")
        logger.info("=" * 60)

        # Use Popen for real-time output streaming
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Merge stderr into stdout
            text=True,
            bufsize=1,  # Line buffered
            universal_newlines=True
        )

        # Stream output in real-time
        output_lines = []
        if process.stdout:
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    # Print directly to console for immediate feedback
                    line = output.strip()
                    flush_print(line)
                    output_lines.append(line)

                    # Log important progress indicators
                    if any(keyword in line.lower() for keyword in ['iter:', 'loss:', 'eval:', 'saving']):
                        logger.info(line)

        # Wait for process to complete and get return code
        return_code = process.poll()

        if return_code == 0:
            logger.info("=" * 60)
            logger.info("Training completed successfully!")

            # Parse training metrics and create visualization
            logger.info("📊 Creating training visualization...")
            metrics = parse_training_metrics(output_lines)

            if metrics['iteration']:
                # Create and display visualization
                df = create_training_visualization(metrics, output_dir)
                if df is not None and not df.empty:
                    print_training_summary_table(df)
                else:
                    logger.warning("No training data available for visualization")
            else:
                logger.warning("No training metrics found in output")
        else:
            logger.error(f"Training failed with return code {return_code}")
            if return_code is not None:
                raise subprocess.CalledProcessError(return_code, cmd)
            else:
                raise RuntimeError("Training process failed with unknown return code")

    except subprocess.CalledProcessError as e:
        logger.error(f"Training failed with return code {e.returncode}")
        raise
    except FileNotFoundError:
        logger.error("MLX-LM not found. Make sure it's installed: pip install mlx-lm")
        raise
    
    return output_dir

class Phi3QueryGenerator:
    """Inference class for generating PostgreSQL queries with the fine-tuned model."""
    
    def __init__(self, model_path: str = "microsoft/Phi-3-mini-4k-instruct", adapter_path: str = "./lora_adapters"):
        """
        Initialize the query generator.
        
        Args:
            model_path: Path to the base model
            adapter_path: Path to the LoRA adapters
        """
        self.model_path = model_path
        self.adapter_path = adapter_path
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """Load the model with LoRA adapters."""
        logger.info(f"Loading model from {self.model_path}")
        
        if Path(self.adapter_path).exists():
            logger.info(f"Loading LoRA adapters from {self.adapter_path}")
            # Load model with adapters
            self.model, self.tokenizer = load(
                self.model_path,
                adapter_path=self.adapter_path
            )
        else:
            logger.info("No LoRA adapters found, loading base model only")
            # Load base model only
            self.model, self.tokenizer = load(self.model_path)
        
        logger.info("Model loaded successfully!")
    
    def _create_prompt(self, query: str) -> str:
        """Create a structured prompt for the model."""
        return f"""<|system|>
You are a medical database search expert. Convert natural language medical queries into PostgreSQL full-text search expressions using tsquery syntax.

Rules:
- Use & for AND operations
- Use | for OR operations  
- Use quotes for exact phrases
- Favor broader terms to avoid false negatives
- Don't include subset terms (e.g., use 'thoracotomy' not 'emergency thoracotomy')
- Let PostgreSQL stemming handle plurals

<|user|>
Convert this medical query to a PostgreSQL tsquery expression:
{query}

<|assistant|>
"""
    
    def generate_query(self, input_text: str, max_tokens: int = 100, temperature: float = 0.1) -> str:
        """
        Generate a PostgreSQL tsquery expression from natural language.
        
        Args:
            input_text: Natural language medical query
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (lower = more deterministic)
            
        Returns:
            Generated tsquery expression
        """
        prompt = self._create_prompt(input_text)
        
        # Generate response (MLX-LM returns only the generated part)
        try:
            response = generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=max_tokens
            )
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return f"Error: {e}"
        
        # MLX-LM returns just the generated part, no need to remove prompt
        generated_text = response.strip()
        
        # Clean up the response (remove special tokens and extra text)
        # Remove <|end|> and other special tokens
        generated_text = generated_text.replace('<|end|>', '').replace('<|endoftext|>', '')
        
        # If it starts with "tsquery", remove that prefix
        if generated_text.startswith('tsquery '):
            generated_text = generated_text[8:]  # Remove "tsquery "
        
        # Take only the first line if there are multiple lines
        lines = generated_text.split('\n')
        if lines:
            result = lines[0].strip()
        else:
            result = generated_text.strip()
        
        return result
    
    def batch_generate(self, queries: list, **kwargs) -> list:
        """Generate tsquery expressions for multiple queries."""
        results = []
        for query in queries:
            result = self.generate_query(query, **kwargs)
            results.append(result)
        return results

def test_model(adapter_path: str = "./lora_adapters", model_path: str = "microsoft/Phi-3-mini-4k-instruct"):
    """Test the trained model with sample queries."""
    logger.info("Testing the trained model...")
    
    # Initialize generator
    generator = Phi3QueryGenerator(model_path, adapter_path)
    
    # Test queries
    test_queries = [
        "What is the sensitivity of troponin for myocardial infarction?",
        "What are the contraindications for lumbar puncture?",
        "What is the treatment for septic shock?",
        "What are the risk factors for pulmonary embolism?",
        "What is the Glasgow Coma Scale scoring system?",
    ]
    
    logger.info("Sample generations:")
    logger.info("=" * 80)
    
    for i, query in enumerate(test_queries, 1):
        logger.info(f"\nTest {i}:")
        logger.info(f"Input:  {query}")
        
        # Generate tsquery
        try:
            generated_query = generator.generate_query(query)
            logger.info(f"Output: {generated_query}")
            
            # Basic validation
            if any(char in generated_query for char in ['&', '|']):
                logger.info("✓ Contains logical operators")
            else:
                logger.info("⚠ No logical operators detected")
                
        except Exception as e:
            logger.error(f"✗ Generation failed: {e}")
        
        logger.info("-" * 60)

def check_mlx_lm_installation():
    """Check if MLX-LM is properly installed and has the lora module."""
    try:
        result = subprocess.run([sys.executable, "-m", "mlx_lm", "lora", "--help"], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            logger.info("MLX-LM LoRA training module is available")
            return True
        else:
            logger.error("MLX-LM LoRA module not working properly")
            return False
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        logger.error("MLX-LM LoRA module not found")
        logger.info("Try: pip install --upgrade mlx-lm")
        return False

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Phi-3-mini for PostgreSQL query generation")
    parser.add_argument("--data-file", type=str, help="Path to training data file (JSONL or JSON format)")
    parser.add_argument("--data-dir", type=str, default="./human_verified_training_data", help="Directory containing training data")
    parser.add_argument("--model_path", type=str, default="microsoft/Phi-3-mini-4k-instruct", help="Path to base model")
    parser.add_argument("--num_iters", type=int, default=1000, help="Number of training iterations")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--num_layers", type=int, default=16, help="Number of LoRA layers")
    parser.add_argument("--output_dir", type=str, default="./lora_adapters", help="Output directory for LoRA adapters")
    parser.add_argument("--test_only", action="store_true", help="Only run testing (skip training)")
    parser.add_argument("--prepare_only", action="store_true", help="Only prepare dataset (skip training)")
    parser.add_argument("--check_install", action="store_true", help="Check MLX-LM installation")
    
    args = parser.parse_args()
    
    try:
        if args.check_install:
            # Check MLX-LM installation
            check_mlx_lm_installation()
            return
        
        if args.prepare_only:
            # Only prepare dataset
            logger.info("Preparing dataset only...")
            
            # Determine input file
            if args.data_file:
                input_file = args.data_file
            else:
                # Look in data directory for default files
                data_dir = Path(args.data_dir)
                if not data_dir.exists():
                    logger.info(f"Creating data directory: {data_dir}")
                    data_dir.mkdir(parents=True, exist_ok=True)
                    
                jsonl_candidates = list(data_dir.glob("*.jsonl"))
                json_candidates = list(data_dir.glob("*.json"))
                
                if jsonl_candidates:
                    input_file = str(jsonl_candidates[0])
                    logger.info(f"Found JSONL file: {input_file}")
                elif json_candidates:
                    input_file = str(json_candidates[0])
                    logger.info(f"Found JSON file: {input_file}")
                else:
                    input_file = None
                    logger.warning(f"No training data found in {data_dir}")
                    logger.info("Please:")
                    logger.info(f"1. Place your JSONL file in {data_dir}/")
                    logger.info(f"2. Or specify --data-file path/to/your/data.jsonl")
                    logger.info(f"3. Or run: bash setup_jsonl_workflow.sh")
                    return
            
            if input_file and input_file.endswith('.jsonl'):
                train_file, val_file = prepare_dataset(jsonl_file=input_file)
            elif input_file and input_file.endswith('.json'):
                train_file, val_file = prepare_dataset(json_file=input_file)
            else:
                train_file, val_file = prepare_dataset()  # Use defaults
                
            logger.info("Dataset preparation completed!")
            logger.info(f"Next step: python {__file__} --data-file {input_file if input_file else 'your_data_file'}")
            return
        
        if args.test_only:
            # Only run testing
            test_model(args.output_dir, args.model_path)
            return
        
        # Check MLX-LM installation first
        if not check_mlx_lm_installation():
            logger.error("Please fix MLX-LM installation before training")
            return
        
        # Prepare dataset
        logger.info("Preparing dataset...")
        
        # Determine input file
        if args.data_file:
            input_file = args.data_file
            if input_file.endswith('.jsonl'):
                train_file, val_file = prepare_dataset(jsonl_file=input_file)
            elif input_file.endswith('.json'):
                train_file, val_file = prepare_dataset(json_file=input_file)
            else:
                logger.error(f"Unsupported file format: {input_file}. Use .jsonl or .json")
                return
        else:
            # Use default discovery
            train_file, val_file = prepare_dataset()
        
        # Train model
        logger.info("Starting training...")
        output_dir = train_model_with_mlx_lm(
            model_path=args.model_path,
            train_file=train_file,
            val_file=val_file,
            output_dir=args.output_dir,
            num_iters=args.num_iters,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            num_layers=args.num_layers,
        )
        
        # Test the trained model
        logger.info("Testing trained model...")
        test_model(output_dir, args.model_path)
        
        logger.info("Training and testing completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()
