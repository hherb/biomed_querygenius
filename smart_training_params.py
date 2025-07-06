#!/usr/bin/env python3
"""
Smart parameter detection based on dataset size
"""

import json
from pathlib import Path

def detect_optimal_parameters(data_file, current_args):
    """
    Analyze dataset and suggest optimal training parameters.
    
    Args:
        data_file: Path to training data file
        current_args: Current argument values
        
    Returns:
        dict: Suggested parameters with reasoning
    """
    
    # Count examples in dataset
    if data_file.endswith('.jsonl'):
        example_count = count_jsonl_examples(data_file)
    elif data_file.endswith('.json'):
        example_count = count_json_examples(data_file)
    else:
        return None
    
    print(f"📊 Dataset analysis: {example_count} training examples")
    
    # Determine optimal parameters based on size
    if example_count < 500:
        # Small dataset
        suggested = {
            'num_iters': 1000,
            'batch_size': 4,
            'learning_rate': 1e-5,
            'num_layers': 16,
            'category': 'Small Dataset',
            'reasoning': 'Conservative parameters to avoid overfitting'
        }
    elif example_count < 2000:
        # Medium dataset  
        suggested = {
            'num_iters': 1500,
            'batch_size': 8,
            'learning_rate': 5e-6,
            'num_layers': 32,
            'category': 'Medium Dataset',
            'reasoning': 'Balanced parameters for stable training'
        }
    else:
        # Large dataset
        suggested = {
            'num_iters': 2500,
            'batch_size': 12,
            'learning_rate': 3e-6,
            'num_layers': 32,
            'category': 'Large Dataset',
            'reasoning': 'Aggressive parameters to leverage data volume'
        }
    
    # Check if current parameters are optimal
    needs_adjustment = (
        current_args.num_iters != suggested['num_iters'] or
        current_args.batch_size != suggested['batch_size'] or
        current_args.learning_rate != suggested['learning_rate'] or
        current_args.num_layers != suggested['num_layers']
    )
    
    return {
        'example_count': example_count,
        'suggested': suggested,
        'current': {
            'num_iters': current_args.num_iters,
            'batch_size': current_args.batch_size,
            'learning_rate': current_args.learning_rate,
            'num_layers': current_args.num_layers
        },
        'needs_adjustment': needs_adjustment
    }

def count_jsonl_examples(file_path):
    """Count examples in JSONL file."""
    count = 0
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        json.loads(line)
                        count += 1
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        print(f"⚠️ Error reading {file_path}: {e}")
    return count

def count_json_examples(file_path):
    """Count examples in JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return len(data) if isinstance(data, list) else 0
    except Exception as e:
        print(f"⚠️ Error reading {file_path}: {e}")
        return 0

def show_parameter_recommendations(analysis):
    """Display parameter recommendations to user."""
    if not analysis:
        return
    
    suggested = analysis['suggested']
    current = analysis['current']
    
    print(f"\n🎯 {suggested['category']} Detected ({analysis['example_count']} examples)")
    print(f"💡 {suggested['reasoning']}")
    print("-" * 60)
    
    print("📋 Parameter Comparison:")
    params = ['num_iters', 'batch_size', 'learning_rate', 'num_layers']
    
    for param in params:
        current_val = current[param]
        suggested_val = suggested[param]
        
        if current_val == suggested_val:
            status = "✅ Optimal"
        else:
            status = f"📈 Suggested: {suggested_val}"
        
        print(f"  {param:15}: {current_val:>10} | {status}")
    
    if analysis['needs_adjustment']:
        print(f"\n🚀 Recommended command:")
        print(f"python phi3_mlx_training.py \\")
        print(f"  --num_iters {suggested['num_iters']} \\")
        print(f"  --batch_size {suggested['batch_size']} \\")
        print(f"  --learning_rate {suggested['learning_rate']} \\")
        print(f"  --num_layers {suggested['num_layers']}")
        
        print(f"\n⏱️ Estimated training time: {estimate_training_time(suggested, analysis['example_count'])}")
    else:
        print(f"\n✅ Current parameters are optimal for this dataset size!")

def estimate_training_time(params, example_count):
    """Estimate training time based on parameters and dataset size."""
    # Rough estimates for M3 Max
    iterations = params['num_iters']
    batch_size = params['batch_size']
    
    # Approximate iterations per second on M3 Max
    iters_per_second = 2.5
    
    # Adjust for batch size (larger batches are slightly slower per iteration)
    iters_per_second *= (4 / batch_size) * 0.8
    
    total_seconds = iterations / iters_per_second
    
    if total_seconds < 3600:
        return f"{total_seconds/60:.0f} minutes"
    else:
        return f"{total_seconds/3600:.1f} hours"

def check_file_selection(data_dir):
    """Check which file will be selected and warn about multiple files."""
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"⚠️ Directory {data_dir} does not exist")
        return None
    
    jsonl_files = list(data_path.glob("*.jsonl"))
    json_files = list(data_path.glob("*.json"))
    
    print(f"📁 Files in {data_dir}:")
    
    if jsonl_files:
        print("  JSONL files found:")
        for i, file in enumerate(jsonl_files):
            marker = "← WILL USE" if i == 0 else ""
            size = count_jsonl_examples(file)
            print(f"    {file.name} ({size} examples) {marker}")
        
        if len(jsonl_files) > 1:
            print(f"  ⚠️ Multiple JSONL files found! Will use: {jsonl_files[0].name}")
            print(f"  💡 To use a specific file: --data-file path/to/your/file.jsonl")
        
        return str(jsonl_files[0])
    
    elif json_files:
        print("  JSON files found:")
        for i, file in enumerate(json_files):
            marker = "← WILL USE" if i == 0 else ""
            size = count_json_examples(file)
            print(f"    {file.name} ({size} examples) {marker}")
        
        return str(json_files[0])
    
    else:
        print("  ❌ No .jsonl or .json files found")
        return None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze dataset and suggest optimal training parameters")
    parser.add_argument("--data-file", type=str, help="Specific data file to analyze")
    parser.add_argument("--data-dir", type=str, default="./human_verified_training_data", help="Data directory to scan")
    
    args = parser.parse_args()
    
    print("🔍 Smart Training Parameter Detection")
    print("=" * 50)
    
    if args.data_file:
        data_file = args.data_file
        print(f"📄 Analyzing specified file: {data_file}")
    else:
        data_file = check_file_selection(args.data_dir)
        if not data_file:
            print("❌ No training data found")
            print(f"💡 Place your .jsonl file in {args.data_dir}/ or use --data-file")
            exit(1)
    
    # Create mock args for analysis
    class MockArgs:
        num_iters = 1000
        batch_size = 4
        learning_rate = 1e-5
        num_layers = 16
    
    mock_args = MockArgs()
    analysis = detect_optimal_parameters(data_file, mock_args)
    
    if analysis:
        show_parameter_recommendations(analysis)
    else:
        print("❌ Could not analyze dataset")
