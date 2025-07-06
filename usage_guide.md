# Phi-3-mini PostgreSQL Query Generation Training Guide (Updated)

This guide will help you fine-tune Phi-3-mini on your MacBook M3 Max using the latest MLX-LM API to generate PostgreSQL full-text search queries from natural language medical queries.

## Quick Setup

1. **Download the training files** to a directory
2. **Run the setup script**:
   ```bash
   chmod +x setup.sh
   ./setup.sh
   source phi3_env/bin/activate
   ```

3. **Place your training data** (`postgres_training_data.json`) in the same directory

## Training Workflow

### Step 1: Prepare Dataset
```bash
python phi3_mlx_training.py --prepare_only
```
This converts your JSON data into the JSONL format required by MLX-LM.

### Step 2: Basic Training
```bash
python phi3_mlx_training.py
```

### Step 3: Test the Model
```bash
python phi3_mlx_training.py --test_only
```

## Advanced Training Options

### Custom Training Parameters
```bash
python phi3_mlx_training.py \
    --num_iters 2000 \
    --batch_size 8 \
    --learning_rate 5e-6 \
    --lora_layers 32
```

### Training Parameters Explained

- **`--num_iters`**: Number of training iterations (default: 1000)
  - For your 100 examples: 1000-2000 iterations should be sufficient
  - Monitor for overfitting beyond 2000 iterations

- **`--batch_size`**: Batch size (default: 4)
  - Your 128GB RAM can handle 8-16 easily
  - Larger batches = more stable training

- **`--learning_rate`**: Learning rate (default: 1e-5)
  - Lower (5e-6) for more conservative training
  - Higher (5e-5) for faster learning but risk instability

- **`--lora_layers`**: Number of LoRA layers (default: 16)
  - Higher values (32, 64) may improve performance but increase training time

## Expected Performance

### Training Time (M3 Max)
- **1000 iterations**: ~1-2 hours
- **2000 iterations**: ~2-4 hours

### Memory Usage
- **Model loading**: ~8GB
- **Training**: ~12-16GB total
- **Peak usage**: ~20GB (well within your 128GB)

### Expected Results
- **Training loss**: Should drop to <1.0 within 500 iterations
- **Syntax accuracy**: >95% for well-formed tsquery expressions
- **Medical domain accuracy**: Should improve significantly over base model

## Monitoring Training

### Log Files
- **Console output**: Real-time training progress
- **`phi3_training.log`**: Detailed training logs
- **LoRA adapters**: Saved in `./lora_adapters/` directory

### Key Metrics to Watch
1. **Loss decreasing**: Should steadily decrease
2. **Validation loss**: Reported every 100 iterations
3. **Iterations/second**: Should be 1-3 iter/sec on M3 Max

## Using the Trained Model

### Quick Test
```bash
python inference_example.py
```

### Programmatic Usage
```python
from inference_example import PostgresQueryGenerator

# Load model with trained adapters
generator = PostgresQueryGenerator(
    model_path="microsoft/Phi-3-mini-4k-instruct",
    adapter_path="./lora_adapters"
)

query = "What is the sensitivity of troponin for MI?"
result = generator.generate_query(query)
print(result)  # Expected: sensitivity & troponin & (MI | "myocardial infarction")
```

## Integration with PostgreSQL Testing

After training, integrate with your database testing script:

```python
# In your postgres testing script
from inference_example import PostgresQueryGenerator

generator = PostgresQueryGenerator(
    adapter_path="./lora_adapters"
)

# Replace static dataset with model generation
for test_case in test_cases:
    input_query = test_case['input']
    generated_output = generator.generate_query(input_query)
    
    # Test the generated query
    result = test_single_query(input_query, generated_output)
    # Log results...
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   - Make sure you're using the latest MLX-LM: `pip install --upgrade mlx-lm`
   - Activate the virtual environment: `source phi3_env/bin/activate`

2. **Out of Memory Error**
   - Reduce `batch_size` to 2 or 1
   - Reduce `max_seq_length` to 256

3. **Very High Loss**
   - Lower learning rate to 5e-6
   - Check that data format is correct (run `--prepare_only` first)

4. **Poor Generation Quality**
   - Train for more iterations (2000+)
   - Try higher LoRA layers (32 or 64)
   - Use lower temperature (0.05) during inference

5. **Training Appears Stuck**
   - Check GPU memory usage
   - Try smaller batch size
   - Ensure dataset was prepared correctly

### Performance Optimization

1. **For faster training**: Increase batch_size to 8-16
2. **For better quality**: Train for 2000+ iterations with lora_layers=32
3. **For inference speed**: Use temperature=0.0 for deterministic output

## Model Output Structure

The training saves LoRA adapters in `./lora_adapters/`:
- **`adapters.safetensors`**: The trained LoRA weights
- **`adapter_config.json`**: Configuration for the adapters

## Workflow Summary

```bash
# Complete workflow
source phi3_env/bin/activate

# 1. Prepare data (once)
python phi3_mlx_training.py --prepare_only

# 2. Train model
python phi3_mlx_training.py --num_iters 1500

# 3. Test model
python phi3_mlx_training.py --test_only

# 4. Use for inference
python inference_example.py
```

## Next Steps

1. **Evaluate on your PostgreSQL database** using the testing script
2. **Collect edge cases** where the model fails
3. **Add those cases to training data** and retrain
4. **Deploy the model** in your medical search application

## Hardware Utilization

Your M3 Max setup is excellent for this task:
- **128GB RAM**: Massive headroom for large batches
- **Apple Silicon + MLX**: Optimal performance for this setup
- **Training efficiency**: Should be 3-5x faster than comparable x86 systems

You could experiment with:
- **Larger LoRA configurations** (lora_layers=64) for better performance
- **Multiple training runs** with different hyperparameters
- **Ensemble approaches** using multiple adapter sets