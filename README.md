# PostgreSQL Full-Text Search Query Generator

A machine learning project that fine-tunes Phi-3-mini to convert natural language medical queries into PostgreSQL full-text search expressions (tsquery syntax). Optimized for Apple Silicon using MLX-LM.

## 🎯 Project Overview

This project addresses the challenge of translating natural language medical queries into precise PostgreSQL full-text search expressions. By fine-tuning Microsoft's Phi-3-mini model, we create a specialized tool that understands medical terminology and generates syntactically correct tsquery expressions for database searches.

### Key Features

- **Medical Domain Specialization**: Trained specifically on medical terminology and concepts
- **PostgreSQL tsquery Generation**: Produces valid full-text search expressions
- **Apple Silicon Optimization**: Uses MLX-LM for efficient training on M1/M2/M3 Macs
- **LoRA Fine-tuning**: Memory-efficient training using Low-Rank Adaptation
- **Comprehensive Testing**: Includes database validation and query testing tools
- **Ollama Integration**: Pipeline for deploying models to Ollama for easy inference

## 🚀 Quick Start

### Prerequisites

- macOS with Apple Silicon (M1/M2/M3)
- Python 3.12+
- PostgreSQL database (for testing)
- 16GB+ RAM recommended (128GB for optimal performance)

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd pg_fulltext
   ```

2. **Set up the environment**:
   ```bash
   chmod +x setup_requirements.sh
   ./setup_requirements.sh
   ```

3. **Install dependencies**:
   ```bash
   uv sync .
   ```

### Basic Usage

1. **Prepare training data**:
   ```bash
   python phi3_mlx_training.py --prepare_only
   ```

2. **Train the model**:
   ```bash
   python phi3_mlx_training.py
   ```

3. **Test the trained model**:
   ```bash
   python inference_example.py
   ```

## 📁 Project Structure

```
pg_fulltext/
├── README.md                          # This file
├── pyproject.toml                     # Project configuration
├── main.py                           # Main entry point
├── phi3_mlx_training.py              # Core training script
├── inference_example.py              # Model inference examples
├── postgres_query_tester.py          # Database query validation
├── mlx_to_ollama_pipeline.py         # Ollama deployment pipeline
├── smart_training_params.py          # Intelligent parameter selection
├── data/                             # Training datasets
│   ├── train.jsonl                   # Training data
│   └── valid.jsonl                   # Validation data
├── synthetic_data/                   # Generated training data
├── human_verified_training_data/     # Manually verified datasets
├── lora_adapters/                    # Trained LoRA adapters
├── models/                           # Model artifacts
├── scripts/                          # Utility scripts
├── logs/                             # Training and testing logs
└── llama.cpp/                        # llama.cpp integration
```

## 🔧 Training Configuration

### Default Parameters

- **Model**: microsoft/Phi-3-mini-4k-instruct
- **Training Method**: LoRA (Low-Rank Adaptation)
- **Iterations**: 1000 (configurable)
- **Batch Size**: 4 (adjustable based on memory)
- **Learning Rate**: 1e-5
- **LoRA Layers**: 16

### Custom Training

```bash
python phi3_mlx_training.py \
    --num_iters 2000 \
    --batch_size 8 \
    --learning_rate 5e-6 \
    --lora_layers 32
```

### Smart Parameter Selection

Use the intelligent parameter selection tool:

```bash
python smart_training_params.py
```

This analyzes your dataset and hardware to recommend optimal training parameters.

## 💾 Data Format

The training data uses a prompt-completion format in JSONL:

```json
{
  "prompt": "Convert this medical query to a PostgreSQL tsquery expression:\nEffect of aspirin on cardiovascular mortality",
  "completion": "aspirin & cardiovascular & mortality"
}
```

### Converting to Phi3 MLX Format

If you need to convert existing JSONL files to the phi3 MLX format (required for MLX training), use the conversion script:

```bash
# Preview conversion
python scripts/convert2phi3mlxformat.py input.jsonl --preview

# Convert to phi3 format
python scripts/convert2phi3mlxformat.py input.jsonl output_phi3.jsonl
```

This converts from the prompt/completion format to phi3's chat template format:
```json
{"text": "<|user|>\nEffect of aspirin on cardiovascular mortality <|end|>\n<|assistant|> \naspirin & cardiovascular & mortality <|end|>"}
```

### Medical Query Examples

| Natural Language | Generated tsquery |
|------------------|-------------------|
| "MRI vs CT for stroke detection" | `(MRI \| "magnetic resonance imaging") & (CT \| "computed tomography") & stroke & detect` |
| "Beta-blockers in heart failure" | `(beta-blockers \| "beta adrenergic blockers") & (heart & failure \| HF)` |
| "Troponin sensitivity for MI" | `troponin & sensitivity & (MI \| "myocardial infarction")` |

## 🧪 Testing and Validation

### Database Testing

Test generated queries against a PostgreSQL database:

```bash
# Set up environment variables
export POSTGRES_HOST=localhost
export POSTGRES_DB=medical_db
export POSTGRES_USER=username
export POSTGRES_PASSWORD=password

# Run database tests
python postgres_query_tester.py
```

### Query Generation Testing

Test the model's inference capabilities:

```bash
python quick_generation_test.py
```

## 🚀 Deployment

### Ollama Integration

Deploy your trained model to Ollama for easy inference:

```bash
python mlx_to_ollama_pipeline.py
```

This creates a quantized GGUF model and Ollama Modelfile for deployment.

### Programmatic Usage

```python
from inference_example import PostgresQueryGenerator

# Initialize the generator
generator = PostgresQueryGenerator(
    model_path="microsoft/Phi-3-mini-4k-instruct",
    adapter_path="./lora_adapters"
)

# Generate a query
query = "What is the sensitivity of troponin for MI?"
result = generator.generate_query(query)
print(result)  # Output: troponin & sensitivity & (MI | "myocardial infarction")
```

## 📊 Performance Metrics

### Training Performance (M3 Max)

- **Training Time**: 1-2 hours for 1000 iterations
- **Memory Usage**: ~12-16GB during training
- **Iterations/Second**: 1-3 iter/sec
- **Expected Loss**: <1.0 within 500 iterations

### Model Accuracy

- **Syntax Accuracy**: >95% for well-formed tsquery expressions
- **Medical Domain Accuracy**: Significant improvement over base model
- **Query Effectiveness**: Validated against real medical databases

## 🛠️ Development Tools

### Data Generation Scripts

Generate training data from your PostgreSQL database:

```bash
# Generate Q&A pairs from abstracts (phi3 format by default)
python abstract_qa.py --limit 50 --output-jsonl qa_data.jsonl

# Generate summaries from abstracts (phi3 format by default)
python abstract_summarizer.py --limit 50 --output-jsonl summary_data.jsonl

# Use legacy prompt/completion format
python abstract_qa.py --format prompt_completion --output-jsonl qa_legacy.jsonl
python abstract_summarizer.py --format prompt_completion --output-jsonl summary_legacy.jsonl
```

### Data Conversion Scripts

```bash
# Convert between formats
python scripts/text_to_jsonl.py input.txt output.jsonl
python scripts/jsonl_to_text.py input.jsonl output.txt
python scripts/convert_data.py

# Convert existing prompt/completion format to phi3 MLX format
python scripts/convert2phi3mlxformat.py input.jsonl output_phi3.jsonl
```

### JSONL Workflow Setup

```bash
chmod +x setup_jsonl_workflow.sh
./setup_jsonl_workflow.sh
```

## 🔍 Troubleshooting

### Common Issues

1. **Import Errors**:
   ```bash
   pip install --upgrade mlx-lm
   ```

2. **Memory Issues**:
   - Reduce batch_size to 2 or 1
   - Reduce max_seq_length to 256

3. **Poor Generation Quality**:
   - Train for more iterations (2000+)
   - Use higher LoRA layers (32 or 64)
   - Lower temperature during inference (0.05)

### Performance Optimization

- **Faster Training**: Increase batch_size to 8-16
- **Better Quality**: Train for 2000+ iterations with lora_layers=32
- **Inference Speed**: Use temperature=0.0 for deterministic output

## 📚 Documentation

- **[Usage Guide](usage_guide.md)**: Detailed training and usage instructions
- **Training Logs**: Check `phi3_training.log` for detailed training progress
- **Query Test Results**: Database validation results in timestamped log files

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Microsoft**: For the Phi-3-mini model
- **MLX Team**: For the excellent MLX-LM library
- **PostgreSQL**: For robust full-text search capabilities
- **Medical Community**: For domain expertise and validation

## 📞 Support

For questions, issues, or contributions:

1. Check the [Usage Guide](usage_guide.md) for detailed instructions
2. Review existing issues in the repository
3. Create a new issue with detailed information
4. Join the discussion in project forums

---

**Note**: This project is optimized for Apple Silicon Macs. For other platforms, consider using alternative training frameworks or cloud-based solutions.