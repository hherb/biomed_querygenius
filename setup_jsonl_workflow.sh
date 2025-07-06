#!/bin/bash
# Setup script for JSONL-based Phi-3-mini training workflow

echo "🚀 Setting up JSONL Training Workflow for Phi-3-mini"
echo "=" * 60

# Create directory structure
echo "📁 Creating directory structure..."
mkdir -p ./human_verified_training_data
mkdir -p ./data
mkdir -p ./lora_adapters
mkdir -p ./models

# Create sample JSONL file if none exists
if [ ! -f "./human_verified_training_data/training_data.jsonl" ]; then
    echo "📝 Creating sample training data..."
    cat > ./human_verified_training_data/training_data.jsonl << 'EOF'
{"prompt": "Convert this medical query to a PostgreSQL tsquery expression:\nWhat is the sensitivity of troponin for myocardial infarction?", "completion": "sensitivity & troponin & (MI | \"myocardial infarction\")"}
{"prompt": "Convert this medical query to a PostgreSQL tsquery expression:\nWhat are the contraindications for lumbar puncture?", "completion": "contraindication & (\"lumbar puncture\" | LP)"}
{"prompt": "Convert this medical query to a PostgreSQL tsquery expression:\nWhat is the treatment for septic shock?", "completion": "treatment & septic & shock"}
{"prompt": "Convert this medical query to a PostgreSQL tsquery expression:\nWhat are the risk factors for pulmonary embolism?", "completion": "risk & factor & (PE | \"pulmonary embolism\")"}
{"prompt": "Convert this medical query to a PostgreSQL tsquery expression:\nWhat is the Glasgow Coma Scale scoring system?", "completion": "(GCS | \"Glasgow Coma Scale\") & scoring"}
EOF
    echo "✅ Sample training data created at ./human_verified_training_data/training_data.jsonl"
else
    echo "✅ Training data already exists"
fi

# Create .gitignore if needed
if [ ! -f "./.gitignore" ]; then
    echo "📝 Creating .gitignore..."
    cat > ./.gitignore << 'EOF'
# Training artifacts
./lora_adapters/
./models/fused_model/
./data/train.jsonl
./data/valid.jsonl
*.log

# Python
__pycache__/
*.pyc
*.pyo
.venv/
phi3_env/

# System
.DS_Store
.env

# Large files
*.gguf
*.safetensors
*.bin

# Keep structure but ignore contents
./human_verified_training_data/*
!./human_verified_training_data/.gitkeep
EOF
    touch ./human_verified_training_data/.gitkeep
    echo "✅ .gitignore created"
fi

# Show directory structure
echo ""
echo "📂 Directory structure created:"
tree -a -I '.git|__pycache__|*.pyc' . 2>/dev/null || ls -la

echo ""
echo "💡 Quick Start Guide:"
echo "=" * 40

echo ""
echo "1️⃣ Add your training data:"
echo "   - Place your JSONL file in ./human_verified_training_data/"
echo "   - Format: {\"prompt\": \"Convert...\", \"completion\": \"tsquery...\"}"
echo ""

echo "2️⃣ Prepare and train:"
echo "   python debug_data_format.py --input ./human_verified_training_data/your_data.jsonl"
echo "   python phi3_mlx_training.py --data-file ./human_verified_training_data/your_data.jsonl"
echo ""

echo "3️⃣ Test the model:"
echo "   python phi3_mlx_training.py --test_only"
echo "   python postgres_query_tester.py --data ./data/valid.jsonl --jsonl --compare"
echo ""

echo "4️⃣ Deploy to Ollama:"
echo "   python mlx_to_ollama_pipeline.py"
echo ""

echo "📚 Data Format Examples:"
echo "=" * 30

echo ""
echo "JSONL Format (recommended):"
cat << 'EOF'
{"prompt": "Convert this medical query to a PostgreSQL tsquery expression:\nWhat is troponin sensitivity for MI?", "completion": "sensitivity & troponin & (MI | \"myocardial infarction\")"}
EOF

echo ""
echo "JSON Format (legacy):"
cat << 'EOF'
[
  {
    "input": "What is troponin sensitivity for MI?",
    "output": "sensitivity & troponin & (MI | \"myocardial infarction\")"
  }
]
EOF

echo ""
echo "🔧 Available Commands:"
echo "=" * 25

commands=(
    "python debug_data_format.py --show-formats          # Show supported formats"
    "python phi3_mlx_training.py --prepare_only          # Just prepare data"
    "python phi3_mlx_training.py --check_install         # Check MLX-LM installation"
    "python phi3_mlx_training.py --test_only             # Test trained model"
    "python simple_fused_inference.py --test             # Test without Ollama"
    "python postgres_query_tester.py --data file.jsonl --jsonl  # Test against DB"
    "python mlx_to_ollama_pipeline.py                    # Deploy to Ollama"
)

for cmd in "${commands[@]}"; do
    echo "   $cmd"
done

echo ""
echo "✨ Setup complete! Ready for JSONL-based training workflow."
