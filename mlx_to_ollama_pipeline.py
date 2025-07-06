#!/usr/bin/env python3
"""
Complete MLX to Ollama Deployment Pipeline

This script takes your MLX LoRA adapters and deploys them as an Ollama model:
1. Fuse LoRA adapters with base model using mlx_lm.fuse
2. Convert to GGUF format using llama.cpp 
3. Create Ollama Modelfile
4. Import into Ollama using ollama create
"""

import subprocess
import sys
import json
import shutil
from pathlib import Path
import argparse
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ollama_deployment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MLXToOllamaDeployer:
    """Handles the complete MLX to Ollama deployment pipeline."""
    
    def __init__(
        self,
        model_name: str = "postgres-query-expert",
        base_model: str = "microsoft/Phi-3-mini-4k-instruct",
        adapter_path: str = "./lora_adapters",
        output_dir: str = "./models",
        quantization: str = "q8_0"
    ):
        self.model_name = model_name
        self.base_model = base_model
        self.adapter_path = adapter_path
        self.output_dir = Path(output_dir)
        self.quantization = quantization
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
        # Paths for intermediate files
        self.fused_model_path = self.output_dir / "fused_model"
        self.gguf_path = self.output_dir / f"{self.model_name}.gguf"
        self.modelfile_path = self.output_dir / "Modelfile"
        self.llama_cpp_path = self.output_dir.parent / "llama.cpp"
    
    def check_dependencies(self) -> bool:
        """Check if required dependencies are available."""
        logger.info("🔍 Checking dependencies...")
        
        dependencies = {
            "MLX-LM": ["python", "-m", "mlx_lm.fuse", "--help"],
            "Ollama": ["ollama", "--version"],
            "Git": ["git", "--version"]
        }
        
        missing = []
        for name, cmd in dependencies.items():
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    logger.info(f"✅ {name} is available")
                else:
                    missing.append(name)
                    logger.error(f"❌ {name} not working properly")
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
                missing.append(name)
                logger.error(f"❌ {name} not found")
        
        if missing:
            logger.error(f"Missing dependencies: {', '.join(missing)}")
            return False
        
        return True
    
    def setup_llama_cpp(self) -> bool:
        """Clone and setup llama.cpp for GGUF conversion."""
        logger.info("🔧 Setting up llama.cpp...")
        
        if self.llama_cpp_path.exists():
            logger.info("✅ llama.cpp already exists")
        else:
            try:
                logger.info("📥 Cloning llama.cpp...")
                subprocess.run([
                    "git", "clone", 
                    "https://github.com/ggerganov/llama.cpp.git",
                    str(self.llama_cpp_path)
                ], check=True)
                logger.info("✅ llama.cpp cloned successfully")
            except subprocess.CalledProcessError as e:
                logger.error(f"❌ Failed to clone llama.cpp: {e}")
                return False
        
        # Setup Python dependencies for llama.cpp conversion
        try:
            logger.info("🐍 Installing dependencies for llama.cpp conversion...")
            
            # Required packages for convert_hf_to_gguf.py
            required_packages = [
                "torch",
                "transformers>=4.32.0", 
                "sentencepiece",
                "protobuf",
                "numpy",
                "safetensors"
            ]
            
            # Check if requirements.txt exists, if not use our list
            requirements_file = self.llama_cpp_path / "requirements.txt"
            if requirements_file.exists():
                logger.info("📋 Found requirements.txt, installing from file...")
                subprocess.run([
                    sys.executable, "-m", "pip", "install", 
                    "-r", str(requirements_file)
                ], check=True)
            else:
                logger.info("📦 Installing required packages for GGUF conversion...")
                for package in required_packages:
                    logger.info(f"Installing {package}...")
                    subprocess.run([
                        sys.executable, "-m", "pip", "install", package
                    ], check=True)
                
            logger.info("✅ llama.cpp dependencies installed successfully")
                
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to install dependencies: {e}")
            # Try to continue anyway - the conversion might still work
            logger.warning("⚠️ Continuing anyway - conversion might still work...")
        
        # Verify the conversion script exists
        convert_script = self.llama_cpp_path / "convert_hf_to_gguf.py"
        if convert_script.exists():
            logger.info("✅ GGUF conversion script found")
            return True
        else:
            logger.error(f"❌ Conversion script not found: {convert_script}")
            return False
    
    def fuse_lora_adapters(self) -> bool:
        """Fuse LoRA adapters with the base model (fallback method)."""
        logger.info("🔀 Fusing LoRA adapters with base model...")
        
        if not Path(self.adapter_path).exists():
            logger.error(f"❌ Adapter path {self.adapter_path} not found!")
            return False
        
        cmd = [
            sys.executable, "-m", "mlx_lm.fuse",
            "--model", self.base_model,
            "--adapter-path", self.adapter_path,
            "--save-path", str(self.fused_model_path),
            "--de-quantize"  # Important for GGUF conversion
        ]
        
        try:
            logger.info(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            logger.info("✅ LoRA adapters fused successfully!")
            if result.stdout:
                logger.info(f"Output: {result.stdout}")
            if result.stderr:
                logger.warning(f"Warnings: {result.stderr}")
                
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to fuse adapters: {e}")
            logger.error(f"Error output: {e.stderr}")
            return False
    
    def fuse_and_convert_direct(self) -> bool:
        """Alternative: Fuse LoRA adapters and convert to GGUF in one step using MLX-LM."""
        logger.info("🔀 Fusing LoRA adapters and converting to GGUF directly...")
        
        if not Path(self.adapter_path).exists():
            logger.error(f"❌ Adapter path {self.adapter_path} not found!")
            return False
        
        # Try direct GGUF conversion (only works for some model types)
        cmd = [
            sys.executable, "-m", "mlx_lm", "fuse",
            "--model", self.base_model,
            "--adapter-path", self.adapter_path,
            "--save-path", str(self.fused_model_path),
            "--gguf-path", str(self.gguf_path),
            "--de-quantize"
        ]
        
        try:
            logger.info(f"Trying direct GGUF conversion...")
            logger.info(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            if result.stdout:
                logger.info(f"Output: {result.stdout}")
            if result.stderr:
                logger.warning(f"Warnings: {result.stderr}")
                
            # Check if GGUF file was created
            if self.gguf_path.exists():
                logger.info("✅ Model fused and converted to GGUF successfully!")
                logger.info(f"✅ GGUF file created: {self.gguf_path}")
                return True
            else:
                logger.warning("⚠️ Direct GGUF conversion didn't create GGUF file")
                logger.info("📝 Model was fused but GGUF conversion failed - will try two-step process")
                return False
                
        except subprocess.CalledProcessError as e:
            logger.warning(f"⚠️ Direct conversion failed: {e}")
            if e.stderr:
                logger.warning(f"Error output: {e.stderr}")
            logger.info("🔄 Falling back to two-step process...")
            return False
        """Fuse LoRA adapters with the base model."""
        logger.info("🔀 Fusing LoRA adapters with base model...")
        
        if not Path(self.adapter_path).exists():
            logger.error(f"❌ Adapter path {self.adapter_path} not found!")
            return False
        
        cmd = [
            sys.executable, "-m", "mlx_lm.fuse",
            "--model", self.base_model,
            "--adapter-path", self.adapter_path,
            "--save-path", str(self.fused_model_path),
            "--de-quantize"  # Important for GGUF conversion
        ]
        
        try:
            logger.info(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            logger.info("✅ LoRA adapters fused successfully!")
            if result.stdout:
                logger.info(f"Output: {result.stdout}")
            if result.stderr:
                logger.warning(f"Warnings: {result.stderr}")
                
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to fuse adapters: {e}")
            logger.error(f"Error output: {e.stderr}")
            return False
    
    def convert_to_gguf(self) -> bool:
        """Convert the fused model to GGUF format."""
        logger.info("📦 Converting model to GGUF format...")
        
        if not self.fused_model_path.exists():
            logger.error(f"❌ Fused model path {self.fused_model_path} not found!")
            return False
        
        # Path to conversion script (use absolute path to avoid issues)
        convert_script = self.llama_cpp_path / "convert_hf_to_gguf.py"
        if not convert_script.exists():
            logger.error(f"❌ Conversion script not found: {convert_script}")
            # Try alternate location
            convert_script = self.llama_cpp_path / "convert-hf-to-gguf.py"
            if not convert_script.exists():
                logger.error(f"❌ Conversion script not found at {convert_script} either")
                return False
        
        # Use absolute paths to avoid path resolution issues
        cmd = [
            sys.executable, 
            str(convert_script.absolute()),
            str(self.fused_model_path.absolute()),
            "--outfile", str(self.gguf_path.absolute()),
            "--outtype", self.quantization
        ]
        
        try:
            logger.info(f"Running: {' '.join(cmd)}")
            # Don't change working directory - use absolute paths instead
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            logger.info("✅ Model converted to GGUF successfully!")
            logger.info(f"GGUF model saved to: {self.gguf_path}")
            
            if result.stdout:
                logger.info(f"Conversion output: {result.stdout}")
                
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to convert to GGUF: {e}")
            logger.error(f"Error output: {e.stderr}")
            return False
    
    def create_modelfile(self) -> bool:
        """Create Ollama Modelfile."""
        logger.info("📝 Creating Ollama Modelfile...")
        
        if not self.gguf_path.exists():
            logger.error(f"❌ GGUF file not found: {self.gguf_path}")
            return False
        
        # Get relative path to GGUF file from Modelfile location
        relative_gguf_path = self.gguf_path.relative_to(self.modelfile_path.parent)
        
        modelfile_content = f"""FROM ./{relative_gguf_path}

TEMPLATE \"\"\"<|system|>
You are a medical database search expert. Convert natural language medical queries into PostgreSQL full-text search expressions using tsquery syntax.

Rules:
- Use & for AND operations
- Use | for OR operations  
- Use quotes for exact phrases
- Favor broader terms to avoid false negatives
- Don't include subset terms (e.g., use 'thoracotomy' not 'emergency thoracotomy')
- Let PostgreSQL stemming handle plurals

<|user|>
{{{{ .Prompt }}}}

<|assistant|>
\"\"\"

PARAMETER stop \"<|end|>\"
PARAMETER stop \"<|endoftext|>\"
PARAMETER temperature 0.1
PARAMETER top_p 0.9
"""
        
        try:
            with open(self.modelfile_path, 'w') as f:
                f.write(modelfile_content)
            
            logger.info(f"✅ Modelfile created: {self.modelfile_path}")
            logger.info("📋 Modelfile content:")
            logger.info(modelfile_content)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create Modelfile: {e}")
            return False
    
    def import_to_ollama(self) -> bool:
        """Import the model into Ollama."""
        logger.info("📥 Importing model into Ollama...")
        
        if not self.modelfile_path.exists():
            logger.error(f"❌ Modelfile not found: {self.modelfile_path}")
            return False
        
        # Change to the directory containing the Modelfile
        original_cwd = Path.cwd()
        
        try:
            cmd = ["ollama", "create", self.model_name, "-f", "Modelfile"]
            
            logger.info(f"Running: {' '.join(cmd)} (in {self.output_dir})")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=self.output_dir)
            
            logger.info("✅ Model imported to Ollama successfully!")
            if result.stdout:
                logger.info(f"Import output: {result.stdout}")
                
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to import to Ollama: {e}")
            logger.error(f"Error output: {e.stderr}")
            return False
    
    def test_ollama_model(self) -> bool:
        """Test the imported Ollama model."""
        logger.info("🧪 Testing Ollama model...")
        
        test_prompt = "What is the sensitivity of troponin for myocardial infarction?"
        
        cmd = ["ollama", "run", self.model_name, test_prompt]
        
        try:
            logger.info(f"Testing with prompt: '{test_prompt}'")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=30)
            
            response = result.stdout.strip()
            logger.info(f"🎯 Model response: '{response}'")
            
            # Basic quality check
            if any(op in response for op in ['&', '|']):
                logger.info("✅ Response contains logical operators - looks good!")
                return True
            else:
                logger.warning("⚠️ Response doesn't contain expected operators")
                return False
                
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to test model: {e}")
            return False
        except subprocess.TimeoutExpired:
            logger.error("❌ Model test timed out")
            return False
    
    def list_ollama_models(self):
        """List all Ollama models."""
        try:
            result = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=True)
            logger.info("📋 Available Ollama models:")
            logger.info(result.stdout)
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to list Ollama models: {e}")
    
    def deploy(self) -> bool:
        """Run the complete deployment pipeline."""
        logger.info("🚀 Starting MLX to Ollama deployment pipeline...")
        logger.info(f"Model name: {self.model_name}")
        logger.info(f"Base model: {self.base_model}")
        logger.info(f"Adapter path: {self.adapter_path}")
        logger.info(f"Output directory: {self.output_dir}")
        
        # Step 1: Check dependencies
        logger.info(f"\n{'='*60}")
        logger.info(f"Step: Checking dependencies")
        logger.info(f"{'='*60}")
        if not self.check_dependencies():
            logger.error(f"❌ Pipeline failed at step: Checking dependencies")
            return False
        
        # Step 2: Try direct fuse and convert (simpler approach)
        logger.info(f"\n{'='*60}")
        logger.info(f"Step: Direct fuse and GGUF conversion")
        logger.info(f"{'='*60}")
        
        direct_success = self.fuse_and_convert_direct()
        
        if not direct_success:
            # Fall back to two-step process
            logger.info("🔄 Direct conversion failed, trying two-step process...")
            
            steps = [
                ("Setting up llama.cpp", self.setup_llama_cpp),
                ("Fusing LoRA adapters", self.fuse_lora_adapters),
                ("Converting to GGUF", self.convert_to_gguf)
            ]
            
            for step_name, step_func in steps:
                logger.info(f"\n{'='*60}")
                logger.info(f"Step: {step_name}")
                logger.info(f"{'='*60}")
                
                if not step_func():
                    logger.error(f"❌ Pipeline failed at step: {step_name}")
                    return False
        
        # Continue with remaining steps
        final_steps = [
            ("Creating Modelfile", self.create_modelfile),
            ("Importing to Ollama", self.import_to_ollama),
            ("Testing model", self.test_ollama_model)
        ]
        
        for step_name, step_func in final_steps:
            logger.info(f"\n{'='*60}")
            logger.info(f"Step: {step_name}")
            logger.info(f"{'='*60}")
            
            if not step_func():
                logger.error(f"❌ Pipeline failed at step: {step_name}")
                return False
        
        logger.info(f"\n🎉 Deployment completed successfully!")
        logger.info(f"Your model '{self.model_name}' is now available in Ollama")
        logger.info(f"\nTo use your model:")
        logger.info(f"  ollama run {self.model_name} \"What are the contraindications for MRI?\"")
        
        # List models
        self.list_ollama_models()
        
        return True

def main():
    parser = argparse.ArgumentParser(description="Deploy MLX LoRA model to Ollama")
    parser.add_argument("--model-name", default="postgres-query-expert", help="Name for the Ollama model")
    parser.add_argument("--base-model", default="microsoft/Phi-3-mini-4k-instruct", help="Base model path")
    parser.add_argument("--adapter-path", default="./lora_adapters", help="Path to LoRA adapters")
    parser.add_argument("--output-dir", default="./models", help="Output directory")
    parser.add_argument("--quantization", default="q8_0", choices=["q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "f16", "f32"], help="GGUF quantization type")
    parser.add_argument("--skip-test", action="store_true", help="Skip model testing")
    
    args = parser.parse_args()
    
    deployer = MLXToOllamaDeployer(
        model_name=args.model_name,
        base_model=args.base_model,
        adapter_path=args.adapter_path,
        output_dir=args.output_dir,
        quantization=args.quantization
    )
    
    try:
        success = deployer.deploy()
        if success:
            logger.info("🎊 Deployment pipeline completed successfully!")
            sys.exit(0)
        else:
            logger.error("💥 Deployment pipeline failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("❌ Deployment cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"💥 Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
