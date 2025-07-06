#!/usr/bin/env python3
"""
Quick fix script for llama.cpp setup issues
"""

import subprocess
import sys
from pathlib import Path

def fix_llama_cpp_setup():
    """Fix the llama.cpp setup by installing required packages."""
    
    print("🔧 Fixing llama.cpp setup...")
    
    # Required packages for GGUF conversion
    required_packages = [
        "torch",
        "transformers>=4.32.0", 
        "sentencepiece",
        "protobuf",
        "numpy",
        "safetensors"
    ]
    
    print("📦 Installing required packages for GGUF conversion...")
    
    for package in required_packages:
        try:
            print(f"Installing {package}...")
            subprocess.run([
                sys.executable, "-m", "pip", "install", package
            ], check=True, capture_output=True)
            print(f"✅ {package} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"⚠️ Warning: Failed to install {package}: {e}")
    
    # Verify llama.cpp conversion script exists
    llama_cpp_path = Path("./llama.cpp")
    convert_script = llama_cpp_path / "convert_hf_to_gguf.py"
    
    if convert_script.exists():
        print(f"✅ GGUF conversion script found at {convert_script}")
    else:
        print(f"❌ Conversion script not found. Make sure llama.cpp is cloned properly.")
        print("Run: git clone https://github.com/ggerganov/llama.cpp.git")
    
    print("\n🎉 Setup fixed! You can now run the deployment pipeline again.")
    print("python mlx_to_ollama_pipeline.py")

if __name__ == "__main__":
    fix_llama_cpp_setup()
