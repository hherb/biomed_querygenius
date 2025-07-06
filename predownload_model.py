#!/usr/bin/env python3
"""Download Phi-3-mini with visible progress"""

from huggingface_hub import snapshot_download
from tqdm import tqdm
import os

def download_with_progress():
    print("Downloading Phi-3-mini with progress tracking...")
    
    # This should show progress bars
    snapshot_download(
        "microsoft/Phi-3-mini-4k-instruct",
        cache_dir="~/.cache/huggingface/hub",
        resume_download=True,
        local_files_only=False,
    )
    
    print("Download complete!")

if __name__ == "__main__":
    download_with_progress()