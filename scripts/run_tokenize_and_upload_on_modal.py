"""
Script to run tokenize_and_upload.py on Modal GPUs.

Usage:
    1. Deploy: modal deploy run_on_modal.py
    2. Run detached: modal run --detach run_on_modal.py::app.run --repo-name username/dataset --batch-size batch_size_value
    3. Check logs: modal app logs libritts-r-tokenizer
"""

import modal

# Modal setup
app = modal.App("libritts-r-tokenizer")

# Create Modal image with required dependencies
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "libsndfile1")  # Install FFmpeg and audio libraries
    .run_commands("pip install --upgrade pip")
    .pip_install(
        "datasets[audio]>=4.2.0",
        "torch>=2.9.0",
        "torchaudio>=2.9.0",
        "neucodec>=0.0.4",
        "transformers>=4.57.1",
        "huggingface-hub>=0.27.0",
    )
)

# Create Modal volume for caching
volume = modal.Volume.from_name("libritts-cache", create_if_missing=True)

# HuggingFace secrets for authentication
HF_TOKEN = modal.Secret.from_name("HF_TOKEN")


@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={"/cache": volume},
    secrets=[HF_TOKEN],
    timeout=86400,  # 24 hours timeout
)
def run(
    repo_name: str,
    batch_size: int = 32,
    private: bool = False,
):
    """
    Run the tokenization and upload script on Modal with GPU.
    Imports and calls tokenize_dataset from tokenize_and_upload module.

    Args:
        repo_name: HuggingFace repo name (e.g., 'username/libritts_r_neucodec')
        batch_size: Batch size for processing
        private: Whether to make the dataset private
    """
    # Import the shared module
    from tokenize_and_upload import tokenize_dataset

    # Run tokenization with cache directory on volume
    tokenize_dataset(
        repo_name=repo_name,
        batch_size=batch_size,
        private=private,
        cache_dir="/cache/huggingface",
    )
