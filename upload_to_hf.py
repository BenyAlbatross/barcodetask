"""
Script to upload barcode detection dataset to Hugging Face Hub
"""

from huggingface_hub import HfApi, create_repo
from pathlib import Path
import os

api = HfApi()

# Upload the entire dataset folder (includes images, labels, and data.yaml)
api.upload_large_folder(
    repo_id="albagon/barcode",
    repo_type="dataset",
    folder_path=Path("barcode_dataset")
)