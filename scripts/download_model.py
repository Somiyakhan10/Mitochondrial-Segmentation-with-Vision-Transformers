# scripts/download_model.py
import os
import urllib.request
from pathlib import Path

def download_model():
    """Download trained segmentation model during build"""
    
    model_dir = Path("data/models")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = model_dir / "segmentation_unet.pt"
    
    if model_path.exists():
        print(f"✅ Model already exists: {model_path}")
        return
    
    print("📥 Downloading segmentation model...")
    
    # Try downloading from Hugging Face
    url = "https://huggingface.co/somiyakhan10/mitomorph/resolve/main/segmentation_unet.pt"
    
    try:
        urllib.request.urlretrieve(url, model_path)
        print(f"✅ Model downloaded to: {model_path}")
        print(f"   Size: {model_path.stat().st_size / (1024*1024):.2f} MB")
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        print("   Please upload model manually to data/models/")

if __name__ == "__main__":
    download_model()
