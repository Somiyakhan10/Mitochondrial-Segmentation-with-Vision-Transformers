#!/bin/bash
echo "🚀 Starting build process..."

echo "📦 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "📦 Installing mitomorph package..."
pip install -e .

echo "📥 Downloading trained model..."
python scripts/download_model.py

echo "✅ Build complete!"
