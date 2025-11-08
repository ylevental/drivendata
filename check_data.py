#!/usr/bin/env python3
"""
Data verification script for wildlife classification competition
Run this FIRST to make sure everything is set up correctly
"""

import os
import pandas as pd
from pathlib import Path

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

def check_data():
    """Verify all data files are present and valid"""
    print("=" * 60)
    print("CHECKING DATA SETUP")
    print("=" * 60)
    
    errors = []
    warnings = []
    
    # Check directory exists
    if not os.path.exists(DATA_DIR):
        errors.append(f"Data directory not found: {DATA_DIR}")
        print("\n❌ ERRORS:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    print(f"\n✓ Data directory found: {DATA_DIR}")
    
    # Check CSV files
    train_labels = os.path.join(DATA_DIR, 'train_labels.csv')
    submission_format = os.path.join(DATA_DIR, 'submission_format.csv')
    
    if not os.path.exists(train_labels):
        errors.append("train_labels.csv not found")
    else:
        print(f"✓ train_labels.csv found")
        df = pd.read_csv(train_labels)
        print(f"  - {len(df)} training labels")
        # Multi-label format - get class columns
        class_cols = [col for col in df.columns if col != 'id']
        print(f"  - Classes: {class_cols}")
        print(f"  - Class counts:")
        for col in class_cols:
            print(f"    {col}: {int(df[col].sum())}")
    
    if not os.path.exists(submission_format):
        errors.append("submission_format.csv not found")
    else:
        print(f"✓ submission_format.csv found")
        df = pd.read_csv(submission_format)
        print(f"  - {len(df)} test samples")
    
    # Check image directories
    train_dir = os.path.join(DATA_DIR, 'train_features')
    test_dir = os.path.join(DATA_DIR, 'test_features')
    
    if not os.path.exists(train_dir):
        errors.append(f"Training directory not found: {train_dir}")
    else:
        train_images = list(Path(train_dir).rglob('*.jpg'))
        print(f"✓ train_features directory found")
        print(f"  - {len(train_images)} training images")
        
        if len(train_images) == 0:
            errors.append("No training images found!")
    
    if not os.path.exists(test_dir):
        errors.append(f"Test directory not found: {test_dir}")
    else:
        test_images = list(Path(test_dir).rglob('*.jpg'))
        print(f"✓ test_features directory found")
        print(f"  - {len(test_images)} test images")
        
        if len(test_images) == 0:
            errors.append("No test images found!")
    
    # Check for CUDA
    print(f"\n{'=' * 60}")
    print("SYSTEM CHECK")
    print("=" * 60)
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
            print(f"  - CUDA version: {torch.version.cuda}")
            print(f"  - GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            warnings.append("No GPU detected - training will be SLOW on CPU")
            print("⚠ No GPU detected - will use CPU (training will be slower)")
    except ImportError:
        warnings.append("PyTorch not installed yet")
        print("⚠ PyTorch not installed - run: pip install -r requirements.txt")
    
    # Print summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print("=" * 60)
    
    if errors:
        print("\n❌ ERRORS:")
        for error in errors:
            print(f"  - {error}")
    
    if warnings:
        print("\n⚠ WARNINGS:")
        for warning in warnings:
            print(f"  - {warning}")
    
    if not errors:
        print("\n✅ All checks passed! Ready to train.")
        print("\nNext steps:")
        print("  1. Install dependencies: pip install -r requirements.txt")
        print("  2. Train model: python3 train_wildlife.py --mode train")
        print("  3. Generate predictions: python3 train_wildlife.py --mode predict")
        print("  Or run both: python3 train_wildlife.py --mode both")
        return True
    else:
        print("\n❌ Please fix errors before training.")
        return False


if __name__ == '__main__':
    check_data()
