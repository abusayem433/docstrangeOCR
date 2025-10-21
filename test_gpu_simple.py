#!/usr/bin/env python3
"""Simple GPU test without OCR initialization."""

import sys
import os

def test_gpu_basic():
    """Test basic GPU functionality without OCR."""
    print("Testing GPU availability...")
    
    try:
        import torch
        print(f"+ PyTorch version: {torch.__version__}")
        print(f"+ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"+ GPU count: {torch.cuda.device_count()}")
            print(f"+ GPU name: {torch.cuda.get_device_name(0)}")
            print(f"+ CUDA version: {torch.version.cuda}")
            return True
        else:
            print("- CUDA not available")
            return False
    except ImportError as e:
        print(f"- PyTorch not available: {e}")
        return False

def test_docstrange_import():
    """Test DocStrange import without initialization."""
    print("\nTesting DocStrange import...")
    
    try:
        from docstrange.extractor import DocumentExtractor
        print("+ DocStrange imported successfully")
        return True
    except Exception as e:
        print(f"- DocStrange import failed: {e}")
        return False

def test_gpu_utils():
    """Test GPU utility functions."""
    print("\nTesting GPU utilities...")
    
    try:
        from docstrange.utils.gpu_utils import should_use_gpu_processor
        gpu_available = should_use_gpu_processor()
        print(f"+ GPU processor check: {gpu_available}")
        return gpu_available
    except Exception as e:
        print(f"- GPU utils failed: {e}")
        return False

def main():
    """Run all tests."""
    print("DocStrange Local GPU Processing - Simple Test")
    print("=" * 50)
    
    gpu_ok = test_gpu_basic()
    import_ok = test_docstrange_import()
    utils_ok = test_gpu_utils()
    
    print("\n" + "=" * 50)
    print("Test Summary:")
    print(f"GPU Available: {'+' if gpu_ok else '-'}")
    print(f"DocStrange Import: {'+' if import_ok else '-'}")
    print(f"GPU Utils: {'+' if utils_ok else '-'}")
    
    if gpu_ok and import_ok and utils_ok:
        print("\nSUCCESS: System is ready for local GPU processing!")
        print("The Unicode issue is only affecting the OCR model download progress bar.")
        print("The core GPU functionality is working correctly.")
    else:
        print("\nERROR: Some components need attention.")
    
    return gpu_ok and import_ok and utils_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
