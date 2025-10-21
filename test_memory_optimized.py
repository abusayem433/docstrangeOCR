#!/usr/bin/env python3
"""Test document extraction with memory optimization."""

import os
import sys
from pathlib import Path

def test_simple_extraction():
    """Test simple text extraction without OCR."""
    print("Testing simple text extraction...")
    
    try:
        from docstrange import DocumentExtractor
        
        # Create a simple text file for testing
        test_content = "Hello World! This is a test document for DocStrange local GPU processing."
        test_file = "test_simple.txt"
        
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(test_content)
        
        print(f"Created test file: {test_file}")
        
        # Test extraction
        extractor = DocumentExtractor()
        result = extractor.extract(test_file)
        
        print("SUCCESS: Text extraction working!")
        print(f"Extracted content: {result.content[:100]}...")
        
        # Cleanup
        os.remove(test_file)
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        return False

def test_gpu_memory():
    """Test GPU memory status."""
    print("\nTesting GPU memory status...")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            total_memory = torch.cuda.get_device_properties(device).total_memory
            allocated_memory = torch.cuda.memory_allocated(device)
            cached_memory = torch.cuda.memory_reserved(device)
            
            print(f"GPU: {torch.cuda.get_device_name(device)}")
            print(f"Total Memory: {total_memory / 1024**3:.2f} GB")
            print(f"Allocated Memory: {allocated_memory / 1024**3:.2f} GB")
            print(f"Cached Memory: {cached_memory / 1024**3:.2f} GB")
            print(f"Free Memory: {(total_memory - allocated_memory) / 1024**3:.2f} GB")
            
            # Clear cache
            torch.cuda.empty_cache()
            print("GPU cache cleared")
            
        return True
        
    except Exception as e:
        print(f"ERROR checking GPU memory: {e}")
        return False

def main():
    """Run tests."""
    print("DocStrange Memory-Optimized Extraction Test")
    print("=" * 50)
    
    gpu_ok = test_gpu_memory()
    extraction_ok = test_simple_extraction()
    
    print("\n" + "=" * 50)
    print("Test Summary:")
    print(f"GPU Memory Check: {'PASS' if gpu_ok else 'FAIL'}")
    print(f"Simple Extraction: {'PASS' if extraction_ok else 'FAIL'}")
    
    if gpu_ok and extraction_ok:
        print("\nSUCCESS: Basic extraction is working!")
        print("The memory optimizations should help with OCR processing.")
    else:
        print("\nSome issues detected. Check the errors above.")
    
    return gpu_ok and extraction_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
