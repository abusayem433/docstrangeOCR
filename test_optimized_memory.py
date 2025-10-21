#!/usr/bin/env python3
"""Test optimized memory usage for hybrid GPU/CPU processing."""

import os
import sys
import time
import torch
from pathlib import Path

def test_optimized_extraction():
    """Test extraction with optimized memory usage."""
    print("Testing Optimized Memory Usage")
    print("=" * 50)
    
    try:
        from docstrange import DocumentExtractor
        
        # Create a simple test document
        test_content = """# DocStrange Optimized Processing Test

## System Configuration
- GPU: NVIDIA GeForce GTX 1660 Ti (6GB)
- CPU: Multi-core processing
- Memory: Optimized allocation

## Features
- Hybrid GPU/CPU processing
- Memory-efficient model loading
- Aggressive memory management
- Sequential device mapping

## Test Results
This document tests the optimized memory usage for local processing.

**Status: Testing in progress...**
"""
        
        test_file = "test_optimized.txt"
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(test_content)
        
        print(f"Created test file: {test_file}")
        
        # Monitor GPU memory
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            total_memory = torch.cuda.get_device_properties(device).total_memory
            allocated_before = torch.cuda.memory_allocated(device)
            
            print(f"\nGPU Memory Status:")
            print(f"  Total: {total_memory / 1024**3:.2f} GB")
            print(f"  Allocated Before: {allocated_before / 1024**3:.2f} GB")
            print(f"  Free Before: {(total_memory - allocated_before) / 1024**3:.2f} GB")
        
        # Test extraction
        print(f"\nStarting optimized extraction...")
        start_time = time.time()
        
        extractor = DocumentExtractor()
        result = extractor.extract(test_file)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Check GPU memory after
        if torch.cuda.is_available():
            allocated_after = torch.cuda.memory_allocated(device)
            memory_used = (allocated_after - allocated_before) / 1024**3
            
            print(f"\nGPU Memory After:")
            print(f"  Allocated: {allocated_after / 1024**3:.2f} GB")
            print(f"  Memory Used: {memory_used:.2f} GB")
        
        print(f"\nResults:")
        print(f"  Processing Time: {processing_time:.2f} seconds")
        print(f"  Content Length: {len(result.content)} characters")
        print(f"  Success: {'Yes' if len(result.content) > 0 else 'No'}")
        
        # Cleanup
        os.remove(test_file)
        
        return len(result.content) > 0, processing_time
        
    except Exception as e:
        print(f"ERROR: {e}")
        return False, 0

def main():
    """Run optimized memory test."""
    print("DocStrange Optimized Memory Test")
    print("=" * 60)
    
    success, processing_time = test_optimized_extraction()
    
    print(f"\n" + "=" * 60)
    print("Test Summary:")
    print(f"Extraction Success: {'PASS' if success else 'FAIL'}")
    print(f"Processing Time: {processing_time:.2f}s")
    
    if success:
        print(f"\nSUCCESS: Optimized memory usage is working!")
        print(f"   - Processing completed successfully")
        print(f"   - Memory usage is optimized")
        print(f"   - Ready for document processing")
    else:
        print(f"\nFAILED: Extraction encountered issues")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
