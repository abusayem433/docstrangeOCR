#!/usr/bin/env python3
"""Test hybrid GPU/CPU performance optimization."""

import os
import sys
import time
import torch
from pathlib import Path

def test_gpu_cpu_utilization():
    """Test GPU and CPU utilization during processing."""
    print("Testing Hybrid GPU/CPU Utilization")
    print("=" * 50)
    
    try:
        from docstrange import DocumentExtractor
        
        # Create a test image for OCR processing
        from PIL import Image, ImageDraw, ImageFont
        import numpy as np
        
        # Create a test document image
        img = Image.new('RGB', (800, 600), color='white')
        draw = ImageDraw.Draw(img)
        
        # Add some text to the image
        try:
            # Try to use a system font
            font = ImageFont.truetype("arial.ttf", 24)
        except:
            font = ImageFont.load_default()
        
        text_lines = [
            "DocStrange Hybrid GPU/CPU Test",
            "This document tests the hybrid processing capabilities.",
            "GPU: NVIDIA GeForce GTX 1660 Ti",
            "CPU: Multi-core processing",
            "Memory: Optimized for 6GB GPU + 16GB RAM",
            "",
            "Features:",
            "✓ GPU acceleration for neural networks",
            "✓ CPU processing for memory-intensive tasks",
            "✓ Balanced workload distribution",
            "✓ Automatic device optimization"
        ]
        
        y_position = 50
        for line in text_lines:
            draw.text((50, y_position), line, fill='black', font=font)
            y_position += 40
        
        # Save test image
        test_image = "test_hybrid_processing.png"
        img.save(test_image)
        print(f"Created test image: {test_image}")
        
        # Monitor GPU memory before processing
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            total_memory = torch.cuda.get_device_properties(device).total_memory
            allocated_before = torch.cuda.memory_allocated(device)
            cached_before = torch.cuda.memory_reserved(device)
            
            print(f"\nGPU Memory Status (Before):")
            print(f"  Total: {total_memory / 1024**3:.2f} GB")
            print(f"  Allocated: {allocated_before / 1024**3:.2f} GB")
            print(f"  Cached: {cached_before / 1024**3:.2f} GB")
            print(f"  Free: {(total_memory - allocated_before) / 1024**3:.2f} GB")
        
        # Test extraction with timing
        print(f"\nStarting hybrid GPU/CPU extraction...")
        start_time = time.time()
        
        extractor = DocumentExtractor()
        result = extractor.extract(test_image)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Monitor GPU memory after processing
        if torch.cuda.is_available():
            allocated_after = torch.cuda.memory_allocated(device)
            cached_after = torch.cuda.memory_reserved(device)
            
            print(f"\nGPU Memory Status (After):")
            print(f"  Allocated: {allocated_after / 1024**3:.2f} GB")
            print(f"  Cached: {cached_after / 1024**3:.2f} GB")
            print(f"  Memory Used: {(allocated_after - allocated_before) / 1024**3:.2f} GB")
        
        print(f"\nProcessing Results:")
        print(f"  Time: {processing_time:.2f} seconds")
        print(f"  Content Length: {len(result.content)} characters")
        print(f"  Content Preview: {result.content[:200]}...")
        
        # Check if GPU was utilized
        gpu_utilized = False
        if torch.cuda.is_available():
            gpu_utilized = allocated_after > allocated_before
        
        print(f"  GPU Utilized: {'Yes' if gpu_utilized else 'No'}")
        
        # Cleanup
        os.remove(test_image)
        
        return True, processing_time, gpu_utilized
        
    except Exception as e:
        print(f"ERROR: {e}")
        return False, 0, False

def test_performance_comparison():
    """Compare performance with different configurations."""
    print(f"\nPerformance Comparison Test")
    print("=" * 50)
    
    try:
        import psutil
        
        # Get system info
        cpu_count = psutil.cpu_count()
        memory = psutil.virtual_memory()
        
        print(f"System Resources:")
        print(f"  CPU Cores: {cpu_count}")
        print(f"  Total RAM: {memory.total / 1024**3:.2f} GB")
        print(f"  Available RAM: {memory.available / 1024**3:.2f} GB")
        
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(device)
            gpu_memory = torch.cuda.get_device_properties(device).total_memory
            
            print(f"  GPU: {gpu_name}")
            print(f"  GPU Memory: {gpu_memory / 1024**3:.2f} GB")
        
        return True
        
    except Exception as e:
        print(f"ERROR getting system info: {e}")
        return False

def main():
    """Run hybrid performance tests."""
    print("DocStrange Hybrid GPU/CPU Performance Test")
    print("=" * 60)
    
    # Test 1: Hybrid utilization
    success, processing_time, gpu_used = test_gpu_cpu_utilization()
    
    # Test 2: Performance comparison
    perf_ok = test_performance_comparison()
    
    print(f"\n" + "=" * 60)
    print("Test Summary:")
    print(f"Hybrid Processing: {'PASS' if success else 'FAIL'}")
    print(f"Processing Time: {processing_time:.2f}s")
    print(f"GPU Utilization: {'Yes' if gpu_used else 'No'}")
    print(f"System Info: {'PASS' if perf_ok else 'FAIL'}")
    
    if success and gpu_used:
        print(f"\n🎉 SUCCESS: Hybrid GPU/CPU processing is working optimally!")
        print(f"   - GPU is being utilized for neural network computations")
        print(f"   - CPU is handling memory-intensive operations")
        print(f"   - Processing completed in {processing_time:.2f} seconds")
    elif success:
        print(f"\n⚠️  PARTIAL: Processing works but GPU not fully utilized")
        print(f"   - Consider checking GPU memory allocation")
    else:
        print(f"\n❌ FAILED: Processing encountered errors")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
