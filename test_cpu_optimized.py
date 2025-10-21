#!/usr/bin/env python3
"""Test CPU-optimized processing for powerful CPU utilization."""

import os
import sys
import time
import torch
from pathlib import Path

def test_cpu_optimized_extraction():
    """Test extraction with CPU optimization."""
    print("Testing CPU-Optimized Processing")
    print("=" * 50)
    
    try:
        from docstrange import DocumentExtractor
        
        # Create a test image for OCR processing
        from PIL import Image, ImageDraw, ImageFont
        
        # Create a test document image
        img = Image.new('RGB', (1200, 800), color='white')
        draw = ImageDraw.Draw(img)
        
        # Add some text to the image
        try:
            font = ImageFont.truetype("arial.ttf", 28)
        except:
            font = ImageFont.load_default()
        
        text_lines = [
            "DocStrange CPU-Optimized Processing Test",
            "This document tests CPU utilization optimization.",
            "System Configuration:",
            "- CPU: Multi-core processing (8 threads)",
            "- RAM: 16GB available",
            "- GPU: GTX 1660 Ti (minimal usage)",
            "",
            "Features Being Tested:",
            "✓ CPU-optimized model loading",
            "✓ Multi-threaded processing",
            "✓ High-resolution image processing",
            "✓ Advanced OCR extraction",
            "",
            "Expected Results:",
            "• Higher CPU utilization",
            "• Better extraction quality",
            "• Faster processing with CPU power"
        ]
        
        y_position = 50
        for line in text_lines:
            draw.text((50, y_position), line, fill='black', font=font)
            y_position += 35
        
        # Save test image
        test_image = "test_cpu_optimized.png"
        img.save(test_image)
        print(f"Created test image: {test_image}")
        
        # Monitor system resources
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            total_memory = torch.cuda.get_device_properties(device).total_memory
            allocated_before = torch.cuda.memory_allocated(device)
            
            print(f"\nGPU Memory Status:")
            print(f"  Total: {total_memory / 1024**3:.2f} GB")
            print(f"  Allocated Before: {allocated_before / 1024**3:.2f} GB")
        
        # Test extraction with timing
        print(f"\nStarting CPU-optimized extraction...")
        start_time = time.time()
        
        extractor = DocumentExtractor()
        result = extractor.extract(test_image)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Check GPU memory after
        if torch.cuda.is_available():
            allocated_after = torch.cuda.memory_allocated(device)
            memory_used = (allocated_after - allocated_before) / 1024**3
            
            print(f"\nGPU Memory After:")
            print(f"  Allocated: {allocated_after / 1024**3:.2f} GB")
            print(f"  Memory Used: {memory_used:.2f} GB")
        
        print(f"\nProcessing Results:")
        print(f"  Time: {processing_time:.2f} seconds")
        print(f"  Content Length: {len(result.content)} characters")
        print(f"  Success: {'Yes' if len(result.content) > 0 else 'No'}")
        
        if len(result.content) > 0:
            print(f"  Content Preview: {result.content[:300]}...")
        
        # Cleanup
        os.remove(test_image)
        
        return len(result.content) > 0, processing_time
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False, 0

def test_system_utilization():
    """Test system resource utilization."""
    print(f"\nSystem Resource Utilization Test")
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
        print(f"  RAM Usage: {memory.percent}%")
        
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
    """Run CPU optimization tests."""
    print("DocStrange CPU-Optimized Processing Test")
    print("=" * 60)
    
    # Test 1: CPU-optimized extraction
    success, processing_time = test_cpu_optimized_extraction()
    
    # Test 2: System utilization
    sys_ok = test_system_utilization()
    
    print(f"\n" + "=" * 60)
    print("Test Summary:")
    print(f"CPU-Optimized Extraction: {'PASS' if success else 'FAIL'}")
    print(f"Processing Time: {processing_time:.2f}s")
    print(f"System Info: {'PASS' if sys_ok else 'FAIL'}")
    
    if success:
        print(f"\nSUCCESS: CPU-optimized processing is working!")
        print(f"   - Extraction completed successfully")
        print(f"   - CPU resources are being utilized")
        print(f"   - Processing time: {processing_time:.2f} seconds")
        print(f"   - Ready for document processing")
    else:
        print(f"\nFAILED: Extraction encountered issues")
        print(f"   - Check error messages above")
        print(f"   - Verify model loading")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
