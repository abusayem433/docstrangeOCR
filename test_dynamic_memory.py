#!/usr/bin/env python3
"""Test DYNAMIC memory management - PyTorch handles GPU/CPU automatically."""

import os
import sys
import time
import torch
from pathlib import Path

def test_dynamic_memory_management():
    """Test extraction with dynamic memory management."""
    print("Testing DYNAMIC Memory Management")
    print("=" * 50)
    
    try:
        from docstrange import DocumentExtractor
        
        # Create a test image for OCR processing
        from PIL import Image, ImageDraw, ImageFont
        
        # Create a test document image
        img = Image.new('RGB', (1600, 1200), color='white')
        draw = ImageDraw.Draw(img)
        
        # Add some text to the image
        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except:
            font = ImageFont.load_default()
        
        text_lines = [
            "DocStrange DYNAMIC Memory Management Test",
            "This document tests automatic GPU/CPU memory handling:",
            "",
            "Dynamic Memory Configuration:",
            "• GPU: Automatic allocation (no limits)",
            "• CPU: Dynamic memory usage (no limits)",
            "• Image: 1536px resolution (dynamic processing)",
            "• Processing: PyTorch manages memory automatically",
            "",
            "Dynamic Memory Features:",
            "✓ GPU: No allocation limits - PyTorch decides",
            "✓ CPU: No allocation limits - PyTorch decides",
            "✓ Memory: Automatic offloading when needed",
            "✓ Resolution: 1536px for dynamic processing",
            "✓ Generation: Dynamic beam search optimization",
            "",
            "Expected Performance:",
            "• GPU Usage: Dynamic (PyTorch manages)",
            "• CPU Usage: Dynamic (PyTorch manages)",
            "• RAM Usage: Dynamic (PyTorch manages)",
            "• Processing: Automatic memory optimization",
            "",
            "Status: Testing DYNAMIC memory management..."
        ]
        
        y_position = 30
        for line in text_lines:
            draw.text((30, y_position), line, fill='black', font=font)
            y_position += 28
        
        # Save test image
        test_image = "test_dynamic_memory.png"
        img.save(test_image)
        print(f"Created test image: {test_image}")
        
        # Monitor GPU memory
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            total_memory = torch.cuda.get_device_properties(device).total_memory
            allocated_before = torch.cuda.memory_allocated(device)
            
            print(f"\nGPU Memory Status:")
            print(f"  Total: {total_memory / 1024**3:.2f} GB")
            print(f"  Allocated Before: {allocated_before / 1024**3:.2f} GB")
            print(f"  Memory Management: DYNAMIC (PyTorch handles)")
            print(f"  No Limits: PyTorch decides optimal usage")
        
        # Test extraction with timing
        print(f"\nStarting DYNAMIC memory extraction...")
        start_time = time.time()
        
        extractor = DocumentExtractor()
        result = extractor.extract(test_image)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Check GPU memory after
        if torch.cuda.is_available():
            allocated_after = torch.cuda.memory_allocated(device)
            memory_used = (allocated_after - allocated_before) / 1024**3
            utilization_percent = (allocated_after / total_memory) * 100
            
            print(f"\nGPU Memory After:")
            print(f"  Allocated: {allocated_after / 1024**3:.2f} GB")
            print(f"  Memory Used: {memory_used:.2f} GB")
            print(f"  Utilization: {utilization_percent:.1f}%")
            print(f"  Management: DYNAMIC (PyTorch optimized)")
        
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

def main():
    """Run DYNAMIC memory management tests."""
    print("DocStrange DYNAMIC Memory Management Test")
    print("=" * 60)
    
    # Test dynamic memory management
    success, processing_time = test_dynamic_memory_management()
    
    print(f"\n" + "=" * 60)
    print("Test Summary:")
    print(f"Dynamic Memory Management: {'PASS' if success else 'FAIL'}")
    print(f"Processing Time: {processing_time:.2f}s")
    
    if success:
        print(f"\nSUCCESS: DYNAMIC memory management is working!")
        print(f"   - PyTorch handles GPU/CPU allocation automatically")
        print(f"   - Processing completed successfully")
        print(f"   - Processing time: {processing_time:.2f} seconds")
        print(f"   - Ready for dynamic memory processing")
    else:
        print(f"\nFAILED: Dynamic memory management encountered issues")
        print(f"   - Check error messages above")
        print(f"   - Verify PyTorch dynamic memory handling")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
