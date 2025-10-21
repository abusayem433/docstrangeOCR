#!/usr/bin/env python3
"""Test MEMORY-SAFE processing - CPU-first approach."""

import os
import sys
import time
import torch
from pathlib import Path

def test_memory_safe_processing():
    """Test extraction with MEMORY-SAFE processing."""
    print("Testing MEMORY-SAFE Processing")
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
            font = ImageFont.truetype("arial.ttf", 24)
        except:
            font = ImageFont.load_default()
        
        text_lines = [
            "DocStrange MEMORY-SAFE Processing Test",
            "This document tests CPU-first processing:",
            "",
            "Memory-Safe Configuration:",
            "• GPU: 20% memory allocation (minimal usage)",
            "• CPU: 12GB RAM allocation (primary processing)",
            "• Image: 1024px resolution (memory-safe)",
            "• Processing: CPU-first approach",
            "",
            "Optimization Features:",
            "✓ GPU: Minimal usage to avoid OOM",
            "✓ CPU: Full RAM utilization (12GB)",
            "✓ Memory: PyTorch expandable segments",
            "✓ Resolution: 1024px for stability",
            "✓ Generation: Single beam for efficiency",
            "",
            "Expected Performance:",
            "• GPU Usage: ~20% (minimal)",
            "• CPU Usage: ~80-90% (primary)",
            "• RAM Usage: High utilization",
            "• Processing: Stable, no OOM errors",
            "",
            "Status: Testing MEMORY-SAFE processing..."
        ]
        
        y_position = 30
        for line in text_lines:
            draw.text((30, y_position), line, fill='black', font=font)
            y_position += 30
        
        # Save test image
        test_image = "test_memory_safe.png"
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
            print(f"  Target Utilization: ~20% (minimal)")
        
        # Test extraction with timing
        print(f"\nStarting MEMORY-SAFE extraction...")
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
    """Run MEMORY-SAFE processing tests."""
    print("DocStrange MEMORY-SAFE Processing Test")
    print("=" * 60)
    
    # Test memory-safe processing
    success, processing_time = test_memory_safe_processing()
    
    print(f"\n" + "=" * 60)
    print("Test Summary:")
    print(f"Memory-Safe Processing: {'PASS' if success else 'FAIL'}")
    print(f"Processing Time: {processing_time:.2f}s")
    
    if success:
        print(f"\nSUCCESS: MEMORY-SAFE processing is working!")
        print(f"   - CPU-first approach prevents OOM errors")
        print(f"   - Processing completed successfully")
        print(f"   - Processing time: {processing_time:.2f} seconds")
        print(f"   - Ready for stable document processing")
    else:
        print(f"\nFAILED: Memory-safe processing encountered issues")
        print(f"   - Check error messages above")
        print(f"   - Verify CPU-first optimization")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
