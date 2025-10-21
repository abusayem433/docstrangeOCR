#!/usr/bin/env python3
"""Test CPU-ONLY processing - complete GPU avoidance."""

import os
import sys
import time
import torch
from pathlib import Path

def test_cpu_only_processing():
    """Test extraction with CPU-ONLY processing."""
    print("Testing CPU-ONLY Processing")
    print("=" * 50)
    
    try:
        # Force CPU-only processing
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        
        from docstrange import DocumentExtractor
        
        # Create a test image for OCR processing
        from PIL import Image, ImageDraw, ImageFont
        
        # Create a test document image
        img = Image.new('RGB', (1000, 700), color='white')
        draw = ImageDraw.Draw(img)
        
        # Add some text to the image
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        text_lines = [
            "DocStrange CPU-ONLY Processing Test",
            "This document tests complete CPU processing:",
            "",
            "CPU-Only Configuration:",
            "• GPU: DISABLED (CUDA_VISIBLE_DEVICES='')",
            "• CPU: 12GB RAM allocation (primary processing)",
            "• Image: 1024px resolution (CPU-optimized)",
            "• Processing: 100% CPU-only approach",
            "",
            "CPU-Only Features:",
            "✓ GPU: Completely disabled",
            "✓ CPU: Full RAM utilization (12GB)",
            "✓ Memory: PyTorch CPU backend",
            "✓ Resolution: 1024px for CPU efficiency",
            "✓ Generation: CPU-optimized settings",
            "",
            "Expected Performance:",
            "• GPU Usage: 0% (disabled)",
            "• CPU Usage: ~90-95% (maximum)",
            "• RAM Usage: High utilization",
            "• Processing: Stable, no GPU OOM",
            "",
            "Status: Testing CPU-ONLY processing..."
        ]
        
        y_position = 25
        for line in text_lines:
            draw.text((25, y_position), line, fill='black', font=font)
            y_position += 25
        
        # Save test image
        test_image = "test_cpu_only.png"
        img.save(test_image)
        print(f"Created test image: {test_image}")
        
        # Check CUDA status
        print(f"\nCUDA Status:")
        print(f"  CUDA Available: {torch.cuda.is_available()}")
        print(f"  CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
        print(f"  Target: CPU-only processing")
        
        # Test extraction with timing
        print(f"\nStarting CPU-ONLY extraction...")
        start_time = time.time()
        
        extractor = DocumentExtractor()
        result = extractor.extract(test_image)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
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
    """Run CPU-ONLY processing tests."""
    print("DocStrange CPU-ONLY Processing Test")
    print("=" * 60)
    
    # Test CPU-only processing
    success, processing_time = test_cpu_only_processing()
    
    print(f"\n" + "=" * 60)
    print("Test Summary:")
    print(f"CPU-Only Processing: {'PASS' if success else 'FAIL'}")
    print(f"Processing Time: {processing_time:.2f}s")
    
    if success:
        print(f"\nSUCCESS: CPU-ONLY processing is working!")
        print(f"   - GPU completely disabled")
        print(f"   - Processing completed successfully")
        print(f"   - Processing time: {processing_time:.2f} seconds")
        print(f"   - Ready for stable CPU-only processing")
    else:
        print(f"\nFAILED: CPU-only processing encountered issues")
        print(f"   - Check error messages above")
        print(f"   - Verify CPU-only optimization")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
