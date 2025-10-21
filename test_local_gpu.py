#!/usr/bin/env python3
"""
Test script for DocStrange Local GPU Processing System
This script demonstrates the local GPU-only functionality
"""

import sys
import os

def test_gpu_availability():
    """Test GPU availability."""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        print(f"CUDA Available: {cuda_available}")
        print(f"PyTorch Version: {torch.__version__}")
        
        if cuda_available:
            print(f"GPU Count: {torch.cuda.device_count()}")
            print(f"Current GPU: {torch.cuda.current_device()}")
            print(f"GPU Name: {torch.cuda.get_device_name()}")
        else:
            print("No GPU detected - CPU-only PyTorch installation")
        
        return cuda_available
    except ImportError:
        print("PyTorch not installed")
        return False

def test_docstrange_import():
    """Test DocStrange import."""
    try:
        from docstrange import DocumentExtractor
        print("YES - DocStrange imported successfully")
        return True
    except ImportError as e:
        print(f"NO - Failed to import DocStrange: {e}")
        return False

def test_extractor_creation():
    """Test extractor creation with GPU requirement."""
    try:
        from docstrange import DocumentExtractor
        
        print("\nTesting extractor creation...")
        
        # This should fail if GPU is not available
        try:
            extractor = DocumentExtractor()
            print("YES - Extractor created successfully")
            print(f"Processing mode: {extractor.get_processing_mode()}")
            print(f"Cloud enabled: {extractor.is_cloud_enabled()}")
            return True
        except RuntimeError as e:
            print(f"NO - Extractor creation failed (expected): {e}")
            return False
            
    except Exception as e:
        print(f"NO - Unexpected error: {e}")
        return False

def test_cli_functionality():
    """Test CLI functionality."""
    try:
        from docstrange.cli import main
        print("YES - CLI imported successfully")
        return True
    except ImportError as e:
        print(f"NO - Failed to import CLI: {e}")
        return False

def test_web_interface():
    """Test web interface functionality."""
    try:
        from docstrange.web_app import run_web_app, check_gpu_availability
        print("YES - Web interface imported successfully")
        
        gpu_available = check_gpu_availability()
        print(f"GPU check from web app: {gpu_available}")
        
        return True
    except ImportError as e:
        print(f"NO - Failed to import web interface: {e}")
        return False

def main():
    """Main test function."""
    print("DocStrange Local GPU Processing System Test")
    print("=" * 50)
    
    # Test GPU availability
    print("\n1. Testing GPU Availability:")
    gpu_available = test_gpu_availability()
    
    # Test DocStrange import
    print("\n2. Testing DocStrange Import:")
    import_success = test_docstrange_import()
    
    # Test extractor creation
    print("\n3. Testing Extractor Creation:")
    extractor_success = test_extractor_creation()
    
    # Test CLI functionality
    print("\n4. Testing CLI Functionality:")
    cli_success = test_cli_functionality()
    
    # Test web interface
    print("\n5. Testing Web Interface:")
    web_success = test_web_interface()
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Summary:")
    print(f"GPU Available: {'YES' if gpu_available else 'NO'}")
    print(f"DocStrange Import: {'YES' if import_success else 'NO'}")
    print(f"Extractor Creation: {'YES' if extractor_success else 'NO'}")
    print(f"CLI Functionality: {'YES' if cli_success else 'NO'}")
    print(f"Web Interface: {'YES' if web_success else 'NO'}")
    
    if gpu_available:
        print("\nSystem is ready for local GPU processing!")
        print("You can now run:")
        print("  - docstrange web (for web interface)")
        print("  - docstrange document.pdf (for CLI processing)")
    else:
        print("\nGPU is required for local processing")
        print("To enable GPU processing:")
        print("  1. Install CUDA toolkit")
        print("  2. Install PyTorch with CUDA support:")
        print("     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        print("  3. Ensure you have a compatible NVIDIA GPU")

if __name__ == "__main__":
    main()
