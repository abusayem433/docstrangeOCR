"""Main extractor class for handling document conversion."""

import os
import logging
from typing import List, Optional

from .processors import (
    PDFProcessor,
    DOCXProcessor,
    TXTProcessor,
    ExcelProcessor,
    URLProcessor,
    HTMLProcessor,
    PPTXProcessor,
    ImageProcessor,
    GPUProcessor,
)
from .result import ConversionResult
from .exceptions import ConversionError, UnsupportedFormatError, FileNotFoundError
from .utils.gpu_utils import should_use_gpu_processor

# Configure logging
logger = logging.getLogger(__name__)


class DocumentExtractor:
    """Main class for converting documents to LLM-ready formats."""
    
    def __init__(
        self,
        preserve_layout: bool = True,
        include_images: bool = True,
        ocr_enabled: bool = True,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        gpu: bool = True
    ):
        """Initialize the file extractor for LOCAL GPU processing only.
        
        Args:
            preserve_layout: Whether to preserve document layout
            include_images: Whether to include images in output
            ocr_enabled: Whether to enable OCR for image and PDF processing
            api_key: DEPRECATED - No longer used (local processing only)
            model: DEPRECATED - No longer used (local processing only)
            gpu: Force local GPU processing (default: True)
        
        Note:
            - LOCAL GPU processing is now the default and only mode
            - All processing happens locally on your GPU
            - No cloud dependencies or API calls
        """
        self.preserve_layout = preserve_layout
        self.include_images = include_images
        self.api_key = None  # Disabled for local-only mode
        self.model = None    # Disabled for local-only mode
        self.gpu = True      # Always use GPU for local processing
        
        # Force local processing mode
        self.cloud_mode = False
        
        # Check GPU availability - required for local processing
        if not should_use_gpu_processor():
            raise RuntimeError(
                "GPU is required for local processing but not available. "
                "Please ensure CUDA is installed and a compatible GPU is present."
            )
        
        # Default to True if not explicitly set
        if ocr_enabled is None:
            self.ocr_enabled = True
        else:
            self.ocr_enabled = ocr_enabled
        
        # Initialize local processors only
        self.processors = []
        logger.info("Local GPU processing mode enabled - no cloud dependencies")
        self._setup_local_processors()
    
    def _setup_local_processors(self):
        """Setup local processors for GPU processing only."""
        local_processors = [
            PDFProcessor(preserve_layout=self.preserve_layout, include_images=self.include_images, ocr_enabled=self.ocr_enabled),
            DOCXProcessor(preserve_layout=self.preserve_layout, include_images=self.include_images),
            TXTProcessor(preserve_layout=self.preserve_layout, include_images=self.include_images),
            ExcelProcessor(preserve_layout=self.preserve_layout, include_images=self.include_images),
            HTMLProcessor(preserve_layout=self.preserve_layout, include_images=self.include_images),
            PPTXProcessor(preserve_layout=self.preserve_layout, include_images=self.include_images),
            ImageProcessor(preserve_layout=self.preserve_layout, include_images=self.include_images, ocr_enabled=self.ocr_enabled),
            URLProcessor(preserve_layout=self.preserve_layout, include_images=self.include_images),
        ]
        
        # Always add GPU processor for enhanced OCR capabilities
        logger.info("Adding GPU processor with Nanonets OCR for enhanced processing")
        gpu_processor = GPUProcessor(preserve_layout=self.preserve_layout, include_images=self.include_images, ocr_enabled=self.ocr_enabled)
        local_processors.append(gpu_processor)
        
        self.processors.extend(local_processors)
    
    def extract(self, file_path: str) -> ConversionResult:
        """Convert a file to internal format.
        
        Args:
            file_path: Path to the file to extract
            
        Returns:
            ConversionResult containing the processed content
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            UnsupportedFormatError: If the format is not supported
            ConversionError: If conversion fails
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Find the appropriate processor
        processor = self._get_processor(file_path)
        if not processor:
            raise UnsupportedFormatError(f"No processor found for file: {file_path}")
        
        logger.info(f"Using processor {processor.__class__.__name__} for {file_path}")
        
        # Process the file
        return processor.process(file_path)
    
    def convert_with_output_type(self, file_path: str, output_type: str) -> ConversionResult:
        """Convert a file with specific output type using local GPU processing.
        
        Args:
            file_path: Path to the file to extract
            output_type: Desired output type (markdown, flat-json, html)
            
        Returns:
            ConversionResult containing the processed content
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            UnsupportedFormatError: If the format is not supported
            ConversionError: If conversion fails
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Use local GPU processing for all conversions
        logger.info(f"Using local GPU processing with output_type={output_type} for {file_path}")
        return self.extract(file_path)
    
    def extract_url(self, url: str) -> ConversionResult:
        """Convert a URL to internal format using local processing.
        
        Args:
            url: URL to extract
            
        Returns:
            ConversionResult containing the processed content
            
        Raises:
            ConversionError: If conversion fails
        """
        # Find the URL processor
        url_processor = None
        for processor in self.processors:
            if isinstance(processor, URLProcessor):
                url_processor = processor
                break
        
        if not url_processor:
            raise ConversionError("URL processor not available")
        
        logger.info(f"Converting URL using local processing: {url}")
        return url_processor.process(url)
    
    def extract_text(self, text: str) -> ConversionResult:
        """Convert plain text to internal format using local processing.
        
        Args:
            text: Plain text to extract
            
        Returns:
            ConversionResult containing the processed content
        """
        metadata = {
            "content_type": "text",
            "processor": "TextConverter",
            "preserve_layout": self.preserve_layout,
            "processing_mode": "local_gpu"
        }
        
        return ConversionResult(text, metadata)
    
    def is_cloud_enabled(self) -> bool:
        """Check if cloud processing is enabled and configured.
        
        Returns:
            False - Local GPU processing only
        """
        return False
    
    def get_processing_mode(self) -> str:
        """Get the current processing mode.
        
        Returns:
            String describing the current processing mode
        """
        return "local_gpu"
    
    def _get_processor(self, file_path: str):
        """Get the appropriate processor for the file, prioritizing GPU processing.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Processor that can handle the file, or None if none found
        """
        # Define GPU-supported formats
        gpu_supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.gif', '.pdf']
        
        # Check file extension
        _, ext = os.path.splitext(file_path.lower())
        
        # Always try GPU processor first for supported formats
        if ext in gpu_supported_formats:
            for processor in self.processors:
                if isinstance(processor, GPUProcessor):
                    logger.info(f"Using GPU processor with Nanonets OCR for {file_path}")
                    return processor
        
        # Fallback to other processors for non-GPU formats
        for processor in self.processors:
            if processor.can_process(file_path):
                # Skip GPU processor in fallback mode to avoid infinite loops
                if isinstance(processor, GPUProcessor):
                    continue
                logger.info(f"Using {processor.__class__.__name__} for {file_path}")
                return processor
        return None
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported file formats.
        
        Returns:
            List of supported file extensions
        """
        formats = []
        for processor in self.processors:
            if hasattr(processor, 'can_process'):
                # This is a simplified way to get formats
                # In a real implementation, you might want to store this info
                if isinstance(processor, PDFProcessor):
                    formats.extend(['.pdf'])
                elif isinstance(processor, DOCXProcessor):
                    formats.extend(['.docx', '.doc'])
                elif isinstance(processor, TXTProcessor):
                    formats.extend(['.txt', '.text'])
                elif isinstance(processor, ExcelProcessor):
                    formats.extend(['.xlsx', '.xls', '.csv'])
                elif isinstance(processor, HTMLProcessor):
                    formats.extend(['.html', '.htm'])
                elif isinstance(processor, PPTXProcessor):
                    formats.extend(['.ppt', '.pptx'])
                elif isinstance(processor, ImageProcessor):
                    formats.extend(['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.gif'])
                elif isinstance(processor, URLProcessor):
                    formats.append('URLs')
                # Cloud processor removed - local processing only
                elif isinstance(processor, GPUProcessor):
                    # GPU processor supports all image formats and PDFs
                    formats.extend(['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.gif', '.pdf'])
        
        return list(set(formats))  # Remove duplicates 