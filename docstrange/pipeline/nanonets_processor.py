"""Neural Document Processor using Nanonets OCR for superior document understanding."""

import logging
import os
from typing import Optional
from pathlib import Path
from PIL import Image

logger = logging.getLogger(__name__)


class NanonetsDocumentProcessor:
    """Neural Document Processor using Nanonets OCR model."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize the Neural Document Processor with Nanonets OCR."""
        logger.info("Initializing Neural Document Processor with Nanonets OCR...")
        
        # Initialize models
        self._initialize_models(cache_dir)
        
        logger.info("Neural Document Processor initialized successfully")
    
    def _initialize_models(self, cache_dir: Optional[Path] = None):
        """Initialize Nanonets OCR model from local cache."""
        try:
            from transformers import AutoTokenizer, AutoProcessor, AutoModelForImageTextToText
            from .model_downloader import ModelDownloader
            
            # Get model downloader instance
            model_downloader = ModelDownloader(cache_dir)
            
            # Get the path to the locally cached Nanonets model
            model_path = model_downloader.get_model_path('nanonets-ocr')
            
            if model_path is None:
                raise RuntimeError(
                        "Failed to download Nanonets OCR model. "
                        "Please ensure you have sufficient disk space and internet connection."
                    )
            
            # The actual model files are in a subdirectory with the same name
            actual_model_path = model_path / "Nanonets-OCR-ss"
            
            if not actual_model_path.exists():
                raise RuntimeError(
                    f"Model files not found at expected path: {actual_model_path}"
                )
            
            logger.info(f"Loading Nanonets OCR model from local cache: {actual_model_path}")
            
            # Load model with CPU-optimized processing for powerful CPU
            self.model = AutoModelForImageTextToText.from_pretrained(
                str(actual_model_path), 
                torch_dtype="auto", 
                device_map="cpu",  # Use CPU for main processing to utilize powerful CPU
                local_files_only=True,  # Use only local files
                low_cpu_mem_usage=False,  # Allow full CPU memory usage
                max_memory={"cpu": "14GB"}  # Use most of your 16GB RAM
            )
            self.model.eval()
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(actual_model_path),
                local_files_only=True
            )
            self.processor = AutoProcessor.from_pretrained(
                str(actual_model_path),
                local_files_only=True
            )
            
            logger.info("Nanonets OCR model loaded successfully from local cache")
            
        except ImportError as e:
            logger.error(f"Transformers library not available: {e}")
            raise ImportError(
                "Transformers library is required for Nanonets OCR. "
                "Please install it: pip install transformers"
            )
        except Exception as e:
            logger.error(f"Failed to initialize Nanonets OCR model: {e}")
            raise
    
    def extract_text(self, image_path: str) -> str:
        """Extract text from image using Nanonets OCR."""
        try:
            if not os.path.exists(image_path):
                logger.error(f"Image file does not exist: {image_path}")
                return ""
            
            return self._extract_text_with_nanonets(image_path)
                
        except Exception as e:
            logger.error(f"Nanonets OCR extraction failed: {e}")
            return ""
    
    def extract_text_with_layout(self, image_path: str) -> str:
        """Extract text with layout awareness using Nanonets OCR.
        
        Note: Nanonets OCR already provides layout-aware extraction,
        so this method returns the same result as extract_text().
        """
        return self.extract_text(image_path)
    
    def _extract_text_with_nanonets(self, image_path: str, max_new_tokens: int = 4096) -> str:
        """Extract text using Nanonets OCR model optimized for powerful CPU."""
        try:
            import torch
            import gc
            import os
            
            prompt = """Extract the text from the above document as if you were reading it naturally. Return the tables in html format. Return the equations in LaTeX representation. If there is an image in the document and image caption is not present, add a small description of the image inside the <img></img> tag; otherwise, add the image caption inside <img></img>. Watermarks should be wrapped in brackets. Ex: <watermark>OFFICIAL COPY</watermark>. Page numbers should be wrapped in brackets. Ex: <page_number>14</page_number> or <page_number>9/22</page_number>. Prefer using ☐ and ☑ for check boxes."""
            
            # Optimize for CPU processing
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                # Reserve minimal GPU for other processes
                torch.cuda.set_per_process_memory_fraction(0.3)
            
            # Force garbage collection
            gc.collect()
            
            # Check if image exists
            if not os.path.exists(image_path):
                logger.error(f"Image file does not exist: {image_path}")
                return ""
            
            image = Image.open(image_path)
            
            # Keep higher resolution for better OCR quality with powerful CPU
            max_size = 2048  # Increased for better quality with CPU processing
            if max(image.size) > max_size:
                ratio = max_size / max(image.size)
                new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
                image = image.resize(new_size, Image.Resampling.LANCZOS)
                logger.info(f"Resized image to {new_size} for CPU-optimized processing")
            
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": [
                    {"type": "image", "image": f"file://{image_path}"},
                    {"type": "text", "text": prompt},
                ]},
            ]
            
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[text], images=[image], padding=True, return_tensors="pt")
            
            # Process on CPU to utilize powerful CPU
            inputs = inputs.to("cpu")
            logger.info(f"Processing on CPU to utilize powerful CPU resources")
            
            # Generate with CPU-optimized settings
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs, 
                    max_new_tokens=max_new_tokens, 
                    do_sample=True,  # Enable sampling for better quality
                    temperature=0.7,  # Balanced creativity
                    top_p=0.9,  # Nucleus sampling
                    top_k=50,  # Top-k sampling
                    repetition_penalty=1.1,  # Reduce repetition
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                    use_cache=True,
                    num_beams=2,  # Use beam search for better quality
                    early_stopping=True
                )
            
            # Decode results
            generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
            output_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            
            # Clean up intermediate variables
            del inputs, output_ids, generated_ids
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Force garbage collection
            gc.collect()
            
            logger.info(f"Successfully extracted {len(output_text[0])} characters")
            return output_text[0]
            
        except Exception as e:
            logger.error(f"Nanonets OCR extraction failed: {e}")
            # Cleanup on error
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            return ""
    
    def __del__(self):
        """Cleanup resources."""
        pass 