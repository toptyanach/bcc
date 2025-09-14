"""
TrOCR Wrapper Module
Alternative OCR solution for high-quality plain text extraction.
Used as fallback or in ensemble approaches when PaddleOCR struggles.
"""

import logging
from typing import Optional, List, Union
from pathlib import Path
import warnings

import torch
from PIL import Image
import numpy as np

try:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    TROCR_AVAILABLE = True
except ImportError:
    TROCR_AVAILABLE = False
    warnings.warn("transformers library not found. Install with: pip install transformers")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrOCRWrapper:
    """
    Wrapper for Microsoft's TrOCR model.
    Provides high-quality text extraction without bounding boxes.
    """
    
    def __init__(
        self,
        model_name: str = 'microsoft/trocr-base-printed',
        device: Optional[str] = None,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize TrOCR model and processor.
        
        Args:
            model_name: HuggingFace model identifier
                - 'microsoft/trocr-base-printed' for printed text
                - 'microsoft/trocr-base-handwritten' for handwritten text
                - 'microsoft/trocr-large-printed' for better quality (slower)
            device: 'cuda', 'cpu', or None (auto-detect)
            cache_dir: Directory for caching downloaded models
        """
        if not TROCR_AVAILABLE:
            raise ImportError("transformers library is required for TrOCR")
        
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.model = None
        self.processor = None
        
        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        logger.info(f"TrOCR will run on: {self.device}")
        
        # Initialize model
        self._load_model()
    
    def _load_model(self):
        """Load TrOCR model and processor."""
        try:
            logger.info(f"Loading TrOCR model: {self.model_name}")
            
            # Load processor
            self.processor = TrOCRProcessor.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            )
            
            # Load model
            self.model = VisionEncoderDecoderModel.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            )
            
            # Move model to device
            if self.device == 'cuda':
                self.model = self.model.cuda()
                # Enable mixed precision for faster inference
                self.model = self.model.half()
            
            # Set to evaluation mode
            self.model.eval()
            
            logger.info("TrOCR model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load TrOCR model: {e}")
            raise
    
    def process_image(self, image: Union[Image.Image, np.ndarray]) -> str:
        """
        Process single image and extract text.

        Args:
            image: PIL Image or numpy array

        Returns:
            Extracted text string
        """
        return self.extract_text(image)

    def extract_text(self, image: Union[Image.Image, np.ndarray]) -> str:
        """
        Extract text from image using TrOCR model.

        Args:
            image: PIL Image or numpy array

        Returns:
            Extracted text string
        """
        # Convert numpy array to PIL Image if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Check image size and warn if too complex
        width, height = image.size
        
        # For very large images, return a warning instead of trying to process
        if width > 1500 or height > 1500:
            logger.warning(f"Large document image detected ({width}x{height}). TrOCR is designed for single text lines, not full documents.")
            return "TrOCR: Изображение слишком большое для данной модели. TrOCR предназначен для распознавания отдельных строк текста, а не полных документов. Используйте PaddleOCR или Tesseract."
        
        if width > 1000 or height > 1000:
            logger.warning(f"Large image detected ({width}x{height}). TrOCR works best on cropped text regions.")

        try:
            # Process image
            with torch.no_grad():
                pixel_values = self.processor(image, return_tensors="pt").pixel_values

                if self.device == 'cuda':
                    pixel_values = pixel_values.cuda()

                # Generate text with more controlled parameters
                generated_ids = self.model.generate(
                    pixel_values,
                    max_length=256,  # Reduced limit for better quality
                    num_beams=4,     # Increased beam search for better quality
                    early_stopping=True,
                    do_sample=False  # Disable sampling for more deterministic results
                )
                generated_text = self.processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True
                )[0]

            # Enhanced validation - check if result looks like nonsense
            if generated_text and len(generated_text.strip()) > 0:
                # Clean the text
                generated_text = generated_text.strip()
                
                # Check for common hallucination patterns
                suspicious_patterns = [
                    'INVOICEBOOK', 'COMPERIENCE', 'AAAAAAA', 'XXXXXXX', 
                    'THANK YOU FOR FULL', 'GENERATED TEXT', 'NO TEXT FOUND',
                    'ABCDEFGH', 'IIIIIII', 'OOOOOOO', 'MMMMMMM'
                ]
                
                # Check if the text is mostly repeating characters
                if len(set(generated_text.replace(' ', ''))) < 3 and len(generated_text) > 5:
                    logger.warning(f"TrOCR generated repetitive text: {generated_text}")
                    return "TrOCR: Обнаружен повторяющийся результат. Возможно, изображение нечеткое или содержит много текста."
                
                # Check for suspicious patterns
                if any(pattern in generated_text.upper() for pattern in suspicious_patterns):
                    logger.warning(f"TrOCR generated suspicious text: {generated_text}")
                    return "TrOCR: Обнаружен подозрительный результат распознавания. Изображение может быть слишком сложным для данной модели."

                return generated_text
            else:
                return "TrOCR: Текст не распознан"

        except Exception as e:
            logger.error(f"TrOCR processing error: {e}")
            return f"TrOCR: Ошибка обработки - {str(e)}"
    
    def process_image_regions(self, image: Image.Image, regions: List[tuple]) -> List[str]:
        """
        Process multiple regions from an image.
        
        Args:
            image: PIL Image
            regions: List of (x1, y1, x2, y2) bounding boxes
            
        Returns:
            List of extracted text strings
        """
        results = []
        
        for region in regions:
            x1, y1, x2, y2 = region
            # Crop region
            cropped = image.crop((x1, y1, x2, y2))
            # Extract text
            text = self.process_image(cropped)
            results.append(text)
        
        return results
    
    def run(self, path: str) -> str:
        """
        Main method to extract text from an image file.

        Args:
            path: Path to image file

        Returns:
            Extracted text string
        """
        try:
            from pathlib import Path

            file_path = Path(path)

            # Проверяем расширение файла
            if file_path.suffix.lower() == '.pdf':
                # Для PDF файлов возвращаем заглушку
                logger.warning(f"PDF files not supported in TrOCR: {path}")
                return "TrOCR не поддерживает PDF файлы. Используйте изображения PNG/JPG."

            # Load image
            image = Image.open(path)

            # Add warning for complex documents
            width, height = image.size
            if width > 1500 or height > 1500:
                logger.warning(f"Large complex image ({width}x{height}). TrOCR is designed for single text lines.")
                return f"TrOCR: Изображение слишком большое ({width}x{height}). Данная модель предназначена для распознавания отдельных строк текста, а не полных документов. Рекомендуется использовать PaddleOCR или Tesseract для обработки сложных документов."

            # For medium-large images, warn but still try to process
            if width > 800 or height > 800:
                logger.info(f"Processing large image ({width}x{height}). TrOCR may not perform optimally on full documents.")

            # Process entire image
            text = self.extract_text(image)

            return text if text else "TrOCR не смог распознать текст в изображении"

        except Exception as e:
            logger.error(f"Error processing {path}: {e}")
            return f"Ошибка TrOCR: {str(e)}"


# Convenience functions for direct usage

def init_trocr(
    model_name: str = 'microsoft/trocr-base-printed',
    device: Optional[str] = None,
    cache_dir: Optional[str] = None,
    language: str = 'en'
) -> TrOCRWrapper:
    """
    Initialize TrOCR model.

    Args:
        model_name: Model identifier from HuggingFace
        device: 'cuda', 'cpu', or None (auto-detect)
        cache_dir: Cache directory for models
        language: Language hint for processing

    Returns:
        TrOCRWrapper instance
    """
    # Для русского языка пытаемся использовать multilingual модель
    if language == 'ru' and model_name == 'microsoft/trocr-base-printed':
        # Можно попробовать другие модели для русского
        logger.info("Russian language detected. Using base model with warning.")

    return TrOCRWrapper(model_name=model_name, device=device, cache_dir=cache_dir)


def run_trocr(
    path: str,
    model: Optional[TrOCRWrapper] = None,
    model_name: str = 'microsoft/trocr-base-printed',
    device: Optional[str] = None
) -> str:
    """
    Extract text from image using TrOCR.
    
    Args:
        path: Path to image file
        model: Existing TrOCRWrapper instance (optional)
        model_name: Model to use if creating new instance
        device: Device to use if creating new instance
        
    Returns:
        Extracted text string
    """
    # Use provided model or create new one
    if model is None:
        model = init_trocr(model_name=model_name, device=device)
    
    return model.run(path)


# Ensemble helper function
def ensemble_with_paddle(
    path: str,
    paddle_result: Optional[str] = None,
    confidence_threshold: float = 0.8
) -> str:
    """
    Combine TrOCR with PaddleOCR results for better accuracy.
    
    Args:
        path: Path to image
        paddle_result: Text from PaddleOCR (optional)
        confidence_threshold: Threshold to decide when to use TrOCR
        
    Returns:
        Best text result
    """
    # This is a simplified example
    # In practice, you'd want more sophisticated ensemble logic
    
    trocr_text = run_trocr(path)
    
    if paddle_result is None:
        return trocr_text
    
    # Simple heuristic: use longer text or combine
    if len(trocr_text) > len(paddle_result) * 1.2:
        return trocr_text
    elif len(paddle_result) > len(trocr_text) * 1.2:
        return paddle_result
    else:
        # Could implement more sophisticated merging
        return trocr_text  # Default to TrOCR for quality


# Batch processing for efficiency
class BatchTrOCR:
    """
    Batch processing wrapper for TrOCR to improve throughput.
    """
    
    def __init__(
        self,
        model_name: str = 'microsoft/trocr-base-printed',
        device: Optional[str] = None,
        batch_size: int = 8
    ):
        """
        Initialize batch processor.
        
        Args:
            model_name: TrOCR model to use
            device: Device for processing
            batch_size: Number of images to process at once
        """
        self.wrapper = TrOCRWrapper(model_name=model_name, device=device)
        self.batch_size = batch_size
    
    def process_batch(self, image_paths: List[str]) -> List[str]:
        """
        Process multiple images in batches.
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            List of extracted text strings
        """
        results = []
        
        for i in range(0, len(image_paths), self.batch_size):
            batch_paths = image_paths[i:i + self.batch_size]
            batch_images = []
            
            # Load batch
            for path in batch_paths:
                try:
                    img = Image.open(path).convert('RGB')
                    batch_images.append(img)
                except Exception as e:
                    logger.error(f"Error loading {path}: {e}")
                    batch_images.append(None)
            
            # Process batch
            for img in batch_images:
                if img is not None:
                    text = self.wrapper.process_image(img)
                    results.append(text)
                else:
                    results.append("")
        
        return results


# Performance monitoring
class TrOCRWithMetrics(TrOCRWrapper):
    """
    TrOCR wrapper with performance metrics.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.total_processed = 0
        self.total_time = 0
        self.avg_time = 0
    
    def run(self, path: str) -> str:
        """Run with timing metrics."""
        import time
        
        start = time.time()
        result = super().run(path)
        elapsed = time.time() - start
        
        self.total_processed += 1
        self.total_time += elapsed
        self.avg_time = self.total_time / self.total_processed
        
        logger.info(f"Processed in {elapsed:.2f}s (avg: {self.avg_time:.2f}s)")
        
        return result


# Usage example and testing
if __name__ == "__main__":
    # Example usage
    print("TrOCR Wrapper Module")
    print("-" * 50)
    
    # Check if transformers is available
    if not TROCR_AVAILABLE:
        print("Please install transformers: pip install transformers")
        exit(1)
    
    # Initialize model
    print("Initializing TrOCR model...")
    trocr = init_trocr(
        model_name='microsoft/trocr-base-printed',
        device='cpu'  # Use 'cuda' if available
    )
    
    # Example: Process single image
    # text = run_trocr("path/to/image.jpg", model=trocr)
    # print(f"Extracted text: {text}")
    
    # Example: Batch processing
    # batch_processor = BatchTrOCR(device='cuda', batch_size=16)
    # texts = batch_processor.process_batch(["img1.jpg", "img2.jpg", "img3.jpg"])
    
    print("\nAvailable models:")
    print("- microsoft/trocr-base-printed (default, fast)")
    print("- microsoft/trocr-base-handwritten (for handwritten text)")
    print("- microsoft/trocr-large-printed (best quality, slower)")
    print("- microsoft/trocr-small-printed (fastest, lower quality)")
    
    print("\nKey features:")
    print("✓ High-quality text extraction")
    print("✓ No bounding boxes (text only)")
    print("✓ Good for ensemble with PaddleOCR")
    print("✓ Supports batch processing")
    print("✓ CPU/GPU support with auto-detection")
    print("✗ Slower than PaddleOCR on CPU")
    print("✗ Requires more memory")