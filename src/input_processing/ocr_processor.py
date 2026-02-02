# src/input_processing/ocr_processor.py
import pytesseract
from PIL import Image
from .schemas import CanonicalInput
import os
import platform
import shutil

class OCRProcessor:
    def __init__(self):
        """Initialize Tesseract OCR processor."""
        self.is_available = False
        
        # 1. Detect Operating System
        system_os = platform.system()
        
        if system_os == "Windows":
            # Windows Paths
            tesseract_paths = [
                r'C:\Program Files\Tesseract-OCR\tesseract.exe',
                r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
            ]
            for path in tesseract_paths:
                if os.path.exists(path):
                    pytesseract.pytesseract.tesseract_cmd = path
                    self.is_available = True
                    break
        else:
            # Linux (Docker/Cloud)
            # Use shutil to find 'tesseract' in the system PATH
            tesseract_cmd = shutil.which("tesseract")
            if tesseract_cmd:
                pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
                self.is_available = True
            else:
                # Fallback check
                if os.path.exists('/usr/bin/tesseract'):
                    pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
                    self.is_available = True

        # 2. Verify availability
        if self.is_available:
            try:
                # specific check to see if it runs
                pytesseract.get_tesseract_version()
                print("✅ Tesseract initialized successfully")
            except Exception as e:
                print(f"⚠️ Tesseract found but failed to run: {e}")
                self.is_available = False
        else:
            print("⚠️ Tesseract not found. OCR will be disabled.")

    def process(self, image_path: str) -> CanonicalInput:
        """Process image and return canonical input."""
        # Fail gracefully if Tesseract isn't loaded
        if not self.is_available:
            return CanonicalInput(
                input_type="image",
                raw_file_path=image_path,
                extracted_text="",
                original_extraction="",
                confidence_score=0.0,
                metadata={
                    "error": "OCR engine not installed on server.",
                    "engine": "tesseract"
                }
            )

        try:
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # --psm 6 assumes a single uniform block of text
            data = pytesseract.image_to_data(
                image, 
                output_type=pytesseract.Output.DICT,
                config='--psm 6'
            )
            
            texts = []
            confidences = []
            
            for i, conf in enumerate(data['conf']):
                conf_value = int(conf) if str(conf).lstrip('-').isdigit() else -1
                if conf_value > 0:
                    text = data['text'][i].strip()
                    if text:
                        texts.append(text)
                        confidences.append(conf_value / 100.0)
            
            if not texts:
                return CanonicalInput(
                    input_type="image",
                    raw_file_path=image_path,
                    extracted_text="",
                    original_extraction="",
                    confidence_score=0.0,
                    metadata={"error": "No text detected", "engine": "tesseract"}
                )
            
            combined_text = " ".join(texts)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            return CanonicalInput(
                input_type="image",
                raw_file_path=image_path,
                extracted_text=combined_text,
                original_extraction=combined_text,
                confidence_score=avg_confidence,
                metadata={
                    "num_words": len(texts),
                    "individual_confidences": confidences,
                    "engine": "tesseract"
                }
            )
            
        except Exception as e:
            return CanonicalInput(
                input_type="image",
                raw_file_path=image_path,
                extracted_text="",
                original_extraction="",
                confidence_score=0.0,
                metadata={"error": str(e), "engine": "tesseract"}
            )