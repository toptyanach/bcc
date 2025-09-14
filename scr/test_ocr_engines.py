"""
Test script to check individual OCR engines without dependencies conflicts
"""
import sys
import os
from pathlib import Path

# Test PaddleOCR first (most important fix)
def test_paddle_ocr():
    print("Testing PaddleOCR...")
    try:
        from ocr_paddle import run_paddle
        
        # Use a sample image
        test_image = Path("uploads/20250914_224416_1_page-0001_1.jpg")
        if not test_image.exists():
            print(f"❌ Test image not found: {test_image}")
            return False
            
        print("Initializing PaddleOCR...")
        result = run_paddle(str(test_image), lang='ru')
        
        if result:
            print(f"✅ PaddleOCR successful! Found {len(result)} text blocks")
            for i, item in enumerate(result[:3]):  # Show first 3 results
                print(f"  {i+1}. Text: '{item['text'][:50]}...' (conf: {item['conf']:.2f})")
            return True
        else:
            print("❌ PaddleOCR returned empty result")
            return False
            
    except Exception as e:
        print(f"❌ PaddleOCR failed: {e}")
        return False

# Test Tesseract
def test_tesseract():
    print("\nTesting Tesseract...")
    try:
        import pytesseract
        from PIL import Image
        
        test_image = Path("uploads/20250914_224416_1_page-0001_1.jpg")
        if not test_image.exists():
            print(f"❌ Test image not found: {test_image}")
            return False
            
        image = Image.open(test_image)
        text = pytesseract.image_to_string(image, lang='rus')
        
        if text.strip():
            print(f"✅ Tesseract successful! Extracted {len(text)} characters")
            print(f"  Sample: '{text[:100]}...'")
            return True
        else:
            print("❌ Tesseract returned empty result")
            return False
            
    except Exception as e:
        print(f"❌ Tesseract failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 OCR Engine Testing Suite")
    print("=" * 50)
    
    # Test PaddleOCR (main issue)
    paddle_ok = test_paddle_ocr()
    
    # Test Tesseract (reference)
    tesseract_ok = test_tesseract()
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    print(f"  PaddleOCR: {'✅ WORKING' if paddle_ok else '❌ FAILED'}")
    print(f"  Tesseract: {'✅ WORKING' if tesseract_ok else '❌ FAILED'}")
    
    if paddle_ok:
        print("\n🎉 PaddleOCR is now working! The model download issue has been fixed.")
    else:
        print("\n⚠️  PaddleOCR still has issues. Check the error messages above.")