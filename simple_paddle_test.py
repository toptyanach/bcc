#!/usr/bin/env python3
"""
Simple PaddleOCR test to see actual result structure
"""

from paddleocr import PaddleOCR
import os

def test_paddle():
    print("🔍 Testing PaddleOCR directly...")
    
    # Initialize PaddleOCR
    ocr = PaddleOCR(use_angle_cls=True, lang='ru')
    
    # Test image
    image_path = "scr/uploads/20250914_231837_2025-09-14_225307.png"
    
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return
        
    print(f"📷 Processing: {image_path}")
    
    # Run OCR
    result = ocr.ocr(image_path)
    
    print(f"\n📊 Result type: {type(result)}")
    print(f"📊 Result length: {len(result) if result else 0}")
    
    if result and len(result) > 0:
        first_page = result[0]
        print(f"\n📊 First page type: {type(first_page)}")
        
        if hasattr(first_page, '__dict__'):
            print("\n🔍 OCRResult object attributes:")
            attributes = [attr for attr in dir(first_page) if not attr.startswith('_')]
            for attr in attributes:
                try:
                    value = getattr(first_page, attr)
                    if not callable(value):
                        print(f"  {attr}: {type(value)}")
                        if hasattr(value, '__len__') and not isinstance(value, str):
                            print(f"    Length: {len(value)}")
                            if len(value) > 0:
                                print(f"    First item: {value[0]}")
                except:
                    print(f"  {attr}: <unable to access>")
        else:
            print(f"\n📊 First page length: {len(first_page) if hasattr(first_page, '__len__') else 'N/A'}")
            if hasattr(first_page, '__len__') and len(first_page) > 0:
                print(f"📊 First item in page: {type(first_page[0])}")
                if len(first_page) > 0:
                    print(f"   Content: {first_page[0]}")

if __name__ == "__main__":
    test_paddle()