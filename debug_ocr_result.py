#!/usr/bin/env python3
"""
Debug script to inspect PaddleOCR OCRResult object structure
"""

from scr.ocr_paddle import get_paddle_instance
import json

def debug_ocr_result():
    """Debug the OCRResult object structure"""
    
    # Test image path
    test_image = "scr/uploads/20250914_231837_2025-09-14_225307.png"
    
    print("🔍 Debugging PaddleOCR OCRResult structure...")
    
    # Get PaddleOCR instance
    ocr = get_paddle_instance(lang='ru')
    
    # Run OCR
    print(f"📷 Processing image: {test_image}")
    result = ocr.ocr(test_image)
    
    print(f"\n📊 Raw result type: {type(result)}")
    print(f"📊 Raw result length: {len(result) if result else 0}")
    
    if result and len(result) > 0:
        first_item = result[0]
        print(f"\n📊 First item type: {type(first_item)}")
        
        # Check if it's OCRResult object
        if hasattr(first_item, '__dict__'):
            print(f"\n🔍 OCRResult attributes:")
            for attr_name in dir(first_item):
                if not attr_name.startswith('_'):
                    try:
                        attr_value = getattr(first_item, attr_name)
                        if not callable(attr_value):
                            print(f"  {attr_name}: {type(attr_value)} = {attr_value}")
                    except Exception as e:
                        print(f"  {attr_name}: Error accessing - {e}")
        
        # Check specific attributes we're looking for
        expected_attrs = ['rec_texts', 'rec_scores', 'rec_polys']
        for attr in expected_attrs:
            if hasattr(first_item, attr):
                attr_value = getattr(first_item, attr)
                print(f"\n✅ {attr} found:")
                print(f"   Type: {type(attr_value)}")
                print(f"   Length: {len(attr_value) if hasattr(attr_value, '__len__') else 'N/A'}")
                if hasattr(attr_value, '__len__') and len(attr_value) > 0:
                    print(f"   First item: {attr_value[0] if len(attr_value) > 0 else 'Empty'}")
                    if len(attr_value) > 1:
                        print(f"   Second item: {attr_value[1]}")
            else:
                print(f"\n❌ {attr} NOT found")
        
        # Try to access as legacy format
        print(f"\n🔍 Checking legacy format compatibility...")
        try:
            if isinstance(first_item, list) and len(first_item) > 0:
                print(f"   Legacy format - list length: {len(first_item)}")
                for i, item in enumerate(first_item[:3]):  # Show first 3 items
                    print(f"   Item {i}: {type(item)} = {item}")
        except Exception as e:
            print(f"   Legacy format check failed: {e}")
            
    else:
        print("\n❌ No results returned from OCR")

if __name__ == "__main__":
    debug_ocr_result()