#!/usr/bin/env python3
"""
Test OCRResult structure and methods 
"""

import sys
import os
sys.path.append(os.path.abspath('.'))

from scr.ocr_paddle import get_paddle_instance

def test_ocr_result_structure():
    """Test OCRResult object structure and available methods"""
    
    # Test image path
    test_image = "scr/uploads/20250914_231837_2025-09-14_225307.png"
    
    if not os.path.exists(test_image):
        print(f"❌ Image not found: {test_image}")
        return
    
    print("🔍 Testing OCRResult object structure...")
    
    # Get PaddleOCR instance
    ocr = get_paddle_instance(lang='ru')
    
    # Run OCR
    print(f"📷 Processing image: {test_image}")
    result = ocr.ocr(test_image)
    
    print(f"\n📊 Raw result type: {type(result)}")
    print(f"📊 Raw result length: {len(result) if result else 0}")
    
    if result and len(result) > 0:
        ocr_result = result[0]
        print(f"\n📊 OCRResult type: {type(ocr_result)}")
        print(f"📊 OCRResult class name: {type(ocr_result).__name__}")
        
        # Test dictionary-like behavior
        print(f"\n🔍 Testing dictionary-like behavior:")
        
        # Test keys method
        if hasattr(ocr_result, 'keys'):
            try:
                keys = list(ocr_result.keys())
                print(f"✅ Keys method exists. Keys: {keys}")
                
                # Try accessing some values
                for key in keys[:5]:  # Show first 5 keys
                    try:
                        value = ocr_result[key]
                        print(f"   ocr_result['{key}']: {type(value)}")
                        if isinstance(value, list) and len(value) > 0:
                            print(f"      List length: {len(value)}, first item: {value[0]}")
                        elif isinstance(value, str):
                            print(f"      String: '{value[:50]}{'...' if len(value) > 50 else ''}'")
                        else:
                            print(f"      Value: {value}")
                    except Exception as e:
                        print(f"   Error accessing '{key}': {e}")
                        
            except Exception as e:
                print(f"❌ Error with keys(): {e}")
        else:
            print("❌ No keys() method")
            
        # Test json method
        if hasattr(ocr_result, 'json'):
            try:
                json_data = ocr_result.json()
                print(f"\n✅ JSON method works, type: {type(json_data)}")
                if isinstance(json_data, dict):
                    print(f"   JSON dict keys: {list(json_data.keys())}")
                    # Look for text data
                    for key, value in json_data.items():
                        if 'text' in key.lower() or 'rec' in key.lower():
                            print(f"   Found text-related key '{key}': {type(value)}")
                            if isinstance(value, list) and len(value) > 0:
                                print(f"      List length: {len(value)}, sample: {value[:2]}")
                elif isinstance(json_data, list):
                    print(f"   JSON list length: {len(json_data)}")
                    if json_data:
                        print(f"   First JSON item: {json_data[0]}")
            except Exception as e:
                print(f"❌ JSON method error: {e}")
        else:
            print("❌ No json() method")
            
        # Test string representation to see internal structure
        try:
            str_repr = str(ocr_result)
            print(f"\n📝 String representation (first 300 chars):")
            print(f"   {str_repr[:300]}...")
        except:
            print("❌ String representation failed")

if __name__ == "__main__":
    test_ocr_result_structure()