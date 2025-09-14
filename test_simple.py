#!/usr/bin/env python3
"""
Test script for working OCR engines only
"""

import sys
import os
from pathlib import Path

# Add src directory to path
src_path = Path(__file__).parent / "scr"
sys.path.insert(0, str(src_path))

def test_paddle_ocr():
    """Test PaddleOCR with fixed API call"""
    try:
        print("🧪 Тестирование PaddleOCR...")
        from ocr_paddle import run_paddle
        
        # Test with a sample image
        test_image = "scr/uploads/20250914_225421_2025-09-14_225307.png"
        if os.path.exists(test_image):
            result = run_paddle(test_image, lang='ru')
            print(f"✅ PaddleOCR работает! Результат длиной: {len(result)} символов")
            print(f"Первые 100 символов: {result[:100]}")
            return True
        else:
            print(f"❌ Тестовое изображение не найдено: {test_image}")
            return False
            
    except Exception as e:
        print(f"❌ Ошибка PaddleOCR: {e}")
        return False

def test_tesseract():
    """Test Tesseract as reference"""
    try:
        print("🧪 Тестирование Tesseract...")
        from ocr_baseline import run_tesseract
        
        # Test with a sample image
        test_image = "scr/uploads/20250914_225421_2025-09-14_225307.png"
        if os.path.exists(test_image):
            result = run_tesseract(test_image, lang='rus')
            print(f"✅ Tesseract работает! Результат длиной: {len(result)} символов")
            print(f"Первые 100 символов: {result[:100]}")
            return True
        else:
            print(f"❌ Тестовое изображение не найдено: {test_image}")
            return False
            
    except Exception as e:
        print(f"❌ Ошибка Tesseract: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Запуск тестов основных OCR движков...\n")
    
    results = {
        "Tesseract": test_tesseract(),
        "PaddleOCR": test_paddle_ocr()
    }
    
    print("\n📊 Результаты тестирования:")
    for engine, success in results.items():
        status = "✅ Работает" if success else "❌ Не работает"
        print(f"  {engine}: {status}")
    
    working_count = sum(results.values())
    total_count = len(results)
    
    print(f"\n🎯 Итого: {working_count}/{total_count} движков работают")
    
    if working_count == total_count:
        print("🎉 Все движки работают корректно!")
    elif working_count > 0:
        print("⚠️ Некоторые движки требуют дополнительной настройки")
    else:
        print("🔧 Все движки требуют исправления")