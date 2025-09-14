#!/usr/bin/env python3
"""
Simple debug test for PaddleOCR raw output
"""

import sys
import os
from pathlib import Path

# Add src directory to path
src_path = Path(__file__).parent / "scr"
sys.path.insert(0, str(src_path))

def debug_paddle_raw():
    """Debug PaddleOCR raw output directly"""
    try:
        print("🔍 Отладка сырого вывода PaddleOCR...")
        from ocr_paddle import get_paddle_instance
        
        # Test with a sample image
        test_image = "scr/uploads/20250914_230933_2025-09-14_225307.png"
        if not os.path.exists(test_image):
            print(f"❌ Тестовое изображение не найдено: {test_image}")
            return False
            
        print(f"📁 Тестируем с изображением: {test_image}")
        
        # Get PaddleOCR instance
        ocr = get_paddle_instance(lang='ru')
        
        # Get raw output directly
        print("🔧 Получаем сырой вывод PaddleOCR...")
        raw_result = ocr.ocr(str(test_image))
        
        print("=" * 60)
        print(f"📊 Сырой результат:")
        print(f"   Тип: {type(raw_result)}")
        print(f"   Длина: {len(raw_result) if hasattr(raw_result, '__len__') else 'N/A'}")
        print(f"   Содержимое: {repr(raw_result)}")
        print("=" * 60)
        
        if raw_result and len(raw_result) > 0:
            print(f"📊 Первый элемент:")
            print(f"   Тип: {type(raw_result[0])}")
            print(f"   Длина: {len(raw_result[0]) if hasattr(raw_result[0], '__len__') else 'N/A'}")
            
            if hasattr(raw_result[0], '__iter__') and not isinstance(raw_result[0], str):
                print(f"   Первые 3 элемента:")
                try:
                    for i, item in enumerate(list(raw_result[0])[:3]):
                        print(f"     {i}: {repr(item)}")
                except Exception as e:
                    print(f"     Ошибка итерации: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Запуск отладки сырого PaddleOCR...")
    debug_paddle_raw()
    print("\n🔧 Эта отладка покажет точную структуру вывода PaddleOCR")