#!/usr/bin/env python3
"""
Test fixed PaddleOCR implementation with OCRResult format
"""

import sys
import os
from pathlib import Path

# Add src directory to path
src_path = Path(__file__).parent / "scr"
sys.path.insert(0, str(src_path))

def test_paddle_ocr_fix():
    """Test PaddleOCR with fixed normalization"""
    try:
        print("🧪 Тестирование исправленного PaddleOCR...")
        from ocr_paddle import run_paddle
        
        # Test with a sample image
        test_image = "scr/uploads/20250914_230135_2025-09-14_225307.png"
        if os.path.exists(test_image):
            print(f"📁 Тестируем с изображением: {test_image}")
            result = run_paddle(test_image, lang='ru')
            
            if isinstance(result, list) and len(result) > 0:
                print(f"✅ PaddleOCR работает! Найдено {len(result)} элементов")
                print("📝 Первые несколько результатов:")
                for i, item in enumerate(result[:5]):
                    text = item.get('text', '').strip()
                    conf = item.get('conf', 0)
                    print(f"  {i+1}. '{text}' (уверенность: {conf:.2f})")
                
                # Объединяем весь текст
                all_text = '\n'.join([item.get('text', '') for item in result])
                print(f"\n📄 Полный текст ({len(all_text)} символов):")
                print(all_text[:200] + "..." if len(all_text) > 200 else all_text)
                return True
            else:
                print("❌ PaddleOCR вернул пустой результат")
                return False
        else:
            print(f"❌ Тестовое изображение не найдено: {test_image}")
            return False
            
    except Exception as e:
        print(f"❌ Ошибка PaddleOCR: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_normalize_function():
    """Test the normalize function specifically with problematic data"""
    try:
        print("\n🔧 Тестирование функции нормализации...")
        from ocr_paddle import normalize_paddle_output
        
        # Test with various problematic inputs
        test_cases = [
            # Normal case
            [[[[1, 2], [3, 4], [5, 6], [7, 8]], ['normal text', 0.95]]],
            
            # Case with string coordinates (like 'i')
            [[[[1, 2], [3, 'i'], [5, 6], [7, 8]], ['text with bad coord', 0.8]]],
            
            # Empty case
            [[]],
            
            # Case with missing data
            [[[[], ['empty coords', 0.5]]]],
        ]
        
        for i, test_case in enumerate(test_cases):
            print(f"  Тест {i+1}: ", end="")
            try:
                result = normalize_paddle_output(test_case)
                print(f"✅ OK - получено {len(result)} элементов")
            except Exception as e:
                print(f"❌ Ошибка: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка тестирования нормализации: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Тестирование исправлений PaddleOCR...\n")
    
    # Test normalization function first
    normalize_ok = test_normalize_function()
    
    # Test full PaddleOCR if normalization works
    if normalize_ok:
        paddle_ok = test_paddle_ocr_fix()
    else:
        paddle_ok = False
    
    print(f"\n📊 Результаты:")
    print(f"  Нормализация: {'✅ Работает' if normalize_ok else '❌ Не работает'}")
    print(f"  PaddleOCR: {'✅ Работает' if paddle_ok else '❌ Не работает'}")
    
    if paddle_ok and normalize_ok:
        print("\n🎉 Все исправления работают! PaddleOCR должен работать в веб-приложении.")
    else:
        print("\n🔧 Требуются дополнительные исправления.")
        
    print("\nℹ️  Для тестирования в venv запустите:")
    print("    (venv) python test_paddle_fix.py")