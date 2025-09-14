#!/usr/bin/env python3
"""
Diagnostic test for PaddleOCR output structure
"""

import sys
import os
from pathlib import Path

# Add src directory to path
src_path = Path(__file__).parent / "scr"
sys.path.insert(0, str(src_path))

def test_paddle_debug():
    """Test PaddleOCR with debug output"""
    try:
        print("🔍 Диагностика PaddleOCR...")
        from ocr_paddle import run_paddle
        
        # Test with a sample image
        test_image = "scr/uploads/20250914_230630_2025-09-14_225307.png"
        if os.path.exists(test_image):
            print(f"📁 Тестируем с изображением: {test_image}")
            print("=" * 50)
            result = run_paddle(test_image, lang='ru')
            print("=" * 50)
            
            print(f"📊 Результат: тип={type(result)}, длина={len(result) if hasattr(result, '__len__') else 'N/A'}")
            
            if isinstance(result, list):
                print(f"📝 Первые 5 элементов результата:")
                for i, item in enumerate(result[:5]):
                    print(f"  {i+1}. {item}")
            else:
                print(f"📝 Результат (первые 200 символов): {str(result)[:200]}")
                
            return True
        else:
            print(f"❌ Тестовое изображение не найдено: {test_image}")
            return False
            
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Запуск диагностики PaddleOCR...")
    test_paddle_debug()
    print("\n🔧 Эта диагностика поможет понять структуру данных PaddleOCR")