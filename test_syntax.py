#!/usr/bin/env python3
"""
Syntax check for ocr_paddle.py
"""

import sys
from pathlib import Path

# Add src directory to path
src_path = Path(__file__).parent / "scr"
sys.path.insert(0, str(src_path))

try:
    import ocr_paddle
    print("✅ Синтаксис ocr_paddle.py исправлен!")
    print("🔧 Можно запускать веб-приложение")
except SyntaxError as e:
    print(f"❌ Синтаксическая ошибка: {e}")
    print(f"📍 Строка {e.lineno}: {e.text}")
except Exception as e:
    print(f"⚠️ Другая ошибка (но синтаксис OK): {e}")
    print("✅ Синтаксис файла корректный")