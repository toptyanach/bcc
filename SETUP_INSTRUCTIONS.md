# Инструкции по настройке OCR системы

## Проблемы из JSON-отчета:

### 1. PaddleOCR - Отсутствует модуль 'paddle'
**Ошибка**: `No module named 'paddle'`

**Решение**:
```bash
pip install paddlepaddle-gpu==2.5.2
# или для CPU версии:
pip install paddlepaddle==2.5.2
```

### 2. Tesseract - Нет русского языка
**Ошибка**: `Error opening data file C:\Program Files\Tesseract-OCR/tessdata/ru.traineddata`

**Решение**:
1. Скачайте русские языковые данные:
   - Перейдите на: https://github.com/tesseract-ocr/tessdata/blob/main/rus.traineddata
   - Нажмите "Download" и сохраните файл `rus.traineddata`

2. Скопируйте файл в папку tessdata:
   ```
   C:\Program Files\Tesseract-OCR\tessdata\rus.traineddata
   ```

3. Или установите переменную окружения:
   ```bash
   set TESSDATA_PREFIX=C:\Program Files\Tesseract-OCR\tessdata
   ```

### 3. TrOCR - Неправильное распознавание русского текста
**Проблема**: TrOCR возвращает "THANK YOU FOR FULL" для русского текста

**Объяснение**:
- TrOCR предназначен в основном для английского текста
- Модель `microsoft/trocr-base-printed` обучена на английском языке
- Для русского языка лучше использовать PaddleOCR или Tesseract

**Рекомендации**:
- Для русских документов используйте PaddleOCR (после установки paddle)
- Для англо-русских документов используйте Tesseract с `rus+eng`

## Быстрое решение для тестирования:

Система уже настроена на автоопределение доступных языков в Tesseract. После установки `rus.traineddata` Tesseract будет работать с русским языком автоматически.

## Приоритет движков для русского языка:
1. **PaddleOCR** - лучший для русского (после установки paddle)
2. **Tesseract** - хорош для русского (после установки rus.traineddata)
3. **TrOCR** - только для английского или коротких фраз

## Команды для быстрой установки:

```bash
# Установка PaddlePaddle
pip install paddlepaddle==2.5.2

# Проверка Tesseract языков
python -c "import pytesseract; print(pytesseract.get_languages())"
```

После выполнения этих шагов все три OCR движка должны работать корректно.