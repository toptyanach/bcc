# OCR Web Application

Веб-приложение для сравнения OCR движков с поддержкой кириллицы.

## 🚀 Быстрый старт

```bash
# Запуск веб-приложения
python working_web_app.py
```

Затем откройте браузер: **http://localhost:5000**

## � Описание

Приложение сравнивает три OCR движка:
- **PaddleOCR** - высокое качество для кириллицы  
- **Tesseract** - быстрый базовый OCR
- **TrOCR** - AI-модель для текста

### Возможности
- 📤 Загрузка изображений (PNG, JPG) и PDF
- 🔍 Сравнение результатов OCR движков
- 📊 Метрики качества (CER, WER)
- 📝 Извлечение данных (даты, ИНН, телефоны)
- 🌐 REST API

## 🛠 Установка

```bash
# Создание виртуального окружения
python -m venv venv

# Активация (Windows)
venv\Scripts\activate

# Установка зависимостей
pip install -r requirements.txt
```

## 📁 Структура

```
CASE 2/
├── scr/
│   ├── working_web_app.py    # 🚀 Главное приложение
│   ├── ocr_coordinator.py    # Координатор OCR
│   ├── ocr_paddle.py        # PaddleOCR
│   ├── ocr_baseline.py      # Tesseract  
│   ├── ocr_trocr.py         # TrOCR
│   └── templates/           # HTML шаблоны
├── data/samples/            # Тестовые изображения
└── requirements.txt         # Зависимости
```

## 🌐 API

### Сравнение OCR
```http
POST /api/ocr/compare
Content-Type: multipart/form-data

- file: изображение или PDF
- engines: список движков (необязательно)
- language: язык (по умолчанию: ru)
```

### Список движков
```http
GET /api/ocr/engines
```

### Пример
```python
import requests

with open('document.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:5000/api/ocr/compare',
        files={'file': f},
        data={'language': 'ru'}
    )
    result = response.json()
```

## 📊 Результат

```json
{
  "results": {
    "PaddleOCR": {
      "raw_text": "Распознанный текст...",
      "extracted_fields": {
        "date": "2025-09-14",
        "inn": "1234567890"
      },
      "avg_confidence": 0.89
    }
  },
  "comparison_metrics": {
    "PaddleOCR_vs_Tesseract": {
      "cer": 0.15,
      "wer": 0.08
    }
  }
}
```

## 🚨 Известные проблемы

- **PaddleOCR**: Первый запуск медленный (загрузка моделей)
- **TrOCR**: Работает лучше с отдельными строками, чем с документами  
- **Tesseract**: Требует системную установку tesseract-ocr

## 🧪 Тестирование

```bash
# Тест PaddleOCR
python test_paddle_fix.py

# Тест всех движков
python test_ocr_fix.py
```

---

**Запуск:** `python working_web_app.py` → http://localhost:5000

```python
comparison = coordinator.compare_engines(
    image_path="contract.pdf",
    engines=["PaddleOCR", "Tesseract", "TrOCR"]
)

for engine, result in comparison['results'].items():
    print(f"{engine}: {result['avg_confidence']:.2f}")
```

### 3. Вычисление метрик качества:

```python
metrics = coordinator.calculate_metrics(
    reference_text="Иванов Иван Иванович",
    hypothesis_text="Иванов Иван Иваненич"
)

print(f"CER: {metrics['cer']:.3f}")
print(f"WER: {metrics['wer']:.3f}")
```

## 🚨 Устранение неполадок

### Проблема: PaddleOCR не работает
```bash
# Переустановка PaddleOCR
pip uninstall paddlepaddle paddleocr
pip install paddlepaddle==2.5.1 paddleocr==2.7.0.3
```

### Проблема: Tesseract не найден
```bash
# Linux
sudo apt-get update && sudo apt-get install tesseract-ocr

# Windows - добавить в PATH:
# C:\Program Files\Tesseract-OCR
```

### Проблема: TrOCR медленно работает
```bash
# Установка с GPU поддержкой
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Проблема: LLM не работает
```bash
# Проверка переменных окружения
echo $OPENAI_API_KEY

# Или использование mock режима
# В postprocess.py автоматически включается fallback
```

## 🤝 Интеграция с другими системами

### Использование как микросервис:

```yaml
# docker-compose.yml
version: '3.8'
services:
  ocr-service:
    build: .
    ports:
      - "5000:5000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    volumes:
      - ./uploads:/app/uploads
```

### Интеграция с CI/CD:

```yaml
# .github/workflows/test.yml
- name: Test OCR Service
  run: |
    python -m pytest scr/tests/
    curl -f http://localhost:5000/health
```

## 📈 Производительность

### Benchmarks (примерные):
- **PaddleOCR**: ~2-3 сек/страница, высокое качество для кириллицы
- **Tesseract**: ~1-2 сек/страница, быстрая обработка
- **TrOCR**: ~5-10 сек/страница, лучшее качество для сложных текстов
- **LLM обработка**: +1-3 сек для структурирования данных

### Рекомендации:
- Для продакшена используйте PaddleOCR + LLM постобработку
- Для быстрой обработки используйте Tesseract
- Для максимального качества комбинируйте несколько движков

## 📄 Лицензия

MIT License - см. файл LICENSE

## 👥 Авторы

Разработано для демонстрации современных подходов к OCR обработке документов.

---

**🎯 Готов к использованию! Запустите `python web_app.py` и откройте http://localhost:5000**