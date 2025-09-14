# OCR Web Application

Веб-приложение для распознавания текста с поддержкой кириллицы. Сравнивает несколько OCR движков и выбирает лучший результат.

## Установка и запуск

### Требования
- Python 3.8+
- Tesseract OCR (для движка Tesseract)

### Развертывание проекта

1. Клонируйте репозиторий и перейдите в папку проекта

2. Создайте виртуальное окружение:
```bash
python -m venv venv
```

3. Активируйте виртуальное окружение:
```bash
# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

4. Установите зависимости:
```bash
pip install -r requirements.txt
```

5. Запустите приложение:
```bash
python working_web_app.py
```

Приложение будет доступно по адресу: http://localhost:5000

## Описание

Приложение позволяет загружать изображения и PDF документы для распознавания текста. Используются три OCR движка:

- **PaddleOCR** - хорошо работает с кириллицей, особенно с документами на русском языке
- **Tesseract** - классический OCR, быстрый но иногда менее точный
- **TrOCR** - нейросетевая модель, медленнее но может давать хорошие результаты на сложных текстах

После распознавания система автоматически извлекает ключевые данные: даты, ИНН, телефоны и другую структурированную информацию.

## Структура проекта

```
CASE 2/
├── scr/
│   ├── working_web_app.py    # Основное Flask приложение
│   ├── ocr_coordinator.py    # Координатор для управления OCR движками
│   ├── ocr_paddle.py        # Модуль PaddleOCR
│   ├── ocr_baseline.py      # Модуль Tesseract
│   ├── ocr_trocr.py         # Модуль TrOCR
│   └── templates/           # HTML шаблоны
├── data/samples/            # Примеры изображений для тестирования
└── requirements.txt         # Файл зависимостей
```

## API endpoints

### Веб-интерфейс
- `GET /` - главная страница с формой загрузки

### REST API

#### Сравнение OCR движков
```
POST /api/ocr/compare
```
Параметры:
- `file` - изображение или PDF файл
- `engines` - список движков для сравнения (опционально)
- `language` - язык распознавания (по умолчанию: ru)

Пример ответа:
```json
{
  "results": {
    "PaddleOCR": {
      "raw_text": "Распознанный текст документа",
      "extracted_fields": {
        "date": "14.09.2025",
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

#### Список доступных движков
```
GET /api/ocr/engines
```

## Использование через Python

```python
import requests

# Загрузка и обработка документа
with open('document.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:5000/api/ocr/compare',
        files={'file': f},
        data={'language': 'ru'}
    )
    result = response.json()
    
    # Вывод результатов
    for engine, data in result['results'].items():
        print(f"{engine}: уверенность {data['avg_confidence']:.2%}")
        print(f"Текст: {data['raw_text'][:100]}...")
```

## Известные особенности

### PaddleOCR
При первом запуске загружает модели (~300MB), это может занять время. Последующие запуски будут быстрее.

### Tesseract
Требует установки в системе. На Windows нужно добавить путь к tesseract.exe в переменную PATH.

Установка Tesseract:
```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr tesseract-ocr-rus

# Windows
# Скачать установщик с https://github.com/UB-Mannheim/tesseract/wiki
```

### TrOCR
Использует transformers модели, может работать медленно на CPU. Рекомендуется GPU для больших объемов.

## Проверка установки

После установки всех зависимостей проверьте работоспособность каждого движка:

```bash
# Проверка Tesseract
tesseract --version

# Проверка PaddleOCR (должен импортироваться без ошибок)
python -c "from paddleocr import PaddleOCR; print('PaddleOCR OK')"

# Проверка TrOCR
python -c "from transformers import TrOCRProcessor; print('TrOCR OK')"
```

Если все проверки прошли успешно, можете запускать приложение.

## Тестирование

Для проверки работы отдельных компонентов:

```bash
# Тест PaddleOCR
python test_paddle_fix.py

# Тест всех движков
python test_ocr_fix.py
```

## Решение проблем

### Ошибка импорта PaddleOCR
```bash
pip uninstall paddlepaddle paddleocr
pip install paddlepaddle==2.5.1
pip install paddleocr==2.7.0.3
```

### Tesseract не найден
Убедитесь что tesseract установлен и доступен из командной строки:
```bash
tesseract --version
```

### Нехватка памяти при использовании TrOCR
Можно уменьшить размер батча или использовать более легкую модель в файле `ocr_trocr.py`.

## Производительность

Примерная скорость обработки одной страницы A4:
- PaddleOCR: 2-3 секунды
- Tesseract: 1-2 секунды  
- TrOCR: 5-10 секунд

Для production рекомендуется использовать PaddleOCR как основной движок, с Tesseract как запасной вариант.

## Дополнительная настройка

### Переменные окружения
Если используется LLM для постобработки, установите:
```bash
export OPENAI_API_KEY=your_api_key_here
```

### Конфигурация портов
По умолчанию используется порт 5000. Для изменения отредактируйте `working_web_app.py`:
```python
app.run(host='0.0.0.0', port=8080, debug=False)
```

## Лицензия

MIT

---

Для быстрого старта после установки зависимостей просто запустите `python working_web_app.py`