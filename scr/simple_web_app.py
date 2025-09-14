"""
Упрощенная версия Flask веб-приложения для OCR
Без внешних шаблонов - все HTML встроено
"""

from flask import Flask, request, jsonify
import os
import json
import tempfile
from datetime import datetime

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB

# Создаем папки
os.makedirs('uploads', exist_ok=True)
os.makedirs('results', exist_ok=True)

@app.route('/')
def index():
    """Главная страница с встроенным HTML"""
    html = """
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OCR 2.0 - Простой интерфейс</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
        .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .upload-area { border: 2px dashed #ccc; padding: 40px; text-align: center; border-radius: 8px; margin: 20px 0; }
        .upload-area:hover { border-color: #007bff; background: #f8f9ff; }
        input[type="file"] { margin: 20px 0; }
        select, button { padding: 10px; margin: 10px 5px; border: 1px solid #ddd; border-radius: 4px; }
        button { background: #007bff; color: white; cursor: pointer; }
        button:hover { background: #0056b3; }
        .result { margin-top: 30px; padding: 20px; background: #f8f9fa; border-radius: 8px; }
        .error { color: red; background: #fee; padding: 15px; border-radius: 8px; margin: 10px 0; }
        .success { color: green; background: #efe; padding: 15px; border-radius: 8px; margin: 10px 0; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 OCR 2.0 - Распознавание текста</h1>

        <div class="upload-area" id="uploadArea">
            <h3>📁 Загрузите файл для обработки</h3>
            <p>Поддерживаются: PNG, JPG, PDF (макс. 50MB)</p>
            <input type="file" id="fileInput" accept=".png,.jpg,.jpeg,.pdf,.bmp,.tiff" />
        </div>

        <div>
            <label>🔧 OCR Движок:</label>
            <select id="engine">
                <option value="mock">Тестовый режим (без OCR)</option>
            </select>
        </div>

        <button onclick="processFile()">🚀 Запустить обработку</button>

        <div id="result" class="result" style="display:none;">
            <h3>📊 Результаты:</h3>
            <div id="resultContent"></div>
        </div>
    </div>

    <script>
        function processFile() {
            const fileInput = document.getElementById('fileInput');
            const engine = document.getElementById('engine').value;
            const resultDiv = document.getElementById('result');
            const contentDiv = document.getElementById('resultContent');

            if (!fileInput.files[0]) {
                alert('Выберите файл!');
                return;
            }

            const formData = new FormData();
            formData.append('file', fileInput.files[0]);
            formData.append('engine', engine);

            contentDiv.innerHTML = '⏳ Обработка файла...';
            resultDiv.style.display = 'block';

            fetch('/api/ocr/process', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    contentDiv.innerHTML = `
                        <div class="success">✅ Файл успешно обработан!</div>
                        <h4>📝 Информация о файле:</h4>
                        <p><strong>Имя:</strong> ${fileInput.files[0].name}</p>
                        <p><strong>Размер:</strong> ${(fileInput.files[0].size / 1024 / 1024).toFixed(2)} MB</p>
                        <p><strong>Движок:</strong> ${engine}</p>
                        <p><strong>Время:</strong> ${new Date().toLocaleString()}</p>

                        <h4>📋 Результат (JSON):</h4>
                        <textarea rows="10" cols="80" readonly>${JSON.stringify(data.result, null, 2)}</textarea>
                    `;
                } else {
                    contentDiv.innerHTML = `<div class="error">❌ Ошибка: ${data.error}</div>`;
                }
            })
            .catch(error => {
                contentDiv.innerHTML = `<div class="error">❌ Ошибка сети: ${error.message}</div>`;
            });
        }
    </script>
</body>
</html>
    """
    return html

@app.route('/api/ocr/process', methods=['POST'])
def process_document():
    """Упрощенная обработка документа"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'Файл не найден'})

        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'Файл не выбран'})

        engine = request.form.get('engine', 'mock')

        # Сохраняем файл
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{file.filename}"
        filepath = os.path.join('uploads', filename)
        file.save(filepath)

        # Простая имитация OCR обработки
        result = {
            'engine': engine,
            'filename': file.filename,
            'file_size': os.path.getsize(filepath),
            'timestamp': timestamp,
            'status': 'processed',
            'raw_text': 'Пример распознанного текста (тестовый режим)',
            'extracted_fields': {
                'doc_type': 'тестовый документ',
                'processing_note': 'Для реальной обработки установите OCR движки'
            },
            'message': 'Файл успешно загружен и сохранен. Для полной обработки установите зависимости OCR.'
        }

        return jsonify({
            'success': True,
            'result': result
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/status')
def status():
    """Статус сервиса"""
    # Проверяем какие модули доступны
    available_modules = []

    try:
        import numpy
        available_modules.append('numpy')
    except ImportError:
        pass

    try:
        import pytesseract
        available_modules.append('tesseract')
    except ImportError:
        pass

    try:
        import paddleocr
        available_modules.append('paddleocr')
    except ImportError:
        pass

    return jsonify({
        'status': 'running',
        'available_modules': available_modules,
        'total_modules': len(available_modules),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/health')
def health():
    """Простая проверка здоровья"""
    return jsonify({
        'status': 'healthy',
        'message': 'OCR сервис работает',
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    print("🚀 Запуск упрощенного OCR веб-сервиса...")
    print("📁 Папка загрузок: uploads")
    print("🌐 Веб-интерфейс: http://localhost:5000")
    print("📋 API эндпоинты:")
    print("  GET  / - главная страница")
    print("  POST /api/ocr/process - обработка файла")
    print("  GET  /api/status - статус модулей")
    print("  GET  /health - проверка здоровья")
    print()

    app.run(debug=True, host='0.0.0.0', port=5000)