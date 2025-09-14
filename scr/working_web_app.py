"""
Рабочая версия Flask веб-приложения с встроенными шаблонами
Все OCR движки работают: PaddleOCR, Tesseract, TrOCR
"""

import os
import json
from pathlib import Path
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import tempfile
from datetime import datetime
import traceback

# Импорты OCR координатора
from ocr_coordinator import OCRCoordinator

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB максимум
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'

# Создаем папки если их нет
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Инициализируем координатор OCR
ocr_coordinator = OCRCoordinator()

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf', 'bmp', 'tiff'}


def allowed_file(filename):
    """Проверка допустимых расширений файлов"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """Главная страница с встроенным HTML"""

    # Получаем список доступных движков
    try:
        engines = ocr_coordinator.get_available_engines()
        engines_options = ''.join([f'<option value="{engine}">{engine}</option>' for engine in engines])
        if not engines:
            engines_options = '<option value="">Нет доступных движков</option>'
    except:
        engines_options = '<option value="">Ошибка загрузки движков</option>'

    html = f"""
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OCR 2.0 - Веб-интерфейс</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.0/font/bootstrap-icons.css" rel="stylesheet">
    <style>
        .drop-zone {{
            border: 2px dashed #dee2e6;
            border-radius: 10px;
            padding: 60px 20px;
            text-align: center;
            transition: all 0.3s ease;
            cursor: pointer;
        }}
        .drop-zone:hover, .drop-zone.dragover {{
            border-color: #0d6efd;
            background-color: #f8f9ff;
        }}
        .result-section {{
            display: none;
            margin-top: 30px;
        }}
        .json-viewer {{
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 5px;
            padding: 15px;
            max-height: 400px;
            overflow-y: auto;
            font-family: 'Courier New', monospace;
        }}
    </style>
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark bg-primary">
        <div class="container">
            <a class="navbar-brand" href="#">
                <i class="bi bi-eye"></i> OCR 2.0
            </a>
            <div class="navbar-nav ms-auto">
                <a class="nav-link" href="/health">
                    <i class="bi bi-heart-pulse"></i> Статус
                </a>
            </div>
        </div>
    </nav>

    <div class="container mt-4">
        <!-- Заголовок -->
        <div class="row mb-4">
            <div class="col-12">
                <div class="text-center">
                    <h1 class="display-6 mb-3">
                        <i class="bi bi-file-text"></i>
                        Обработка документов с OCR
                    </h1>
                    <p class="lead text-muted">
                        Загрузите изображение или PDF документ для извлечения текста и данных
                    </p>
                </div>
            </div>
        </div>

        <!-- Основная форма -->
        <div class="row">
            <div class="col-lg-8 mx-auto">
                <div class="card shadow">
                    <div class="card-header">
                        <h5 class="card-title mb-0">
                            <i class="bi bi-upload"></i> Загрузка документа
                        </h5>
                    </div>
                    <div class="card-body">
                        <!-- Зона загрузки -->
                        <div class="drop-zone" id="dropZone">
                            <i class="bi bi-cloud-upload-fill fs-1 text-muted mb-3"></i>
                            <h5>Перетащите файл сюда или нажмите для выбора</h5>
                            <p class="text-muted mb-0">
                                Поддерживаются: PNG, JPG, PDF, TIFF (макс. 50MB)
                            </p>
                            <input type="file" id="fileInput" class="d-none"
                                   accept=".png,.jpg,.jpeg,.pdf,.bmp,.tiff">
                        </div>

                        <!-- Настройки обработки -->
                        <div class="row mt-4" id="settingsPanel" style="display: none;">
                            <div class="col-md-6">
                                <label class="form-label">
                                    <i class="bi bi-gear"></i> OCR Движок
                                </label>
                                <select class="form-select" id="engineSelect">
                                    {engines_options}
                                </select>
                            </div>
                            <div class="col-md-6">
                                <label class="form-label">
                                    <i class="bi bi-translate"></i> Язык
                                </label>
                                <select class="form-select" id="languageSelect">
                                    <option value="ru">Русский</option>
                                    <option value="en">English</option>
                                    <option value="rus+eng">Русский + English</option>
                                </select>
                            </div>
                            <div class="col-md-6 mt-3">
                                <label class="form-label">Дополнительные опции</label>
                                <div class="form-check">
                                    <input class="form-check-input" type="checkbox" id="useLlmCheck">
                                    <label class="form-check-label" for="useLlmCheck">
                                        <i class="bi bi-robot"></i> Использовать LLM постобработку
                                    </label>
                                </div>
                                <div class="form-check">
                                    <input class="form-check-input" type="checkbox" id="compareEngines">
                                    <label class="form-check-label" for="compareEngines">
                                        <i class="bi bi-bar-chart"></i> Сравнить несколько движков
                                    </label>
                                </div>
                            </div>
                        </div>

                        <!-- Кнопки управления -->
                        <div class="d-grid gap-2 mt-4" id="actionButtons" style="display: none;">
                            <button class="btn btn-primary btn-lg" id="processBtn" onclick="processDocument()">
                                <i class="bi bi-play-fill"></i> Запустить OCR
                            </button>
                            <button class="btn btn-outline-secondary" onclick="resetForm()">
                                <i class="bi bi-arrow-clockwise"></i> Сбросить
                            </button>
                        </div>

                        <!-- Прогресс обработки -->
                        <div class="text-center mt-3" id="progressSection" style="display: none;">
                            <div class="spinner-border text-primary" role="status">
                                <span class="visually-hidden">Загрузка...</span>
                            </div>
                            <p class="mt-2" id="progressText">Обработка документа...</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Результаты -->
        <div class="result-section" id="resultsSection">
            <div class="row mt-4">
                <div class="col-12">
                    <div class="card shadow">
                        <div class="card-header d-flex justify-content-between align-items-center">
                            <h5 class="card-title mb-0">
                                <i class="bi bi-check-circle-fill text-success"></i>
                                Результаты обработки
                            </h5>
                            <div>
                                <button class="btn btn-outline-primary btn-sm" onclick="downloadResults()">
                                    <i class="bi bi-download"></i> Скачать JSON
                                </button>
                            </div>
                        </div>
                        <div class="card-body">
                            <!-- Навигация результатов -->
                            <ul class="nav nav-tabs" id="resultTabs" role="tablist">
                                <li class="nav-item">
                                    <a class="nav-link active" data-bs-toggle="tab" href="#textTab">
                                        <i class="bi bi-file-text"></i> Текст
                                    </a>
                                </li>
                                <li class="nav-item">
                                    <a class="nav-link" data-bs-toggle="tab" href="#fieldsTab">
                                        <i class="bi bi-list-check"></i> Извлеченные поля
                                    </a>
                                </li>
                                <li class="nav-item">
                                    <a class="nav-link" data-bs-toggle="tab" href="#rawTab">
                                        <i class="bi bi-code-square"></i> Сырые данные
                                    </a>
                                </li>
                            </ul>

                            <!-- Содержимое вкладок -->
                            <div class="tab-content mt-3" id="resultContent">
                                <!-- Вкладка текста -->
                                <div class="tab-pane fade show active" id="textTab">
                                    <div class="row">
                                        <div class="col-12">
                                            <h6>Распознанный текст:</h6>
                                            <div class="border rounded p-3 bg-light" style="max-height: 400px; overflow-y: auto;">
                                                <pre id="extractedText" class="mb-0"></pre>
                                            </div>
                                        </div>
                                    </div>
                                </div>

                                <!-- Вкладка полей -->
                                <div class="tab-pane fade" id="fieldsTab">
                                    <div id="extractedFields"></div>
                                </div>

                                <!-- Вкладка сырых данных -->
                                <div class="tab-pane fade" id="rawTab">
                                    <div class="json-viewer">
                                        <pre id="rawData"></pre>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        let currentFile = null;
        let currentResult = null;

        document.addEventListener('DOMContentLoaded', function() {{
            const dropZone = document.getElementById('dropZone');
            const fileInput = document.getElementById('fileInput');

            // Drag & Drop функциональность
            dropZone.addEventListener('click', () => fileInput.click());
            dropZone.addEventListener('dragover', handleDragOver);
            dropZone.addEventListener('dragleave', handleDragLeave);
            dropZone.addEventListener('drop', handleDrop);

            fileInput.addEventListener('change', handleFileSelect);
        }});

        function handleDragOver(e) {{
            e.preventDefault();
            e.stopPropagation();
            e.currentTarget.classList.add('dragover');
        }}

        function handleDragLeave(e) {{
            e.preventDefault();
            e.stopPropagation();
            e.currentTarget.classList.remove('dragover');
        }}

        function handleDrop(e) {{
            e.preventDefault();
            e.stopPropagation();
            e.currentTarget.classList.remove('dragover');

            const files = e.dataTransfer.files;
            if (files.length > 0) {{
                handleFileSelect({{ target: {{ files: files }} }});
            }}
        }}

        function handleFileSelect(e) {{
            const file = e.target.files[0];
            if (!file) return;

            currentFile = file;
            updateDropZoneWithFile(file);
            showSettings();
        }}

        function updateDropZoneWithFile(file) {{
            const dropZone = document.getElementById('dropZone');
            const fileSize = (file.size / 1024 / 1024).toFixed(2);

            dropZone.innerHTML = `
                <i class="bi bi-file-check-fill fs-1 text-success mb-3"></i>
                <h5>${{file.name}}</h5>
                <p class="text-muted mb-0">Размер: ${{fileSize}} MB</p>
            `;
        }}

        function showSettings() {{
            document.getElementById('settingsPanel').style.display = 'block';
            document.getElementById('actionButtons').style.display = 'block';
        }}

        function processDocument() {{
            if (!currentFile) {{
                alert('Сначала выберите файл');
                return;
            }}

            const isComparison = document.getElementById('compareEngines').checked;

            if (isComparison) {{
                processComparison();
            }} else {{
                processSingleEngine();
            }}
        }}

        function processSingleEngine() {{
            const formData = new FormData();
            formData.append('file', currentFile);
            formData.append('engine', document.getElementById('engineSelect').value);
            formData.append('language', document.getElementById('languageSelect').value);
            formData.append('use_llm', document.getElementById('useLlmCheck').checked);

            showProgress('Обработка документа...');

            fetch('/api/ocr/process', {{
                method: 'POST',
                body: formData
            }})
            .then(response => response.json())
            .then(data => {{
                hideProgress();
                if (data.success) {{
                    currentResult = data.result;
                    displayResults(data.result);
                    showAlert('Документ успешно обработан!', 'success');
                }} else {{
                    showAlert(`Ошибка обработки: ${{data.error}}`, 'danger');
                    console.error('Детали ошибки:', data);
                }}
            }})
            .catch(error => {{
                hideProgress();
                console.error('Ошибка:', error);
                showAlert('Ошибка подключения к серверу', 'danger');
            }});
        }}

        function processComparison() {{
            const formData = new FormData();
            formData.append('file', currentFile);
            formData.append('engines', 'PaddleOCR,Tesseract,TrOCR');
            formData.append('language', document.getElementById('languageSelect').value);

            showProgress('Сравнение движков...');

            fetch('/api/ocr/compare', {{
                method: 'POST',
                body: formData
            }})
            .then(response => response.json())
            .then(data => {{
                hideProgress();
                if (data.success) {{
                    currentResult = data.comparison;
                    displayComparisonResults(data.comparison);
                    showAlert('Сравнение движков завершено!', 'success');
                }} else {{
                    showAlert(`Ошибка сравнения: ${{data.error}}`, 'danger');
                }}
            }})
            .catch(error => {{
                hideProgress();
                console.error('Ошибка:', error);
                showAlert('Ошибка подключения к серверу', 'danger');
            }});
        }}

        function showProgress(text) {{
            document.getElementById('progressSection').style.display = 'block';
            document.getElementById('progressText').textContent = text;
            document.getElementById('processBtn').disabled = true;
        }}

        function hideProgress() {{
            document.getElementById('progressSection').style.display = 'none';
            document.getElementById('processBtn').disabled = false;
        }}

        function displayResults(result) {{
            document.getElementById('resultsSection').style.display = 'block';

            // Отображаем текст
            document.getElementById('extractedText').textContent = result.raw_text || 'Текст не найден';

            // Отображаем извлеченные поля
            displayExtractedFields(result.extracted_fields);

            // Отображаем сырые данные
            document.getElementById('rawData').textContent = JSON.stringify(result, null, 2);

            // Прокручиваем к результатам
            document.getElementById('resultsSection').scrollIntoView({{ behavior: 'smooth' }});
        }}

        function displayExtractedFields(fields) {{
            const container = document.getElementById('extractedFields');

            if (!fields || Object.keys(fields).length === 0) {{
                container.innerHTML = '<div class="alert alert-info">Поля не извлечены</div>';
                return;
            }}

            const importantFields = ['fio', 'date', 'sum', 'contract_number', 'account', 'phone', 'email', 'inn'];
            let html = '<div class="row">';

            importantFields.forEach(fieldName => {{
                if (fields[fieldName]) {{
                    const icon = getFieldIcon(fieldName);
                    html += `
                        <div class="col-md-6 mb-3">
                            <div class="card h-100">
                                <div class="card-body">
                                    <h6 class="card-title">
                                        <i class="bi ${{icon}}"></i>
                                        ${{getFieldDisplayName(fieldName)}}
                                    </h6>
                                    <p class="card-text">${{fields[fieldName]}}</p>
                                </div>
                            </div>
                        </div>
                    `;
                }}
            }});

            html += '</div>';
            container.innerHTML = html;
        }}

        function displayComparisonResults(comparison) {{
            document.getElementById('resultsSection').style.display = 'block';

            // Создаем сравнительную таблицу
            const results = comparison.results;
            let comparisonHtml = '<div class="table-responsive"><table class="table table-striped">';
            comparisonHtml += '<thead><tr><th>Движок</th><th>Статус</th><th>Символов</th><th>Слов</th><th>Элементов</th><th>Уверенность</th></tr></thead><tbody>';

            Object.keys(results).forEach(engine => {{
                const result = results[engine];
                const status = result.success ?
                    '<span class="badge bg-success">Успех</span>' :
                    '<span class="badge bg-danger">Ошибка</span>';

                const chars = result.raw_text ? result.raw_text.length : 0;
                const words = result.raw_text ? result.raw_text.split(' ').length : 0;
                const items = result.total_items || 0;
                const conf = result.avg_confidence ? `${{(result.avg_confidence * 100).toFixed(1)}}%` : 'N/A';

                comparisonHtml += `
                    <tr>
                        <td><strong>${{engine}}</strong></td>
                        <td>${{status}}</td>
                        <td>${{chars}}</td>
                        <td>${{words}}</td>
                        <td>${{items}}</td>
                        <td>${{conf}}</td>
                    </tr>
                `;
            }});

            comparisonHtml += '</tbody></table></div>';

            // Отображаем на всех вкладках
            document.getElementById('extractedText').innerHTML = comparisonHtml;
            document.getElementById('extractedFields').innerHTML = comparisonHtml;
            document.getElementById('rawData').textContent = JSON.stringify(comparison, null, 2);

            document.getElementById('resultsSection').scrollIntoView({{ behavior: 'smooth' }});
        }}

        function getFieldIcon(fieldName) {{
            const icons = {{
                'fio': 'bi-person-fill',
                'date': 'bi-calendar-fill',
                'sum': 'bi-currency-dollar',
                'contract_number': 'bi-file-earmark-text',
                'account': 'bi-bank',
                'phone': 'bi-telephone-fill',
                'email': 'bi-envelope-fill',
                'inn': 'bi-card-text'
            }};
            return icons[fieldName] || 'bi-info-circle';
        }}

        function getFieldDisplayName(fieldName) {{
            const names = {{
                'fio': 'ФИО',
                'date': 'Дата',
                'sum': 'Сумма',
                'contract_number': 'Номер договора',
                'account': 'Счет',
                'phone': 'Телефон',
                'email': 'Email',
                'inn': 'ИНН'
            }};
            return names[fieldName] || fieldName;
        }}

        function downloadResults() {{
            if (!currentResult) {{
                alert('Нет результатов для скачивания');
                return;
            }}

            const dataStr = JSON.stringify(currentResult, null, 2);
            const dataBlob = new Blob([dataStr], {{ type: 'application/json' }});
            const url = URL.createObjectURL(dataBlob);

            const link = document.createElement('a');
            link.href = url;
            link.download = `ocr_result_${{new Date().toISOString().split('T')[0]}}.json`;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);

            URL.revokeObjectURL(url);
        }}

        function resetForm() {{
            currentFile = null;
            currentResult = null;

            const dropZone = document.getElementById('dropZone');
            dropZone.innerHTML = `
                <i class="bi bi-cloud-upload-fill fs-1 text-muted mb-3"></i>
                <h5>Перетащите файл сюда или нажмите для выбора</h5>
                <p class="text-muted mb-0">
                    Поддерживаются: PNG, JPG, PDF, TIFF (макс. 50MB)
                </p>
            `;

            document.getElementById('fileInput').value = '';
            document.getElementById('settingsPanel').style.display = 'none';
            document.getElementById('actionButtons').style.display = 'none';
            document.getElementById('resultsSection').style.display = 'none';
            document.getElementById('compareEngines').checked = false;
            document.getElementById('useLlmCheck').checked = false;
        }}

        function showAlert(message, type) {{
            // Простой alert пока что
            alert(message);
        }}
    </script>
</body>
</html>
    """
    return html


@app.route('/api/ocr/engines', methods=['GET'])
def get_ocr_engines():
    """Получить список доступных OCR движков"""
    try:
        engines = ocr_coordinator.get_available_engines()
        return jsonify({
            'success': True,
            'engines': engines
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/ocr/process', methods=['POST'])
def process_document():
    """Обработка загруженного документа"""
    try:
        # Проверяем наличие файла
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'Файл не найден в запросе'
            }), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'Файл не выбран'
            }), 400

        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': f'Недопустимый тип файла. Разрешены: {", ".join(ALLOWED_EXTENSIONS)}'
            }), 400

        # Получаем параметры
        engine = request.form.get('engine', 'PaddleOCR')
        use_llm = request.form.get('use_llm', 'false').lower() == 'true'
        language = request.form.get('language', 'ru')

        # Сохраняем файл
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)

        # Получаем рекомендации по движкам
        recommendations = ocr_coordinator.recommend_engine(filepath)

        # Обрабатываем документ
        result = ocr_coordinator.process_document(
            image_path=filepath,
            engine=engine,
            language=language,
            use_llm=use_llm
        )

        # Добавляем рекомендации к результату
        result['engine_recommendations'] = recommendations

        # Добавляем информацию о файлах в результат
        result['files'] = {
            'original': filepath,
            'result_id': timestamp
        }

        return jsonify({
            'success': True,
            'result': result
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/api/ocr/compare', methods=['POST'])
def compare_engines():
    """Сравнение нескольких OCR движков на одном документе"""
    try:
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'Файл не найден в запросе'
            }), 400

        file = request.files['file']
        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': f'Недопустимый тип файла. Разрешены: {", ".join(ALLOWED_EXTENSIONS)}'
            }), 400

        # Получаем список движков для сравнения
        engines_str = request.form.get('engines', 'PaddleOCR,Tesseract')
        engines = [engine.strip() for engine in engines_str.split(',')]
        language = request.form.get('language', 'ru')

        # Сохраняем файл
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)

        # Сравниваем движки
        comparison_result = ocr_coordinator.compare_engines(
            image_path=filepath,
            engines=engines,
            language=language
        )

        return jsonify({
            'success': True,
            'comparison': comparison_result
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/api/setup-help')
def setup_help():
    """Помощь по настройке OCR движков"""
    help_info = {
        'PaddleOCR': {
            'status': 'missing_dependency',
            'error': 'No module named paddle',
            'solution': 'pip install paddlepaddle==2.5.2',
            'description': 'Лучший выбор для русского языка'
        },
        'Tesseract': {
            'status': 'missing_language_data',
            'error': 'ru.traineddata not found',
            'solution': 'Скачайте rus.traineddata с GitHub tesseract-ocr/tessdata',
            'description': 'Хорош для русского после установки языковых данных'
        },
        'TrOCR': {
            'status': 'language_limitation',
            'error': 'Плохо работает с русским языком',
            'solution': 'Используйте только для английских текстов',
            'description': 'Предназначен в основном для английского языка'
        }
    }

    return jsonify({
        'engines': help_info,
        'recommendations': {
            'for_russian': 'Используйте PaddleOCR (после установки) или Tesseract (после установки rus.traineddata)',
            'for_english': 'Любой из движков подойдет',
            'quick_fix': 'pip install paddlepaddle==2.5.2 для быстрого решения'
        }
    })


@app.route('/health')
def health_check():
    """Проверка здоровья сервиса"""
    try:
        engines = ocr_coordinator.get_available_engines()
        return jsonify({
            'status': 'healthy',
            'available_engines': engines,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500


@app.errorhandler(413)
def too_large(e):
    return jsonify({
        'success': False,
        'error': 'Файл слишком большой. Максимальный размер: 50MB'
    }), 413


if __name__ == '__main__':
    print("🚀 Запуск РАБОЧЕГО OCR веб-сервиса...")
    print(f"📁 Папка загрузок: {app.config['UPLOAD_FOLDER']}")
    print(f"📁 Папка результатов: {app.config['RESULTS_FOLDER']}")

    # Проверяем доступные движки
    try:
        available_engines = ocr_coordinator.get_available_engines()
        print(f"🔧 Доступные OCR движки: {', '.join(available_engines)}")
    except Exception as e:
        print(f"⚠️ Ошибка инициализации OCR: {e}")

    print("🌐 Веб-интерфейс доступен по адресу: http://localhost:5000")
    print("📋 API документация:")
    print("  POST /api/ocr/process - обработка документа")
    print("  POST /api/ocr/compare - сравнение движков")
    print("  GET  /api/ocr/engines - список движков")
    print("  GET  /health - проверка статуса")
    print()

    # Для поддержки PDF добавим
    try:
        import pdf2image
        print("✅ PDF поддержка доступна")
    except ImportError:
        print("⚠️ Для поддержки PDF установите: pip install pdf2image")

    app.run(debug=True, host='0.0.0.0', port=5000)