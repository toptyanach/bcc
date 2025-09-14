"""
Flask веб-приложение для OCR обработки документов
Обеспечивает REST API и веб-интерфейс для загрузки и обработки документов
"""

import os
import json
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_file
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
os.makedirs('templates', exist_ok=True)
os.makedirs('static', exist_ok=True)

# Инициализируем координатор OCR
ocr_coordinator = OCRCoordinator()

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf', 'bmp', 'tiff'}


def allowed_file(filename):
    """Проверка допустимых расширений файлов"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """Главная страница с формой загрузки"""
    return render_template('index.html')


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
        confidence_threshold = float(request.form.get('confidence_threshold', '0.5'))

        # Сохраняем файл
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)

        # Обрабатываем документ
        result = ocr_coordinator.process_document(
            image_path=filepath,
            engine=engine,
            language=language,
            use_llm=use_llm,
            confidence_threshold=confidence_threshold
        )

        # Сохраняем результат
        result_filename = f"result_{timestamp}.json"
        result_path = os.path.join(app.config['RESULTS_FOLDER'], result_filename)

        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        # Добавляем информацию о файлах в результат
        result['files'] = {
            'original': filepath,
            'result': result_path,
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

        # Сохраняем результат сравнения
        result_filename = f"comparison_{timestamp}.json"
        result_path = os.path.join(app.config['RESULTS_FOLDER'], result_filename)

        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(comparison_result, f, ensure_ascii=False, indent=2)

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


@app.route('/api/metrics/calculate', methods=['POST'])
def calculate_metrics():
    """Вычисление метрик качества OCR"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'JSON данные не найдены'
            }), 400

        reference_text = data.get('reference_text', '')
        hypothesis_text = data.get('hypothesis_text', '')

        if not reference_text or not hypothesis_text:
            return jsonify({
                'success': False,
                'error': 'Необходимы reference_text и hypothesis_text'
            }), 400

        # Вычисляем метрики
        metrics_result = ocr_coordinator.calculate_metrics(
            reference_text,
            hypothesis_text
        )

        return jsonify({
            'success': True,
            'metrics': metrics_result
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/results/<result_id>')
def get_result(result_id):
    """Получить сохраненный результат по ID"""
    try:
        result_path = os.path.join(app.config['RESULTS_FOLDER'], f"result_{result_id}.json")

        if not os.path.exists(result_path):
            return jsonify({
                'success': False,
                'error': 'Результат не найден'
            }), 404

        with open(result_path, 'r', encoding='utf-8') as f:
            result = json.load(f)

        return jsonify({
            'success': True,
            'result': result
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/results/<result_id>/download')
def download_result(result_id):
    """Скачать результат в JSON формате"""
    try:
        result_path = os.path.join(app.config['RESULTS_FOLDER'], f"result_{result_id}.json")

        if not os.path.exists(result_path):
            return jsonify({
                'success': False,
                'error': 'Результат не найден'
            }), 404

        return send_file(
            result_path,
            as_attachment=True,
            download_name=f"ocr_result_{result_id}.json"
        )

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


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


@app.errorhandler(500)
def internal_error(e):
    return jsonify({
        'success': False,
        'error': 'Внутренняя ошибка сервера'
    }), 500


if __name__ == '__main__':
    print("🚀 Запуск OCR веб-сервиса...")
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
    print("  POST /api/metrics/calculate - вычисление метрик")
    print("  GET  /health - проверка статуса")

    app.run(debug=True, host='0.0.0.0', port=5000)