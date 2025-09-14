"""
OCR Coordinator - единый интерфейс для всех OCR движков
Обеспечивает интеграцию PaddleOCR, Tesseract, TrOCR и постобработки
"""

import json
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from PIL import Image
import tempfile

# Импортируем OCR модули с обработкой ошибок
try:
    from ocr_paddle import run_paddle, get_plaintext
    PADDLE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ PaddleOCR недоступен: {e}")
    PADDLE_AVAILABLE = False

try:
    from ocr_baseline import run_tesseract, run_tesseract_with_data
    TESSERACT_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Tesseract недоступен: {e}")
    TESSERACT_AVAILABLE = False

try:
    from ocr_trocr import run_trocr
    TROCR_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ TrOCR недоступен: {e}")
    TROCR_AVAILABLE = False

try:
    from extract import extract_fields_from_paddle
    EXTRACT_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Extract модуль недоступен: {e}")
    EXTRACT_AVAILABLE = False

try:
    from metrics import cer, wer, normalized_levenshtein, field_metrics
    METRICS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Metrics модуль недоступен: {e}")
    METRICS_AVAILABLE = False


class OCRCoordinator:
    """Координатор для всех OCR движков"""

    def __init__(self):
        """Инициализация координатора"""
        self.engines = {
            'PaddleOCR': {
                'available': PADDLE_AVAILABLE,
                'function': self._run_paddle_ocr,
                'description': 'PaddleOCR - высокое качество для кириллицы'
            },
            'Tesseract': {
                'available': TESSERACT_AVAILABLE,
                'function': self._run_tesseract_ocr,
                'description': 'Tesseract - быстрый базовый OCR'
            },
            'TrOCR': {
                'available': TROCR_AVAILABLE,
                'function': self._run_trocr_ocr,
                'description': 'TrOCR - AI-модель для сложных текстов'
            }
        }

    def get_available_engines(self) -> List[str]:
        """Получить список доступных OCR движков"""
        return [name for name, config in self.engines.items() if config['available']]

    def get_engine_info(self) -> Dict[str, Dict]:
        """Получить подробную информацию о движках"""
        return {
            name: {
                'available': config['available'],
                'description': config['description']
            }
            for name, config in self.engines.items()
        }

    def _run_paddle_ocr(self, image_path: str, **kwargs) -> Dict[str, Any]:
        """Запуск PaddleOCR"""
        if not PADDLE_AVAILABLE:
            raise RuntimeError("PaddleOCR не установлен")

        language = kwargs.get('language', 'ru')

        # Запускаем OCR
        ocr_output = run_paddle(image_path, lang=language)
        raw_text = get_plaintext(ocr_output)

        # Извлекаем поля если модуль доступен
        extracted_fields = {}
        if EXTRACT_AVAILABLE:
            try:
                extracted_fields = extract_fields_from_paddle(ocr_output)
            except Exception as e:
                print(f"Ошибка извлечения полей: {e}")

        return {
            'engine': 'PaddleOCR',
            'raw_text': raw_text,
            'ocr_data': ocr_output,
            'extracted_fields': extracted_fields,
            'total_items': len(ocr_output) if ocr_output else 0,
            'avg_confidence': sum(item.get('conf', 0) for item in ocr_output) / len(ocr_output) if ocr_output else 0
        }

    def _run_tesseract_ocr(self, image_path: str, **kwargs) -> Dict[str, Any]:
        """Запуск Tesseract OCR"""
        if not TESSERACT_AVAILABLE:
            raise RuntimeError("Tesseract не установлен")

        language = kwargs.get('language', 'eng')  # Изменен на английский по умолчанию пока не установлен русский

        # Запускаем OCR с автоопределением языка
        raw_text = run_tesseract(image_path, lang=None)  # None = автоопределение
        ocr_data = run_tesseract_with_data(image_path, lang=None)

        # Простое извлечение полей для Tesseract
        extracted_fields = self._extract_fields_simple(raw_text)

        return {
            'engine': 'Tesseract',
            'raw_text': raw_text,
            'ocr_data': ocr_data,
            'extracted_fields': extracted_fields,
            'total_items': len(ocr_data) if ocr_data else 0,
            'avg_confidence': sum(item.get('conf', 0) for item in ocr_data) / len(ocr_data) if ocr_data else 0
        }

    def _run_trocr_ocr(self, image_path: str, **kwargs) -> Dict[str, Any]:
        """Запуск TrOCR"""
        if not TROCR_AVAILABLE:
            raise RuntimeError("TrOCR не установлен")

        # Запускаем OCR
        raw_text = run_trocr(image_path)

        # TrOCR возвращает только текст, создаем простую структуру
        extracted_fields = self._extract_fields_simple(raw_text)

        return {
            'engine': 'TrOCR',
            'raw_text': raw_text,
            'ocr_data': [],
            'extracted_fields': extracted_fields,
            'total_items': 1,
            'avg_confidence': 0.85  # TrOCR не предоставляет confidence
        }

    def _extract_fields_simple(self, text: str) -> Dict[str, Any]:
        """Простое извлечение полей с помощью регулярных выражений"""
        import re
        from datetime import datetime

        fields = {}

        # Даты (простой поиск)
        date_patterns = [
            r'\b(\d{1,2})[.\-/](\d{1,2})[.\-/](\d{4})\b',
            r'\b(\d{4})[.\-/](\d{1,2})[.\-/](\d{1,2})\b'
        ]

        for pattern in date_patterns:
            matches = re.findall(pattern, text)
            if matches:
                try:
                    if len(matches[0][0]) == 4:  # YYYY-MM-DD
                        date_obj = datetime(int(matches[0][0]), int(matches[0][1]), int(matches[0][2]))
                    else:  # DD.MM.YYYY
                        date_obj = datetime(int(matches[0][2]), int(matches[0][1]), int(matches[0][0]))
                    fields['date'] = date_obj.strftime('%Y-%m-%d')
                    break
                except ValueError:
                    continue

        # Суммы
        sum_pattern = r'(\d{1,3}(?:[\s,]\d{3})*(?:[.,]\d{1,2})?)\s*(?:руб|рублей|р\.|₽)'
        sum_matches = re.findall(sum_pattern, text, re.IGNORECASE)
        if sum_matches:
            try:
                sum_str = sum_matches[0].replace(' ', '').replace(',', '.')
                fields['sum'] = float(sum_str)
            except ValueError:
                pass

        # Телефоны
        phone_pattern = r'(?:\+7|8|7)?[\s\-]?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}'
        phone_matches = re.findall(phone_pattern, text)
        if phone_matches:
            phone = re.sub(r'[^\d+]', '', phone_matches[0])
            if len(phone) >= 10:
                fields['phone'] = phone

        # Email
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        email_matches = re.findall(email_pattern, text, re.IGNORECASE)
        if email_matches:
            fields['email'] = email_matches[0].lower()

        # ИНН
        inn_pattern = r'\b\d{10}\b|\b\d{12}\b'
        inn_matches = re.findall(inn_pattern, text)
        if inn_matches:
            fields['inn'] = inn_matches[0]

        fields['raw_text'] = text
        fields['total_chars'] = len(text)
        fields['total_words'] = len(text.split())

        return fields

    def recommend_engine(self, image_path: str) -> Dict[str, Any]:
        """
        Рекомендация OCR движка на основе характеристик изображения

        Args:
            image_path: Путь к изображению

        Returns:
            Словарь с рекомендацией и обоснованием
        """
        try:
            from PIL import Image

            # Анализируем изображение
            image = Image.open(image_path)
            width, height = image.size
            total_pixels = width * height
            aspect_ratio = width / height if height > 0 else 1

            file_path = Path(image_path)
            file_size = file_path.stat().st_size

            recommendation = {
                'image_stats': {
                    'width': width,
                    'height': height,
                    'total_pixels': total_pixels,
                    'aspect_ratio': round(aspect_ratio, 2),
                    'file_size_mb': round(file_size / 1024 / 1024, 2)
                },
                'recommendations': []
            }

            # Логика рекомендаций
            if total_pixels > 2000000:  # Большие изображения
                recommendation['recommendations'].append({
                    'engine': 'PaddleOCR',
                    'priority': 1,
                    'reason': 'Большое изображение с множественным текстом - PaddleOCR лучше справляется с комплексными документами'
                })
                recommendation['recommendations'].append({
                    'engine': 'Tesseract',
                    'priority': 2,
                    'reason': 'Альтернатива для больших документов'
                })
                recommendation['recommendations'].append({
                    'engine': 'TrOCR',
                    'priority': 3,
                    'reason': 'НЕ рекомендуется - TrOCR предназначен для отдельных строк текста'
                })
            elif aspect_ratio > 5 or aspect_ratio < 0.2:  # Длинные узкие изображения
                recommendation['recommendations'].append({
                    'engine': 'TrOCR',
                    'priority': 1,
                    'reason': 'Узкое изображение, вероятно содержит одну строку текста - идеально для TrOCR'
                })
                recommendation['recommendations'].append({
                    'engine': 'PaddleOCR',
                    'priority': 2,
                    'reason': 'Универсальная альтернатива'
                })
            elif total_pixels < 500000:  # Небольшие изображения
                recommendation['recommendations'].append({
                    'engine': 'TrOCR',
                    'priority': 1,
                    'reason': 'Небольшое изображение - TrOCR может дать хорошее качество'
                })
                recommendation['recommendations'].append({
                    'engine': 'PaddleOCR',
                    'priority': 2,
                    'reason': 'Надежная альтернатива'
                })
            else:  # Средние изображения
                recommendation['recommendations'].append({
                    'engine': 'PaddleOCR',
                    'priority': 1,
                    'reason': 'Универсальный выбор для большинства документов'
                })
                recommendation['recommendations'].append({
                    'engine': 'Tesseract',
                    'priority': 2,
                    'reason': 'Хорошая альтернатива, особенно для текстовых документов'
                })
                recommendation['recommendations'].append({
                    'engine': 'TrOCR',
                    'priority': 3,
                    'reason': 'Может не подойти для сложных документов'
                })

            # Добавляем информацию о доступности движков
            for rec in recommendation['recommendations']:
                engine = rec['engine']
                if engine in self.engines:
                    rec['available'] = self.engines[engine]['available']
                    if not rec['available']:
                        rec['reason'] += ' (НЕ УСТАНОВЛЕН)'

            return recommendation

        except Exception as e:
            return {
                'error': f'Ошибка анализа изображения: {str(e)}',
                'recommendations': [{
                    'engine': 'PaddleOCR',
                    'priority': 1,
                    'reason': 'Универсальный выбор по умолчанию'
                }]
            }

    def process_document(
        self,
        image_path: str,
        engine: str = 'PaddleOCR',
        language: str = 'ru',
        use_llm: bool = False,
        confidence_threshold: float = 0.5,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Обработка документа выбранным OCR движком

        Args:
            image_path: Путь к изображению
            engine: Название OCR движка
            language: Язык распознавания
            use_llm: Использовать ли LLM постобработку
            confidence_threshold: Порог уверенности

        Returns:
            Результат обработки
        """
        try:
            if engine not in self.engines:
                raise ValueError(f"Неизвестный движок: {engine}")

            if not self.engines[engine]['available']:
                raise RuntimeError(f"Движок {engine} недоступен")

            # Проверяем файл
            if not Path(image_path).exists():
                raise FileNotFoundError(f"Файл не найден: {image_path}")

            # Запускаем OCR
            ocr_function = self.engines[engine]['function']
            result = ocr_function(image_path, language=language, **kwargs)

            # Добавляем метаданные
            result.update({
                'processing_params': {
                    'engine': engine,
                    'language': language,
                    'use_llm': use_llm,
                    'confidence_threshold': confidence_threshold
                },
                'file_info': {
                    'path': image_path,
                    'size': Path(image_path).stat().st_size,
                    'name': Path(image_path).name
                },
                'timestamp': str(Path(image_path).stat().st_mtime),
                'success': True
            })

            # TODO: LLM постобработка когда будет реализована
            if use_llm:
                result['llm_postprocessed'] = False
                result['llm_note'] = "LLM постобработка пока не реализована"

            return result

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc(),
                'engine': engine
            }

    def compare_engines(
        self,
        image_path: str,
        engines: List[str] = None,
        language: str = 'ru',
        **kwargs
    ) -> Dict[str, Any]:
        """
        Сравнение нескольких OCR движков на одном документе

        Args:
            image_path: Путь к изображению
            engines: Список движков для сравнения
            language: Язык распознавания

        Returns:
            Результаты сравнения
        """
        if engines is None:
            engines = self.get_available_engines()

        results = {}
        comparison_metrics = {}

        # Обрабатываем каждым движком
        for engine in engines:
            if engine in self.engines and self.engines[engine]['available']:
                print(f"Обработка {engine}...")
                result = self.process_document(
                    image_path=image_path,
                    engine=engine,
                    language=language,
                    **kwargs
                )
                results[engine] = result

        # Вычисляем метрики сравнения
        if len(results) > 1 and METRICS_AVAILABLE:
            texts = {engine: result.get('raw_text', '') for engine, result in results.items() if result.get('success')}

            # Сравниваем тексты попарно
            for engine1 in texts:
                for engine2 in texts:
                    if engine1 != engine2:
                        key = f"{engine1}_vs_{engine2}"
                        try:
                            comparison_metrics[key] = {
                                'cer': cer(texts[engine1], texts[engine2]),
                                'wer': wer(texts[engine1], texts[engine2]),
                                'similarity': normalized_levenshtein(texts[engine1], texts[engine2]),
                                'length_diff': abs(len(texts[engine1]) - len(texts[engine2]))
                            }
                        except Exception as e:
                            comparison_metrics[key] = {'error': str(e)}

        return {
            'results': results,
            'comparison_metrics': comparison_metrics,
            'summary': {
                'engines_tested': len(results),
                'successful_engines': len([r for r in results.values() if r.get('success')]),
                'failed_engines': len([r for r in results.values() if not r.get('success')])
            }
        }

    def calculate_metrics(
        self,
        reference_text: str,
        hypothesis_text: str
    ) -> Dict[str, float]:
        """
        Вычисление метрик качества OCR

        Args:
            reference_text: Эталонный текст
            hypothesis_text: Распознанный текст

        Returns:
            Словарь с метриками
        """
        if not METRICS_AVAILABLE:
            raise RuntimeError("Модуль метрик недоступен")

        return {
            'cer': cer(reference_text, hypothesis_text),
            'wer': wer(reference_text, hypothesis_text),
            'similarity': normalized_levenshtein(reference_text, hypothesis_text),
            'length_reference': len(reference_text),
            'length_hypothesis': len(hypothesis_text),
            'length_ratio': len(hypothesis_text) / len(reference_text) if len(reference_text) > 0 else 0,
            'words_reference': len(reference_text.split()),
            'words_hypothesis': len(hypothesis_text.split())
        }

    def batch_process(
        self,
        image_paths: List[str],
        engine: str = 'PaddleOCR',
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Пакетная обработка нескольких документов"""
        results = []

        for i, image_path in enumerate(image_paths):
            print(f"Обработка {i+1}/{len(image_paths)}: {Path(image_path).name}")
            result = self.process_document(
                image_path=image_path,
                engine=engine,
                **kwargs
            )
            results.append(result)

        return results


# Пример использования
if __name__ == "__main__":
    coordinator = OCRCoordinator()

    print("🔧 Доступные OCR движки:")
    engines_info = coordinator.get_engine_info()
    for name, info in engines_info.items():
        status = "✅" if info['available'] else "❌"
        print(f"  {status} {name}: {info['description']}")

    # Пример обработки файла (если есть тестовое изображение)
    test_image = "../data/samples/image.png"
    if Path(test_image).exists():
        print(f"\n📄 Тестирование на файле: {test_image}")

        # Обработка одним движком
        if coordinator.get_available_engines():
            engine = coordinator.get_available_engines()[0]
            result = coordinator.process_document(test_image, engine=engine)

            if result.get('success'):
                print(f"✅ {engine} успешно обработал документ")
                print(f"📝 Текст: {result['raw_text'][:100]}...")
                print(f"📊 Извлечено полей: {len(result.get('extracted_fields', {}))}")
            else:
                print(f"❌ Ошибка {engine}: {result.get('error')}")

        # Сравнение движков
        available = coordinator.get_available_engines()
        if len(available) > 1:
            print(f"\n🔄 Сравнение движков: {', '.join(available[:2])}")
            comparison = coordinator.compare_engines(test_image, engines=available[:2])
            print(f"📊 Успешно обработано: {comparison['summary']['successful_engines']}/{comparison['summary']['engines_tested']}")
    else:
        print(f"\n⚠️  Тестовый файл не найден: {test_image}")