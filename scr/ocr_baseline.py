"""
ocr_baseline.py - Tesseract wrapper для baseline OCR распознавания
Обеспечивает извлечение текста и табличных данных с bbox и confidence.
"""

import json
import csv
from pathlib import Path
from typing import List, Dict, Optional, Union
import pytesseract
from pytesseract import Output
from PIL import Image
import numpy as np
import subprocess
try:
    from pdf2image import convert_from_path
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    print("Предупреждение: pdf2image не установлен. Поддержка PDF недоступна.")


def get_available_languages() -> List[str]:
    """
    Получает список доступных языков Tesseract

    Returns:
        Список кодов языков
    """
    try:
        langs = pytesseract.get_languages(config='')
        return langs
    except Exception as e:
        print(f"Ошибка получения языков: {e}")
        return ['eng']  # По умолчанию только английский


def choose_best_language(text_hint: str = None) -> str:
    """
    Выбирает лучший доступный язык для распознавания

    Args:
        text_hint: Подсказка о содержимом текста

    Returns:
        Код языка для Tesseract
    """
    available = get_available_languages()

    # Проверяем наличие русского
    if 'rus' in available:
        if 'eng' in available:
            return 'rus+eng'  # Оба языка
        return 'rus'  # Только русский

    # Если русского нет, используем английский
    return 'eng'


def run_tesseract(path: str, lang: str = None) -> str:
    """
    Выполняет OCR распознавание и возвращает извлеченный текст.

    Args:
        path: Путь к изображению или PDF файлу
        lang: Языки для распознавания (автоопределение если None)

    Returns:
        Распознанный текст как строка

    Raises:
        FileNotFoundError: Если файл не найден
        ValueError: Если формат файла не поддерживается
    """
    # Автоопределение языка если не указан
    if lang is None:
        lang = choose_best_language()
    file_path = Path(path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")
    
    # Обработка PDF
    if file_path.suffix.lower() == '.pdf':
        if not PDF_SUPPORT:
            return "Ошибка: pdf2image не установлен. Установите: pip install pdf2image"

        try:
            pages = convert_from_path(path)
            text_parts = []

            for i, page in enumerate(pages):
                page_text = pytesseract.image_to_string(page, lang=lang)
                if len(pages) > 1:
                    text_parts.append(f"--- Страница {i+1} ---\n{page_text}")
                else:
                    text_parts.append(page_text)

            return "\n\n".join(text_parts)

        except Exception as e:
            return f"Ошибка обработки PDF: {e}. Возможно, нужно установить Poppler: https://github.com/oschwartz10612/poppler-windows/releases/"
    
    # Обработка изображений
    try:
        image = Image.open(path)
        text = pytesseract.image_to_string(image, lang=lang)
        return text
    except Exception as e:
        # Не выбрасываем исключение, возвращаем сообщение об ошибке
        error_msg = str(e)
        if "tesseract is not installed" in error_msg:
            return "Ошибка: Tesseract не установлен. Скачайте с https://github.com/UB-Mannheim/tesseract/wiki и добавьте в PATH"
        else:
            return f"Ошибка Tesseract: {error_msg}"


def run_tesseract_with_data(path: str, lang: str = None) -> List[Dict]:
    """
    Выполняет OCR и возвращает детальные данные с bbox и confidence.

    Args:
        path: Путь к изображению или PDF файлу
        lang: Языки для распознавания (автоопределение если None)

    Returns:
        Список словарей с полями:
        - text: распознанный текст
        - left, top, width, height: координаты bbox
        - conf: уверенность распознавания
        - line_num: номер строки
        - block_num: номер блока
        - page_num: номер страницы (для PDF)
    """
    # Автоопределение языка если не указан
    if lang is None:
        lang = choose_best_language()
    file_path = Path(path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")
    
    results = []
    
    # Обработка PDF
    if file_path.suffix.lower() == '.pdf':
        if not PDF_SUPPORT:
            raise ValueError("Поддержка PDF не доступна. Установите pdf2image.")
        
        pages = convert_from_path(path)
        
        for page_idx, page in enumerate(pages):
            page_data = pytesseract.image_to_data(page, lang=lang, output_type=Output.DICT)
            page_results = _process_tesseract_data(page_data, page_num=page_idx+1)
            results.extend(page_results)
    else:
        # Обработка изображения
        image = Image.open(path)
        data = pytesseract.image_to_data(image, lang=lang, output_type=Output.DICT)
        results = _process_tesseract_data(data)
    
    return results


def _process_tesseract_data(data: Dict, page_num: int = 1) -> List[Dict]:
    """
    Обрабатывает сырые данные от Tesseract.
    
    Args:
        data: Словарь данных от pytesseract.image_to_data
        page_num: Номер страницы для многостраничных документов
    
    Returns:
        Список обработанных записей
    """
    results = []
    n_boxes = len(data['text'])
    
    for i in range(n_boxes):
        text = str(data['text'][i]).strip()
        
        # Пропускаем пустые элементы
        if not text:
            continue
        
        # Обрабатываем confidence
        conf = data['conf'][i]
        if conf == -1 or conf == 0:
            # Помечаем низкой уверенностью
            conf = 0
        
        # Собираем запись
        record = {
            'text': text,
            'left': data['left'][i],
            'top': data['top'][i],
            'width': data['width'][i],
            'height': data['height'][i],
            'conf': conf,
            'line_num': data['line_num'][i] if 'line_num' in data else 0,
            'block_num': data['block_num'][i] if 'block_num' in data else 0,
            'page_num': page_num
        }
        
        # Добавляем дополнительные поля если доступны
        if 'par_num' in data:
            record['par_num'] = data['par_num'][i]
        if 'word_num' in data:
            record['word_num'] = data['word_num'][i]
        
        results.append(record)
    
    return results


def save_results_text(text: str, out_path: str) -> None:
    """
    Сохраняет распознанный текст в файл.
    
    Args:
        text: Текст для сохранения
        out_path: Путь для выходного .txt файла
    """
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_file, 'w', encoding='utf-8') as f:
        f.write(text)
    
    print(f"Текст сохранен в: {out_file}")


def save_results_tsv(data: List[Dict], out_path: str) -> None:
    """
    Сохраняет данные OCR в TSV или JSON формат.
    
    Args:
        data: Список словарей с данными OCR
        out_path: Путь для выходного файла (.tsv или .json)
    """
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    
    if out_file.suffix.lower() == '.json':
        # Сохранение в JSON
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Данные сохранены в JSON: {out_file}")
    
    elif out_file.suffix.lower() in ['.tsv', '.csv']:
        # Сохранение в TSV/CSV
        if not data:
            print("Нет данных для сохранения")
            return
        
        # Определяем разделитель
        delimiter = '\t' if out_file.suffix.lower() == '.tsv' else ','
        
        # Определяем все возможные поля
        fieldnames = []
        for record in data:
            for key in record.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
        
        with open(out_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=delimiter)
            writer.writeheader()
            writer.writerows(data)
        
        format_name = "TSV" if delimiter == '\t' else "CSV"
        print(f"Данные сохранены в {format_name}: {out_file}")
    
    else:
        raise ValueError(f"Неподдерживаемый формат файла: {out_file.suffix}")


def process_document(
    input_path: str,
    output_dir: str = "./output",
    lang: str = 'rus+eng',
    save_text: bool = True,
    save_data: bool = True,
    data_format: str = 'tsv'
) -> Dict:
    """
    Полная обработка документа с сохранением результатов.
    
    Args:
        input_path: Путь к входному документу
        output_dir: Директория для сохранения результатов
        lang: Языки для распознавания
        save_text: Сохранять ли текстовый файл
        save_data: Сохранять ли структурированные данные
        data_format: Формат для структурированных данных ('tsv', 'json', 'csv')
    
    Returns:
        Словарь с путями к сохраненным файлам и статистикой
    """
    input_file = Path(input_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    base_name = input_file.stem
    results = {}
    
    print(f"Обработка: {input_file}")
    
    # Извлечение текста
    if save_text:
        print("Извлечение текста...")
        text = run_tesseract(input_path, lang=lang)
        text_file = output_path / f"{base_name}_text.txt"
        save_results_text(text, str(text_file))
        results['text_file'] = str(text_file)
        results['text_length'] = len(text)
    
    # Извлечение структурированных данных
    if save_data:
        print("Извлечение структурированных данных...")
        data = run_tesseract_with_data(input_path, lang=lang)
        
        # Определяем расширение файла
        if data_format.lower() == 'json':
            data_file = output_path / f"{base_name}_data.json"
        elif data_format.lower() == 'csv':
            data_file = output_path / f"{base_name}_data.csv"
        else:
            data_file = output_path / f"{base_name}_data.tsv"
        
        save_results_tsv(data, str(data_file))
        results['data_file'] = str(data_file)
        results['total_items'] = len(data)
        
        # Статистика по confidence
        if data:
            confidences = [item['conf'] for item in data]
            results['avg_confidence'] = sum(confidences) / len(confidences)
            results['min_confidence'] = min(confidences)
            results['max_confidence'] = max(confidences)
            results['low_conf_items'] = sum(1 for c in confidences if c < 50)
    
    print(f"\nОбработка завершена!")
    return results


def batch_process(
    input_dir: str,
    output_dir: str = "./output",
    extensions: List[str] = None,
    **kwargs
) -> List[Dict]:
    """
    Пакетная обработка файлов в директории.
    
    Args:
        input_dir: Директория с входными файлами
        output_dir: Директория для результатов
        extensions: Список расширений файлов для обработки
        **kwargs: Дополнительные параметры для process_document
    
    Returns:
        Список результатов обработки
    """
    if extensions is None:
        extensions = ['.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.pdf']
    
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Директория не найдена: {input_dir}")
    
    # Находим все подходящие файлы
    files = []
    for ext in extensions:
        files.extend(input_path.glob(f"*{ext}"))
        files.extend(input_path.glob(f"*{ext.upper()}"))
    
    if not files:
        print(f"Файлы с расширениями {extensions} не найдены в {input_dir}")
        return []
    
    results = []
    print(f"Найдено файлов для обработки: {len(files)}")
    
    for i, file_path in enumerate(files, 1):
        print(f"\n[{i}/{len(files)}] Обработка: {file_path.name}")
        try:
            result = process_document(
                str(file_path),
                output_dir=output_dir,
                **kwargs
            )
            result['input_file'] = str(file_path)
            result['status'] = 'success'
            results.append(result)
        except Exception as e:
            print(f"Ошибка при обработке {file_path}: {e}")
            results.append({
                'input_file': str(file_path),
                'status': 'error',
                'error': str(e)
            })
    
    # Сохраняем сводку
    summary_file = Path(output_dir) / "processing_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n\nСводка обработки сохранена в: {summary_file}")
    return results


if __name__ == "__main__":
    # Пример использования
    import argparse
    
    parser = argparse.ArgumentParser(description="OCR Baseline с Tesseract")
    parser.add_argument("input", help="Путь к файлу или директории")
    parser.add_argument("-o", "--output", default="./output", help="Директория для результатов")
    parser.add_argument("-l", "--lang", default="rus+eng", help="Языки для OCR")
    parser.add_argument("-f", "--format", default="tsv", choices=["tsv", "json", "csv"],
                       help="Формат для структурированных данных")
    parser.add_argument("--batch", action="store_true", help="Пакетная обработка директории")
    
    args = parser.parse_args()
    
    if args.batch:
        results = batch_process(
            args.input,
            output_dir=args.output,
            lang=args.lang,
            data_format=args.format
        )
        print(f"\nОбработано файлов: {len(results)}")
        successful = sum(1 for r in results if r.get('status') == 'success')
        print(f"Успешно: {successful}/{len(results)}")
    else:
        result = process_document(
            args.input,
            output_dir=args.output,
            lang=args.lang,
            data_format=args.format
        )
        print("\nРезультаты:")
        for key, value in result.items():
            print(f"  {key}: {value}")