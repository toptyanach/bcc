"""
ocr_paddle.py - PaddleOCR wrapper для распознавания кириллицы и банковских форм
Обеспечивает извлечение текста с bbox и confidence.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from paddleocr import PaddleOCR
import cv2
from PIL import Image


# Глобальный объект PaddleOCR для эффективности
_paddle_instance = None


def get_paddle_instance(lang: str = 'ru', use_angle_cls: bool = True) -> PaddleOCR:
    """
    Получает или создает единственный экземпляр PaddleOCR.
    
    Args:
        lang: Язык для распознавания ('ru', 'en', 'ch')
        use_angle_cls: Использовать ли классификатор угла поворота
    
    Returns:
        Экземпляр PaddleOCR
    """
    global _paddle_instance
    
    if _paddle_instance is None:
        print("Инициализация PaddleOCR...")
        _paddle_instance = PaddleOCR(
            use_angle_cls=use_angle_cls,
            lang=lang
        )
        print("PaddleOCR инициализирован")
    
    return _paddle_instance


def normalize_paddle_output(raw_output: List) -> List[Dict]:
    """
    Нормализует вложенную структуру вывода PaddleOCR.
    
    Args:
        raw_output: Сырой вывод от PaddleOCR
    
    Returns:
        Список словарей с полями box, text, conf
    """
    if not raw_output or not raw_output[0]:
        return []
    
    # Проверяем тип результата
    result_obj = raw_output[0]
    print(f"DEBUG: PaddleOCR result type: {type(result_obj)}")
    
    # Проверим класс объекта более подробно
    class_name = str(type(result_obj).__name__)
    print(f"DEBUG: Class name: {class_name}")
    
    # Новый формат: OCRResult объект (проверяем по имени класса тоже)
    is_ocr_result = (class_name == 'OCRResult' or 
                     hasattr(result_obj, 'rec_texts') or 
                     hasattr(result_obj, 'rec_scores') or 
                     hasattr(result_obj, 'rec_polys'))
    
    if is_ocr_result:
        print("DEBUG: Using new OCRResult format")
        
        # Debug: посмотрим что доступно в объекте
        print(f"DEBUG: Available attributes: {[attr for attr in dir(result_obj) if not attr.startswith('_')]}")
        
        # Пробуем разные способы получить данные
        rec_texts = None
        rec_scores = None
        rec_polys = None
        
        # Способ 1: Проверяем методы словаря
        if hasattr(result_obj, 'keys'):
            try:
                keys = list(result_obj.keys())
                print(f"DEBUG: OCRResult keys: {keys}")
                
                # Пробуем обычные ключи
                for key in ['rec_texts', 'texts', 'text']:
                    if key in keys:
                        rec_texts = result_obj[key]
                        print(f"DEBUG: Found texts via key '{key}': {len(rec_texts) if rec_texts else 0}")
                        break
                        
                for key in ['rec_scores', 'scores', 'confidences']:
                    if key in keys:
                        rec_scores = result_obj[key]
                        print(f"DEBUG: Found scores via key '{key}': {len(rec_scores) if rec_scores else 0}")
                        break
                        
                for key in ['rec_polys', 'polys', 'boxes', 'coordinates']:
                    if key in keys:
                        rec_polys = result_obj[key]
                        print(f"DEBUG: Found polys via key '{key}': {len(rec_polys) if rec_polys else 0}")
                        break
            except Exception as e:
                print(f"DEBUG: Error accessing keys: {e}")
        
        # Способ 2: Используем метод get
        if rec_texts is None and hasattr(result_obj, 'get'):
            try:
                rec_texts = result_obj.get('rec_texts', result_obj.get('texts', result_obj.get('text', [])))
                rec_scores = result_obj.get('rec_scores', result_obj.get('scores', result_obj.get('confidences', [])))
                rec_polys = result_obj.get('rec_polys', result_obj.get('polys', result_obj.get('boxes', [])))
                print(f"DEBUG: Got via get() - texts: {len(rec_texts) if rec_texts else 0}, scores: {len(rec_scores) if rec_scores else 0}, polys: {len(rec_polys) if rec_polys else 0}")
            except Exception as e:
                print(f"DEBUG: Error using get(): {e}")
        
        # Способ 3: Проверяем метод json()
        if rec_texts is None and hasattr(result_obj, 'json'):
            try:
                json_data = result_obj.json()
                print(f"DEBUG: JSON data type: {type(json_data)}")
                if isinstance(json_data, dict):
                    print(f"DEBUG: JSON keys: {list(json_data.keys())}")
                    rec_texts = json_data.get('rec_texts', json_data.get('texts', []))
                    rec_scores = json_data.get('rec_scores', json_data.get('scores', []))
                    rec_polys = json_data.get('rec_polys', json_data.get('polys', []))
                elif isinstance(json_data, list) and len(json_data) > 0:
                    print(f"DEBUG: JSON is list with {len(json_data)} items")
                    # Проверяем структуру первого элемента
                    first_item = json_data[0]
                    print(f"DEBUG: First JSON item type: {type(first_item)}")
                    
                    if isinstance(first_item, dict):
                        # Это может быть словарь с результатами OCR
                        print(f"DEBUG: First item keys: {list(first_item.keys())}")
                        # Пытаемся найти текстовые данные
                        if 'rec_texts' in first_item:
                            rec_texts = first_item['rec_texts']
                            rec_scores = first_item.get('rec_scores', [])
                            rec_polys = first_item.get('rec_polys', [])
                        elif 'text' in first_item:
                            # Возможно это отдельные элементы
                            rec_texts = [item.get('text', '') for item in json_data if 'text' in item]
                            rec_scores = [item.get('score', item.get('confidence', 1.0)) for item in json_data]
                            rec_polys = [item.get('poly', item.get('box', [])) for item in json_data]
                    elif isinstance(first_item, list) and len(first_item) >= 2:
                        # Старый формат: [[bbox, (text, score)], ...]
                        print("DEBUG: Detected legacy format in JSON")
                        rec_texts = []
                        rec_scores = []
                        rec_polys = []
                        
                        for item in json_data:
                            if isinstance(item, list) and len(item) >= 2:
                                bbox = item[0]
                                text_info = item[1]
                                
                                if isinstance(text_info, list) and len(text_info) >= 2:
                                    text = text_info[0]
                                    score = text_info[1]
                                    
                                    rec_texts.append(text)
                                    rec_scores.append(score)
                                    rec_polys.append(bbox)
                                    
                        print(f"DEBUG: Extracted from legacy JSON - texts: {len(rec_texts)}, scores: {len(rec_scores)}, polys: {len(rec_polys)}")
                    else:
                        # Возможно это список текстов напрямую
                        rec_texts = json_data
                        rec_scores = [1.0] * len(json_data)  # Заглушка для scores
                        rec_polys = [[] for _ in json_data]  # Заглушка для координат
            except Exception as e:
                print(f"DEBUG: Error using json(): {e}")
                import traceback
                traceback.print_exc()
        
        # Способ 4: Проверяем items()
        if rec_texts is None and hasattr(result_obj, 'items'):
            try:
                items = list(result_obj.items())
                print(f"DEBUG: Items: {items[:3]}...")  # Показываем первые 3 элемента
                items_dict = dict(items)
                rec_texts = items_dict.get('rec_texts', items_dict.get('texts', []))
                rec_scores = items_dict.get('rec_scores', items_dict.get('scores', []))
                rec_polys = items_dict.get('rec_polys', items_dict.get('polys', []))
            except Exception as e:
                print(f"DEBUG: Error using items(): {e}")
        
        # Способ 5: Прямой доступ к атрибутам (как было раньше)
        if rec_texts is None:
            if hasattr(result_obj, 'rec_texts'):
                rec_texts = getattr(result_obj, 'rec_texts', [])
                print(f"DEBUG: Got rec_texts via attribute: {len(rec_texts) if rec_texts else 0}")
            
            if hasattr(result_obj, 'rec_scores'):
                rec_scores = getattr(result_obj, 'rec_scores', [])
                print(f"DEBUG: Got rec_scores via attribute: {len(rec_scores) if rec_scores else 0}")
            
            if hasattr(result_obj, 'rec_polys'):
                rec_polys = getattr(result_obj, 'rec_polys', [])
                print(f"DEBUG: Got rec_polys via attribute: {len(rec_polys) if rec_polys else 0}")
                
        # Способ 6: Проверяем __dict__ если атрибуты не найдены
        if rec_texts is None and hasattr(result_obj, '__dict__'):
            obj_dict = result_obj.__dict__
            print(f"DEBUG: Object dict keys: {list(obj_dict.keys())}")
            rec_texts = obj_dict.get('rec_texts', [])
            rec_scores = obj_dict.get('rec_scores', [])
            rec_polys = obj_dict.get('rec_polys', [])
            
        # Убеждаемся что у нас есть списки
        rec_texts = rec_texts or []
        rec_scores = rec_scores or []
        rec_polys = rec_polys or []
        
        print(f"DEBUG: Final counts - texts: {len(rec_texts)}, scores: {len(rec_scores)}, polys: {len(rec_polys)}")
        
        if len(rec_texts) > 0:
            print(f"DEBUG: First text: {rec_texts[0]}")
        if len(rec_scores) > 0:
            print(f"DEBUG: First score: {rec_scores[0]}")
        if len(rec_polys) > 0:
            print(f"DEBUG: First poly: {rec_polys[0]}")
        
        normalized = []
        
        # Объединяем данные из трех списков
        for i in range(min(len(rec_texts), len(rec_scores), len(rec_polys))):
            try:
                text = rec_texts[i] if i < len(rec_texts) else ""
                conf = rec_scores[i] if i < len(rec_scores) else 0.0
                poly = rec_polys[i] if i < len(rec_polys) else []
                
                print(f"DEBUG: Item {i}: text='{text}', conf={conf}, poly={poly}")
                
                if not text or not poly:
                    continue
                
                # Обрабатываем координаты полигона
                box = []
                if isinstance(poly, (list, tuple)) and len(poly) > 0:
                    for point in poly:
                        try:
                            if isinstance(point, (list, tuple)) and len(point) >= 2:
                                x = float(point[0])
                                y = float(point[1])
                                box.append([x, y])
                        except (ValueError, TypeError, IndexError) as e:
                            print(f"Warning: Could not convert point {point}: {e}")
                            continue
                
                # Проверяем, что у нас есть валидные координаты
                if not box:
                    print(f"Warning: No valid coordinates for text '{text}', skipping")
                    continue
                
                # Вычисляем дополнительные метрики bbox
                x_coords = [p[0] for p in box]
                y_coords = [p[1] for p in box]
                
                normalized.append({
                    'box': box,
                    'text': text,
                    'conf': float(conf),
                    'left': min(x_coords),
                    'top': min(y_coords),
                    'right': max(x_coords),
                    'bottom': max(y_coords),
                    'width': max(x_coords) - min(x_coords),
                    'height': max(y_coords) - min(y_coords),
                    'center_x': (min(x_coords) + max(x_coords)) / 2,
                    'center_y': (min(y_coords) + max(y_coords)) / 2
                })
                
            except Exception as e:
                print(f"Error processing item {i}: {e}")
                continue
        
        print(f"DEBUG: Normalized {len(normalized)} items")
        return normalized
    
    # Старый формат: список кортежей (оставляем для совместимости)
    else:
        print("DEBUG: Trying legacy format")
        return normalize_legacy_format(raw_output)


def normalize_legacy_format(raw_output: List) -> List[Dict]:
    """
    Обрабатывает старый формат PaddleOCR (список кортежей)
    """
    normalized = []
    
    try:
        for line_idx, line in enumerate(raw_output[0]):
            if line is None or not isinstance(line, (list, tuple)) or len(line) < 2:
                continue
                
            try:
                box_coords = line[0]
                text_data = line[1]
                
                # Извлекаем текст и confidence
                text = text_data[0] if isinstance(text_data, (list, tuple)) else str(text_data)
                conf = text_data[1] if isinstance(text_data, (list, tuple)) and len(text_data) > 1 else 0.0
                
                # Преобразуем координаты bbox
                box = []
                for point in box_coords:
                    try:
                        if isinstance(point, (list, tuple)):
                            x = float(point[0])
                            y = float(point[1])
                            box.append([x, y])
                        else:
                            x = float(point)
                            box.append([x, 0])
                    except (ValueError, TypeError, IndexError):
                        continue
                
                if not box or not text:
                    continue
                
                # Вычисляем дополнительные метрики bbox
                x_coords = [p[0] for p in box]
                y_coords = [p[1] for p in box]
                
                normalized.append({
                    'box': box,
                    'text': text,
                    'conf': float(conf),
                    'left': min(x_coords),
                    'top': min(y_coords),
                    'right': max(x_coords),
                    'bottom': max(y_coords),
                    'width': max(x_coords) - min(x_coords),
                    'height': max(y_coords) - min(y_coords),
                    'center_x': (min(x_coords) + max(x_coords)) / 2,
                    'center_y': (min(y_coords) + max(y_coords)) / 2
                })
                
            except Exception as e:
                print(f"Error processing legacy line {line_idx}: {e}")
                continue
                
    except Exception as e:
        print(f"Error in legacy format processing: {e}")
    
    return normalized


def run_paddle(path: str, lang: str = 'ru') -> List[Dict]:
    """
    Выполняет OCR с помощью PaddleOCR.
    
    Args:
        path: Путь к изображению
        lang: Язык для распознавания
    
    Returns:
        Список объектов с полями:
        - box: [[x,y], ...] - координаты углов bbox
        - text: распознанный текст
        - conf: уверенность распознавания
        - left, top, right, bottom: границы bbox
        - width, height: размеры bbox
        - center_x, center_y: центр bbox
    """
    file_path = Path(path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")
    
    # Получаем экземпляр PaddleOCR
    ocr = get_paddle_instance(lang=lang)
    
    # Загружаем изображение
    if file_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
        # Выполняем OCR
        result = ocr.ocr(str(file_path))
        
        # Нормализуем вывод
        normalized = normalize_paddle_output(result)
        
        # Сортируем по Y-координате для правильного порядка
        normalized = sort_by_reading_order(normalized)
        
        return normalized
    else:
        raise ValueError(f"Неподдерживаемый формат файла: {file_path.suffix}")


def sort_by_reading_order(ocr_output: List[Dict], line_threshold: float = 10) -> List[Dict]:
    """
    Сортирует результаты OCR в порядке чтения (сверху вниз, слева направо).
    
    Args:
        ocr_output: Список результатов OCR
        line_threshold: Порог для группировки элементов в строки (по Y-координате)
    
    Returns:
        Отсортированный список результатов
    """
    if not ocr_output:
        return []
    
    # Группируем элементы по строкам
    lines = []
    for item in ocr_output:
        placed = False
        
        # Ищем подходящую строку
        for line in lines:
            # Проверяем, находится ли элемент на той же строке
            line_y = sum(i['center_y'] for i in line) / len(line)
            if abs(item['center_y'] - line_y) < line_threshold:
                line.append(item)
                placed = True
                break
        
        # Если не нашли подходящую строку, создаем новую
        if not placed:
            lines.append([item])
    
    # Сортируем строки по Y-координате
    lines.sort(key=lambda line: sum(i['center_y'] for i in line) / len(line))
    
    # Сортируем элементы внутри каждой строки по X-координате
    sorted_output = []
    for line_idx, line in enumerate(lines):
        line.sort(key=lambda item: item['center_x'])
        # Добавляем номер строки
        for item in line:
            item['line_num'] = line_idx
        sorted_output.extend(line)
    
    return sorted_output


def get_plaintext(ocr_output: List[Dict], join_lines: str = '\n') -> str:
    """
    Собирает распознанный текст в единую строку.
    
    Args:
        ocr_output: Список результатов OCR
        join_lines: Разделитель между строками
    
    Returns:
        Объединенный текст
    """
    if not ocr_output:
        return ""
    
    # Группируем по строкам
    lines_dict = {}
    for item in ocr_output:
        line_num = item.get('line_num', 0)
        if line_num not in lines_dict:
            lines_dict[line_num] = []
        lines_dict[line_num].append(item)
    
    # Собираем текст по строкам
    text_lines = []
    for line_num in sorted(lines_dict.keys()):
        line_items = lines_dict[line_num]
        # Сортируем элементы в строке по X-координате
        line_items.sort(key=lambda x: x.get('left', x.get('center_x', 0)))
        # Объединяем текст в строке
        line_text = ' '.join(item['text'] for item in line_items if item.get('text'))
        if line_text:
            text_lines.append(line_text)
    
    return join_lines.join(text_lines)


def save_paddle_output(ocr_output: List[Dict], out_path: str) -> None:
    """
    Сохраняет результаты PaddleOCR в JSON.
    
    Args:
        ocr_output: Список результатов OCR
        out_path: Путь для сохранения JSON файла
    """
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Подготавливаем данные для сериализации
    serializable_output = []
    for item in ocr_output:
        serializable_item = {}
        for key, value in item.items():
            # Преобразуем numpy arrays если есть
            if isinstance(value, np.ndarray):
                serializable_item[key] = value.tolist()
            else:
                serializable_item[key] = value
        serializable_output.append(serializable_item)
    
    # Сохраняем в JSON
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_output, f, ensure_ascii=False, indent=2)
    
    print(f"Результаты PaddleOCR сохранены в: {out_file}")


def visualize_results(
    image_path: str,
    ocr_output: List[Dict],
    output_path: Optional[str] = None,
    show: bool = False
) -> np.ndarray:
    """
    Визуализирует результаты OCR на изображении.
    
    Args:
        image_path: Путь к исходному изображению
        ocr_output: Результаты OCR
        output_path: Путь для сохранения визуализации
        show: Показать ли изображение
    
    Returns:
        Изображение с нанесенными bbox и текстом
    """
    # Загружаем изображение
    image = cv2.imread(image_path)
    if image is None:
        image = np.array(Image.open(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Копируем для рисования
    vis_image = image.copy()
    
    # Настройки визуализации
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    
    for item in ocr_output:
        box = item['box']
        text = item['text']
        conf = item['conf']
        
        # Преобразуем координаты в целые числа
        pts = np.array(box, dtype=np.int32)
        
        # Цвет в зависимости от confidence
        if conf > 0.9:
            color = (0, 255, 0)  # Зеленый - высокая уверенность
        elif conf > 0.7:
            color = (0, 165, 255)  # Оранжевый - средняя уверенность
        else:
            color = (0, 0, 255)  # Красный - низкая уверенность
        
        # Рисуем bbox
        cv2.polylines(vis_image, [pts], True, color, 2)
        
        # Добавляем текст и confidence
        x, y = int(item['left']), int(item['top'])
        label = f"{text[:20]}... ({conf:.2f})" if len(text) > 20 else f"{text} ({conf:.2f})"
        
        # Фон для текста
        (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
        cv2.rectangle(vis_image, (x, y - text_height - 4), (x + text_width, y), color, -1)
        
        # Текст
        cv2.putText(vis_image, label, (x, y - 2), font, font_scale, (255, 255, 255), thickness)
    
    # Сохраняем результат
    if output_path:
        cv2.imwrite(output_path, vis_image)
        print(f"Визуализация сохранена в: {output_path}")
    
    # Показываем результат
    if show:
        cv2.imshow('OCR Results', vis_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return vis_image


def process_document(
    input_path: str,
    output_dir: str = "./output",
    lang: str = 'ru',
    save_json: bool = True,
    save_text: bool = True,
    save_visualization: bool = False
) -> Dict:
    """
    Полная обработка документа с PaddleOCR.
    
    Args:
        input_path: Путь к входному изображению
        output_dir: Директория для сохранения результатов
        lang: Язык распознавания
        save_json: Сохранять ли JSON с результатами
        save_text: Сохранять ли текстовый файл
        save_visualization: Сохранять ли визуализацию
    
    Returns:
        Словарь с результатами и статистикой
    """
    input_file = Path(input_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    base_name = input_file.stem
    results = {'input_file': str(input_file)}
    
    print(f"Обработка: {input_file}")
    
    # Выполняем OCR
    print("Выполнение OCR с PaddleOCR...")
    ocr_output = run_paddle(str(input_file), lang=lang)
    results['total_items'] = len(ocr_output)
    
    # Статистика
    if ocr_output:
        confidences = [item['conf'] for item in ocr_output]
        results['avg_confidence'] = sum(confidences) / len(confidences)
        results['min_confidence'] = min(confidences)
        results['max_confidence'] = max(confidences)
        results['low_conf_items'] = sum(1 for c in confidences if c < 0.7)
    
    # Сохраняем JSON
    if save_json:
        json_file = output_path / f"{base_name}_paddle.json"
        save_paddle_output(ocr_output, str(json_file))
        results['json_file'] = str(json_file)
    
    # Извлекаем и сохраняем текст
    if save_text:
        text = get_plaintext(ocr_output)
        text_file = output_path / f"{base_name}_text.txt"
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"Текст сохранен в: {text_file}")
        results['text_file'] = str(text_file)
        results['text_length'] = len(text)
    
    # Сохраняем визуализацию
    if save_visualization:
        vis_file = output_path / f"{base_name}_visualization.jpg"
        visualize_results(str(input_file), ocr_output, str(vis_file))
        results['visualization_file'] = str(vis_file)
    
    print(f"Обработка завершена!")
    return results


def compare_with_tesseract(
    input_path: str,
    output_dir: str = "./output"
) -> Dict:
    """
    Сравнивает результаты PaddleOCR с Tesseract.
    
    Args:
        input_path: Путь к изображению
        output_dir: Директория для результатов
    
    Returns:
        Словарь со сравнительной статистикой
    """
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        print("Tesseract не установлен. Сравнение невозможно.")
        return {}
    
    print("Сравнение PaddleOCR и Tesseract...")
    
    # PaddleOCR
    paddle_output = run_paddle(input_path)
    paddle_text = get_plaintext(paddle_output)
    
    # Tesseract
    image = Image.open(input_path)
    tesseract_text = pytesseract.image_to_string(image, lang='rus+eng')
    
    # Сравнение
    comparison = {
        'paddle_chars': len(paddle_text),
        'tesseract_chars': len(tesseract_text),
        'paddle_words': len(paddle_text.split()),
        'tesseract_words': len(tesseract_text.split()),
        'paddle_lines': len(paddle_text.splitlines()),
        'tesseract_lines': len(tesseract_text.splitlines()),
        'paddle_items': len(paddle_output),
        'paddle_avg_conf': sum(item['conf'] for item in paddle_output) / len(paddle_output) if paddle_output else 0
    }
    
    # Сохраняем оба результата
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    base_name = Path(input_path).stem
    
    with open(output_path / f"{base_name}_paddle.txt", 'w', encoding='utf-8') as f:
        f.write(paddle_text)
    
    with open(output_path / f"{base_name}_tesseract.txt", 'w', encoding='utf-8') as f:
        f.write(tesseract_text)
    
    with open(output_path / f"{base_name}_comparison.json", 'w', encoding='utf-8') as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
    
    print(f"Результаты сравнения сохранены в: {output_dir}")
    return comparison


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="PaddleOCR wrapper для кириллицы")
    parser.add_argument("input", help="Путь к изображению")
    parser.add_argument("-o", "--output", default="./output", help="Директория для результатов")
    parser.add_argument("-l", "--lang", default="ru", help="Язык OCR (ru, en, ch)")
    parser.add_argument("--visualize", action="store_true", help="Создать визуализацию")
    parser.add_argument("--compare", action="store_true", help="Сравнить с Tesseract")
    
    args = parser.parse_args()
    
    if args.compare:
        comparison = compare_with_tesseract(args.input, args.output)
        print("\nСравнение PaddleOCR vs Tesseract:")
        for key, value in comparison.items():
            print(f"  {key}: {value}")
    else:
        result = process_document(
            args.input,
            output_dir=args.output,
            lang=args.lang,
            save_visualization=args.visualize
        )
        print("\nРезультаты:")
        for key, value in result.items():
            print(f"  {key}: {value}")