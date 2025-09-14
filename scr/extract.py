"""
extract.py - Извлечение ключевых полей из результатов OCR
Использует комбинацию поиска по меткам, анализа bbox и регулярных выражений.
"""

import re
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Union
import numpy as np
from collections import defaultdict


# Словари меток для различных полей
FIELD_LABELS = {
    'fio': [
        'ФИО', 'Фамилия', 'Имя', 'Отчество', 
        'Ф.И.О.', 'Ф.И.О', 'ФИО:', 'Фамилия Имя Отчество',
        'Заявитель', 'Клиент', 'Плательщик', 'Получатель',
        'От кого', 'Кому', 'Гражданин', 'Гражданка'
    ],
    'date': [
        'Дата', 'Дата:', 'Дата документа', 'Дата выдачи',
        'Дата заполнения', 'Дата подписания', 'Дата составления',
        'от', 'От', 'Число', 'Когда'
    ],
    'sum': [
        'Сумма', 'Итого', 'К оплате', 'Всего', 'Итог',
        'Сумма:', 'Итого:', 'Всего к оплате', 'Сумма платежа',
        'Сумма договора', 'Стоимость', 'Цена', 'Размер'
    ],
    'contract_number': [
        '№ договора', 'Номер договора', '№', 'Номер',
        'Договор №', 'Договор', 'Контракт №', 'Соглашение №',
        'Счет №', 'Счёт №', 'Заявка №'
    ],
    'account': [
        'Счет', 'Счёт', 'Расчетный счет', 'Р/с', 'Р/счет',
        'Банковский счет', 'Лицевой счет', 'Л/с', 'Номер счета'
    ],
    'phone': [
        'Телефон', 'Тел.', 'Тел:', 'Телефон:', 'Моб.',
        'Мобильный', 'Контактный телефон', 'Тел. для связи'
    ],
    'email': [
        'Email', 'E-mail', 'Электронная почта', 'Эл. почта',
        'Почта', 'E-mail:', 'Email:', '@'
    ],
    'inn': [
        'ИНН', 'ИНН:', 'ИНН/КПП', 'Инн'
    ],
    'passport': [
        'Паспорт', 'Паспортные данные', 'Серия', 'Номер паспорта',
        'Документ', 'Удостоверение личности'
    ],
    'address': [
        'Адрес', 'Адрес:', 'Адрес регистрации', 'Адрес проживания',
        'Место жительства', 'Проживает', 'Зарегистрирован'
    ]
}

# Регулярные выражения для различных типов данных
REGEX_PATTERNS = {
    'date_dmy': r'\b(\d{1,2})[.\-/](\d{1,2})[.\-/](\d{4})\b',
    'date_ymd': r'\b(\d{4})[.\-/](\d{1,2})[.\-/](\d{1,2})\b',
    'date_text': r'\b(\d{1,2})\s+(января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)\s+(\d{4})\b',
    'sum': r'(\d{1,3}(?:[\s\u00A0]\d{3})*(?:[.,]\d{1,2})?)\s*(?:руб|рублей|р\.|₽)?',
    'fio': r'\b([А-ЯЁ][а-яё]+)\s+([А-ЯЁ][а-яё]+)(?:\s+([А-ЯЁ][а-яё]+))?\b',
    'phone': r'(?:\+7|8|7)?[\s\-]?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}',
    'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    'inn': r'\b\d{10}\b|\b\d{12}\b',
    'passport_series': r'\b\d{2}\s?\d{2}\b',
    'passport_number': r'\b\d{6}\b',
    'account': r'\b\d{20}\b',
    'contract_number': r'№?\s*(\d+[\-/]?\d*)',
}

# Месяцы для парсинга дат
MONTHS = {
    'января': 1, 'февраля': 2, 'марта': 3, 'апреля': 4,
    'мая': 5, 'июня': 6, 'июля': 7, 'августа': 8,
    'сентября': 9, 'октября': 10, 'ноября': 11, 'декабря': 12
}


def calculate_distance(bbox1: Dict, bbox2: Dict) -> float:
    """
    Вычисляет евклидово расстояние между центрами двух bbox.
    
    Args:
        bbox1, bbox2: Словари с полями center_x, center_y или left, top, width, height
    
    Returns:
        Расстояние между центрами
    """
    # Получаем центры
    if 'center_x' in bbox1:
        x1, y1 = bbox1['center_x'], bbox1['center_y']
    else:
        x1 = bbox1.get('left', 0) + bbox1.get('width', 0) / 2
        y1 = bbox1.get('top', 0) + bbox1.get('height', 0) / 2
    
    if 'center_x' in bbox2:
        x2, y2 = bbox2['center_x'], bbox2['center_y']
    else:
        x2 = bbox2.get('left', 0) + bbox2.get('width', 0) / 2
        y2 = bbox2.get('top', 0) + bbox2.get('height', 0) / 2
    
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def is_right_of(bbox1: Dict, bbox2: Dict, threshold: float = 0) -> bool:
    """Проверяет, находится ли bbox2 справа от bbox1."""
    left1 = bbox1.get('left', 0)
    right1 = left1 + bbox1.get('width', 0)
    left2 = bbox2.get('left', 0)
    
    return left2 > right1 - threshold


def is_below_of(bbox1: Dict, bbox2: Dict, threshold: float = 0) -> bool:
    """Проверяет, находится ли bbox2 ниже bbox1."""
    top1 = bbox1.get('top', 0)
    bottom1 = top1 + bbox1.get('height', 0)
    top2 = bbox2.get('top', 0)
    
    return top2 > bottom1 - threshold


def is_same_line(bbox1: Dict, bbox2: Dict, threshold: float = 10) -> bool:
    """Проверяет, находятся ли два bbox на одной строке."""
    y1 = bbox1.get('center_y', bbox1.get('top', 0) + bbox1.get('height', 0) / 2)
    y2 = bbox2.get('center_y', bbox2.get('top', 0) + bbox2.get('height', 0) / 2)
    
    return abs(y1 - y2) < threshold


def group_lines_by_y(ocr_output: List[Dict], threshold: float = 10) -> List[Dict]:
    """
    Группирует элементы OCR в строки по Y-координате.
    
    Args:
        ocr_output: Список результатов OCR
        threshold: Порог для группировки в строки
    
    Returns:
        Список строк с объединенным текстом и bbox
    """
    if not ocr_output:
        return []
    
    lines = []
    used = set()
    
    for i, item in enumerate(ocr_output):
        if i in used:
            continue
        
        # Начинаем новую строку
        line_items = [item]
        used.add(i)
        
        # Ищем элементы на той же строке
        for j, other in enumerate(ocr_output):
            if j in used:
                continue
            
            if is_same_line(item, other, threshold):
                line_items.append(other)
                used.add(j)
        
        # Сортируем элементы строки по X
        line_items.sort(key=lambda x: x.get('left', x.get('center_x', 0)))
        
        # Объединяем информацию о строке
        texts = [it['text'] for it in line_items if it.get('text')]
        if not texts:
            continue
        
        # Вычисляем общий bbox для строки
        lefts = [it.get('left', 0) for it in line_items]
        tops = [it.get('top', 0) for it in line_items]
        rights = [it.get('left', 0) + it.get('width', 0) for it in line_items]
        bottoms = [it.get('top', 0) + it.get('height', 0) for it in line_items]
        
        line_data = {
            'text': ' '.join(texts),
            'items': line_items,
            'left': min(lefts),
            'top': min(tops),
            'right': max(rights),
            'bottom': max(bottoms),
            'width': max(rights) - min(lefts),
            'height': max(bottoms) - min(tops),
            'center_x': (min(lefts) + max(rights)) / 2,
            'center_y': (min(tops) + max(bottoms)) / 2,
            'confidence': sum(it.get('conf', 0) for it in line_items) / len(line_items)
        }
        
        lines.append(line_data)
    
    # Сортируем строки по Y
    lines.sort(key=lambda x: x['center_y'])
    
    return lines


def find_by_label(
    ocr_output: List[Dict],
    labels_list: List[str],
    max_distance: float = 300,
    prefer_right: bool = True,
    prefer_below: bool = False
) -> Optional[Tuple[str, str, Dict]]:
    """
    Ищет метку и извлекает ближайшее значение.
    
    Args:
        ocr_output: Результаты OCR
        labels_list: Список возможных меток
        max_distance: Максимальное расстояние для поиска значения
        prefer_right: Предпочитать значения справа
        prefer_below: Предпочитать значения снизу
    
    Returns:
        Кортеж (найденная метка, значение, bbox) или None
    """
    # Нормализуем метки для поиска
    normalized_labels = [label.lower().replace(':', '').strip() for label in labels_list]
    
    for i, item in enumerate(ocr_output):
        item_text = item.get('text', '').lower().replace(':', '').strip()
        
        # Проверяем, является ли текст меткой
        for j, norm_label in enumerate(normalized_labels):
            if norm_label in item_text or item_text in norm_label:
                # Нашли метку, ищем значение
                candidates = []
                
                for k, other in enumerate(ocr_output):
                    if k == i:
                        continue
                    
                    distance = calculate_distance(item, other)
                    if distance > max_distance:
                        continue
                    
                    # Вычисляем приоритет
                    priority = distance
                    
                    if prefer_right and is_right_of(item, other):
                        priority *= 0.5  # Уменьшаем приоритет (лучше)
                    
                    if prefer_below and is_below_of(item, other):
                        priority *= 0.5
                    
                    # Бонус за нахождение на той же строке
                    if is_same_line(item, other):
                        priority *= 0.3
                    
                    candidates.append((priority, other))
                
                if candidates:
                    # Выбираем лучшего кандидата
                    candidates.sort(key=lambda x: x[0])
                    best_candidate = candidates[0][1]
                    
                    return (
                        labels_list[j],
                        best_candidate.get('text', ''),
                        best_candidate
                    )
    
    return None


def normalize_date(date_str: str) -> Optional[str]:
    """
    Нормализует дату в формат YYYY-MM-DD.
    
    Args:
        date_str: Строка с датой
    
    Returns:
        Нормализованная дата или None
    """
    if not date_str:
        return None
    
    # Пробуем различные форматы
    # DD.MM.YYYY
    match = re.search(REGEX_PATTERNS['date_dmy'], date_str)
    if match:
        day, month, year = match.groups()
        try:
            date = datetime(int(year), int(month), int(day))
            return date.strftime('%Y-%m-%d')
        except ValueError:
            pass
    
    # YYYY-MM-DD
    match = re.search(REGEX_PATTERNS['date_ymd'], date_str)
    if match:
        year, month, day = match.groups()
        try:
            date = datetime(int(year), int(month), int(day))
            return date.strftime('%Y-%m-%d')
        except ValueError:
            pass
    
    # DD месяц YYYY
    match = re.search(REGEX_PATTERNS['date_text'], date_str, re.IGNORECASE)
    if match:
        day, month_name, year = match.groups()
        month = MONTHS.get(month_name.lower())
        if month:
            try:
                date = datetime(int(year), month, int(day))
                return date.strftime('%Y-%m-%d')
            except ValueError:
                pass
    
    return None


def normalize_sum(sum_str: str) -> Optional[float]:
    """
    Нормализует сумму в число.
    
    Args:
        sum_str: Строка с суммой
    
    Returns:
        Число или None
    """
    if not sum_str:
        return None
    
    # Извлекаем числовую часть
    match = re.search(REGEX_PATTERNS['sum'], sum_str)
    if match:
        number_str = match.group(1)
        # Убираем пробелы и неразрывные пробелы
        number_str = re.sub(r'[\s\u00A0]', '', number_str)
        # Заменяем запятую на точку
        number_str = number_str.replace(',', '.')
        
        try:
            return float(number_str)
        except ValueError:
            pass
    
    return None


def normalize_fio(fio_str: str) -> Optional[str]:
    """
    Нормализует ФИО.
    
    Args:
        fio_str: Строка с ФИО
    
    Returns:
        Нормализованное ФИО или None
    """
    if not fio_str:
        return None
    
    # Ищем паттерн ФИО
    match = re.search(REGEX_PATTERNS['fio'], fio_str)
    if match:
        parts = [part for part in match.groups() if part]
        return ' '.join(parts)
    
    # Если не нашли по регулярке, пробуем простую эвристику
    words = fio_str.split()
    fio_words = []
    
    for word in words:
        # Проверяем, что слово начинается с заглавной буквы и содержит кириллицу
        if word and word[0].isupper() and re.match(r'^[А-ЯЁа-яё]+$', word):
            fio_words.append(word)
            if len(fio_words) == 3:  # Максимум 3 слова для ФИО
                break
    
    if len(fio_words) >= 2:  # Минимум фамилия и имя
        return ' '.join(fio_words)
    
    return None


def regex_fallback(raw_text: str) -> Dict[str, Any]:
    """
    Извлекает поля с помощью регулярных выражений из сырого текста.
    
    Args:
        raw_text: Полный текст документа
    
    Returns:
        Словарь с найденными полями
    """
    results = {}
    
    # Даты
    dates = []
    for pattern_name in ['date_dmy', 'date_ymd', 'date_text']:
        for match in re.finditer(REGEX_PATTERNS[pattern_name], raw_text, re.IGNORECASE):
            date_str = match.group(0)
            normalized = normalize_date(date_str)
            if normalized:
                dates.append(normalized)
    
    if dates:
        results['dates'] = dates
        results['date'] = dates[0]  # Берем первую найденную дату
    
    # Суммы
    sums = []
    for match in re.finditer(REGEX_PATTERNS['sum'], raw_text):
        sum_str = match.group(0)
        normalized = normalize_sum(sum_str)
        if normalized:
            sums.append(normalized)
    
    if sums:
        results['sums'] = sums
        results['sum'] = max(sums)  # Берем максимальную сумму
    
    # ФИО
    fios = []
    for match in re.finditer(REGEX_PATTERNS['fio'], raw_text):
        fio_str = match.group(0)
        normalized = normalize_fio(fio_str)
        if normalized:
            fios.append(normalized)
    
    if fios:
        results['fios'] = fios
        results['fio'] = fios[0]  # Берем первое найденное ФИО
    
    # Телефоны
    phones = []
    for match in re.finditer(REGEX_PATTERNS['phone'], raw_text):
        phone = re.sub(r'[^\d+]', '', match.group(0))
        if len(phone) >= 10:
            phones.append(phone)
    
    if phones:
        results['phones'] = phones
        results['phone'] = phones[0]
    
    # Email
    emails = []
    for match in re.finditer(REGEX_PATTERNS['email'], raw_text, re.IGNORECASE):
        emails.append(match.group(0).lower())
    
    if emails:
        results['emails'] = emails
        results['email'] = emails[0]
    
    # ИНН
    inns = []
    for match in re.finditer(REGEX_PATTERNS['inn'], raw_text):
        inn = match.group(0)
        if len(inn) in [10, 12]:
            inns.append(inn)
    
    if inns:
        results['inns'] = inns
        results['inn'] = inns[0]
    
    # Номера договоров
    contract_numbers = []
    for match in re.finditer(REGEX_PATTERNS['contract_number'], raw_text):
        number = match.group(1) if match.groups() else match.group(0)
        contract_numbers.append(number)
    
    if contract_numbers:
        results['contract_numbers'] = contract_numbers
        results['contract_number'] = contract_numbers[0]
    
    # Банковские счета
    accounts = []
    for match in re.finditer(REGEX_PATTERNS['account'], raw_text):
        accounts.append(match.group(0))
    
    if accounts:
        results['accounts'] = accounts
        results['account'] = accounts[0]
    
    return results


def detect_document_type(raw_text: str, extracted_fields: Dict) -> str:
    """
    Определяет тип документа на основе текста и извлеченных полей.
    
    Args:
        raw_text: Полный текст документа
        extracted_fields: Извлеченные поля
    
    Returns:
        Предполагаемый тип документа
    """
    text_lower = raw_text.lower()
    
    # Ключевые слова для различных типов документов
    doc_types = {
        'договор': ['договор', 'контракт', 'соглашение'],
        'счет': ['счет на оплату', 'счёт', 'invoice'],
        'акт': ['акт выполненных работ', 'акт оказанных услуг', 'акт приема-передачи'],
        'заявление': ['заявление', 'заявка', 'обращение'],
        'доверенность': ['доверенность', 'доверяю', 'уполномочиваю'],
        'паспорт': ['паспорт', 'удостоверение личности'],
        'справка': ['справка', 'выписка', 'подтверждение'],
        'квитанция': ['квитанция', 'чек', 'оплата'],
        'накладная': ['накладная', 'товарная накладная', 'торг-12']
    }
    
    # Подсчитываем вхождения ключевых слов
    scores = {}
    for doc_type, keywords in doc_types.items():
        score = sum(1 for keyword in keywords if keyword in text_lower)
        if score > 0:
            scores[doc_type] = score
    
    # Дополнительные эвристики на основе полей
    if extracted_fields.get('inn') and extracted_fields.get('account'):
        scores['счет'] = scores.get('счет', 0) + 2
    
    if extracted_fields.get('contract_number'):
        scores['договор'] = scores.get('договор', 0) + 1
    
    if extracted_fields.get('passport'):
        scores['паспорт'] = scores.get('паспорт', 0) + 3
    
    # Выбираем тип с максимальным счетом
    if scores:
        return max(scores, key=scores.get)
    
    return 'неизвестный'


def extract_fields_from_paddle(ocr_output: List[Dict], image=None) -> Dict:
    """
    Основная функция извлечения полей из результатов PaddleOCR.
    
    Args:
        ocr_output: Результаты OCR от PaddleOCR
        image: Опциональное изображение для дополнительного анализа
    
    Returns:
        Словарь с извлеченными полями
    """
    if not ocr_output:
        return {
            'status': 'error',
            'message': 'Пустой OCR вывод',
            'fields': {}
        }
    
    # Собираем полный текст
    raw_text = ' '.join(item.get('text', '') for item in ocr_output)
    
    # Группируем в строки
    lines = group_lines_by_y(ocr_output)
    
    # Результаты извлечения
    extracted = {
        'raw_text': raw_text,
        'total_items': len(ocr_output),
        'total_lines': len(lines)
    }
    
    # Средняя уверенность
    confidences = [item.get('conf', 0) for item in ocr_output if item.get('conf')]
    if confidences:
        extracted['confidence_summary'] = {
            'avg': sum(confidences) / len(confidences),
            'min': min(confidences),
            'max': max(confidences)
        }
    
    # Поиск по меткам
    fields_found = {}
    
    # ФИО
    fio_result = find_by_label(ocr_output, FIELD_LABELS['fio'], prefer_right=True)
    if fio_result:
        fio_normalized = normalize_fio(fio_result[1])
        if fio_normalized:
            fields_found['fio'] = fio_normalized
    
    # Дата
    date_result = find_by_label(ocr_output, FIELD_LABELS['date'], prefer_right=True)
    if date_result:
        date_normalized = normalize_date(date_result[1])
        if date_normalized:
            fields_found['date'] = date_normalized
    
    # Сумма
    sum_result = find_by_label(ocr_output, FIELD_LABELS['sum'], prefer_right=True)
    if sum_result:
        sum_normalized = normalize_sum(sum_result[1])
        if sum_normalized:
            fields_found['sum'] = sum_normalized
    
    # Номер договора
    contract_result = find_by_label(ocr_output, FIELD_LABELS['contract_number'], prefer_right=True)
    if contract_result:
        fields_found['contract_number'] = contract_result[1]
    
    # Счет
    account_result = find_by_label(ocr_output, FIELD_LABELS['account'], prefer_right=True, prefer_below=True)
    if account_result:
        # Очищаем от лишних символов
        account_clean = re.sub(r'[^\d]', '', account_result[1])
        if len(account_clean) == 20:
            fields_found['account'] = account_clean
    
    # Применяем regex fallback для полей, которые не нашли по меткам
    regex_results = regex_fallback(raw_text)
    
    # Объединяем результаты (приоритет у найденных по меткам)
    for field_name in ['fio', 'date', 'sum', 'contract_number', 'account', 'phone', 'email', 'inn']:
        if field_name not in fields_found and field_name in regex_results:
            fields_found[field_name] = regex_results[field_name]
    
    # Определяем тип документа
    doc_type = detect_document_type(raw_text, fields_found)
    fields_found['doc_type'] = doc_type
    
    # Формируем финальный результат
    extracted.update(fields_found)
    
    return extracted


def test_extraction(ocr_output: List[Dict], output_file: Optional[str] = None):
    """
    Тестирует извлечение полей и выводит подробную информацию.
    
    Args:
        ocr_output: Результаты OCR
        output_file: Файл для сохранения результатов тестирования
    """
    print("=" * 80)
    print("ТЕСТИРОВАНИЕ ИЗВЛЕЧЕНИЯ ПОЛЕЙ")
    print("=" * 80)
    
    # Извлекаем поля
    result = extract_fields_from_paddle(ocr_output)
    
    # Выводим raw_text
    print("\n1. ПОЛНЫЙ ТЕКСТ:")
    print("-" * 40)
    print(result.get('raw_text', '')[:500] + "..." if len(result.get('raw_text', '')) > 500 else result.get('raw_text', ''))
    
    # Выводим строки с координатами
    print("\n2. СТРОКИ С КООРДИНАТАМИ:")
    print("-" * 40)
    lines = group_lines_by_y(ocr_output)
    for i, line in enumerate(lines[:10]):  # Первые 10 строк
        print(f"Строка {i+1}: Y={line['center_y']:.0f}, X={line['center_x']:.0f}")
        print(f"  Текст: {line['text'][:60]}..." if len(line['text']) > 60 else f"  Текст: {line['text']}")
        print(f"  Bbox: [{line['left']:.0f}, {line['top']:.0f}, {line['right']:.0f}, {line['bottom']:.0f}]")
    
    if len(lines) > 10:
        print(f"... и еще {len(lines) - 10} строк")
    
    # Выводим найденные поля
    print("\n3. ИЗВЛЕЧЕННЫЕ ПОЛЯ:")
    print("-" * 40)
    
    field_names = ['fio', 'date', 'sum', 'doc_type', 'contract_number', 'account', 'phone', 'email', 'inn']
    
    for field in field_names:
        if field in result:
            value = result[field]
            print(f"  {field.upper():15} : {value}")
    
    # Выводим статистику confidence
    if 'confidence_summary' in result:
        print("\n4. СТАТИСТИКА УВЕРЕННОСТИ:")
        print("-" * 40)
        conf = result['confidence_summary']
        print(f"  Средняя: {conf['avg']:.2f}")
        print(f"  Минимум: {conf['min']:.2f}")
        print(f"  Максимум: {conf['max']:.2f}")
    
    # Сохраняем результаты если указан файл
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        test_report = {
            'timestamp': datetime.now().isoformat(),
            'extraction_results': result,
            'lines': [
                {
                    'index': i,
                    'text': line['text'],
                    'center_y': line['center_y'],
                    'center_x': line['center_x'],
                    'bbox': [line['left'], line['top'], line['right'], line['bottom']]
                }
                for i, line in enumerate(lines)
            ]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(test_report, f, ensure_ascii=False, indent=2)
        
        print(f"\n5. РЕЗУЛЬТАТЫ СОХРАНЕНЫ В: {output_path}")
    
    print("=" * 80)
    
    return result


def visualize_extraction(image_path: str, ocr_output: List[Dict], extracted_fields: Dict, output_path: Optional[str] = None):
    """
    Визуализирует результаты извлечения полей на изображении.
    
    Args:
        image_path: Путь к исходному изображению
        ocr_output: Результаты OCR
        extracted_fields: Извлеченные поля
        output_path: Путь для сохранения визуализации
    """
    try:
        import cv2
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("Для визуализации необходимы библиотеки cv2 и PIL")
        return
    
    # Загружаем изображение
    image = cv2.imread(image_path)
    if image is None:
        image = np.array(Image.open(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    vis_image = image.copy()
    
    # Цвета для разных типов полей
    field_colors = {
        'fio': (0, 255, 0),      # Зеленый
        'date': (255, 0, 0),      # Синий
        'sum': (0, 0, 255),       # Красный
        'contract_number': (255, 255, 0),  # Голубой
        'account': (255, 0, 255),  # Фиолетовый
        'phone': (0, 255, 255),   # Желтый
        'email': (128, 255, 0),   # Салатовый
        'inn': (255, 128, 0)      # Оранжевый
    }
    
    # Находим bbox для каждого извлеченного поля
    for field_name, field_value in extracted_fields.items():
        if field_name in field_colors and isinstance(field_value, str):
            # Ищем соответствующий bbox в OCR результатах
            for item in ocr_output:
                if field_value in item.get('text', ''):
                    # Рисуем bbox
                    if 'box' in item:
                        pts = np.array(item['box'], dtype=np.int32)
                        cv2.polylines(vis_image, [pts], True, field_colors[field_name], 2)
                    else:
                        x = int(item.get('left', 0))
                        y = int(item.get('top', 0))
                        w = int(item.get('width', 0))
                        h = int(item.get('height', 0))
                        cv2.rectangle(vis_image, (x, y), (x + w, y + h), field_colors[field_name], 2)
                    
                    # Добавляем подпись
                    cv2.putText(vis_image, field_name, 
                               (int(item.get('left', 0)), int(item.get('top', 0)) - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, field_colors[field_name], 2)
    
    # Сохраняем или показываем результат
    if output_path:
        cv2.imwrite(output_path, vis_image)
        print(f"Визуализация сохранена в: {output_path}")
    else:
        cv2.imshow('Extracted Fields', vis_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return vis_image


def process_document_with_extraction(
    image_path: str,
    ocr_function=None,
    output_dir: str = "./output",
    save_visualization: bool = False
) -> Dict:
    """
    Полный пайплайн обработки документа: OCR + извлечение полей.
    
    Args:
        image_path: Путь к изображению документа
        ocr_function: Функция OCR (если None, пытается импортировать PaddleOCR)
        output_dir: Директория для сохранения результатов
        save_visualization: Сохранять ли визуализацию
    
    Returns:
        Словарь с результатами обработки
    """
    results = {}
    input_file = Path(image_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    base_name = input_file.stem
    
    print(f"Обработка документа: {input_file}")
    
    # Выполняем OCR
    if ocr_function is None:
        try:
            from ocr_paddle import run_paddle
            ocr_function = run_paddle
        except ImportError:
            print("Не удалось импортировать PaddleOCR. Установите модуль ocr_paddle.py")
            return results
    
    print("1. Выполнение OCR...")
    ocr_output = ocr_function(str(image_path))
    results['ocr_items'] = len(ocr_output)
    
    # Извлекаем поля
    print("2. Извлечение полей...")
    extraction_results = extract_fields_from_paddle(ocr_output)
    results['extracted_fields'] = extraction_results
    
    # Сохраняем результаты
    results_file = output_path / f"{base_name}_extraction.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(extraction_results, f, ensure_ascii=False, indent=2)
    print(f"3. Результаты сохранены в: {results_file}")
    
    # Визуализация
    if save_visualization:
        vis_file = output_path / f"{base_name}_extraction_vis.jpg"
        visualize_extraction(str(image_path), ocr_output, extraction_results, str(vis_file))
        results['visualization'] = str(vis_file)
    
    # Создаем текстовый отчет
    report_file = output_path / f"{base_name}_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("ОТЧЕТ ОБ ИЗВЛЕЧЕНИИ ПОЛЕЙ\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Документ: {input_file}\n")
        f.write(f"Дата обработки: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("ИЗВЛЕЧЕННЫЕ ПОЛЯ:\n")
        f.write("-" * 30 + "\n")
        for field in ['doc_type', 'fio', 'date', 'sum', 'contract_number', 'account', 'phone', 'email', 'inn']:
            if field in extraction_results:
                f.write(f"{field.upper():15} : {extraction_results[field]}\n")
        
        if 'confidence_summary' in extraction_results:
            f.write("\nСТАТИСТИКА УВЕРЕННОСТИ:\n")
            f.write("-" * 30 + "\n")
            conf = extraction_results['confidence_summary']
            f.write(f"Средняя: {conf['avg']:.2f}\n")
            f.write(f"Минимум: {conf['min']:.2f}\n")
            f.write(f"Максимум: {conf['max']:.2f}\n")
    
    print(f"4. Отчет сохранен в: {report_file}")
    results['report'] = str(report_file)
    
    return results


def extract_fields_from_tesseract(tesseract_data: List[Dict], image=None) -> Dict:
    """
    Извлечение полей из результатов Tesseract OCR.

    Args:
        tesseract_data: Результаты OCR от Tesseract с bbox и confidence
        image: Опциональное изображение для дополнительного анализа

    Returns:
        Словарь с извлеченными полями
    """
    if not tesseract_data:
        return {
            'status': 'error',
            'message': 'Пустой OCR вывод от Tesseract',
            'fields': {}
        }

    # Преобразуем формат Tesseract в формат, совместимый с PaddleOCR
    paddle_compatible = []
    for item in tesseract_data:
        paddle_item = {
            'text': item.get('text', ''),
            'left': item.get('left', 0),
            'top': item.get('top', 0),
            'width': item.get('width', 0),
            'height': item.get('height', 0),
            'conf': item.get('conf', 0) / 100.0,  # Tesseract возвращает 0-100, приводим к 0-1
            'center_x': item.get('left', 0) + item.get('width', 0) / 2,
            'center_y': item.get('top', 0) + item.get('height', 0) / 2
        }
        paddle_compatible.append(paddle_item)

    # Используем существующую функцию для PaddleOCR
    return extract_fields_from_paddle(paddle_compatible, image)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Извлечение полей из результатов OCR")
    parser.add_argument("input", help="Путь к изображению или JSON с OCR результатами")
    parser.add_argument("-o", "--output", default="./output", help="Директория для результатов")
    parser.add_argument("--test", action="store_true", help="Режим тестирования с подробным выводом")
    parser.add_argument("--visualize", action="store_true", help="Создать визуализацию")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    if input_path.suffix.lower() == '.json':
        # Загружаем OCR результаты из JSON
        with open(input_path, 'r', encoding='utf-8') as f:
            ocr_output = json.load(f)
        
        if args.test:
            test_extraction(ocr_output, args.output + "/test_report.json")
        else:
            results = extract_fields_from_paddle(ocr_output)
            print("\nИзвлеченные поля:")
            for key, value in results.items():
                if key not in ['raw_text', 'total_items', 'total_lines']:
                    print(f"  {key}: {value}")
    
    else:
        # Обрабатываем изображение
        results = process_document_with_extraction(
            str(input_path),
            output_dir=args.output,
            save_visualization=args.visualize
        )
        
        print("\nРезультаты обработки:")
        if 'extracted_fields' in results:
            for field in ['doc_type', 'fio', 'date', 'sum', 'contract_number', 'account']:
                if field in results['extracted_fields']:
                    print(f"  {field}: {results['extracted_fields'][field]}")