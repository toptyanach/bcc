"""
postprocess.py - Постобработка результатов OCR
Включает LLM обработку, правила и улучшение качества извлеченных данных
"""

import re
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
import requests
import os


def call_llm_to_json(raw_text: str, model_type: str = 'openai') -> Dict[str, Any]:
    """
    Обработка текста с помощью LLM для извлечения структурированных данных

    Args:
        raw_text: Сырой текст от OCR
        model_type: Тип модели ('openai', 'local', 'mock')

    Returns:
        Словарь с извлеченными полями
    """
    if not raw_text or not raw_text.strip():
        return {
            'error': 'Пустой входной текст',
            'success': False
        }

    # Проверяем какой тип LLM использовать
    if model_type == 'openai' and os.getenv('OPENAI_API_KEY'):
        return _call_openai_api(raw_text)
    elif model_type == 'local':
        return _call_local_llm(raw_text)
    else:
        # Fallback на правила-базированную обработку
        print("⚠️ LLM недоступен, используем правила-базированную обработку")
        return _mock_llm_response(raw_text)


def _call_openai_api(text: str) -> Dict[str, Any]:
    """Вызов OpenAI API для обработки текста"""
    try:
        import openai

        client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

        prompt = f"""
        Проанализируй следующий текст документа и извлеки структурированную информацию в JSON формате.
        Найди и верни следующие поля если они есть:
        - fio: полное имя человека
        - date: любые даты в формате YYYY-MM-DD
        - sum: денежные суммы как числа
        - contract_number: номера договоров/контрактов
        - phone: номера телефонов
        - email: адреса электронной почты
        - inn: ИНН (10 или 12 цифр)
        - address: адреса
        - doc_type: тип документа (договор, счет, заявление и т.д.)

        Текст документа:
        {text}

        Ответ в формате JSON:
        """

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Ты эксперт по извлечению данных из документов. Отвечай только в JSON формате."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0
        )

        result_text = response.choices[0].message.content

        # Пытаемся распарсить JSON
        try:
            result = json.loads(result_text)
            result['llm_processed'] = True
            result['llm_model'] = 'openai'
            result['success'] = True
            return result
        except json.JSONDecodeError:
            # Если JSON невалидный, возвращаем как есть
            return {
                'raw_llm_response': result_text,
                'llm_processed': True,
                'llm_model': 'openai',
                'success': True,
                'error': 'Невалидный JSON от LLM'
            }

    except Exception as e:
        return {
            'error': f'Ошибка OpenAI API: {str(e)}',
            'success': False,
            'llm_processed': False
        }


def _call_local_llm(text: str) -> Dict[str, Any]:
    """Вызов локальной LLM модели"""
    try:
        # Пример вызова локальной модели (Ollama, LocalAI и т.д.)
        # Здесь должен быть код для вашей локальной модели

        # Пример для Ollama
        response = requests.post('http://localhost:11434/api/generate', json={
            'model': 'llama2',  # или другая модель
            'prompt': f'''
            Extract structured data from this document text as JSON:

            {text}

            Return JSON with fields: fio, date, sum, phone, email, inn, doc_type
            ''',
            'stream': False
        }, timeout=30)

        if response.status_code == 200:
            result = response.json()
            llm_text = result.get('response', '')

            # Пытаемся извлечь JSON из ответа
            json_match = re.search(r'\{.*\}', llm_text, re.DOTALL)
            if json_match:
                try:
                    parsed = json.loads(json_match.group())
                    parsed['llm_processed'] = True
                    parsed['llm_model'] = 'local'
                    parsed['success'] = True
                    return parsed
                except json.JSONDecodeError:
                    pass

            return {
                'raw_llm_response': llm_text,
                'llm_processed': True,
                'llm_model': 'local',
                'success': True
            }

    except Exception as e:
        return {
            'error': f'Ошибка локальной LLM: {str(e)}',
            'success': False,
            'llm_processed': False
        }


def _mock_llm_response(text: str) -> Dict[str, Any]:
    """Имитация LLM ответа с помощью правил"""
    # Используем простые регулярные выражения для извлечения данных
    result = {
        'llm_processed': False,
        'llm_model': 'mock_rules',
        'success': True
    }

    # Даты
    date_pattern = r'\b(\d{1,2})[.\-/](\d{1,2})[.\-/](\d{4})\b'
    date_matches = re.findall(date_pattern, text)
    if date_matches:
        try:
            day, month, year = date_matches[0]
            date_obj = datetime(int(year), int(month), int(day))
            result['date'] = date_obj.strftime('%Y-%m-%d')
        except ValueError:
            pass

    # Суммы
    sum_pattern = r'(\d{1,3}(?:[\s,]\d{3})*(?:[.,]\d{1,2})?)\s*(?:руб|рублей|р\.|₽)'
    sum_matches = re.findall(sum_pattern, text, re.IGNORECASE)
    if sum_matches:
        try:
            sum_str = sum_matches[0].replace(' ', '').replace(',', '.')
            result['sum'] = float(sum_str)
        except ValueError:
            pass

    # ФИО (простое извлечение)
    fio_pattern = r'\b([А-ЯЁ][а-яё]+)\s+([А-ЯЁ][а-яё]+)(?:\s+([А-ЯЁ][а-яё]+))?\b'
    fio_matches = re.findall(fio_pattern, text)
    if fio_matches:
        fio_parts = [part for part in fio_matches[0] if part]
        result['fio'] = ' '.join(fio_parts)

    # Телефон
    phone_pattern = r'(?:\+7|8|7)?[\s\-]?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}'
    phone_matches = re.findall(phone_pattern, text)
    if phone_matches:
        phone = re.sub(r'[^\d+]', '', phone_matches[0])
        if len(phone) >= 10:
            result['phone'] = phone

    # Email
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    email_matches = re.findall(email_pattern, text, re.IGNORECASE)
    if email_matches:
        result['email'] = email_matches[0].lower()

    # ИНН
    inn_pattern = r'\b\d{10}\b|\b\d{12}\b'
    inn_matches = re.findall(inn_pattern, text)
    if inn_matches:
        result['inn'] = inn_matches[0]

    # Тип документа (простая эвристика)
    doc_keywords = {
        'договор': ['договор', 'контракт', 'соглашение'],
        'счет': ['счет', 'invoice', 'фактура'],
        'заявление': ['заявление', 'заявка'],
        'акт': ['акт'],
        'справка': ['справка', 'выписка']
    }

    text_lower = text.lower()
    for doc_type, keywords in doc_keywords.items():
        if any(keyword in text_lower for keyword in keywords):
            result['doc_type'] = doc_type
            break

    return result


def rules_based_postprocess(extracted_data: Dict, raw_text: str) -> Dict[str, Any]:
    """
    Правила-базированная постобработка извлеченных данных

    Args:
        extracted_data: Данные извлеченные из OCR
        raw_text: Сырой текст для дополнительного анализа

    Returns:
        Очищенные и улучшенные данные
    """
    result = extracted_data.copy() if extracted_data else {}

    # Очистка и валидация данных
    result = _clean_and_validate_fields(result)

    # Дополнительное извлечение из raw_text если что-то пропущено
    result = _extract_missing_fields(result, raw_text)

    # Применение бизнес-правил
    result = _apply_business_rules(result)

    result['postprocessed'] = True
    result['postprocess_method'] = 'rules_based'

    return result


def _clean_and_validate_fields(data: Dict) -> Dict[str, Any]:
    """Очистка и валидация полей"""
    cleaned = {}

    for field, value in data.items():
        if value is None or value == '':
            continue

        # Очистка ФИО
        if field == 'fio' and isinstance(value, str):
            # Убираем лишние пробелы, приводим к нормальному регистру
            fio_clean = ' '.join(word.capitalize() for word in value.split() if word.isalpha())
            if len(fio_clean.split()) >= 2:  # Минимум имя и фамилия
                cleaned[field] = fio_clean

        # Очистка телефонов
        elif field == 'phone' and isinstance(value, str):
            phone_clean = re.sub(r'[^\d+]', '', value)
            if len(phone_clean) >= 10:
                # Приводим к стандартному формату
                if phone_clean.startswith('8'):
                    phone_clean = '+7' + phone_clean[1:]
                elif phone_clean.startswith('7'):
                    phone_clean = '+' + phone_clean
                elif not phone_clean.startswith('+'):
                    phone_clean = '+7' + phone_clean
                cleaned[field] = phone_clean

        # Валидация email
        elif field == 'email' and isinstance(value, str):
            email_pattern = r'^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}$'
            if re.match(email_pattern, value, re.IGNORECASE):
                cleaned[field] = value.lower()

        # Валидация ИНН
        elif field == 'inn' and isinstance(value, str):
            inn_clean = re.sub(r'[^\d]', '', value)
            if len(inn_clean) in [10, 12]:
                cleaned[field] = inn_clean

        # Валидация дат
        elif field == 'date' and isinstance(value, str):
            try:
                # Пытаемся парсить дату
                if re.match(r'^\d{4}-\d{2}-\d{2}$', value):
                    datetime.strptime(value, '%Y-%m-%d')
                    cleaned[field] = value
                else:
                    # Пытаемся конвертировать другие форматы
                    normalized_date = _normalize_date_string(value)
                    if normalized_date:
                        cleaned[field] = normalized_date
            except ValueError:
                pass

        # Валидация сумм
        elif field == 'sum' and (isinstance(value, (int, float)) or isinstance(value, str)):
            try:
                if isinstance(value, str):
                    sum_clean = re.sub(r'[^\d.,]', '', value).replace(',', '.')
                    value = float(sum_clean)
                if value > 0:
                    cleaned[field] = value
            except ValueError:
                pass

        else:
            # Остальные поля копируем как есть
            cleaned[field] = value

    return cleaned


def _extract_missing_fields(data: Dict, raw_text: str) -> Dict[str, Any]:
    """Дополнительное извлечение пропущенных полей"""
    result = data.copy()

    # Если не найден адрес, пытаемся извлечь
    if 'address' not in result:
        address_keywords = ['адрес', 'проживает', 'зарегистрирован', 'местонахождение']
        for keyword in address_keywords:
            pattern = rf'{keyword}[:\s]+([^,\n]+(?:,[^,\n]+)*)'
            match = re.search(pattern, raw_text, re.IGNORECASE)
            if match:
                result['address'] = match.group(1).strip()
                break

    # Дополнительные номера документов
    if 'contract_number' not in result:
        # Ищем различные варианты номеров
        number_patterns = [
            r'№\s*(\d+(?:[-/]\d+)*)',
            r'номер[:\s]+(\d+(?:[-/]\d+)*)',
            r'договор[:\s]+№?\s*(\d+(?:[-/]\d+)*)'
        ]

        for pattern in number_patterns:
            match = re.search(pattern, raw_text, re.IGNORECASE)
            if match:
                result['contract_number'] = match.group(1)
                break

    return result


def _apply_business_rules(data: Dict) -> Dict[str, Any]:
    """Применение бизнес-правил и логических проверок"""
    result = data.copy()

    # Правило: если есть ИНН и счет, скорее всего это коммерческий документ
    if result.get('inn') and result.get('account'):
        if 'doc_type' not in result:
            result['doc_type'] = 'коммерческий_документ'

    # Правило: если есть большая сумма, добавляем валюту
    if 'sum' in result and isinstance(result['sum'], (int, float)):
        if result['sum'] > 1000:
            result['currency'] = 'RUB'
        result['sum_formatted'] = f"{result['sum']:,.2f} ₽"

    # Правило: нормализация типов документов
    if 'doc_type' in result:
        doc_type_mapping = {
            'договор': ['договор', 'контракт', 'соглашение'],
            'счет': ['счет', 'счёт', 'invoice', 'фактура'],
            'заявление': ['заявление', 'заявка', 'обращение'],
            'справка': ['справка', 'выписка', 'подтверждение']
        }

        current_type = result['doc_type'].lower()
        for normalized_type, variants in doc_type_mapping.items():
            if any(variant in current_type for variant in variants):
                result['doc_type'] = normalized_type
                break

    # Добавляем метаданные
    result['extraction_confidence'] = _calculate_extraction_confidence(result)
    result['fields_extracted'] = len([k for k, v in result.items() if v and not k.startswith('_')])

    return result


def _normalize_date_string(date_str: str) -> Optional[str]:
    """Нормализация строки даты в формат YYYY-MM-DD"""
    if not date_str:
        return None

    # Различные форматы дат
    date_formats = [
        '%d.%m.%Y', '%d/%m/%Y', '%Y-%m-%d', '%d-%m-%Y',
        '%d.%m.%y', '%d/%m/%y', '%y-%m-%d'
    ]

    for fmt in date_formats:
        try:
            date_obj = datetime.strptime(date_str.strip(), fmt)
            return date_obj.strftime('%Y-%m-%d')
        except ValueError:
            continue

    return None


def _calculate_extraction_confidence(data: Dict) -> float:
    """Вычисление уверенности извлечения на основе найденных полей"""
    important_fields = ['fio', 'date', 'sum', 'phone', 'email']
    found_important = sum(1 for field in important_fields if field in data and data[field])

    total_fields = len([k for k, v in data.items() if v and not k.startswith('_')])

    # Базовая уверенность
    confidence = found_important / len(important_fields) * 0.7

    # Бонус за общее количество полей
    confidence += min(total_fields / 10, 0.3)

    return min(confidence, 1.0)


# Дополнительные утилиты для постобработки

def merge_extraction_results(ocr_result: Dict, llm_result: Dict) -> Dict[str, Any]:
    """
    Слияние результатов OCR и LLM обработки

    Args:
        ocr_result: Результат извлечения из OCR
        llm_result: Результат LLM обработки

    Returns:
        Объединенный результат с приоритетом LLM
    """
    merged = ocr_result.copy() if ocr_result else {}

    if llm_result:
        # LLM результаты имеют приоритет для основных полей
        priority_fields = ['fio', 'date', 'sum', 'doc_type', 'phone', 'email']

        for field in priority_fields:
            if field in llm_result and llm_result[field]:
                merged[field] = llm_result[field]
                merged[f'{field}_source'] = 'llm'
            elif field in merged and merged[field]:
                merged[f'{field}_source'] = 'ocr'

        # Добавляем остальные поля из LLM
        for field, value in llm_result.items():
            if field not in merged and value:
                merged[field] = value
                merged[f'{field}_source'] = 'llm'

    # Добавляем метаданные о слиянии
    merged['merged'] = True
    merged['merge_timestamp'] = datetime.now().isoformat()

    return merged


def validate_extraction_quality(result: Dict) -> Dict[str, Any]:
    """
    Валидация качества извлечения данных

    Args:
        result: Результат извлечения

    Returns:
        Словарь с оценками качества
    """
    quality_report = {
        'overall_quality': 'unknown',
        'field_quality': {},
        'recommendations': []
    }

    # Проверяем качество отдельных полей
    field_checks = {
        'fio': lambda x: len(x.split()) >= 2 if isinstance(x, str) else False,
        'phone': lambda x: len(re.sub(r'[^\d]', '', str(x))) >= 10,
        'email': lambda x: '@' in str(x) and '.' in str(x),
        'inn': lambda x: len(re.sub(r'[^\d]', '', str(x))) in [10, 12],
        'date': lambda x: re.match(r'^\d{4}-\d{2}-\d{2}$', str(x)) is not None
    }

    total_score = 0
    checked_fields = 0

    for field, check_func in field_checks.items():
        if field in result and result[field]:
            is_valid = check_func(result[field])
            quality_report['field_quality'][field] = 'good' if is_valid else 'poor'

            if is_valid:
                total_score += 1
            else:
                quality_report['recommendations'].append(f'Проверьте поле {field}: {result[field]}')

            checked_fields += 1

    # Общая оценка
    if checked_fields == 0:
        quality_report['overall_quality'] = 'no_data'
    elif total_score / checked_fields >= 0.8:
        quality_report['overall_quality'] = 'excellent'
    elif total_score / checked_fields >= 0.6:
        quality_report['overall_quality'] = 'good'
    elif total_score / checked_fields >= 0.4:
        quality_report['overall_quality'] = 'fair'
    else:
        quality_report['overall_quality'] = 'poor'
        quality_report['recommendations'].append('Рекомендуется проверка и корректировка данных')

    quality_report['score'] = total_score / max(checked_fields, 1)

    return quality_report


if __name__ == "__main__":
    # Пример использования
    test_text = """
    Договор № 123/2024
    Иванов Иван Иванович
    Дата: 15.03.2024
    Сумма: 50 000 руб.
    Телефон: +7 (495) 123-45-67
    Email: ivan@example.com
    ИНН: 1234567890
    """

    print("🧪 Тестирование постобработки")

    # Тест rules-based обработки
    print("\n1. Rules-based постобработка:")
    result = rules_based_postprocess({}, test_text)
    print(json.dumps(result, ensure_ascii=False, indent=2))

    # Тест LLM обработки (mock)
    print("\n2. Mock LLM обработка:")
    llm_result = call_llm_to_json(test_text, model_type='mock')
    print(json.dumps(llm_result, ensure_ascii=False, indent=2))

    # Тест валидации качества
    print("\n3. Валидация качества:")
    quality = validate_extraction_quality(result)
    print(json.dumps(quality, ensure_ascii=False, indent=2))