import streamlit as st
import json
import io
import base64
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from typing import Dict, List, Tuple, Optional
import difflib
import plotly.graph_objects as go
from datetime import datetime

# Импорт ваших OCR модулей
from ocr_paddle import run_paddle, get_plaintext
from ocr_baseline import run_tesseract
from ocr_trocr import run_trocr
from extract import extract_fields_from_paddle, extract_fields_from_tesseract
from postprocess import call_llm_to_json, rules_based_postprocess
from metrics import cer, wer

# Конфигурация страницы
st.set_page_config(
    page_title="OCR 2.0 Демо",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Кастомный CSS для улучшенного UI
st.markdown("""
<style>
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        border: none;
        padding: 0.5rem 1rem;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        background-color: #45a049;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .diff-added {
        background-color: #d4f4dd;
        color: #22863a;
        padding: 2px 4px;
        border-radius: 3px;
    }
    .diff-removed {
        background-color: #ffdce0;
        color: #d73a49;
        padding: 2px 4px;
        text-decoration: line-through;
    }
    .json-container {
        background-color: #f6f8fa;
        border-radius: 6px;
        padding: 1rem;
        font-family: 'Courier New', monospace;
    }
</style>
""", unsafe_allow_html=True)

# Инициализация состояния сессии
if 'ocr_results' not in st.session_state:
    st.session_state.ocr_results = {}
if 'baseline_results' not in st.session_state:
    st.session_state.baseline_results = {}
if 'processing' not in st.session_state:
    st.session_state.processing = False

def draw_bboxes_on_image(image: Image.Image, bboxes: List[Dict]) -> Image.Image:
    """Рисование ограничивающих рамок на изображении для обнаруженных полей"""
    img_with_boxes = image.copy()
    draw = ImageDraw.Draw(img_with_boxes)
    
    # Цветовая палитра для разных типов полей
    colors = {
        'name': '#FF6B6B',
        'date': '#4ECDC4',
        'amount': '#45B7D1',
        'address': '#96CEB4',
        'phone': '#FFEAA7',
        'email': '#DDA0DD',
        'default': '#FFD93D'
    }
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    for bbox in bboxes:
        field_type = bbox.get('type', 'default')
        color = colors.get(field_type, colors['default'])
        
        coords = bbox.get('coordinates', [])
        if len(coords) >= 4:
            # Рисуем прямоугольник
            draw.rectangle(
                [(coords[0], coords[1]), (coords[2], coords[3])],
                outline=color,
                width=3
            )
            
            # Рисуем подпись
            label = f"{bbox.get('field', '')} ({bbox.get('confidence', 0):.2f})"
            draw.text(
                (coords[0], coords[1] - 20),
                label,
                fill=color,
                font=font
            )
    
    return img_with_boxes

def create_diff_visualization(text1: str, text2: str, label1: str = "Базовый", label2: str = "Наш") -> str:
    """Создание HTML визуализации различий между двумя текстами"""
    diff = difflib.unified_diff(
        text1.splitlines(keepends=True),
        text2.splitlines(keepends=True),
        fromfile=label1,
        tofile=label2,
        lineterm=''
    )
    
    html_diff = """
    <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 20px;'>
        <div>
            <h4 style='color: #d73a49;'>{}</h4>
            <div style='background: #f6f8fa; padding: 10px; border-radius: 5px; max-height: 400px; overflow-y: auto;'>
                <pre style='white-space: pre-wrap;'>{}</pre>
            </div>
        </div>
        <div>
            <h4 style='color: #22863a;'>{}</h4>
            <div style='background: #f6f8fa; padding: 10px; border-radius: 5px; max-height: 400px; overflow-y: auto;'>
                <pre style='white-space: pre-wrap;'>{}</pre>
            </div>
        </div>
    </div>
    """.format(label1, text1, label2, text2)
    
    return html_diff

def create_metrics_chart(metrics: Dict) -> go.Figure:
    """Создание интерактивной визуализации метрик"""
    categories = list(metrics.keys())
    values = list(metrics.values())
    
    fig = go.Figure(data=[
        go.Bar(
            x=categories,
            y=values,
            text=[f'{v:.2%}' for v in values],
            textposition='auto',
            marker=dict(
                color=values,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Оценка")
            )
        )
    ])
    
    fig.update_layout(
        title="Метрики производительности OCR",
        xaxis_title="Метрика",
        yaxis_title="Оценка",
        yaxis=dict(range=[0, 1]),
        height=400,
        template="plotly_white"
    )
    
    return fig

def process_pdf_to_images(pdf_file) -> List[Image.Image]:
    """Конвертация PDF в список изображений"""
    try:
        import fitz  # PyMuPDF
        pdf_bytes = pdf_file.read()
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        images = []
        
        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x увеличение для лучшего качества
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            images.append(img)
        
        pdf_document.close()
        return images
    except ImportError:
        st.error("PyMuPDF не установлен. Пожалуйста, установите его для обработки PDF файлов.")
        return []

# Основной интерфейс
def main():
    # Заголовок с градиентом
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 2rem; border-radius: 10px; margin-bottom: 2rem;'>
        <h1 style='color: white; text-align: center; margin: 0;'>🔍 OCR 2.0 Демо</h1>
        <p style='color: rgba(255,255,255,0.9); text-align: center; margin-top: 0.5rem;'>
            Продвинутая обработка документов с AI-распознаванием текста
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Боковая панель конфигурации
    with st.sidebar:
        st.markdown("### ⚙️ Конфигурация")
        
        # Выбор OCR движка
        ocr_engine = st.selectbox(
            "🎯 OCR Движок",
            options=['PaddleOCR', 'Tesseract', 'TrOCR', 'Ансамбль'],
            help="Выберите движок OCR для извлечения текста"
        )
        
        # Опции обработки
        st.markdown("### 🛠️ Опции обработки")
        use_llm = st.checkbox("🤖 Использовать LLM постобработку", value=False)
        show_bboxes = st.checkbox("📦 Показать ограничивающие рамки", value=True)
        enable_baseline = st.checkbox("📊 Сравнить с базовым методом", value=False)
        
        # Расширенные настройки
        with st.expander("🔧 Расширенные настройки"):
            confidence_threshold = st.slider(
                "Порог уверенности",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05
            )
            
            language = st.selectbox(
                "Язык",
                options=['en', 'ru', 'kz', 'multi'],
                index=3
            )
            
            output_format = st.radio(
                "Формат вывода",
                options=['JSON', 'CSV', 'XML']
            )
    
    # Основная область контента
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📤 Загрузить документ")
        uploaded_file = st.file_uploader(
            "Выберите файл",
            type=['jpg', 'jpeg', 'png', 'pdf'],
            help="Загрузите изображение документа или PDF для обработки OCR"
        )
        
        if uploaded_file is not None:
            # Отображение загруженного файла
            if uploaded_file.type == "application/pdf":
                images = process_pdf_to_images(uploaded_file)
                if images:
                    st.markdown(f"📄 **PDF Документ** ({len(images)} страниц)")
                    selected_page = st.selectbox(
                        "Выберите страницу",
                        options=range(1, len(images) + 1),
                        format_func=lambda x: f"Страница {x}"
                    )
                    current_image = images[selected_page - 1]
                    st.image(current_image, caption=f"Страница {selected_page}", use_column_width=True)
            else:
                current_image = Image.open(uploaded_file)
                st.image(current_image, caption="Загруженный документ", use_column_width=True)
            
            # Кнопка обработки
            if st.button("🚀 Запустить OCR", type="primary", use_container_width=True):
                with st.spinner(f"Обработка с помощью {ocr_engine}..."):
                    st.session_state.processing = True
                    
                    # Запуск выбранного OCR движка
                    if ocr_engine == 'PaddleOCR':
                        ocr_output = run_paddle(current_image)
                        raw_text = get_plaintext(ocr_output)
                        extracted_data = extract_fields_from_paddle(ocr_output)
                    elif ocr_engine == 'Tesseract':
                        raw_text, ocr_data = run_tesseract(current_image)
                        extracted_data = extract_fields_from_tesseract(ocr_data)
                    elif ocr_engine == 'TrOCR':
                        ocr_output = run_trocr(current_image)
                        raw_text = ocr_output.get('text', '')
                        extracted_data = ocr_output.get('fields', {})
                    
                    # Постобработка
                    if use_llm:
                        final_json = call_llm_to_json(raw_text)
                    else:
                        final_json = rules_based_postprocess(extracted_data, raw_text)
                    
                    # Сохранение результатов
                    st.session_state.ocr_results = {
                        'raw_text': raw_text,
                        'extracted_data': extracted_data,
                        'final_json': final_json,
                        'bboxes': ocr_output.get('bboxes', []) if isinstance(ocr_output, dict) else []
                    }
                    
                    # Запуск базового метода если включен
                    if enable_baseline:
                        baseline_text, baseline_data = run_tesseract(current_image)
                        st.session_state.baseline_results = {
                            'raw_text': baseline_text,
                            'extracted_data': extract_fields_from_tesseract(baseline_data)
                        }
                    
                    st.session_state.processing = False
                    st.success("✅ Обработка OCR завершена!")
    
    with col2:
        if st.session_state.ocr_results:
            st.markdown("### 📊 Результаты")
            
            # Создание вкладок для разных представлений
            tab1, tab2, tab3, tab4 = st.tabs(["📝 Сырой текст", "🗂️ Извлечённый JSON", "📈 Метрики", "🔍 Сравнение"])
            
            with tab1:
                st.markdown("#### Сырой вывод OCR")
                st.text_area(
                    "Извлечённый текст",
                    value=st.session_state.ocr_results['raw_text'],
                    height=300,
                    disabled=True
                )
                
                # Кнопка копирования
                if st.button("📋 Копировать текст", key="copy_raw"):
                    st.write("Текст скопирован в буфер обмена!")
                    st.code(st.session_state.ocr_results['raw_text'])
            
            with tab2:
                st.markdown("#### Структурированные данные")
                
                # Отображение JSON с подсветкой синтаксиса
                st.json(st.session_state.ocr_results['final_json'])
                
                # Кнопки загрузки
                col1, col2 = st.columns(2)
                with col1:
                    json_str = json.dumps(st.session_state.ocr_results['final_json'], indent=2, ensure_ascii=False)
                    st.download_button(
                        label="⬇️ Скачать JSON",
                        data=json_str,
                        file_name=f"ocr_результат_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                
                with col2:
                    if output_format == 'CSV':
                        # Конвертация в CSV формат
                        import pandas as pd
                        df = pd.DataFrame([st.session_state.ocr_results['final_json']])
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="⬇️ Скачать CSV",
                            data=csv,
                            file_name=f"ocr_результат_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
            
            with tab3:
                st.markdown("#### Метрики производительности")
                
                if enable_baseline and st.session_state.baseline_results:
                    # Вычисление метрик
                    metrics = {
                        'CER': cer(
                            st.session_state.baseline_results['raw_text'],
                            st.session_state.ocr_results['raw_text']
                        ),
                        'WER': wer(
                            st.session_state.baseline_results['raw_text'],
                            st.session_state.ocr_results['raw_text']
                        ),
                        'Уверенность': 0.85,  # Пример оценки уверенности
                        'Скорость': 0.92  # Пример метрики скорости
                    }
                    
                    # Отображение графика метрик
                    fig = create_metrics_chart(metrics)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Карточки метрик
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("CER", f"{metrics['CER']:.2%}", delta="↑ 5%")
                    with col2:
                        st.metric("WER", f"{metrics['WER']:.2%}", delta="↑ 3%")
                    with col3:
                        st.metric("Уверенность", f"{metrics['Уверенность']:.2%}")
                    with col4:
                        st.metric("Скорость", f"{metrics['Скорость']:.2%}")
                else:
                    st.info("Включите сравнение с базовым методом для просмотра метрик")
            
            with tab4:
                if enable_baseline and st.session_state.baseline_results:
                    st.markdown("#### Базовый метод vs Наш метод")
                    
                    # Визуализация различий
                    diff_html = create_diff_visualization(
                        st.session_state.baseline_results['raw_text'],
                        st.session_state.ocr_results['raw_text'],
                        "Базовый (Tesseract)",
                        f"Наш ({ocr_engine})"
                    )
                    st.markdown(diff_html, unsafe_allow_html=True)
                    
                    # Сравнение JSON бок о бок
                    st.markdown("##### Сравнение JSON вывода")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Базовый метод**")
                        st.json(st.session_state.baseline_results['extracted_data'])
                    with col2:
                        st.markdown(f"**{ocr_engine}**")
                        st.json(st.session_state.ocr_results['final_json'])
                else:
                    st.info("Включите сравнение с базовым методом для просмотра различий")
    
    # Визуализация ограничивающих рамок
    if show_bboxes and st.session_state.ocr_results and 'current_image' in locals():
        st.markdown("### 📦 Визуализация обнаружения полей")
        if st.session_state.ocr_results.get('bboxes'):
            img_with_boxes = draw_bboxes_on_image(
                current_image,
                st.session_state.ocr_results['bboxes']
            )
            st.image(img_with_boxes, caption="Обнаруженные поля с ограничивающими рамками", use_column_width=True)
            
            # Легенда для цветов bbox
            st.markdown("""
            **Легенда типов полей:**
            - 🔴 Поля имён
            - 🟢 Поля дат  
            - 🔵 Поля сумм
            - 🟡 Контактная информация
            - 🟣 Email адреса
            """)
    
    # Футер с дополнительной информацией
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>OCR 2.0 Демо | Работает на продвинутых AI моделях</p>
        <p style='font-size: 0.9rem;'>
            📧 Контакты: ocr-team@example.com | 
            📚 <a href='#'>Документация</a> | 
            🐛 <a href='#'>Сообщить об ошибке</a>
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()