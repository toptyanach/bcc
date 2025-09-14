// OCR Web App JavaScript
let currentFile = null;
let currentResult = null;
let availableEngines = [];

// Инициализация при загрузке страницы
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
    setupEventListeners();
    loadAvailableEngines();
});

function initializeApp() {
    console.log('🚀 Инициализация OCR веб-приложения');

    // Обновляем отображение слайдера уверенности
    const confidenceSlider = document.getElementById('confidenceSlider');
    const confidenceValue = document.getElementById('confidenceValue');

    confidenceSlider.addEventListener('input', function() {
        confidenceValue.textContent = this.value;
    });
}

function setupEventListeners() {
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const compareEngines = document.getElementById('compareEngines');

    // Drag & Drop функциональность
    dropZone.addEventListener('click', () => fileInput.click());
    dropZone.addEventListener('dragover', handleDragOver);
    dropZone.addEventListener('dragleave', handleDragLeave);
    dropZone.addEventListener('drop', handleDrop);

    fileInput.addEventListener('change', handleFileSelect);
    compareEngines.addEventListener('change', toggleComparisonMode);
}

function handleDragOver(e) {
    e.preventDefault();
    e.stopPropagation();
    e.currentTarget.classList.add('dragover');
}

function handleDragLeave(e) {
    e.preventDefault();
    e.stopPropagation();
    e.currentTarget.classList.remove('dragover');
}

function handleDrop(e) {
    e.preventDefault();
    e.stopPropagation();
    e.currentTarget.classList.remove('dragover');

    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFileSelect({ target: { files: files } });
    }
}

function handleFileSelect(e) {
    const file = e.target.files[0];
    if (!file) return;

    // Проверяем тип файла
    const allowedTypes = ['image/png', 'image/jpeg', 'image/jpg', 'application/pdf', 'image/bmp', 'image/tiff'];
    if (!allowedTypes.includes(file.type)) {
        showAlert('Неподдерживаемый тип файла. Разрешены: PNG, JPG, PDF, BMP, TIFF', 'danger');
        return;
    }

    // Проверяем размер файла (50MB)
    if (file.size > 50 * 1024 * 1024) {
        showAlert('Файл слишком большой. Максимальный размер: 50MB', 'danger');
        return;
    }

    currentFile = file;
    updateDropZoneWithFile(file);
    showSettings();
}

function updateDropZoneWithFile(file) {
    const dropZone = document.getElementById('dropZone');
    const fileSize = (file.size / 1024 / 1024).toFixed(2);

    dropZone.innerHTML = `
        <i class="bi bi-file-check-fill fs-1 text-success mb-3"></i>
        <h5>${file.name}</h5>
        <p class="text-muted mb-0">Размер: ${fileSize} MB</p>
    `;
}

function showSettings() {
    document.getElementById('settingsPanel').style.display = 'block';
    document.getElementById('actionButtons').style.display = 'block';
}

function loadAvailableEngines() {
    fetch('/api/ocr/engines')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                availableEngines = data.engines;
                updateEngineSelect();
                updateEnginesList();
            } else {
                console.error('Ошибка загрузки движков:', data.error);
                showAlert('Ошибка загрузки доступных OCR движков', 'warning');
            }
        })
        .catch(error => {
            console.error('Ошибка сети:', error);
            showAlert('Ошибка подключения к серверу', 'danger');
        });
}

function updateEngineSelect() {
    const select = document.getElementById('engineSelect');
    select.innerHTML = '';

    if (availableEngines.length === 0) {
        select.innerHTML = '<option value="">Нет доступных движков</option>';
        return;
    }

    availableEngines.forEach(engine => {
        const option = document.createElement('option');
        option.value = engine;
        option.textContent = engine;
        if (engine === 'PaddleOCR') option.selected = true;
        select.appendChild(option);
    });
}

function updateEnginesList() {
    const enginesList = document.getElementById('enginesList');
    if (availableEngines.length === 0) {
        enginesList.innerHTML = '<small class="text-muted">Нет доступных движков</small>';
        return;
    }

    enginesList.innerHTML = availableEngines.map(engine =>
        `<span class="badge bg-primary me-2">${engine}</span>`
    ).join('');
}

function toggleComparisonMode() {
    const comparisonMode = document.getElementById('comparisonMode');
    const engineSelect = document.getElementById('engineSelect');
    const isComparison = document.getElementById('compareEngines').checked;

    if (isComparison) {
        comparisonMode.style.display = 'block';
        engineSelect.disabled = true;
    } else {
        comparisonMode.style.display = 'none';
        engineSelect.disabled = false;
    }
}

function processDocument() {
    if (!currentFile) {
        showAlert('Сначала выберите файл', 'warning');
        return;
    }

    const isComparison = document.getElementById('compareEngines').checked;

    if (isComparison) {
        processComparison();
    } else {
        processSingleEngine();
    }
}

function processSingleEngine() {
    const formData = new FormData();
    formData.append('file', currentFile);
    formData.append('engine', document.getElementById('engineSelect').value);
    formData.append('language', document.getElementById('languageSelect').value);
    formData.append('use_llm', document.getElementById('useLlmCheck').checked);
    formData.append('confidence_threshold', document.getElementById('confidenceSlider').value);

    showProgress('Обработка документа...');

    fetch('/api/ocr/process', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        hideProgress();
        if (data.success) {
            currentResult = data.result;
            displayResults(data.result);
            showAlert('Документ успешно обработан!', 'success');
        } else {
            showAlert(`Ошибка обработки: ${data.error}`, 'danger');
        }
    })
    .catch(error => {
        hideProgress();
        console.error('Ошибка:', error);
        showAlert('Ошибка подключения к серверу', 'danger');
    });
}

function processComparison() {
    const formData = new FormData();
    formData.append('file', currentFile);
    formData.append('engines', availableEngines.join(','));
    formData.append('language', document.getElementById('languageSelect').value);

    showProgress('Сравнение движков...');

    fetch('/api/ocr/compare', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        hideProgress();
        if (data.success) {
            currentResult = data.comparison;
            displayComparisonResults(data.comparison);
            showAlert('Сравнение движков завершено!', 'success');
        } else {
            showAlert(`Ошибка сравнения: ${data.error}`, 'danger');
        }
    })
    .catch(error => {
        hideProgress();
        console.error('Ошибка:', error);
        showAlert('Ошибка подключения к серверу', 'danger');
    });
}

function showProgress(text) {
    document.getElementById('progressSection').style.display = 'block';
    document.getElementById('progressText').textContent = text;
    document.getElementById('processBtn').disabled = true;

    // Анимация прогресс-бара
    let progress = 0;
    const interval = setInterval(() => {
        progress += Math.random() * 15;
        if (progress > 90) progress = 90;
        document.getElementById('progressBar').style.width = `${progress}%`;
    }, 500);

    // Сохраняем интервал для остановки
    document.getElementById('progressSection').dataset.interval = interval;
}

function hideProgress() {
    document.getElementById('progressSection').style.display = 'none';
    document.getElementById('processBtn').disabled = false;
    document.getElementById('progressBar').style.width = '0%';

    // Останавливаем анимацию
    const interval = document.getElementById('progressSection').dataset.interval;
    if (interval) clearInterval(interval);
}

function displayResults(result) {
    document.getElementById('resultsSection').style.display = 'block';

    // Отображаем текст
    document.getElementById('extractedText').textContent = result.raw_text || 'Текст не найден';

    // Отображаем извлеченные поля
    displayExtractedFields(result.extracted_fields);

    // Отображаем метрики
    displayMetrics(result);

    // Отображаем сырые данные
    document.getElementById('rawData').textContent = JSON.stringify(result, null, 2);

    // Прокручиваем к результатам
    document.getElementById('resultsSection').scrollIntoView({ behavior: 'smooth' });
}

function displayExtractedFields(fields) {
    const container = document.getElementById('extractedFields');

    if (!fields || Object.keys(fields).length === 0) {
        container.innerHTML = '<div class="alert alert-info">Поля не извлечены</div>';
        return;
    }

    const importantFields = ['fio', 'date', 'sum', 'contract_number', 'account', 'phone', 'email', 'inn'];
    let html = '<div class="row">';

    importantFields.forEach(fieldName => {
        if (fields[fieldName]) {
            const icon = getFieldIcon(fieldName);
            html += `
                <div class="col-md-6 mb-3">
                    <div class="card h-100">
                        <div class="card-body">
                            <h6 class="card-title">
                                <i class="bi ${icon}"></i>
                                ${getFieldDisplayName(fieldName)}
                            </h6>
                            <p class="card-text">${fields[fieldName]}</p>
                        </div>
                    </div>
                </div>
            `;
        }
    });

    html += '</div>';
    container.innerHTML = html;
}

function displayMetrics(result) {
    const container = document.getElementById('metricsContent');
    let html = '<div class="row">';

    // Основные метрики
    const metrics = [
        {
            name: 'Количество элементов',
            value: result.total_items || 0,
            icon: 'bi-list-ol',
            color: 'primary'
        },
        {
            name: 'Средняя уверенность',
            value: result.avg_confidence ? `${(result.avg_confidence * 100).toFixed(1)}%` : 'N/A',
            icon: 'bi-speedometer2',
            color: 'success'
        },
        {
            name: 'Движок OCR',
            value: result.engine || 'N/A',
            icon: 'bi-gear-fill',
            color: 'info'
        },
        {
            name: 'Символов в тексте',
            value: result.raw_text ? result.raw_text.length : 0,
            icon: 'bi-type',
            color: 'warning'
        }
    ];

    metrics.forEach(metric => {
        html += `
            <div class="col-md-6 mb-3">
                <div class="metric-card bg-${metric.color}">
                    <div class="d-flex align-items-center">
                        <i class="bi ${metric.icon} fs-2 me-3"></i>
                        <div>
                            <h6 class="mb-1">${metric.name}</h6>
                            <h4 class="mb-0">${metric.value}</h4>
                        </div>
                    </div>
                </div>
            </div>
        `;
    });

    html += '</div>';
    container.innerHTML = html;
}

function displayComparisonResults(comparison) {
    document.getElementById('resultsSection').style.display = 'block';

    // Создаем сравнительную таблицу
    const results = comparison.results;
    let comparisonHtml = '<div class="table-responsive"><table class="table table-striped">';
    comparisonHtml += '<thead><tr><th>Движок</th><th>Статус</th><th>Символов</th><th>Слов</th><th>Элементов</th><th>Уверенность</th></tr></thead><tbody>';

    Object.keys(results).forEach(engine => {
        const result = results[engine];
        const status = result.success ?
            '<span class="badge bg-success">Успех</span>' :
            '<span class="badge bg-danger">Ошибка</span>';

        const chars = result.raw_text ? result.raw_text.length : 0;
        const words = result.raw_text ? result.raw_text.split(' ').length : 0;
        const items = result.total_items || 0;
        const conf = result.avg_confidence ? `${(result.avg_confidence * 100).toFixed(1)}%` : 'N/A';

        comparisonHtml += `
            <tr>
                <td><strong>${engine}</strong></td>
                <td>${status}</td>
                <td>${chars}</td>
                <td>${words}</td>
                <td>${items}</td>
                <td>${conf}</td>
            </tr>
        `;
    });

    comparisonHtml += '</tbody></table></div>';

    // Отображаем на всех вкладках
    document.getElementById('extractedText').innerHTML = comparisonHtml;
    document.getElementById('extractedFields').innerHTML = comparisonHtml;
    document.getElementById('metricsContent').innerHTML = comparisonHtml;
    document.getElementById('rawData').textContent = JSON.stringify(comparison, null, 2);

    document.getElementById('resultsSection').scrollIntoView({ behavior: 'smooth' });
}

function getFieldIcon(fieldName) {
    const icons = {
        'fio': 'bi-person-fill',
        'date': 'bi-calendar-fill',
        'sum': 'bi-currency-dollar',
        'contract_number': 'bi-file-earmark-text',
        'account': 'bi-bank',
        'phone': 'bi-telephone-fill',
        'email': 'bi-envelope-fill',
        'inn': 'bi-card-text'
    };
    return icons[fieldName] || 'bi-info-circle';
}

function getFieldDisplayName(fieldName) {
    const names = {
        'fio': 'ФИО',
        'date': 'Дата',
        'sum': 'Сумма',
        'contract_number': 'Номер договора',
        'account': 'Счет',
        'phone': 'Телефон',
        'email': 'Email',
        'inn': 'ИНН'
    };
    return names[fieldName] || fieldName;
}

function downloadResults() {
    if (!currentResult) {
        showAlert('Нет результатов для скачивания', 'warning');
        return;
    }

    const dataStr = JSON.stringify(currentResult, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);

    const link = document.createElement('a');
    link.href = url;
    link.download = `ocr_result_${new Date().toISOString().split('T')[0]}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);

    URL.revokeObjectURL(url);
}

function resetForm() {
    currentFile = null;
    currentResult = null;

    // Сброс интерфейса
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
    document.getElementById('comparisonMode').style.display = 'none';
    document.getElementById('resultsSection').style.display = 'none';
    document.getElementById('compareEngines').checked = false;
    document.getElementById('useLlmCheck').checked = false;
    document.getElementById('confidenceSlider').value = 0.5;
    document.getElementById('confidenceValue').textContent = '0.5';
}

function showAlert(message, type) {
    // Создаем alert если его нет
    let alertContainer = document.getElementById('alertContainer');
    if (!alertContainer) {
        alertContainer = document.createElement('div');
        alertContainer.id = 'alertContainer';
        alertContainer.className = 'position-fixed top-0 end-0 p-3';
        alertContainer.style.zIndex = '1050';
        document.body.appendChild(alertContainer);
    }

    const alertId = 'alert_' + Date.now();
    const alertHtml = `
        <div class="alert alert-${type} alert-dismissible fade show" id="${alertId}" role="alert">
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        </div>
    `;

    alertContainer.insertAdjacentHTML('beforeend', alertHtml);

    // Автоматически убираем через 5 секунд
    setTimeout(() => {
        const alert = document.getElementById(alertId);
        if (alert) {
            const bsAlert = new bootstrap.Alert(alert);
            bsAlert.close();
        }
    }, 5000);
}

function showApiDocs() {
    const modal = new bootstrap.Modal(document.getElementById('apiModal'));
    modal.show();
}

function showHealth() {
    fetch('/health')
        .then(response => response.json())
        .then(data => {
            const status = data.status === 'healthy' ? 'success' : 'danger';
            const engines = data.available_engines ? data.available_engines.join(', ') : 'N/A';
            showAlert(`Статус: ${data.status}. Движки: ${engines}`, status);
        })
        .catch(error => {
            showAlert('Ошибка проверки статуса', 'danger');
        });
}