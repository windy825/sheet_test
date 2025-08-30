// 선수금/선급금 대시보드 JavaScript

let currentData = null;
let chart = null;

// 파일 드롭 핸들러
function handleDrop(e) {
    e.preventDefault();
    e.stopPropagation();
    
    const dropzone = document.getElementById('dropzone');
    dropzone.classList.remove('border-blue-400', 'bg-blue-50');
    dropzone.classList.add('border-gray-300');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        processFile(files[0]);
    }
}

function handleDragOver(e) {
    e.preventDefault();
    e.stopPropagation();
}

function handleDragEnter(e) {
    e.preventDefault();
    e.stopPropagation();
    
    const dropzone = document.getElementById('dropzone');
    dropzone.classList.remove('border-gray-300');
    dropzone.classList.add('border-blue-400', 'bg-blue-50');
}

function handleDragLeave(e) {
    e.preventDefault();
    e.stopPropagation();
    
    const dropzone = document.getElementById('dropzone');
    dropzone.classList.remove('border-blue-400', 'bg-blue-50');
    dropzone.classList.add('border-gray-300');
}

// 파일 선택 핸들러
function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        processFile(file);
    }
}

// 파일 처리 함수
async function processFile(file) {
    try {
        // 파일 형식 검증
        if (!isValidExcelFile(file)) {
            showError('지원되지 않는 파일 형식입니다. Excel 파일(.xlsx, .xlsm, .xls)을 업로드해주세요.');
            return;
        }

        // 업로드 진행상황 표시
        showUploadProgress();

        // Excel 파일 읽기 (클라이언트 사이드)
        const data = await readExcelFile(file);
        
        // 서버로 데이터 전송 및 처리 (실제로는 분석 로직을 여기서 실행)
        const processedData = await sendDataToServer(data);
        
        // 대시보드 업데이트
        updateDashboard(processedData);
        
        hideUploadProgress();
        showSuccess(`파일 "${file.name}" 처리가 완료되었습니다.`);
        
    } catch (error) {
        console.error('파일 처리 오류:', error);
        hideUploadProgress();
        showError('파일 처리 중 오류가 발생했습니다: ' + error.message);
    }
}

// Excel 파일 형식 검증
function isValidExcelFile(file) {
    const validTypes = [
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        'application/vnd.ms-excel.sheet.macroEnabled.12',
        'application/vnd.ms-excel'
    ];
    
    const validExtensions = ['.xlsx', '.xlsm', '.xls'];
    const fileName = file.name.toLowerCase();
    
    return validTypes.includes(file.type) || 
           validExtensions.some(ext => fileName.endsWith(ext));
}

// Excel 파일 읽기 (SheetJS 사용)
async function readExcelFile(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        
        reader.onload = function(e) {
            try {
                const data = new Uint8Array(e.target.result);
                const workbook = XLSX.read(data, { type: 'array' });
                
                // 시트 이름 분석하여 선수금/선급금 시트 찾기
                const sheets = {};
                
                workbook.SheetNames.forEach(sheetName => {
                    const sheet = workbook.Sheets[sheetName];
                    const jsonData = XLSX.utils.sheet_to_json(sheet, { header: 1 });
                    
                    // 빈 행 제거
                    const filteredData = jsonData.filter(row => 
                        row.some(cell => cell !== undefined && cell !== null && cell !== '')
                    );
                    
                    if (filteredData.length > 0) {
                        sheets[sheetName] = filteredData;
                    }
                });
                
                resolve(sheets);
            } catch (error) {
                reject(new Error('Excel 파일 읽기 실패: ' + error.message));
            }
        };
        
        reader.onerror = function() {
            reject(new Error('파일 읽기 실패'));
        };
        
        reader.readAsArrayBuffer(file);
    });
}

// 실제 Excel 데이터 분석
async function sendDataToServer(excelData) {
    try {
        // 진행률 업데이트
        updateProgress(20);
        
        // 실제 Excel 분석 엔진 사용
        console.log('Excel 분석 시작...');
        const analysisResult = await window.excelAnalyzer.analyzeExcelData(excelData);
        
        updateProgress(70);
        await sleep(300);
        
        updateProgress(90);
        await sleep(200);
        
        console.log('Excel 분석 완료:', analysisResult);
        
        updateProgress(100);
        await sleep(200);
        
        return analysisResult;
        
    } catch (error) {
        console.error('Excel 분석 오류:', error);
        throw new Error('Excel 파일 분석 중 오류가 발생했습니다: ' + error.message);
    }
}

// 분석 결과 검증 및 후처리
function validateAnalysisResult(result) {
    // 기본 구조 검증
    if (!result || !result.summary || !result.contracts) {
        throw new Error('분석 결과 형식이 올바르지 않습니다.');
    }
    
    // 데이터 유효성 검증
    if (result.contracts.length === 0) {
        throw new Error('분석할 수 있는 계약 데이터가 없습니다. Excel 파일의 데이터 형식을 확인해주세요.');
    }
    
    // 필수 필드 검증
    result.contracts.forEach((contract, index) => {
        if (!contract.contract_id || contract.contract_id.trim() === '') {
            console.warn(`계약 ${index + 1}: 계약ID가 없습니다.`);
        }
        if (typeof contract.receipts !== 'number' || typeof contract.advances !== 'number') {
            console.warn(`계약 ${contract.contract_id}: 금액 데이터가 올바르지 않습니다.`);
        }
    });
    
    return result;
}

// 대시보드 업데이트
function updateDashboard(data) {
    try {
        // 데이터 검증
        const validatedData = validateAnalysisResult(data);
        currentData = validatedData;
        
        // 분석 정보 표시
        if (validatedData.validation_info) {
            showAnalysisInfo(validatedData.validation_info);
        }
        
        // KPI 업데이트
        document.getElementById('totalReceipts').textContent = formatCurrency(validatedData.summary.total_receipts);
        document.getElementById('totalAdvances').textContent = formatCurrency(validatedData.summary.total_advances);
        
        const gapElement = document.getElementById('totalGap');
        gapElement.textContent = formatCurrency(validatedData.summary.gap);
        gapElement.className = `text-lg font-semibold ${validatedData.summary.gap >= 0 ? 'text-green-600' : 'text-red-600'}`;
        
        document.getElementById('contractCount').textContent = validatedData.summary.contract_count + '개';
        document.getElementById('overdueCount').textContent = validatedData.summary.overdue_count + '건';
        
        // 테이블 업데이트
        updateContractsTable(validatedData.contracts);
        
        // 차트 업데이트
        updateChart(validatedData.contracts);
        
        // 상위 계약 업데이트
        updateTopContracts(validatedData.contracts);
        
        // 대시보드 표시
        document.getElementById('dashboard').classList.remove('hidden');
        
    } catch (error) {
        console.error('대시보드 업데이트 오류:', error);
        showError('대시보드 업데이트 중 오류가 발생했습니다: ' + error.message);
    }
}

// 계약 테이블 업데이트
function updateContractsTable(contracts) {
    const tbody = document.getElementById('contractsTable');
    tbody.innerHTML = '';
    
    contracts.forEach(contract => {
        const row = document.createElement('tr');
        row.className = 'hover:bg-gray-50';
        
        const statusClass = {
            '진행중': 'bg-blue-100 text-blue-800',
            '완료': 'bg-green-100 text-green-800',
            '검토중': 'bg-yellow-100 text-yellow-800',
            '보류': 'bg-gray-100 text-gray-800'
        }[contract.status] || 'bg-gray-100 text-gray-800';
        
        const gapColor = contract.gap >= 0 ? 'text-green-600' : 'text-red-600';
        
        row.innerHTML = `
            <td class="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                ${contract.contract_id}
            </td>
            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                ${contract.party}
            </td>
            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                ${contract.owner}
            </td>
            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900 text-right">
                ${formatCurrency(contract.receipts)}
            </td>
            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900 text-right">
                ${formatCurrency(contract.advances)}
            </td>
            <td class="px-6 py-4 whitespace-nowrap text-sm font-semibold text-right ${gapColor}">
                ${formatCurrency(contract.gap)}
            </td>
            <td class="px-6 py-4 whitespace-nowrap">
                <span class="inline-flex px-2 py-1 text-xs font-semibold rounded-full ${statusClass}">
                    ${contract.status}
                </span>
            </td>
        `;
        
        tbody.appendChild(row);
    });
}

// 차트 업데이트
function updateChart(contracts) {
    const ctx = document.getElementById('gapChart').getContext('2d');
    
    if (chart) {
        chart.destroy();
    }
    
    // Gap 데이터 준비
    const positiveGaps = contracts.filter(c => c.gap > 0);
    const negativeGaps = contracts.filter(c => c.gap < 0);
    const zeroGaps = contracts.filter(c => c.gap === 0);
    
    chart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['양수 Gap', '음수 Gap', '제로 Gap'],
            datasets: [{
                data: [
                    positiveGaps.reduce((sum, c) => sum + c.gap, 0),
                    Math.abs(negativeGaps.reduce((sum, c) => sum + c.gap, 0)),
                    zeroGaps.length * 1000000 // 시각화를 위한 임의값
                ],
                backgroundColor: [
                    '#10B981', // green-500
                    '#EF4444', // red-500  
                    '#6B7280'  // gray-500
                ],
                borderWidth: 2,
                borderColor: '#ffffff'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom'
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const label = context.label || '';
                            const value = formatCurrency(context.raw);
                            return label + ': ' + value;
                        }
                    }
                }
            }
        }
    });
}

// 상위 계약 업데이트
function updateTopContracts(contracts) {
    const topContainer = document.getElementById('topContracts');
    topContainer.innerHTML = '';
    
    // Gap 기준으로 정렬 (절댓값)
    const sorted = [...contracts]
        .sort((a, b) => Math.abs(b.gap) - Math.abs(a.gap))
        .slice(0, 5);
    
    sorted.forEach((contract, index) => {
        const item = document.createElement('div');
        item.className = 'flex items-center justify-between p-3 bg-gray-50 rounded-lg';
        
        const gapColor = contract.gap >= 0 ? 'text-green-600' : 'text-red-600';
        const icon = contract.gap >= 0 ? 'fa-arrow-up text-green-500' : 'fa-arrow-down text-red-500';
        
        item.innerHTML = `
            <div class="flex items-center">
                <div class="flex items-center justify-center w-8 h-8 bg-white rounded-full border text-sm font-semibold text-gray-600 mr-3">
                    ${index + 1}
                </div>
                <div>
                    <div class="font-medium text-gray-900">${contract.contract_id}</div>
                    <div class="text-sm text-gray-500">${contract.party}</div>
                </div>
            </div>
            <div class="text-right">
                <div class="flex items-center ${gapColor} font-semibold">
                    <i class="fas ${icon} mr-1"></i>
                    ${formatCurrency(Math.abs(contract.gap))}
                </div>
            </div>
        `;
        
        topContainer.appendChild(item);
    });
}

// 유틸리티 함수들
function formatCurrency(amount) {
    return new Intl.NumberFormat('ko-KR').format(amount) + '원';
}

function showUploadProgress() {
    document.getElementById('uploadProgress').classList.remove('hidden');
    updateProgress(0);
}

function hideUploadProgress() {
    document.getElementById('uploadProgress').classList.add('hidden');
}

function updateProgress(percent) {
    const progressBar = document.getElementById('progressBar');
    progressBar.style.width = percent + '%';
}

function showSuccess(message) {
    showNotification(message, 'success');
}

function showError(message) {
    showNotification(message, 'error');
}

function showNotification(message, type) {
    // 간단한 알림 표시
    const alertClass = type === 'error' ? 'alert-error' : 'alert-success';
    const bgClass = type === 'error' ? 'bg-red-100 border-red-500 text-red-700' : 'bg-green-100 border-green-500 text-green-700';
    
    const notification = document.createElement('div');
    notification.className = `fixed top-4 right-4 p-4 border-l-4 ${bgClass} shadow-lg rounded z-50`;
    notification.innerHTML = `
        <div class="flex items-center">
            <i class="fas ${type === 'error' ? 'fa-exclamation-circle' : 'fa-check-circle'} mr-2"></i>
            <span>${message}</span>
            <button onclick="this.parentElement.parentElement.remove()" class="ml-4 text-lg">&times;</button>
        </div>
    `;
    
    document.body.appendChild(notification);
    
    // 5초 후 자동 제거
    setTimeout(() => {
        if (notification.parentElement) {
            notification.remove();
        }
    }, 5000);
}

// 분석 정보 표시
function showAnalysisInfo(validationInfo) {
    const notification = document.createElement('div');
    notification.className = 'fixed top-4 left-4 bg-blue-50 border border-blue-200 text-blue-800 p-4 rounded-lg shadow-lg z-50 max-w-sm';
    notification.innerHTML = `
        <div class="flex items-start">
            <i class="fas fa-info-circle text-blue-600 mr-2 mt-1"></i>
            <div>
                <h4 class="font-semibold mb-2">분석 완료</h4>
                <ul class="text-sm space-y-1">
                    <li>📄 감지된 시트: ${validationInfo.시트_목록?.join(', ') || 'N/A'}</li>
                    <li>📊 선수금: ${validationInfo.선수금_표준화_행수 || 0}건 (원본: ${validationInfo.선수금_원본_행수 || 0}건)</li>
                    <li>📊 선급금: ${validationInfo.선급금_표준화_행수 || 0}건 (원본: ${validationInfo.선급금_원본_행수 || 0}건)</li>
                    <li>✅ 총 처리: ${validationInfo.총_데이터_건수 || 0}건</li>
                </ul>
                <button onclick="this.parentElement.parentElement.parentElement.remove()" 
                        class="mt-2 text-xs text-blue-600 hover:text-blue-800">닫기</button>
            </div>
        </div>
    `;
    
    document.body.appendChild(notification);
    
    // 10초 후 자동 제거
    setTimeout(() => {
        if (notification.parentElement) {
            notification.remove();
        }
    }, 10000);
}

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

// 초기화
document.addEventListener('DOMContentLoaded', function() {
    console.log('📊 선수/선급금 대시보드가 로드되었습니다.');
    
    // Excel 분석 엔진 로드 확인
    if (window.excelAnalyzer) {
        console.log('✅ Excel 분석 엔진이 로드되었습니다.');
    } else {
        console.error('❌ Excel 분석 엔진을 로드할 수 없습니다.');
    }
    
    // API 연결 테스트
    fetch('/api/hello')
        .then(response => response.json())
        .then(data => console.log('API 연결 성공:', data))
        .catch(error => console.error('API 연결 실패:', error));
});