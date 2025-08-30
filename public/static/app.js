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

// 서버로 데이터 전송
async function sendDataToServer(excelData) {
    // 실제로는 Excel 데이터를 분석하여 선수금/선급금을 매칭하는 로직이 필요
    // 지금은 mock 데이터를 반환
    
    // 진행률 업데이트
    updateProgress(30);
    
    await sleep(500);
    updateProgress(60);
    
    await sleep(500);
    updateProgress(90);
    
    // Mock 분석 로직 - 실제로는 여기서 복잡한 매칭 알고리즘 실행
    const mockResult = await analyzeMockData(excelData);
    
    updateProgress(100);
    await sleep(200);
    
    return mockResult;
}

// Mock 데이터 분석 (실제 구현에서는 복잡한 매칭 로직)
async function analyzeMockData(excelData) {
    // 시트 개수와 데이터 크기에 따라 동적으로 생성
    const sheetCount = Object.keys(excelData).length;
    const totalRows = Object.values(excelData).reduce((sum, sheet) => sum + sheet.length, 0);
    
    const contracts = [];
    const contractIds = ['PJT-001', 'PJT-002', 'PJT-003', 'PJT-004', 'PJT-005'];
    const companies = ['삼성전자', 'LG전자', 'SK하이닉스', '현대자동차', 'POSCO'];
    const owners = ['김철수', '박영희', '이민수', '정하나', '최민준'];
    const statuses = ['진행중', '완료', '검토중', '보류'];
    
    for (let i = 0; i < Math.min(sheetCount + 2, 8); i++) {
        const receipts = Math.floor(Math.random() * 50000000) + 5000000; // 5M~55M
        const advances = Math.floor(receipts * (0.6 + Math.random() * 0.3)); // 60-90% of receipts
        
        contracts.push({
            contract_id: contractIds[i % contractIds.length] + (i > 4 ? `-${Math.floor(i/5) + 1}` : ''),
            receipts: receipts,
            advances: advances,
            gap: receipts - advances,
            party: companies[i % companies.length],
            owner: owners[i % owners.length],
            status: statuses[i % statuses.length]
        });
    }
    
    // 총계 계산
    const totalReceipts = contracts.reduce((sum, c) => sum + c.receipts, 0);
    const totalAdvances = contracts.reduce((sum, c) => sum + c.advances, 0);
    const totalGap = totalReceipts - totalAdvances;
    const overdueCount = Math.floor(contracts.length * 0.2); // 20% 연체
    
    return {
        summary: {
            total_receipts: totalReceipts,
            total_advances: totalAdvances,
            gap: totalGap,
            contract_count: contracts.length,
            overdue_count: overdueCount
        },
        contracts: contracts
    };
}

// 대시보드 업데이트
function updateDashboard(data) {
    currentData = data;
    
    // KPI 업데이트
    document.getElementById('totalReceipts').textContent = formatCurrency(data.summary.total_receipts);
    document.getElementById('totalAdvances').textContent = formatCurrency(data.summary.total_advances);
    
    const gapElement = document.getElementById('totalGap');
    gapElement.textContent = formatCurrency(data.summary.gap);
    gapElement.className = `text-lg font-semibold ${data.summary.gap >= 0 ? 'text-green-600' : 'text-red-600'}`;
    
    document.getElementById('contractCount').textContent = data.summary.contract_count + '개';
    document.getElementById('overdueCount').textContent = data.summary.overdue_count + '건';
    
    // 테이블 업데이트
    updateContractsTable(data.contracts);
    
    // 차트 업데이트
    updateChart(data.contracts);
    
    // 상위 계약 업데이트
    updateTopContracts(data.contracts);
    
    // 대시보드 표시
    document.getElementById('dashboard').classList.remove('hidden');
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

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

// 초기화
document.addEventListener('DOMContentLoaded', function() {
    console.log('📊 선수/선급금 대시보드가 로드되었습니다.');
    
    // API 연결 테스트
    fetch('/api/hello')
        .then(response => response.json())
        .then(data => console.log('API 연결 성공:', data))
        .catch(error => console.error('API 연결 실패:', error));
});