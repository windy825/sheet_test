import { Hono } from 'hono'
import { cors } from 'hono/cors'
import { serveStatic } from 'hono/cloudflare-workers'

const app = new Hono()

// Enable CORS for frontend-backend communication
app.use('/api/*', cors())

// Serve static files from public directory
app.use('/static/*', serveStatic({ root: './public' }))

// API routes
app.get('/api/hello', (c) => {
  return c.json({ message: 'Hello from 선수금/선급금 대시보드!' })
})

// API endpoint for processing Excel data (mock for now)
app.post('/api/process-excel', async (c) => {
  try {
    // In a real implementation, we would process Excel data here
    // For now, return mock data
    const mockData = {
      summary: {
        total_receipts: 150000000,
        total_advances: 120000000,
        gap: 30000000,
        contract_count: 15,
        overdue_count: 3
      },
      contracts: [
        {
          contract_id: "PJT-001",
          receipts: 10000000,
          advances: 8000000,
          gap: 2000000,
          party: "삼성전자",
          owner: "김철수",
          status: "진행중"
        },
        {
          contract_id: "PJT-002", 
          receipts: 15000000,
          advances: 12000000,
          gap: 3000000,
          party: "LG전자",
          owner: "박영희",
          status: "완료"
        }
      ]
    }
    
    return c.json(mockData)
  } catch (error) {
    return c.json({ error: '데이터 처리 중 오류가 발생했습니다.' }, 500)
  }
})

// Default route - Main Dashboard
app.get('/', (c) => {
  return c.html(`
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>📊 선수/선급금 계약별 대시보드</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <link href="https://cdn.jsdelivr.net/npm/@fortawesome/fontawesome-free@6.4.0/css/all.min.css" rel="stylesheet">
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/xlsx/0.18.5/xlsx.full.min.js"></script>
        <link href="/static/style.css" rel="stylesheet">
    </head>
    <body class="bg-gray-50 min-h-screen">
        <!-- Header -->
        <header class="bg-white shadow-sm border-b">
            <div class="max-w-7xl mx-auto px-4 py-4">
                <div class="flex items-center justify-between">
                    <div class="flex items-center">
                        <i class="fas fa-chart-line text-blue-600 text-2xl mr-3"></i>
                        <h1 class="text-2xl font-bold text-gray-900">선수/선급금 계약별 대시보드</h1>
                    </div>
                    <div class="text-sm text-gray-500">
                        <i class="fas fa-sync-alt mr-1"></i>
                        실시간 업데이트
                    </div>
                </div>
            </div>
        </header>

        <div class="max-w-7xl mx-auto px-4 py-6">
            <!-- 파일 업로드 섹션 -->
            <div class="bg-white rounded-lg shadow-sm border p-6 mb-6">
                <h2 class="text-lg font-semibold text-gray-800 mb-4">
                    <i class="fas fa-upload mr-2 text-green-600"></i>
                    엑셀 파일 업로드
                </h2>
                <div class="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center" 
                     id="dropzone"
                     ondrop="handleDrop(event)" 
                     ondragover="handleDragOver(event)"
                     ondragenter="handleDragEnter(event)"
                     ondragleave="handleDragLeave(event)">
                    <i class="fas fa-cloud-upload-alt text-4xl text-gray-400 mb-4"></i>
                    <p class="text-gray-600 mb-2">엑셀 파일을 드래그하여 놓거나 클릭하여 선택하세요</p>
                    <p class="text-sm text-gray-500 mb-4">지원 형식: .xlsx, .xlsm, .xls</p>
                    <input type="file" id="fileInput" accept=".xlsx,.xlsm,.xls" class="hidden" onchange="handleFileSelect(event)">
                    <button onclick="document.getElementById('fileInput').click()" 
                            class="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition-colors">
                        <i class="fas fa-folder-open mr-2"></i>파일 선택
                    </button>
                </div>
                
                <!-- 업로드 진행상황 -->
                <div id="uploadProgress" class="hidden mt-4">
                    <div class="bg-blue-100 rounded-lg p-3">
                        <div class="flex items-center">
                            <i class="fas fa-spinner fa-spin text-blue-600 mr-2"></i>
                            <span class="text-blue-800">파일을 분석하고 있습니다...</span>
                        </div>
                        <div class="w-full bg-blue-200 rounded-full h-2 mt-2">
                            <div class="bg-blue-600 h-2 rounded-full transition-all duration-300" 
                                 id="progressBar" style="width: 0%"></div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- KPI 대시보드 -->
            <div id="dashboard" class="hidden">
                <!-- 주요 지표 -->
                <div class="grid grid-cols-1 md:grid-cols-5 gap-4 mb-6">
                    <div class="bg-white rounded-lg shadow-sm border p-4">
                        <div class="flex items-center">
                            <div class="p-2 bg-green-100 rounded-lg">
                                <i class="fas fa-arrow-down text-green-600"></i>
                            </div>
                            <div class="ml-3">
                                <p class="text-sm text-gray-500">총 선수금</p>
                                <p class="text-lg font-semibold text-gray-900" id="totalReceipts">-</p>
                            </div>
                        </div>
                    </div>
                    
                    <div class="bg-white rounded-lg shadow-sm border p-4">
                        <div class="flex items-center">
                            <div class="p-2 bg-red-100 rounded-lg">
                                <i class="fas fa-arrow-up text-red-600"></i>
                            </div>
                            <div class="ml-3">
                                <p class="text-sm text-gray-500">총 선급금</p>
                                <p class="text-lg font-semibold text-gray-900" id="totalAdvances">-</p>
                            </div>
                        </div>
                    </div>
                    
                    <div class="bg-white rounded-lg shadow-sm border p-4">
                        <div class="flex items-center">
                            <div class="p-2 bg-blue-100 rounded-lg">
                                <i class="fas fa-balance-scale text-blue-600"></i>
                            </div>
                            <div class="ml-3">
                                <p class="text-sm text-gray-500">전체 Gap</p>
                                <p class="text-lg font-semibold" id="totalGap">-</p>
                            </div>
                        </div>
                    </div>
                    
                    <div class="bg-white rounded-lg shadow-sm border p-4">
                        <div class="flex items-center">
                            <div class="p-2 bg-purple-100 rounded-lg">
                                <i class="fas fa-file-contract text-purple-600"></i>
                            </div>
                            <div class="ml-3">
                                <p class="text-sm text-gray-500">계약 수</p>
                                <p class="text-lg font-semibold text-gray-900" id="contractCount">-</p>
                            </div>
                        </div>
                    </div>
                    
                    <div class="bg-white rounded-lg shadow-sm border p-4">
                        <div class="flex items-center">
                            <div class="p-2 bg-yellow-100 rounded-lg">
                                <i class="fas fa-exclamation-triangle text-yellow-600"></i>
                            </div>
                            <div class="ml-3">
                                <p class="text-sm text-gray-500">연체 건</p>
                                <p class="text-lg font-semibold text-gray-900" id="overdueCount">-</p>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- 차트와 테이블 -->
                <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
                    <!-- 차트 -->
                    <div class="bg-white rounded-lg shadow-sm border p-6">
                        <h3 class="text-lg font-semibold text-gray-800 mb-4">
                            <i class="fas fa-chart-pie mr-2 text-blue-600"></i>
                            계약별 Gap 분포
                        </h3>
                        <div class="relative h-64">
                            <canvas id="gapChart"></canvas>
                        </div>
                    </div>
                    
                    <!-- 상위 계약 -->
                    <div class="bg-white rounded-lg shadow-sm border p-6">
                        <h3 class="text-lg font-semibold text-gray-800 mb-4">
                            <i class="fas fa-trophy mr-2 text-yellow-600"></i>
                            Gap 상위 계약
                        </h3>
                        <div id="topContracts" class="space-y-3">
                            <!-- 동적으로 생성 -->
                        </div>
                    </div>
                </div>

                <!-- 상세 테이블 -->
                <div class="bg-white rounded-lg shadow-sm border">
                    <div class="p-6 border-b">
                        <h3 class="text-lg font-semibold text-gray-800">
                            <i class="fas fa-table mr-2 text-green-600"></i>
                            계약별 상세 현황
                        </h3>
                    </div>
                    <div class="overflow-x-auto">
                        <table class="w-full">
                            <thead class="bg-gray-50">
                                <tr>
                                    <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                        계약ID
                                    </th>
                                    <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                        거래처
                                    </th>
                                    <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                        담당자
                                    </th>
                                    <th class="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                                        선수금
                                    </th>
                                    <th class="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                                        선급금
                                    </th>
                                    <th class="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                                        Gap
                                    </th>
                                    <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                        상태
                                    </th>
                                </tr>
                            </thead>
                            <tbody id="contractsTable" class="bg-white divide-y divide-gray-200">
                                <!-- 동적으로 생성 -->
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>

        <script src="/static/app.js"></script>
    </body>
    </html>
  `)
})

export default app