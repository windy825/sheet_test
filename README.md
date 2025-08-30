# 📊 선수금/선급금 계약별 대시보드

## 프로젝트 개요
- **이름**: 선수금/선급금 계약별 대시보드 (webapp)
- **목적**: 엑셀 데이터를 기반으로 선수금과 선급금을 매칭하여 계약별 Gap을 분석하는 인터랙티브 대시보드
- **주요 기능**: 엑셀 파일 업로드, 데이터 자동 매칭, KPI 시각화, 상세 분석

## 🌐 URL 정보
- **개발 서버**: https://3000-isgf7jst48imgnfl8kz13-6532622b.e2b.dev
- **API 헬스체크**: https://3000-isgf7jst48imgnfl8kz13-6532622b.e2b.dev/api/hello
- **GitHub**: https://github.com/windy825/sheet_test (원본 Streamlit 버전)

## 🎯 현재 구현된 기능

### ✅ 완료된 기능
1. **반응형 웹 대시보드**
   - 모던한 UI/UX (TailwindCSS + FontAwesome)
   - 드래그 앤 드롭 파일 업로드
   - 실시간 진행률 표시

2. **엑셀 파일 처리**
   - 클라이언트 사이드 Excel 파싱 (SheetJS)
   - 지원 형식: .xlsx, .xlsm, .xls
   - 안전한 파일 검증

3. **KPI 대시보드**
   - 총 선수금/선급금 표시
   - Gap 분석 (선수금 - 선급금)
   - 계약 수 및 연체 건수 추적

4. **시각화 기능**
   - Gap 분포 도넛 차트 (Chart.js)
   - 상위 계약 랭킹
   - 계약별 상세 테이블

5. **API 구조**
   - `/api/hello` - 헬스체크
   - `/api/process-excel` - 엑셀 데이터 처리 (POST)

### 📊 현재 기능별 URI

| 기능 | HTTP 메소드 | 경로 | 설명 |
|------|-------------|------|------|
| 메인 대시보드 | GET | `/` | 웹 대시보드 메인 페이지 |
| API 헬스체크 | GET | `/api/hello` | 서버 상태 확인 |
| 엑셀 데이터 처리 | POST | `/api/process-excel` | 업로드된 엑셀 파일 분석 |
| 정적 파일 | GET | `/static/*` | CSS, JS, 이미지 파일 |

## 📋 아직 구현되지 않은 기능

### 🔧 백엔드 개선 필요
1. **실제 Excel 분석 로직**
   - 현재: Mock 데이터 사용
   - 필요: 실제 컬럼 매칭 및 데이터 표준화
   - 필요: 선수금/선급금 지능형 매칭 알고리즘

2. **데이터 저장소 연동**
   - Cloudflare D1 데이터베이스 설정
   - 분석 결과 영구 저장
   - 사용자별 세션 관리

3. **고급 매칭 설정**
   - 매칭 조건 설정 UI
   - 금액 허용 오차 설정
   - 일자 윈도우 설정
   - 거래처 텍스트 유사도 매칭

### 🎨 프론트엔드 개선
1. **에러 핸들링**
   - 파일 업로드 실패 처리
   - 네트워크 오류 대응
   - 사용자 친화적 오류 메시지

2. **고급 필터링**
   - 테이블 정렬 기능
   - 검색 및 필터
   - 데이터 내보내기 (Excel, CSV)

3. **실시간 업데이트**
   - WebSocket 연결
   - 자동 새로고침
   - 배치 처리 상태 표시

## 🚀 다음 단계 권장사항

### 우선순위 1: 핵심 기능 구현
1. **Excel 분석 엔진 구현**
   ```typescript
   // src/services/excelAnalyzer.ts 생성 필요
   - 컬럼 자동 감지 및 매칭
   - 데이터 표준화 및 검증
   - 선수금/선급금 매칭 알고리즘
   ```

2. **Cloudflare D1 데이터베이스 설정**
   ```bash
   npx wrangler d1 create webapp-production
   # 마이그레이션 파일 생성 필요
   ```

3. **실제 API 로직 구현**
   - `/api/process-excel` 엔드포인트 완성
   - 파일 업로드 처리 로직
   - 결과 데이터 반환 구조 정의

### 우선순위 2: 사용자 경험 개선
1. **에러 처리 강화**
2. **로딩 상태 개선**
3. **모바일 반응형 최적화**

### 우선순위 3: 고급 기능
1. **사용자 인증 (Cloudflare Access)**
2. **데이터 내보내기**
3. **히스토리 관리**

## 🏗️ 데이터 아키텍처

### 현재 Mock 데이터 구조
```typescript
{
  summary: {
    total_receipts: number,    // 총 선수금
    total_advances: number,    // 총 선급금  
    gap: number,              // 전체 Gap
    contract_count: number,   // 계약 수
    overdue_count: number     // 연체 건수
  },
  contracts: [{
    contract_id: string,      // 계약 ID
    receipts: number,         // 선수금
    advances: number,         // 선급금
    gap: number,             // Gap (선수금-선급금)
    party: string,           // 거래처명
    owner: string,           // 담당자
    status: string           // 상태 (진행중/완료/검토중/보류)
  }]
}
```

### 권장 D1 테이블 구조
```sql
-- 분석 세션
CREATE TABLE analysis_sessions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  filename TEXT NOT NULL,
  upload_time DATETIME DEFAULT CURRENT_TIMESTAMP,
  status TEXT DEFAULT 'processing'
);

-- 계약 데이터
CREATE TABLE contracts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id INTEGER,
  contract_id TEXT,
  receipts INTEGER DEFAULT 0,
  advances INTEGER DEFAULT 0,
  gap INTEGER GENERATED ALWAYS AS (receipts - advances),
  party TEXT,
  owner TEXT,
  status TEXT,
  FOREIGN KEY (session_id) REFERENCES analysis_sessions(id)
);
```

## 📝 사용자 가이드

### 1. Excel 파일 업로드
1. 메인 페이지의 업로드 영역에 Excel 파일을 드래그하거나 클릭하여 선택
2. 지원 형식: .xlsx, .xlsm, .xls
3. 파일 분석 진행률을 실시간으로 확인

### 2. 대시보드 확인
1. **KPI 메트릭**: 상단 5개 카드에서 주요 지표 확인
2. **Gap 분포 차트**: 왼쪽 도넛 차트에서 전체 Gap 분포 시각화
3. **상위 계약**: 오른쪽에서 Gap이 큰 계약 순위 확인
4. **상세 테이블**: 하단에서 모든 계약의 상세 정보 확인

### 3. 데이터 해석
- **양수 Gap**: 선수금이 선급금보다 많음 (유리한 상황)
- **음수 Gap**: 선급금이 선수금보다 많음 (현금흐름 주의)
- **연체 건**: 기한이 경과된 거래로 별도 관리 필요

## 🛠️ 배포 정보
- **플랫폼**: Cloudflare Pages (계획)
- **현재 상태**: ✅ 개발 서버 활성
- **기술 스택**: 
  - Backend: Hono + TypeScript
  - Frontend: HTML + TailwindCSS + Vanilla JS
  - 차트: Chart.js
  - Excel 처리: SheetJS
- **마지막 업데이트**: 2025-08-30

## 📞 연락처
- **개발자**: GitHub @windy825
- **원본 저장소**: https://github.com/windy825/sheet_test (Streamlit 버전)
- **현재 프로젝트**: Cloudflare Pages 웹앱 버전