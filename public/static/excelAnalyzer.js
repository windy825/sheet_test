// 엑셀 분석 엔진 - 원본 Streamlit 로직 포팅

// 표준 컬럼 정의
const STANDARD_COLS = [
    "contract_id", "direction", "amount", "date", "party", 
    "owner", "status", "note", "overdue_flag"
];

// 컬럼 매칭 후보들 (원본에서 그대로 가져옴)
const COLUMN_CANDIDATES = {
    contract: ["계약번호", "금형마스터", "프로젝트", "프로젝트코드", "PJT", "PJT코드", "고유번호", "계약코드", "계약id", "계약ID"],
    amount: ["금액", "선수금", "선급금", "선수금금액", "선급금금액", "합계", "잔액", "amount"],
    date: ["일자", "청구일", "지급일", "납기일", "요청일", "등록일", "기준일", "date"],
    party: ["업체명", "거래처", "고객사", "고객명", "상대방", "회사", "vendor", "customer", "업체"],
    owner: ["담당자", "담당", "담당자명", "PM", "담당부서", "owner"],
    status: ["진행현황", "정산여부", "상태", "status"],
    note: ["비고", "메모", "특이사항", "코멘트", "note", "설명"],
    overdue: ["기한경과", "연체", "overdue", "경과"]
};

class ExcelAnalyzer {
    constructor() {
        this.debug = true;
    }

    log(message, data = null) {
        if (this.debug) {
            console.log(`[ExcelAnalyzer] ${message}`, data || '');
        }
    }

    // 안전한 숫자 변환
    toFloat(value) {
        try {
            if (value === null || value === undefined || value === '') return 0.0;
            if (typeof value === 'number' && !isNaN(value)) return value;
            
            let str = String(value).trim();
            if (str === '' || str.toLowerCase().match(/^(nan|none|null)$/)) return 0.0;
            
            // 숫자가 아닌 문자 제거 (쉼표, 원화 기호 등)
            str = str.replace(/[^0-9\-\.]/g, '');
            if (str === '' || str === '-' || str === '.') return 0.0;
            
            const result = parseFloat(str);
            return isNaN(result) ? 0.0 : result;
        } catch (e) {
            this.log(`숫자 변환 실패: ${value} -> ${e.message}`);
            return 0.0;
        }
    }

    // 컬럼명 정규화
    normalizeColumns(data) {
        if (!data || data.length === 0) return data;
        
        // 첫 번째 행을 헤더로 가정
        const headers = data[0];
        const normalizedHeaders = [];
        const seen = new Set();
        
        headers.forEach(header => {
            let normalized = String(header || '').trim();
            normalized = normalized.replace(/\s+/g, ' '); // 공백 정규화
            
            // 중복 컬럼명 처리
            if (seen.has(normalized)) {
                let counter = 2;
                let newName = normalized;
                while (seen.has(newName)) {
                    newName = `${normalized}_${counter}`;
                    counter++;
                }
                normalized = newName;
            }
            
            seen.add(normalized);
            normalizedHeaders.push(normalized);
        });
        
        // 첫 번째 행을 새로운 헤더로 교체
        return [normalizedHeaders, ...data.slice(1)];
    }

    // 컬럼 매칭
    findColumnMatch(headers, candidates) {
        if (!headers || headers.length === 0) return null;
        
        const lowerMap = {};
        headers.forEach(header => {
            lowerMap[String(header).toLowerCase()] = header;
        });
        
        // 정확한 매치 우선
        for (const candidate of candidates) {
            if (headers.includes(candidate)) return candidate;
            if (lowerMap[candidate.toLowerCase()]) {
                return lowerMap[candidate.toLowerCase()];
            }
        }
        
        // 부분 매치
        for (const candidate of candidates) {
            const pattern = new RegExp(candidate.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'i');
            for (const header of headers) {
                if (pattern.test(String(header))) {
                    return header;
                }
            }
        }
        
        return null;
    }

    // 시트에서 키워드 기반으로 데이터 찾기
    findSheetByKeywords(sheets, keywords) {
        const normalizeSheetName = (name) => name.replace(/\s+/g, '').toLowerCase();
        
        for (const sheetName of Object.keys(sheets)) {
            const normalized = normalizeSheetName(sheetName);
            if (keywords.some(keyword => normalized.includes(normalizeSheetName(keyword)))) {
                return sheets[sheetName];
            }
        }
        
        this.log(`키워드 ${keywords}에 해당하는 시트를 찾을 수 없습니다.`);
        return null;
    }

    // 데이터 표준화
    standardizeData(rawData, direction) {
        if (!rawData || rawData.length === 0) {
            this.log(`${direction}: 빈 데이터`);
            return [];
        }
        
        try {
            // 컬럼명 정규화
            const normalizedData = this.normalizeColumns(rawData);
            const headers = normalizedData[0];
            const rows = normalizedData.slice(1);
            
            this.log(`${direction} 원본 데이터: ${rows.length}행, 컬럼: ${headers.join(', ')}`);
            
            // 컬럼 매칭
            const columnMapping = {};
            for (const [columnType, candidates] of Object.entries(COLUMN_CANDIDATES)) {
                const matchedColumn = this.findColumnMatch(headers, candidates);
                columnMapping[columnType] = matchedColumn;
                if (matchedColumn) {
                    this.log(`${direction} - ${columnType} 매칭: '${matchedColumn}'`);
                }
            }
            
            // 계약ID가 없으면 첫 번째 컬럼 사용
            if (!columnMapping.contract && headers.length > 0) {
                columnMapping.contract = headers[0];
                this.log(`${direction} - 계약ID 컬럼을 찾지 못해 첫 번째 컬럼 '${headers[0]}'을 사용합니다.`);
            }
            
            // 표준 데이터 생성
            const standardizedRows = [];
            
            rows.forEach((row, index) => {
                try {
                    // 빈 행 건너뛰기
                    if (!row || row.every(cell => !cell || String(cell).trim() === '')) {
                        return;
                    }
                    
                    const contractId = this.getColumnValue(row, headers, columnMapping.contract);
                    const amount = this.toFloat(this.getColumnValue(row, headers, columnMapping.amount));
                    
                    // 유효하지 않은 데이터 필터링
                    if (!contractId || String(contractId).trim() === '' || 
                        String(contractId).toLowerCase() === 'nan' || amount === 0) {
                        return;
                    }
                    
                    const standardRow = {
                        contract_id: String(contractId).trim(),
                        direction: direction,
                        amount: amount,
                        date: this.parseDate(this.getColumnValue(row, headers, columnMapping.date)),
                        party: String(this.getColumnValue(row, headers, columnMapping.party) || '').trim(),
                        owner: String(this.getColumnValue(row, headers, columnMapping.owner) || '').trim(),
                        status: String(this.getColumnValue(row, headers, columnMapping.status) || '').trim(),
                        note: String(this.getColumnValue(row, headers, columnMapping.note) || '').trim(),
                        overdue_flag: this.parseOverdueFlag(this.getColumnValue(row, headers, columnMapping.overdue))
                    };
                    
                    standardizedRows.push(standardRow);
                    
                } catch (rowError) {
                    this.log(`${direction} 행 ${index + 1} 처리 오류: ${rowError.message}`);
                }
            });
            
            this.log(`${direction} 데이터 ${standardizedRows.length}건 표준화 완료`);
            return standardizedRows;
            
        } catch (error) {
            this.log(`${direction} 데이터 표준화 오류: ${error.message}`);
            return [];
        }
    }

    // 헤더 인덱스로 값 추출
    getColumnValue(row, headers, columnName) {
        if (!columnName || !headers.includes(columnName)) return '';
        const index = headers.indexOf(columnName);
        return index >= 0 && index < row.length ? row[index] : '';
    }

    // 날짜 파싱
    parseDate(dateValue) {
        if (!dateValue) return null;
        
        try {
            const date = new Date(dateValue);
            return isNaN(date.getTime()) ? null : date.toISOString().split('T')[0];
        } catch (e) {
            return null;
        }
    }

    // 연체 플래그 파싱
    parseOverdueFlag(overdueValue) {
        if (!overdueValue) return false;
        
        const str = String(overdueValue).toLowerCase().trim();
        return str === 'y' || str === 'yes' || str === 'true' || str === '1' || 
               str === 'o' || str.includes('경과') || str.includes('연체') || 
               str.includes('over');
    }

    // 집계 테이블 생성
    createAggregationTable(baseData) {
        if (!baseData || baseData.length === 0) {
            this.log('집계 테이블: 입력 데이터가 비어있음');
            return [];
        }
        
        try {
            this.log(`집계 테이블 생성 시작: ${baseData.length}건의 데이터`);
            
            // 계약ID별 그룹핑
            const contractGroups = {};
            baseData.forEach(row => {
                const contractId = row.contract_id;
                if (!contractGroups[contractId]) {
                    contractGroups[contractId] = [];
                }
                contractGroups[contractId].push(row);
            });
            
            const contracts = Object.keys(contractGroups);
            this.log(`총 계약 수: ${contracts.length}개`);
            
            const resultRows = [];
            
            contracts.forEach(contractId => {
                try {
                    const contractData = contractGroups[contractId];
                    
                    // 금액 집계
                    const receiptsData = contractData.filter(row => row.direction === '선수금');
                    const advancesData = contractData.filter(row => row.direction === '선급금');
                    
                    const receiptsSum = receiptsData.reduce((sum, row) => sum + row.amount, 0);
                    const advancesSum = advancesData.reduce((sum, row) => sum + row.amount, 0);
                    const gap = receiptsSum - advancesSum;
                    
                    // 메타 정보 추출
                    const owners = contractData.map(row => row.owner).filter(owner => owner && owner.trim() !== '');
                    const parties = contractData.map(row => row.party).filter(party => party && party.trim() !== '');
                    const dates = contractData.map(row => row.date).filter(date => date);
                    
                    // 가장 빈번한 담당자/거래처
                    const mostFrequent = (arr) => {
                        if (arr.length === 0) return '';
                        const frequency = {};
                        arr.forEach(item => {
                            frequency[item] = (frequency[item] || 0) + 1;
                        });
                        return Object.keys(frequency).reduce((a, b) => frequency[a] > frequency[b] ? a : b);
                    };
                    
                    const owner = mostFrequent(owners);
                    const party = mostFrequent(parties);
                    const latestDate = dates.length > 0 ? new Date(Math.max(...dates.map(d => new Date(d)))).toISOString().split('T')[0] : null;
                    
                    const aggregatedRow = {
                        contract_id: contractId,
                        receipts: receiptsSum,
                        advances: advancesSum,
                        gap: gap,
                        party: party,
                        owner: owner,
                        latest_date: latestDate,
                        count: contractData.length,
                        status: contractData[0].status || '진행중'
                    };
                    
                    resultRows.push(aggregatedRow);
                    
                } catch (contractError) {
                    this.log(`계약 ${contractId} 처리 오류: ${contractError.message}`);
                }
            });
            
            this.log(`집계 테이블 생성 완료: ${resultRows.length}개 계약`);
            return resultRows;
            
        } catch (error) {
            this.log(`집계 테이블 생성 오류: ${error.message}`);
            return [];
        }
    }

    // 메인 분석 함수
    async analyzeExcelData(sheetsData) {
        try {
            this.log('Excel 데이터 분석 시작');
            this.log('감지된 시트:', Object.keys(sheetsData));
            
            // 선수금/선급금 시트 찾기
            const receiptsSheet = this.findSheetByKeywords(sheetsData, ['선수금']);
            const advancesSheet = this.findSheetByKeywords(sheetsData, ['선급금']);
            
            if (!receiptsSheet && !advancesSheet) {
                throw new Error('선수금 또는 선급금 시트를 찾을 수 없습니다. 시트 이름을 확인해주세요.');
            }
            
            // 데이터 표준화
            const receiptsData = receiptsSheet ? this.standardizeData(receiptsSheet, '선수금') : [];
            const advancesData = advancesSheet ? this.standardizeData(advancesSheet, '선급금') : [];
            
            // 통합 데이터
            const baseData = [...receiptsData, ...advancesData];
            this.log(`통합 데이터: ${baseData.length}건`);
            
            // 집계 테이블 생성
            const contracts = this.createAggregationTable(baseData);
            
            // 요약 통계
            const totalReceipts = contracts.reduce((sum, c) => sum + c.receipts, 0);
            const totalAdvances = contracts.reduce((sum, c) => sum + c.advances, 0);
            const totalGap = totalReceipts - totalAdvances;
            const overdueCount = baseData.filter(row => row.overdue_flag).length;
            
            const result = {
                summary: {
                    total_receipts: totalReceipts,
                    total_advances: totalAdvances,
                    gap: totalGap,
                    contract_count: contracts.length,
                    overdue_count: overdueCount
                },
                contracts: contracts,
                validation_info: {
                    시트_목록: Object.keys(sheetsData),
                    선수금_원본_행수: receiptsSheet ? receiptsSheet.length - 1 : 0, // 헤더 제외
                    선급금_원본_행수: advancesSheet ? advancesSheet.length - 1 : 0,
                    선수금_표준화_행수: receiptsData.length,
                    선급금_표준화_행수: advancesData.length,
                    총_데이터_건수: baseData.length
                }
            };
            
            this.log('분석 완료:', result.summary);
            return result;
            
        } catch (error) {
            this.log(`Excel 분석 오류: ${error.message}`);
            throw error;
        }
    }
}

// 전역 인스턴스 생성
window.excelAnalyzer = new ExcelAnalyzer();