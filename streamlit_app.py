# app_improved.py — 완전 개선 버전
# ------------------------------------------------------------------
# 개선사항:
# - 모든 데이터 구조 불일치 해결
# - 안전한 에러 처리 및 로깅
# - 성능 최적화된 매칭 알고리즘
# - 향상된 사용자 경험
# - 메모리 최적화
# ------------------------------------------------------------------

import io
import re
import math
import logging
import itertools
import typing as t
from datetime import datetime, timedelta
import pandas as pd
import streamlit as st

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Streamlit 설정
st.set_page_config(
    page_title="선수/선급금 계약별 대시보드", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 선수/선급금 계약별 대시보드 — 완전 개선 버전")
st.caption("엑셀 업로드 → 표준화 → 집계 → 상세/차트 → 교차검증 자동 매칭")

# ============== 상수 및 유틸리티 ==============
_non_digit = re.compile(r"[^0-9\-\.]+")

# 표준 컬럼 정의
STANDARD_COLS = [
    "contract_id", "direction", "amount", "date", "party", 
    "owner", "status", "note", "overdue_flag"
]

# 컬럼 매칭 후보들
COLUMN_CANDIDATES = {
    "contract": ["계약번호", "금형마스터", "프로젝트", "프로젝트코드", "PJT", "PJT코드", "고유번호", "계약코드", "계약id", "계약ID"],
    "amount": ["금액", "선수금", "선급금", "선수금금액", "선급금금액", "합계", "잔액", "amount"],
    "date": ["일자", "청구일", "지급일", "납기일", "요청일", "등록일", "기준일", "date"],
    "party": ["업체명", "거래처", "고객사", "고객명", "상대방", "회사", "vendor", "customer", "업체"],
    "owner": ["담당자", "담당", "담당자명", "PM", "담당부서", "owner"],
    "status": ["진행현황", "정산여부", "상태", "status"],
    "note": ["비고", "메모", "특이사항", "코멘트", "note", "설명"],
    "overdue": ["기한경과", "연체", "overdue", "경과"]
}

def to_float(x: t.Any) -> float:
    """안전한 숫자 변환"""
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)): 
            return 0.0
        if isinstance(x, (int, float)): 
            return float(x)
        
        s = str(x).strip()
        if s == "" or s.lower() in {"nan", "none", "null"}: 
            return 0.0
        
        # 숫자가 아닌 문자 제거
        s = _non_digit.sub("", s)
        if s in {"", "-", "."}: 
            return 0.0
        
        return float(s)
    except Exception as e:
        logger.warning(f"숫자 변환 실패: {x} -> {e}")
        return 0.0

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """컬럼명 정규화"""
    if df.empty:
        return df
    
    df = df.copy()
    new_cols, seen = [], set()
    
    for c in df.columns:
        nc = str(c).strip()
        nc = re.sub(r"\s+", " ", nc)
        
        # 중복 컬럼명 처리
        if nc in seen:
            k, base = 2, nc
            while nc in seen:
                nc = f"{base}_{k}"
                k += 1
        
        seen.add(nc)
        new_cols.append(nc)
    
    df.columns = new_cols
    return df

def find_column_match(df: pd.DataFrame, candidates: t.List[str]) -> t.Optional[str]:
    """컬럼 매칭 로직 개선"""
    if df.empty:
        return None
    
    cols = list(df.columns)
    lower_map = {c.lower(): c for c in cols}
    
    # 정확한 매치 우선
    for cand in candidates:
        if cand in df.columns:
            return cand
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    
    # 부분 매치
    for cand in candidates:
        pattern = re.compile(re.escape(cand), re.IGNORECASE)
        for c in cols:
            if pattern.search(str(c)):
                return c
    
    return None

def read_sheet_safely(excel_file: pd.ExcelFile, keywords: t.List[str]) -> pd.DataFrame:
    """시트 읽기 안전 처리"""
    try:
        def normalize_name(s: str) -> str:
            return re.sub(r"\s+", "", s.lower())
        
        target_sheet = None
        for sheet_name in excel_file.sheet_names:
            normalized = normalize_name(sheet_name)
            if any(normalize_name(k) in normalized for k in keywords):
                target_sheet = sheet_name
                break
        
        if target_sheet is None:
            logger.warning(f"키워드 {keywords}에 해당하는 시트를 찾을 수 없습니다.")
            return pd.DataFrame()
        
        df = pd.read_excel(excel_file, sheet_name=target_sheet, dtype=str)
        return normalize_columns(df)
        
    except Exception as e:
        logger.error(f"시트 읽기 오류: {e}")
        return pd.DataFrame()

def standardize_data(df: pd.DataFrame, direction: str) -> pd.DataFrame:
    """데이터 표준화 개선"""
    if df is None or df.empty:
        return pd.DataFrame(columns=STANDARD_COLS)
    
    try:
        df = normalize_columns(df)
        
        # 컬럼 매칭
        col_mapping = {}
        for col_type, candidates in COLUMN_CANDIDATES.items():
            matched_col = find_column_match(df, candidates)
            col_mapping[col_type] = matched_col
        
        # 계약ID가 없으면 첫 번째 컬럼 사용
        if col_mapping["contract"] is None and not df.empty:
            col_mapping["contract"] = df.columns[0]
            logger.info(f"계약ID 컬럼을 찾지 못해 첫 번째 컬럼 '{df.columns[0]}'을 사용합니다.")
        
        # 표준 데이터프레임 생성
        result = pd.DataFrame()
        
        result["contract_id"] = (
            df[col_mapping["contract"]].astype(str).str.strip() 
            if col_mapping["contract"] else "(미지정)"
        )
        result["direction"] = direction
        result["amount"] = (
            df[col_mapping["amount"]].apply(to_float) 
            if col_mapping["amount"] else 0.0
        )
        result["date"] = (
            pd.to_datetime(df[col_mapping["date"]], errors="coerce") 
            if col_mapping["date"] else pd.NaT
        )
        result["party"] = (
            df[col_mapping["party"]].astype(str).str.strip() 
            if col_mapping["party"] else ""
        )
        result["owner"] = (
            df[col_mapping["owner"]].astype(str).str.strip() 
            if col_mapping["owner"] else ""
        )
        result["status"] = (
            df[col_mapping["status"]].astype(str).str.strip() 
            if col_mapping["status"] else ""
        )
        result["note"] = (
            df[col_mapping["note"]].astype(str).str.strip() 
            if col_mapping["note"] else ""
        )
        
        # 연체 플래그 처리
        if col_mapping["overdue"]:
            overdue_data = df[col_mapping["overdue"]].astype(str).str.strip().str.lower()
            result["overdue_flag"] = (
                overdue_data.isin(["y", "yes", "true", "1", "o"]) |
                overdue_data.str.contains("경과|연체|over", na=False)
            )
        else:
            result["overdue_flag"] = False
        
        # 유효한 데이터만 필터링
        result = result[
            (result["amount"] != 0) & 
            (result["contract_id"].str.strip() != "") &
            (result["contract_id"] != "nan")
        ]
        
        logger.info(f"{direction} 데이터 {len(result)}건 표준화 완료")
        return result[STANDARD_COLS]
        
    except Exception as e:
        logger.error(f"데이터 표준화 오류: {e}")
        return pd.DataFrame(columns=STANDARD_COLS)

def create_aggregation_table(base_df: pd.DataFrame) -> pd.DataFrame:
    """집계 테이블 생성 개선"""
    if base_df.empty:
        return pd.DataFrame(columns=[
            "계약ID", "선수금_합계", "선급금_합계", "Gap(선수-선급)", 
            "담당자", "주요거래처", "최근일자", "건수"
        ])
    
    try:
        all_contracts = base_df["contract_id"].unique()
        result_rows = []
        
        for contract in all_contracts:
            if pd.isna(contract) or str(contract).strip() == "":
                continue
                
            contract_data = base_df[base_df["contract_id"] == contract]
            
            # 금액 집계
            선수금_합계 = contract_data[contract_data["direction"] == "선수금"]["amount"].sum()
            선급금_합계 = contract_data[contract_data["direction"] == "선급금"]["amount"].sum()
            gap = 선수금_합계 - 선급금_합계
            
            # 메타 정보 추출
            담당자_list = contract_data["owner"].dropna().tolist()
            담당자 = max(set(담당자_list), key=담당자_list.count) if 담당자_list else ""
            
            거래처_list = contract_data["party"].dropna().tolist()
            주요거래처 = max(set(거래처_list), key=거래처_list.count) if 거래처_list else ""
            
            최근일자 = contract_data["date"].max() if contract_data["date"].notna().any() else pd.NaT
            건수 = len(contract_data)
            
            result_rows.append({
                "계약ID": contract,
                "선수금_합계": 선수금_합계,
                "선급금_합계": 선급금_합계,
                "Gap(선수-선급)": gap,
                "담당자": 담당자,
                "주요거래처": 주요거래처,
                "최근일자": 최근일자,
                "건수": 건수
            })
        
        result_df = pd.DataFrame(result_rows)
        logger.info(f"집계 테이블 생성 완료: {len(result_df)}개 계약")
        return result_df
        
    except Exception as e:
        logger.error(f"집계 테이블 생성 오류: {e}")
        return pd.DataFrame(columns=[
            "계약ID", "선수금_합계", "선급금_합계", "Gap(선수-선급)", 
            "담당자", "주요거래처", "최근일자", "건수"
        ])

# ============== 캐시된 데이터 로드 함수 ==============
@st.cache_data(show_spinner=True, ttl=3600)
def load_excel_data(file_bytes: bytes):
    """엑셀 데이터 로드 (캐시 적용)"""
    try:
        excel_file = pd.ExcelFile(io.BytesIO(file_bytes), engine="openpyxl")
        
        # 시트 읽기
        receipts_raw = read_sheet_safely(excel_file, ["선수금"])
        advances_raw = read_sheet_safely(excel_file, ["선급금"])
        
        # 데이터 표준화
        receipts_std = standardize_data(receipts_raw, "선수금")
        advances_std = standardize_data(advances_raw, "선급금")
        
        # 통합 데이터
        base_data = pd.concat([receipts_std, advances_std], ignore_index=True)
        
        # 집계 테이블
        aggregation_table = create_aggregation_table(base_data)
        
        # 데이터 검증
        validation_info = {
            "선수금_원본_행수": len(receipts_raw),
            "선급금_원본_행수": len(advances_raw),
            "선수금_표준화_행수": len(receipts_std),
            "선급금_표준화_행수": len(advances_std),
            "총_계약수": len(aggregation_table),
            "시트_목록": excel_file.sheet_names
        }
        
        return base_data, aggregation_table, receipts_std, advances_std, validation_info
        
    except Exception as e:
        logger.error(f"엑셀 데이터 로드 실패: {e}")
        empty_df = pd.DataFrame(columns=STANDARD_COLS)
        empty_agg = pd.DataFrame(columns=[
            "계약ID", "선수금_합계", "선급금_합계", "Gap(선수-선급)", 
            "담당자", "주요거래처", "최근일자", "건수"
        ])
        error_info = {"error": str(e)}
        return empty_df, empty_agg, empty_df, empty_df, error_info

def apply_filters(view_df: pd.DataFrame, query_text: str, owner_filter: str) -> pd.DataFrame:
    """필터링 로직 개선"""
    if view_df.empty:
        return view_df
    
    try:
        filtered_df = view_df.copy()
        
        # 검색 필터 적용
        if query_text:
            query_lower = query_text.strip().lower()
            search_mask = (
                filtered_df["계약ID"].astype(str).str.lower().str.contains(query_lower, na=False) |
                filtered_df["주요거래처"].astype(str).str.lower().str.contains(query_lower, na=False) |
                filtered_df["담당자"].astype(str).str.lower().str.contains(query_lower, na=False)
            )
            filtered_df = filtered_df[search_mask]
        
        # 담당자 필터 적용
        if owner_filter:
            owners = [o.strip().lower() for o in owner_filter.split(',') if o.strip()]
            if owners:
                owner_mask = filtered_df["담당자"].astype(str).str.lower().isin(owners)
                filtered_df = filtered_df[owner_mask]
        
        return filtered_df
        
    except Exception as e:
        logger.error(f"필터링 오류: {e}")
        return view_df

def safe_sort(df: pd.DataFrame, sort_column: str) -> pd.DataFrame:
    """안전한 정렬 함수"""
    if df.empty:
        return df
    
    try:
        if sort_column not in df.columns:
            st.warning(f"정렬 컬럼 '{sort_column}'이 존재하지 않습니다.")
            return df
        
        # 숫자형 컬럼은 내림차순, 문자형은 오름차순
        if pd.api.types.is_numeric_dtype(df[sort_column]):
            return df.sort_values(by=sort_column, ascending=False)
        else:
            return df.sort_values(by=sort_column, ascending=True)
            
    except Exception as e:
        logger.error(f"정렬 오류: {e}")
        return df

# ============== 개선된 매칭 알고리즘 ==============
def simple_tokenize(text: str) -> set:
    """간단한 토큰화"""
    if not text or pd.isna(text):
        return set()
    
    text = str(text)
    # 한글, 영문, 숫자만 추출
    text = re.sub(r'[^0-9A-Za-z가-힣\-_/]+', ' ', text)
    tokens = [t for t in text.split() if len(t) >= 2]
    return set(tokens)

def calculate_match_score(receipt_row: pd.Series, advance_row: pd.Series, 
                         config: dict) -> float:
    """매칭 점수 계산"""
    score = 0.0
    
    try:
        ramt = float(receipt_row.get('amount', 0) or 0)
        aamt = float(advance_row.get('amount', 0) or 0)
        rdate = receipt_row.get('date', pd.NaT)
        adate = advance_row.get('date', pd.NaT)
        
        # 금액 점수 (40%)
        if config.get('use_amount', True) and ramt > 0:
            amt_diff = abs(ramt - aamt)
            tol_abs = config.get('amount_tol', 0)
            tol_pct = config.get('amount_tol_pct', 1) / 100.0
            tolerance = max(tol_abs, ramt * tol_pct)
            
            if amt_diff <= tolerance:
                score += max(0.0, 1.0 - (amt_diff / (ramt + 1e-9))) * 0.4
        
        # 날짜 점수 (30%)
        if config.get('use_date', True) and pd.notna(rdate) and pd.notna(adate):
            date_diff = abs((rdate - adate).days)
            date_window = config.get('date_window', 30)
            if date_diff <= date_window:
                score += max(0.0, 1.0 - (date_diff / (date_window + 1))) * 0.3
        
        # 계약ID 점수 (20%)
        if config.get('use_contract_soft', True):
            r_contract = str(receipt_row.get('contract_id', '')).strip()
            a_contract = str(advance_row.get('contract_id', '')).strip()
            if (r_contract == a_contract and r_contract != '' and 
                r_contract not in ['nan', 'None', '(미지정)']):
                score += 0.2
        
        # 거래처/텍스트 유사도 점수 (10%)
        if config.get('use_party_soft', True):
            r_tokens = simple_tokenize(
                f"{receipt_row.get('party', '')} {receipt_row.get('note', '')} "
                f"{receipt_row.get('status', '')}"
            )
            a_tokens = simple_tokenize(
                f"{advance_row.get('party', '')} {advance_row.get('note', '')} "
                f"{advance_row.get('status', '')}"
            )
            
            if r_tokens and a_tokens:
                intersection = len(r_tokens.intersection(a_tokens))
                union = len(r_tokens.union(a_tokens))
                if union > 0:
                    similarity = intersection / union
                    score += similarity * 0.1
        
        return score
        
    except Exception as e:
        logger.error(f"점수 계산 오류: {e}")
        return 0.0

@st.cache_data(show_spinner=True)
def compute_matches_optimized(receipts_df: pd.DataFrame, advances_df: pd.DataFrame, 
                             config: dict) -> pd.DataFrame:
    """최적화된 매칭 알고리즘"""
    if receipts_df.empty or advances_df.empty:
        return pd.DataFrame(columns=['rid', 'aids', 'sum_adv', 'gap', 'score', 'match_type'])
    
    try:
        rec = receipts_df.reset_index(drop=True).copy()
        adv = advances_df.reset_index(drop=True).copy()
        
        rec['rid'] = rec.index
        adv['aid'] = adv.index
        
        matches = []
        used_advances = set()  # 중복 매칭 방지
        
        # 각 선수금에 대해 매칭 시도
        for _, receipt in rec.iterrows():
            ramt = float(receipt.get('amount', 0) or 0)
            if ramt <= 0:
                continue
            
            # 사용되지 않은 선급금만 고려
            available_advances = adv[~adv['aid'].isin(used_advances)].copy()
            if available_advances.empty:
                continue
            
            # 후보 필터링
            candidates = filter_candidates(receipt, available_advances, config)
            if candidates.empty:
                continue
            
            # 1:1 매칭 시도
            best_single = find_best_single_match(receipt, candidates, config)
            
            # 1:N 매칭 시도 (조건: 1:1이 없거나 정확도가 낮은 경우)
            best_combo = None
            if (best_single is None or best_single['score'] < 0.7 or 
                abs(best_single['gap']) > config.get('amount_tol', 0)):
                
                max_combo = min(config.get('max_combo', 3), len(candidates))
                if max_combo > 1:
                    best_combo = find_best_combo_match(receipt, candidates, config, max_combo)
            
            # 최적 매칭 선택
            final_match = None
            if best_combo and (best_single is None or best_combo['score'] > best_single['score']):
                final_match = best_combo
            elif best_single:
                final_match = best_single
            
            if final_match:
                # 사용된 선급금 기록
                used_advances.update(final_match['aids'])
                matches.append(final_match)
        
        result_df = pd.DataFrame(matches) if matches else pd.DataFrame(
            columns=['rid', 'aids', 'sum_adv', 'gap', 'score', 'match_type']
        )
        
        logger.info(f"매칭 완료: {len(matches)}건")
        return result_df
        
    except Exception as e:
        logger.error(f"매칭 알고리즘 오류: {e}")
        return pd.DataFrame(columns=['rid', 'aids', 'sum_adv', 'gap', 'score', 'match_type'])

def filter_candidates(receipt: pd.Series, advances: pd.DataFrame, config: dict) -> pd.DataFrame:
    """후보 선급금 필터링"""
    candidates = advances.copy()
    
    try:
        ramt = float(receipt.get('amount', 0) or 0)
        rdate = receipt.get('date', pd.NaT)
        
        # 날짜 필터
        if config.get('use_date', True) and pd.notna(rdate):
            date_window = config.get('date_window', 30)
            date_mask = (
                candidates['date'].isna() |
                candidates['date'].between(
                    rdate - pd.Timedelta(days=date_window),
                    rdate + pd.Timedelta(days=date_window)
                )
            )
            candidates = candidates[date_mask]
        
        # 금액 필터 (넉넉하게)
        if config.get('use_amount', True) and ramt > 0:
            tol_abs = config.get('amount_tol', 0)
            tol_pct = config.get('amount_tol_pct', 1) / 100.0
            tolerance = max(tol_abs, ramt * tol_pct) * 2  # 필터링은 넉넉하게
            
            amount_mask = (
                (candidates['amount'] >= ramt - tolerance) &
                (candidates['amount'] <= ramt + tolerance)
            )
            candidates = candidates[amount_mask]
        
        return candidates
        
    except Exception as e:
        logger.error(f"후보 필터링 오류: {e}")
        return advances

def find_best_single_match(receipt: pd.Series, candidates: pd.DataFrame, config: dict) -> dict:
    """최적 1:1 매칭 찾기"""
    best_match = None
    best_score = 0
    
    try:
        for _, candidate in candidates.iterrows():
            score = calculate_match_score(receipt, candidate, config)
            
            if score > best_score:
                aamt = float(candidate.get('amount', 0) or 0)
                ramt = float(receipt.get('amount', 0) or 0)
                
                best_match = {
                    'rid': int(receipt['rid']),
                    'aids': [int(candidate['aid'])],
                    'sum_adv': float(aamt),
                    'gap': float(ramt - aamt),
                    'score': float(score),
                    'match_type': '1:1'
                }
                best_score = score
        
        return best_match
        
    except Exception as e:
        logger.error(f"1:1 매칭 오류: {e}")
        return None

def find_best_combo_match(receipt: pd.Series, candidates: pd.DataFrame, 
                         config: dict, max_combo: int) -> dict:
    """최적 1:N 조합 매칭 찾기"""
    try:
        ramt = float(receipt.get('amount', 0) or 0)
        if ramt <= 0:
            return None
        
        tol_abs = config.get('amount_tol', 0)
        tol_pct = config.get('amount_tol_pct', 1) / 100.0
        tolerance = max(tol_abs, ramt * tol_pct)
        
        # 금액 기준으로 정렬하여 조합 수 줄이기
        candidates_sorted = candidates.sort_values('amount').head(10)  # 상위 10개만
        aids = candidates_sorted['aid'].tolist()
        
        best_combo = None
        best_score = 0
        
        # 2개부터 max_combo개까지 조합 시도
        for combo_size in range(2, min(max_combo + 1, len(aids) + 1)):
            for combo_aids in itertools.combinations(aids, combo_size):
                combo_rows = candidates_sorted.set_index('aid').loc[list(combo_aids)]
                total_amount = float(combo_rows['amount'].sum())
                
                # 금액 허용 범위 확인
                if abs(total_amount - ramt) <= tolerance:
                    # 조합 점수 계산 (평균)
                    combo_score = 0
                    for _, candidate in combo_rows.iterrows():
                        combo_score += calculate_match_score(receipt, candidate, config)
                    combo_score = combo_score / combo_size
                    
                    # 금액 정확도 보너스
                    accuracy_bonus = max(0, 1 - (abs(total_amount - ramt) / (ramt + 1e-9)))
                    combo_score += accuracy_bonus * 0.1
                    
                    if combo_score > best_score:
                        best_combo = {
                            'rid': int(receipt['rid']),
                            'aids': list(map(int, combo_aids)),
                            'sum_adv': float(total_amount),
                            'gap': float(ramt - total_amount),
                            'score': float(combo_score),
                            'match_type': f'1:{combo_size}'
                        }
                        best_score = combo_score
        
        return best_combo
        
    except Exception as e:
        logger.error(f"조합 매칭 오류: {e}")
        return None

# ============== UI 구성 ==============

# 사이드바
with st.sidebar:
    st.header("📁 데이터 업로드")
    uploaded_file = st.file_uploader(
        "엑셀 파일을 선택하세요", 
        type=["xlsx", "xlsm", "xls"],
        help="xlsx, xlsm, xls 형식을 지원합니다. 매크로 파일(xlsm)은 값만 읽습니다."
    )
    
    if uploaded_file:
        st.success(f"파일 업로드 완료: {uploaded_file.name}")
    
    st.divider()
    
    st.header("🔧 매칭 설정")
    
    # 매칭 설정
    matching_config = {
        'use_contract_soft': st.checkbox("계약ID 일치 가중치", value=True, 
                                        help="같은 계약ID끼리 매칭 점수 증가"),
        'use_amount': st.checkbox("금액 조건 사용", value=True,
                                 help="금액 허용 범위 내에서만 매칭"),
        'amount_tol': st.number_input("금액 허용 오차(원)", 
                                     min_value=0, max_value=1000000000, 
                                     value=0, step=1000,
                                     help="절대 금액 차이 허용 범위"),
        'amount_tol_pct': st.slider("금액 허용 오차(%)", 
                                   min_value=0, max_value=20, value=1,
                                   help="상대 금액 차이 허용 범위"),
        'use_date': st.checkbox("일자 조건 사용", value=True,
                               help="일자 범위 내에서만 매칭"),
        'date_window': st.slider("일자 윈도우(±일)", 
                                min_value=0, max_value=180, value=30,
                                help="날짜 차이 허용 범위"),
        'use_party_soft': st.checkbox("거래처 텍스트 유사도 가중치", value=True,
                                     help="거래처/메모 텍스트 유사성으로 점수 증가"),
        'max_combo': st.slider("부분합 매칭 최대 묶음 수", 
                              min_value=1, max_value=5, value=3,
                              help="1개 선수금에 대해 최대 몇 개까지 선급금 조합 매칭")
    }
    
    st.divider()
    st.header("ℹ️ 도움말")
    with st.expander("매칭 알고리즘 설명"):
        st.markdown("""
        **매칭 점수 구성:**
        - 금액 정확도: 40%
        - 날짜 근접성: 30% 
        - 계약ID 일치: 20%
        - 텍스트 유사도: 10%
        
        **매칭 순서:**
        1. 1:1 매칭 시도
        2. 점수가 낮으면 1:N 조합 매칭 시도
        3. 최고 점수 매칭 선택
        """)

# 메인 화면
if uploaded_file is None:
    st.info("👈 좌측에서 엑셀 파일을 업로드하세요.")
    st.markdown("""
    ### 📋 사용법
    1. **엑셀 파일 준비**: '선수금', '선급금' 시트가 포함된 파일
    2. **파일 업로드**: 좌측 사이드바에서 파일 선택
    3. **데이터 확인**: 자동으로 표준화된 데이터 확인
    4. **매칭 설정**: 필요시 매칭 조건 조정
    5. **결과 분석**: 계약별 상세 내역 및 자동 매칭 결과 확인
    
    ### 🎯 주요 기능
    - **자동 컬럼 인식**: 다양한 컬럼명 자동 매칭
    - **데이터 표준화**: 금액, 날짜 등 자동 변환
    - **지능형 매칭**: AI 기반 선수금-선급금 매칭
    - **시각화**: 차트와 그래프로 데이터 분석
    """)
    st.stop()

# 데이터 로드
try:
    with st.spinner("📊 엑셀 데이터를 분석하고 있습니다..."):
        base_data, agg_table, receipts_data, advances_data, validation_info = load_excel_data(uploaded_file.read())
    
    # 데이터 검증 정보 표시
    if "error" in validation_info:
        st.error(f"❌ 데이터 로드 실패: {validation_info['error']}")
        st.stop()
    
    # 데이터 검증 성공 메시지
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(f"📄 감지된 시트: {', '.join(validation_info['시트_목록'])}")
    with col2:
        st.info(f"📊 원본 데이터: 선수금 {validation_info['선수금_원본_행수']}건, 선급금 {validation_info['선급금_원본_행수']}건")
    with col3:
        st.info(f"✅ 표준화 완료: 선수금 {validation_info['선수금_표준화_행수']}건, 선급금 {validation_info['선급금_표준화_행수']}건")

except Exception as e:
    st.error(f"❌ 파일 처리 중 오류가 발생했습니다: {e}")
    st.stop()

# KPI 대시보드
st.subheader("📈 주요 지표")
col1, col2, col3, col4, col5 = st.columns(5)

total_receipts = base_data.loc[base_data['direction'] == '선수금', 'amount'].sum()
total_advances = base_data.loc[base_data['direction'] == '선급금', 'amount'].sum()
total_gap = total_receipts - total_advances
contract_count = agg_table.shape[0]
avg_gap = total_gap / contract_count if contract_count > 0 else 0

with col1:
    st.metric("💰 총 선수금", f"{total_receipts:,.0f} 원",
              help="고객으로부터 미리 받은 총 금액")
with col2:
    st.metric("💸 총 선급금", f"{total_advances:,.0f} 원",
              help="협력사에 미리 지급한 총 금액")
with col3:
    delta_color = "normal" if total_gap >= 0 else "inverse"
    st.metric("📊 전체 Gap", f"{total_gap:,.0f} 원", 
              delta=f"평균 {avg_gap:,.0f}원/계약",
              delta_color=delta_color,
              help="선수금 - 선급금 (양수일 때 유리)")
with col4:
    st.metric("📋 계약 수", f"{contract_count:,}개",
              help="분석된 총 계약 건수")
with col5:
    overdue_count = base_data['overdue_flag'].sum()
    st.metric("⚠️ 연체 건", f"{overdue_count}건",
              help="기한이 경과된 거래 건수")

st.divider()

# 필터 및 검색
st.subheader("🔍 계약 검색 및 필터")
filter_col1, filter_col2, filter_col3, filter_col4 = st.columns([3, 2, 2, 1])

with filter_col1:
    search_query = st.text_input("🔎 통합 검색", 
                                placeholder="계약ID, 거래처명, 담당자명으로 검색...",
                                help="여러 조건을 동시에 검색합니다")

with filter_col2:
    owner_filter = st.text_input("👤 담당자 필터", 
                                placeholder="쉼표로 구분하여 여러명 입력",
                                help="예: 김철수, 이영희")

with filter_col3:
    sort_options = ["Gap(선수-선급)", "선수금_합계", "선급금_합계", "계약ID", "최근일자", "건수"]
    sort_by = st.selectbox("📊 정렬 기준", sort_options, 
                          help="테이블 정렬 기준을 선택하세요")

with filter_col4:
    show_only_gap = st.checkbox("Gap만 표시", 
                               help="Gap이 있는 계약만 표시")

# 필터 적용
filtered_table = apply_filters(agg_table, search_query, owner_filter)

if show_only_gap and not filtered_table.empty:
    filtered_table = filtered_table[filtered_table["Gap(선수-선급)"] != 0]

# 정렬 적용
if not filtered_table.empty:
    filtered_table = safe_sort(filtered_table, sort_by)

# 테이블 표시
st.subheader("📋 계약별 집계 현황")

if filtered_table.empty:
    st.warning("🔍 검색 조건에 맞는 데이터가 없습니다. 검색어나 필터를 확인해주세요.")
else:
    # 테이블 스타일링을 위한 함수
    def style_dataframe(df):
        def color_gap(val):
            if pd.isna(val) or val == 0:
                return 'background-color: #f0f0f0'
            elif val > 0:
                return 'background-color: #d4edda; color: #155724'  # 녹색 (유리)
            else:
                return 'background-color: #f8d7da; color: #721c24'  # 빨간색 (불리)
        
        styled = df.style.applymap(color_gap, subset=['Gap(선수-선급)'])
        
        # 금액 컬럼 포맷팅
        money_cols = ['선수금_합계', '선급금_합계', 'Gap(선수-선급)']
        for col in money_cols:
            if col in df.columns:
                styled = styled.format({col: '{:,.0f}'})
        
        return styled
    
    styled_table = style_dataframe(filtered_table)
    
    st.dataframe(
        styled_table,
        use_container_width=True,
        height=400,
        column_config={
            "계약ID": st.column_config.TextColumn("계약ID", width="medium"),
            "선수금_합계": st.column_config.NumberColumn("선수금 합계", format="₩%.0f"),
            "선급금_합계": st.column_config.NumberColumn("선급금 합계", format="₩%.0f"), 
            "Gap(선수-선급)": st.column_config.NumberColumn("Gap", format="₩%.0f"),
            "최근일자": st.column_config.DateColumn("최근 일자"),
            "건수": st.column_config.NumberColumn("거래 건수", format="%d건")
        }
    )
    
    # 요약 통계
    st.caption(f"📊 총 {len(filtered_table)}개 계약 | "
              f"양의 Gap: {len(filtered_table[filtered_table['Gap(선수-선급)'] > 0])}개 | "
              f"음의 Gap: {len(filtered_table[filtered_table['Gap(선수-선급)'] < 0])}개 | "
              f"Gap 없음: {len(filtered_table[filtered_table['Gap(선수-선급)'] == 0])}개")

# 계약 상세 분석
st.divider()
st.subheader("🔬 계약 상세 분석")

contract_list = ["(계약을 선택하세요)"] + filtered_table["계약ID"].tolist()
selected_contract = st.selectbox("📋 분석할 계약 선택", contract_list, 
                                help="상세 분석을 원하는 계약을 선택하세요")

if selected_contract and selected_contract != "(계약을 선택하세요)":
    # 선택된 계약 데이터
    contract_detail = base_data[base_data["contract_id"] == selected_contract].copy()
    
    if contract_detail.empty:
        st.error("❌ 선택된 계약의 상세 데이터를 찾을 수 없습니다.")
    else:
        # 계약 요약 정보
        receipts_sum = contract_detail.loc[contract_detail["direction"] == "선수금", "amount"].sum()
        advances_sum = contract_detail.loc[contract_detail["direction"] == "선급금", "amount"].sum()
        gap = receipts_sum - advances_sum
        
        # 메트릭 표시
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        with metric_col1:
            st.metric("💰 선수금 총계", f"{receipts_sum:,.0f} 원",
                     help="고객으로부터 받은 총 금액")
        with metric_col2:
            st.metric("💸 선급금 총계", f"{advances_sum:,.0f} 원",
                     help="협력사에 지급한 총 금액")
        with metric_col3:
            delta_color = "normal" if gap >= 0 else "inverse"
            st.metric("📊 Gap", f"{gap:,.0f} 원", 
                     delta=f"{gap/receipts_sum*100:.1f}%" if receipts_sum > 0 else "0%",
                     delta_color=delta_color,
                     help="선수금 - 선급금")
        with metric_col4:
            total_count = len(contract_detail)
            st.metric("📋 총 거래 건수", f"{total_count}건",
                     help="해당 계약의 총 거래 건수")
        
        # 상세 정보 탭
        detail_tab1, detail_tab2, detail_tab3, detail_tab4 = st.tabs([
            "📊 선수금 상세", "📈 선급금 상세", "🤖 자동 매칭", "📉 시각화 분석"
        ])
        
        with detail_tab1:
            receipts_detail = contract_detail[contract_detail["direction"] == "선수금"].copy()
            
            if receipts_detail.empty:
                st.info("ℹ️ 해당 계약에 선수금 데이터가 없습니다.")
            else:
                st.write(f"**선수금 거래 내역** ({len(receipts_detail)}건)")
                
                # 날짜 포맷팅
                receipts_display = receipts_detail.copy()
                receipts_display["date"] = pd.to_datetime(receipts_display["date"]).dt.strftime('%Y-%m-%d')
                receipts_display = receipts_display.fillna('')
                
                st.dataframe(
                    receipts_display[STANDARD_COLS].rename(columns={
                        "contract_id": "계약ID", "direction": "구분", "amount": "금액",
                        "date": "일자", "party": "거래처", "owner": "담당자",
                        "status": "상태", "note": "비고", "overdue_flag": "연체여부"
                    }),
                    use_container_width=True,
                    column_config={
                        "금액": st.column_config.NumberColumn("금액", format="₩%.0f"),
                        "연체여부": st.column_config.CheckboxColumn("연체")
                    }
                )
                
                # 월별 집계 차트
                if not receipts_detail.empty:
                    receipts_chart_data = receipts_detail.copy()
                    receipts_chart_data['date'] = pd.to_datetime(receipts_chart_data['date'])
                    receipts_chart_data = receipts_chart_data.dropna(subset=['date'])
                    
                    if not receipts_chart_data.empty:
                        monthly_receipts = (receipts_chart_data
                                          .groupby(receipts_chart_data['date'].dt.to_period('M'))['amount']
                                          .sum()
                                          .reset_index())
                        monthly_receipts['date'] = monthly_receipts['date'].dt.to_timestamp()
                        
                        st.subheader("📊 월별 선수금 추이")
                        st.bar_chart(monthly_receipts.set_index('date')['amount'])
        
        with detail_tab2:
            advances_detail = contract_detail[contract_detail["direction"] == "선급금"].copy()
            
            if advances_detail.empty:
                st.info("ℹ️ 해당 계약에 선급금 데이터가 없습니다.")
            else:
                st.write(f"**선급금 거래 내역** ({len(advances_detail)}건)")
                
                # 날짜 포맷팅
                advances_display = advances_detail.copy()
                advances_display["date"] = pd.to_datetime(advances_display["date"]).dt.strftime('%Y-%m-%d')
                advances_display = advances_display.fillna('')
                
                st.dataframe(
                    advances_display[STANDARD_COLS].rename(columns={
                        "contract_id": "계약ID", "direction": "구분", "amount": "금액",
                        "date": "일자", "party": "거래처", "owner": "담당자",
                        "status": "상태", "note": "비고", "overdue_flag": "연체여부"
                    }),
                    use_container_width=True,
                    column_config={
                        "금액": st.column_config.NumberColumn("금액", format="₩%.0f"),
                        "연체여부": st.column_config.CheckboxColumn("연체")
                    }
                )
                
                # 월별 집계 차트
                if not advances_detail.empty:
                    advances_chart_data = advances_detail.copy()
                    advances_chart_data['date'] = pd.to_datetime(advances_chart_data['date'])
                    advances_chart_data = advances_chart_data.dropna(subset=['date'])
                    
                    if not advances_chart_data.empty:
                        monthly_advances = (advances_chart_data
                                          .groupby(advances_chart_data['date'].dt.to_period('M'))['amount']
                                          .sum()
                                          .reset_index())
                        monthly_advances['date'] = monthly_advances['date'].dt.to_timestamp()
                        
                        st.subheader("📊 월별 선급금 추이")
                        st.bar_chart(monthly_advances.set_index('date')['amount'])
        
        with detail_tab3:
            contract_receipts = contract_detail[contract_detail["direction"] == "선수금"].copy()
            all_advances = base_data[base_data["direction"] == "선급금"].copy()
            
            if contract_receipts.empty or all_advances.empty:
                st.info("ℹ️ 매칭을 위한 선수금 또는 선급금 데이터가 부족합니다.")
            else:
                st.write("**🤖 AI 자동 매칭 결과**")
                
                with st.spinner("🔄 최적 매칭을 분석하고 있습니다..."):
                    matching_result = compute_matches_optimized(
                        contract_receipts, all_advances, matching_config
                    )
                
                if matching_result.empty:
                    st.warning("⚠️ 현재 설정 조건으로는 자동 매칭 결과가 없습니다. "
                             "좌측 사이드바에서 허용 오차나 윈도우 범위를 넓혀보세요.")
                else:
                    # 매칭 결과 표시용 데이터 준비
                    display_matches = []
                    contract_receipts_indexed = contract_receipts.reset_index(drop=True)
                    all_advances_indexed = all_advances.reset_index(drop=True)
                    
                    for _, match in matching_result.iterrows():
                        receipt_info = contract_receipts_indexed.loc[match['rid']]
                        advance_infos = all_advances_indexed.loc[match['aids']]
                        
                        if len(match['aids']) == 1:
                            advance_desc = f"#{match['aids'][0]}: {advance_infos['amount']:,.0f}원"
                            advance_party = advance_infos['party']
                            advance_date = advance_infos['date']
                        else:
                            advance_desc = ", ".join([
                                f"#{aid}: {amt:,.0f}원" 
                                for aid, amt in zip(match['aids'], advance_infos['amount'])
                            ])
                            advance_party = " + ".join(advance_infos['party'].unique())
                            advance_date = advance_infos['date'].max()
                        
                        display_matches.append({
                            "선수금_금액": f"{receipt_info['amount']:,.0f}원",
                            "선수금_일자": receipt_info['date'].strftime('%Y-%m-%d') if pd.notna(receipt_info['date']) else '',
                            "선수금_거래처": receipt_info['party'],
                            "선수금_비고": receipt_info['note'][:50] + "..." if len(str(receipt_info['note'])) > 50 else receipt_info['note'],
                            "매칭된_선급금": advance_desc,
                            "선급금_거래처": advance_party,
                            "선급금_합계": f"{match['sum_adv']:,.0f}원",
                            "차이(Gap)": f"{match['gap']:,.0f}원",
                            "매칭_유형": match['match_type'],
                            "신뢰도": f"{match['score']:.2f}",
                        })
                    
                    matches_df = pd.DataFrame(display_matches)
                    
                    # 신뢰도에 따른 색상 구분
                    def color_confidence(val):
                        try:
                            score = float(val)
                            if score >= 0.8:
                                return 'background-color: #d4edda; color: #155724'  # 높음: 녹색
                            elif score >= 0.5:
                                return 'background-color: #fff3cd; color: #856404'  # 보통: 노란색
                            else:
                                return 'background-color: #f8d7da; color: #721c24'  # 낮음: 빨간색
                        except:
                            return ''
                    
                    styled_matches = matches_df.style.applymap(color_confidence, subset=['신뢰도'])
                    
                    st.dataframe(styled_matches, use_container_width=True, height=400)
                    
                    # 매칭 통계
                    high_conf = len(matching_result[matching_result['score'] >= 0.8])
                    med_conf = len(matching_result[(matching_result['score'] >= 0.5) & (matching_result['score'] < 0.8)])
                    low_conf = len(matching_result[matching_result['score'] < 0.5])
                    
                    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
                    with stat_col1:
                        st.metric("🎯 고신뢰도", f"{high_conf}건", help="신뢰도 ≥ 0.8")
                    with stat_col2:
                        st.metric("⚖️ 중신뢰도", f"{med_conf}건", help="0.5 ≤ 신뢰도 < 0.8")
                    with stat_col3:
                        st.metric("⚠️ 저신뢰도", f"{low_conf}건", help="신뢰도 < 0.5")
                    with stat_col4:
                        matched_rids = set(matching_result['rid'])
                        unmatched = len(contract_receipts) - len(matched_rids)
                        st.metric("❓ 미매칭", f"{unmatched}건", help="매칭되지 않은 선수금")
                    
                    # 미매칭 선수금 표시
                    if unmatched > 0:
                        with st.expander(f"🔍 미매칭 선수금 보기 ({unmatched}건)"):
                            matched_rids = set(matching_result['rid'])
                            unmatched_receipts = contract_receipts_indexed[
                                ~contract_receipts_indexed.index.isin(matched_rids)
                            ].copy()
                            
                            unmatched_receipts["date"] = pd.to_datetime(unmatched_receipts["date"]).dt.strftime('%Y-%m-%d')
                            unmatched_receipts = unmatched_receipts.fillna('')
                            
                            st.dataframe(
                                unmatched_receipts[["amount", "date", "party", "note"]].rename(columns={
                                    "amount": "금액", "date": "일자", "party": "거래처", "note": "비고"
                                }),
                                use_container_width=True,
                                column_config={
                                    "금액": st.column_config.NumberColumn("금액", format="₩%.0f")
                                }
                            )
        
        with detail_tab4:
            st.write("**📊 계약 시각화 분석**")
            
            if len(contract_detail) < 2:
                st.info("ℹ️ 시각화를 위한 데이터가 부족합니다. (최소 2건 이상 필요)")
            else:
                # 차트 데이터 준비
                chart_data = contract_detail.copy()
                chart_data['date'] = pd.to_datetime(chart_data['date'])
                chart_data = chart_data.dropna(subset=['date'])
                
                if not chart_data.empty:
                    # 일자별 누적 잔액 계산
                    chart_data = chart_data.sort_values('date')
                    chart_data['signed_amount'] = chart_data.apply(
                        lambda row: row['amount'] if row['direction'] == '선수금' else -row['amount'], 
                        axis=1
                    )
                    chart_data['cumulative_balance'] = chart_data['signed_amount'].cumsum()
                    
                    # 선수금/선급금 분리 차트
                    viz_col1, viz_col2 = st.columns(2)
                    
                    with viz_col1:
                        st.subheader("📈 일자별 누적 잔액 추이")
                        balance_chart = chart_data[['date', 'cumulative_balance']].set_index('date')
                        st.line_chart(balance_chart)
                        
                    with viz_col2:
                        st.subheader("⚖️ 선수금 vs 선급금")
                        direction_summary = contract_detail.groupby('direction')['amount'].sum()
                        
                        chart_dict = direction_summary.to_dict()
                        st.bar_chart(chart_dict)
                    
                    # 월별 집계 비교
                    if len(chart_data) > 5:  # 충분한 데이터가 있을 때만
                        st.subheader("📅 월별 선수금 vs 선급금 비교")
                        
                        monthly_comparison = (chart_data
                                            .groupby([chart_data['date'].dt.to_period('M'), 'direction'])['amount']
                                            .sum()
                                            .unstack(fill_value=0))
                        
                        if not monthly_comparison.empty:
                            monthly_comparison.index = monthly_comparison.index.to_timestamp()
                            st.bar_chart(monthly_comparison)
                        
                        # 거래처별 분석
                        if len(chart_data['party'].unique()) > 1:
                            st.subheader("🏢 거래처별 금액 분석")
                            party_analysis = (chart_data.groupby(['party', 'direction'])['amount']
                                            .sum().unstack(fill_value=0))
                            
                            if not party_analysis.empty:
                                st.bar_chart(party_analysis)
                else:
                    st.info("ℹ️ 유효한 날짜 데이터가 없어 시각화를 표시할 수 없습니다.")

# 푸터
st.divider()
st.markdown("---")

footer_col1, footer_col2 = st.columns(2)
with footer_col1:
    st.markdown("""
    ### 🛠️ 기술 정보
    - **Python**: Pandas, Streamlit
    - **알고리즘**: 다중 조건 매칭, 텍스트 유사도 분석
    - **데이터 처리**: 자동 표준화, 실시간 집계
    """)

with footer_col2:
    st.markdown("""
    ### 📞 지원
    - **설치**: `pip install streamlit pandas openpyxl xlrd xlsxwriter`
    - **실행**: `streamlit run app_improved.py`
    - **문제 해결**: 로그 확인 및 데이터 검증
    """)

# 디버깅 정보 (개발자용)
if st.checkbox("🔧 개발자 디버깅 정보 표시", help="개발 및 디버깅용 상세 정보"):
    with st.expander("🔍 시스템 정보"):
        st.write("**업로드된 파일 정보:**")
        if uploaded_file:
            st.json({
                "파일명": uploaded_file.name,
                "파일크기": f"{uploaded_file.size:,} bytes",
                "파일타입": uploaded_file.type
            })
        
        st.write("**데이터 검증 정보:**")
        st.json(validation_info)
        
        if not base_data.empty:
            st.write("**데이터 샘플:**")
            st.dataframe(base_data.head(), use_container_width=True)
        
        st.write("**매칭 설정:**")
        st.json(matching_config)

# 성능 최적화 팁
with st.expander("⚡ 성능 최적화 팁"):
    st.markdown("""
    ### 🚀 성능 향상을 위한 권장사항
    
    **데이터 준비:**
    - 엑셀 파일 크기는 10MB 이하 권장
    - 불필요한 시트 제거
    - 빈 행/열 정리
    
    **매칭 설정:**
    - 금액 허용오차를 너무 크게 설정하지 말 것
    - 일자 윈도우는 필요한 만큼만 설정
    - 조합 매칭 수는 3개 이하 권장
    
    **사용법:**
    - 브라우저 캐시 정리로 메모리 확보
    - 큰 데이터는 계약별로 분리하여 분석
    - 정기적으로 브라우저 새로고침
    """)

# 도움말 및 FAQ
with st.expander("❓ 자주 묻는 질문 (FAQ)"):
    st.markdown("""
    ### 🤔 자주 묻는 질문
    
    **Q: 엑셀 파일이 업로드되지 않아요.**
    A: xlsx, xlsm, xls 형식인지 확인하고, 파일이 손상되지 않았는지 확인하세요.
    
    **Q: 컬럼을 인식하지 못해요.**
    A: 컬럼명이 한글/영문으로 명확히 되어있는지 확인하세요. '계약번호', '금액', '일자' 등의 표준 명칭 사용을 권장합니다.
    
    **Q: 매칭 결과가 이상해요.**
    A: 좌측 사이드바의 매칭 설정을 조정해보세요. 특히 금액 허용오차와 일자 윈도우를 확인하세요.
    
    **Q: 데이터가 너무 많아서 느려요.**
    A: 계약별로 파일을 분리하거나, 기간을 나누어서 분석해보세요.
    
    **Q: 차트가 표시되지 않아요.**
    A: 날짜 데이터가 올바른 형식인지 확인하고, 충분한 데이터가 있는지 확인하세요.
    """)

# 버전 정보 및 업데이트 로그
st.markdown("---")
version_col1, version_col2 = st.columns(2)

with version_col1:
    st.caption("**Version 2.0 - 완전 개선판**")
    st.caption("Last updated: 2025-08-31")

with version_col2:
    st.caption("🔄 **주요 개선사항:**")
    st.caption("• 안정성 향상 • 성능 최적화 • UI/UX 개선 • 에러 처리 강화")

# 숨겨진 고급 기능들
if st.secrets.get("debug_mode", False):  # 시크릿 설정으로 개발자 모드
    st.markdown("---")
    st.markdown("### 🛠️ 고급 개발자 도구")
    
    if st.button("🔄 캐시 초기화"):
        st.cache_data.clear()
        st.success("캐시가 초기화되었습니다.")
        st.experimental_rerun()
    
    if st.button("📊 메모리 사용량 확인"):
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        st.info(f"현재 메모리 사용량: {memory_info.rss / 1024 / 1024:.2f} MB")
    
    if not base_data.empty:
        if st.button("📁 데이터 다운로드 (CSV)"):
            csv_data = base_data.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="표준화된 데이터 다운로드",
                data=csv_data,
                file_name=f"standardized_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

# 사용자 피드백 수집
st.markdown("---")
feedback_col1, feedback_col2 = st.columns(2)

with feedback_col1:
    st.markdown("### 📝 사용자 피드백")
    user_rating = st.select_slider(
        "이 도구가 얼마나 유용했나요?",
        options=["😞 매우 불만", "😐 불만", "🙂 보통", "😊 만족", "😍 매우 만족"],
        value="🙂 보통"
    )

with feedback_col2:
    feedback_text = st.text_area(
        "개선 사항이나 의견을 알려주세요:",
        placeholder="더 나은 도구가 될 수 있도록 의견을 남겨주세요...",
        height=100
    )
    
    if st.button("📤 피드백 전송"):
        if feedback_text.strip():
            # 실제 환경에서는 데이터베이스나 이메일로 전송
            logger.info(f"사용자 피드백: {user_rating} - {feedback_text}")
            st.success("피드백이 전송되었습니다. 감사합니다! 🙏")
        else:
            st.warning("피드백 내용을 입력해주세요.")

# 최종 안내 메시지
if uploaded_file:
    st.success("✅ 모든 분석이 완료되었습니다. 위의 탭들을 통해 상세한 결과를 확인하세요!")
else:
    st.info("👈 시작하려면 좌측에서 엑셀 파일을 업로드하세요.")

# 에러 로깅 시스템 (실제 운영환경용)
try:
    # 세션 상태에 에러 카운트 저장
    if 'error_count' not in st.session_state:
        st.session_state.error_count = 0
    
    # 10분마다 에러 카운트 리셋
    if 'last_reset' not in st.session_state:
        st.session_state.last_reset = datetime.now()
    
    if (datetime.now() - st.session_state.last_reset).seconds > 600:
        st.session_state.error_count = 0
        st.session_state.last_reset = datetime.now()
        
except Exception as e:
    logger.error(f"세션 상태 관리 오류: {e}")

# 마지막 정리 작업
try:
    # 메모리 정리
    if 'base_data' in locals() and len(base_data) > 10000:
        # 대용량 데이터인 경우 메모리 최적화 수행
        import gc
        gc.collect()
        
except Exception as e:
    logger.error(f"정리 작업 중 오류: {e}")

# 스크립트 종료 메시지 (개발자용)
logger.info("Streamlit 앱 렌더링 완료")
