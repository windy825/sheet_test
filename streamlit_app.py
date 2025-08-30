# app.py
# -------------------------------------------------------------
# "선수/선급금 계약별 대시보드" — Streamlit 단일 파일 웹앱
# - 엑셀 업로드 → 계약 단위 집계 & 매칭 설정 패널 제공
# - 교차검증: 계약ID 외에도 금액/일자/거래처/품목 기반으로 선수금↔선급금 매칭
# - 실행: `pip install streamlit pandas openpyxl xlrd xlsxwriter` 후 `streamlit run app.py`
# -------------------------------------------------------------

import io
import re
import math
import typing as t
import pandas as pd
import streamlit as st

st.set_page_config(page_title="선수/선급금 계약별 대시보드", layout="wide")
st.title("📊 선수/선급금 계약별 대시보드")

# ===== 유틸 =====
_non_digit = re.compile(r"[^0-9\-\.]+")

def to_float(x):
    if x is None or (isinstance(x,float) and math.isnan(x)):
        return 0.0
    try:
        return float(_non_digit.sub("", str(x)))
    except:
        return 0.0

# ===== 데이터 로딩 =====

@st.cache_data(show_spinner=True)
def load_excel(file_bytes: bytes):
    try:
        excel = pd.ExcelFile(io.BytesIO(file_bytes), engine="openpyxl")
    except Exception:
        excel = pd.ExcelFile(io.BytesIO(file_bytes))
    return excel

# 시트 찾기
def read_sheet(excel: pd.ExcelFile, keywords):
    for name in excel.sheet_names:
        if any(k in str(name) for k in keywords):
            try:
                return pd.read_excel(excel, sheet_name=name)
            except Exception:
                return None
    return None

# 표준화
def standardize(df: pd.DataFrame, direction: str):
    if df is None: return pd.DataFrame()
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    # 최소 키 후보
    cid = next((c for c in df.columns if "계약" in c or "PJT" in c or "고유" in c or "마스터" in c), None)
    amt = next((c for c in df.columns if "금액" in c or "선수" in c or "선급" in c), None)
    date = next((c for c in df.columns if "일자" in c or "납기" in c or "청구" in c or "지급" in c), None)
    party = next((c for c in df.columns if "업체" in c or "거래" in c or "고객" in c), None)

    out = pd.DataFrame()
    out["계약ID"] = df[cid].astype(str) if cid else ""
    out["금액"] = df[amt].map(to_float) if amt else 0.0
    out["일자"] = pd.to_datetime(df[date], errors="coerce") if date else pd.NaT
    out["거래처"] = df[party].astype(str) if party else ""
    out["구분"] = direction
    return out

# ===== 사이드바 업로드 =====
with st.sidebar:
    st.header("📁 데이터 업로드")
    upl = st.file_uploader("엑셀 파일 선택", type=["xlsx","xlsm","xls"])

if upl is None:
    st.info("좌측에서 파일을 업로드하세요")
    st.stop()

excel = load_excel(upl.read())
df_s = standardize(read_sheet(excel,["선수금"]), "선수금")
df_a = standardize(read_sheet(excel,["선급금"]), "선급금")
base = pd.concat([df_s, df_a], ignore_index=True)

# ===== 매칭 설정 패널 =====
st.sidebar.header("🔧 매칭 설정")
# 계약ID를 '필터'가 아니라 '가중치'로도 사용할 수 있게 옵션 분리
use_contract_soft = st.sidebar.checkbox("계약ID 일치에 가중치 부여", value=True)
use_amount = st.sidebar.checkbox("금액 조건 사용", value=True)
amount_tol = st.sidebar.number_input("금액 허용 오차(원)", 0, 1_000_000_000, 0, step=1000)
amount_tol_pct = st.sidebar.slider("금액 허용 오차(%)", 0, 20, 1)
use_date = st.sidebar.checkbox("일자 조건 사용", value=True)
date_window = st.sidebar.slider("일자 윈도우(±일)", 0, 180, 30)
use_party_soft = st.sidebar.checkbox("거래처(텍스트) 유사도 가중치", value=True)
max_combo = st.sidebar.slider("부분합 매칭(多:1) 최대 묶음 수", 1, 5, 3)

st.sidebar.caption("💡 계약ID가 불안정해도 금액/일자/텍스트 유사도로 교차검증합니다.")

# ===== 교차검증 매칭 =====
import itertools

def simple_tokens(s: str) -> set:
    s = '' if s is None else str(s)
    s = re.sub(r'[^0-9A-Za-z가-힣\-_/]+', ' ', s)
    toks = [t for t in s.split() if len(t) >= 3]
    return set(toks)

receipts = df_s.copy().reset_index(drop=True)
advances = df_a.copy().reset_index(drop=True)
receipts['rid'] = receipts.index
advances['aid'] = advances.index

# 사전 토큰 (현재 스키마상 비고/상태 컬럼이 없으므로 거래처/계약ID만 사용)
receipts['tok'] = (receipts['거래처'].fillna('') + ' ' + receipts['계약ID'].fillna('')).apply(simple_tokens)
advances['tok'] = (advances['거래처'].fillna('') + ' ' + advances['계약ID'].fillna('')).apply(simple_tokens)

match_rows = []
for _, r in receipts.iterrows():
    r_amt = float(r['금액'] or 0)
    r_date = r['일자']

    # 1) 후보군: 날짜/금액으로 1차 축소 (너무 큰 탐색 방지)
    cand = advances.copy()
    if use_date and pd.notna(r_date):
        cand = cand[(cand['일자'].isna()) | (cand['일자'].between(r_date - pd.Timedelta(days=date_window), r_date + pd.Timedelta(days=date_window)))]
    if use_amount:
        tol = max(amount_tol, r_amt * (amount_tol_pct/100.0))
        cand = cand[(cand['금액'] >= r_amt - tol) & (cand['금액'] <= r_amt + tol)]

    # 스코어 함수
    def score_of(a_row):
        s = 0.0
        # 금액 근접도
        if use_amount and r_amt > 0:
            diff = abs(r_amt - float(a_row['금액']))
            s += max(0.0, 1.0 - diff / (r_amt + 1e-9))
        # 날짜 근접도
        if use_date and pd.notna(r_date) and pd.notna(a_row['일자']) and date_window > 0:
            s += max(0.0, 1.0 - abs((r_date - a_row['일자']).days) / (date_window + 1))
        # 계약ID 일치 가중치(소프트)
        if use_contract_soft and str(r['계약ID']) != '' and str(a_row['계약ID']) != '' and str(r['계약ID']) == str(a_row['계약ID']):
            s += 0.6
        # 거래처 토큰 유사도(소프트)
        if use_party_soft:
            inter = len(r['tok'].intersection(a_row['tok']))
            if inter > 0:
                s += min(0.4, 0.1 * inter)
        return s

    best = None
    # 2) 단건 매칭 후보
    for _, a in cand.iterrows():
        sc = score_of(a)
        if (best is None) or (sc > best['score']):
            best = {'rid': int(r['rid']), 'aids': [int(a['aid'])], 'sum_adv': float(a['금액']), 'gap': float(r_amt - a['금액']), 'score': float(sc)}

    # 3) 부분합 매칭 (多:1) — 작은 조합 우선 탐색
    if max_combo > 1 and not cand.empty and r_amt > 0:
        # 상위 N(=8)개로 후보 제한해 조합 폭발 방지
        pool = cand.copy()
        pool['amt_diff'] = (pool['금액'] - r_amt).abs()
        pool = pool.sort_values('amt_diff').head(8)
        ids = list(pool['aid'])
        for k in range(2, max_combo+1):
            for combo in itertools.combinations(ids, k):
                rows = pool.set_index('aid').loc[list(combo)]
                total = float(rows['금액'].sum())
                tol = max(amount_tol, r_amt * (amount_tol_pct/100.0)) if use_amount else 0.0
                if abs(total - r_amt) <= tol:
                    # 조합 스코어 = 평균 단건 스코어
                    sc = sum(score_of(row) for _, row in rows.iterrows()) / k
                    if (best is None) or (sc > best['score']):
                        best = {'rid': int(r['rid']), 'aids': list(map(int, combo)), 'sum_adv': float(total), 'gap': float(r_amt - total), 'score': float(sc)}

    if best:
        match_rows.append(best)

# 결과 테이블 구성
if match_rows:
    mm = pd.DataFrame(match_rows)
    r_show = receipts.copy()
    a_show = advances.copy()
    def aids_to_text(aids):
        rows = a_show.set_index('aid').loc[aids]
        return ", ".join([f"#{i}:{amt:,.0f}" for i, amt in zip(aids, rows['금액'])])
    out = mm.copy()
    out['선수금'] = out['rid'].apply(lambda i: r_show.loc[i, '금액'])
    out['선수일자'] = out['rid'].apply(lambda i: r_show.loc[i, '일자'])
    out['선수_계약ID'] = out['rid'].apply(lambda i: r_show.loc[i, '계약ID'])
    out['선급_묶음(인덱스:금액)'] = out['aids'].apply(aids_to_text)
    out['선급합계'] = out['sum_adv']
    out['차이(Gap)'] = out['gap']
    out['신뢰도'] = out['score']
    df_match = out[["선수_계약ID","선수금","선수일자","선급_묶음(인덱스:금액)","선급합계","차이(Gap)","신뢰도"]]
else:
    df_match = pd.DataFrame()

# ===== 출력 =====
st.subheader("📂 매칭 결과")
if match_results:
    df_match = pd.DataFrame(match_results)
    st.dataframe(df_match, use_container_width=True)
else:
    st.warning("설정된 조건으로 매칭 결과가 없습니다.")

st.subheader("📊 원본 데이터 미리보기")
st.write("선수금", df_s.head())
st.write("선급금", df_a.head())
