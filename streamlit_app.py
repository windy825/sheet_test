# app.py
# -------------------------------------------------------------
# "선수/선급금 계약별 대시보드" — Streamlit 단일 파일 웹앱
# - 엑셀(xlsx/xlsm/xls) 업로드 → 계약 단위로 선수금/선급금 집계 → 클릭 시 상세 확장
# - 한국어 UI, 불완전한 컬럼명/시트명도 최대한 자동 인식
# - 실행: 1) `pip install -r requirements.txt`  2) `streamlit run app.py`
# -------------------------------------------------------------

import io
import re
import sys
import math
import json
import typing as t
from datetime import datetime

import pandas as pd
import streamlit as st

# ====== 기본 설정 ======
st.set_page_config(page_title="선수/선급금 계약별 대시보드", layout="wide")
st.title("📊 선수/선급금 계약별 대시보드")
st.caption("엑셀 데이터를 업로드하면 계약별로 선수금/선급금을 집계하고, 계약을 선택하면 상세가 확장됩니다.")

# ====== 유틸: 안전한 숫자 변환 ======
_non_digit = re.compile(r"[^0-9\-\.]+")

def to_float(x: t.Any) -> float:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return 0.0
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return 0.0
    s = _non_digit.sub("", s)
    if s in {"", "-", "."}:
        return 0.0
    try:
        return float(s)
    except Exception:
        return 0.0

# ====== 유틸: 컬럼 정규화 ======

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # 원본 보존
    df = df.copy()
    # 중복 컬럼 방지 & 문자열 컬럼 통일
    new_cols = []
    seen = set()
    for c in df.columns:
        nc = str(c).strip()
        nc = re.sub(r"\s+", " ", nc)
        if nc in seen:
            # 동일 컬럼명 중복 시 접미어 부여
            k = 2
            base = nc
            while nc in seen:
                nc = f"{base}_{k}"
                k += 1
        seen.add(nc)
        new_cols.append(nc)
    df.columns = new_cols
    return df

# ====== 유틸: 후보 리스트 중 실제 존재 컬럼 찾기 ======

def first_col(df: pd.DataFrame, candidates: t.List[str]) -> t.Optional[str]:
    cols = list(df.columns)
    lower_map = {c.lower(): c for c in cols}
    # 정확 일치 먼저
    for cand in candidates:
        if cand in df.columns:
            return cand
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    # 부분 일치(정규식 대략 매칭)
    joined = "\n".join(cols)
    for cand in candidates:
        pattern = re.compile(re.escape(cand), re.IGNORECASE)
        for c in cols:
            if pattern.search(str(c)):
                return c
    return None

# ====== 시트 읽기 ======

def read_sheet_by_keywords(excel: pd.ExcelFile, keywords: t.List[str]) -> t.Optional[pd.DataFrame]:
    def _norm(s: str) -> str:
        return re.sub(r"\s+", "", s.lower())

    target = None
    for name in excel.sheet_names:
        nm = _norm(name)
        if any(_norm(k) in nm for k in keywords):
            target = name
            break
    if target is None:
        return None
    try:
        df = pd.read_excel(excel, sheet_name=target, dtype=str)
        return normalize_columns(df)
    except Exception:
        return None

# ====== 선수/선급금 표준화 ======

STANDARD_COLS = [
    "contract_id",  # 계약 식별자 (계약번호/금형마스터/PJT/프로젝트코드 등)
    "direction",    # '선수금' 또는 '선급금'
    "amount",       # 금액(양수)
    "date",         # 일자/청구일/지급일 등
    "party",        # 업체/고객/거래처(상대방 명칭)
    "owner",        # 담당자
    "status",       # 진행현황/정산상태
    "note",         # 비고/메모
    "overdue_flag", # 기한경과 여부(Y/N/True/False)
]

CONTRACT_CANDIDATES = ["계약번호", "금형마스터", "프로젝트", "프로젝트코드", "PJT", "PJT코드", "고유번호", "계약코드", "계약id"]
AMOUNT_CANDIDATES  = ["금액", "선수금", "선급금", "선수금금액", "선급금금액", "합계", "잔액"]
DATE_CANDIDATES    = ["일자", "청구일", "지급일", "납기일", "요청일", "등록일", "기준일", "date"]
PARTY_CANDIDATES   = ["업체명", "거래처", "고객사", "고객명", "상대방", "회사", "vendor", "customer"]
OWNER_CANDIDATES   = ["담당자", "담당", "담당자명", "PM", "담당부서", "owner"]
STATUS_CANDIDATES  = ["진행현황", "정산여부", "상태", "status"]
NOTE_CANDIDATES    = ["비고", "메모", "특이사항", "코멘트", "note"]
OD_CANDIDATES      = ["기한경과", "연체", "overdue", "경과"]


def standardize(df: pd.DataFrame, direction: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=STANDARD_COLS)

    df = normalize_columns(df)

    c_contract = first_col(df, CONTRACT_CANDIDATES)
    c_amount   = first_col(df, AMOUNT_CANDIDATES)
    c_date     = first_col(df, DATE_CANDIDATES)
    c_party    = first_col(df, PARTY_CANDIDATES)
    c_owner    = first_col(df, OWNER_CANDIDATES)
    c_status   = first_col(df, STATUS_CANDIDATES)
    c_note     = first_col(df, NOTE_CANDIDATES)
    c_overdue  = first_col(df, OD_CANDIDATES)

    # 일부 시트에 계약 식별자가 없는 경우(예: 요약) → 스킵
    if c_contract is None and direction in ("선수금", "선급금"):
        # 간혹 '계약번호' 대신 '프로젝트명(코드)' 같은 합쳐진 컬럼이 있을 수 있어 임시 대체
        for col in df.columns:
            if any(k in str(col) for k in ["PJT", "프로젝트", "금형", "계약"]):
                c_contract = col
                break

    data = pd.DataFrame()
    data["contract_id"] = df[c_contract].astype(str).str.strip() if c_contract in df.columns else "(미지정)"
    data["direction"] = direction
    data["amount"] = df[c_amount].apply(to_float) if c_amount in df.columns else 0.0

    if c_date in df.columns:
        data["date"] = pd.to_datetime(df[c_date], errors="coerce")
    else:
        data["date"] = pd.NaT

    data["party"] = df[c_party].astype(str).str.strip() if c_party in df.columns else ""
    data["owner"] = df[c_owner].astype(str).str.strip() if c_owner in df.columns else ""
    data["status"] = df[c_status].astype(str).str.strip() if c_status in df.columns else ""
    data["note"] = df[c_note].astype(str).str.strip() if c_note in df.columns else ""

    if c_overdue in df.columns:
        od = df[c_overdue].astype(str).str.strip().str.lower()
        data["overdue_flag"] = od.isin(["y", "yes", "true", "1", "o", "경과", "연체", "x", "기한경과", "있음"]) | od.str.contains("경과|연체|over", na=False)
    else:
        data["overdue_flag"] = False

    # 금액이 0이거나 contract_id 비어있으면 제거 (노이즈 방지)
    data = data[(data["amount"] != 0) & (data["contract_id"].astype(str).str.strip() != "")]
    return data[STANDARD_COLS]

# ====== 엑셀 업로드 ======

with st.sidebar:
    st.header("📁 데이터 업로드")
    upl = st.file_uploader("엑셀 파일 선택 (xlsx/xlsm/xls)", type=["xlsx", "xlsm", "xls"])   
    st.markdown("— 매크로(xlsm)는 **값만 읽어옵니다** (매크로 실행 없음)")

@st.cache_data(show_spinner=True)
def load_data(file_bytes: bytes) -> pd.DataFrame:
    excel = pd.ExcelFile(io.BytesIO(file_bytes))

    # 후보 키워드로 시트 탐색
    df_receipts_raw = read_sheet_by_keywords(excel, ["선수금"])  # 고객에게서 받은 돈
    df_advances_raw = read_sheet_by_keywords(excel, ["선급금"])  # 협력사에 준 돈

    # 표준화
    df_receipts = standardize(df_receipts_raw, "선수금")
    df_advances = standardize(df_advances_raw, "선급금")

    # 합치기
    base = pd.concat([df_receipts, df_advances], ignore_index=True)

    # 계약별 집계
    agg = base.groupby("contract_id").agg(
        선수금_합계=(lambda x: base.loc[x.index][base.loc[x.index, "direction"]=="선수금", "amount"].sum()),
        선급금_합계=(lambda x: base.loc[x.index][base.loc[x.index, "direction"]=="선급금", "amount"].sum()),
    )

    agg = agg.fillna(0.0)
    agg["Gap(선수-선급)"] = agg["선수금_합계"] - agg["선급금_합계"]

    # 대표 담당자/거래처 등 메타 (첫 값 사용)
    meta_cols = ["owner", "party", "status"]
    meta = base.groupby("contract_id")[meta_cols].agg(lambda s: s.dropna().astype(str).replace({"", "nan", "None"}, pd.NA).dropna().unique()[:1])
    for c in meta_cols:
        meta[c] = meta[c].apply(lambda arr: arr[0] if isinstance(arr, (list, tuple, pd.Series)) and len(arr)>0 else "")

    table = agg.join(meta, how="left").reset_index().rename(columns={"contract_id": "계약ID", "owner": "담당자", "party": "주요거래처", "status": "진행현황"})

    # 상세 원장도 함께 반환
    return base, table

# ====== 데이터 로딩 & 기본 요약 ======

if upl is None:
    st.info("좌측에서 엑셀 파일을 업로드하세요. 예: '선수금', '선급금' 시트가 포함된 파일")
    st.stop()

base, table = load_data(upl.read())

# 상단 KPI
col1, col2, col3, col4 = st.columns(4)
with col1:
    total_receipts = base.loc[base["direction"]=="선수금", "amount"].sum()
    st.metric("총 선수금", f"{total_receipts:,.0f} 원")
with col2:
    total_advances = base.loc[base["direction"]=="선급금", "amount"].sum()
    st.metric("총 선급금", f"{total_advances:,.0f} 원")
with col3:
    st.metric("Gap(선수-선급)", f"{(total_receipts-total_advances):,.0f} 원")
with col4:
    st.metric("계약 수", f"{table.shape[0]:,}")

st.divider()

# ====== 필터 & 테이블 ======

fc1, fc2, fc3 = st.columns([2,2,1])
with fc1:
    q = st.text_input("계약ID/거래처/담당자 검색", "")
with fc2:
    owner_filter = st.text_input("담당자 필터 (쉼표로 여러명)", "")
with fc3:
    sort_opt = st.selectbox("정렬", ["Gap(선수-선급)", "선수금_합계", "선급금_합계", "계약ID"], index=0)

view = table.copy()
if q:
    ql = q.strip().lower()
    view = view[view.apply(lambda r: ql in str(r["계약ID"]).lower() or ql in str(r["주요거래처"]).lower() or ql in str(r["담당자"]).lower(), axis=1)]

if owner_filter:
    owners = [o.strip().lower() for o in owner_filter.split(',') if o.strip()]
    if owners:
        view = view[view["담당자"].str.lower().isin(owners)]

view = view.sort_values(by=sort_opt, ascending=False)

st.subheader("📂 계약별 집계")
st.dataframe(view, use_container_width=True, height=400)

# ====== 계약 선택 → 상세 확장 ======

st.subheader("🔎 계약 상세 보기")
contract_ids = view["계약ID"].tolist()
sel = st.selectbox("계약을 선택하세요", ["(미선택)"] + contract_ids, index=0)

if sel and sel != "(미선택)":
    detail = base[base["contract_id"]==sel].copy()

    r_sum = detail.loc[detail["direction"]=="선수금", "amount"].sum()
    a_sum = detail.loc[detail["direction"]=="선급금", "amount"].sum()
    gap = r_sum - a_sum

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("선수금(받은 돈)", f"{r_sum:,.0f} 원")
    with c2:
        st.metric("선급금(먼저 준 돈)", f"{a_sum:,.0f} 원")
    with c3:
        st.metric("Gap(선수-선급)", f"{gap:,.0f} 원")

    with st.expander("ℹ️ 개념 도움말 (초보자용)", expanded=False):
        st.markdown(
            """
            - **선수금**: 고객에게서 **미리 받은 돈** (나중에 납품/청구 시 매출에서 차감)
            - **선급금**: 협력사에 **미리 준 돈** (나중에 납품 대금 지급 시 차감)
            - **Gap(선수-선급)**: 받은 돈에서 먼저 준 돈을 뺀 금액 → **플러스**면 유리, **마이너스**면 불리
            """
        )

    # 탭: 선수금 / 선급금 상세
    t1, t2 = st.tabs(["선수금 상세", "선급금 상세"])

    with t1:
        df_r = detail[detail["direction"]=="선수금"][STANDARD_COLS].copy()
        if df_r.empty:
            st.info("해당 계약에 선수금 데이터가 없습니다.")
        else:
            df_r["date"] = pd.to_datetime(df_r["date"], errors="coerce").dt.date
            st.dataframe(df_r.rename(columns={
                "contract_id":"계약ID", "direction":"구분", "amount":"금액", "date":"일자",
                "party":"거래처/고객", "owner":"담당자", "status":"진행현황", "note":"비고", "overdue_flag":"기한경과"
            }), use_container_width=True)

            # 월별 추이
            df_r_plot = detail[detail["direction"]=="선수금"][['date','amount']].dropna().copy()
            if not df_r_plot.empty:
                df_r_plot['month'] = pd.to_datetime(df_r_plot['date']).dt.to_period('M').dt.to_timestamp()
                st.bar_chart(df_r_plot.groupby('month')['amount'].sum())

    with t2:
        df_a = detail[detail["direction"]=="선급금"][STANDARD_COLS].copy()
        if df_a.empty:
            st.info("해당 계약에 선급금 데이터가 없습니다.")
        else:
            df_a["date"] = pd.to_datetime(df_a["date"], errors="coerce").dt.date
            st.dataframe(df_a.rename(columns={
                "contract_id":"계약ID", "direction":"구분", "amount":"금액", "date":"일자",
                "party":"업체/협력사", "owner":"담당자", "status":"진행현황", "note":"비고", "overdue_flag":"기한경과"
            }), use_container_width=True)

            # 월별 추이
            df_a_plot = detail[detail["direction"]=="선급금"][['date','amount']].dropna().copy()
            if not df_a_plot.empty:
                df_a_plot['month'] = pd.to_datetime(df_a_plot['date']).dt.to_period('M').dt.to_timestamp()
                st.bar_chart(df_a_plot.groupby('month')['amount'].sum())

    # 다운로드 옵션 (계약별 상세)
    def to_excel_bytes(df_dict: t.Dict[str, pd.DataFrame]) -> bytes:
        bio = io.BytesIO()
        with pd.ExcelWriter(bio, engine="xlsxwriter") as writer:
            for name, d in df_dict.items():
                d.to_excel(writer, sheet_name=name, index=False)
        bio.seek(0)
        return bio.read()

    dl = st.download_button(
        "⬇️ 현재 계약 상세 다운로드 (Excel)",
        data=to_excel_bytes({
            "선수금": df_r if 'df_r' in locals() and not df_r.empty else pd.DataFrame(columns=STANDARD_COLS),
            "선급금": df_a if 'df_a' in locals() and not df_a.empty else pd.DataFrame(columns=STANDARD_COLS),
        }),
        file_name=f"contract_{sel}_detail.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

st.divider()

with st.expander("⚙️ 사용 팁 / 설정"):
    st.markdown(
        """
        **시트/컬럼 자동 인식 규칙**
        - 시트명에 `선수금`/`선급금` 텍스트가 포함되어 있으면 각 상세로 인식합니다.
        - 계약 식별자는 아래 후보 중 첫 번째 일치 컬럼을 사용합니다.
            - `계약번호`, `금형마스터`, `프로젝트`, `프로젝트코드`, `PJT`, `PJT코드`, `고유번호`, `계약코드`, `계약id`
        - 금액 컬럼 후보: `금액`, `선수금`, `선급금`, `선수금금액`, `선급금금액`, `합계`, `잔액`
        - 일자 후보: `일자`, `청구일`, `지급일`, `납기일`, `요청일`, `등록일`, `기준일`, `date`
        - 거래처/업체 후보: `업체명`, `거래처`, `고객사`, `고객명`, `상대방`, `회사`, `vendor`, `customer`
        - 담당자 후보: `담당자`, `담당`, `담당자명`, `PM`, `담당부서`, `owner`
        - 상태/비고/기한경과도 유사 텍스트를 자동 매핑합니다.

        **Gap 계산**
        - Gap = 총 선수금 − 총 선급금 (양수면 유리, 음수면 불리)

        **참고**
        - 매크로(XLSM)는 실행하지 않으며, 시트의 **보이는 값만** 읽습니다.
        - 데이터 형식이 매우 불규칙하면 일부 항목은 공란으로 표시될 수 있습니다.
        - 필요 시 컬럼명을 위 후보 중 하나로 맞추면 인식률이 높아집니다.
        """
    )

# ====== requirements.txt (참고) ======
st.code("""\nstreamlit>=1.36.0\npandas>=2.2.0\nopenpyxl>=3.1.2\nxlsxwriter>=3.2.0\n""", language="text")
