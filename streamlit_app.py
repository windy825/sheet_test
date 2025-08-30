# app.py
# -------------------------------------------------------------
# 선수/선급금 계약별 대시보드 (교차검증 매칭 포함) — Streamlit 단일 파일
# - 엑셀 업로드 → 표준화 → 계약별 집계 → 상세/차트 → 자동 매칭(계약ID 의존 X, 가중치·부분합 지원)
# - requirements.txt 없이도 동작(미설치시 설치 명령 안내)
# 실행: 1) pip install streamlit pandas openpyxl xlrd xlsxwriter  2) streamlit run app.py
# -------------------------------------------------------------

import io
import re
import math
import typing as t
from datetime import datetime

import pandas as pd
import streamlit as st

# ====== 기본 설정 ======
st.set_page_config(page_title="선수/선급금 계약별 대시보드", layout="wide")
st.title("📊 선수/선급금 계약별 대시보드 (교차검증 매칭)")
st.caption("엑셀을 업로드하면 계약 기준으로 선수/선급을 집계하고, 교차검증 매칭으로 선수↔선급을 연결합니다.")

# ====== 유틸: 숫자 변환 ======
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
    df = df.copy()
    new_cols, seen = [], set()
    for c in df.columns:
        nc = str(c).strip()
        nc = re.sub(r"\s+", " ", nc)
        if nc in seen:
            k, base = 2, nc
            while nc in seen:
                nc = f"{base}_{k}"; k += 1
        seen.add(nc); new_cols.append(nc)
    df.columns = new_cols
    return df

# ====== 후보 컬럼 ======
STANDARD_COLS = [
    "contract_id", "direction", "amount", "date", "party", "owner", "status", "note", "overdue_flag"
]
CONTRACT_CANDIDATES = ["계약번호", "금형마스터", "프로젝트", "프로젝트코드", "PJT", "PJT코드", "고유번호", "계약코드", "계약id"]
AMOUNT_CANDIDATES  = ["금액", "선수금", "선급금", "선수금금액", "선급금금액", "합계", "잔액"]
DATE_CANDIDATES    = ["일자", "청구일", "지급일", "납기일", "요청일", "등록일", "기준일", "date"]
PARTY_CANDIDATES   = ["업체명", "거래처", "고객사", "고객명", "상대방", "회사", "vendor", "customer"]
OWNER_CANDIDATES   = ["담당자", "담당", "담당자명", "PM", "담당부서", "owner"]
STATUS_CANDIDATES  = ["진행현황", "정산여부", "상태", "status"]
NOTE_CANDIDATES    = ["비고", "메모", "특이사항", "코멘트", "note"]
OD_CANDIDATES      = ["기한경과", "연체", "overdue", "경과"]

# ====== 헬퍼 ======

def first_col(df: pd.DataFrame, candidates: t.List[str]) -> t.Optional[str]:
    cols = list(df.columns)
    lower_map = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand in df.columns:
            return cand
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    for cand in candidates:
        pat = re.compile(re.escape(cand), re.IGNORECASE)
        for c in cols:
            if pat.search(str(c)):
                return c
    return None

# ====== 시트 읽기 ======

def read_sheet_by_keywords(excel: pd.ExcelFile, keywords: t.List[str]) -> pd.DataFrame:
    def _norm(s: str) -> str:
        return re.sub(r"\s+", "", s.lower())
    target = None
    for name in excel.sheet_names:
        nm = _norm(name)
        if any(_norm(k) in nm for k in keywords):
            target = name; break
    if target is None:
        return pd.DataFrame()
    try:
        df = pd.read_excel(excel, sheet_name=target, dtype=str)
        return normalize_columns(df)
    except Exception:
        return pd.DataFrame()

# ====== 표준화 ======

def standardize(df: pd.DataFrame, direction: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=STANDARD_COLS)
    df = normalize_columns(df)

    c_contract = first_col(df, CONTRACT_CANDIDATES)
    if c_contract is None:
        for col in df.columns:
            if any(k in str(col) for k in ["PJT", "프로젝트", "금형", "계약"]):
                c_contract = col; break

    c_amount   = first_col(df, AMOUNT_CANDIDATES)
    c_date     = first_col(df, DATE_CANDIDATES)
    c_party    = first_col(df, PARTY_CANDIDATES)
    c_owner    = first_col(df, OWNER_CANDIDATES)
    c_status   = first_col(df, STATUS_CANDIDATES)
    c_note     = first_col(df, NOTE_CANDIDATES)
    c_overdue  = first_col(df, OD_CANDIDATES)

    data = pd.DataFrame()
    data["contract_id"] = df[c_contract].astype(str).str.strip() if c_contract in df.columns else "(미지정)"
    data["direction"] = direction
    data["amount"] = df[c_amount].apply(to_float) if c_amount in df.columns else 0.0
    data["date"] = pd.to_datetime(df[c_date], errors="coerce") if c_date in df.columns else pd.NaT
    data["party"] = df[c_party].astype(str).str.strip() if c_party in df.columns else ""
    data["owner"] = df[c_owner].astype(str).str.strip() if c_owner in df.columns else ""
    data["status"] = df[c_status].astype(str).str.strip() if c_status in df.columns else ""
    data["note"] = df[c_note].astype(str).str.strip() if c_note in df.columns else ""

    if c_overdue in df.columns:
        od = df[c_overdue].astype(str).str.strip().str.lower()
        data["overdue_flag"] = od.isin(["y", "yes", "true", "1", "o"]) | od.str.contains("경과|연체|over", na=False)
    else:
        data["overdue_flag"] = False

    data = data[(data["amount"] != 0) & (data["contract_id"].astype(str).str.strip() != "")]
    return data[STANDARD_COLS]

# ====== 캐시: DataFrame만 반환 ======

@st.cache_data(show_spinner=True)
def load_data(file_bytes: bytes):
    try:
        import openpyxl  # noqa: F401
    except ImportError:
        st.error("`openpyxl`이 없습니다. 아래 설치 명령을 실행하세요: `pip install openpyxl`.")
        # 빈 DF 4개 반환(앱은 계속 동작)
        empty = pd.DataFrame(columns=STANDARD_COLS)
        return empty, pd.DataFrame(), empty, empty

    try:
        excel = pd.ExcelFile(io.BytesIO(file_bytes), engine="openpyxl")
    except Exception as e:
        st.error(f"엑셀 파일을 열 수 없습니다: {e}")
        empty = pd.DataFrame(columns=STANDARD_COLS)
        return empty, pd.DataFrame(), empty, empty

    df_receipts_raw = read_sheet_by_keywords(excel, ["선수금"])  # 고객에서 받은
    df_advances_raw = read_sheet_by_keywords(excel, ["선급금"])  # 협력사에 준

    df_receipts = standardize(df_receipts_raw, "선수금")
    df_advances = standardize(df_advances_raw, "선급금")

    base = pd.concat([df_receipts, df_advances], ignore_index=True)

    if base.empty:
        table = pd.DataFrame(columns=["계약ID","선수금_합계","선급금_합계","Gap(선수-선급)","담당자","주요거래처","진행현황"]) 
        return base, table, df_receipts, df_advances

    agg = base.groupby("contract_id").agg(
        선수금_합계=(lambda x: base.loc[x.index][base.loc[x.index, "direction"]=="선수금", "amount"].sum()),
        선급금_합계=(lambda x: base.loc[x.index][base.loc[x.index, "direction"]=="선급금", "amount"].sum()),
    ).fillna(0.0)
    agg["Gap(선수-선급)"] = agg["선수금_합계"] - agg["선급금_합계"]

    meta_cols = ["owner", "party", "status"]
    meta = base.groupby("contract_id")[meta_cols].agg(lambda s: s.dropna().astype(str).replace({"", "nan", "None"}, pd.NA).dropna().unique()[:1])
    for c in meta_cols:
        meta[c] = meta[c].apply(lambda arr: arr[0] if isinstance(arr, (list, tuple, pd.Series)) and len(arr)>0 else "")

    table = agg.join(meta, how="left").reset_index().rename(columns={
        "contract_id":"계약ID", "owner":"담당자", "party":"주요거래처", "status":"진행현황"
    })

    return base, table, df_receipts, df_advances

# ====== 사이드바: 업로드 + 매칭 설정 ======
with st.sidebar:
    st.header("📁 데이터 업로드")
    upl = st.file_uploader("엑셀 파일 (xlsx/xlsm/xls)", type=["xlsx","xlsm","xls"]) 
    st.markdown("— 매크로(xlsm)는 **값만 읽습니다** (매크로 실행 없음)")

    st.header("🔧 매칭 설정")
    st.caption("계약ID에 덜 의존하고 금액/일자/텍스트로 교차검증합니다.")
    use_contract_soft = st.checkbox("계약ID 일치 가중치 적용", value=True)
    use_amount = st.checkbox("금액 조건 사용", value=True)
    amount_tol = st.number_input("금액 허용 오차(원)", min_value=0, value=0, step=1000, format="%d")
    amount_tol_pct = st.slider("금액 허용 오차(%)", 0, 20, 1)
    use_date = st.checkbox("일자 조건 사용", value=True)
    date_window = st.slider("일자 윈도우(±일)", 0, 180, 30)
    use_party_soft = st.checkbox("거래처/텍스트 유사도 가중치", value=True)
    max_combo = st.slider("부분합 매칭 최대 묶음 수(多:1)", 1, 5, 3)

st.markdown("---")

if upl is None:
    st.info("좌측에서 엑셀 파일을 업로드하세요. 아래 패키지가 없으면 설치하세요:")
    st.code("pip install streamlit pandas openpyxl xlrd xlsxwriter", language="bash")
    st.stop()

base, table, receipts, advances = load_data(upl.read())

# ====== KPI ======
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

# ====== 필터 & 집계 테이블 ======
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
st.dataframe(view, use_container_width=True, height=420)

# ====== 매칭 엔진 ======
import itertools

def simple_tokens(s: str) -> set:
    s = '' if s is None else str(s)
    s = re.sub(r'[^0-9A-Za-z가-힣\-_/]+', ' ', s)
    toks = [t for t in s.split() if len(t) >= 3]
    return set(toks)


def compute_matches(receipts_df: pd.DataFrame, advances_df: pd.DataFrame,
                    date_window: int, tol_abs: int, tol_pct: int,
                    use_contract_soft: bool, use_party_soft: bool,
                    use_amount: bool, use_date: bool, max_combo: int) -> pd.DataFrame:
    rec = receipts_df.reset_index(drop=True).copy()
    adv = advances_df.reset_index(drop=True).copy()
    if rec.empty or adv.empty:
        return pd.DataFrame(columns=['rid','aids','sum_adv','gap','score'])

    rec['rid'] = rec.index
    adv['aid'] = adv.index

    rec['tok'] = (rec['note'].fillna('') + ' ' + rec['status'].fillna('') + ' ' + rec['party'].fillna('') + ' ' + rec['contract_id'].fillna('')).apply(simple_tokens)
    adv['tok'] = (adv['note'].fillna('') + ' ' + adv['status'].fillna('') + ' ' + adv['party'].fillna('') + ' ' + adv['contract_id'].fillna('')).apply(simple_tokens)

    matches = []
    for _, r in rec.iterrows():
        rdate = r.get('date', pd.NaT)
        ramt = float(r.get('amount', 0) or 0)

        cand = adv.copy()
        if use_date and pd.notna(rdate):
            cand = cand[(cand['date'].isna()) | (cand['date'].between(rdate - pd.Timedelta(days=date_window), rdate + pd.Timedelta(days=date_window)))]

        if use_amount:
            tol = max(tol_abs, ramt * tol_pct / 100.0)
            cand = cand[(cand['amount'] >= ramt - tol) & (cand['amount'] <= ramt + tol)]

        best = None
        # 1) 단건 매칭
        for _, a in cand.iterrows():
            aamt = float(a.get('amount', 0) or 0)
            amt_diff = abs(ramt - aamt)
            pct_ok = (ramt == 0) or (amt_diff <= (ramt * tol_pct / 100.0))
            if (not use_amount) or amt_diff <= tol_abs or pct_ok:
                score = 0.0
                # 금액 근접
                if use_amount and ramt > 0:
                    score += max(0.0, 1.0 - (amt_diff / (ramt + 1e-9)))
                # 날짜 근접
                if use_date and pd.notna(rdate) and pd.notna(a['date']) and date_window > 0:
                    score += max(0.0, 1 - (abs((rdate - a['date']).days) / (date_window + 1)))
                # 계약ID 소프트 가중치
                if use_contract_soft and str(r['contract_id']) != '' and str(a['contract_id']) != '' and str(r['contract_id']) == str(a['contract_id']):
                    score += 0.6
                # 텍스트 유사도(거래처/비고/상태 등)
                if use_party_soft:
                    inter = len(r['tok'].intersection(a['tok']))
                    if inter > 0:
                        score += min(0.4, 0.1 * inter)

                if (best is None) or (score > best['score']):
                    best = {'rid': int(r['rid']), 'aids': [int(a['aid'])], 'sum_adv': float(aamt), 'gap': float(ramt - aamt), 'score': float(score)}

        # 2) 부분합 매칭 (여러 선급 → 한 선수)
        if (best is None) and max_combo > 1 and not cand.empty and ramt > 0:
            tol = max(tol_abs, ramt * tol_pct / 100.0) if use_amount else 0.0
            pool = cand.copy()
            if use_amount:
                pool['amt_diff'] = (pool['amount'] - ramt).abs()
                pool = pool.sort_values('amt_diff').head(10)
            ids = list(pool['aid'])
            for k in range(2, max_combo+1):
                for combo in itertools.combinations(ids, k):
                    rows = pool.set_index('aid').loc[list(combo)]
                    total = float(rows['amount'].sum())
                    if abs(total - ramt) <= tol:
                        # 조합 스코어 = 평균 단건 스코어
                        sc = 0.0
                        for _, row in rows.iterrows():
                            aamt = float(row['amount'])
                            s = 0.0
                            if use_amount and ramt > 0:
                                s += max(0.0, 1.0 - abs(ramt - aamt) / (ramt + 1e-9))
                            if use_date and pd.notna(rdate) and pd.notna(row['date']) and date_window > 0:
                                s += max(0.0, 1 - (abs((rdate - row['date']).days) / (date_window + 1)))
                            if use_contract_soft and str(r['contract_id']) == str(row['contract_id']) and str(r['contract_id']) != '':
                                s += 0.6
                            if use_party_soft:
                                inter = len(r['tok'].intersection(row['tok']))
                                if inter > 0:
                                    s += min(0.4, 0.1 * inter)
                            sc += s
                        sc = sc / k
                        best = {'rid': int(r['rid']), 'aids': list(map(int, combo)), 'sum_adv': float(total), 'gap': float(ramt - total), 'score': float(sc)}
                        break
                if best is not None:
                    break

        if best:
            matches.append(best)

    if not matches:
        return pd.DataFrame(columns=['rid','aids','sum_adv','gap','score'])
    return pd.DataFrame(matches)

# ====== 계약 상세 ======
st.subheader("🔎 계약 상세 보기")
contract_ids = view["계약ID"].tolist()
sel = st.selectbox("계약을 선택하세요", ["(미선택)"] + contract_ids, index=0)

if sel and sel != "(미선택)":
    detail = base[base["contract_id"]==sel].copy()

    r_sum = detail.loc[detail["direction"]=="선수금", "amount"].sum()
    a_sum = detail.loc[detail["direction"]=="선급금", "amount"].sum()
    gap = r_sum - a_sum

    c1, c2, c3 = st.columns(3)
    with c1: st.metric("선수금(받은 돈)", f"{r_sum:,.0f} 원")
    with c2: st.metric("선급금(먼저 준 돈)", f"{a_sum:,.0f} 원")
    with c3: st.metric("Gap(선수-선급)", f"{gap:,.0f} 원")

    with st.expander("ℹ️ 개념 도움말 (초보자용)", expanded=False):
        st.markdown(
            """
            - **선수금**: 고객에게서 **미리 받은 돈** (나중에 납품/청구 시 매출에서 차감)
            - **선급금**: 협력사에 **미리 준 돈** (나중에 대금 지급 시 차감)
            - **Gap(선수-선급)**: 받은 돈에서 먼저 준 돈을 뺀 금액 → **플러스**면 유리, **마이너스**면 불리
            """
        )

    t1, t2, t3 = st.tabs(["선수금 상세", "선급금 상세", "자동 매칭(교차검증)"])

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
            df_a_plot = detail[detail["direction"]=="선급금"][['date','amount']].dropna().copy()
            if not df_a_plot.empty:
                df_a_plot['month'] = pd.to_datetime(df_a_plot['date']).dt.to_period('M').dt.to_timestamp()
                st.bar_chart(df_a_plot.groupby('month')['amount'].sum())

    with t3:
        r_sel = detail[detail["direction"]=="선수금"][STANDARD_COLS].copy()
        a_all = base[base["direction"]=="선급금"][STANDARD_COLS].copy()
        if r_sel.empty or a_all.empty:
            st.info("매칭할 데이터가 부족합니다. (선수금/선급금 모두 필요)")
        else:
            mm = compute_matches(r_sel, a_all, date_window, amount_tol, amount_tol_pct,
                                 use_contract_soft, use_party_soft, use_amount, use_date, max_combo)
            if mm.empty:
                st.warning("현재 설정으로 자동 매칭 결과가 없습니다. 윈도우/허용오차를 넓혀 보세요.")
            else:
                r_show = r_sel.reset_index(drop=True)
                a_show = a_all.reset_index(drop=True)
                def aid_to_str(aids):
                    rows = a_show.loc[aids]
                    return ", ".join([f"#{i}: {amt:,.0f}" for i, amt in zip(aids, rows['amount'])])
                view_mm = mm.copy()
                view_mm['선수금'] = view_mm['rid'].apply(lambda i: r_show.loc[i, 'amount'])
                view_mm['선수금_일자'] = view_mm['rid'].apply(lambda i: r_show.loc[i, 'date'])
                view_mm['선수금_비고'] = view_mm['rid'].apply(lambda i: r_show.loc[i, 'note'])
                view_mm['선급금_묶음(인덱스:금액)'] = view_mm['aids'].apply(aid_to_str)
                view_mm = view_mm.rename(columns={'sum_adv':'선급금_합계','gap':'차이(Gap)','score':'신뢰도'})
                st.dataframe(view_mm[["선수금","선수금_일자","선수금_비고","선급금_묶음(인덱스:금액)","선급금_합계","차이(Gap)","신뢰도"]], use_container_width=True)

                with st.expander("미매칭 항목 보기"):
                    matched_rids = set(view_mm['rid'].tolist())
                    unmatched_r = r_show.loc[~r_show.index.isin(matched_rids)]
                    st.markdown("**미매칭 선수금**")
                    st.dataframe(unmatched_r, use_container_width=True)

    # 다운로드
    def to_excel_bytes(df_dict: t.Dict[str, pd.DataFrame]) -> bytes:
        bio = io.BytesIO()
        with pd.ExcelWriter(bio, engine="xlsxwriter") as writer:
            for name, d in df_dict.items():
                d.to_excel(writer, sheet_name=name, index=False)
        bio.seek(0)
        return bio.read()

    st.download_button(
        "⬇️ 현재 계약 상세 다운로드 (Excel)",
        data=to_excel_bytes({
            "선수금": df_r if 'df_r' in locals() and not df_r.empty else pd.DataFrame(columns=STANDARD_COLS),
            "선급금": df_a if 'df_a' in locals() and not df_a.empty else pd.DataFrame(columns=STANDARD_COLS),
        }),
        file_name=f"contract_{sel}_detail.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

st.markdown("---")
st.markdown("설치가 필요하면 아래 명령을 실행하세요:")
st.code("pip install streamlit pandas openpyxl xlrd xlsxwriter", language="bash")
