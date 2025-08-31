
# -*- coding: utf-8 -*-
"""
streamlit_app.py
영업 담당자 관점 대시보드 + 매칭/검색 심화

개선 사항
- 👤 영업 대시보드: 현재 시점 기준으로 처리된/처리할/연체/당월예정 KPI 및 차트
- 🔍 고급 검색: 전역 키워드, 다중 필터(담당자/업체/계약/기간/금액/상태), 결과 내보내기
- 매칭 탭: 빈 데이터/빈 후보에서도 안전하게 동작
- 공통 전처리: 컬럼 개행/공백 정규화, 영업담당 표준화, YY/MM 기한 파싱

실행:
    pip install -r requirements.txt
    streamlit run streamlit_app.py
"""
from __future__ import annotations

import math
import re
from datetime import datetime, date
from calendar import monthrange
from difflib import SequenceMatcher
from typing import Dict, Optional, Tuple, List

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------
# 기본 설정
# -----------------------------
st.set_page_config(page_title="선수·선급금 매칭 & 영업 대시보드", layout="wide", page_icon="📊")
DEFAULT_EXCEL_PATH = "./2025.07월말 선수선급금 현황_20250811.xlsx"

# -----------------------------
# 유틸
# -----------------------------
def norm_col(c: str) -> str:
    if not isinstance(c, str):
        return c
    return (
        c.replace("\\n", "")
         .replace("\n", "")
         .replace("\r", "")
         .replace(" ", "")
         .strip()
    )

def normalize_text(s: Optional[str]) -> str:
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return ""
    return "".join(ch for ch in str(s).upper().strip())

def text_sim(a: Optional[str], b: Optional[str]) -> float:
    a, b = normalize_text(a), normalize_text(b)
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()

def to_number(x) -> Optional[float]:
    try:
        if pd.isna(x):
            return None
        if isinstance(x, str):
            x = x.replace(",", "").replace(" ", "")
            # "184-150" 같은 코드 문자열 방지
            if re.search(r"[^\d.\-+]", x):
                return None
        v = float(x)
        if math.isfinite(v):
            return v
        return None
    except Exception:
        return None

def choose_amount_row(row: pd.Series) -> Optional[float]:
    for key in ["전표통화액", "현지통화액"]:
        if key in row.index:
            v = to_number(row[key])
            if v is not None:
                return v
    return None

def to_date(x) -> Optional[pd.Timestamp]:
    try:
        if pd.isna(x):
            return None
        return pd.to_datetime(x)
    except Exception:
        return None

def ensure_keycols(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        "정산\n선수금\n고유번호": "정산선수금고유번호",
        "정산여부\n(O/X)": "정산여부",
        "고객명\n(드롭다운)": "고객명",
        "회수목표일정\n(YY/MM)": "회수목표일정(YY/MM)",
        "경과기간\n(개월)": "경과기간(개월)",
        "영업담당\n(변경시)": "영업담당_변경시",
    }
    for old, new in mapping.items():
        if old in df.columns and new not in df.columns:
            df.rename(columns={old: new}, inplace=True)
    # 영업담당 표준 컬럼
    if "영업담당" in df.columns:
        df["영업담당_표준"] = df["영업담당"]
    elif "영업담당_변경시" in df.columns:
        df["영업담당_표준"] = df["영업담당_변경시"]
    else:
        df["영업담당_표준"] = None
    return df

def parse_due_yy_mm(val) -> Optional[pd.Timestamp]:
    """ 'YY/MM' 또는 'YY-MM' 등 기한 문자열을 월 말일로 파싱 """
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return None
    s = str(val).strip()
    if not s:
        return None
    m = re.match(r"^\s*(\d{2})[./\- ]?(\d{1,2})\s*$", s)
    if not m:
        return None
    yy = int(m.group(1))
    mm = int(m.group(2))
    year = 2000 + yy if yy <= 79 else 1900 + yy  # 00~79 → 2000~2079, 그 외는 1900대 처리
    mm = max(1, min(12, mm))
    last_day = monthrange(year, mm)[1]
    return pd.Timestamp(year=year, month=mm, day=last_day)

def add_common_fields(df: pd.DataFrame) -> pd.DataFrame:
    if "금액" not in df.columns:
        df["금액"] = df.apply(choose_amount_row, axis=1)
    if "전기일" in df.columns and "전기일_parsed" not in df.columns:
        df["전기일_parsed"] = pd.to_datetime(df["전기일"], errors="coerce")
    # 회수목표일정(YY/MM) → due_date
    if "회수목표일정(YY/MM)" in df.columns and "회수목표일자" not in df.columns:
        df["회수목표일자"] = df["회수목표일정(YY/MM)"].apply(parse_due_yy_mm)
    # 정산 여부 판단 필드
    df["is_settled"] = False
    if "정산여부" in df.columns:
        df["is_settled"] = df["is_settled"] | df["정산여부"].astype(str).str.contains("O", na=False)
    if "정산선수금고유번호" in df.columns:
        df["is_settled"] = df["is_settled"] | df["정산선수금고유번호"].astype(str).str.strip().ne("")
    # 금액 정수화 보조
    df["금액_num"] = df["금액"].apply(to_number)
    return df

# -----------------------------
# 데이터 로드
# -----------------------------
@st.cache_data(show_spinner=False)
def load_excel(excel_bytes_or_path) -> Dict[str, pd.DataFrame]:
    if excel_bytes_or_path is None:
        return {}
    xls = pd.ExcelFile(excel_bytes_or_path)
    sheets = {}
    for s in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=s)
        df.columns = [norm_col(c) for c in df.columns]
        df = df.dropna(how="all")
        df = ensure_keycols(df)
        df = add_common_fields(df)
        sheets[s] = df
    return sheets

def find_sheet(sheets: Dict[str, pd.DataFrame], target: str) -> Optional[str]:
    tgt = normalize_text(target)
    for name in sheets.keys():
        if normalize_text(name) == tgt:
            return name
    for name in sheets.keys():
        if tgt in normalize_text(name):
            return name
    return None

# -----------------------------
# 매칭 점수
# -----------------------------
def calc_match_score(sunsu: pd.Series, seongeup: pd.Series, date_half_life_days: int = 90) -> Tuple[float, Dict[str, float]]:
    weights = {"linked_id": 60.0, "contract": 20.0, "name": 10.0, "date": 5.0, "text": 5.0, "amount": 10.0}
    def get(row: pd.Series, key: str) -> Optional[str]:
        return row.get(key) if key in row.index else None

    linked = 0.0
    seon_link = get(seongeup, "정산선수금고유번호") or get(seongeup, "정산\n선수금\n고유번호")
    sun_id = get(sunsu, "고유넘버")
    if seon_link and sun_id and str(seon_link).strip() == str(sun_id).strip():
        linked = 1.0

    contract_equal = 0.0
    if get(sunsu, "계약번호") and get(seongeup, "계약번호"):
        if str(get(sunsu, "계약번호")).strip() == str(get(seongeup, "계약번호")).strip():
            contract_equal = 1.0

    name_sim = text_sim(get(sunsu, "업체명"), get(seongeup, "업체명"))

    d1 = to_date(get(sunsu, "전기일"))
    d2 = to_date(get(seongeup, "전기일"))
    date_score = 0.0
    if d1 is not None and d2 is not None:
        dd = abs((d1 - d2).days)
        date_score = 0.5 ** (dd / float(max(date_half_life_days, 1)))

    text_contains = 0.0
    if get(seongeup, "텍스트") and get(sunsu, "계약번호"):
        if str(get(sunsu, "계약번호")).strip() in str(get(seongeup, "텍스트")):
            text_contains = 1.0

    amt_sun = sunsu.get("금액_num")
    amt_seon = seongeup.get("금액_num")
    amount_score = 0.0
    if amt_sun is not None and amt_seon is not None and amt_sun != 0:
        diff = abs(amt_sun - amt_seon)
        amount_score = max(0.0, 1.0 - (diff / abs(amt_sun)))

    parts = {
        "linked_id": linked * weights["linked_id"],
        "contract": contract_equal * weights["contract"],
        "name": name_sim * weights["name"],
        "date": date_score * weights["date"],
        "text": text_contains * weights["text"],
        "amount": amount_score * weights["amount"],
    }
    total = sum(parts.values())
    return total, parts

# -----------------------------
# UI: 데이터 업로드
# -----------------------------
st.sidebar.header("데이터")
excel_file = st.sidebar.file_uploader("엑셀 업로드 (.xlsx)", type=["xlsx"], accept_multiple_files=False)

default_used = False
sheets = {}
if excel_file is not None:
    sheets = load_excel(excel_file)
else:
    try:
        with open(DEFAULT_EXCEL_PATH, "rb") as f:
            sheets = load_excel(f)
            default_used = True
            st.sidebar.info("기본 경로에서 엑셀을 불러왔습니다.")
    except Exception:
        st.sidebar.warning("엑셀을 업로드하거나, 리포지토리에 엑셀을 포함시켜 주세요.")

if not sheets:
    st.info("데이터가 없어서 기능을 비활성화합니다. 왼쪽에서 엑셀을 업로드하세요.")
    st.stop()

s_sunsu = find_sheet(sheets, "선수금")
s_seon = find_sheet(sheets, "선급금")
if s_sunsu is None or s_seon is None:
    st.error("시트 이름 '선수금'과 '선급금'을 찾지 못했습니다. 엑셀 시트명을 확인해주세요.")
    st.stop()

df_sunsu = sheets[s_sunsu].copy()
df_seon = sheets[s_seon].copy()

# -----------------------------
# 사이드바: 필터 프리셋
# -----------------------------
st.sidebar.header("공통 필터")
owner_all = sorted(set([x for x in df_sunsu["영업담당_표준"].dropna().unique().tolist() + df_seon["영업담당_표준"].dropna().unique().tolist()]))
owner = st.sidebar.multiselect("영업담당 선택(복수)", options=owner_all, default=[])
only_unsettled = st.sidebar.checkbox("미정산만 보기", value=False)
only_overdue = st.sidebar.checkbox("연체만 보기(현재 기준)", value=False)

def apply_owner_status_filter(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if owner:
        out = out[out["영업담당_표준"].isin(owner)]
    if only_unsettled:
        out = out[~out["is_settled"]]
    # overdue 계산
    now = pd.Timestamp.now()
    if only_overdue:
        if "회수목표일자" in out.columns:
            out = out[(~out["is_settled"]) & (out["회수목표일자"].notna()) & (out["회수목표일자"] < now)]
        else:
            out = out[~out["is_settled"]]  # 기한 없으면 미정산으로만 필터
    return out

df_sunsu_f = apply_owner_status_filter(df_sunsu)
df_seon_f = apply_owner_status_filter(df_seon)

# -----------------------------
# 탭
# -----------------------------
tab0, tab1, tab2, tab3, tab4 = st.tabs(["👤 영업 대시보드", "🔎 매칭 조회", "⚙️ 일괄 매칭", "📊 요약 대시보드", "🧭 고급 검색"])

# -----------------------------
# 👤 영업 대시보드
# -----------------------------
with tab0:
    st.subheader("영업 담당자 관점 KPI & 상태 보드 (현재 시점 기준)")
    now = pd.Timestamp.now()
    this_year = now.year
    this_month = now.month
    start_month = pd.Timestamp(year=this_year, month=this_month, day=1)
    end_month = pd.Timestamp(year=this_year, month=this_month, day=monthrange(this_year, this_month)[1])

    def status_bucket(df: pd.DataFrame) -> pd.DataFrame:
        base = df.copy()
        base["상태"] = "정보없음"
        # 기본 상태
        base.loc[base["is_settled"] == True, "상태"] = "처리완료"
        # 미정산 + 기한 존재
        has_due = base["회수목표일자"].notna() if "회수목표일자" in base.columns else pd.Series(False, index=base.index)
        cond_un = (base["is_settled"] == False)
        base.loc[cond_un & has_due & (base["회수목표일자"] < now), "상태"] = "연체"
        base.loc[cond_un & has_due & (base["회수목표일자"].dt.to_period("M") == start_month.to_period("M")), "상태"] = "당월예정"
        base.loc[cond_un & (~has_due), "상태"] = "기한미설정"
        base.loc[cond_un & has_due & (base["회수목표일자"] > end_month), "상태"] = "향후예정"
        return base

    sunsu_s = status_bucket(df_sunsu_f)
    seon_s = status_bucket(df_seon_f)

    def kpi_block(title: str, df: pd.DataFrame):
        total = len(df)
        done = int(df["is_settled"].sum())
        overdue = int(((~df["is_settled"]) & (df.get("회수목표일자").notna()) & (df["회수목표일자"] < now)).sum())
        due_this = int(((~df["is_settled"]) & (df.get("회수목표일자").notna()) & (df["회수목표일자"].dt.to_period("M") == start_month.to_period("M"))).sum())
        amt_total = df["금액_num"].sum(skipna=True)
        amt_un = df.loc[~df["is_settled"], "금액_num"].sum(skipna=True)
        cols = st.columns(5)
        with cols[0]: st.metric(f"{title} 건수", f"{total:,}")
        with cols[1]: st.metric("처리완료", f"{done:,}")
        with cols[2]: st.metric("연체", f"{overdue:,}")
        with cols[3]: st.metric("당월예정", f"{due_this:,}")
        with cols[4]: st.metric("미정산금액", f"{amt_un:,.0f}" if pd.notna(amt_un) else "-")

    st.markdown("### 선수금")
    kpi_block("선수금", sunsu_s)
    st.markdown("### 선급금")
    kpi_block("선급금", seon_s)

    # 상태별 금액 합계 차트 (담당자 필터 반영)
    def status_chart(df: pd.DataFrame, title: str):
        base = df.copy()
        base = base.dropna(subset=["금액_num"])
        agg = base.groupby("상태")["금액_num"].sum().reset_index()
        chart = alt.Chart(agg).mark_bar().encode(
            x=alt.X("상태:N", title="상태"),
            y=alt.Y("금액_num:Q", title="금액 합계")
        ).properties(height=280, title=f"{title} - 상태별 금액")
        st.altair_chart(chart, use_container_width=True)
        st.dataframe(agg.rename(columns={"금액_num": "금액합계"}), use_container_width=True, height=240)

    c1, c2 = st.columns(2)
    with c1: status_chart(sunsu_s, "선수금")
    with c2: status_chart(seon_s, "선급금")

    # 담당자별 미정산 현황
    def owner_unsettled(df: pd.DataFrame, title: str):
        base = df[~df["is_settled"]].copy()
        if base.empty:
            st.info(f"{title}: 미정산 없음")
            return
        agg = base.groupby("영업담당_표준", dropna=False)["금액_num"].sum().reset_index().sort_values(by="금액_num", ascending=False)
        chart = alt.Chart(agg.dropna()).mark_bar().encode(
            x=alt.X("금액_num:Q", title="미정산 금액"),
            y=alt.Y("영업담당_표준:N", sort="-x", title="영업담당")
        ).properties(height=360, title=f"{title} - 담당자별 미정산")
        st.altair_chart(chart, use_container_width=True)
        st.dataframe(agg.rename(columns={"금액_num": "미정산금액"}), use_container_width=True, height=260)

    c3, c4 = st.columns(2)
    with c3: owner_unsettled(sunsu_s, "선수금")
    with c4: owner_unsettled(seon_s, "선급금")

# -----------------------------
# 🔎 매칭 조회
# -----------------------------
with tab1:
    st.subheader("특정 선수금과 매칭되는 선급금 후보 조회")
    date_half_life_days = st.slider("일자 근접도 절반감쇠일(일)", 15, 180, 90, 15, key="m_dhl")
    score_threshold = st.slider("후보 표시 최소점수", 0, 100, 40, 5, key="m_th")
    if df_sunsu_f.empty or df_seon_f.empty:
        st.warning("필터 적용 후 데이터가 비어 있습니다. 사이드바 필터를 조정하세요.")
    else:
        def sunsu_label(row: pd.Series) -> str:
            gid = str(row.get("고유넘버", ""))
            comp = str(row.get("업체명", ""))
            contract = str(row.get("계약번호", ""))
            datev = row.get("전기일_parsed", row.get("전기일", ""))
            amt = row.get("금액_num", None)
            amt_str = f"{amt:,.0f}" if isinstance(amt, (int, float, np.number)) and not pd.isna(amt) else "-"
            dstr = datev.strftime("%Y-%m-%d") if isinstance(datev, pd.Timestamp) else (str(datev) if datev is not None else "")
            return f"[{gid}] {comp} | 계약:{contract} | 일자:{dstr} | 금액:{amt_str}"

        sunsu_options = df_sunsu_f.index.tolist()
        selectable = [(i, sunsu_label(df_sunsu_f.loc[i])) for i in sunsu_options]
        selected_idx = st.selectbox("선수금 선택", options=[i for i, _ in selectable], format_func=lambda i: dict(selectable)[i])

        if selected_idx is not None:
            target_row = df_sunsu_f.loc[selected_idx]
            scores: List[dict] = []
            for i, row in df_seon_f.iterrows():
                total, parts = calc_match_score(target_row, row, date_half_life_days=date_half_life_days)
                if total >= score_threshold:
                    scores.append({
                        "선급_index": i,
                        "총점": round(total, 2),
                        **{f"점수:{k}": round(v, 2) for k, v in parts.items()},
                        "계약번호": row.get("계약번호"),
                        "업체명": row.get("업체명"),
                        "전기일": row.get("전기일_parsed", row.get("전기일")),
                        "금액": row.get("금액_num"),
                        "정산선수금고유번호": row.get("정산선수금고유번호"),
                        "텍스트": row.get("텍스트"),
                        "고유넘버": row.get("고유넘버"),
                        "영업담당": row.get("영업담당_표준"),
                    })
            if not scores:
                st.info("후보가 없습니다. 점수 임계값 또는 필터를 조정해 주세요.")
            else:
                cand_df = pd.DataFrame(scores)
                if "총점" in cand_df.columns:
                    cand_df = cand_df.sort_values(by=["총점"], ascending=False).reset_index(drop=True)
                st.dataframe(cand_df, use_container_width=True, height=430)

# -----------------------------
# ⚙️ 일괄 매칭
# -----------------------------
with tab2:
    st.subheader("일괄 매칭 제안(Top-1)")
    score_threshold2 = st.slider("후보 표시 최소점수", 0, 100, 40, 5, key="b_th")
    limit = st.number_input("대상 선수금 수", min_value=10, max_value=max(10, len(df_sunsu_f)), value=min(200, len(df_sunsu_f) if len(df_sunsu_f)>0 else 10), step=10)
    if df_sunsu_f.empty or df_seon_f.empty:
        st.info("데이터가 비어 있어 일괄 제안을 생략합니다.")
    else:
        rows = []
        for si, srow in df_sunsu_f.head(int(limit)).iterrows():
            best_score = -1.0
            best_idx = None
            for ei, erow in df_seon_f.iterrows():
                total, _ = calc_match_score(srow, erow, date_half_life_days=90)
                if total > best_score:
                    best_score = total
                    best_idx = ei
            if best_idx is not None and best_score >= score_threshold2:
                erow = df_seon_f.loc[best_idx]
                rows.append({
                    "선수_index": si, "선급_index": best_idx, "총점": round(best_score, 2),
                    "선수_고유넘버": srow.get("고유넘버"), "선수_계약번호": srow.get("계약번호"),
                    "선수_업체명": srow.get("업체명"), "선수_전기일": srow.get("전기일_parsed", srow.get("전기일")),
                    "선수_금액": srow.get("금액_num"),
                    "선급_고유넘버": erow.get("고유넘버"), "선급_계약번호": erow.get("계약번호"),
                    "선급_업체명": erow.get("업체명"), "선급_전기일": erow.get("전기일_parsed", erow.get("전기일")),
                    "선급_금액": erow.get("금액_num"),
                    "선급_정산선수금고유번호": erow.get("정산선수금고유번호"),
                    "영업담당": srow.get("영업담당_표준")
                })
        if not rows:
            st.info("제안 가능한 매칭이 없습니다.")
        else:
            dfb = pd.DataFrame(rows).sort_values(by="총점", ascending=False).reset_index(drop=True)
            st.dataframe(dfb, use_container_width=True, height=450)
            st.download_button("CSV 다운로드", dfb.to_csv(index=False).encode("utf-8-sig"), file_name="batch_match_suggestions.csv", mime="text/csv")

# -----------------------------
# 📊 요약 대시보드
# -----------------------------
with tab3:
    st.subheader("요약 지표 & 시각화")
    def group_unsettled(df: pd.DataFrame, title: str):
        base = df[~df["is_settled"]].copy()
        base = base.dropna(subset=["금액_num"])
        agg = base.groupby("업체명", dropna=False)["금액_num"].sum().reset_index().sort_values(by="금액_num", ascending=False).head(20)
        st.markdown(f"**{title} - 미정산 금액 상위 20 업체**")
        chart = alt.Chart(agg.dropna()).mark_bar().encode(
            x=alt.X("금액_num:Q", title="금액 합계"),
            y=alt.Y("업체명:N", sort="-x", title="업체명")
        ).properties(height=360)
        st.altair_chart(chart, use_container_width=True)
        st.dataframe(agg.rename(columns={"금액_num": "금액합계"}), use_container_width=True, height=260)

    c1, c2 = st.columns(2)
    with c1:
        if not df_sunsu_f.empty: group_unsettled(df_sunsu_f, "선수금")
        else: st.info("선수금 데이터 없음")
    with c2:
        if not df_seon_f.empty: group_unsettled(df_seon_f, "선급금")
        else: st.info("선급금 데이터 없음")

    # 에이징
    def aging_chart(df: pd.DataFrame, title: str):
        col = "경과기간(개월)" if "경과기간(개월)" in df.columns else None
        if col is None or df.empty:
            st.info(f"{title}: '경과기간(개월)' 컬럼이 없어 에이징 차트를 생략합니다.")
            return
        base = df.copy()
        base = base.dropna(subset=["금액_num"])
        def bucket(x):
            try: v = float(x)
            except Exception: return "미상"
            if v < 1: return "0-1개월"
            if v < 3: return "1-3개월"
            if v < 6: return "3-6개월"
            if v < 12: return "6-12개월"
            if v < 24: return "12-24개월"
            return "24개월+"
        base["버킷"] = base[col].apply(bucket)
        agg = base.groupby("버킷")["금액_num"].sum().reset_index()
        order = ["0-1개월", "1-3개월", "3-6개월", "6-12개월", "12-24개월", "24개월+"]
        agg["버킷"] = pd.Categorical(agg["버킷"], categories=order, ordered=True)
        agg = agg.sort_values("버킷")
        chart = alt.Chart(agg).mark_bar().encode(
            x=alt.X("버킷:N", sort=order, title="경과기간 버킷"),
            y=alt.Y("금액_num:Q", title="금액 합계")
        ).properties(height=300)
        st.markdown(f"**{title} - 에이징(개월) 분포**")
        st.altair_chart(chart, use_container_width=True)
        st.dataframe(agg.rename(columns={"금액_num": "금액합계"}), use_container_width=True, height=240)

    c3, c4 = st.columns(2)
    with c3: aging_chart(df_sunsu_f, "선수금")
    with c4: aging_chart(df_seon_f, "선급금")

# -----------------------------
# 🧭 고급 검색
# -----------------------------
with tab4:
    st.subheader("강화된 검색(선수금/선급금 통합)")

    # 통합 뷰
    sunsu_view = df_sunsu_f.copy()
    sunsu_view["구분"] = "선수금"
    seon_view = df_seon_f.copy()
    seon_view["구분"] = "선급금"
    all_view = pd.concat([sunsu_view, seon_view], ignore_index=True, sort=False)

    # 필터 UI
    col1, col2, col3 = st.columns(3)
    with col1:
        kw = st.text_input("키워드(업체/계약/텍스트/전표번호/금형/고유넘버)", value="")
    with col2:
        min_amt = st.number_input("최소 금액", value=0, step=1000)
    with col3:
        max_amt = st.number_input("최대 금액(0=제한없음)", value=0, step=1000)

    col4, col5, col6 = st.columns(3)
    with col4:
        start_date = st.date_input("전기일 From", value=None)
    with col5:
        end_date = st.date_input("전기일 To", value=None)
    with col6:
        only_due_this_month = st.checkbox("당월예정만", value=False)

    # 적용
    res = all_view.copy()

    # 금액 필터
    res = res[(res["금액_num"].fillna(0) >= (min_amt or 0))]
    if max_amt and max_amt > 0:
        res = res[(res["금액_num"].fillna(0) <= max_amt)]

    # 날짜 필터
    if start_date:
        sdt = pd.Timestamp(start_date)
        res = res[res["전기일_parsed"].fillna(pd.Timestamp("1900-01-01")) >= sdt]
    if end_date:
        edt = pd.Timestamp(end_date)
        res = res[res["전기일_parsed"].fillna(pd.Timestamp("2999-12-31")) <= edt]

    # 당월예정
    if only_due_this_month:
        now = pd.Timestamp.now()
        res = res[(~res["is_settled"]) & (res["회수목표일자"].notna()) & (res["회수목표일자"].dt.to_period("M") == now.to_period("M"))]

    # 키워드(contains, 대소문자 무시)
    if kw.strip():
        k = kw.strip().upper()
        def contains_any(s):
            if s is None or (isinstance(s, float) and math.isnan(s)): return False
            return k in str(s).upper()
        search_cols = [c for c in ["업체명","계약번호","텍스트","전표번호","금형마스터","금형마스터내역","고유넘버"] if c in res.columns]
        mask = False
        for c in search_cols:
            mask = mask | res[c].apply(contains_any)
        res = res[mask]

    # 정렬 & 표시
    sort_cols = [c for c in ["구분","is_settled","회수목표일자","전기일_parsed","금액_num"] if c in res.columns]
    if sort_cols:
        res = res.sort_values(by=sort_cols, ascending=[True, True, True, True, False]).reset_index(drop=True)
    show_cols = [c for c in ["구분","영업담당_표준","업체명","계약번호","고유넘버","전기일_parsed","회수목표일자","금액_num","정산선수금고유번호","정산여부","진행현황","텍스트"] if c in res.columns]
    st.dataframe(res[show_cols], use_container_width=True, height=520)
    st.download_button("검색결과 CSV 다운로드", res[show_cols].to_csv(index=False).encode("utf-8-sig"), file_name="search_results.csv", mime="text/csv")

st.caption("ⓘ 담당자/상태 중심 KPI·차트, 통합 검색 강화 완료. 필요 시 상태 정의/가중치/컬럼 매핑 커스터마이즈 가능.")
