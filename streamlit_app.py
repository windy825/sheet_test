
# -*- coding: utf-8 -*-
"""
streamlit_app.py
로컬/클라우드 공용: 선수·선급금 매칭 & 대시보드 (안전가드 강화판)

주요 개선
- 데이터 미로딩/빈 데이터프레임일 때에도 앱이 죽지 않고 안내문 표시
- 점수 후보가 없을 때 '총점' 정렬에서 KeyError 나는 문제 방지
- 금액 변환은 지정 컬럼(전표통화액/현지통화액)만 대상으로 안전 처리
- 시트명/컬럼명 개행·공백 정규화

실행 방법(로컬):
    pip install -r requirements.txt
    streamlit run streamlit_app.py
"""
from __future__ import annotations

import math
from datetime import datetime
from difflib import SequenceMatcher
from typing import Dict, Optional, Tuple, List

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------
# 기본 설정
# -----------------------------
st.set_page_config(page_title="선수·선급금 매칭 & 대시보드", layout="wide", page_icon="📊")

DEFAULT_EXCEL_PATH = "./2025.07월말 선수선급금 현황_20250811.xlsx"  # 로컬에서 동일 폴더에 둘 경우 자동 로드 시도

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
        v = float(x)
        if math.isfinite(v):
            return v
        return None
    except Exception:
        return None

def choose_amount_row(row: pd.Series) -> Optional[float]:
    # 금액 우선순위: 전표통화액 -> 현지통화액
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
    # 자주 쓰는 개행 포함 컬럼을 단일 키로 통일
    mapping = {
        "정산\n선수금\n고유번호": "정산선수금고유번호",
        "정산여부\n(O/X)": "정산여부",
        "고객명\n(드롭다운)": "고객명",
        "회수목표일정\n(YY/MM)": "회수목표일정(YY/MM)",
        "경과기간\n(개월)": "경과기간(개월)",
    }
    for old, new in mapping.items():
        if old in df.columns and new not in df.columns:
            df.rename(columns={old: new}, inplace=True)
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
        sheets[s] = ensure_keycols(df)
    return sheets

def find_sheet(sheets: Dict[str, pd.DataFrame], target: str) -> Optional[str]:
    target_norm = normalize_text(target)
    for name in sheets.keys():
        if normalize_text(name) == target_norm:
            return name
    for name in sheets.keys():
        if target_norm in normalize_text(name):
            return name
    return None

# -----------------------------
# 매칭 점수
# -----------------------------
def calc_match_score(sunsu: pd.Series, seongeup: pd.Series,
                     today: datetime, date_half_life_days: int = 90) -> Tuple[float, Dict[str, float]]:
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

    amt_sun = choose_amount_row(sunsu)
    amt_seon = choose_amount_row(seongeup)
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
# UI: 데이터 업로드/로드
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
    st.info("데이터가 없어서 기능을 비활성화합니다. 왼쪽 사이드바에서 엑셀을 업로드하면 모든 기능이 켜집니다.")
    st.stop()

s_sunsu = find_sheet(sheets, "선수금")
s_seon = find_sheet(sheets, "선급금")
if s_sunsu is None or s_seon is None:
    st.error("시트 이름 '선수금'과 '선급금'을 찾지 못했습니다. 엑셀 시트명을 확인해주세요.")
    st.stop()

df_sunsu = sheets[s_sunsu].copy()
df_seon = sheets[s_seon].copy()

# 금액/일자 파생
for df in (df_sunsu, df_seon):
    if "금액" not in df.columns:
        df["금액"] = df.apply(choose_amount_row, axis=1)
    if "전기일" in df.columns:
        df["전기일_parsed"] = pd.to_datetime(df["전기일"], errors="coerce")

# -----------------------------
# 매칭 옵션
# -----------------------------
st.sidebar.header("매칭 옵션")
date_half_life_days = st.sidebar.slider("일자 근접도 절반감쇠일(일)", 15, 180, 90, 15)
score_threshold = st.sidebar.slider("후보 표시 최소점수", 0, 100, 40, 5)

# -----------------------------
# 탭
# -----------------------------
tab1, tab2, tab3 = st.tabs(["🔎 매칭 조회", "⚙️ 일괄 매칭 제안", "📊 대시보드"])

with tab1:
    st.subheader("특정 선수금과 매칭되는 선급금 후보 조회")

    if df_sunsu.empty or df_seon.empty:
        st.warning("선수금/선급금 데이터가 비어있습니다. 업로드 파일을 확인해주세요.")
        st.stop()

    def sunsu_label(row: pd.Series) -> str:
        gid = str(row.get("고유넘버", ""))
        comp = str(row.get("업체명", ""))
        contract = str(row.get("계약번호", ""))
        date = row.get("전기일_parsed", row.get("전기일", ""))
        amt = row.get("금액", None)
        amt_str = f"{amt:,.0f}" if isinstance(amt, (int, float, np.number)) and not pd.isna(amt) else "-"
        dstr = date.strftime("%Y-%m-%d") if isinstance(date, pd.Timestamp) else (str(date) if date is not None else "")
        return f"[{gid}] {comp} | 계약:{contract} | 일자:{dstr} | 금액:{amt_str}"

    sunsu_options = df_sunsu.index.tolist()
    if not sunsu_options:
        st.info("선수금 행이 없습니다.")
        st.stop()

    selectable = [(i, sunsu_label(df_sunsu.loc[i])) for i in sunsu_options]
    selected_idx = st.selectbox("선수금 선택", options=[i for i, _ in selectable], format_func=lambda i: dict(selectable)[i])

    if selected_idx is None:
        st.info("선수금을 선택하면 후보가 표시됩니다.")
    else:
        target_row = df_sunsu.loc[selected_idx]
        today = datetime.now()

        # 단일 후보 점수 계산
        scores: List[dict] = []
        for i, row in df_seon.iterrows():
            total, parts = calc_match_score(target_row, row, today=today, date_half_life_days=date_half_life_days)
            if total >= score_threshold:
                scores.append({
                    "선급_index": i,
                    "총점": round(total, 2),
                    **{f"점수:{k}": round(v, 2) for k, v in parts.items()},
                    "계약번호": row.get("계약번호"),
                    "업체명": row.get("업체명"),
                    "전기일": row.get("전기일_parsed", row.get("전기일")),
                    "금액": row.get("금액"),
                    "정산선수금고유번호": row.get("정산선수금고유번호"),
                    "텍스트": row.get("텍스트"),
                    "고유넘버": row.get("고유넘버"),
                })

        if not scores:
            st.info("점수 임계값을 낮추거나, 업로드 데이터(금액/계약번호/일자/업체명) 컬럼을 확인해 주세요. 현재 조건에 부합하는 후보가 없습니다.")
        else:
            cand_df = pd.DataFrame(scores)
            # 안전 정렬
            if "총점" in cand_df.columns:
                cand_df = cand_df.sort_values(by=["총점"], ascending=False).reset_index(drop=True)
            st.markdown("**단일 항목 후보(점수 순)**")
            st.dataframe(cand_df, use_container_width=True, height=420)

with tab2:
    st.subheader("일괄 매칭 제안(Top-1 단일 후보)")

    if df_sunsu.empty or df_seon.empty:
        st.info("데이터가 비어 있어 일괄 제안을 생략합니다.")
    else:
        max_rows = st.number_input("대상 선수금 수(상위 N행)", 10, max(10, len(df_sunsu)), min(200, len(df_sunsu)), 10)
        today = datetime.now()
        batch_rows = []

        sunsu_iter = df_sunsu.head(int(max_rows))
        for si, srow in sunsu_iter.iterrows():
            best_score = -1.0
            best_idx = None
            for ei, erow in df_seon.iterrows():
                total, _ = calc_match_score(srow, erow, today=today, date_half_life_days=date_half_life_days)
                if total > best_score:
                    best_score = total
                    best_idx = ei
            if best_idx is not None and best_score >= score_threshold:
                erow = df_seon.loc[best_idx]
                batch_rows.append({
                    "선수_index": si,
                    "선급_index": best_idx,
                    "총점": round(best_score, 2),
                    "선수_고유넘버": srow.get("고유넘버"),
                    "선수_계약번호": srow.get("계약번호"),
                    "선수_업체명": srow.get("업체명"),
                    "선수_전기일": srow.get("전기일_parsed", srow.get("전기일")),
                    "선수_금액": srow.get("금액"),
                    "선급_고유넘버": erow.get("고유넘버"),
                    "선급_계약번호": erow.get("계약번호"),
                    "선급_업체명": erow.get("업체명"),
                    "선급_전기일": erow.get("전기일_parsed", erow.get("전기일")),
                    "선급_금액": erow.get("금액"),
                    "선급_정산선수금고유번호": erow.get("정산선수금고유번호"),
                })

        if not batch_rows:
            st.info("제안 가능한 매칭이 없습니다. 점수 임계값을 낮춰보세요.")
        else:
            batch_df = pd.DataFrame(batch_rows).sort_values(by="총점", ascending=False).reset_index(drop=True)
            st.dataframe(batch_df, use_container_width=True, height=450)
            csv = batch_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button("일괄 제안 CSV 다운로드", data=csv, file_name="batch_match_suggestions.csv", mime="text/csv")

with tab3:
    st.subheader("요약 지표 & 시각화")

    def is_matched_row(df: pd.DataFrame) -> pd.Series:
        cond = pd.Series(False, index=df.index)
        if "정산여부" in df.columns:
            cond = cond | df["정산여부"].astype(str).str.contains("O", na=False)
        if "정산선수금고유번호" in df.columns:
            cond = cond | df["정산선수금고유번호"].astype(str).str.strip().ne("")
        return cond

    if df_sunsu.empty and df_seon.empty:
        st.info("대시보드를 렌더링할 데이터가 없습니다.")
    else:
        sunsu_matched = is_matched_row(df_sunsu) if not df_sunsu.empty else pd.Series([], dtype=bool)
        seon_matched = is_matched_row(df_seon) if not df_seon.empty else pd.Series([], dtype=bool)

        kpi = st.columns(4)
        with kpi[0]:
            st.metric("선수금 건수", f"{len(df_sunsu):,}")
        with kpi[1]:
            st.metric("선수금 미정산 건수", f"{int((~sunsu_matched).sum()) if len(sunsu_matched) else 0:,}")
        with kpi[2]:
            st.metric("선급금 건수", f"{len(df_seon):,}")
        with kpi[3]:
            st.metric("선급금 미정산 건수", f"{int((~seon_matched).sum()) if len(seon_matched) else 0:,}")

        def to_amt_df(df: pd.DataFrame) -> pd.DataFrame:
            base = df.copy()
            base["금액"] = base["금액"].apply(to_number)
            return base.dropna(subset=["금액"])

        def group_unsettled(df: pd.DataFrame, title: str):
            if df.empty:
                st.info(f"{title}: 데이터 없음")
                return
            base = df.copy()
            m = is_matched_row(base)
            base = base[~m]
            base = to_amt_df(base)
            if base.empty:
                st.info(f"{title}: 미정산 항목 없음")
                return
            agg = base.groupby("업체명", dropna=False)["금액"].sum().reset_index().sort_values(by="금액", ascending=False).head(20)
            chart = alt.Chart(agg.dropna()).mark_bar().encode(
                x=alt.X("금액:Q", title="금액 합계"),
                y=alt.Y("업체명:N", sort="-x", title="업체명")
            ).properties(height=360)
            st.markdown(f"**{title} - 미정산 금액 상위 20 업체**")
            st.altair_chart(chart, use_container_width=True)
            st.dataframe(agg, use_container_width=True, height=260)

        c1, c2 = st.columns(2)
        with c1: group_unsettled(df_sunsu, "선수금")
        with c2: group_unsettled(df_seon, "선급금")

        def aging_chart(df: pd.DataFrame, title: str):
            col = "경과기간(개월)" if "경과기간(개월)" in df.columns else None
            if df.empty or col is None:
                st.info(f"{title}: '경과기간(개월)' 컬럼이 없어 에이징 차트를 생략합니다.")
                return
            base = to_amt_df(df)
            if base.empty:
                st.info(f"{title}: 금액 데이터 없음")
                return
            def bucket(x):
                try:
                    v = float(x)
                except Exception:
                    return "미상"
                if v < 1: return "0-1개월"
                if v < 3: return "1-3개월"
                if v < 6: return "3-6개월"
                if v < 12: return "6-12개월"
                if v < 24: return "12-24개월"
                return "24개월+"
            base["버킷"] = base[col].apply(bucket)
            agg = base.groupby("버킷")["금액"].sum().reset_index()
            order = ["0-1개월", "1-3개월", "3-6개월", "6-12개월", "12-24개월", "24개월+"]
            agg["버킷"] = pd.Categorical(agg["버킷"], categories=order, ordered=True)
            agg = agg.sort_values("버킷")
            chart = alt.Chart(agg).mark_bar().encode(
                x=alt.X("버킷:N", sort=order, title="경과기간 버킷"),
                y=alt.Y("금액:Q", title="금액 합계")
            ).properties(height=300)
            st.markdown(f"**{title} - 에이징(개월) 분포**")
            st.altair_chart(chart, use_container_width=True)
            st.dataframe(agg, use_container_width=True, height=240)

        c3, c4 = st.columns(2)
        with c3: aging_chart(df_sunsu, "선수금")
        with c4: aging_chart(df_seon, "선급금")

st.caption("ⓘ 데이터가 비어 있거나 후보가 없는 경우에도 오류 없이 안내 메시지만 표시되도록 보강했습니다.")
