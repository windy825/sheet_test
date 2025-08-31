
# -*- coding: utf-8 -*-
"""
로컬용 선수/선급금 매칭 & 대시보드 (Streamlit)
------------------------------------------------
요구사항 요약
- 특정 "선수금"과 매칭될 수 있는 "선급금"들을 쉽게 조회
- 자동 매칭 제안(1:N 조합 탐색 포함)
- 심화 대시보드

실행 방법
1) Python 3.9+ 권장
2) 아래 라이브러리 설치
   pip install streamlit pandas numpy openpyxl altair
3) 실행
   streamlit run app.py

기본 파일 경로(옵션):
- 본 앱은 기본적으로 같은 디렉토리에 있는 Excel 파일을 찾거나, 사이드바에서 업로드할 수 있습니다.
- 예시: /mnt/data/2025.07월말 선수선급금 현황_20250811.xlsx
"""
from __future__ import annotations

import itertools
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from difflib import SequenceMatcher

# -----------------------------
# 설정
# -----------------------------
st.set_page_config(
    page_title="선수·선급금 매칭 & 대시보드",
    layout="wide",
    page_icon="📊",
)

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
    s = str(s)
    return "".join(ch for ch in s.upper().strip())

def text_sim(a: str, b: str) -> float:
    a, b = normalize_text(a), normalize_text(b)
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()

def to_number(x) -> Optional[float]:
    try:
        if pd.isna(x):
            return None
        # 문자열 금액에 콤마 처리
        if isinstance(x, str):
            x = x.replace(",", "").replace(" ", "")
        v = float(x)
        if math.isfinite(v):
            return v
        return None
    except Exception:
        return None

def to_date(x) -> Optional[pd.Timestamp]:
    try:
        if pd.isna(x):
            return None
        return pd.to_datetime(x)
    except Exception:
        return None

def choose_amount_row(row: pd.Series) -> Optional[float]:
    # 금액 우선순위: 전표통화액 -> 현지통화액
    for key in ["전표통화액", "현지통화액"]:
        if key in row.index:
            v = to_number(row[key])
            if v is not None:
                return v
    # 백업: 금액 혹은 Debit/Credit 계열이 있으면 확장 가능
    return None

def days_between(d1: Optional[pd.Timestamp], d2: Optional[pd.Timestamp]) -> Optional[int]:
    if d1 is None or d2 is None:
        return None
    return abs((pd.to_datetime(d1) - pd.to_datetime(d2)).days)

# -----------------------------
# 데이터 로드 & 전처리
# -----------------------------
@st.cache_data(show_spinner=False)
def load_excel(excel_bytes_or_path) -> Dict[str, pd.DataFrame]:
    if excel_bytes_or_path is None:
        return {}
    try:
        xls = pd.ExcelFile(excel_bytes_or_path)
        sheets = {}
        for s in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=s)
            # 컬럼 정규화
            df.columns = [norm_col(c) for c in df.columns]
            # 빈 행 제거(전부 NaN)
            df = df.dropna(how="all")
            sheets[s] = df
        return sheets
    except Exception as e:
        st.error(f"엑셀 로드 실패: {e}")
        return {}

def find_sheet_case_insensitive(sheets: Dict[str, pd.DataFrame], target: str) -> Optional[str]:
    target_norm = normalize_text(target)
    for name in sheets.keys():
        if normalize_text(name) == target_norm:
            return name
    # 넓게 포함 검색
    for name in sheets.keys():
        if target_norm in normalize_text(name):
            return name
    return None

# -----------------------------
# 매칭 스코어
# -----------------------------
def calc_match_score(sunsu: pd.Series, seongeup: pd.Series, today: datetime, date_half_life_days: int = 90) -> Tuple[float, Dict[str, float]]:
    """
    두 행(선수금 1개 vs 선급금 1개)에 대한 매칭 점수(0~100)와 구성요소 세부 점수 반환
    가중치 설계(경험치 기반, 필요시 사이드바에서 조정 가능하도록 변경 가능)
    """
    weights = {
        "linked_id": 60.0,   # 정산선수금고유번호 = 선수금 고유넘버
        "contract": 20.0,    # 계약번호 완전일치
        "name": 10.0,        # 업체명 유사도(0~1)*10
        "date": 5.0,         # 일자 근접도
        "text": 5.0,         # 텍스트에 계약번호 포함
        "amount": 10.0,      # 금액 근접도
    }

    # 공통 컬럼 안전 접근
    def get(row: pd.Series, key: str) -> Optional[str]:
        return row.get(key) if key in row.index else None

    # Linked ID
    linked = 0.0
    seon_link = get(seongeup, "정산선수금고유번호") or get(seongeup, "정산\n선수금\n고유번호")
    sun_id = get(sunsu, "고유넘버")
    if seon_link and sun_id and str(seon_link).strip() == str(sun_id).strip():
        linked = 1.0

    # 계약번호
    contract_equal = 0.0
    if get(sunsu, "계약번호") and get(seongeup, "계약번호"):
        if str(get(sunsu, "계약번호")).strip() == str(get(seongeup, "계약번호")).strip():
            contract_equal = 1.0

    # 업체명 유사도
    name_sim = text_sim(get(sunsu, "업체명"), get(seongeup, "업체명"))

    # 일자 근접(지수감쇠)
    d1 = to_date(get(sunsu, "전기일"))
    d2 = to_date(get(seongeup, "전기일"))
    date_score = 0.0
    if d1 is not None and d2 is not None:
        dd = abs((d1 - d2).days)
        # 절반감쇠일(date_half_life_days) 기준의 지수 점수
        # dd=0 ->1, dd=half_life ->0.5, dd=2*half_life->0.25 ...
        if date_half_life_days <= 0:
            date_score = 1.0 if dd == 0 else 0.0
        else:
            date_score = 0.5 ** (dd / float(date_half_life_days))

    # 텍스트에 계약번호 포함
    text_contains = 0.0
    if get(seongeup, "텍스트") and get(sunsu, "계약번호"):
        if str(get(sunsu, "계약번호")).strip() in str(get(seongeup, "텍스트")):
            text_contains = 1.0

    # 금액 근접
    amt_sun = choose_amount_row(sunsu)
    amt_seon = choose_amount_row(seongeup)
    amount_score = 0.0
    if amt_sun is not None and amt_seon is not None and amt_sun != 0:
        diff = abs(amt_sun - amt_seon)
        rel = max(0.0, 1.0 - (diff / abs(amt_sun)))  # 0~1
        amount_score = rel

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
# 1:N 조합 탐색(가벼운 휴리스틱)
# -----------------------------
def propose_combinations(target_amount: float,
                         candidates: pd.DataFrame,
                         amount_col: str = "금액",
                         max_depth: int = 6,
                         max_nodes: int = 800,
                         tolerance: float = 0.05) -> List[Dict]:
    """
    Greedy + 제한적 백트래킹으로 1:N 조합 제안
    - candidates에는 금액 열이 존재해야 함(미리 추출해서 넣음)
    - tolerance: 목표 대비 허용 오차(비율)
    """
    rows = candidates.copy()
    rows = rows[rows[amount_col].notna()].reset_index(drop=True)
    rows = rows.sort_values(by=amount_col, key=lambda s: (target_amount - s).abs())

    best_solutions: List[Dict] = []
    visited = 0

    def backtrack(start_idx: int, chosen_idx: List[int], sum_amt: float):
        nonlocal visited, best_solutions
        if visited >= max_nodes or len(chosen_idx) > max_depth:
            return
        visited += 1

        gap = target_amount - sum_amt
        # 수용 범위 내면 솔루션 추가
        if abs(gap) <= abs(target_amount) * tolerance:
            combo = rows.loc[chosen_idx]
            best_solutions.append({
                "indices": chosen_idx.copy(),
                "rows": combo.copy(),
                "sum": float(sum_amt),
                "gap": float(gap),
                "count": len(chosen_idx),
            })
            # 더 깊게 가지 않고 종료(추가 확장 방지)
            return

        # 가지치기: 초과되었고 target > 0인 경우
        if (target_amount >= 0 and sum_amt > target_amount * (1 + tolerance)) or \
           (target_amount < 0 and sum_amt < target_amount * (1 + tolerance)):
            return

        # 탐색 확장
        for i in range(start_idx, len(rows)):
            amt = rows.iloc[i][amount_col]
            if pd.isna(amt):
                continue
            backtrack(i + 1, chosen_idx + [i], sum_amt + amt)

    backtrack(0, [], 0.0)
    # 상위 솔루션 정렬: 오차 -> 항목수 -> 금액합 근접
    best_solutions = sorted(best_solutions, key=lambda x: (abs(x["gap"]), x["count"]))
    # 중복 제거(같은 인덱스 조합)
    unique = []
    seen = set()
    for sol in best_solutions:
        key = tuple(sol["indices"])
        if key not in seen:
            seen.add(key)
            unique.append(sol)
    return unique[:10]

# -----------------------------
# 사이드바 - 데이터 로딩
# -----------------------------
st.sidebar.header("데이터")
excel_file = st.sidebar.file_uploader("엑셀 업로드 (.xlsx)", type=["xlsx"], accept_multiple_files=False)

default_used = False
sheets = {}
if excel_file is not None:
    sheets = load_excel(excel_file)
else:
    # 로컬 기본 경로 자동 로드 시도
    try:
        with open(DEFAULT_EXCEL_PATH, "rb") as f:
            sheets = load_excel(f)
            default_used = True
            st.sidebar.info("기본 경로에서 엑셀을 불러왔습니다.")
    except Exception:
        st.sidebar.warning("엑셀을 업로드 해주세요.")

if not sheets:
    st.stop()

s_sunsu = find_sheet_case_insensitive(sheets, "선수금")
s_seongeup = find_sheet_case_insensitive(sheets, "선급금")

if s_sunsu is None or s_seongeup is None:
    st.error("시트 이름 '선수금', '선급금'을 찾지 못했습니다. 시트명을 확인해주세요.")
    st.stop()

df_sunsu = sheets[s_sunsu].copy()
df_seon = sheets[s_seongeup].copy()

# 중요 키 컬럼 가시화용 별칭/존재 체크
for df in (df_sunsu, df_seon):
    for old, new in [
        ("정산\n선수금\n고유번호", "정산선수금고유번호"),
        ("정산여부\n(O/X)", "정산여부"),
        ("고객명\n(드롭다운)", "고객명"),
        ("정산\n선수금\n고유번호", "정산선수금고유번호"),
        ("회수목표일정\n(YY/MM)", "회수목표일정(YY/MM)"),
        ("경과기간\n(개월)", "경과기간(개월)"),
    ]:
        if old in df.columns and new not in df.columns:
            df.rename(columns={old: new}, inplace=True)

# 금액 컬럼 생성(전표통화액/현지통화액 중 하나를 '금액'으로)
def unify_amount_col(df: pd.DataFrame) -> pd.DataFrame:
    if "금액" not in df.columns:
        df["금액"] = df.apply(choose_amount_row, axis=1)
    return df

df_sunsu = unify_amount_col(df_sunsu)
df_seon = unify_amount_col(df_seon)

# 날짜 파싱 컬럼
for df in (df_sunsu, df_seon):
    if "전기일" in df.columns:
        df["전기일_parsed"] = pd.to_datetime(df["전기일"], errors="coerce")

# -----------------------------
# 사이드바 - 매칭 옵션
# -----------------------------
st.sidebar.header("매칭 옵션")
date_half_life_days = st.sidebar.slider("일자 근접도 절반감쇠일(일)", min_value=15, max_value=180, value=90, step=15)
score_threshold = st.sidebar.slider("후보 표시 최소점수", min_value=0, max_value=100, value=40, step=5)
combo_tolerance = st.sidebar.slider("조합 합계 허용오차(±%)", min_value=1, max_value=20, value=5, step=1) / 100.0
combo_max_depth = st.sidebar.slider("조합 최대 항목 수", min_value=2, max_value=10, value=6, step=1)

st.sidebar.caption("※ 점수는 링크드ID>계약번호>금액/업체명/일자/텍스트 순으로 가중치가 적용됩니다.")

# -----------------------------
# 탭 구성
# -----------------------------
tab1, tab2, tab3 = st.tabs(["🔎 매칭 조회", "⚙️ 일괄 매칭 제안", "📊 대시보드"])

# -----------------------------
# 🔎 매칭 조회
# -----------------------------
with tab1:
    st.subheader("특정 선수금과 매칭되는 선급금 후보 조회")
    # 선수금 선택 셀렉트박스
    def sunsu_label(row: pd.Series) -> str:
        gid = str(row.get("고유넘버", ""))
        comp = str(row.get("업체명", ""))
        contract = str(row.get("계약번호", ""))
        date = row.get("전기일_parsed", row.get("전기일", ""))
        amt = row.get("금액", None)
        amt_str = f"{amt:,.0f}" if isinstance(amt, (int, float, np.number)) and not pd.isna(amt) else "-"
        dstr = ""
        if isinstance(date, pd.Timestamp):
            dstr = date.strftime("%Y-%m-%d")
        else:
            dstr = str(date) if date is not None else ""
        return f"[{gid}] {comp} | 계약:{contract} | 일자:{dstr} | 금액:{amt_str}"

    sunsu_options = df_sunsu.index.tolist()
    selectable = [(i, sunsu_label(df_sunsu.loc[i])) for i in sunsu_options]
    selected_idx = st.selectbox("선수금 선택", options=[i for i, _ in selectable], format_func=lambda i: dict(selectable)[i])

    if selected_idx is not None:
        target_row = df_sunsu.loc[selected_idx]

        # 개별 선급금과의 점수 계산
        today = datetime.now()
        scores = []
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

        cand_df = pd.DataFrame(scores).sort_values(by=["총점"], ascending=False).reset_index(drop=True)
        st.markdown("**단일 항목 후보(점수 순)**")
        st.dataframe(cand_df, use_container_width=True, height=400)

        # 1:N 조합 제안
        target_amt = target_row.get("금액", None)
        if isinstance(target_amt, (int, float, np.number)) and not pd.isna(target_amt):
            # 금액 필터링 강도 조절을 위한 후보 축소 (계약번호/업체명 근접 우선)
            seed = df_seon.copy()
            def contract_match(r): 
                return str(r.get("계약번호","")).strip() == str(target_row.get("계약번호","")).strip() if r.get("계약번호") is not None else False
            seed["name_sim"] = seed["업체명"].apply(lambda x: text_sim(x, target_row.get("업체명")) if "업체명" in seed.columns else 0.0)
            # 1차 필터: 계약번호 일치 혹은 업체명 유사도 >= 0.75
            seed = seed[(seed.apply(contract_match, axis=1)) | (seed["name_sim"] >= 0.75)].copy()
            seed["금액"] = seed["금액"].apply(to_number)
            # 금액이 0이거나 NaN 제거
            seed = seed[seed["금액"].notna() & (seed["금액"] != 0)]
            # 절대값이 target 대비 너무 큰 항목 제외
            seed = seed[seed["금액"].abs() <= abs(target_amt) * (1 + combo_tolerance)].copy()
            # 최대 80개로 제한(성능 보호)
            seed = seed.sort_values(by="금액", key=lambda s: (abs(target_amt - s))).head(80).reset_index(drop=True)

            if seed.empty:
                st.info("조합 탐색을 위한 후보가 충분하지 않아 조합 제안이 없습니다.")
            else:
                combos = propose_combinations(
                    target_amount=target_amt,
                    candidates=seed,
                    amount_col="금액",
                    max_depth=combo_max_depth,
                    max_nodes=800,
                    tolerance=combo_tolerance,
                )
                st.markdown("**1:N 조합 후보(오차·항목 수 기준 상위)**")
                if not combos:
                    st.info("조건에 맞는 조합이 발견되지 않았습니다. 허용오차(±%) 또는 후보 필터를 완화해보세요.")
                else:
                    # 상위 5개만 표로 변환
                    combo_tables = []
                    for rank, c in enumerate(combos[:5], start=1):
                        tbl = c["rows"].copy()
                        tbl = tbl.assign(_조합순위=rank)
                        combo_tables.append(tbl.assign(_합계=c["sum"], _갭=c["gap"]))
                    combo_df = pd.concat(combo_tables, ignore_index=True)
                    # 보기 편하게 선택 컬럼만 노출
                    show_cols = [col for col in ["_조합순위", "고유넘버", "계약번호", "업체명", "전기일", "금액", "텍스트", "_합계", "_갭"] if col in combo_df.columns]
                    st.dataframe(combo_df[show_cols], use_container_width=True, height=400)

                    # 다운로드
                    csv = combo_df.to_csv(index=False).encode("utf-8-sig")
                    st.download_button("조합 제안 CSV 다운로드", data=csv, file_name="match_combinations.csv", mime="text/csv")

        else:
            st.warning("선수금 금액이 확인되지 않아 조합 제안을 수행할 수 없습니다. '전표통화액' 또는 '현지통화액'을 확인해주세요.")

# -----------------------------
# ⚙️ 일괄 매칭 제안
# -----------------------------
with tab2:
    st.subheader("일괄 매칭 제안(Top-1 단일 후보)")
    max_rows = st.number_input("대상 선수금 수(상위 N행)", min_value=10, max_value=min(2000, len(df_sunsu)), value=min(200, len(df_sunsu)), step=10)
    today = datetime.now()
    batch_rows = []
    # 정산여부가 미정(O/X가 아닌)인 건 우선
    sunsu_order = df_sunsu.copy()
    if "정산여부" in sunsu_order.columns:
        sunsu_order["미정산"] = ~sunsu_order["정산여부"].astype(str).str.contains("O", na=False)
        sunsu_order = sunsu_order.sort_values(by=["미정산"], ascending=False)
    sunsu_order = sunsu_order.head(int(max_rows))

    for si, srow in sunsu_order.iterrows():
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

# -----------------------------
# 📊 대시보드
# -----------------------------
with tab3:
    st.subheader("요약 지표 & 시각화")

    # 매칭 상태 파악(정산여부, 링크드ID 존재 여부 활용)
    def is_matched_row(df: pd.DataFrame) -> pd.Series:
        cond = pd.Series(False, index=df.index)
        if "정산여부" in df.columns:
            cond = cond | df["정산여부"].astype(str).str.contains("O", na=False)
        if "정산선수금고유번호" in df.columns:
            cond = cond | df["정산선수금고유번호"].astype(str).str.strip().ne("")
        return cond

    sunsu_matched = is_matched_row(df_sunsu)
    seon_matched = is_matched_row(df_seon)

    kpi = st.columns(4)
    with kpi[0]:
        st.metric("선수금 건수", f"{len(df_sunsu):,}")
    with kpi[1]:
        st.metric("선수금 미정산 건수", f"{int((~sunsu_matched).sum()):,}")
    with kpi[2]:
        st.metric("선급금 건수", f"{len(df_seon):,}")
    with kpi[3]:
        st.metric("선급금 미정산 건수", f"{int((~seon_matched).sum()):,}")

    # 업체별 미정산 금액 상위
    def group_unsettled(df: pd.DataFrame, title: str):
        base = df.copy()
        base["is_unsettled"] = ~is_matched_row(base)
        base = base[base["is_unsettled"]]
        base["금액"] = base["금액"].apply(to_number)
        agg = base.groupby("업체명", dropna=False)["금액"].sum().reset_index().sort_values(by="금액", ascending=False).head(20)
        st.markdown(f"**{title} - 미정산 금액 상위 20 업체**")
        chart = alt.Chart(agg.dropna()).mark_bar().encode(
            x=alt.X("금액:Q", title="금액 합계"),
            y=alt.Y("업체명:N", sort="-x", title="업체명")
        ).properties(height=400)
        st.altair_chart(chart, use_container_width=True)
        st.dataframe(agg, use_container_width=True, height=300)

    c1, c2 = st.columns(2)
    with c1:
        group_unsettled(df_sunsu, "선수금")
    with c2:
        group_unsettled(df_seon, "선급금")

    # 경과기간(개월) 버킷
    def aging_chart(df: pd.DataFrame, title: str):
        col = "경과기간(개월)" if "경과기간(개월)" in df.columns else None
        if col is None:
            st.info(f"{title}: '경과기간(개월)' 컬럼이 없어 에이징 차트를 생략합니다.")
            return
        base = df.copy()
        base["금액"] = base["금액"].apply(to_number)
        base = base.dropna(subset=["금액"])
        # 버킷팅
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
        st.markdown(f"**{title} - 에이징(개월) 분포**")
        chart = alt.Chart(agg).mark_bar().encode(
            x=alt.X("버킷:N", sort=order, title="경과기간 버킷"),
            y=alt.Y("금액:Q", title="금액 합계")
        ).properties(height=300)
        st.altair_chart(chart, use_container_width=True)
        st.dataframe(agg, use_container_width=True, height=250)

    c3, c4 = st.columns(2)
    with c3:
        aging_chart(df_sunsu, "선수금")
    with c4:
        aging_chart(df_seon, "선급금")

    # 월별 추이(연도/월 혹은 전기일 기준)
    def monthly_trend(df: pd.DataFrame, title: str):
        base = df.copy()
        base["금액"] = base["금액"].apply(to_number)
        base = base.dropna(subset=["금액"])
        if "연도/월" in base.columns:
            base["연월"] = base["연도/월"].astype(str)
        elif "전기일_parsed" in base.columns:
            base["연월"] = base["전기일_parsed"].dt.strftime("%Y-%m")
        else:
            return
        agg = base.groupby("연월")["금액"].sum().reset_index()
        st.markdown(f"**{title} - 월별 금액 추이**")
        chart = alt.Chart(agg).mark_line(point=True).encode(
            x=alt.X("연월:N", sort=None, title="연월"),
            y=alt.Y("금액:Q", title="금액 합계")
        ).properties(height=300)
        st.altair_chart(chart, use_container_width=True)
        st.dataframe(agg, use_container_width=True, height=250)

    monthly_trend(df_sunsu, "선수금")
    monthly_trend(df_seon, "선급금")

st.caption("ⓘ 점수/규칙/차트는 현 데이터 구조에 맞춰 기본값으로 구성되었습니다. 필요 시 규칙 가중치, 조합 탐색 한도, 시트/컬럼명 매핑 등을 커스터마이즈해드립니다.")
