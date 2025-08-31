
# -*- coding: utf-8 -*-
from __future__ import annotations

import io
import json
import math
import os
import re
import hashlib
import zipfile
from calendar import monthrange
from difflib import SequenceMatcher
from typing import Dict, Optional, Tuple, List

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="선수·선급금 매칭 & 영업 대시보드", layout="wide", page_icon="📊")
DEFAULT_EXCEL_PATH = "./2025.07월말 선수선급금 현황_20250811.xlsx"
HISTORY_PATH = "./_history_snapshot.jsonl"

# -----------------------------
# 유틸
# -----------------------------
def norm_col(c: str) -> str:
    if not isinstance(c, str):
        return c
    return (
        c.replace("\\n", "").replace("\n", "").replace("\r", "").replace(" ", "").strip()
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
        "회수목표일정\n(YY/MM)": "정산목표일정(YY/MM)",
        "경과기간\n(개월)": "경과기간(개월)",
        "담당팀\n(변경시)": "담당팀_변경시",
        "영업담당\n(변경시)": "영업담당_변경시",
        "연락이력": "연락이력",
        "연락 이력": "연락이력",
        "진행\n현황": "진행현황",
        "진행 현황": "진행현황",
        "정산진행현황": "정산진행현황",
    }
    for old, new in mapping.items():
        if old in df.columns and new not in df.columns:
            df.rename(columns={old: new}, inplace=True)

    if "영업담당" in df.columns:
        df["영업담당_표준"] = df["영업담당"]
    elif "영업담당_변경시" in df.columns:
        df["영업담당_표준"] = df["영업담당_변경시"]
    else:
        df["영업담당_표준"] = None

    for c in ["진행현황", "연락이력", "정산진행현황"]:
        if c not in df.columns:
            df[c] = None

    return df

def parse_due_yy_mm(val) -> Optional[pd.Timestamp]:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return None
    s = str(val).strip()
    if not s:
        return None
    m = re.match(r"^\s*(\d{2})[./\- ]?(\d{1,2})\s*$", s)
    if not m:
        return None
    yy = int(m.group(1)); mm = int(m.group(2))
    year = 2000 + yy if yy <= 79 else 1900 + yy
    mm = max(1, min(12, mm))
    last_day = monthrange(year, mm)[1]
    return pd.Timestamp(year=year, month=mm, day=last_day)

SETTLED_FLAG_TRUE = {"O","o","Y","y","1","True","TRUE","true"}
SETTLED_FLAG_FALSE = {"X","x","N","n","0","False","FALSE","false"}
POSITIVE_PATTERNS = [
    r"(?<!미)완료(?![가-힣])",  # '미완료' 제외
    r"입금\s*완료",
    r"수납\s*완료",
    r"\bsettled\b|\bcleared\b|\bclosed\b"
]
NEGATIVE_HINTS = [
    "미완료","미정산","미입금","연체","보류","대기","예정","진행","진행중","협의","검토","요청","확인중"
]

def looks_settled(row: pd.Series) -> bool:
    v1 = str(row.get("정산여부", "")).strip()
    if v1 in SETTLED_FLAG_TRUE:
        return True
    if v1 in SETTLED_FLAG_FALSE:
        return False

    v2 = str(row.get("정산진행현황", "")).strip().lower()
    v3 = str(row.get("진행현황", "")).strip().lower()
    t = f"{v2} {v3}".strip()

    if any(h in t for h in [h.lower() for h in NEGATIVE_HINTS]):
        return False
    for pat in POSITIVE_PATTERNS:
        if re.search(pat, t):
            return True
    return False

def add_common_fields(df: pd.DataFrame) -> pd.DataFrame:
    if "금액" not in df.columns:
        df["금액"] = df.apply(choose_amount_row, axis=1)
    if "전기일" in df.columns and "전기일_parsed" not in df.columns:
        df["전기일_parsed"] = pd.to_datetime(df["전기일"], errors="coerce")
    if "회수목표일자" in df.columns:
        df["회수목표일자"] = pd.to_datetime(df["회수목표일자"], errors="coerce")
    elif "정산목표일정(YY/MM)" in df.columns:
        df["회수목표일자"] = df["정산목표일정(YY/MM)"].apply(parse_due_yy_mm)
    else:
        df["회수목표일자"] = pd.NaT

    df["is_settled"] = df.apply(looks_settled, axis=1)
    df["금액_num"] = df["금액"].apply(to_number)
    for c in ["진행현황","연락이력","정산진행현황"]:
        if c not in df.columns: df[c] = None
    return df

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
    if (amt_sun is not None) and (amt_seon is not None) and (amt_sun != 0):
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
# 공통 표현: 금액 포맷 (정수 원, 콤마) - 문자열 렌더링로 오류 회피
# -----------------------------
def _money_like(col: str) -> bool:
    return ("금액" in col) or ("합계" in col) or (col in ["금액_num", "금액"])

def _fmt_won(v):
    if pd.isna(v):
        return ""
    try:
        return f"{int(round(float(v))):,}"
    except Exception:
        return str(v)

def display_df(df: pd.DataFrame, height: int = 420):
    df2 = df.copy()
    money_cols = [c for c in df2.columns if _money_like(c)]
    for c in money_cols:
        df2[c] = df2[c].apply(_fmt_won)
    st.dataframe(df2, width='stretch', height=height)

# -----------------------------
# 데이터 로드 & 권한
# -----------------------------
st.sidebar.header("데이터")
excel_file = st.sidebar.file_uploader("엑셀 업로드 (.xlsx)", type=["xlsx"], accept_multiple_files=False)

sheets = {}
try:
    if excel_file is not None:
        sheets = load_excel(excel_file)
    else:
        with open(DEFAULT_EXCEL_PATH, "rb") as f:
            sheets = load_excel(f)
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

# 권한뷰
st.sidebar.header("권한/담당자")
owners_all = sorted(set([x for x in df_sunsu["영업담당_표준"].dropna().unique().tolist() + df_seon["영업담당_표준"].dropna().unique().tolist()]))
my_only = st.sidebar.checkbox("본인 건만 보기", value=False)
my_name = st.sidebar.selectbox("내 담당자명", options=["(선택)"] + owners_all, index=0)

def apply_my_view(df: pd.DataFrame) -> pd.DataFrame:
    if my_only and my_name and my_name != "(선택)":
        return df[df["영업담당_표준"] == my_name].copy()
    return df

df_sunsu = apply_my_view(df_sunsu)
df_seon = apply_my_view(df_seon)

# 공통 필터
st.sidebar.header("공통 필터")
owner_multi = st.sidebar.multiselect("영업담당 선택(복수)", options=owners_all, default=[])
only_unsettled = st.sidebar.checkbox("미정산만 보기", value=False)
only_overdue = st.sidebar.checkbox("연체만 보기(현재 기준)", value=False)

def apply_owner_status_filter(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if owner_multi:
        out = out[out["영업담당_표준"].isin(owner_multi)]
    if only_unsettled:
        out = out[~out["is_settled"]]
    now = pd.Timestamp.now()
    if only_overdue:
        due = out["회수목표일자"] if "회수목표일자" in out.columns else pd.Series(pd.NaT, index=out.index)
        out = out[(~out["is_settled"]) & (due.notna()) & (due < now)]
    return out

df_sunsu_f = apply_owner_status_filter(df_sunsu)
df_seon_f = apply_owner_status_filter(df_seon)

# -----------------------------
# 상태 정의 & 파이프라인
# -----------------------------
def enrich_status_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    base = df.copy()
    now = pd.Timestamp.now()
    this_month = now.to_period("M")

    base["상태"] = "정보없음"
    base.loc[base["is_settled"] == True, "상태"] = "처리완료"
    due = base["회수목표일자"] if "회수목표일자" in base.columns else pd.Series(pd.NaT, index=base.index)
    cond_un = (base["is_settled"] == False)
    has_due = due.notna()

    base.loc[cond_un & has_due & (due < now), "상태"] = "연체"
    base.loc[cond_un & has_due & (due.dt.to_period("M") == this_month), "상태"] = "당월예정"
    base.loc[cond_un & (~has_due), "상태"] = "기한미설정"
    base.loc[cond_un & has_due & (due.dt.to_period("M") > this_month), "상태"] = "향후예정"

    def map_progress(*values: str) -> Optional[str]:
        t = " ".join([str(x).strip().lower() for x in values if isinstance(x, str)])
        if not t: return None
        if any(k in t for k in ["회수중", "수금중", "징수중", "collection"]): return "회수중"
        if any(k in t for k in ["협의", "논의", "컨펌", "조율"]): return "협의중"
        if any(k in t for k in ["보류", "hold", "대기"]): return "보류"
        if any(k in t for k in ["소송", "분쟁", "법무"]): return "분쟁/소송"
        if any(k in t for k in ["무응답", "연락두절"]): return "무응답"
        if any(re.search(p, t) for p in POSITIVE_PATTERNS) and not any(h in t for h in [h.lower() for h in NEGATIVE_HINTS]): return "완료"
        return None

    if "진행현황" not in base.columns: base["진행현황"] = None
    if "정산진행현황" not in base.columns: base["정산진행현황"] = None
    base["세부상태"] = base.apply(lambda r: map_progress(r.get("진행현황"), r.get("정산진행현황")), axis=1)
    base["상태(세부)"] = base["상태"]
    mask = base["세부상태"].notna()
    base.loc[mask, "상태(세부)"] = base.loc[mask, "상태"] + "-" + base.loc[mask, "세부상태"]

    stage = pd.Series("", index=base.index, dtype="object")
    if "회수목표일자" in base.columns:
        ddays = (due.dt.normalize() - now.normalize()).dt.days
        stage[(base["상태"] == "당월예정") & (ddays <= -1)] = "지연"
        stage[(base["상태"] == "당월예정") & (ddays == 0)] = "당일"
        stage[(base["상태"] == "당월예정") & (ddays.between(1, 3))] = "D-3"
        stage[(base["상태"] == "당월예정") & (ddays.between(4, 7))] = "D-7"
        stage[(base["상태"] == "당월예정") & (ddays >= 8)] = "당월(8일+)"
    base["파이프라인"] = stage.where(stage != "", other=None)

    if "연락이력" not in base.columns: base["연락이력"] = None

    return base

sunsu_s = enrich_status_pipeline(df_sunsu_f)
seon_s = enrich_status_pipeline(df_seon_f)

# -----------------------------
# AI 히스토리 추적 (간단)
# -----------------------------
KEY_COLS_FOR_HISTORY = ["구분","키","영업담당_표준","업체명","계약번호","고유넘버","전기일_parsed","회수목표일자","상태","상태(세부)","파이프라인","정산진행현황","정산여부","연락이력","비고","금액_num"]

def make_key(row: pd.Series) -> str:
    gid = str(row.get("고유넘버", "")).strip()
    if gid:
        return gid
    parts = [
        str(row.get("전표번호","")).strip(),
        str(row.get("계약번호","")).strip(),
        str(row.get("업체명","")).strip(),
        str(row.get("전기일_parsed","")).strip()
    ]
    base = "|".join(parts)
    return hashlib.sha1(base.encode("utf-8")).hexdigest()[:12]

def build_current_snapshot() -> pd.DataFrame:
    a = sunsu_s.assign(구분="선수금").copy()
    b = seon_s.assign(구분="선급금").copy()
    allv = pd.concat([a, b], ignore_index=True, sort=False)
    allv["키"] = allv.apply(make_key, axis=1)
    return allv

def load_last_snapshot() -> Optional[pd.DataFrame]:
    if not os.path.exists(HISTORY_PATH):
        return None
    try:
        records = [json.loads(line) for line in open(HISTORY_PATH, "r", encoding="utf-8")]
        if not records:
            return None
        last = records[-1]
        return pd.DataFrame(last["data"])
    except Exception:
        return None

def save_snapshot(df: pd.DataFrame):
    rec = {
        "ts": pd.Timestamp.now().isoformat(),
        "data": df[KEY_COLS_FOR_HISTORY].fillna("").astype(str).to_dict(orient="records"),
    }
    with open(HISTORY_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def diff_snapshots(prev: Optional[pd.DataFrame], curr: pd.DataFrame) -> pd.DataFrame:
    if prev is None or prev.empty:
        return pd.DataFrame(columns=["키","변경항목","이전값","현재값","구분","업체명","계약번호","영업담당_표준","변경시각"])
    prev_i = prev.set_index("키")
    curr_i = curr.set_index("키")
    changed_rows = []
    common_keys = set(prev_i.index).intersection(set(curr_i.index))
    cols_to_check = [c for c in KEY_COLS_FOR_HISTORY if c not in ["키"]]
    for k in common_keys:
        p = prev_i.loc[k]; c = curr_i.loc[k]
        for col in cols_to_check:
            pv = str(p.get(col, "")); cv = str(c.get(col, ""))
            if pv != cv:
                changed_rows.append({
                    "키": k, "변경항목": col, "이전값": pv, "현재값": cv,
                    "구분": c.get("구분",""), "업체명": c.get("업체명",""),
                    "계약번호": c.get("계약번호",""), "영업담당_표준": c.get("영업담당_표준",""),
                    "변경시각": pd.Timestamp.now()
                })
    return pd.DataFrame(changed_rows)

# -----------------------------
# 탭
# -----------------------------
tab0, tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "👤 영업 대시보드", "🔎 매칭 조회", "⚙️ 일괄 매칭", "📊 요약 대시보드",
    "🧭 고급 검색", "🗂 3단 그리드", "🗓 주간 계획표", "📜 히스토리"
])

# -----------------------------
# 👤 영업 대시보드
# -----------------------------
with tab0:
    st.subheader("영업 담당자 관점 KPI & 상태 보드 (현재 시점 기준)")
    now = pd.Timestamp.now()
    sm = pd.Timestamp(now.year, now.month, 1)

    def kpi_block(title: str, df: pd.DataFrame):
        if df.empty:
            st.info(f"{title}: 데이터 없음")
            return
        due = df["회수목표일자"] if "회수목표일자" in df.columns else pd.Series(pd.NaT, index=df.index)
        total = len(df)
        done = int(df["is_settled"].sum())
        overdue = int(((~df["is_settled"]) & due.notna() & (due < now)).sum())
        due_this = int(((~df["is_settled"]) & due.notna() & (due.dt.to_period("M") == sm.to_period("M"))).sum())
        amt_un = df.loc[~df["is_settled"], "금액_num"].sum(skipna=True)
        cols = st.columns(5)
        with cols[0]: st.metric(f"{title} 건수", f"{total:,}")
        with cols[1]: st.metric("처리완료", f"{done:,}")
        with cols[2]: st.metric("연체", f"{overdue:,}")
        with cols[3]: st.metric("당월예정", f"{due_this:,}")
        with cols[4]: st.metric("미정산금액", f"{amt_un:,.0f}" if pd.notna(amt_un) else "-")

    st.markdown("### 선수금"); kpi_block("선수금", sunsu_s)
    st.markdown("### 선급금"); kpi_block("선급금", seon_s)

    def status_chart(df: pd.DataFrame, title: str):
        if df.empty: st.info(f"{title}: 데이터 없음"); return
        agg = df.dropna(subset=["금액_num"]).groupby("상태")["금액_num"].sum().reset_index()
        chart = alt.Chart(agg).mark_bar().encode(x="상태:N", y=alt.Y("금액_num:Q", title="금액 합계")).properties(height=280, title=f"{title} - 상태별 금액")
        st.altair_chart(chart, use_container_width=True)
        display_df(agg.rename(columns={"금액_num": "금액합계"}), height=240)

    c1, c2 = st.columns(2)
    with c1: status_chart(sunsu_s, "선수금")
    with c2: status_chart(seon_s, "선급금")

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

        options = df_sunsu_f.index.tolist()
        selectable = [(i, sunsu_label(df_sunsu_f.loc[i])) for i in options]
        selected_idx = st.selectbox("선수금 선택", options=[i for i, _ in selectable], format_func=lambda i: dict(selectable)[i])

        if selected_idx is not None:
            target_row = df_sunsu_f.loc[selected_idx]
            scores: List[dict] = []
            for i, row in df_seon_f.iterrows():
                total, parts = calc_match_score(target_row, row, date_half_life_days=date_half_life_days)
                if total >= score_threshold:
                    scores.append({
                        "선급_index": i, "총점": round(total, 2),
                        **{f"점수:{k}": round(v, 2) for k, v in parts.items()},
                        "계약번호": row.get("계약번호"), "업체명": row.get("업체명"),
                        "전기일": row.get("전기일_parsed", row.get("전기일")), "금액": row.get("금액_num"),
                        "정산선수금고유번호": row.get("정산선수금고유번호"), "텍스트": row.get("텍스트"),
                        "고유넘버": row.get("고유넘버"), "영업담당": row.get("영업담당_표준"),
                    })
            if not scores:
                st.info("후보가 없습니다. 점수 임계값 또는 필터를 조정해 주세요.")
            else:
                cand_df = pd.DataFrame(scores).sort_values(by="총점", ascending=False).reset_index(drop=True)
                display_df(cand_df, height=430)

# -----------------------------
# ⚙️ 일괄 매칭
# -----------------------------
with tab2:
    st.subheader("일괄 매칭 제안(Top-1)")
    score_threshold2 = st.slider("후보 표시 최소점수", 0, 100, 40, 5, key="b_th")
    limit = st.number_input("대상 선수금 수", min_value=10, max_value=max(10, len(df_sunsu_f) if len(df_sunsu_f)>0 else 10), value=min(200, len(df_sunsu_f) if len(df_sunsu_f)>0 else 10), step=10)
    if df_sunsu_f.empty or df_seon_f.empty:
        st.info("데이터가 비어 있어 일괄 제안을 생략합니다.")
    else:
        rows = []
        for si, srow in df_sunsu_f.head(int(limit)).iterrows():
            best_score = -1.0; best_idx = None
            for ei, erow in df_seon_f.iterrows():
                total, _ = calc_match_score(srow, erow, date_half_life_days=90)
                if total > best_score:
                    best_score = total; best_idx = ei
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
            display_df(dfb, height=450)
            st.download_button("CSV 다운로드", dfb.to_csv(index=False).encode("utf-8-sig"), file_name="batch_match_suggestions.csv", mime="text/csv")

# -----------------------------
# 📊 요약 대시보드
# -----------------------------
with tab3:
    st.subheader("요약 지표 & 시각화")
    def group_unsettled(df: pd.DataFrame, title: str):
        base = df[~df["is_settled"]].copy().dropna(subset=["금액_num"])
        if base.empty: st.info(f"{title}: 미정산 없음"); return
        agg = base.groupby("업체명", dropna=False)["금액_num"].sum().reset_index().sort_values(by="금액_num", ascending=False).head(20)
        st.markdown(f"**{title} - 미정산 금액 상위 20 업체**")
        chart = alt.Chart(agg.dropna()).mark_bar().encode(x=alt.X("금액_num:Q", title="금액 합계"), y=alt.Y("업체명:N", sort="-x", title="업체명")).properties(height=360)
        st.altair_chart(chart, use_container_width=True)
        display_df(agg.rename(columns={"금액_num": "금액합계"}), height=260)

    c1, c2 = st.columns(2)
    with c1: group_unsettled(sunsu_s, "선수금")
    with c2: group_unsettled(seon_s, "선급금")

    def aging_chart(df: pd.DataFrame, title: str):
        col = "경과기간(개월)" if "경과기간(개월)" in df.columns else None
        if col is None or df.empty: st.info(f"{title}: '경과기간(개월)' 컬럼이 없어 에이징 차트를 생략합니다."); return
        base = df.dropna(subset=["금액_num"]).copy()
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
        agg["버킷"] = pd.Categorical(agg["버킷"], categories=order, ordered=True); agg = agg.sort_values("버킷")
        chart = alt.Chart(agg).mark_bar().encode(x=alt.X("버킷:N", sort=order, title="경과기간 버킷"), y=alt.Y("금액_num:Q", title="금액 합계")).properties(height=300)
        st.markdown(f"**{title} - 에이징(개월) 분포**")
        st.altair_chart(chart, use_container_width=True)
        display_df(agg.rename(columns={"금액_num": "금액합계"}), height=240)

    c3, c4 = st.columns(2)
    with c3: aging_chart(sunsu_s, "선수금")
    with c4: aging_chart(seon_s, "선급금")

# -----------------------------
# 🧭 고급 검색
# -----------------------------
with tab4:
    st.subheader("강화된 검색(선수금/선급금 통합)")
    sunsu_view = sunsu_s.copy(); sunsu_view["구분"] = "선수금"
    seon_view = seon_s.copy();   seon_view["구분"] = "선급금"
    all_view = pd.concat([sunsu_view, seon_view], ignore_index=True, sort=False)

    col1, col2, col3 = st.columns(3)
    with col1: kw = st.text_input("키워드(업체/계약/텍스트/전표번호/금형/고유넘버)", value="")
    with col2: min_amt = st.number_input("최소 금액", value=0, step=1000)
    with col3: max_amt = st.number_input("최대 금액(0=제한없음)", value=0, step=1000)

    col4, col5, col6 = st.columns(3)
    with col4: start_date = st.date_input("전기일 From", value=None)
    with col5: end_date = st.date_input("전기일 To", value=None)
    with col6: only_due_this_month = st.checkbox("당월예정만", value=False)

    col7, col8 = st.columns(2)
    with col7: status_sel = st.multiselect("상태(세부) 선택", options=sorted(all_view["상태(세부)"].dropna().unique().tolist()), default=[])
    with col8: pipeline_sel = st.multiselect("파이프라인 단계 선택", options=[x for x in ["지연","당일","D-3","D-7","당월(8일+)"] if x in all_view["파이프라인"].dropna().unique()], default=[])

    res = all_view.copy()
    res = res[(res["금액_num"].fillna(0) >= (min_amt or 0))]
    if max_amt and max_amt > 0: res = res[(res["금액_num"].fillna(0) <= max_amt)]
    if start_date: res = res[res["전기일_parsed"].fillna(pd.Timestamp("1900-01-01")) >= pd.Timestamp(start_date)]
    if end_date:   res = res[res["전기일_parsed"].fillna(pd.Timestamp("2999-12-31")) <= pd.Timestamp(end_date)]
    if only_due_this_month:
        now = pd.Timestamp.now()
        due = res["회수목표일자"] if "회수목표일자" in res.columns else pd.Series(pd.NaT, index=res.index)
        res = res[(~res["is_settled"]) & (due.notna()) & (due.dt.to_period("M") == now.to_period("M"))]
    if status_sel:   res = res[res["상태(세부)"].isin(status_sel)]
    if pipeline_sel: res = res[res["파이프라인"].isin(pipeline_sel)]

    if kw.strip():
        k = kw.strip().upper()
        def contains_any(s):
            if s is None or (isinstance(s, float) and math.isnan(s)): return False
            return k in str(s).upper()
        search_cols = [c for c in ["업체명","계약번호","텍스트","전표번호","금형마스터","금형마스터내역","고유넘버"] if c in res.columns]
        mask = False
        for c in search_cols: mask = mask | res[c].apply(contains_any)
        res = res[mask]

    sort_cols = [c for c in ["구분","is_settled","회수목표일자","전기일_parsed","금액_num"] if c in res.columns]
    if sort_cols: res = res.sort_values(by=sort_cols, ascending=[True, True, True, True, False]).reset_index(drop=True)
    show_cols = [c for c in ["구분","영업담당_표준","업체명","계약번호","고유넘버","전기일_parsed","회수목표일자","파이프라인","상태(세부)","금액_num","정산선수금고유번호","정산여부","정산진행현황","진행현황","연락이력","텍스트","비고"] if c in res.columns]
    display_df(res[show_cols], height=520)
    st.download_button("검색결과 CSV 다운로드", res[show_cols].to_csv(index=False).encode("utf-8-sig"), file_name="search_results.csv", mime="text/csv")

# -----------------------------
# 🗂 3단 그리드
# -----------------------------
with tab5:
    st.subheader("담당자 → 고객 → 계약 3단 그리드")
    all3 = pd.concat([sunsu_s.assign(구분="선수금"), seon_s.assign(구분="선급금")], ignore_index=True, sort=False)
    if all3.empty:
        st.info("데이터 없음")
    else:
        owners = sorted(all3["영업담당_표준"].dropna().unique().tolist())
        for owner in owners:
            with st.expander(f"담당자: {owner}"):
                sub = all3[all3["영업담당_표준"] == owner].copy()
                customers = sorted(sub["업체명"].dropna().unique().tolist())
                for cust in customers:
                    st.markdown(f"**고객: {cust}**")
                    sub2 = sub[sub["업체명"] == cust].copy()
                    cols = [c for c in ["계약번호","구분","전기일_parsed","회수목표일자","파이프라인","상태(세부)","금액_num","진행현황","정산진행현황","연락이력","텍스트","비고"] if c in sub2.columns]
                    display_df(sub2[cols], height=240)

# -----------------------------
# 🗓 주간 계획표
# -----------------------------
with tab6:
    st.subheader("주간 회수 계획표 (미정산 + 이번주 기한/지연 포함)")
    now = pd.Timestamp.now().normalize()
    week_start = now - pd.Timedelta(days=now.weekday())
    week_end = week_start + pd.Timedelta(days=6)

    def weekly_plan(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty: return df
        if "회수목표일자" not in df.columns:
            return df.iloc[0:0].copy()
        due = df["회수목표일자"]
        cond = (~df["is_settled"]) & (due.notna()) & ((due < week_start) | ((due >= week_start) & (due <= week_end)))
        out = df[cond].copy()
        weekday = out["회수목표일자"].dt.weekday
        names = {0:"월",1:"화",2:"수",3:"목",4:"금",5:"토",6:"일"}
        out["요일"] = weekday.map(names)
        return out

    week_all = pd.concat([weekly_plan(sunsu_s.assign(구분="선수금")), weekly_plan(seon_s.assign(구분="선급금"))], ignore_index=True, sort=False)
    if week_all.empty:
        st.info("이번 주 계획 대상이 없습니다.")
    else:
        show_cols = [c for c in ["요일","구분","영업담당_표준","업체명","계약번호","고유넘버","회수목표일자","파이프라인","상태(세부)","금액_num","연락이력","텍스트","비고"] if c in week_all.columns]
        week_all = week_all.sort_values(by=["요일","영업담당_표준","업체명","계약번호"])
        display_df(week_all[show_cols], height=520)
        st.download_button("주간 계획표 CSV 다운로드", week_all[show_cols].to_csv(index=False).encode("utf-8-sig"), file_name="weekly_collection_plan.csv", mime="text/csv")

# -----------------------------
# 📜 히스토리
# -----------------------------
with tab7:
    st.subheader("📜 변경 히스토리")
    curr = build_current_snapshot()
    prev = load_last_snapshot()
    diffs = diff_snapshots(prev, curr)
    if diffs.empty: st.info("직전 스냅샷 대비 변경 없음");
    else: display_df(diffs[["변경시각","구분","영업담당_표준","업체명","계약번호","키","변경항목","이전값","현재값"]], height=360)
    if st.button("현재 상태 스냅샷 저장"):
        save_snapshot(curr)
        st.success("스냅샷을 저장했습니다. 다음 실행부터 변경내역이 비교됩니다.")

# -----------------------------
# 📣 알림 리포트
# -----------------------------
st.markdown("---")
st.subheader("📣 알림 리포트: 영업담당별 당월예정/연체 목록")

def build_alert_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    for needed in ["is_settled", "상태"]:
        if needed not in df.columns: return df.iloc[0:0].copy()
    base = df[(~df["is_settled"]) & (df["상태"].isin(["연체","당월예정"]))].copy()
    desired = ["영업담당_표준","업체명","계약번호","고유넘버","상태","파이프라인","회수목표일자","금액_num","진행현황","정산진행현황","연락이력","텍스트","비고","구분"]
    cols = [c for c in desired if c in base.columns]
    base = base[cols]
    return base

alert_all = pd.concat([build_alert_df(sunsu_s.assign(구분="선수금")), build_alert_df(seon_s.assign(구분="선급금"))], ignore_index=True, sort=False)
if alert_all.empty:
    st.info("알림 대상(당월예정/연체)이 없습니다.")
else:
    owners = sorted(alert_all["영업담당_표준"].dropna().unique().tolist()) if "영업담당_표준" in alert_all.columns else []
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for o in owners:
            sub = alert_all[alert_all["영업담당_표준"] == o].copy()
            if sub.empty: continue
            csv_bytes = sub.to_csv(index=False).encode("utf-8-sig")
            zf.writestr(f"{o}_알림대상.csv", csv_bytes)
    mem.seek(0)
    st.download_button("담당자별 알림 CSV(zip) 다운로드", data=mem, file_name="alerts_by_owner.zip", mime="application/zip")

st.caption("완료 판정은 정산여부 O/Y/1 또는 정산/진행 현황의 명확한 '완료' 신호가 있고 부정 키워드(미완료/예정/진행중 등)가 없을 때만 True입니다. 표의 금액은 화면에서만 콤마 문자열로 표시되며, 내부 계산/CSV는 숫자를 사용합니다.")
