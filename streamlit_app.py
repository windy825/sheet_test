# app_v4.py — Autodetect & Robust Matching
# ------------------------------------------------------------------
# Run:
#   pip install streamlit pandas openpyxl xlrd xlsxwriter
#   streamlit run app_v4.py
# ------------------------------------------------------------------

import io, re, math, itertools, typing as t
import pandas as pd
import streamlit as st

st.set_page_config(page_title="선수/선급금 계약별 대시보드 (v4)", layout="wide")
st.title("📊 선수/선급금 계약별 대시보드 — v4 (Autodetect)")
st.caption("엑셀 업로드 → 자동 시트/컬럼 탐지 → 집계/검색 → 교차검증 매칭")

# ===================== Utilities =====================
_non_digit = re.compile(r"[^0-9\-\.\(\)]+")  # allow parentheses too

def to_float(x: t.Any) -> float:
    """Parse numbers like 1,234, (1,234), 1 234, 1.234,00"""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return 0.0
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip().replace("\u00A0", " ")  # nbsp
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return 0.0
    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg = True
    s = _non_digit.sub("", s)
    if s in {"", "-", "."}:
        return 0.0
    try:
        val = float(s.replace(",", ""))
    except Exception:
        try:
            # EU style: 1.234,56 -> 1234.56
            if "," in s and s.count(",") == 1 and "." in s:
                s2 = s.replace(".", "").replace(",", ".")
                val = float(s2)
            else:
                val = float(s)
        except Exception:
            return 0.0
    return -val if neg else val

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

STANDARD_COLS = ["contract_id","direction","amount","date","party","owner","status","note","overdue_flag"]
CONTRACT_CANDS = ["계약", "프로젝트", "PJT", "금형", "고유", "Contract", "Project", "Code", "ID"]
AMOUNT_CANDS  = ["금액", "선수", "선급", "합계", "Amount", "금"]
DATE_CANDS    = ["일자", "청구", "지급", "납기", "요청", "등록", "기준", "date", "날짜"]
PARTY_CANDS   = ["업체", "거래처", "고객", "상대", "회사", "vendor", "customer"]
OWNER_CANDS   = ["담당", "PM", "부서", "owner"]
STATUS_CANDS  = ["진행", "상태", "정산", "status"]
NOTE_CANDS    = ["비고", "메모", "특이", "코멘트", "note"]
OD_CANDS      = ["경과", "연체", "overdue"]

def first_col_like(df: pd.DataFrame, keys: t.List[str]) -> t.Optional[str]:
    cols = list(df.columns)
    for c in cols:
        low = str(c).lower()
        if any(k.lower() in low for k in keys):
            return c
    return None

def read_sheet_by_keywords(excel: pd.ExcelFile, keywords: t.List[str]) -> pd.DataFrame:
    def _norm(s: str) -> str: return re.sub(r"\s+","",s.lower())
    for name in excel.sheet_names:
        nm = _norm(name)
        if any(_norm(k) in nm for k in keywords):
            try:
                return normalize_columns(pd.read_excel(excel, sheet_name=name, dtype=str))
            except Exception:
                pass
    return pd.DataFrame()

def standardize(df: pd.DataFrame, direction: str, strict_filter: bool) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=STANDARD_COLS)
    df = normalize_columns(df)

    c_contract = first_col_like(df, CONTRACT_CANDS)
    c_amount   = first_col_like(df, AMOUNT_CANDS)
    c_date     = first_col_like(df, DATE_CANDS)
    c_party    = first_col_like(df, PARTY_CANDS)
    c_owner    = first_col_like(df, OWNER_CANDS)
    c_status   = first_col_like(df, STATUS_CANDS)
    c_note     = first_col_like(df, NOTE_CANDS)
    c_overdue  = first_col_like(df, OD_CANDS)

    out = pd.DataFrame()
    out["contract_id"] = df[c_contract].astype(str).str.strip() if c_contract in df.columns else ""
    out["direction"]   = direction
    out["amount"]      = df[c_amount].apply(to_float) if c_amount in df.columns else 0.0
    out["date"]        = pd.to_datetime(df[c_date], errors="coerce") if c_date in df.columns else pd.NaT
    out["party"]       = df[c_party].astype(str).str.strip() if c_party in df.columns else ""
    out["owner"]       = df[c_owner].astype(str).str.strip() if c_owner in df.columns else ""
    out["status"]      = df[c_status].astype(str).str.strip() if c_status in df.columns else ""
    out["note"]        = df[c_note].astype(str).str.strip() if c_note in df.columns else ""

    if c_overdue in df.columns:
        od = df[c_overdue].astype(str).str.strip().str.lower()
        out["overdue_flag"] = od.isin(["y","yes","true","1","o"]) | od.str.contains("경과|연체|over", na=False)
    else:
        out["overdue_flag"] = False

    if strict_filter:
        out = out[(out["amount"] != 0) & (out["contract_id"].astype(str).str.strip() != "")]
    return out[STANDARD_COLS]

# ===================== Autodetect across all sheets =====================
def autodetect_frames(excel: pd.ExcelFile, strict_filter: bool):
    receipts_list, advances_list = [], []

    # First, try named sheets
    r_named = read_sheet_by_keywords(excel, ["선수금","receipt","advance_in"])
    a_named = read_sheet_by_keywords(excel, ["선급금","prepay","advance_out"])
    if not r_named.empty:
        receipts_list.append(standardize(r_named, "선수금", strict_filter))
    if not a_named.empty:
        advances_list.append(standardize(a_named, "선급금", strict_filter))

    # Then, scan all sheets heuristically
    for name in excel.sheet_names:
        try:
            raw = normalize_columns(pd.read_excel(excel, sheet_name=name, dtype=str))
        except Exception:
            continue

        # Heuristic: decide direction by sheet name
        low = name.lower()
        if any(k in low for k in ["선수", "receipt", "입금", "customer"]):
            receipts_list.append(standardize(raw, "선수금", strict_filter))
            continue
        if any(k in low for k in ["선급", "prepay", "출금", "협력", "vendor"]):
            advances_list.append(standardize(raw, "선급금", strict_filter))
            continue

        # Otherwise infer by content
        party_col = first_col_like(raw, PARTY_CANDS) or ""
        party_vals = "|".join(str(x) for x in raw.get(party_col, [])[:50]).lower()
        if any(k in party_vals for k in ["고객","buyer","client"]):
            receipts_list.append(standardize(raw, "선수금", strict_filter))
        elif any(k in party_vals for k in ["협력","vendor","supplier"]):
            advances_list.append(standardize(raw, "선급금", strict_filter))

    rec = pd.concat([df for df in receipts_list if not df.empty], ignore_index=True) if receipts_list else pd.DataFrame(columns=STANDARD_COLS)
    adv = pd.concat([df for df in advances_list if not df.empty], ignore_index=True) if advances_list else pd.DataFrame(columns=STANDARD_COLS)
    return rec, adv

# ===================== Cache =====================
@st.cache_data(show_spinner=True)
def load_data(file_bytes: bytes, strict_filter: bool, synth_contract: bool):
    # Choose engine by signature
    bio = io.BytesIO(file_bytes)
    header = bio.getvalue()[:2]
    try:
        if header == b"PK":
            excel = pd.ExcelFile(bio, engine="openpyxl")
        else:
            excel = pd.ExcelFile(bio, engine="xlrd")
    except Exception as e:
        st.error(f"엑셀 열기 실패: {e}")
        empty = pd.DataFrame(columns=STANDARD_COLS)
        return empty, pd.DataFrame(), empty, empty

    # Try named sheets first
    df_r_raw = read_sheet_by_keywords(excel, ["선수금"])
    df_a_raw = read_sheet_by_keywords(excel, ["선급금"])

    df_r = standardize(df_r_raw, "선수금", strict_filter)
    df_a = standardize(df_a_raw, "선급금", strict_filter)

    # If nothing detected, autodetect across all sheets
    if df_r.empty and df_a.empty:
        df_r, df_a = autodetect_frames(excel, strict_filter)

    # Synthesize contract_id if missing and option is enabled
    if synth_contract:
        def synth(df: pd.DataFrame) -> pd.DataFrame:
            miss = df["contract_id"].fillna("").astype(str).str.strip()== ""
            if miss.any():
                df = df.copy()
                reparty = df["party"].fillna("").astype(str).str.strip()
                rdate = pd.to_datetime(df["date"], errors="coerce")
                df.loc[miss, "contract_id"] = (
                    reparty.where(reparty!="", "NO_PARTY") + "_" +
                    rdate.dt.strftime("%Y%m%d").where(rdate.notna(), "NO_DATE") + "_" +
                    df.index.astype(str)
                )
            return df
        df_r = synth(df_r)
        df_a = synth(df_a)

    base = pd.concat([df_r, df_a], ignore_index=True)

    if base.empty:
        table = pd.DataFrame(columns=["계약ID","선수금_합계","선급금_합계","Gap(선수-선급)","담당자","주요거래처","진행현황"])
        return base, table, df_r, df_a

    agg_r = base[base["direction"]=="선수금"].groupby("contract_id")["amount"].sum().rename("선수금_합계")
    agg_a = base[base["direction"]=="선급금"].groupby("contract_id")["amount"].sum().rename("선급금_합계")
    agg = pd.concat([agg_r, agg_a], axis=1).fillna(0.0)
    agg["Gap(선수-선급)"] = agg["선수금_합계"] - agg["선급금_합계"]

    meta_cols = ["owner","party","status"]
    meta = base.groupby("contract_id")[meta_cols].agg(lambda s: s.dropna().astype(str).replace({"","nan","None"}, pd.NA).dropna().unique()[:1])
    for c in meta_cols:
        meta[c] = meta[c].apply(lambda arr: arr[0] if isinstance(arr,(list,tuple,pd.Series)) and len(arr)>0 else "")

    table = agg.join(meta, how="left").reset_index().rename(columns={"contract_id":"계약ID","owner":"담당자","party":"주요거래처","status":"진행현황"})
    return base, table, df_r, df_a

# ===================== Sidebar =====================
with st.sidebar:
    st.header("📁 데이터 업로드")
    upl = st.file_uploader("엑셀 파일 (xlsx/xlsm/xls)", type=["xlsx","xlsm","xls"])
    st.caption("매크로(xlsm)는 값만 읽음")

    st.header("⚙️ 인식 옵션")
    strict_filter = st.checkbox("엄격 필터(금액=0 제외, 계약ID 빈값 제외)", value=True)
    synth_contract = st.checkbox("계약ID 없으면 자동 생성", value=True)

    st.header("🔧 매칭 설정")
    use_contract_soft = st.checkbox("계약ID 일치 가중치", value=True)
    use_amount = st.checkbox("금액 조건 사용", value=True)
    amount_tol = st.number_input("금액 허용 오차(원)", min_value=0, value=0, step=1000, format="%d")
    amount_tol_pct = st.slider("금액 허용 오차(%)", 0, 20, 1)
    use_date = st.checkbox("일자 조건 사용", value=True)
    date_window = st.slider("일자 윈도우(±일)", 0, 180, 30)
    use_party_soft = st.checkbox("거래처/텍스트 유사도 가중치", value=True)
    max_combo = st.slider("부분합 매칭 최대 묶음 수(多:1)", 1, 5, 3)

st.markdown("---")
if upl is None:
    st.info("좌측에서 엑셀을 업로드하세요.")
    st.stop()

file_bytes = upl.read()
base, table, receipts, advances = load_data(file_bytes, strict_filter, synth_contract)

# Diagnostics section
with st.expander("🧪 업로드 진단"):
    st.write({"rows_detected": len(base), "receipts_rows": len(receipts), "advances_rows": len(advances)})
    if table.empty:
        st.warning("집계 테이블이 비어 있습니다. 인식 옵션을 조정해 보세요.")

# ===================== KPIs =====================
col1, col2, col3, col4 = st.columns(4)
with col1:
    total_r = base.loc[base["direction"]=="선수금", "amount"].sum()
    st.metric("총 선수금", f"{total_r:,.0f} 원")
with col2:
    total_a = base.loc[base["direction"]=="선급금", "amount"].sum()
    st.metric("총 선급금", f"{total_a:,.0f} 원")
with col3:
    st.metric("Gap(선수-선급)", f"{(total_r-total_a):,.0f} 원")
with col4:
    st.metric("계약 수", f"{table.shape[0]:,}")

st.divider()

# ===================== Filters & Table =====================
fc1, fc2, fc3 = st.columns([2,2,1])
with fc1:
    q = st.text_input("계약ID/거래처/담당자 검색", "")
with fc2:
    owner_filter = st.text_input("담당자 필터 (쉼표로 여러명)", "")
with fc3:
    sort_opt = st.selectbox("정렬", ["Gap(선수-선급)", "선수금_합계", "선급금_합계", "계약ID"], index=0)

view = table.copy()
if not view.empty and q:
    ql = q.strip().lower()
    view = view[view.apply(lambda r: ql in str(r.get("계약ID","" )).lower() or 
                                     ql in str(r.get("주요거래처","" )).lower() or 
                                     ql in str(r.get("담당자",""  )).lower(), axis=1)]
if not view.empty and owner_filter:
    owners = [o.strip().lower() for o in owner_filter.split(',') if o.strip()]
    if owners and "담당자" in view.columns:
        view = view[view["담당자"].str.lower().isin(owners)]

if not view.empty:
    sortable_cols = ["Gap(선수-선급)", "선수금_합계", "선급금_합계", "계약ID"]
    for c in sortable_cols:
        if c not in view.columns:
            view[c] = 0 if c != "계약ID" else ""
    try:
        view = view.sort_values(by=sort_opt if sort_opt in view.columns else "계약ID", ascending=False)
    except Exception:
        pass

st.subheader("📂 계약별 집계")
st.dataframe(view, use_container_width=True, height=420)

# ===================== Matching Engine =====================
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
        for _, a in cand.iterrows():
            aamt = float(a.get('amount', 0) or 0)
            amt_diff = abs(ramt - aamt)
            pct_ok = (ramt == 0) or (amt_diff <= (ramt * tol_pct / 100.0))
            if (not use_amount) or amt_diff <= tol_abs or pct_ok:
                score = 0.0
                if use_amount and ramt > 0:
                    score += max(0.0, 1.0 - (amt_diff / (ramt + 1e-9)))
                if use_date and pd.notna(rdate) and pd.notna(a['date']) and date_window > 0:
                    score += max(0.0, 1 - (abs((rdate - a['date']).days) / (date_window + 1)))
                if use_contract_soft and str(r['contract_id']) != '' and str(a['contract_id']) != '' and str(r['contract_id']) == str(a['contract_id']):
                    score += 0.6
                if use_party_soft:
                    inter = len(r['tok'].intersection(a['tok']))
                    if inter > 0:
                        score += min(0.4, 0.1 * inter)
                if (best is None) or (score > best['score']):
                    best = {'rid': int(r['rid']), 'aids': [int(a['aid'])], 'sum_adv': float(aamt), 'gap': float(ramt - aamt), 'score': float(score)}

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

# ===================== Detail & Matching Tabs =====================
st.subheader("🔎 계약 상세 보기")
contract_ids = table["계약ID"].tolist() if not table.empty else []
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

    t1, t2, t3 = st.tabs(["선수금 상세", "선급금 상세", "자동 매칭(교차검증)"])

    with t1:
        df_r = detail[detail["direction"]=="선수금"][STANDARD_COLS].copy()
        if df_r.empty:
            st.info("해당 계약에 선수금 데이터가 없습니다.")
        else:
            df_r["date"] = pd.to_datetime(df_r["date"], errors="coerce").dt.date
            st.dataframe(df_r.rename(columns={
                "contract_id":"계약ID","direction":"구분","amount":"금액","date":"일자",
                "party":"거래처/고객","owner":"담당자","status":"진행현황","note":"비고","overdue_flag":"기한경과"
            }), use_container_width=True)

    with t2:
        df_a = detail[detail["direction"]=="선급금"][STANDARD_COLS].copy()
        if df_a.empty:
            st.info("해당 계약에 선급금 데이터가 없습니다.")
        else:
            df_a["date"] = pd.to_datetime(df_a["date"], errors="coerce").dt.date
            st.dataframe(df_a.rename(columns={
                "contract_id":"계약ID","direction":"구분","amount":"금액","date":"일자",
                "party":"업체/협력사","owner":"담당자","status":"진행현황","note":"비고","overdue_flag":"기한경과"
            }), use_container_width=True)

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

st.markdown("---")
st.code("pip install streamlit pandas openpyxl xlrd xlsxwriter", language="bash")
