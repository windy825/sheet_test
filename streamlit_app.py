# 안전 정렬: 선택한 컬럼이 없으면 대체 컬럼으로 정렬
sortable_cols = ["Gap(선수-선급)", "선수금_합계", "선급금_합계", "계약ID"]
# 누락 컬럼을 0/빈값으로 보완
for c in sortable_cols:
    if c not in view.columns:
        view[c] = 0 if c != "계약ID" else ""

fallback = next((c for c in [sort_opt] + sortable_cols if c in view.columns), "계약ID")
try:
    view = view.sort_values(by=fallback, ascending=False)
except Exception:
    # 정렬 실패 시 원본 유지
    pass
