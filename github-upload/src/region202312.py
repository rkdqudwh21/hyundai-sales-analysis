import os
import platform
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# =========================
# 0) 파일 경로/설정
# =========================
CSV_PATH = r"C:\sqlite\region202312.csv"  # ✅ 본인 PC 경로로 수정
# 업로드 파일로 테스트하려면:
# CSV_PATH = "/mnt/data/region202312.csv"

OUT_DIR = "viz_output_region202312"
TOP_TABLE = 15
TOP_BAR   = 10
TOP_LINE  = 5
TOP_HEAT  = 15
os.makedirs(OUT_DIR, exist_ok=True)

# =========================
# 1) CSV 읽기(인코딩 자동 시도)
# =========================
def read_csv_safely(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"파일 경로를 찾지 못했습니다: {path}")
    for enc in ["utf-8-sig", "cp949", "euc-kr", "utf-8"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    return pd.read_csv(path)

df = read_csv_safely(CSV_PATH)

# =========================
# 2) 한글 폰트 설정(가능한 폰트 자동 선택)
# =========================
def set_korean_font():
    candidates = []
    sysname = platform.system()
    if sysname == "Windows":
        candidates += ["Malgun Gothic"]
    elif sysname == "Darwin":
        candidates += ["AppleGothic"]
    candidates += ["NanumGothic", "Noto Sans KR", "Noto Sans CJK KR"]

    installed = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in installed:
            plt.rcParams["font.family"] = name
            break
    else:
        plt.rcParams["font.family"] = "DejaVu Sans"  # 한글 폰트 없으면 깨질 수 있음
    plt.rcParams["axes.unicode_minus"] = False

set_korean_font()

# =========================
# 3) 컬럼 자동 인식(지역/월/합계)
# =========================
cols = list(df.columns)

def find_col(cands):
    for c in cands:
        if c in df.columns:
            return c
    return None

# 지역 컬럼
region_col = find_col(["Region", "지역", "권역", "시도", "Province", "State"])
if region_col is None:
    obj_cols = [c for c in cols if df[c].dtype == "object"]
    if not obj_cols:
        raise ValueError(f"지역(문자열) 컬럼을 찾지 못했습니다. 현재 컬럼: {cols}")
    region_col = obj_cols[0]

# 합계 컬럼(있으면 사용)
total_col = find_col(["Total", "TOTAL", "total", "합계", "연간합계", "Grand Total"])

# 월 컬럼(Jan/Jan. 둘 다 대응)
month_variants = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
month_map = {"Jan":"1월","Feb":"2월","Mar":"3월","Apr":"4월","May":"5월","Jun":"6월",
             "Jul":"7월","Aug":"8월","Sep":"9월","Oct":"10월","Nov":"11월","Dec":"12월"}

month_cols, month_kr = [], []
for m in month_variants:
    if f"{m}." in df.columns:
        month_cols.append(f"{m}."); month_kr.append(month_map[m])
    elif m in df.columns:
        month_cols.append(m);       month_kr.append(month_map[m])

if len(month_cols) == 0:
    raise ValueError(f"월 컬럼(Jan~Dec)을 찾지 못했습니다. 현재 컬럼: {cols}")

print("✅ 감지된 컬럼")
print(f"- 지역 컬럼: {region_col}")
print(f"- 합계 컬럼: {total_col if total_col else '(없음)'}")
print(f"- 월 컬럼: {month_cols}")

# =========================
# 4) 전처리: 숫자 변환 / 구분행 제거 / 소계 제거 / 중복 합산
# =========================
num_targets = month_cols + ([total_col] if total_col else [])

for c in num_targets:
    df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", "", regex=False), errors="coerce")

data = df.copy()
data[region_col] = data[region_col].astype(str).str.strip()

# Total이 있으면, Total이 비어있는 구분/헤더행 제거
if total_col:
    data = data[data[total_col].notna()].copy()

# Sub-total/소계/합계 같은 집계행 제거(지역 텍스트에 들어있는 경우)
data = data[~data[region_col].str.contains("sub-total|subtotal|소계|합계", case=False, na=False)].copy()

# NaN -> 0
for c in num_targets:
    data[c] = data[c].fillna(0)

# ✅ 지역 중복이 있으면 합산
use_cols = month_cols + ([total_col] if total_col else [])
agg = data.groupby(region_col, as_index=True)[use_cols].sum()

# 정렬 기준: Total 있으면 Total, 없으면 월합(파생값, 데이터 “추가” 없이 계산만)
if total_col and total_col in agg.columns:
    rank_series = agg[total_col]
    rank_label_kr = "연간합계"
else:
    rank_series = agg[month_cols].sum(axis=1)
    rank_label_kr = "연간합계(월합)"

agg_sorted = agg.loc[rank_series.sort_values(ascending=False).index].copy()

# =========================
# 5) 한글 표(상위 N) + 저장
# =========================
top_table = agg_sorted.head(TOP_TABLE).copy()

rename_map = {mc: mk for mc, mk in zip(month_cols, month_kr)}
if total_col and total_col in top_table.columns:
    rename_map[total_col] = "연간합계"

top_table_kr = top_table.rename(columns=rename_map)
top_table_kr.index.name = "지역"

# Total이 없으면 표에 월합(연간합계) 열을 “추가”하지 않고, 저장도 원본 컬럼만 저장합니다.
top_table_kr.to_excel(os.path.join(OUT_DIR, "상위지역_표.xlsx"))
top_table_kr.to_csv(os.path.join(OUT_DIR, "상위지역_표.csv"), encoding="utf-8-sig")

print("\n[상위 지역 표]")
print(top_table_kr)

# 표 이미지 저장
plt.figure(figsize=(14, 0.6 + 0.35 * len(top_table_kr)))
plt.axis("off")
tbl = plt.table(
    cellText=top_table_kr.reset_index().values,
    colLabels=["지역"] + list(top_table_kr.columns),
    cellLoc="center",
    loc="center"
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1, 1.2)
plt.title(f"상위 {TOP_TABLE}개 지역 월별/연간 값(표)", pad=12)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "표_상위지역.png"), dpi=200)
plt.show()

# =========================
# 6) 차트 1: 월별 전체 합계 추이
# =========================
monthly_total = agg[month_cols].sum()
plt.figure(figsize=(10, 4))
plt.plot(month_kr, monthly_total.values, marker="o")
plt.title("월별 전체 합계 추이")
plt.xlabel("월")
plt.ylabel("합계")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "차트_월별전체추이.png"), dpi=200)
plt.show()

# =========================
# 7) 차트 2: 지역별 연간합계 상위 N (막대)
# =========================
top_bar_idx = rank_series.sort_values(ascending=False).head(TOP_BAR).index
top_bar_vals = rank_series.loc[top_bar_idx]

plt.figure(figsize=(10, 5))
plt.barh(top_bar_idx.astype(str)[::-1], top_bar_vals.values[::-1])
plt.title(f"지역별 {rank_label_kr} 상위 {TOP_BAR}")
plt.xlabel("값")
plt.ylabel("지역")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "차트_지역상위막대.png"), dpi=200)
plt.show()

# =========================
# 8) 차트 3: 상위 N개 지역 월별 추이 (라인)
# =========================
top_line = agg_sorted.head(TOP_LINE)[month_cols].copy()
top_line.columns = month_kr

plt.figure(figsize=(10, 5))
for region, row in top_line.iterrows():
    plt.plot(month_kr, row.values, marker="o", label=str(region))
plt.title(f"상위 {TOP_LINE}개 지역 월별 추이")
plt.xlabel("월")
plt.ylabel("값")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "차트_상위지역_월별추이.png"), dpi=200)
plt.show()

# =========================
# 9) 차트 4: 상위 N개 지역 월별 히트맵
# =========================
top_heat = agg_sorted.head(TOP_HEAT)[month_cols].copy()
top_heat.columns = month_kr

plt.figure(figsize=(12, 6))
im = plt.imshow(top_heat.values, aspect="auto")
plt.title(f"상위 {TOP_HEAT}개 지역 월별 히트맵")
plt.yticks(range(len(top_heat.index)), top_heat.index.astype(str))
plt.xticks(range(len(month_kr)), month_kr)
plt.colorbar(im, label="값")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "차트_히트맵.png"), dpi=200)
plt.show()

print(f"\n✅ 저장 완료: {os.path.abspath(OUT_DIR)}")
print("   - 표: 상위지역_표.xlsx / 상위지역_표.csv / 표_상위지역.png")
print("   - 차트: 차트_월별전체추이.png / 차트_지역상위막대.png / 차트_상위지역_월별추이.png / 차트_히트맵.png")
