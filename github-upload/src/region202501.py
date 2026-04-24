import os
import platform
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# =========================
# 0) 파일 경로/설정
# =========================
CSV_PATH = r"C:\sqlite\region202501.csv"   # ✅ 본인 PC 파일 위치로 수정
# 업로드 파일로 테스트하려면:
# CSV_PATH = "/mnt/data/region202501.csv"

OUT_DIR = "viz_output_region202501"
TOP_TABLE = 20   # 표 상위 N
TOP_BAR   = 15   # 막대 상위 N
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

# 컬럼명 공백 제거(가끔 "Region " 같이 들어옴)
df.columns = [str(c).strip() for c in df.columns]

# =========================
# 2) 한글 폰트 설정
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
# 3) 컬럼 자동 인식 (지역 컬럼 + 1개월 값 컬럼)
# =========================
cols = list(df.columns)

def find_col(cands):
    for c in cands:
        if c in df.columns:
            return c
    return None

# (1) 지역 컬럼
region_col = find_col(["Region", "지역", "권역", "시도", "Province", "State", "지역명"])
if region_col is None:
    obj_cols = [c for c in cols if df[c].dtype == "object"]
    if not obj_cols:
        raise ValueError(f"지역(문자열) 컬럼을 찾지 못했습니다. 현재 컬럼: {cols}")
    region_col = obj_cols[0]

# (2) 값 컬럼(2025-01 / Jan / 1월 / Total 등)
month_like_keys = ["2025-01", "202501", "2025/01", "2025.01", "jan", "1월", "01월"]
value_col = None

for c in cols:
    if c == region_col:
        continue
    cl = str(c).lower()
    if any(k in cl for k in month_like_keys):
        value_col = c
        break

# Total/합계가 있으면 차선으로 사용
if value_col is None:
    value_col = find_col(["Total", "TOTAL", "total", "합계"])

# 그래도 없으면 숫자 컬럼 1개 선택
if value_col is None:
    num_cols = [c for c in cols if c != region_col and pd.api.types.is_numeric_dtype(df[c])]
    if not num_cols:
        # 숫자로 변환 가능한 컬럼 찾기
        for c in cols:
            if c == region_col:
                continue
            s = pd.to_numeric(df[c].astype(str).str.replace(",", "", regex=False), errors="coerce")
            if s.notna().mean() >= 0.7:
                num_cols = [c]
                break
    if not num_cols:
        raise ValueError(f"값(숫자) 컬럼을 찾지 못했습니다. 현재 컬럼: {cols}")
    value_col = num_cols[0]

print("✅ 감지된 컬럼")
print(f"- 지역 컬럼: {region_col}")
print(f"- 값 컬럼: {value_col}")

# =========================
# 4) 전처리: 숫자 변환 / 소계행 제거 / 중복 합산
# =========================
df[region_col] = df[region_col].astype(str).str.strip()

# 값 컬럼 숫자화(쉼표 제거 포함)
df[value_col] = pd.to_numeric(
    df[value_col].astype(str).str.replace(",", "", regex=False),
    errors="coerce"
).fillna(0)

# Sub-total/소계/합계 같은 집계행 제거(지역 텍스트에 들어있는 경우)
data = df[~df[region_col].str.contains("sub-total|subtotal|소계|합계", case=False, na=False)].copy()

# 지역명이 비어있는 행 제거(구분행 방지)
data = data[data[region_col].notna() & (data[region_col].astype(str).str.strip() != "")].copy()

# ✅ 지역 중복이 있으면 합산
agg = data.groupby(region_col, as_index=True)[value_col].sum().sort_values(ascending=False)
total_value = float(agg.sum())

# 라벨(한글) 만들기: 컬럼명이 Jan/2025-01 류면 “2025년 1월”
col_lower = str(value_col).lower()
if ("jan" in col_lower) or ("2025-01" in col_lower) or ("202501" in col_lower) or ("1월" in str(value_col)):
    value_label = "2025년 1월"
else:
    value_label = str(value_col)

# =========================
# 5) 한글 표(상위 N) + 저장
# =========================
top_table = agg.head(TOP_TABLE).to_frame(name=f"{value_label} 값")
top_table.index.name = "지역"

top_table.to_excel(os.path.join(OUT_DIR, "상위지역_표.xlsx"))
top_table.to_csv(os.path.join(OUT_DIR, "상위지역_표.csv"), encoding="utf-8-sig")

print("\n[상위 지역 표]")
print(top_table)

# 표 이미지 저장
plt.figure(figsize=(12, 0.6 + 0.35 * len(top_table)))
plt.axis("off")
tbl = plt.table(
    cellText=top_table.reset_index().values,
    colLabels=["지역", f"{value_label} 값"],
    cellLoc="center",
    loc="center"
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1, 1.2)
plt.title(f"{value_label} 상위 {TOP_TABLE}개 지역(표)", pad=12)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "표_상위지역.png"), dpi=200)
plt.show()

# =========================
# 6) 차트 1: 상위 N 지역 막대그래프(한글)
# =========================
top_bar = agg.head(TOP_BAR)

plt.figure(figsize=(10, 5))
plt.barh(top_bar.index.astype(str)[::-1], top_bar.values[::-1])
plt.title(f"{value_label} 상위 {TOP_BAR}개 지역")
plt.xlabel("값")
plt.ylabel("지역")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "차트_상위막대.png"), dpi=200)
plt.show()

# =========================
# 7) 차트 2: 파레토(누적비율) — 1개월 데이터에 유용
# =========================
cum_rate = (agg.cumsum() / total_value) * 100 if total_value != 0 else agg.cumsum() * 0

pareto_n = min(30, len(cum_rate))
x = list(range(1, pareto_n + 1))
y = cum_rate.head(pareto_n).values

plt.figure(figsize=(10, 4))
plt.plot(x, y, marker="o")
plt.title(f"{value_label} 누적비율(파레토)")
plt.xlabel("순위(상위부터)")
plt.ylabel("누적비율(%)")
plt.grid(True, alpha=0.3)
plt.ylim(0, 105)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "차트_파레토누적비율.png"), dpi=200)
plt.show()

print(f"\n✅ 저장 완료: {os.path.abspath(OUT_DIR)}")
print("   - 표: 상위지역_표.xlsx / 상위지역_표.csv / 표_상위지역.png")
print("   - 차트: 차트_상위막대.png / 차트_파레토누적비율.png")
