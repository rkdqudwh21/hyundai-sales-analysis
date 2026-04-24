import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform

# =========================
# 설정
# =========================
DATASETS = {
    "EU": ["europesales202312.csv", "eu202412sales.csv", "eusales202501.csv"],
    "GLOBAL": ["globalsales202312.csv", "globalsales202412.csv", "globalsales202501.csv"],
    "USRETAIL": ["usretail202312.csv", "usretail202412.csv", "usretail202501.csv"],
}

OUT_BASE = "reco_output"
os.makedirs(OUT_BASE, exist_ok=True)

TOP_N_DEFAULT = 10
ALPHA_RECENCY = 0.55
BETA_MOMENTUM = 0.25
GAMMA_POP     = 0.20

MONTHS_EN = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

# =========================
# 유틸
# =========================
def set_korean_font():
    candidates = []
    sysname = platform.system()
    if sysname == "Windows":
        candidates += ["Malgun Gothic"]
    elif sysname == "Darwin":
        candidates += ["AppleGothic"]
    candidates += ["NanumGothic", "Noto Sans CJK KR", "Noto Sans KR"]
    installed = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in installed:
            plt.rcParams["font.family"] = name
            break
    else:
        plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False

def resolve_path(filename: str) -> str:
    candidates = [
        rf"C:\sqlite\{filename}",
        rf"C:\Pythonexam\{filename}",
        filename,
        f"/mnt/data/{filename}",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"파일을 찾지 못했습니다: {filename}\n시도한 경로:\n- " + "\n- ".join(candidates)
    )

def read_csv_safely(path: str) -> pd.DataFrame:
    for enc in ["utf-8-sig", "cp949", "euc-kr", "utf-8"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    return pd.read_csv(path)

def find_col(df: pd.DataFrame, cands):
    for c in cands:
        if c in df.columns:
            return c
    return None

def parse_yyyymm_from_filename(filename: str):
    m = re.search(r"(20\d{2})(\d{2})", filename)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))

def detect_month_cols(df: pd.DataFrame):
    cols = []
    for m in MONTHS_EN:
        if f"{m}." in df.columns:
            cols.append(f"{m}.")
        elif m in df.columns:
            cols.append(m)
    return cols

def safe_numeric_series(x: pd.Series) -> pd.Series:
    s = pd.to_numeric(x.astype(str).str.replace(",", "", regex=False), errors="coerce").fillna(0)
    return s.clip(lower=0)  # ✅ -1 등 음수는 0 처리

def normalize_01(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    mx = float(s.max()) if len(s) else 0.0
    if mx <= 0:
        return s * 0.0
    return s / mx

# =========================
# 파일 -> 기간합 테이블
# =========================
def file_period_sum(filename: str) -> pd.DataFrame:
    path = resolve_path(filename)
    df = read_csv_safely(path)
    df.columns = [str(c).strip() for c in df.columns]

    # ✅ 리테일 파일 대응 후보 확대
    model_col = find_col(df, ["Models","Model","models","model","모델","항목","Name","Product","제품",
                              "Category","Item","Department","상품","카테고리"])
    total_col = find_col(df, ["Total","TOTAL","total","합계","연간합계","Grand Total"])

    if model_col is None:
        obj_cols = [c for c in df.columns if df[c].dtype == "object"]
        if not obj_cols:
            raise ValueError(f"[{filename}] 항목(문자열) 컬럼을 찾지 못했습니다. 컬럼: {list(df.columns)}")
        model_col = obj_cols[0]

    df[model_col] = df[model_col].astype(str).str.strip()
    df = df[~df[model_col].str.contains("sub-total|subtotal|소계|합계|grand total", case=False, na=False)].copy()

    y, mm = parse_yyyymm_from_filename(filename)
    if y is None or mm is None:
        raise ValueError(f"[{filename}] 파일명에서 YYYYMM을 찾지 못했습니다.")

    period = str(y) if mm == 12 else f"{y}-{mm:02d}"

    month_cols = detect_month_cols(df)
    num_cols = month_cols + ([total_col] if total_col else [])
    for c in num_cols:
        df[c] = safe_numeric_series(df[c])

    if total_col and total_col in df.columns:
        df["기간합"] = df[total_col]
    else:
        if not month_cols:
            raise ValueError(f"[{filename}] Total도 없고 월 컬럼도 없습니다.")
        df["기간합"] = df[month_cols].sum(axis=1)

    out = df.groupby(model_col, as_index=False)["기간합"].sum()
    out = out.rename(columns={model_col: "모델", "기간합": period})
    return out

def build_period_table(file_list):
    tables = [file_period_sum(f) for f in file_list]
    merged = tables[0]
    for t in tables[1:]:
        merged = merged.merge(t, on="모델", how="outer")
    merged = merged.fillna(0)

    period_cols = sorted([c for c in merged.columns if c != "모델"], key=lambda x: str(x))
    merged[period_cols] = merged[period_cols].apply(pd.to_numeric, errors="coerce").fillna(0).clip(lower=0)
    return merged, period_cols

# =========================
# 추천 점수
# =========================
def recommend(period_df: pd.DataFrame, period_cols):
    latest = period_cols[-1]
    prev = period_cols[-2] if len(period_cols) >= 2 else None

    df = period_df.copy()
    df["전체판매"] = df[period_cols].sum(axis=1)
    df["인기점수"] = normalize_01(np.log1p(df["전체판매"]))

    if len(period_cols) == 1:
        w = [1.0]
    elif len(period_cols) == 2:
        w = [0.35, 0.65]
    else:
        w = [0.15, 0.35, 0.50]

    rec = 0.0
    for col, weight in zip(period_cols, w):
        rec += normalize_01(df[col]) * weight
    df["최근점수"] = normalize_01(rec)

    if prev:
        prev_v = df[prev].replace(0, pd.NA)
        mom = (df[latest] - df[prev]) / prev_v
        mom = mom.fillna(0).clip(lower=-1, upper=5)
        df["성장점수"] = normalize_01(mom - mom.min())
    else:
        df["성장점수"] = 0.0

    df["최종점수"] = (
        ALPHA_RECENCY * df["최근점수"] +
        BETA_MOMENTUM * df["성장점수"] +
        GAMMA_POP     * df["인기점수"]
    )
    return df.sort_values("최종점수", ascending=False).reset_index(drop=True)

# =========================
# 출력
# =========================
def save_table_image(df: pd.DataFrame, title: str, out_path_base: str):
    df.to_csv(out_path_base + ".csv", encoding="utf-8-sig", index=False)
    try:
        df.to_excel(out_path_base + ".xlsx", index=False)
    except Exception:
        pass

    plt.close("all")
    plt.figure(figsize=(14, 0.6 + 0.35 * len(df)))
    plt.axis("off")
    tbl = plt.table(
        cellText=df.values,
        colLabels=list(df.columns),
        cellLoc="center",
        loc="center"
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.2)
    plt.title(title, pad=12)
    plt.tight_layout()
    plt.savefig(out_path_base + ".png", dpi=200)
    plt.show()

def bar_chart(labels, values, title, out_png):
    plt.close("all")
    plt.figure(figsize=(10, 5))
    plt.barh(pd.Series(labels).astype(str)[::-1], np.array(values)[::-1])
    plt.title(title)
    plt.xlabel("추천점수")
    plt.ylabel("항목")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.show()

# =========================
# 메뉴
# =========================
def choose_dataset():
    keys = list(DATASETS.keys())
    print("\n추천에 사용할 데이터 그룹 선택:")
    for i, k in enumerate(keys, start=1):
        print(f" {i}) {k}  ->  {', '.join(DATASETS[k])}")
    while True:
        c = input("번호 입력: ").strip()
        if c.isdigit() and 1 <= int(c) <= len(keys):
            return keys[int(c)-1]
        print("❗ 다시 입력해주세요.")

def run_menu():
    set_korean_font()
    ds = choose_dataset()
    files = DATASETS[ds]
    out_dir = os.path.join(OUT_BASE, ds)
    os.makedirs(out_dir, exist_ok=True)

    period_df, period_cols = build_period_table(files)
    reco_df = recommend(period_df, period_cols)

    while True:
        print("\n==============================")
        print(f"[{ds}] 추천 (1개씩 출력)")
        print(" 1) 추천 TOP 표(1개)")
        print(" 2) 추천 TOP 막대차트(1개)")
        print(" 3) 전체 추천 결과 저장")
        print(" 0) 종료")
        print("==============================")

        choice = input("번호 입력: ").strip()

        if choice == "1":
            n = input(f"TOP 몇 개? (기본 10): ").strip()
            n = int(n) if n.isdigit() else TOP_N_DEFAULT
            cols = ["모델"] + period_cols + ["전체판매","최근점수","성장점수","인기점수","최종점수"]
            view = reco_df[cols].head(n).copy()
            save_table_image(
                view,
                title=f"추천 TOP {n} [{ds}]",
                out_path_base=os.path.join(out_dir, f"표_추천_TOP{n}")
            )

        elif choice == "2":
            n = input(f"TOP 몇 개? (기본 10): ").strip()
            n = int(n) if n.isdigit() else TOP_N_DEFAULT
            top = reco_df.head(n)
            bar_chart(
                labels=top["모델"],
                values=top["최종점수"],
                title=f"추천 점수 TOP {n} [{ds}]",
                out_png=os.path.join(out_dir, f"차트_추천_TOP{n}.png")
            )

        elif choice == "3":
            reco_df.to_csv(os.path.join(out_dir, "추천_전체.csv"), encoding="utf-8-sig", index=False)
            try:
                reco_df.to_excel(os.path.join(out_dir, "추천_전체.xlsx"), index=False)
            except Exception:
                pass
            print("✅ 저장 완료:", os.path.abspath(out_dir))

        elif choice == "0":
            print("종료합니다.")
            break
        else:
            print("❗ 0~3 중에서 선택해주세요.")

    print("\n✅ 결과 폴더:", os.path.abspath(out_dir))

if __name__ == "__main__":
    run_menu()
