import os
import re
import argparse
from datetime import datetime
from typing import List, Optional, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import platform
import matplotlib.font_manager as fm


# =========================
# 0) 파일 자동 탐색(Windows/리눅스 모두 대응)
# =========================
MODEL_FILES  = ["model202312.csv", "model202412.csv", "model202501.csv"]
REGION_FILES = ["region202312.csv", "region202412.csv", "region202501.csv"]

CANDIDATE_DIRS = [
    r"C:\Pythonexam",
    os.getcwd(),
    "/mnt/data",  # (업로드 환경)
]

OUT_DIR_BASE = "reco_output_no_price"

MONTHS = ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]


# =========================
# 0-1) 한글 폰트 설정(가능한 경우)
# =========================
def set_korean_font() -> None:
    try:
        sysname = platform.system()
        if sysname == "Windows":
            plt.rcParams["font.family"] = "Malgun Gothic"
        elif sysname == "Darwin":
            plt.rcParams["font.family"] = "AppleGothic"
        else:
            candidates = ["NanumGothic", "Noto Sans CJK KR", "DejaVu Sans"]
            available = {f.name for f in fm.fontManager.ttflist}
            for c in candidates:
                if c in available:
                    plt.rcParams["font.family"] = c
                    break
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass


# =========================
# 1) 유틸: CSV 안전하게 읽기(인코딩 자동 시도)
# =========================
def read_csv_safely(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"파일 경로를 찾지 못했습니다: {path}")
    last_err = None
    for enc in ["utf-8-sig", "cp949", "euc-kr", "utf-8"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_err = e
    raise last_err


def find_existing_paths(filenames: List[str]) -> List[str]:
    found = []
    for fn in filenames:
        hit = None
        for d in CANDIDATE_DIRS:
            p = os.path.join(d, fn)
            if os.path.exists(p):
                hit = p
                break
        if hit is None:
            raise FileNotFoundError(
                f"필요한 파일을 찾지 못했습니다: {fn}\n"
                f"다음 폴더에서 찾았습니다: {CANDIDATE_DIRS}\n"
                f"파일을 위 폴더 중 하나로 옮기거나, 코드의 CANDIDATE_DIRS를 수정하세요."
            )
        found.append(hit)
    return found


def extract_period_from_filename(path: str) -> str:
    m = re.search(r"(\d{6})", os.path.basename(path))
    return m.group(1) if m else "unknown"


def _norm_col(c: str) -> str:
    return re.sub(r"[\s\.\-_]", "", str(c).strip().lower())


def pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    norm = {_norm_col(c): c for c in df.columns}
    for cand in candidates:
        key = _norm_col(cand)
        if key in norm:
            return norm[key]
    return None


def drop_total_like_rows(df: pd.DataFrame, name_col: str) -> pd.DataFrame:
    """합계/소계 같은 집계행 제거(한/영 모두 대응)."""
    s = df[name_col].astype(str).str.strip()
    key = s.str.lower().str.replace(" ", "", regex=False)

    bad_keywords = [
        "total", "subtotal", "sub-total", "grandtotal", "grand-total",
        "합계", "총계", "소계", "전체", "합  계"
    ]
    mask = pd.Series(False, index=df.index)
    for w in bad_keywords:
        mask |= key.str.contains(w.replace(" ", ""), na=False)
    return df[~mask].copy()


def standardize_month_cols(df: pd.DataFrame) -> Dict[str, str]:
    """df의 월 컬럼명을 jan~dec로 매핑"""
    colmap = {}
    for c in df.columns:
        n = _norm_col(c)
        for m in MONTHS:
            if n == m:
                colmap[c] = m
    return colmap


def load_model_file(path: str) -> pd.DataFrame:
    df = read_csv_safely(path)

    model_col = pick_col(df, ["Model", "모델", "차종", "차량", "차종명"])
    if model_col is None:
        raise ValueError(f"[{os.path.basename(path)}] 모델 컬럼을 찾지 못했습니다. 컬럼 목록: {list(df.columns)}")

    total_col = pick_col(df, ["Total", "합계", "총계", "TOTAL"])
    colmap = standardize_month_cols(df)

    keep = [model_col] + list(colmap.keys()) + ([total_col] if total_col else [])
    df = df[keep].copy()
    df.rename(columns={model_col: "model", **colmap}, inplace=True)
    df["model"] = df["model"].astype(str).str.strip()

    df = drop_total_like_rows(df, "model")

    # 월 컬럼 없는 경우 대비(없으면 0으로)
    for m in MONTHS:
        if m not in df.columns:
            df[m] = 0

    for m in MONTHS:
        df[m] = pd.to_numeric(df[m], errors="coerce").fillna(0)

    if total_col:
        df.rename(columns={total_col: "total"}, inplace=True)
        df["total"] = pd.to_numeric(df["total"], errors="coerce").fillna(0)
    else:
        df["total"] = df[MONTHS].sum(axis=1)

    df["period"] = extract_period_from_filename(path)
    df["year"] = pd.to_numeric(df["period"].str[:4], errors="coerce")
    return df


def load_region_file(path: str) -> pd.DataFrame:
    df = read_csv_safely(path)

    region_col = pick_col(df, ["Region", "지역", "권역", "국가", "market"])
    if region_col is None:
        raise ValueError(f"[{os.path.basename(path)}] 지역 컬럼을 찾지 못했습니다. 컬럼 목록: {list(df.columns)}")

    total_col = pick_col(df, ["Total", "합계", "총계", "TOTAL"])
    colmap = standardize_month_cols(df)

    keep = [region_col] + list(colmap.keys()) + ([total_col] if total_col else [])
    df = df[keep].copy()
    df.rename(columns={region_col: "region", **colmap}, inplace=True)
    df["region"] = df["region"].astype(str).str.strip()
    df = drop_total_like_rows(df, "region")

    for m in MONTHS:
        if m not in df.columns:
            df[m] = 0
        df[m] = pd.to_numeric(df[m], errors="coerce").fillna(0)

    if total_col:
        df.rename(columns={total_col: "total"}, inplace=True)
        df["total"] = pd.to_numeric(df["total"], errors="coerce").fillna(0)
    else:
        df["total"] = df[MONTHS].sum(axis=1)

    df["period"] = extract_period_from_filename(path)
    df["year"] = pd.to_numeric(df["period"].str[:4], errors="coerce")
    return df


def minmax(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").astype(float)
    lo, hi = np.nanmin(s.values), np.nanmax(s.values)
    if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < 1e-12:
        return pd.Series([0.0] * len(s), index=s.index)
    return (s - lo) / (hi - lo)


def build_model_metrics(model_paths: List[str]) -> pd.DataFrame:
    frames = [load_model_file(p) for p in model_paths]
    all_df = pd.concat(frames, ignore_index=True)

    # period별 테이블
    pivot_total = all_df.pivot_table(index="model", columns="period", values="total", aggfunc="sum").fillna(0)
    pivot_jan   = all_df.pivot_table(index="model", columns="period", values="jan",   aggfunc="sum").fillna(0)

    p2023 = next((p for p in pivot_total.columns if str(p).startswith("2023")), None)
    p2024 = next((p for p in pivot_total.columns if str(p).startswith("2024")), None)
    p2025 = next((p for p in pivot_total.columns if str(p).startswith("2025")), None)

    out = pd.DataFrame({"model": pivot_total.index})
    out["sales_2023_total"] = pivot_total[p2023].values if p2023 else 0
    out["sales_2024_total"] = pivot_total[p2024].values if p2024 else 0
    out["sales_2024_jan"]   = pivot_jan[p2024].values   if p2024 else 0
    out["sales_2025_jan"]   = pivot_jan[p2025].values   if p2025 else 0

    # 성장률
    out["growth_24_vs_23"] = np.where(
        out["sales_2023_total"] > 0,
        (out["sales_2024_total"] / out["sales_2023_total"]) - 1.0,
        np.nan
    )
    out["yoy_jan_25_vs_24"] = np.where(
        out["sales_2024_jan"] > 0,
        (out["sales_2025_jan"] / out["sales_2024_jan"]) - 1.0,
        np.nan
    )

    # 2024 월별 안정성(CV = std/mean, 낮을수록 안정)
    if p2024:
        df2024 = all_df[all_df["period"] == p2024].copy()
        df2024["mean_2024"] = df2024[MONTHS].mean(axis=1)
        df2024["std_2024"]  = df2024[MONTHS].std(axis=1)
        df2024["cv_2024"]   = np.where(df2024["mean_2024"] > 0, df2024["std_2024"] / df2024["mean_2024"], np.nan)
        stab = df2024[["model","cv_2024"]].groupby("model", as_index=False).mean()
        out = out.merge(stab, on="model", how="left")
    else:
        out["cv_2024"] = np.nan

    # 종합 점수: (판매량 + 최근 상승세 + 장기 성장 + 안정성)
    # 안정성은 cv가 낮을수록 좋으니 (1 - 정규화(cv)) 로 반영
    s_volume = minmax(out["sales_2024_total"])
    s_rising = minmax(out["yoy_jan_25_vs_24"].fillna(out["yoy_jan_25_vs_24"].median(skipna=True) if out["yoy_jan_25_vs_24"].notna().any() else 0))
    s_growth = minmax(out["growth_24_vs_23"].fillna(out["growth_24_vs_23"].median(skipna=True) if out["growth_24_vs_23"].notna().any() else 0))
    s_stable = 1.0 - minmax(out["cv_2024"].fillna(out["cv_2024"].median(skipna=True) if out["cv_2024"].notna().any() else 0))

    out["score_overall"] = (0.55*s_volume + 0.25*s_rising + 0.10*s_growth + 0.10*s_stable) * 100
    out["score_overall"] = out["score_overall"].round(2)

    # 보기 좋게 정렬
    out.sort_values(["score_overall","sales_2024_total"], ascending=False, inplace=True)
    out.reset_index(drop=True, inplace=True)
    return out


def build_region_summary(region_paths: List[str]) -> pd.DataFrame:
    frames = [load_region_file(p) for p in region_paths]
    all_df = pd.concat(frames, ignore_index=True)

    pivot_total = all_df.pivot_table(index="region", columns="period", values="total", aggfunc="sum").fillna(0)
    pivot_jan   = all_df.pivot_table(index="region", columns="period", values="jan",   aggfunc="sum").fillna(0)

    p2023 = next((p for p in pivot_total.columns if str(p).startswith("2023")), None)
    p2024 = next((p for p in pivot_total.columns if str(p).startswith("2024")), None)
    p2025 = next((p for p in pivot_total.columns if str(p).startswith("2025")), None)

    out = pd.DataFrame({"region": pivot_total.index})
    out["sales_2023_total"] = pivot_total[p2023].values if p2023 else 0
    out["sales_2024_total"] = pivot_total[p2024].values if p2024 else 0
    out["sales_2024_jan"]   = pivot_jan[p2024].values   if p2024 else 0
    out["sales_2025_jan"]   = pivot_jan[p2025].values   if p2025 else 0

    out["growth_24_vs_23"] = np.where(
        out["sales_2023_total"] > 0,
        (out["sales_2024_total"] / out["sales_2023_total"]) - 1.0,
        np.nan
    )
    out["yoy_jan_25_vs_24"] = np.where(
        out["sales_2024_jan"] > 0,
        (out["sales_2025_jan"] / out["sales_2024_jan"]) - 1.0,
        np.nan
    )

    out.sort_values("sales_2024_total", ascending=False, inplace=True)
    out.reset_index(drop=True, inplace=True)
    return out


def make_recommendations(metrics: pd.DataFrame, mode: str, top_n: int, min_volume: float) -> pd.DataFrame:
    df = metrics.copy()
    if min_volume is not None:
        df = df[df["sales_2024_total"] >= min_volume].copy()

    if mode == "overall":
        df = df.sort_values(["score_overall","sales_2024_total"], ascending=False)
    elif mode == "bestseller":
        df = df.sort_values(["sales_2024_total","score_overall"], ascending=False)
    elif mode == "rising":
        # yoy_jan_25_vs_24가 NaN이면 뒤로
        df["tmp"] = df["yoy_jan_25_vs_24"].fillna(-9999)
        df = df.sort_values(["tmp","score_overall"], ascending=False).drop(columns=["tmp"])
    elif mode == "stable":
        # cv 낮을수록 안정. NaN은 뒤로
        df["tmp"] = df["cv_2024"].fillna(9999)
        df = df.sort_values(["tmp","sales_2024_total"], ascending=True).drop(columns=["tmp"])
    else:
        raise ValueError("mode는 overall|bestseller|rising|stable 중 하나여야 합니다.")

    cols = [
        "model",
        "score_overall",
        "sales_2024_total",
        "sales_2025_jan",
        "growth_24_vs_23",
        "yoy_jan_25_vs_24",
        "cv_2024",
    ]
    return df[cols].head(top_n).reset_index(drop=True)


def save_charts(out_dir: str, reco: pd.DataFrame, region_sum: pd.DataFrame) -> None:
    set_korean_font()
    os.makedirs(out_dir, exist_ok=True)

    # 추천 점수 막대
    plt.figure()
    plt.bar(reco["model"], reco["score_overall"])
    plt.xticks(rotation=75, ha="right")
    plt.title("추천 모델 점수(종합 스코어)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "reco_score_bar.png"), dpi=200)
    plt.close()

    # 2024 지역 상위 10
    topR = region_sum.head(10).copy()
    plt.figure()
    plt.bar(topR["region"], topR["sales_2024_total"])
    plt.xticks(rotation=75, ha="right")
    plt.title("2024 지역별 판매량 Top 10")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "region_2024_top10.png"), dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="현대차 판매데이터 기반(가격 없이) 모델 추천")
    parser.add_argument("--mode", type=str, default="overall", help="overall|bestseller|rising|stable")
    parser.add_argument("--top", type=int, default=10, help="추천 개수")
    parser.add_argument("--min_volume", type=float, default=0, help="2024 판매량 최소 기준(너무 작은 모델 제외)")
    parser.add_argument("--no_plot", action="store_true", help="차트 저장 생략")
    args = parser.parse_args()

    model_paths  = find_existing_paths(MODEL_FILES)
    region_paths = find_existing_paths(REGION_FILES)

    metrics = build_model_metrics(model_paths)
    region_sum = build_region_summary(region_paths)

    reco = make_recommendations(metrics, mode=args.mode, top_n=args.top, min_volume=args.min_volume)

    print("\n=== 추천 결과 ===")
    print(reco.to_string(index=False))

    print("\n[참고] 2024 지역별 판매량 Top 5")
    print(region_sum[["region","sales_2024_total"]].head(5).to_string(index=False))

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"{OUT_DIR_BASE}_{args.mode}_{stamp}"
    os.makedirs(out_dir, exist_ok=True)

    metrics.to_csv(os.path.join(out_dir, "model_metrics_all.csv"), index=False, encoding="utf-8-sig")
    reco.to_csv(os.path.join(out_dir, "recommendations.csv"), index=False, encoding="utf-8-sig")
    region_sum.to_csv(os.path.join(out_dir, "region_summary.csv"), index=False, encoding="utf-8-sig")

    if not args.no_plot:
        save_charts(out_dir, reco, region_sum)

    print(f"\n저장 완료: {out_dir}/recommendations.csv (및 model_metrics_all.csv, region_summary.csv)")
    if not args.no_plot:
        print(f"차트: {out_dir}/reco_score_bar.png , {out_dir}/region_2024_top10.png")


if __name__ == "__main__":
    main()
