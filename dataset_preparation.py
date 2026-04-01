from __future__ import annotations
import json
from pathlib import Path

import pandas as pd

"""
Pipeline:
  1. Chọn đúng các cột cần thiết
  2. Cắt train/test 80-20 theo thời gian (split_date chung cho mọi Product ID)

Columns giữ lại:
  - Date            : mốc thời gian
  - Product ID      : định danh chuỗi
  - Units Sold      : target (y)
  - Seasonality     : feature
  - Category        : feature
  - Holiday/Promotion: feature

split_date chung: dựa trên 80% số mốc ngày duy nhất
  → mọi Product ID đều train/test trên cùng khoảng thời gian
"""

CSV_PATH = Path("./dataset/retail_store_inventory.csv")
OUT_DIR = Path("./dataset/split")

DATE_COL = "Date"
GROUP_COL = "Product ID"
KEEP_COLS = [
    "Date",
    "Product ID",
    "Units Sold",  # target
    "Seasonality",  # feature
    "Category",  # feature
    "Holiday/Promotion",  # feature
]


# ── Helpers ───────────────────────────────────────────────────────────────────


def select_columns(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in KEEP_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Thiếu cột trong dataset: {missing}")
    return df[KEEP_COLS].copy()


# ── Core split ────────────────────────────────────────────────────────────────


def temporal_train_test_split(
    df: pd.DataFrame,
    date_col: str,
    group_col: str,
    test_ratio: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Chia train/test theo thời gian với split_date CHUNG cho mọi Product ID.

    - Cắt theo số mốc ngày DUY NHẤT (không phải số dòng)
      → tránh mỗi product bị cắt tại thời điểm khác nhau.
    - Kiểm tra unseen product trong test.
    """
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d = d.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)

    unique_dates = d[date_col].sort_values().unique()
    n_dates = len(unique_dates)

    split_idx = int(n_dates * (1.0 - test_ratio))
    split_idx = max(1, min(split_idx, n_dates - 1))
    split_date = pd.Timestamp(unique_dates[split_idx])

    train = d[d[date_col] < split_date].copy()
    test = d[d[date_col] >= split_date].copy()

    # Cảnh báo nếu có product chỉ xuất hiện trong test
    train_groups = set(train[group_col].unique())
    test_groups = set(test[group_col].unique())
    unseen = test_groups - train_groups
    if unseen:
        print(
            f"[WARN] {len(unseen)} Product ID có trong test nhưng KHÔNG có trong train: {unseen}"
        )

    meta = {
        "date_col": date_col,
        "group_col": group_col,
        "keep_cols": KEEP_COLS,
        "split_date": split_date.isoformat(),
        "n_unique_dates": int(n_dates),
        "n_total": int(len(d)),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "test_ratio": float(test_ratio),
        "n_products": int(d[group_col].nunique()),
        "train_date_range": [
            train[date_col].min().date().isoformat(),
            train[date_col].max().date().isoformat(),
        ],
        "test_date_range": [
            test[date_col].min().date().isoformat(),
            test[date_col].max().date().isoformat(),
        ],
    }
    return train, test, meta


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"Không thấy {CSV_PATH}. Đổi CSV_PATH hoặc đặt file vào đúng thư mục."
        )

    df = pd.read_csv(CSV_PATH)

    print(f"[INFO] Đọc dataset gốc: {len(df):,} dòng, {len(df.columns)} cột")

    # Bước 1: Chọn cột
    df = select_columns(df)
    print(f"[INFO] Sau khi chọn cột: {list(df.columns)}")

    # Bước 2: Cắt train/test
    train, test, meta = temporal_train_test_split(
        df, DATE_COL, GROUP_COL, test_ratio=0.2
    )

    # Lưu kết quả
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(OUT_DIR / "train.csv", index=False)
    test.to_csv(OUT_DIR / "test.csv", index=False)
    (OUT_DIR / "split_config.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(json.dumps(meta, indent=2, ensure_ascii=False))
    print(
        f"\n[OK] train.csv ({meta['n_train']:,} dòng) | "
        f"test.csv ({meta['n_test']:,} dòng) → {OUT_DIR}/"
    )


if __name__ == "__main__":
    main()
