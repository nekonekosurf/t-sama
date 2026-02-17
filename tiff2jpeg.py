# -*- coding: utf-8 -*-
"""
tiff_to_jpeg_percentile_top1_255.py

./ 以下のサブフォルダ（例: 88_20251104_船舶, 93_20251028_船舶 など）を走査し、
ファイル名が "QSR-12*.tif"（例: QSR-12_20251028_3_0_L11_SM_SLA.tif）を見つけて
・上位1%（99パーセンタイル）より大きい値は 255
・それ以外は 0〜254 に正規化
として、1つの出力ディレクトリに JPEG 保存します。
"""
import os
import sys
import glob
import math
import numpy as np
from typing import Optional
from tqdm import tqdm

from PIL import Image, ImageFile, UnidentifiedImageError
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

try:
    import rasterio
except ImportError:
    rasterio = None


# ========= 設定 =========
# INPUT_ROOT = "../a_撮影データ/"                   # 走査の起点
INPUT_ROOT = "./tmp_raw"                   # 走査の起点
# OUTPUT_DIR = "./jpeg_out/"           # すべてのJPEGの保存先（1つのディレクトリ）
OUTPUT_DIR = "./tmp_jpeg"           # すべてのJPEGの保存先（1つのディレクトリ）
# TARGET_NAME_PATTERN = "QSR-10_20251025_7_0_L11_SM_SLA.tif" # 対象ファイル名パターン
# TARGET_NAME_PATTERN = "QSR-8_20250601_0_0_L11_SM_SLC.tif" # 対象ファイル名パターン
TARGET_NAME_PATTERN = "*.tif" # 対象ファイル名パターン
OVERWRITE = False                   # Trueで既存JPEGを上書き
# ========================


def safe_makedirs(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def print_stats(arr: np.ndarray, name: str = "array") -> None:
    arr_f = arr.astype(np.float32)
    arr_f = arr_f[np.isfinite(arr_f)]
    if arr_f.size == 0:
        print(f"[WARN] {name}: 有効な値がありません")
        return
    print(f"--- {name} stats ---")
    print("shape:", arr.shape, "dtype:", arr.dtype)
    print("min:", float(np.min(arr_f)))
    print("p1 :", float(np.percentile(arr_f, 1)))
    print("p50:", float(np.percentile(arr_f, 50)))
    print("p99:", float(np.percentile(arr_f, 99)))
    print("max:", float(np.max(arr_f)))
    print("mean:", float(np.mean(arr_f)))
    print("std:", float(np.std(arr_f)))


def load_slc(path: str) -> Optional[np.ndarray]:
    """SLC想定: rasterio で読み、複素数なら絶対値にして float32 化"""
    print(f"Loading SLC: {path}")
    if rasterio is None:
        print("[ERROR] rasterio がインストールされていません。pip install rasterio を実行してください。")
        return None
    try:
        with rasterio.open(path) as src:
            data = src.read(1)
        if np.iscomplexobj(data):
            img = np.abs(data).astype(np.float32)
        else:
            img = np.abs(data.astype(np.float32))
        print_stats(img, "SLC")
        return img
    except Exception as e:
        print(f"[ERROR] Failed to load SLC: {e}")
        return None


def load_sla(path: str) -> Optional[np.ndarray]:
    """SLA想定: Pillow で読み込んで float32 に"""
    print(f"Loading SLA: {path}")
    try:
        img = Image.open(path)
        img_np = np.array(img).astype(np.float32)
        print_stats(img_np, "SLA")
        return img_np
    except (UnidentifiedImageError, OSError, ValueError) as e:
        print(f"[ERROR] Failed to load SLA: {e}")
        return None



def top1_to_255_scale_rest_to_0_254(img: np.ndarray) -> np.ndarray:
    """
    要件: 上位1%（> p99）は255、それ以外（<= p99）は0..254に線形正規化して返す。
    - NaN/Inf は 0 に
    - 下限は 0 にクリップ（SAR 強度/振幅を想定）
    - スケーリングは 0..p99 → 0..254 （係数 254/p99）
    """
    if img is None:
        raise ValueError("img is None")

    arr = img.astype(np.float32)
    arr[~np.isfinite(arr)] = 0.0
    arr = np.maximum(arr, 0.0)

    if arr.size == 0:
        return np.zeros_like(arr, dtype=np.uint8)

    p99 = float(np.percentile(arr, 99.0))
    if not math.isfinite(p99) or p99 <= 0.0:
        # 画像が全ゼロ等の場合はそのままゼロで返す
        print("[Normalize] p99 が 0 または非有限のため全ゼロ出力")
        return np.zeros_like(arr, dtype=np.uint8)

    # ≤ p99 を 0..254 に線形マッピング
    scale = 254.0 / p99
    out = np.empty(arr.shape, dtype=np.uint8)

    # マスク分岐（ベクトル化）
    mask_top1 = arr > p99
    mask_rest = ~mask_top1

    # 0..p99 → 0..254
    out[mask_rest] = np.clip(arr[mask_rest] * scale, 0.0, 254.0).astype(np.uint8)
    # 上位1% → 255
    out[mask_top1] = 255

    # ログ
    count_top1 = int(mask_top1.sum())
    total = arr.size
    print(f"[Normalize] p99 = {p99:.6f}, top1% → 255: {count_top1}/{total} pixels ({count_top1/total*100:.3f}%)")

    return out


def save_as_jpeg(img_uint8: np.ndarray, out_path: str) -> None:
    """グレースケールを3chにしてJPEG保存（quality=100）"""
    if img_uint8.ndim == 2:
        img_rgb = np.stack([img_uint8] * 3, axis=-1)
    else:
        img_rgb = img_uint8
    Image.fromarray(img_rgb).save(out_path, quality=100)
    print(f"[SAVE] {out_path}")


def is_slc_name(path: str) -> bool:
    """簡易判定: ファイル名に 'SLC' を含むか"""
    name = os.path.basename(path).upper()
    return "SLC" in name


def find_target_tifs(root: str, pattern: str) -> list:
    """
    例: root=./, pattern=QSR-12*.tif
    ./88_20251104_船舶/**/QSR-12*.tif, ./93_20251028_船舶/**/QSR-12*.tif など
    サブフォルダを再帰で探します。
    """
    candidates = []
    ship_dirs = glob.glob(os.path.join(root, "*_船舶"))
    for d in ship_dirs:
        candidates += glob.glob(os.path.join(d, "**", pattern), recursive=True)
    candidates += glob.glob(os.path.join(root, "**", pattern), recursive=True)
    return sorted(set(candidates))


def main():
    print("[Start] TIFF → JPEG 変換（上位1%→255／それ以外0..254）")
    print("INPUT_ROOT:", os.path.abspath(INPUT_ROOT))
    print("OUTPUT_DIR:", os.path.abspath(OUTPUT_DIR))
    print("PATTERN   :", TARGET_NAME_PATTERN)
    safe_makedirs(OUTPUT_DIR)

    tifs = find_target_tifs(INPUT_ROOT, TARGET_NAME_PATTERN)
    if not tifs:
        print("[INFO] 対象のTIFFが見つかりませんでした。パターンやパスを確認してください。")
        return

    for tif_path in tqdm(tifs):
        base = os.path.splitext(os.path.basename(tif_path))[0]
        out_jpg = os.path.join(OUTPUT_DIR, f"{base}.jpg")

        # if (not OVERWRITE) and os.path.exists(out_jpg):
        #     print(f"[SKIP] 既に存在: {out_jpg}")
        #     continue

        if "Browse" in tif_path:
            print(f"[SKIP] Browse画像をスキップ: {tif_path}")
            continue

        print(f"\n[PROC] {tif_path}")
        # SLC か SLA かで読込関数を切り替え
        if is_slc_name(tif_path):
            img = load_slc(tif_path)
        else:
            img = load_sla(tif_path)

        if img is None:
            print(f"[SKIP] 読み込み失敗: {tif_path}")
            continue

        try:
            # 正規化（上位1%→255／それ以外0..254）
            img_u8 = top1_to_255_scale_rest_to_0_254(img)
            # img_u8 = tsol(img)
        except Exception as e:  
            print(f"[ERROR] 正規化失敗: {e}")
            continue
        # JPEG保存
        save_as_jpeg(img_u8, out_jpg)

    print("\n[Done] 変換完了")


if __name__ == "__main__":
    main()
