import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from PIL import Image
from collections import Counter
from config import Config
from utils import download_data

def run_analysis():
    download_data()
    print("=== BÁO CÁO PHÂN TÍCH DỮ LIỆU (EDA - SUPER) ===")

    df = pd.read_csv(Config.CAPTIONS_FILE)

    # 1. Image Size Analysis
    print("\n[Analyzing Image Sizes...]")
    widths, heights = [], []
    # Sample 500 ảnh để check size cho nhanh
    for img_name in df['image'].unique()[:500]:
        try:
            with Image.open(os.path.join(Config.IMAGES_DIR, img_name)) as img:
                widths.append(img.width)
                heights.append(img.height)
        except: pass

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    sns.scatterplot(x=widths, y=heights, alpha=0.5)
    plt.title("Image Dimensions Distribution")
    plt.xlabel("Width")
    plt.ylabel("Height")

    # 2. Caption Length Analysis
    print("[Analyzing Caption Lengths...]")
    df['length'] = df['caption'].apply(lambda x: len(str(x).split()))

    plt.subplot(1, 2, 2)
    sns.histplot(df['length'], bins=30, kde=True, color='orange')
    plt.title("Caption Length Distribution")
    plt.axvline(x=df['length'].mean(), color='red', linestyle='--', label=f"Mean: {df['length'].mean():.1f}")
    plt.legend()
    plt.show()

    # 3. Vocab Stats
    all_words = " ".join(df['caption'].astype(str)).lower().split()
    vocab_size = len(Counter(all_words))
    print(f"\n- Tổng số ảnh: {df['image'].nunique()}")
    print(f"- Tổng số từ vựng (raw): {vocab_size}")
    print(f"- Caption dài nhất: {df['length'].max()} từ")
