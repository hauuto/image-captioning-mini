# -*- coding: utf-8 -*-
import random

import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from tqdm import tqdm
from PIL import Image
import os

def analyze_caption_lengths(data_dict, show_plot=True):
    """
    Phân tích độ dài của các caption (số lượng từ).
    Input: data_dict (dict: {image_path: [captions...]})
    """
    all_captions = []
    for captions in data_dict.values():
        all_captions.extend(captions)

    # Tính độ dài theo từ
    lengths = [len(cap.split()) for cap in all_captions]

    stats = {
        "total_captions": len(all_captions),
        "min_len": np.min(lengths),
        "max_len": np.max(lengths),
        "mean_len": np.mean(lengths),
        "median_len": np.median(lengths),
        "std_dev": np.std(lengths)
    }

    print("=== THỐNG KÊ ĐỘ DÀI CAPTION ===")
    for k, v in stats.items():
        print(f"{k}: {v:.2f}")

    if show_plot:
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.hist(lengths, bins=30, color='skyblue', edgecolor='black')
        plt.title("Phân phối độ dài Caption (Histogram)")
        plt.xlabel("Số từ")
        plt.ylabel("Tần suất")
        plt.axvline(stats['mean_len'], color='red', linestyle='dashed', linewidth=1, label=f"Mean: {stats['mean_len']:.1f}")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.boxplot(lengths, vert=False)
        plt.title("Boxplot độ dài Caption")
        plt.xlabel("Số từ")

        plt.tight_layout()
        plt.show()

    return stats


def analyze_vocabulary(data_dict, top_n=20, show_plot=True):
    """
    Phân tích tần suất từ vựng.
    Input: data_dict (dict: {image_path: [captions...]})
    """
    all_words = []
    for captions in data_dict.values():
        for cap in captions:
            # Tokenize đơn giản (lowercase + split)
            # Bạn có thể thay bằng hàm tokenize phức tạp hơn nếu cần
            words = cap.lower().replace('.', '').replace(',', '').split()
            all_words.extend(words)

    counter = Counter(all_words)
    vocab_size = len(counter)

    # Thống kê từ hiếm
    freqs = list(counter.values())
    single_occurrence = freqs.count(1)

    print("=== THỐNG KÊ TỪ VỰNG ===")
    print(f"Tổng số từ (Tokens): {len(all_words)}")
    print(f"Kích thước từ điển (Unique words): {vocab_size}")
    print(f"Số từ chỉ xuất hiện 1 lần: {single_occurrence} ({single_occurrence/vocab_size*100:.2f}%)")

    if show_plot:
        # Lấy top N từ phổ biến nhất
        most_common = counter.most_common(top_n)
        words, counts = zip(*most_common)

        plt.figure(figsize=(12, 6))
        plt.bar(words, counts, color='lightgreen', edgecolor='black')
        plt.title(f"Top {top_n} từ xuất hiện nhiều nhất")
        plt.xticks(rotation=45)
        plt.ylabel("Số lần xuất hiện")
        plt.show()

    return counter




def analyze_image_specs(data_dict, images_root_path, sample_size=1000):
    """
    Phân tích kích thước và tỉ lệ khung hình của ảnh.
    Input:
        data_dict: dict dữ liệu
        images_root_path: đường dẫn thư mục chứa ảnh
        sample_size: số lượng ảnh để analyze (để None nếu muốn chạy hết)
    """
    image_paths = list(data_dict.keys())
    if sample_size and sample_size < len(image_paths):
        image_paths = random.sample(image_paths, sample_size)

    widths = []
    heights = []
    aspect_ratios = []

    print(f"Đang phân tích {len(image_paths)} ảnh...")
    for path in tqdm(image_paths):
        full_path = os.path.join(images_root_path, path)
        try:
            with Image.open(full_path) as img:
                w, h = img.size
                widths.append(w)
                heights.append(h)
                aspect_ratios.append(w / h)
        except FileNotFoundError:
            continue

    print("=== THỐNG KÊ KÍCH THƯỚC ẢNH ===")
    print(f"Width  - Min: {min(widths)}, Max: {max(widths)}, Mean: {np.mean(widths):.1f}")
    print(f"Height - Min: {min(heights)}, Max: {max(heights)}, Mean: {np.mean(heights):.1f}")

    plt.figure(figsize=(14, 5))

    # Scatter plot Width vs Height
    plt.subplot(1, 2, 1)
    plt.scatter(widths, heights, alpha=0.3, c='purple')
    plt.title("Phân bố Kích thước (Width vs Height)")
    plt.xlabel("Width")
    plt.ylabel("Height")

    # Histogram Aspect Ratio
    plt.subplot(1, 2, 2)
    plt.hist(aspect_ratios, bins=30, color='orange', edgecolor='black')
    plt.title("Phân bố Tỉ lệ khung hình (Aspect Ratio = W/H)")
    plt.xlabel("Aspect Ratio")
    plt.axvline(1.0, color='red', linestyle='--', label='Vuông (1.0)')
    plt.legend()

    plt.tight_layout()
    plt.show()


def check_dataset_integrity(data_dict, images_root_path):
    missing_files = []
    for path in data_dict.keys():
        full_path = os.path.join(images_root_path, path)
        if not os.path.exists(full_path):
            missing_files.append(path)

    if len(missing_files) == 0:
        print("✅ Kiểm tra hoàn tất: Tất cả ảnh đều tồn tại.")
    else:
        print(f"❌ CẢNH BÁO: Tìm thấy {len(missing_files)} ảnh bị thiếu!")
        print("Ví dụ 5 ảnh đầu tiên bị thiếu:", missing_files[:5])
    return missing_files



def visualize_random_samples(data_dict, images_root_path, num_samples=6):
    """Hiển thị lưới các ảnh ngẫu nhiên kèm caption đầu tiên"""
    paths = random.sample(list(data_dict.keys()), num_samples)

    rows = (num_samples + 1) // 2
    plt.figure(figsize=(15, 5 * rows))

    for i, path in enumerate(paths):
        full_path = os.path.join(images_root_path, path)
        captions = data_dict[path]

        try:
            img = Image.open(full_path).convert("RGB")
            plt.subplot(rows, 2, i + 1)
            plt.imshow(img)
            plt.title(f"Captions:\n- {captions[0]}\n- {captions[1]}", loc='left', fontsize=10)
            plt.axis('off')
        except Exception as e:
            print(f"Lỗi đọc ảnh {path}: {e}")

    plt.tight_layout()
    plt.show()