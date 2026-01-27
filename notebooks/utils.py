import os
import shutil
import torch
import numpy as np
import kagglehub
from collections import Counter
from config import Config

def download_data():
    if os.path.exists(Config.DATA_DIR):
        print(f"[INFO] Dataset đã tồn tại tại {Config.DATA_DIR}")
        return
    print("[INFO] Đang tải dataset từ Kaggle...")
    try:
        cache_path = kagglehub.dataset_download("adityajn105/flickr8k")
        if not os.path.exists(Config.DATA_DIR):
            shutil.copytree(cache_path, Config.DATA_DIR)
            print(f"[INFO] Đã di chuyển dataset về {Config.DATA_DIR}")
    except Exception as e:
        print(f"[ERROR] Lỗi tải dữ liệu: {e}")

def save_checkpoint(state, filename="best_model.pth"):
    torch.save(state, filename)

def load_checkpoint(checkpoint_path, model, optimizer):
    checkpoint = torch.load(checkpoint_path, map_location=Config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["val_loss"]

# --- HÀM TÍNH BLEU SCORE (MỚI) ---
def calculate_bleu_score(reference, candidate, n=4):
    """Tính BLEU score đơn giản (theo code bạn cung cấp)"""
    ref_tokens = reference.lower().split()
    cand_tokens = candidate.lower().split()
    if len(cand_tokens) == 0: return 0.0

    bp = min(1.0, np.exp(1 - len(ref_tokens) / max(len(cand_tokens), 1)))
    precisions = []

    for i in range(1, n + 1):
        ref_ngrams = Counter([tuple(ref_tokens[j:j+i]) for j in range(len(ref_tokens)-i+1)])
        cand_ngrams = Counter([tuple(cand_tokens[j:j+i]) for j in range(len(cand_tokens)-i+1)])
        overlap = sum(min(cand_ngrams[ng], ref_ngrams[ng]) for ng in cand_ngrams)
        total = sum(cand_ngrams.values())
        precisions.append(overlap / total if total > 0 else 0)

    if 0 in precisions: return 0.0
    log_precision = sum(np.log(p) for p in precisions) / n
    return bp * np.exp(log_precision)
