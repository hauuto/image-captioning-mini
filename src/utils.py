# -*- coding: utf-8 -*-
import shutil
import os
import kagglehub

from src.config import Config
def download_data():
    if os.path.exists(Config.DATA_DIR):
        print("Data đã tồn tại")
        return
    try:
        cache_path = kagglehub.dataset_download("adityajn105/flickr8k")
        if not os.path.exists(Config.DATA_DIR):
            shutil.move(cache_path, Config.DATA_DIR)
            print("Tải dữ liệu thành coông")
    except Exception as e:
        print(f"[ERROR] {e}")
