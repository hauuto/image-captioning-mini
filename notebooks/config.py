import torch

class Config:
    # --- ĐƯỜNG DẪN DỮ LIỆU ---
    # Bạn đổi lại đường dẫn này theo máy của bạn nếu chạy local
    # Nếu chạy Colab thì giữ nguyên để nó tự tải về
    DATA_DIR = "./data/flickr8k"
    IMAGES_DIR = f"{DATA_DIR}/Images"
    CAPTIONS_FILE = f"{DATA_DIR}/captions.txt"

    # --- THAM SỐ MODEL ---
    EMBED_SIZE = 256
    HIDDEN_SIZE = 512
    NUM_LAYERS = 1
    DROPOUT = 0.5           # (Mới) Giúp giảm overfitting
    TRAIN_CNN = False       # (Mới) Có train lại ResNet không? (Fine-tuning)

    # --- THAM SỐ HUẤN LUYỆN ---
    LEARNING_RATE = 3e-4
    BATCH_SIZE = 32
    EPOCHS = 10             # Tăng lên vì có validation
    NUM_WORKERS = 2

    # --- XỬ LÝ DỮ LIỆU ---
    FREQ_THRESHOLD = 2      # (Thay đổi) Lấy từ code mới (threshold=2)
    IMAGE_SIZE = 224

    # --- THIẾT BỊ ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
