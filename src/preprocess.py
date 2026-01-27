import torch
from collections import Counter
from torch.nn.utils.rnn import pad_sequence
import os

class Vocabulary:
    def __init__(self, freq_threshold):
        self.freq_threshold = freq_threshold

        # 4 token đặc biệt bắt buộc
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {v: k for k, v in self.itos.items()}

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer_eng(text):
        """Chuyển câu thành list các từ, viết thường và bỏ ký tự lạ"""
        text = text.lower().strip()
        # Loại bỏ các dấu câu phổ biến dính liền với từ
        for char in '.,!?;:"\'()[]{}':
            text = text.replace(char, ' ')
        return text.split()

    def build_vocabulary(self, sentence_list):
        """Xây dựng từ điển từ danh sách các câu caption"""
        frequencies = Counter()
        idx = 4 # Bắt đầu index từ 4 (sau các token đặc biệt)

        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                frequencies[word] += 1

        # Chỉ giữ lại từ xuất hiện nhiều hơn ngưỡng threshold
        for word, count in frequencies.items():
            if count >= self.freq_threshold:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1

        print(f"Vocabulary built! Size: {len(self.itos)} words (Threshold: {self.freq_threshold})")

    def numericalize(self, text):
        """Chuyển text thành list các số nguyên (indices)"""
        tokenized_text = self.tokenizer_eng(text)

        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
            for token in tokenized_text
        ]

    def denumericalize(self, indices):
        """Chuyển list số nguyên về lại text (để kiểm tra)"""
        words = []
        for idx in indices:
            if isinstance(idx, torch.Tensor): idx = idx.item()
            word = self.itos.get(idx, "<UNK>")
            if word == "<EOS>": break # Dừng khi gặp thẻ kết thúc
            if word not in ["<PAD>", "<SOS>"]: # Bỏ qua thẻ PAD và SOS
                words.append(word)
        return " ".join(words)


from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

# --- 1. Định nghĩa Image Transforms ---
def get_transforms(config):
    # Transform cho tập Train: Có thêm Random Augmentation
    train_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE + 20, config.IMG_SIZE + 20)), # Resize to ra một chút
        transforms.RandomCrop(config.IMG_SIZE),   # Cắt ngẫu nhiên
        transforms.RandomHorizontalFlip(),        # Lật ngang ngẫu nhiên
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # Transform cho Val/Test: Chỉ Resize và Normalize chuẩn
    val_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    return train_transform, val_transform

# --- 2. Dataset Class ---
class FlickrDataset(Dataset):
    def __init__(self, root_dir, data_dict, vocab, transform=None):
        """
        root_dir: Đường dẫn thư mục chứa ảnh
        data_dict: Dictionary {img_name: [captions]} (đã split)
        vocab: Object Vocabulary đã build
        """
        self.root_dir = root_dir
        self.vocab = vocab
        self.transform = transform

        # Flatten dữ liệu: Mỗi mẫu là 1 cặp (ảnh, 1 caption)
        self.samples = []
        for img_name, captions in data_dict.items():
            for cap in captions:
                self.samples.append((img_name, cap))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_id, caption = self.samples[index]
        img_path = os.path.join(self.root_dir, img_id)

        # 1. Load Ảnh
        try:
            img = Image.open(img_path).convert("RGB")
        except:
            # Fallback nếu ảnh lỗi: lấy ảnh tiếp theo
            return self.__getitem__((index + 1) % len(self.samples))

        if self.transform is not None:
            img = self.transform(img)

        # 2. Xử lý Caption (Numericalize)
        # Thêm <SOS> ở đầu và <EOS> ở cuối
        numericalized_caption = [self.vocab.stoi["<SOS>"]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi["<EOS>"])

        return img, torch.tensor(numericalized_caption)


from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# --- 1. Collate Function (Xử lý Batch & Padding) ---
class CollateFn:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        # Batch là list các tuple (img, caption_tensor)
        imgs = [item[0] for item in batch]
        captions = [item[1] for item in batch]

        # Stack ảnh lại thành 1 Tensor 4 chiều (Batch, 3, H, W)
        imgs = torch.stack(imgs, dim=0)

        # Padding caption: Các câu ngắn sẽ được thêm pad_idx vào sau
        # batch_first=True -> Output shape: (Batch, Max_Len_In_Batch)
        targets = pad_sequence(captions, batch_first=True, padding_value=self.pad_idx)

        return imgs, targets

# --- 2. Hàm Main để lấy DataLoaders ---
def get_loaders(config, data_dict):
    """
    Hàm chính để gọi trong quá trình training.
    Input: config, data_dict (toàn bộ dữ liệu)
    Output: train_loader, val_loader, test_loader, vocab
    """

    # B1: Xây dựng Vocabulary từ TOÀN BỘ dữ liệu
    # (Để tránh lỗi <UNK> quá nhiều ở tập test)
    all_captions = []
    for caps in data_dict.values():
        all_captions.extend(caps)

    vocab = Vocabulary(config.FREQ_THRESHOLD)
    vocab.build_vocabulary(all_captions)

    # B2: Chia tập Train/Val/Test theo ẢNH (Image-based Split)
    # Lý do: Để tránh rò rỉ dữ liệu (Data Leakage)
    all_imgs = list(data_dict.keys())

    # Train 70%, Temp 30%
    train_imgs, temp_imgs = train_test_split(all_imgs, test_size=0.3, random_state=42)
    # Temp -> Val 10%, Test 20% (1/3 của 30% là 10%)
    val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.66, random_state=42)

    # Tạo dictionary con
    train_data = {k: data_dict[k] for k in train_imgs}
    val_data = {k: data_dict[k] for k in val_imgs}
    test_data = {k: data_dict[k] for k in test_imgs}

    print(f"Data Split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)} images")

    # B3: Định nghĩa Transforms
    train_tf, val_tf = get_transforms(config)

    # B4: Tạo Dataset
    train_ds = FlickrDataset(config.IMG_DIR, train_data, vocab, train_tf)
    val_ds = FlickrDataset(config.IMG_DIR, val_data, vocab, val_tf)
    test_ds = FlickrDataset(config.IMG_DIR, test_data, vocab, val_tf)

    # B5: Tạo DataLoader
    pad_idx = vocab.stoi["<PAD>"]

    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        collate_fn=CollateFn(pad_idx),
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        collate_fn=CollateFn(pad_idx),
        pin_memory=True
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        collate_fn=CollateFn(pad_idx)
    )

    return train_loader, val_loader, test_loader, vocab