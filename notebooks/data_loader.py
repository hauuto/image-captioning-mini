import os
import pandas as pd
import torch
import random
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
from collections import Counter
from sklearn.model_selection import train_test_split
from config import Config

class Vocabulary:
    def __init__(self, freq_threshold):
        self.freq_threshold = freq_threshold
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {v: k for k, v in self.itos.items()}

    def __len__(self):
        return len(self.itos)

    def tokenize(self, text):
        """Xử lý text: lowercase và bỏ dấu câu cơ bản"""
        text = text.lower().strip()
        for char in '.,!?;:"\'()[]{}':
            text = text.replace(char, ' ')
        return text.split()

    def build_vocab(self, sentence_list):
        frequencies = Counter()
        idx = 4
        for sentence in sentence_list:
            for word in self.tokenize(sentence):
                frequencies[word] += 1

        for word, freq in frequencies.items():
            if freq >= self.freq_threshold:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1

    def numericalize(self, text):
        tokenized_text = self.tokenize(text)
        return [self.stoi.get(token, self.stoi["<UNK>"]) for token in tokenized_text]

    def denumericalize(self, indices):
        """Chuyển số về lại chữ"""
        words = []
        for idx in indices:
            if isinstance(idx, torch.Tensor): idx = idx.item()
            word = self.itos.get(idx, "<UNK>")
            if word == "<EOS>": break
            if word not in ["<PAD>", "<SOS>"]: words.append(word)
        return ' '.join(words)

# --- 2. DATASET CLASS ---
class FlickrDataset(Dataset):
    def __init__(self, data_dict, images_dir, vocab, transform=None):
        """
        data_dict: Dictionary {image_name: [list_of_captions]}
        """
        self.images_dir = images_dir
        self.vocab = vocab
        self.transform = transform

        # Flatten dữ liệu: mỗi mẫu là 1 cặp (ảnh, 1 caption)
        self.samples = []
        for img_name, captions in data_dict.items():
            for cap in captions:
                self.samples.append((img_name, cap))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_name, caption = self.samples[index]
        img_path = os.path.join(self.images_dir, img_name)

        try:
            img = Image.open(img_path).convert("RGB")
        except:
            return self.__getitem__((index + 1) % len(self.samples)) # Fallback

        if self.transform:
            img = self.transform(img)

        # Chuyển text sang vector số
        cap_vec = [self.vocab.stoi["<SOS>"]]
        cap_vec += self.vocab.numericalize(caption)
        cap_vec.append(self.vocab.stoi["<EOS>"])

        return img, torch.tensor(cap_vec)

# --- 3. COLLATE FUNCTION ---
class CollateFn:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        imgs = [item[0] for item in batch]
        captions = [item[1] for item in batch]

        imgs = torch.stack(imgs, dim=0)
        targets = pad_sequence(captions, batch_first=True, padding_value=self.pad_idx)

        return imgs, targets, torch.tensor([len(c) for c in captions])

# --- 4. HÀM CHIA DATASET (SPLIT) ---
def get_loaders(batch_size=32, num_workers=2, transform=None):
    # Đọc dữ liệu thô
    df = pd.read_csv(Config.CAPTIONS_FILE)

    # Gom nhóm caption theo ảnh
    data_dict = {}
    for img, cap in zip(df['image'], df['caption']):
        data_dict.setdefault(img.strip(), []).append(str(cap))

    all_imgs = list(data_dict.keys())

    # Build Vocab từ toàn bộ dữ liệu (tránh OOV ở tập test)
    all_captions = [cap for caps in data_dict.values() for cap in caps]
    vocab = Vocabulary(Config.FREQ_THRESHOLD)
    vocab.build_vocab(all_captions)

    # Chia tập Train (70%), Val (10%), Test (20%)
    train_imgs, temp_imgs = train_test_split(all_imgs, test_size=0.3, random_state=42)
    val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.66, random_state=42) # 0.66 of 0.3 ~ 0.2 total

    # Tạo dictionary cho từng tập
    train_data = {img: data_dict[img] for img in train_imgs}
    val_data = {img: data_dict[img] for img in val_imgs}
    test_data = {img: data_dict[img] for img in test_imgs}

    # Tạo Datasets
    train_ds = FlickrDataset(train_data, Config.IMAGES_DIR, vocab, transform)
    val_ds = FlickrDataset(val_data, Config.IMAGES_DIR, vocab, transform)
    test_ds = FlickrDataset(test_data, Config.IMAGES_DIR, vocab, transform)

    pad_idx = vocab.stoi["<PAD>"]

    # Tạo Loaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, collate_fn=CollateFn(pad_idx), pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, collate_fn=CollateFn(pad_idx), pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, collate_fn=CollateFn(pad_idx))

    return train_loader, val_loader, test_loader, vocab
