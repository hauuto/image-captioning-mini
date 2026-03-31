import os
import re
import random
from collections import Counter
from typing import Dict, List, Tuple

import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader

SPECIAL_TOKENS = {
    "<PAD>": 0,
    "<BOS>": 1,
    "<EOS>": 2,
    "<UNK>": 3,
}

def load_captions_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "image" in df.columns and "caption" in df.columns:
        out = df[["image", "caption"]].copy()
    else:
        out = df.iloc[:, :2].copy()
        out.columns = ["image", "caption"]
    out["image"] = out["image"].astype(str)
    out["caption"] = out["caption"].astype(str)
    return out

def normalize_caption(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def build_vocab(df: pd.DataFrame, min_freq: int = 1) -> Dict[str, int]:
    counter = Counter()
    for caption in df["caption"].tolist():
        counter.update(normalize_caption(caption).split())

    vocab = dict(SPECIAL_TOKENS)
    for token, freq in counter.items():
        if freq >= min_freq and token not in vocab:
            vocab[token] = len(vocab)
    return vocab

def encode_caption(caption: str, vocab: Dict[str, int], max_len: int = 24) -> List[int]:
    tokens = normalize_caption(caption).split()
    token_ids = [vocab["<BOS>"]]
    token_ids += [vocab.get(tok, vocab["<UNK>"]) for tok in tokens][: max_len - 2]
    token_ids.append(vocab["<EOS>"])
    return token_ids

def decode_ids(ids: List[int], idx2token: Dict[int, str]) -> str:
    words = []
    for idx in ids:
        tok = idx2token.get(int(idx), "<UNK>")
        if tok in ("<PAD>", "<BOS>"):
            continue
        if tok == "<EOS>":
            break
        words.append(tok)
    return " ".join(words)

class ImageCaptionDataset(Dataset):
    def __init__(self, image_dir: str, captions_df: pd.DataFrame, vocab: Dict[str, int], transform=None, max_len: int = 24):
        self.image_dir = image_dir
        self.vocab = vocab
        self.transform = transform
        self.max_len = max_len
        self.captions_by_image = captions_df.groupby("image")["caption"].apply(list).to_dict()
        self.image_names = list(self.captions_by_image.keys())

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx: int):
        image_name = self.image_names[idx]
        caption = random.choice(self.captions_by_image[image_name])
        caption_ids = encode_caption(caption, self.vocab, max_len=self.max_len)

        image_path = os.path.join(self.image_dir, image_name)
        with Image.open(image_path) as im:
            image = im.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, torch.tensor(caption_ids, dtype=torch.long), image_name

class CollateWrapper:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        images, captions, names = zip(*batch)
        images = torch.stack(images, dim=0)

        max_len = max(len(c) for c in captions)
        padded = torch.full((len(captions), max_len), self.pad_idx, dtype=torch.long)
        for i, cap in enumerate(captions):
            padded[i, : len(cap)] = cap

        caption_in = padded[:, :-1]
        caption_out = padded[:, 1:]
        return images, caption_in, caption_out, names

def get_dataloaders(images_dir, captions_path, batch_size, image_size, max_len, min_freq=1, num_workers=0):
    from torchvision import transforms
    captions_df = load_captions_df(captions_path)
    vocab = build_vocab(captions_df, min_freq=min_freq)
    
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    dataset = ImageCaptionDataset(
        image_dir=images_dir,
        captions_df=captions_df,
        vocab=vocab,
        transform=transform,
        max_len=max_len,
    )
    
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    
    collate_fn = CollateWrapper(vocab["<PAD>"])
    
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )
    return train_loader, val_loader, vocab, captions_df

def visualize_dataset_sample(images_dir, captions_df):
    captions_by_image = captions_df.groupby("image")["caption"].apply(list).to_dict()
    sample_key = next(iter(captions_by_image))
    
    print("Sample image:", sample_key)
    print("Num captions for sample:", len(captions_by_image[sample_key]))
    print("Captions:")
    for cap in captions_by_image[sample_key][:3]:
        print("- ", cap)
        
    image_path = os.path.join(images_dir, sample_key)
    try:
        with Image.open(image_path) as im:
            plt.figure(figsize=(4, 4))
            plt.imshow(im)
            plt.axis("off")
            plt.title(f"Sample: {sample_key}")
            plt.show()
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
