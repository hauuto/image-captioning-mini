# -*- coding: utf-8 -*-
from datetime import datetime
import os

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np

# Import các module đã viết
from src.config import Config
from src.model import CNNtoRNN

def save_checkpoint(state, filename="best_model.pth"):
    """Lưu checkpoint model"""
    print(f"=> Saving checkpoint to {filename}")
    torch.save(state, filename)

def train_one_epoch(loader, model, optimizer, criterion, device):
    """Huấn luyện 1 Epoch"""
    model.train()
    losses = []

    # Thanh progress bar
    loop = tqdm(loader, total=len(loader), leave=True)

    for batch_idx, (imgs, captions) in enumerate(loop):
        # 1. Đưa dữ liệu vào GPU/CPU
        imgs = imgs.to(device)
        captions = captions.to(device)

        # 2. Forward Pass
        # outputs shape: [Batch, Seq_Len-1, Vocab_Size]
        outputs = model(imgs, captions)

        # 3. Chuẩn bị tính Loss
        # Target: Là caption gốc nhưng bỏ từ <SOS> ở đầu (dịch đi 1 bước)
        # targets shape: [Batch, Seq_Len-1]
        targets = captions[:, 1:]

        # Reshape về 2D để tính CrossEntropy
        # Output: [(Batch * Seq_Len), Vocab_Size]
        # Target: [(Batch * Seq_Len)]
        loss = criterion(outputs.reshape(-1, outputs.shape[2]), targets.reshape(-1))

        # 4. Backward Pass
        optimizer.zero_grad()
        loss.backward()

        # 5. Gradient Clipping (Tránh bùng nổ gradient - Quan trọng cho LSTM)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        # 6. Update weights
        optimizer.step()

        # 7. Log
        losses.append(loss.item())
        loop.set_description(f"Train Loss: {loss.item():.4f}")

    return np.mean(losses)

def validate(loader, model, criterion, device):
    """Đánh giá model trên tập Val"""
    model.eval()
    losses = []

    with torch.no_grad():
        loop = tqdm(loader, total=len(loader), desc="Validating", leave=True)
        for imgs, captions in loop:
            imgs = imgs.to(device)
            captions = captions.to(device)

            outputs = model(imgs, captions)
            targets = captions[:, 1:]

            loss = criterion(outputs.reshape(-1, outputs.shape[2]), targets.reshape(-1))
            losses.append(loss.item())

    return np.mean(losses)

# --- MAIN TRAINING FUNCTION ---
def run_training(train_loader, val_loader, test_loader, vocab, experiment_name = None):
    # 1. Tạo thư mục checkpoints
    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")

    # 2. TỰ ĐỘNG HÓA TÊN FILE (Auto-Naming Logic)
    # Nếu người dùng không điền tên, tự động tạo tên dựa trên Config
    if experiment_name is None:
        # Ví dụ: "bs32_lr0.0003_emb256"
        experiment_name = f"bs{Config.BATCH_SIZE}_lr{Config.LEARNING_RATE}_emb{Config.EMBED_SIZE}"

    # Thêm timestamp để đảm bảo DUY NHẤT (không bao giờ trùng)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M") # Ví dụ: 20260127_2030

    # Tên file cuối cùng: checkpoints/bs32_lr0.0003_emb256_20260127_2030.pth
    checkpoint_filename = f"checkpoints/{experiment_name}_{timestamp}.pth"

    print(f"✅ Auto-save mode: Model sẽ được lưu tại:\n   -> {checkpoint_filename}")

    # 1. Khởi tạo Model
    model = CNNtoRNN(
        embed_size=Config.EMBED_SIZE,
        hidden_size=Config.HIDDEN_SIZE,
        vocab_size=len(vocab),
        num_layers=Config.NUM_LAYERS,
        train_cnn=Config.TRAIN_CNN,
        dropout=Config.DROPOUT
    ).to(Config.DEVICE)

    # 2. Loss & Optimizer
    # QUAN TRỌNG: ignore_index giúp model không tính loss cho phần padding
    pad_idx = vocab.stoi["<PAD>"]
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    # Chỉ optimize các tham số có requires_grad=True (ResNet đã bị freeze)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=Config.LEARNING_RATE)

    # Scheduler: Giảm LR nếu val_loss không giảm sau 2 epochs
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )

    # 3. Training Loop
    best_val_loss = float("inf")

    print(f"Bắt đầu huấn luyện trên {Config.DEVICE}...")
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    for epoch in range(Config.EPOCHS):
        print(f"\nEpoch [{epoch+1}/{Config.EPOCHS}]")

        # Train
        train_loss = train_one_epoch(train_loader, model, optimizer, criterion, Config.DEVICE)

        # Validate
        val_loss = validate(val_loader, model, criterion, Config.DEVICE)

        # Update Scheduler
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\t>>> Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"\t>>> Current Learning Rate: {current_lr}")

        # Save Best Model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "val_loss": val_loss,
                "vocab": vocab.__dict__,
                "config": {
                    "embed": Config.EMBED_SIZE,
                    "hidden": Config.HIDDEN_SIZE,
                    "lr": Config.LEARNING_RATE
                }
            }
            # Lưu với tên file tự động
            save_checkpoint(checkpoint, filename=checkpoint_filename)
            print(f"\t>>> Đã cập nhật model tốt nhất!")

        # --- Demo: Sinh thử caption sau mỗi epoch để xem tiến bộ ---
        model.eval()
        with torch.no_grad():
            # Lấy 1 ảnh từ tập test
            img_tensor, _ = next(iter(test_loader))
            img_tensor = img_tensor[0].to(Config.DEVICE) # Lấy ảnh đầu tiên trong batch

            generated_cap = model.generate_caption(img_tensor, vocab)
            print(f"\t>>> Demo Caption: {generated_cap}")

    print("\nHOÀN TẤT HUẤN LUYỆN!")