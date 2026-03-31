import os
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    pbar = tqdm(loader, desc="Training", leave=False)
    
    for images, caption_in, caption_out, _ in pbar:
        images = images.to(device)
        caption_in = caption_in.to(device)
        caption_out = caption_out.to(device)

        optimizer.zero_grad()
        
        outputs = model(images, caption_in)
        if isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs
            
        loss = criterion(logits.reshape(-1, logits.size(-1)), caption_out.reshape(-1))
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item()
        pbar.set_postfix(loss=loss.item())

    return running_loss / max(len(loader), 1)

@torch.no_grad()
def evaluate_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0

    pbar = tqdm(loader, desc="Evaluating", leave=False)
    
    for images, caption_in, caption_out, _ in pbar:
        images = images.to(device)
        caption_in = caption_in.to(device)
        caption_out = caption_out.to(device)

        outputs = model(images, caption_in)
        if isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs
            
        loss = criterion(logits.reshape(-1, logits.size(-1)), caption_out.reshape(-1))
        
        running_loss += loss.item()
        pbar.set_postfix(loss=loss.item())

    return running_loss / max(len(loader), 1)

def train_model(model, train_loader, val_loader, optimizer, criterion, device, epochs=20, save_path="best_model.pth"):
    best_val_loss = float("inf")
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch:02d}/{epochs}")
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate_one_epoch(model, val_loader, criterion, device)
        
        print(f"-> train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            print(f"🌟 Val loss improved from {best_val_loss:.4f} to {val_loss:.4f}. Saving model...")
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            
    return best_val_loss

@torch.no_grad()
def generate_caption(model, image: torch.Tensor, vocab: dict, decode_ids_fn, idx2token: dict, device, max_len: int = 24) -> str:
    model.eval()
    image = image.unsqueeze(0).to(device)
    image_feats = model.encoder(image)

    # Nếu dùng network2, cnn_feats phải đi qua attention và transformer trước
    if hasattr(model, 'transformer') and hasattr(model, 'attention'):
        attn_feats = model.attention(image_feats)
        image_feats = model.transformer(attn_feats)

    tokens = [vocab["<BOS>"]]
    for _ in range(max_len - 1):
        caption_in = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
        
        if hasattr(model, 'transformer'):
             # Network 2: decode chỉ trả về logits
             logits = model.decoder(image_feats, caption_in)
        else:
             # Network 1: decode trả về logits, attn_weights
             logits, _ = model.decoder(image_feats, caption_in)
             
        next_token = int(torch.argmax(logits[0, -1]).item())
        tokens.append(next_token)
        if next_token == vocab["<EOS>"]:
            break

    return decode_ids_fn(tokens, idx2token)

def predict_folder(model, folder_path, vocab, decode_ids_fn, idx2token, device, image_size=224, max_len=24):
    from torchvision import transforms
    if not os.path.exists(folder_path):
        print(f"Folder {folder_path} does not exist.")
        return
        
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    inv_norm = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
    )
    
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        print(f"No images found in {folder_path}.")
        return
        
    for file in image_files:
        img_path = os.path.join(folder_path, file)
        try:
            with Image.open(img_path) as im:
                image_rgb = im.convert("RGB")
            
            image_tensor = transform(image_rgb)
            pred_caption = generate_caption(model, image_tensor, vocab, decode_ids_fn, idx2token, device, max_len=max_len)
            
            show_img = inv_norm(image_tensor).clamp(0, 1).permute(1, 2, 0).cpu()
            plt.figure(figsize=(6, 6))
            plt.imshow(show_img)
            plt.title(f"{file}\nPred: {pred_caption}")
            plt.axis("off")
            plt.show()
        except Exception as e:
            print(f"Error predicting {file}: {e}")
