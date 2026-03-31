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
    
    # Do chúng ta chỉ sử dụng Network1 theo cấu trúc có sẵn: .encoder và .decoder
    cnn_feats = model.encoder(image)
    tokens = [vocab["<BOS>"]]
    for _ in range(max_len - 1):
        caption_in = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
        outputs = model.decoder(cnn_feats, caption_in)
        
        if isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs
            
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

def calculate_bleu_score(model, val_loader, vocab, idx2token, decode_ids_fn, device, max_len=24, num_batches=None):
    from nltk.translate.bleu_score import corpus_bleu
    import warnings
    warnings.filterwarnings('ignore') # ignore warning when pred is too short
    
    model.eval()
    references = []
    hypotheses = []
    
    pbar = tqdm(val_loader, desc="Calculating BLEU", leave=False)
    
    for i, (images, caption_in, caption_out, _) in enumerate(pbar):
        if num_batches is not None and i >= num_batches:
            break
            
        # Dùng batch size
        bz = images.size(0)
        
        # Batch evaluation
        for j in range(bz):
            # Target (Reference)
            target_ids = caption_out[j].cpu().tolist()
            target_str = decode_ids_fn(target_ids, idx2token)
            references.append([target_str.split()])  # corpus_bleu cần list các reference (1 reference/hình)
            
            # Predict (Hypothesis)
            # Lưu ý generate_caption expects cpu image if unsqueeze is inside, it already does .to(device)
            pred_str = generate_caption(
                model=model, 
                image=images[j].cpu(), 
                vocab=vocab,
                decode_ids_fn=decode_ids_fn,
                idx2token=idx2token,
                device=device,
                max_len=max_len
            )
            hypotheses.append(pred_str.split())
            
    # Tính toán BLEU
    bleu_1 = corpus_bleu(references, hypotheses, weights=(1.0, 0, 0, 0))
    bleu_2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0))
    bleu_3 = corpus_bleu(references, hypotheses, weights=(0.33, 0.33, 0.33, 0))
    bleu_4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25))
    
    print("--- ĐÁNH GIÁ MÔ HÌNH (BLEU SCORE) ---")
    print(f"BLEU-1: {bleu_1*100:.2f}")
    print(f"BLEU-2: {bleu_2*100:.2f}")
    print(f"BLEU-3: {bleu_3*100:.2f}")
    print(f"BLEU-4: {bleu_4*100:.2f}")
    
    return bleu_1, bleu_2, bleu_3, bleu_4
