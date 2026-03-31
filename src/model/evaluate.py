import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu
from PIL import Image
from torchvision import transforms


class Evaluator:
    def __init__(self, model, vocab, device):
        self.model = model
        self.vocab = vocab
        self.device = device
        self.model.eval() # Luôn nhớ bật chế độ eval

    def clean_sentence(self, indices):
        """
        Chuyển list index thành câu hoàn chỉnh, loại bỏ <SOS>, <EOS>, <PAD>
        """
        sentence = []
        for index in indices:
            word = self.vocab.itos.get(index, "<UNK>")
            if word == "<EOS>":
                break
            if word not in ["<SOS>", "<PAD>"]:
                sentence.append(word)
        return sentence

    def calculate_metrics(self, data_loader):
        """
        Tính điểm BLEU-1,2,3,4 trên toàn bộ tập test
        """
        print("📊 Đang tính toán BLEU score...")

        references = [] # Ground Truth (List of Lists)
        hypotheses = [] # Predictions (List)

        with torch.no_grad():
            for imgs, captions in tqdm(data_loader, desc="Evaluating"):
                imgs = imgs.to(self.device)

                # 1. Sinh caption (Generate)
                features = self.model.encoder(imgs)

                # Loop qua từng ảnh trong batch để generate
                for i in range(imgs.size(0)):
                    # Lấy feature của ảnh i: [1, Embed_Size]
                    feature = features[i].unsqueeze(0)

                    # Generate ra list index
                    generated_ids = self.model.decoder.generate(feature, max_len=20, vocab=self.vocab)

                    # Chuyển thành list từ (words)
                    pred_caption = self.clean_sentence(generated_ids)
                    hypotheses.append(pred_caption)

                    # Xử lý caption gốc (Target)
                    target_ids = captions[i].tolist()
                    target_caption = self.clean_sentence(target_ids)

                    # Thêm vào list reference (Lưu ý: corpus_bleu cần list of lists cho reference)
                    references.append([target_caption])

        # 2. Tính điểm bằng NLTK
        bleu1 = corpus_bleu(references, hypotheses, weights=(1.0, 0, 0, 0))
        bleu2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0))
        bleu3 = corpus_bleu(references, hypotheses, weights=(0.33, 0.33, 0.33, 0))
        bleu4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25))

        print("\n" + "="*30)
        print(f"KẾT QUẢ ĐÁNH GIÁ (BLEU SCORE)")
        print("="*30)
        print(f"🔹 BLEU-1: {bleu1*100:.2f}")
        print(f"🔹 BLEU-2: {bleu2*100:.2f}")
        print(f"🔹 BLEU-3: {bleu3*100:.2f}")
        print(f"🌟 BLEU-4: {bleu4*100:.2f} (Quan trọng nhất)")
        print("="*30)

        return bleu4

    def visualize(self, data_loader, num_samples=6):
        """
        Hiển thị ảnh và so sánh caption
        """
        print("🖼️ Đang tạo ảnh minh họa...")
        imgs, captions = next(iter(data_loader))
        imgs = imgs.to(self.device)

        # Un-normalize để hiển thị ảnh đúng màu
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

        plt.figure(figsize=(15, 10))

        for i in range(min(num_samples, imgs.size(0))):
            # 1. Generate
            with torch.no_grad():
                feature = self.model.encoder(imgs[i].unsqueeze(0))
                gen_ids = self.model.decoder.generate(feature, vocab=self.vocab)
                gen_cap = " ".join(self.clean_sentence(gen_ids))

            # 2. Ground Truth
            gt_cap = " ".join(self.clean_sentence(captions[i].tolist()))

            # 3. Xử lý ảnh
            img_np = imgs[i].cpu().permute(1, 2, 0).numpy()
            img_np = std * img_np + mean
            img_np = np.clip(img_np, 0, 1)

            # 4. Vẽ
            plt.subplot(2, 3, i + 1)
            plt.imshow(img_np)
            plt.title(f"Truth: {gt_cap}\nPred: {gen_cap}", fontsize=9, loc='left', color='blue')
            plt.axis("off")

        plt.tight_layout()
        plt.show()

    def predict_image(self, image_path):
        """
        Dự đoán cho 1 file ảnh bất kỳ (.jpg, .png)
        """
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        try:
            image = Image.open(image_path).convert("RGB")
            # Clone ảnh gốc để hiển thị
            original_image = image.copy()

            # Transform để đưa vào model
            img_tensor = transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                caption = self.model.generate_caption(img_tensor, self.vocab)

            plt.figure(figsize=(6, 6))
            plt.imshow(original_image)
            plt.title(f"AI Caption: {caption}", fontsize=12, color='darkgreen', fontweight='bold')
            plt.axis("off")
            plt.show()

            print(f"📝 Caption: {caption}")
            return caption

        except Exception as e:
            print(f"❌ Lỗi: {e}")
            return None