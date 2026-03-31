import torch
import torch.nn as nn
import torchvision.models as models


class CNNEncoder(nn.Module):
    def __init__(self, out_dim: int = 512, backbone_type: str = 'resnet18', fine_tune: bool = True):
        super().__init__()
        if backbone_type == 'resnet18':
            backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            in_features = 512
        else:
            backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            in_features = 2048

        # Loại bỏ 2 layer cuối (AdaptiveAvgPool2d và Linear) để lấy spatial features
        modules = list(backbone.children())[:-2]
        self.feature_extractor = nn.Sequential(*modules)

        # Cải tiến: Cho phép Fine-tune block cuối cùng (layer4) để đặc trưng phù hợp hơn với text
        for name, param in self.feature_extractor.named_parameters():
            if fine_tune and "layer4" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.proj = nn.Sequential(
            nn.Conv2d(in_features, out_dim, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(0.2)  # Thêm regularization
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.feature_extractor(images)
        feats = self.proj(feats)
        b, c, h, w = feats.shape
        # Flatten spatial dimensions: (B, C, H*W) -> (B, H*W, C)
        feats = feats.view(b, c, h * w).permute(0, 2, 1)
        return feats


class AdditiveAttention(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, attn_dim: int):
        super().__init__()
        self.feature_proj = nn.Linear(feature_dim, attn_dim)
        self.hidden_proj = nn.Linear(hidden_dim, attn_dim)
        self.score = nn.Linear(attn_dim, 1)

    def forward(self, image_feats: torch.Tensor, decoder_hidden: torch.Tensor):
        # image_feats: (B, num_pixels, feature_dim)
        # decoder_hidden: (B, hidden_dim)

        feat_proj = self.feature_proj(image_feats)  # (B, num_pixels, attn_dim)
        hid_proj = self.hidden_proj(decoder_hidden).unsqueeze(1)  # (B, 1, attn_dim)

        energy = torch.tanh(feat_proj + hid_proj)  # (B, num_pixels, attn_dim)
        attn_scores = self.score(energy).squeeze(-1)  # (B, num_pixels)

        attn_weights = torch.softmax(attn_scores, dim=1)  # (B, num_pixels)

        # Tính context vector
        context = torch.bmm(attn_weights.unsqueeze(1), image_feats)  # (B, 1, feature_dim)
        context = context.squeeze(1)  # (B, feature_dim)

        return context, attn_weights


class CaptionDecoderLSTM(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, feature_dim: int, attn_dim: int, pad_idx: int,
                 drop_prob: float = 0.5):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        # Cải tiến: Khởi tạo hidden state và cell state từ hình ảnh
        self.init_h = nn.Linear(feature_dim, hidden_dim)
        self.init_c = nn.Linear(feature_dim, hidden_dim)

        # Cải tiến: Dùng LSTMCell để xử lý từng bước (step-by-step), kết hợp image context vào input
        self.lstm_cell = nn.LSTMCell(embed_dim + feature_dim, hidden_dim)
        self.attention = AdditiveAttention(feature_dim=feature_dim, hidden_dim=hidden_dim, attn_dim=attn_dim)

        self.dropout = nn.Dropout(drop_prob)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def init_hidden_state(self, image_feats: torch.Tensor):
        # Lấy trung bình toàn bộ đặc trưng ảnh để khởi tạo trạng thái ban đầu
        mean_image_feats = image_feats.mean(dim=1)
        h = self.init_h(mean_image_feats)
        c = self.init_c(mean_image_feats)
        return h, c

    def forward(self, image_feats: torch.Tensor, caption_in: torch.Tensor):
        """
        caption_in: (B, seq_len) - Đã được shift right (có <sos> ở đầu)
        """
        batch_size = image_feats.size(0)
        seq_len = caption_in.size(1)

        # Khởi tạo h và c
        h, c = self.init_hidden_state(image_feats)

        embeddings = self.embedding(caption_in)  # (B, seq_len, embed_dim)
        embeddings = self.dropout(embeddings)

        predictions = torch.zeros(batch_size, seq_len, self.vocab_size).to(image_feats.device)
        all_alphas = torch.zeros(batch_size, seq_len, image_feats.size(1)).to(image_feats.device)

        for t in range(seq_len):
            # 1. Tính attention tại bước t dựa trên hidden state của bước t-1
            context, alpha = self.attention(image_feats, h)

            # 2. Ghép embedding của từ hiện tại với context của hình ảnh
            lstm_input = torch.cat([embeddings[:, t, :], context], dim=1)

            # 3. Cập nhật LSTM
            h, c = self.lstm_cell(lstm_input, (h, c))

            # 4. Dự đoán từ tiếp theo
            output = self.output(self.dropout(h))

            predictions[:, t, :] = output
            all_alphas[:, t, :] = alpha

        return predictions, all_alphas


class ImageCaptioningModel(nn.Module):
    """Refactored Network: CNN Encoder + Step-by-Step Attention LSTM"""

    def __init__(self, vocab_size: int, pad_idx: int, embed_dim: int = 256, hidden_dim: int = 512,
                 feature_dim: int = 512, attn_dim: int = 256):
        super().__init__()
        self.encoder = CNNEncoder(out_dim=feature_dim, backbone_type='resnet18', fine_tune=True)
        self.decoder = CaptionDecoderLSTM(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            feature_dim=feature_dim,
            attn_dim=attn_dim,
            pad_idx=pad_idx
        )

    def forward(self, images: torch.Tensor, caption_in: torch.Tensor):
        image_feats = self.encoder(images)
        logits, attn_weights = self.decoder(image_feats, caption_in)
        return logits, attn_weights