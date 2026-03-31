import torch
import torch.nn as nn
import torchvision.models as models


# ==========================================
# 1. CNN Encoder
# ==========================================
class CNNEncoder(nn.Module):
    def __init__(self, out_dim: int = 512):
        super().__init__()
        # Sử dụng ResNet18 làm backbone
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Bỏ 2 lớp cuối (AdaptiveAvgPool2d và Linear)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-2])
        self.proj = nn.Conv2d(512, out_dim, kernel_size=1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.feature_extractor(images)  # Shape: (B, 512, H, W)
        feats = self.proj(feats)  # Shape: (B, out_dim, H, W)
        b, c, h, w = feats.shape
        # Flatten (H, W) thành chuỗi các patches
        feats = feats.view(b, c, h * w).permute(0, 2, 1)  # Shape: (B, H*W, out_dim)
        return feats


# ==========================================
# 2. Attention Layer (Self-Attention)
# ==========================================
class SpatialAttention(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int = 8):
        super().__init__()
        # Self-Attention để làm nổi bật các patch ảnh quan trọng
        self.multihead_attn = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, Seq_len, feature_dim)
        attn_out, attn_weights = self.multihead_attn(x, x, x)
        # Residual connection + LayerNorm
        out = self.norm(x + attn_out)
        return out


# ==========================================
# 3. Transformer Encoder
# ==========================================
class TransformerBlock(nn.Module):
    def __init__(self, feature_dim: int, num_layers: int = 2, num_heads: int = 8):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            batch_first=True,
            dim_feedforward=feature_dim * 4,
            dropout=0.1
        )
        # Transformer giúp mô hình hóa mối quan hệ toàn cục giữa các patch ảnh
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.transformer(x)


# ==========================================
# 4. LSTM Decoder
# ==========================================
class LSTMDecoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, feature_dim: int, pad_idx: int,
                 num_layers: int = 1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        # LSTM nhận đầu vào là sự kết hợp (concatenate) giữa embedding text và đặc trưng ảnh
        self.lstm = nn.LSTM(
            input_size=embed_dim + feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.0 if num_layers == 1 else 0.2
        )

        # Lớp tuyến tính để khởi tạo trạng thái ẩn (hidden state) từ đặc trưng ảnh
        self.init_h = nn.Linear(feature_dim, hidden_dim * num_layers)
        self.init_c = nn.Linear(feature_dim, hidden_dim * num_layers)

        self.output = nn.Linear(hidden_dim, vocab_size)

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

    def forward(self, image_feats: torch.Tensor, caption_in: torch.Tensor):
        b = image_feats.size(0)

        # Lấy trung bình toàn cục (Global Average Pooling) để tạo 1 vector đại diện cho cả ảnh
        global_img_feat = image_feats.mean(dim=1)  # Shape: (B, feature_dim)

        # Khởi tạo h0, c0 cho LSTM
        h0 = self.init_h(global_img_feat).view(b, self.num_layers, self.hidden_dim).permute(1, 0, 2).contiguous()
        c0 = self.init_c(global_img_feat).view(b, self.num_layers, self.hidden_dim).permute(1, 0, 2).contiguous()

        # Nhúng (Embed) caption
        emb = self.embedding(caption_in)  # Shape: (B, seq_len, embed_dim)

        # Lặp lại vector đặc trưng ảnh để ghép với mỗi token của caption
        seq_len = emb.size(1)
        global_img_feat_expanded = global_img_feat.unsqueeze(1).expand(-1, seq_len, -1)

        # Nối text embedding và image feature
        lstm_input = torch.cat([emb, global_img_feat_expanded], dim=-1)  # Shape: (B, seq_len, embed_dim + feature_dim)

        # Chạy qua LSTM
        lstm_out, _ = self.lstm(lstm_input, (h0, c0))

        # Dự đoán từ tiếp theo
        logits = self.output(lstm_out)
        return logits


# ==========================================
# 5. Pipeline hoàn chỉnh: ImageCaptioningModel
# ==========================================
class HybridImageCaptioningModel(nn.Module):
    def __init__(self, vocab_size: int, pad_idx: int, embed_dim: int = 256, hidden_dim: int = 512,
                 feature_dim: int = 512):
        super().__init__()
        # 1. CNN
        self.encoder = CNNEncoder(out_dim=feature_dim)
        # 2. Attention
        self.attention = SpatialAttention(feature_dim=feature_dim, num_heads=8)
        # 3. Transformer
        self.transformer = TransformerBlock(feature_dim=feature_dim, num_layers=2, num_heads=8)
        # 4. LSTM
        self.decoder = LSTMDecoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            feature_dim=feature_dim,
            pad_idx=pad_idx
        )

    def forward(self, images: torch.Tensor, caption_in: torch.Tensor):
        # Bước 1: Trích xuất đặc trưng không gian bằng CNN
        cnn_feats = self.encoder(images)

        # Bước 2: Tinh chỉnh đặc trưng bằng Self-Attention
        attn_feats = self.attention(cnn_feats)

        # Bước 3: Mô hình hóa ngữ cảnh toàn cục bằng Transformer
        transformer_feats = self.transformer(attn_feats)

        # Bước 4: Sinh text bằng LSTM dựa trên đầu ra của Transformer
        logits = self.decoder(transformer_feats, caption_in)

        return logits