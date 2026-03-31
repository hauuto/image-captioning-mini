import torch
import torch.nn as nn
import torchvision.models as models

class CNNEncoder(nn.Module):
    def __init__(self, out_dim: int = 512):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-2])
        self.proj = nn.Conv2d(512, out_dim, kernel_size=1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.feature_extractor(images)
        feats = self.proj(feats)
        b, c, h, w = feats.shape
        feats = feats.view(b, c, h * w).permute(0, 2, 1)
        return feats

class AdditiveAttention(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, attn_dim: int):
        super().__init__()
        self.feature_proj = nn.Linear(feature_dim, attn_dim)
        self.hidden_proj = nn.Linear(hidden_dim, attn_dim)
        self.score = nn.Linear(attn_dim, 1)

    def forward(self, image_feats: torch.Tensor, hidden_seq: torch.Tensor):
        feat_proj = self.feature_proj(image_feats).unsqueeze(1)
        hid_proj = self.hidden_proj(hidden_seq).unsqueeze(2)
        energy = torch.tanh(feat_proj + hid_proj)
        attn_scores = self.score(energy).squeeze(-1)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        context = torch.matmul(attn_weights, image_feats)
        return context, attn_weights

class CaptionDecoderGRU(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, feature_dim: int, attn_dim: int, pad_idx: int, num_layers: int = 1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.0 if num_layers == 1 else 0.2,
        )
        self.attention = AdditiveAttention(feature_dim=feature_dim, hidden_dim=hidden_dim, attn_dim=attn_dim)
        
        self.output = nn.Linear(hidden_dim + feature_dim, vocab_size)

    def forward(self, image_feats: torch.Tensor, caption_in: torch.Tensor):
        emb = self.embedding(caption_in)
        hidden_seq, _ = self.gru(emb)
        context, attn_weights = self.attention(image_feats, hidden_seq)
        logits = self.output(torch.cat([hidden_seq, context], dim=-1))
        return logits, attn_weights

class ImageCaptioningModel(nn.Module):
    def __init__(self, vocab_size: int, pad_idx: int, embed_dim: int = 256, hidden_dim: int = 256, feature_dim: int = 512, attn_dim: int = 256):
        super().__init__()
        self.encoder = CNNEncoder(out_dim=feature_dim)
        self.decoder = CaptionDecoderGRU(
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
