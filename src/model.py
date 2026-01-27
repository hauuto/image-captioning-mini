import torch
import torch.nn as nn
import torchvision.models as models

# --- 1. ENCODER (CNN) ---
class EncoderCNN(nn.Module):
    def __init__(self, embed_size, train_cnn=False, dropout=0.5):
        super(EncoderCNN, self).__init__()

        # Tải ResNet-18 đã train trên ImageNet
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Freeze (đóng băng) các tham số của CNN để không train lại (tiết kiệm thời gian)
        for param in resnet.parameters():
            param.requires_grad = train_cnn

        # Thay thế lớp FC cuối cùng của ResNet
        # ResNet18 trả về vector 512 chiều -> ta chuyển về embed_size (256)
        self.resnet = nn.Sequential(*list(resnet.children())[:-1]) # Bỏ lớp FC cũ
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        self.dropout = nn.Dropout(dropout)

    def forward(self, images):
        # images: [Batch, 3, 224, 224]
        features = self.resnet(images)                # [Batch, 512, 1, 1]
        features = features.reshape(features.size(0), -1) # Flatten -> [Batch, 512]
        features = self.linear(features)              # [Batch, Embed_Size]
        features = self.bn(features)
        features = self.dropout(features)
        return features

# --- 2. DECODER (RNN/LSTM) ---
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, dropout=0.5):
        super(DecoderRNN, self).__init__()

        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        # LAYER 1: Word Embedding (Chuyển index thành vector)
        self.embedding = nn.Embedding(vocab_size, embed_size)

        # LAYER 2: LSTM
        self.lstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # LAYER 3: Output Linear (Chuyển hidden state về xác suất từ vựng)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

        # LAYER 4: Init Hidden States (Ánh xạ đặc trưng ảnh sang bộ nhớ LSTM)
        self.init_h = nn.Linear(embed_size, hidden_size)
        self.init_c = nn.Linear(embed_size, hidden_size)

    def init_hidden_state(self, features):
        """
        Khởi tạo trạng thái ẩn (h) và trạng thái tế bào (c) từ đặc trưng ảnh
        """
        h0 = self.init_h(features).unsqueeze(0).repeat(self.num_layers, 1, 1)
        c0 = self.init_c(features).unsqueeze(0).repeat(self.num_layers, 1, 1)
        return h0, c0

    def forward(self, features, captions):
        """
        Forward Pass (Dùng trong Training)
        features: Đặc trưng ảnh từ Encoder [Batch, Embed]
        captions: Ground Truth captions [Batch, Max_Len]
        """

        # 1. Khởi tạo trí nhớ LSTM từ ảnh
        h, c = self.init_hidden_state(features)

        # 2. Bỏ token <EOS> ở cuối caption (vì ta dự đoán từ tiếp theo)
        # Input: <SOS> A cat -> Output: A cat <EOS>
        embeddings = self.embedding(captions[:, :-1])
        embeddings = self.dropout(embeddings)

        # 3. Chạy qua LSTM
        # Lưu ý: Ta KHÔNG nối features vào input nữa, vì features đã dùng để init hidden rồi
        lstm_out, _ = self.lstm(embeddings, (h, c))

        # 4. Tính xác suất từ tiếp theo
        outputs = self.linear(lstm_out)
        return outputs

    def generate(self, features, max_len=20, vocab=None):
        """
        Inference Pass (Dùng để sinh caption cho ảnh mới)
        """
        batch_size = features.size(0)

        # Khởi tạo trí nhớ từ ảnh
        h, c = self.init_hidden_state(features)

        # Bắt đầu bằng token <SOS>
        start_token = vocab.stoi["<SOS>"]
        inputs = self.embedding(torch.tensor([start_token] * batch_size).to(features.device))
        inputs = inputs.unsqueeze(1) # [Batch, 1, Embed]

        captions = []

        for _ in range(max_len):
            # Chạy LSTM 1 bước
            lstm_out, (h, c) = self.lstm(inputs, (h, c))
            outputs = self.linear(lstm_out.squeeze(1))

            # Chọn từ có xác suất cao nhất (Greedy Search)
            predicted = outputs.argmax(1)

            # Lưu lại từ vừa dự đoán
            captions.append(predicted.item())

            # Nếu gặp <EOS> thì dừng
            if vocab.itos[predicted.item()] == "<EOS>":
                break

            # Dùng từ vừa dự đoán làm input cho bước tiếp theo
            inputs = self.embedding(predicted).unsqueeze(1)

        return captions

# --- 3. FULL MODEL (CNN + RNN) ---
class CNNtoRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, train_cnn=False, dropout=0.5):
        super(CNNtoRNN, self).__init__()
        self.encoder = EncoderCNN(embed_size, train_cnn, dropout)
        self.decoder = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers, dropout)

    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs

    def generate_caption(self, image, vocab, max_len=20):
        # Hàm wrapper tiện lợi để gọi từ bên ngoài
        self.eval()
        with torch.no_grad():
            # Thêm chiều batch = 1 nếu chưa có
            if image.dim() == 3:
                image = image.unsqueeze(0)

            features = self.encoder(image)
            generated_ids = self.decoder.generate(features, max_len, vocab)

            # Chuyển list số thành câu chữ
            return vocab.denumericalize(generated_ids)