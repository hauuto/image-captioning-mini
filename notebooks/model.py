import torch
import torch.nn as nn
import torchvision.models as models

class EncoderCNN(nn.Module):
    def __init__(self, embed_size, train_cnn=False, dropout=0.5):
        super(EncoderCNN, self).__init__()
        # Sử dụng ResNet18 Pretrained
        resnet = models.resnet18(pretrained=True)

        # Freeze / Unfreeze layers
        for param in resnet.parameters():
            param.requires_grad = train_cnn

        # Thay thế lớp FC cuối cùng
        self.resnet = nn.Sequential(*list(resnet.children())[:-1]) # Bỏ FC cũ
        self.fc = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        self.dropout = nn.Dropout(dropout)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.fc(features)
        features = self.bn(features)
        features = self.dropout(features)
        return features

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, dropout=0.5):
        super(DecoderRNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

        # --- QUAN TRỌNG: INIT HIDDEN STATES TỪ ẢNH (Từ code mới của bạn) ---
        self.init_h = nn.Linear(embed_size, hidden_size)
        self.init_c = nn.Linear(embed_size, hidden_size)

    def init_hidden_state(self, features):
        """Khởi tạo trạng thái ban đầu của LSTM từ đặc trưng ảnh"""
        h0 = self.init_h(features).unsqueeze(0).repeat(self.num_layers, 1, 1)
        c0 = self.init_c(features).unsqueeze(0).repeat(self.num_layers, 1, 1)
        return h0, c0

    def forward(self, features, captions):
        # features: (Batch, Embed)
        # captions: (Batch, Seq_Len)

        # 1. Khởi tạo hidden state từ ảnh
        h, c = self.init_hidden_state(features)

        # 2. Embedding Caption (bỏ token <EOS> cuối để teacher forcing)
        embeddings = self.embed(captions[:, :-1])
        embeddings = self.dropout(embeddings)

        # 3. LSTM Forward (chỉ đưa embeddings vào, hidden state đã có thông tin ảnh)
        # Lưu ý: Code cũ của mình nối ảnh vào đầu chuỗi. 
        # Code mới của bạn dùng ảnh để init hidden. Cách của bạn (Show and Tell) xịn hơn cho Encoder-Decoder.
        lstm_out, _ = self.lstm(embeddings, (h, c))

        outputs = self.linear(lstm_out)
        return outputs

    def generate(self, features, max_len=20, vocab=None):
        # Inference function
        batch_size = features.size(0)
        h, c = self.init_hidden_state(features)

        # Bắt đầu bằng <SOS>
        inputs = self.embed(torch.tensor([vocab.stoi["<SOS>"]]).to(features.device)).unsqueeze(0)

        captions = []
        for _ in range(max_len):
            lstm_out, (h, c) = self.lstm(inputs, (h, c))
            outputs = self.linear(lstm_out.squeeze(1))
            predicted = outputs.argmax(1)

            if vocab.itos[predicted.item()] == "<EOS>":
                break

            captions.append(predicted.item())
            # Input tiếp theo là từ vừa dự đoán
            inputs = self.embed(predicted).unsqueeze(1)

        return [captions] # Trả về list of lists để khớp format

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
        self.eval()
        with torch.no_grad():
            features = self.encoder(image).unsqueeze(0) # (1, Embed)
            generated_ids = self.decoder.generate(features, max_len, vocab)
            return vocab.denumericalize(generated_ids[0])
