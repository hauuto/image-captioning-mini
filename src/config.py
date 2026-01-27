# -*- coding: utf-8 -*-
import torch

class Config:
    DATA_DIR = '../data/flickr8k'
    IMG_DIR = f"{DATA_DIR}/images"
    CAPTION = f"{DATA_DIR}/captions.txt"


    EMBED_SIZE = 256
    HIDDEN_SIZE = 512
    NUM_LAYERS = 1
    DROPOUT = 0.5
    TRAIN_CNN = False


    LEARNING_RATE = 3e-4
    BATCH_SIZE = 32
    EPOCHS = 10
    NUM_WORKERS = 2


    FREQ_THRESHOLD = 2
    IMG_SIZE = 224

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")