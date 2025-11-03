import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.io
import os
import copy
import math

# 假設這些架構組件位於 ./architecture/ 子模塊中
from .architecture.decoder import Decoder
from .architecture.multihead_attention import MultiHeadAttention
from .architecture.positional_encoding import PositionalEncoding
from .architecture.pointerwise_feedforward import PointerwiseFeedforward
from .architecture.encoder_decoder import EncoderDecoder
from .architecture.encoder import Encoder
from .architecture.encoder_layer import EncoderLayer
from .architecture.decoder_layer import DecoderLayer
from .architecture.embeddings import Embeddings
from .architecture.generator import Generator

class QuantizedTF(nn.Module):
    """
    Quantized Transformer Model (Encoder-Decoder) for Trajectory Prediction.
    
    Encoder Input: Continuous Coordinates (x, y) via nn.Linear.
    Decoder Input: Discrete Action IDs via nn.Embedding.
    Output: Logits for N_DIRECTIONS classes.
    """
    def __init__(self, enc_inp_size, dec_inp_size, dec_out_size, layer=6,
                   d_model=512, d_ff=2048, h=8, dropout=0.1):
        super(QuantizedTF, self).__init__()
        
        if enc_inp_size != 2:
             raise ValueError("Encoder Input Size must be 2 for (x, y) coordinates.")
             
        c = copy.deepcopy
        attn = MultiHeadAttention(h, d_model)
        ff = PointerwiseFeedforward(d_model, d_ff, dropout)
        position = PositionalEncoding(d_model, dropout)
        
    ##### --- Encoder Input ---
        # Encoder Input Layer
        src_embed_layer = nn.Sequential(
            nn.Linear(enc_inp_size, d_model), # nn.Linear(2, 512)
            nn.ReLU(),
            c(position)
        )
        
        # Decoder Input Layer
        tgt_embed_layer = nn.Sequential(
            Embeddings(d_model, dec_inp_size),
            c(position)
        )

        self.model = EncoderDecoder(
            Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), layer),
            Decoder(DecoderLayer(d_model, c(attn), c(attn),
                                 c(ff), dropout), layer),
            src_embed_layer,
            tgt_embed_layer,
            Generator(d_model, dec_out_size))

        # Initialize parameters with Glorot / fan_avg.
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, *input):
        return self.model.generator(self.model(*input))

    def predict(self, *input):
        return F.softmax(self.model.generator(self.model(*input)), dim=-1)