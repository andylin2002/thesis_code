import torch
import torch.nn as nn
import torch.nn.functional as F
from architecture.decoder import Decoder
from architecture.multihead_attention import MultiHeadAttention
from architecture.positional_encoding import PositionalEncoding
from architecture.pointerwise_feedforward import PointerwiseFeedforward
from architecture.encoder_decoder import EncoderDecoder
from architecture.encoder import Encoder
from architecture.encoder_layer import EncoderLayer
from architecture.decoder_layer import DecoderLayer
from architecture.batch import subsequent_mask
from architecture.embeddings import Embeddings
from architecture.generator import Generator
import numpy as np
import scipy.io
import os

import copy
import math

class QuantizedTF(nn.Module):
    def __init__(self, enc_inp_size, dec_inp_size, dec_out_size, layer=6,
                   d_model=512, d_ff=2048, h=8, dropout=0.1):
        super(QuantizedTF, self).__init__()
        "Helper: Construct a model from hyperparameters."
        c = copy.deepcopy
        attn = MultiHeadAttention(h, d_model)
        ff = PointerwiseFeedforward(d_model, d_ff, dropout)
        position = PositionalEncoding(d_model, dropout)
        self.model = EncoderDecoder(
            Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), layer),
            Decoder(DecoderLayer(d_model, c(attn), c(attn),
                                 c(ff), dropout), layer),
            nn.Sequential(Embeddings(d_model,enc_inp_size), c(position)),
            nn.Sequential(Embeddings(d_model,dec_inp_size), c(position)),
            Generator(d_model, dec_out_size))

        # This was important from their code.
        # Initialize parameters with Glorot / fan_avg.
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, *input):
        return self.model.generator(self.model(*input))

    def predict(self,*input):

        return F.softmax(self.model.generator(self.model(*input)), dim=-1)
