#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time: 20:38 2020/8/10
# @Author: Sijie Shen
# @File: seq2seq
# @Project: Seq2seqModel

import torch.nn as nn

from .encoder import EncoderRNN
from .decoder import DecoderRNN


class Seq2seqModel(nn.Module):
    def __init__(self, model_params):
        super().__init__()
        self.src_vocab_size = model_params['src_vocab_size']
        self.tgt_vocab_size = model_params['tgt_vocab_size']
        self.rnn_cell = model_params['rnn_cell']
        self.hidden_size = model_params['hidden_size']
        self.num_encoder_layers = model_params['num_encoder_layers']
        self.num_decoder_layers = model_params['num_decoder_layers']
        self.input_drouput = model_params['input_drouput']
        self.rnn_drouput = model_params['rnn_drouput']
        self.share_vocab = model_params['share_vocab']
        self.weight_tying = model_params['weight_tying']
        self.bidirectional_encoder = model_params['bidirectional_encoder']
        self.use_attention = model_params['use_attention']

        # Build encoder and decoder
        self.encoder = EncoderRNN(self.hidden_size, self.num_encoder_layers, self.input_drouput, self.rnn_drouput,
                                  self.bidirectional_encoder, self.rnn_cell)
        self.decoder = DecoderRNN(self.hidden_size, self.num_decoder_layers, self.input_drouput, self.rnn_drouput,
                                  self.bidirectional_encoder, self.use_attention, self.rnn_cell)

        # Build embedding and out layer
        self.src_embedding = nn.Embedding(self.src_vocab_size, self.hidden_size)
        if self.share_vocab:
            assert self.src_vocab_size == self.tgt_vocab_size
            self.tgt_embedding = self.src_embedding
        else:
            self.tgt_embedding = nn.Embedding(self.tgt_vocab_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.tgt_vocab_size)
        if self.weight_tying:
            self.out.weight = self.tgt_embedding.weight

        self._reset_parameters()

    def _reset_parameters(self):
        init_range = 0.1
        self.src_embedding.weight.data.uniform_(-init_range, init_range)
        if not self.share_vocab:
            self.tgt_embedding.weight.data.uniform_(-init_range, init_range)
        self.out.bias.data.zero_()
        if not self.weight_tying:
            self.out.weight.data.uniform_(-init_range, init_range)

    def forward(self, encoder_inputs, decoder_inputs, encoder_input_lengths=None):
        encoder_embedded = self.src_embedding(encoder_inputs)
        decoder_embedded = self.tgt_embedding(decoder_inputs)

        encoder_outputs, encoder_hidden = self.encoder(encoder_embedded, encoder_input_lengths)
        decoder_outputs = self.decoder(encoder_hidden, decoder_embedded, encoder_outputs, encoder_input_lengths)
        logits = self.out(decoder_outputs)

        return logits
