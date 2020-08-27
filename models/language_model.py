#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time: 22:43 2020/8/20
# @Author: Sijie Shen
# @File: language_model
# @Project: Seq2seqModel

import torch.nn as nn

from .encoder import EncoderRNN


class LanguageModel(nn.Module):
    def __init__(self, model_params):
        super().__init__()
        self.vocab_size = model_params['vocab_size']
        self.rnn_cell = model_params['rnn_cell']
        self.hidden_size = model_params['hidden_size']
        self.num_layers = model_params['num_layers']
        self.input_drouput = model_params['input_drouput']
        self.rnn_drouput = model_params['rnn_drouput']
        self.weight_tying = model_params['weight_tying']

        # Build encoder and decoder
        self.encoder = EncoderRNN(self.hidden_size, self.num_layers, self.input_drouput, self.rnn_drouput,
                                  bidirectional=False, rnn_cell=self.rnn_cell)

        # Build embedding and out layer
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.vocab_size)
        if self.weight_tying:
            self.out.weight = self.embedding.weight

        self._reset_parameters()

    def _reset_parameters(self):
        init_range = 0.1
        self.embedding.weight.data.uniform_(-init_range, init_range)
        self.out.bias.data.zero_()
        if not self.weight_tying:
            self.out.weight.data.uniform_(-init_range, init_range)

    def forward(self, encoder_inputs, encoder_input_lengths=None):
        encoder_embedded = self.embedding(encoder_inputs)

        encoder_outputs, _ = self.encoder(encoder_embedded, encoder_input_lengths)
        logits = self.out(encoder_outputs)

        return logits
