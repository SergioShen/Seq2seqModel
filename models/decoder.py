#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time: 18:54 2020/8/10
# @Author: Sijie Shen
# @File: decoder
# @Project: Seq2seqModel


import torch
import torch.nn as nn

from .attention import Attention


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, n_layers=1, input_dropout_p=0, dropout_p=0, bidirectional_encoder=False,
                 use_attention=True, rnn_cell='lstm'):
        super().__init__()

        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.input_dropout_p = input_dropout_p
        self.dropout_p = dropout_p
        self.bidirectional_encoder = bidirectional_encoder
        self.use_attention = use_attention

        if rnn_cell.lower() == 'lstm':
            self.rnn_cls = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cls = nn.GRU
        elif rnn_cell.lower() == 'rnn':
            self.rnn_cls = nn.RNN
        else:
            raise ValueError("Unsupported RNN Cell: {0}".format(rnn_cell))

        self.input_dropout = nn.Dropout(p=input_dropout_p)
        self.rnn = self.rnn_cls(hidden_size, hidden_size, n_layers, dropout=dropout_p)
        if self.use_attention:
            self.attention = Attention(hidden_size)

    def forward(self, init_hidden, inputs, encoder_outputs=None, encoder_input_lengths=None):
        inputs = self.input_dropout(inputs)
        outputs, hidden = self.rnn(inputs, init_hidden)

        if self.use_attention:
            batch_size = encoder_outputs.size(1)
            input_size = encoder_outputs.size(0)
            output_size = outputs.size(0)
            attn_mask = self._generate_attention_mask(encoder_input_lengths, input_size, output_size, batch_size)
            outputs, _ = self.attention(outputs, encoder_outputs, attn_mask)

        return outputs

    def _generate_attention_mask(self, input_lengths, input_size, output_size, batch_size):
        device = input_lengths.device
        range_tensor = torch.arange(0, input_size, device=device).expand(batch_size, input_size)
        expanded_lengths = input_lengths.unsqueeze(1).expand(batch_size, input_size)
        mask = range_tensor >= expanded_lengths
        mask = mask.unsqueeze(1).expand(batch_size, output_size, input_size)

        return mask
