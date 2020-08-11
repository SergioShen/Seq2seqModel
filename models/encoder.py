#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time: 18:46 2020/8/10
# @Author: Sijie Shen
# @File: encoder
# @Project: Seq2seqModel


import torch.nn as nn


class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, n_layers=1, input_dropout_p=0, dropout_p=0, bidirectional=False, rnn_cell='lstm'):
        super().__init__()

        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.input_dropout_p = input_dropout_p
        self.dropout_p = dropout_p

        if rnn_cell.lower() == 'lstm':
            self.rnn_cls = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cls = nn.GRU
        elif rnn_cell.lower() == 'rnn':
            self.rnn_cls = nn.RNN
        else:
            raise ValueError("Unsupported RNN Cell: {0}".format(rnn_cell))

        self.input_dropout = nn.Dropout(p=input_dropout_p)
        self.rnn = self.rnn_cls(hidden_size, hidden_size, n_layers, bidirectional=bidirectional, dropout=dropout_p)

    def forward(self, inputs, input_lengths=None):
        inputs = self.input_dropout(inputs)

        if input_lengths is not None:
            inputs = nn.utils.rnn.pack_padded_sequence(inputs, input_lengths, enforce_sorted=False)

        outputs, hidden = self.rnn(inputs)

        if input_lengths is not None:
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)

        return outputs, hidden
