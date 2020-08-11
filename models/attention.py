#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time: 21:47 2020/8/10
# @Author: Sijie Shen
# @File: attention
# @Project: Seq2seqModel

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, dim):
        super(Attention, self).__init__()
        self.linear_out = nn.Linear(dim * 2, dim)

    def forward(self, output, memory, attn_mask=None):
        """
        :param output: (T, B, H)
        :param memory: (S, B, H)
        :param attn_mask: (B, T, S)
        :return:
        """
        # (B, T, H) * (B, H, S) -> (B, T, S)
        attn = torch.bmm(output.transpose(0, 1), memory.permute(1, 2, 0))
        if attn_mask is not None:
            attn.masked_fill_(attn_mask, float('-inf'))
        attn = attn.softmax(dim=2)

        # (B, T, S) * (B, S, H) -> (B, T, H) -> (T, B, H)
        context = torch.bmm(attn, memory.transpose(0, 1)).transpose(0, 1)

        # concat -> (T, B, 2 * H)
        combined = torch.cat((context, output), dim=2)

        # output -> (T, B, H) -> (T, B, H)
        output = self.linear_out(combined)
        output = torch.tanh(output)

        return output, attn
