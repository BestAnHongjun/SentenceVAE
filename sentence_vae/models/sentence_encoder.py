# Copyright (c) 2024, School of Artificial Intelligence, OPtics and 
# ElectroNics(iOPEN), Northwestern PolyTechnical University, and Institute of 
# Artificial Intelligence (TeleAI), China Telecom.
#
# Author:   Coder.AN (an.hongjun@foxmail.com)
#           Huasen Chen (chenyifan1@mail.nwpu.edu.cn)
# 
# 
# This software is licensed under the MIT License.
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import torch
import torch.nn as nn
from typing import Union

from sentence_vae.models import PositionalEncodding
from sentence_vae.utils.llm import get_model
from sentence_vae.utils.weights import init_model_weights, load_embedding_state_dict


class SentenceEncoder(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        device: torch.device = None,
        dtype: torch.dtype = None,
        learnable_add: bool = True,
        load_ref_model: Union[bool, nn.Module] = False,
        ref_model_dir: str = None,
        ref_model_dtype: torch.dtype = None,
        finetune_embedding: bool = True,
        word_embed_proj_dim: int = None,
        num_attention_heads: int = 16, 
        num_hidden_layers: int = 1,
        max_seq_len: int = 1024,
        dropout: float = 0.1,
        pad_id=1,
    ):
        super(SentenceEncoder, self).__init__()
        word_embed_proj_dim = word_embed_proj_dim if word_embed_proj_dim is not None else hidden_size

        self.device = device if device is not None else torch.device('cuda')
        self.dtype = dtype if dtype is not None else torch.float16
        self.learnable_add = learnable_add

        self.embed_token = nn.Embedding(vocab_size, word_embed_proj_dim, padding_idx=pad_id)
        self.embed_positions = PositionalEncodding(hidden_size, max_seq_len, dtype=self.dtype, device=self.device)

        if word_embed_proj_dim != hidden_size:
            self.project_in = nn.Linear(word_embed_proj_dim, hidden_size, bias=False)
        else:
            self.project_in = None 

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_attention_heads,
            dim_feedforward=hidden_size * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_hidden_layers)
        if self.learnable_add:
            self.la = nn.Linear(hidden_size, 1)
            # self.conv1d = nn.Conv1d(hidden_size, 1, kernel_size=3, stride=1, padding=1, bias=True)
        self.lnorm = nn.LayerNorm(hidden_size, device=self.device, dtype=self.dtype)

        if isinstance(load_ref_model, nn.Module):
            ref_model_dtype = ref_model_dtype if ref_model_dtype is not None else self.dtype
            ref_model = load_ref_model
            ref_model_state_dict = load_embedding_state_dict(ref_model)
            assert ref_model_state_dict is not None, f"Model {ref_model_dir} does not have an Embedding layer."
            self.embed_token.load_state_dict(ref_model_state_dict)
        elif load_ref_model and ref_model_dir is not None:
            ref_model_dtype = ref_model_dtype if ref_model_dtype is not None else self.dtype
            ref_model = get_model(ref_model_dir, ref_model_dtype, 'cpu')
            ref_model_state_dict = load_embedding_state_dict(ref_model)
            assert ref_model_state_dict is not None, f"Model {ref_model_dir} does not have an Embedding layer."
            self.embed_token.load_state_dict(ref_model_state_dict)
        if not finetune_embedding:
            self.embed_token.weight.requires_grad = False

        self.to(self.dtype)
        self.to(self.device)
    

    def forward(self, input_ids, attention_mask=None):
        _, seq_len = input_ids.shape 
        if attention_mask is not None and attention_mask.dtype is not torch.bool:
            attention_mask = ~attention_mask.to(torch.bool)

        inputs_emb = self.embed_token(input_ids)
        pos_emb = self.embed_positions(seq_len)
        if self.project_in is not None:
            inputs_emb = self.project_in(inputs_emb)
        inputs_emb = inputs_emb + pos_emb 

        hidden_state = self.encoder(inputs_emb, src_key_padding_mask=attention_mask)
        hidden_state[attention_mask] = 0    # (batch, seq_len, hidden)

        if self.learnable_add:
            alpha = self.la(hidden_state)
            # print(alpha.shape)
            # hidden_trans = hidden_state.transpose(2, 1).contiguous() # (batch, hidden, seq_len)
            # alpha = self.conv1d(hidden_trans)   # (batch, 1, seq_len)
            # alpha = alpha.transpose(2, 1).contiguous()   # (batch, seqlen, hidden)
            sentence_emb = torch.sum(hidden_state * alpha, dim=-2, keepdim=True)
        else:
            sentence_emb = torch.sum(hidden_state, dim=-2, keepdim=True)
        sentence_emb_norm = self.lnorm(sentence_emb)

        return sentence_emb_norm
