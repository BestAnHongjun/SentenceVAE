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
import torch.nn.functional as F

from mmengine.model import BaseModel

from .focal_loss import FocalLoss
from .sentence_vae_model import SentenceVAE
from sentence_vae.utils.llm import get_model


class SentenceLLM(BaseModel):
    def __init__(
        self,
        svae_hidden_size: int,
        svae_vocab_size: int,
        svae_learnable_add: bool = True,
        svae_load_ref_model: bool = False,
        svae_ref_model_dir: str = None,
        svae_ref_model_dtype: torch.dtype = None,
        svae_finetune_embedding: bool = True,
        svae_word_embed_proj_dim: int = None,
        svae_num_attention_heads: int = 16,
        svae_num_hidden_layers: int = 1,
        svae_model_path: str = None,
        llm_ref_model_dir: str = None,
        llm_ref_model_dtype: torch.dtype = None,
        llm_finetune_layers: int = -1,
        finetune_svae: bool = True,
        max_sentence_len: int = 512,
        max_sentence_num: int = 512,
        dropout: float = 0.1,
        bos_id=2,
        pad_id=1,
        end_id=2,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ):
        super().__init__()
        self.bos_token_id = bos_id
        self.pad_token_id = pad_id
        self.eos_token_id = end_id
        self.hidden_size = svae_hidden_size
        self.llm_finetune_layers = llm_finetune_layers
        self.max_sentence_len = max_sentence_len
        self.max_sentence_num = max_sentence_num

        self.device = device if device is not None else torch.device("cuda")
        self.dtype = dtype if dtype is not None else torch.float16

        self.svae = SentenceVAE(
            hidden_size=svae_hidden_size, vocab_size=svae_vocab_size, device=self.device, dtype=self.dtype,
            learnable_add=svae_learnable_add, load_ref_model=svae_load_ref_model, ref_model_dir=svae_ref_model_dir, ref_model_dtype=svae_ref_model_dtype,
            finetune_embedding=svae_finetune_embedding, word_embed_proj_dim=svae_word_embed_proj_dim, 
            num_attention_heads=svae_num_attention_heads, num_hidden_layers=svae_num_hidden_layers, max_seq_len=max_sentence_len, dropout=dropout, 
            bos_id=bos_id, pad_id=pad_id, end_id=end_id
        )

        llm_ref_model_dtype = llm_ref_model_dtype if llm_ref_model_dtype else self.dtype
        llm = get_model(llm_ref_model_dir, llm_ref_model_dtype, self.device, True)
        self.llm_pe = llm.model.decoder.embed_positions
        self.llm_layers = llm.model.decoder.layers

        self.fc = nn.Linear(svae_hidden_size, 2)
        self.focal_loss = FocalLoss()

        if svae_model_path is not None:
            print(f"Loading {svae_model_path}")
            ckpt = torch.load(svae_model_path)
            self.svae.load_state_dict(ckpt['state_dict'])

        self.to(self.dtype)
        self.to(self.device)

        if not finetune_svae:
            self._freeze_model(self.svae)
        if llm_finetune_layers >= 0 and llm_finetune_layers * 2 < len(self.llm_layers):
            for i in range(llm_finetune_layers, len(self.llm_layers) - llm_finetune_layers):
                self._freeze_model(self.llm_layers[i])
    

    def _freeze_model(self, model: nn.Module):
        for param in model.parameters():
            param.requires_grad = False


    def forward(self, sentence_mask, sentence_toks, tok_mask, mode='loss'):
        sentence_mask = sentence_mask.to(self.device)
        sentence_toks = sentence_toks.to(self.device)
        tok_mask = tok_mask.to(self.device)
        batch_size, sen_num, seq_len = sentence_toks.shape

        sentence_embedding = torch.zeros(
            (batch_size, sen_num, self.hidden_size),
            dtype=self.dtype, device=self.device
        )

        # encoder
        for b in range(batch_size):
            sentence_embedding[b] = self.svae.encoder(sentence_toks[b], tok_mask[b]).view(sen_num, self.hidden_size)
            sentence_embedding[b][~sentence_mask[b].bool()] = 0

        # llm
        pos_emb = self.llm_pe(sentence_mask, 0)
        hidden_state = sentence_embedding + pos_emb
        for layer in self.llm_layers:
            hidden_state = layer(hidden_state)[0]
        stop_flag = self.fc(hidden_state)

        sentence_pad = torch.zeros((batch_size, 1), dtype=sentence_mask.dtype, device=sentence_mask.device)
        tgt_stop_flag = torch.concat((sentence_mask, sentence_pad), dim=1)[:, 1:]
        sen_lens = torch.sum(sentence_mask, dim=1)
        stop_loss = 0
        for b in range(batch_size):
            sen_len = sen_lens[b]
            stop_loss += self.focal_loss(stop_flag[b, :sen_len], tgt_stop_flag[b, :sen_len])
        stop_loss /= batch_size

        # decoder
        hidden_state = hidden_state.view(batch_size, sen_num, 1, self.hidden_size)
        decode_loss = 0
        ppl_loss = 0
        for b in range(batch_size):
            sen_len = sen_lens[b]
            input_ids = sentence_toks[b, 1:sen_len]
            attention_mask = tok_mask[b, :sen_len-1]
            sentence_embd = hidden_state[b, :sen_len-1]
            output = self.svae.decoder(input_ids, sentence_embd, attention_mask)
            pad_ids = torch.zeros((sen_len-1, 1), device=self.device, dtype=input_ids.dtype).fill_(self.pad_token_id)
            tgt_ids = torch.cat((input_ids, pad_ids), dim=1)
            seq_lens = torch.sum(attention_mask, dim=1, keepdim=True)
            tgt_ids.scatter_(1, seq_lens, self.eos_token_id)
            attention_mask = attention_mask.bool()
            decode_loss += self.focal_loss(output[attention_mask], tgt_ids[:, 1:][attention_mask])
            ppl_loss += F.cross_entropy(output[attention_mask], tgt_ids[:, 1:][attention_mask])
            del input_ids
            del attention_mask
            del sentence_embd
            del output 
            del pad_ids 
            del tgt_ids
        decode_loss /= batch_size
        ppl_loss /= batch_size

        if mode == 'loss':
            if stop_loss < 1e-2:
                return {"decode_loss": decode_loss}
            return {"stop_loss": stop_loss, "decode_loss": decode_loss}
        elif mode == 'predict':
            return stop_loss, ppl_loss
