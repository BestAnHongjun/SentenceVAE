# Copyright (c) 2024, School of Artificial Intelligence, OPtics and ElectroNics(iOPEN), 
# Northwestern PolyTechnical University, 
# and Institute of Artificial Intelligence (TeleAI), China Telecom.
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

from mmengine.model import BaseModel

from .focal_loss import FocalLoss
from .sentence_encoder import SentenceEncoder
from .sentence_decoder import SentenceDecoder
from sentence_vae.utils import get_model


class SentenceVAE(BaseModel):
    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        device: torch.dtype = None,
        dtype: torch.dtype = None,
        learnable_add: bool = True,
        load_ref_model: bool = False,
        ref_model_dir: str = None,
        ref_model_dtype: torch.dtype = None,
        finetune_embedding: bool = True,
        word_embed_proj_dim: int = None,
        num_attention_heads: int = 16, 
        num_hidden_layers: int = 1,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        bos_id=2,
        pad_id=1,
        end_id=2,
    ):
        super().__init__()
        self.bos_token_id = bos_id
        self.pad_token_id = pad_id
        self.eos_token_id = end_id
        self.max_seq_len = max_seq_len

        self.device = device if device is not None else torch.device("cuda")
        self.dtype = dtype if dtype is not None else torch.float16

        if load_ref_model and ref_model_dir is not None:
            ref_model_dtype = ref_model_dtype if ref_model_dtype is not None else self.dtype
            load_ref_model = get_model(ref_model_dir, ref_model_dtype, 'cpu')
        else:
            load_ref_model = False

        self.encoder = SentenceEncoder(
            hidden_size, vocab_size, device, dtype,
            learnable_add, load_ref_model, ref_model_dir, ref_model_dtype,
            finetune_embedding, word_embed_proj_dim, 
            num_attention_heads, num_hidden_layers, max_seq_len,
            dropout, pad_id
        )
        self.decoder = SentenceDecoder(
            hidden_size, vocab_size, device, dtype,
            load_ref_model, ref_model_dir, ref_model_dtype,
            finetune_embedding, word_embed_proj_dim, 
            num_attention_heads, num_hidden_layers, max_seq_len,
            dropout, pad_id
        )

        self.focal_loss = FocalLoss()
    
    def forward(self, input_ids, attention_mask=None, mode='loss'):
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        else:
            attention_mask = torch.ones(input_ids.shape, dtype=torch.int64, device=self.device)

        sentence_embd = self.encoder(input_ids, attention_mask)

        output = self.decoder(input_ids, sentence_embd, attention_mask)
        batch, _ = input_ids.shape
        pad_ids = torch.zeros((batch, 1), device=self.device, dtype=input_ids.dtype).fill_(self.pad_token_id)
        tgt_ids = torch.cat((input_ids, pad_ids), dim=1)
        seq_lens = torch.sum(attention_mask, dim=1, keepdim=True)
        tgt_ids.scatter_(1, seq_lens, self.eos_token_id)
        attention_mask = attention_mask.bool()
        if mode == 'loss':
            loss = self.focal_loss(output[attention_mask], tgt_ids[:, 1:][attention_mask])
            return {'total_loss': loss}
        elif mode == 'predict':
            return output, attention_mask, tgt_ids
        else:
            return output
    
    def streaming_generate(self, input_ids, attention_mask=None, max_output_len=64):
        batch_size = input_ids.size(0)
        assert batch_size == 1

        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        else:
            attention_mask = torch.ones(input_ids.shape, dtype=torch.int64, device=self.device)

        sentence_embd = self.encoder(input_ids, attention_mask)

        for output_id in self.decoder.streaming_generate(
            sentence_embd, 
            max_output_len, 
            self.bos_token_id,
            self.eos_token_id
        ):
            yield output_id

        # output_ids = torch.tensor([[self.bos_token_id]], dtype=torch.long, device=self.device)
        # while len(output_ids) < max_output_len:
        #     logits = self.decoder(output_ids, sentence_embd)
        #     new_id = torch.argmax(logits[:, -1:], dim=-1)
        #     output_ids = torch.concat((output_ids, new_id), dim=1)
        #     yield new_id
        #     if new_id.item() == self.eos_token_id:
        #         break 

    def generate(self, input_ids, attention_mask=None, max_output_len=64):
        output_ids = []
        for output_id in self.streaming_generate(input_ids, attention_mask, max_output_len):
            output_ids.append(output_id.item())
        return output_ids