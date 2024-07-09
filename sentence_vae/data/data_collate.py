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

import re
import torch 


class SentenceCollate:
    def __init__(
        self, 
        tokenizer, 
        max_len=1024, 
        padding=True, 
        fix_len=True
    ):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.padding = padding
        self.fix_len = fix_len
    
    def __call__(self, texts):
        encoded = self.tokenizer.batch_encode_plus(
            texts, 
            padding=self.padding, 
            truncation=True, 
            max_length=self.max_len, 
            return_tensors="pt"
        )

        input_ids = encoded['input_ids']
        attention_mask = encoded['attention_mask']

        if self.fix_len:
            batch, seq_len = input_ids.shape
            if seq_len < self.max_len: 
                pad_ids = torch.zeros(
                    (batch, self.max_len - seq_len), 
                    dtype=input_ids.dtype, 
                    device=input_ids.device
                ).fill_(self.tokenizer.pad_token_id)
                pad_mask = torch.zeros(
                    (batch, self.max_len - seq_len),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device
                )
                
                input_ids = torch.concat((input_ids, pad_ids), dim=1)
                attention_mask = torch.concat((attention_mask, pad_mask), dim=1)

        return input_ids, attention_mask


class PassageCollate:
    def __init__(
        self, 
        tokenizer, 
        max_sentence_len=512, 
        max_sentence_num=512,
        padding=True, 
        fix_len=True
    ):
        self.tokenizer = tokenizer
        self.max_sentence_len = max_sentence_len
        self.max_sentence_num = max_sentence_num
        self.padding = padding
        self.fix_len = fix_len

        self._re_sentence = re.compile(
            '([，。！？；：\?])([^”’])|' + '([,.!?;:\?])([^"\'])|' +
            '(\…{2})([^”’])|' + '(\.{6})([^”’])|' +
            '([。！？\?][”’])([^，。！？\?])|' + '([.!?\?]["\'])([^，。！？\?])'
        )


    def cut_sentence_func(self, para):
        p, last_p, length = 0, 0, len(para)
        sentences = []
        sentence_mask = []

        while p < length and len(sentences) < self.max_sentence_num:
            if p + 1 < length and (
                para[p:p+2] == "……" or 
                para[p:p+2] == "，”" or 
                para[p:p+2] == "”，" or
                para[p:p+2] == "。”" or 
                para[p:p+2] == "”。" or
                para[p:p+2] == "!”" or 
                para[p:p+2] == "”！" or 
                para[p:p+2] == "？”" or 
                para[p:p+2] == "”？" or 
                para[p:p+2] == ',"' or 
                para[p:p+2] == '",' or 
                para[p:p+2] == '."' or 
                para[p:p+2] == '".' or 
                para[p:p+2] == '!"' or 
                para[p:p+2] == '"!' or 
                para[p:p+2] == '?"' or 
                para[p:p+2] == '"?'
            ):
                p += 2
                # cut
            elif para[p] in [
                '，', '。', '”', '！', '？', '；', '：',
                ',', '.', '!', '?', ';', ':', '"', '\n', '\r'
            ]:
                p += 1
                # cut
            else:
                p += 1
                continue

            while p < length and para[p] in [' ', '\n', '\r']: p += 1
            sentences.append(para[last_p:p])
            sentence_mask.append(1)
            last_p = p
        
        delta = self.max_sentence_num - len(sentences)
        if delta:
            sentences.extend(["" for _ in range(delta)])
            sentence_mask.extend([0 for _ in range(delta)])
        
        sentence_mask = torch.tensor(sentence_mask, dtype=torch.int64)

        return sentences, sentence_mask

    
    def __call__(self, texts):
        batch_size = len(texts)

        batch_sentence_mask = torch.zeros(
            (batch_size, self.max_sentence_num),
            dtype=torch.int64
        )
        batch_sentence_toks = torch.zeros(
            (batch_size, self.max_sentence_num, self.max_sentence_len),
            dtype=torch.int64
        )
        batch_tok_mask = torch.zeros(
            (batch_size, self.max_sentence_num, self.max_sentence_len),
            dtype=torch.int64
        )

        for i, text in enumerate(texts):
            sentences, sentence_mask = self.cut_sentence_func(text)
            encoded = self.tokenizer.batch_encode_plus(
                sentences,
                padding=self.padding,
                truncation=True,
                max_length=self.max_sentence_len,
                return_tensors="pt"
            )

            sentence_toks, tok_mask = encoded['input_ids'], encoded['attention_mask']
            sentence_num, seq_len = sentence_toks.shape
            if seq_len < self.max_sentence_len:
                pad_ids = torch.zeros(
                    (sentence_num, self.max_sentence_len - seq_len), 
                    dtype=sentence_toks.dtype, 
                    device=sentence_toks.device
                ).fill_(self.tokenizer.pad_token_id)
                pad_mask = torch.zeros(
                    (sentence_num, self.max_sentence_len - seq_len),
                    dtype=tok_mask.dtype,
                    device=tok_mask.device
                )
                sentence_toks = torch.concat((sentence_toks, pad_ids), dim=1)
                tok_mask = torch.concat((tok_mask, pad_mask), dim=1)
            
            batch_sentence_mask[i] = sentence_mask
            batch_sentence_toks[i] = sentence_toks
            batch_tok_mask[i] = tok_mask

        return batch_sentence_mask, batch_sentence_toks, batch_tok_mask
