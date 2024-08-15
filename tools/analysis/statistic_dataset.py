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

import argparse
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from mmengine.dataset import DefaultSampler

from sentence_vae.utils import get_tokenizer
from sentence_vae.data import TeleDSDataset, SentenceCollate, PassageCollate


def make_parser():
    parser = argparse.ArgumentParser("SentenceVAE statistic parser.")
    parser.add_argument("--tokenizer_dir", type=str)
    parser.add_argument("--max_sen_len", type=int, default=64)
    parser.add_argument("--max_sen_num", type=int, default=64)
    parser.add_argument("--mode", choices=["sentence", "passage"], default='sentence')
    parser.add_argument("--card_size", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--teleds_ip", type=str, default="127.0.0.1")
    parser.add_argument("--teleds_port", type=int, default=8000)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--prefetch_factor", type=int, default=5)
    parser.add_argument("--max_iters", type=int, default=300000)
    return parser


def main(args):
    tokenizer = get_tokenizer(ckpt_path=args.tokenizer_dir, max_seq_len=args.max_sen_len)

    dataset = TeleDSDataset(server_ip=args.teleds_ip, server_port=args.teleds_port)
    sampler = DefaultSampler(dataset, shuffle=False)
    if args.mode == "sentence":
        collate_fn = SentenceCollate(tokenizer=tokenizer, max_len=args.max_sen_len, padding=True)
    else:
        collate_fn = PassageCollate(tokenizer=tokenizer, max_sentence_len=args.max_sen_len, max_sentence_num=args.max_sen_num, padding=True)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size * args.card_size,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor
    )

    all_tokens = 0
    total_len = len(dataloader)
    if total_len > args.max_iters:
        total_len = args.max_iters
    
    pbar = tqdm(range(total_len))
    for i, data in enumerate(dataloader):
        if i >= total_len:
            break
        if args.mode == 'sentence':
            input_ids, attention_mask = data 
            tokens = torch.sum(attention_mask).item()
            all_tokens += tokens
            pbar.set_postfix(tokens=all_tokens)
            pbar.update(1)
        else:
            batch_sentence_mask, batch_sentence_toks, batch_tok_mask = data 
            batch_size = batch_sentence_mask.size(0)
            for b in range(batch_size):
                attention_mask = batch_tok_mask[b][batch_sentence_mask[b]]
                tokens = torch.sum(attention_mask).item()
                all_tokens += tokens
            pbar.set_postfix(tokens=all_tokens)
            pbar.update(1)
    pbar.close()
    print("Total tokens:", all_tokens)


if __name__ == "__main__":
    args = make_parser().parse_args()
    main(args)
