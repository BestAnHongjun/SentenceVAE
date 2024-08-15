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
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from mmengine.dataset import DefaultSampler

from sentence_vae.utils import get_model, get_tokenizer
from sentence_vae.data import TeleDSDataset, SentenceCollate


def make_parser():
    parser = argparse.ArgumentParser("SentenceVAE eval parser.")
    parser.add_argument("--server", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--model_dtype", type=str, default="fp16")
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_eval_samples", type=int, default=1000)
    return parser


def main(args):
    model = get_model(args.model_dir, args.model_dtype, args.device).eval()
    tokenizer = get_tokenizer(ckpt_path=args.model_dir, max_seq_len=args.max_seq_len)

    eval_dataset = TeleDSDataset(server_ip=args.server, server_port=args.port, eval_mode=True, eval_samples=args.max_eval_samples)
    eval_sampler = DefaultSampler(eval_dataset, shuffle=False)
    eval_collate_fn = SentenceCollate(tokenizer=tokenizer, max_len=args.max_seq_len, padding=True)

    eval_dataloader = DataLoader(
        dataset=eval_dataset,
        batch_size=args.batch_size,
        sampler=eval_sampler,
        collate_fn=eval_collate_fn,
        num_workers=8,
        prefetch_factor=20
    )
        
    loss_list = []
    device = torch.device(args.device)
    for data in tqdm(eval_dataloader):
        input_ids, attention_mask = data 
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        with torch.no_grad():
            output = model(input_ids, attention_mask).logits
            batch, _ = input_ids.shape
            pad_ids = torch.zeros((batch, 1), device=device, dtype=input_ids.dtype).fill_(tokenizer.pad_token_id)
            tgt_ids = torch.cat((input_ids, pad_ids), dim=1)
            seq_lens = torch.sum(attention_mask, dim=1, keepdim=True)
            tgt_ids.scatter_(1, seq_lens, tokenizer.eos_token_id)
            attention_mask = attention_mask.bool()
            loss = F.cross_entropy(output[attention_mask], tgt_ids[:, 1:][attention_mask]).item()
            loss_list.append(loss)
    mean_loss = np.mean(np.array(loss_list))
    ppl = np.exp(mean_loss)
    
    print("Exp:", args.model_dir)
    print("Best PPL:", ppl)


if __name__ == "__main__":
    args = make_parser().parse_args()
    main(args)
