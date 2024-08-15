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
import argparse

from sentence_vae.utils import get_model, get_tokenizer
from sentence_vae.data import SentenceCollate


def make_parser():
    parser = argparse.ArgumentParser("SentenceVAE benchmark parser.")
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--model_dtype", type=str, default="fp16")
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--input", type=str, default="Hello,")
    return parser


def main(args):
    model = get_model(args.model_dir, args.model_dtype, args.device).eval()
    tokenizer = get_tokenizer(ckpt_path=args.model_dir, max_seq_len=args.max_seq_len)

    collate_fn = SentenceCollate(tokenizer=tokenizer, max_len=args.max_seq_len, padding=True, fix_len=False)
    input_ids = tokenizer.batch_encode_plus([args.input], return_tensors="pt")['input_ids']

    device = torch.device(args.device)
    input_ids = input_ids.to(device)
    output = model(input_ids)

    print("Input:", args.input)
    print("Output:")
    while True:
        logits = output.logits 
        past_key_values = output.past_key_values 
        new_id = torch.argmax(logits[:1, -1:], dim=-1)
        input_ids = torch.concat((input_ids, new_id), dim=1)
        if new_id.item() == tokenizer.eos_token_id:
            break 
        if input_ids.size(1) >= args.max_seq_len:
            break 
        output_word = tokenizer.decode(new_id.item(), skip_special_tokens=True)
        print(output_word, end="")
        output = model(new_id, past_key_values=past_key_values)
    print()


if __name__ == "__main__":
    args = make_parser().parse_args()
    main(args)
