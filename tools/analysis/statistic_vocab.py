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
import argparse

from sentence_vae.utils import get_model, get_tokenizer
from sentence_vae.data import SentenceCollate


def make_parser():
    parser = argparse.ArgumentParser("SentenceVAE analysis parser.")
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--max_seq_len", type=int, default=1024)
    return parser


def main(args):
    tokenizer = get_tokenizer(ckpt_path=args.model_dir, max_seq_len=args.max_seq_len)
    vocab  = tokenizer.get_vocab()
    total_bytes = 0

    for token in vocab:
        token_bytes = len(token.encode('utf-8'))
        total_bytes += token_bytes

    average_bytes = total_bytes / len(vocab)
    print(f"Average bytes per token: {average_bytes:.2f}")


if __name__ == "__main__":
    args = make_parser().parse_args()
    main(args)
