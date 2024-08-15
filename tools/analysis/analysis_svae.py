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

from mmengine.analysis import get_model_complexity_info

from sentence_vae.utils import get_config, get_dtype, load_yaml
from sentence_vae.models import SentenceVAE


def make_parser():
    parser = argparse.ArgumentParser("SentenceLLM train parser.")
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--cards", type=int, default=1)
    return parser


def main(args):
    cfg = load_yaml(args.config)
    ref_model_cfg = get_config(cfg["ref_model_dir"])
    
    model = SentenceVAE(
        hidden_size=ref_model_cfg.hidden_size,
        vocab_size=ref_model_cfg.vocab_size,
        device=torch.device(cfg["device"]),
        dtype=get_dtype(cfg["dtype"]),
        learnable_add=cfg["learnable_add"],
        load_ref_model=cfg["load_ref_model"],
        ref_model_dir=cfg["ref_model_dir"],
        ref_model_dtype=get_dtype(cfg["ref_model_dtype"]) if cfg["ref_model_dtype"] is not None else None,
        finetune_embedding=cfg["finetune_embedding"],
        num_attention_heads=ref_model_cfg.num_attention_heads,
        num_hidden_layers=cfg["num_hidden_layers"],
        max_seq_len=cfg["max_seq_len"],
        dropout=ref_model_cfg.dropout,
        bos_id=ref_model_cfg.bos_token_id,
        pad_id=ref_model_cfg.pad_token_id,
        end_id=ref_model_cfg.eos_token_id
    ).eval()

    batch_size = cfg["batch_size"] * args.cards
    device = torch.device(args.device)
    sentences = torch.ones((batch_size, cfg["max_seq_len"]), dtype=torch.long, device=device)
    sentence_mask = torch.ones((batch_size, cfg["max_seq_len"]), dtype=torch.long, device=device)
    
    print(get_model_complexity_info(model, inputs=(sentences, sentence_mask))['out_table'])


if __name__ == "__main__":
    args = make_parser().parse_args()
    main(args)
