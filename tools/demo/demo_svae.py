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

import os
import torch
import argparse

from sentence_vae.models import SentenceVAE
from sentence_vae.utils import get_config, get_tokenizer, get_dtype, load_yaml


def make_parser():
    parser = argparse.ArgumentParser("SentenceVAE demo parser.")
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--input", type=str, default=None)
    return parser


def main(args):
    cfg = load_yaml(args.config)
    expn = os.path.splitext(os.path.basename(args.config))[0]
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
    )
    exp_dir = f"exp/{expn}"
    ckpt_list = os.listdir(exp_dir)
    ckpt = args.checkpoint
    if ckpt is None:
        for ckpt_path in ckpt_list:
            if "best" in ckpt_path:
                ckpt_path = os.path.join(exp_dir, ckpt_path)
                ckpt = torch.load(ckpt_path)['state_dict']
        assert ckpt is not None, f"Not found the best checkpoint under {exp_dir}."
    else:
        assert os.path.exists(ckpt), f"Checkpoint {ckpt} not found."
        ckpt = torch.load(ckpt)["state_dict"]
    model.load_state_dict(ckpt)
    model.eval()

    tokenizer = get_tokenizer(ckpt_path=cfg["ref_model_dir"], max_seq_len=cfg["max_seq_len"])

    if args.input is None:
        input_texts = [
            'I love China.',
            'We come from Northwestern Polytechnical University.',
            "Hello,",
            "Welcome to TeleAI!",
            "What's your name?", 
            "What's your problem?", 
            "Hello, my dear friend", 
            "Today is Friday.",
            "There is Institute of Artificial Intelligence (TeleAI), China Telecom.",
            "One two three four five six seven eight nine ten~",
            "Hahaha... and you?", 
            "Yao yao ling xian!"
        ]
    else: 
        input_texts = [args.input]

    input_ids = tokenizer.batch_encode_plus(
        input_texts, 
        padding=True,
        truncation=True,
        max_length=cfg["max_seq_len"],
        return_tensors="pt"
    )
    attention_mask = input_ids['attention_mask']
    input_ids = input_ids['input_ids']

    for idx, input_text in enumerate(input_texts):
        print("--------------------")
        print(f"[{idx}]\tTest Input: ", end="")
        print(input_text)
        input_ids = tokenizer.encode(input_text, return_tensors='pt')
        print(f"\tTokens:{input_ids.size(1)}")
        print("\tVAE Output: ", end="")
        for output_id in model.streaming_generate(input_ids):
            output_word = tokenizer.decode(output_id, skip_special_tokens=True)
            print(output_word, end='')
        print()
    
    # exit(0)
    
    # output_ids, output_mask = model(input_ids, attention_mask, mode='predict')

    # output_texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    # print("VAE input:")
    # print(input_texts)
    # print("VAE output:")
    # print(output_texts)


if __name__ == "__main__":
    args = make_parser().parse_args()
    main(args)
