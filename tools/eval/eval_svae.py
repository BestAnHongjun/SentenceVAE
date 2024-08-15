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

import os
import argparse
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from mmengine.dataset import DefaultSampler

from sentence_vae.models import SentenceVAE
from sentence_vae.utils import get_tokenizer, get_config, get_dtype, load_yaml
from sentence_vae.data import TeleDSDataset, SentenceCollate, SVAE_PPL


def make_parser():
    parser = argparse.ArgumentParser("SentenceVAE eval parser.")
    parser.add_argument("--server", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--max_eval_samples", type=int, default=1000)
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

    eval_dataset = TeleDSDataset(server_ip=args.server, server_port=args.port, eval_mode=True)
    eval_sampler = DefaultSampler(eval_dataset, shuffle=False)
    eval_collate_fn = SentenceCollate(tokenizer=tokenizer, max_len=cfg["max_seq_len"], padding=True)

    eval_dataloader = DataLoader(
        dataset=eval_dataset,
        batch_size=cfg["batch_size"],
        sampler=eval_sampler,
        collate_fn=eval_collate_fn,
        num_workers=cfg["dataloader_num_workers"],
        prefetch_factor=cfg["dataloader_prefetch_factor"]
    )

    metric = SVAE_PPL()
    device = torch.device(args.device)
    for data in tqdm(eval_dataloader):
        input_ids, attention_mask = data 
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        output = model(input_ids, attention_mask, mode='predict')
        metric.process(input_ids, output)
    results = metric.compute_metrics(metric.results)
    
    print("Exp:", expn)
    print("Best PPL:", results["eval_ppl"])


if __name__ == "__main__":
    args = make_parser().parse_args()
    main(args)
