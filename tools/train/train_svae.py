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
import argparse

import torch
from torch.utils.data import DataLoader

from mmengine.runner import Runner
from mmengine.dataset import DefaultSampler
from mmengine.dist.utils import init_dist

from sentence_vae.utils import get_config, get_tokenizer, get_dtype, load_yaml
from sentence_vae.models import SentenceVAE
from sentence_vae.data import TeleDSDataset, SentenceCollate, SVAE_PPL


def make_parser():
    parser = argparse.ArgumentParser("SentenceVAE train parser.")
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("--local-rank", "--local_rank", type=int, default=0)
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none', help='job launcher')
    parser.add_argument("--teleds_ip", type=str, default="127.0.0.1")
    parser.add_argument("--teleds_port", type=int, default=8000)
    return parser


def main(args):
    cfg = load_yaml(args.config)
    expn = os.path.splitext(os.path.basename(args.config))[0]
    ref_model_cfg = get_config(cfg["ref_model_dir"])

    if args.launcher != 'none':
        init_dist(args.launcher)

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

    tokenizer = get_tokenizer(ckpt_path=cfg["ref_model_dir"], max_seq_len=cfg["max_seq_len"])

    train_dataset = TeleDSDataset(server_ip=args.teleds_ip, server_port=args.teleds_port)
    train_sampler = DefaultSampler(train_dataset, shuffle=False)
    train_collate_fn = SentenceCollate(tokenizer=tokenizer, max_len=cfg["max_seq_len"], padding=True)

    eval_dataset = TeleDSDataset(server_ip=args.teleds_ip, server_port=args.teleds_port, eval_mode=True)
    eval_sampleer = DefaultSampler(eval_dataset, shuffle=False)
    eval_collate_fn = SentenceCollate(tokenizer=tokenizer, max_len=cfg["max_seq_len"], padding=True)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg["batch_size"],
        sampler=train_sampler,
        collate_fn=train_collate_fn,
        num_workers=cfg["dataloader_num_workers"],
        prefetch_factor=cfg["dataloader_prefetch_factor"]
    )
    eval_dataloader = DataLoader(
        dataset=eval_dataset,
        batch_size=cfg["batch_size"],
        sampler=eval_sampleer,
        collate_fn=eval_collate_fn,
        num_workers=cfg["dataloader_num_workers"],
        prefetch_factor=cfg["dataloader_prefetch_factor"]
    )

    learning_rate = cfg["batch_size"] * cfg["base_lr"]

    default_hooks=dict(checkpoint=dict(
        type='CheckpointHook', 
        by_epoch=False, 
        interval=cfg["save_checkpoint_iters"],
        max_keep_ckpts=cfg["max_keep_ckpts"],
        save_best='eval_ppl', rule='less', published_keys=['meta', 'state_dict']
    ))
    runner = Runner(
        model=model,
        work_dir=f"exp/{expn}",
        train_dataloader=train_dataloader,
        val_dataloader=eval_dataloader,
        val_cfg=dict(),
        val_evaluator=dict(type=SVAE_PPL),
        train_cfg=dict(by_epoch=False, max_iters=cfg["max_iters"], val_interval=cfg["val_iters"]),
        optim_wrapper=dict(type="AmpOptimWrapper", optimizer=dict(type='AdamW', lr=learning_rate, weight_decay=0.01), clip_grad=dict(max_norm=1)),
        param_scheduler=[
            dict(type='LinearLR', start_factor=1e-3, by_epoch=False, begin=0, end=cfg["warmup_iters"]),
            dict(type='CosineAnnealingLR', by_epoch=False, T_max=cfg["cosineannealinglr_tmax"])
        ],
        visualizer=dict(type='Visualizer', vis_backends=[dict(type='TensorboardVisBackend')]),
        default_hooks=default_hooks,
        custom_hooks=[dict(type='EMAHook')],
        resume=cfg["resume_train"]
    )
    runner.train()


if __name__ == "__main__":
    args = make_parser().parse_args()
    main(args)
