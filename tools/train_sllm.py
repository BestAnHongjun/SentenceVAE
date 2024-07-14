import os
import sys 
import yaml
import json
import argparse

import torch
from torch.utils.data import DataLoader
from torch.optim import SGD

from mmengine.config import Config
from mmengine.runner import Runner, IterBasedTrainLoop 
from mmengine.dataset import DefaultSampler
from mmengine.dist.utils import init_dist
from mmengine.optim import AmpOptimWrapper

from sentence_vae.utils import get_config, get_tokenizer, get_dtype, load_yaml
from sentence_vae.models import SentenceLLM
from sentence_vae.data import TeleDSDataset, PassageCollate

torch.multiprocessing.set_sharing_strategy('file_system')


def make_parser():
    parser = argparse.ArgumentParser("SentenceLLM train parser.")
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("--local-rank", "--local_rank", type=int, default=0)
    parser.add_argument("--launcher", choices=['none', 'pytorch', 'slurm', 'mpi'], default='none', help='job launcher')
    parser.add_argument("--teleds_ip", type=str, default="127.0.0.1")
    parser.add_argument("--teleds_port", type=int, default=8001)
    return parser


def main(args):
    cfg = load_yaml(args.config)
    expn = os.path.splitext(os.path.basename(args.config))[0]
    svae_ref_model_cfg = get_config(cfg["svae"]["ref_model_dir"])

    if args.launcher != 'none':
        init_dist(args.launcher)

    model = SentenceLLM(
        svae_hidden_size=svae_ref_model_cfg.hidden_size,
        svae_vocab_size=svae_ref_model_cfg.vocab_size,
        svae_learnable_add=cfg["svae"]["learnable_add"],
        svae_load_ref_model=cfg["svae"]["load_ref_model"],
        svae_ref_model_dir=cfg["svae"]["ref_model_dir"],
        svae_ref_model_dtype=svae_ref_model_cfg.torch_dtype,
        svae_finetune_embedding=cfg["svae"]["finetune_embedding"],
        svae_word_embed_proj_dim=svae_ref_model_cfg.word_embed_proj_dim,
        svae_num_attention_heads=svae_ref_model_cfg.num_attention_heads,
        svae_num_hidden_layers=cfg["svae"]["num_hidden_layers"],
        svae_model_path=cfg["svae"]["model_path"],
        llm_ref_model_dir=cfg["llm"]["ref_model_dir"],
        llm_ref_model_dtype=svae_ref_model_cfg.torch_dtype,
        llm_finetune_layers=cfg["llm"]["finetune_layers"],
        finetune_svae=cfg["finetune_svae"],
        max_sentence_len=cfg["max_sen_len"],
        max_sentence_num=cfg["max_sen_num"],
        dropout=svae_ref_model_cfg.dropout,
        bos_id=svae_ref_model_cfg.bos_token_id,
        pad_id=svae_ref_model_cfg.pad_token_id,
        end_id=svae_ref_model_cfg.eos_token_id,
        device=torch.device(cfg["device"]),
        dtype=get_dtype(cfg["dtype"])
    )

    tokenizer = get_tokenizer(ckpt_path=cfg["svae"]["ref_model_dir"], max_seq_len=cfg["max_sen_len"])

    train_dataset = TeleDSDataset(server_ip=args.teleds_ip, server_port=args.teleds_port)
    train_sampler = DefaultSampler(train_dataset, shuffle=False)
    train_collate_fn = PassageCollate(tokenizer=tokenizer, max_sentence_len=cfg["max_sen_len"], max_sentence_num=cfg["max_sen_num"], padding=True)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg["batch_size"],
        sampler=train_sampler,
        collate_fn=train_collate_fn,
        num_workers=cfg["dataloader_num_workers"],
        prefetch_factor=cfg["dataloader_prefetch_factor"]
    )

    learning_rate = cfg["batch_size"] * cfg["base_lr"]

    default_hooks=dict(checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=cfg["save_checkpoint_iters"]))
    runner = Runner(
        model=model,
        work_dir=f"exp/SentenceVAE-{expn}",
        train_dataloader=train_dataloader,
        train_cfg=dict(by_epoch=True, max_epochs=cfg["finetune_epoches"]),
        optim_wrapper=dict(type="AmpOptimWrapper", optimizer=dict(type='AdamW', lr=learning_rate, weight_decay=0.01), clip_grad=dict(max_norm=5)),
        param_scheduler=[
            dict(type='LinearLR', start_factor=1e-3, by_epoch=False, begin=0, end=cfg["warmup_iters"]),
            dict(type='CosineAnnealingLR', by_epoch=False, T_max=cfg["cosineannealinglr_tmax"])
        ],
        visualizer=dict(type='Visualizer', vis_backends=[dict(type='TensorboardVisBackend')]),
        default_hooks=default_hooks,
        resume=cfg["resume_train"]
    )
    runner.train()


if __name__ == "__main__":
    args = make_parser().parse_args()
    main(args)
