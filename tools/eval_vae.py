import sys 
import json
import argparse

import torch
from torch.utils.data import DataLoader

from mmengine.config import Config
from mmengine.runner import Runner, IterBasedTrainLoop 
from mmengine.dataset import DefaultSampler
from mmengine.dist.utils import init_dist

from sentence_vae.utils import get_config, get_tokenizer, get_dtype
from sentence_vae.models import SentenceVAE
from sentence_vae.data import TeleDSDataset, SentenceCollate


def make_parser():
    parser = argparse.ArgumentParser("SentenceVAE eval parser.")
    parser.add_argument("-c", "--config", type=str, required=True)
    # parser.add_argument("--local-rank", "--local_rank", type=int, default=0)
    # parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none', help='job launcher')
    return parser


def load_json(json_path):
    f = open(json_path, "r")
    cfg = json.load(f)
    f.close()
    return cfg


def main(args):
    cfg = load_json(args.config)
    ref_model_cfg = get_config(cfg["ref_model_dir"])
    work_dir = f"exp/SentenceVAE-{cfg['expn']}"

    model = SentenceVAE(
        hidden_size=ref_model_cfg.hidden_size,
        vocab_size=ref_model_cfg.vocab_size,
        device=torch.device(cfg["device"]),
        dtype=ref_model_cfg.torch_dtype,
        learnable_add=cfg["learnable_add"],
        load_ref_model=False,
        ref_model_dir=None,
        ref_model_dtype=None,
        finetune_embedding=False,
        num_attention_heads=ref_model_cfg.num_attention_heads,
        num_hidden_layers=cfg["num_hidden_layers"],
        max_seq_len=cfg["max_seq_len"],
        dropout=ref_model_cfg.dropout,
        bos_id=ref_model_cfg.bos_token_id,
        pad_id=ref_model_cfg.pad_token_id,
        end_id=ref_model_cfg.eos_token_id
    )

    # print(work_dir)

    # print(model)
    # exit(0)

    tokenizer = get_tokenizer(ckpt_path=cfg["ref_model_dir"], max_seq_len=cfg["max_seq_len"])

    eval_dataset = TeleDSDataset(server_ip=cfg["teleds_ip"], server_port=cfg["teleds_port"], max_samples=1000)
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

    for data in eval_dataloader:
        print(data)

    # default_hooks=dict(checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=cfg["save_checkpoint_iters"]))
    # runner = Runner(
    #     model=model,
    #     work_dir=f"exp/SentenceVAE-{cfg['expn']}",
    #     train_dataloader=train_dataloader,
    #     train_cfg=dict(by_epoch=True, max_epochs=cfg["pretrain_epoches"]),
    #     optim_wrapper=dict(optimizer=dict(type='SGD', lr=learning_rate, momentum=0.9, weight_decay=5e-3)),
    #     param_scheduler=[
    #         dict(type='LinearLR', start_factor=1e-3, by_epoch=False, begin=0, end=cfg["warmup_iters"]),
    #         dict(type='CosineAnnealingLR', by_epoch=True, T_max=cfg["pretrain_epoches"], convert_to_iter_based=True)
    #     ],
    #     visualizer=dict(type='Visualizer', vis_backends=[dict(type='TensorboardVisBackend')]),
    #     default_hooks=default_hooks,
    #     resume=cfg["resume_train"]
    # )
    # runner.train()


if __name__ == "__main__":
    args = make_parser().parse_args()
    main(args)
