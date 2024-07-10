import os
import re
import sys 
import json
import shutil
import argparse
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

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
    return parser


def load_json(json_path):
    f = open(json_path, "r")
    cfg = json.load(f)
    f.close()
    return cfg


def extract_iter(filename):
    match = re.match(r"iter_(\d+)\.pth", filename)
    if match:
        return int(match.group(1))
    return None


def main(args):
    cfg = load_json(args.config)
    ref_model_cfg = get_config(cfg["ref_model_dir"])
    work_dir = f"exp/SentenceVAE-{cfg['expn']}"
    writer = SummaryWriter(f"exp/eval/SentenceVAE-{cfg['expn']}")

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

    tokenizer = get_tokenizer(ckpt_path=cfg["ref_model_dir"], max_seq_len=cfg["max_seq_len"])

    eval_dataset = TeleDSDataset(server_ip=cfg["teleds_ip"], server_port=cfg["teleds_port"], max_samples=cfg["max_eval_samples"])
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

    ckpt_list = os.listdir(work_dir)
    best_ppl, best_ckpt = np.exp(20), None
    for ckpt_name in tqdm(ckpt_list):
        iter_n = extract_iter(ckpt_name)
        if iter_n is None: continue
        ckpt_path = os.path.join(work_dir, ckpt_name)
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        
        loss_list = []
        for data in eval_dataloader:
            loss = model(*data, mode='loss')['total_loss'].item()
            loss_list.append(loss)
        mean_loss = np.mean(np.array(loss_list))
        ppl = np.exp(mean_loss)

        writer.add_scalar("Eval/PPL", ppl, iter_n)
        writer.add_scalar("Eval/Loss", mean_loss, iter_n)

        if ppl < best_ppl:
            best_ppl = ppl 
            best_ckpt = ckpt_path
    
    with open(f"exp/eval/SentenceVAE-{cfg['expn']}/best_checkpoint", "w") as f:
        f.write(best_ckpt)
    with open(f"exp/eval/SentenceVAE-{cfg['expn']}/log.txt", "w") as f:
        f.write(f"Exp:{work_dir}\nBest PPL:{best_ppl}\nBest ckpt:{best_ckpt}")
    shutil.copy(best_ckpt, f"exp/eval/SentenceVAE-{cfg['expn']}/best_checkpoint.pth")
    
    print("Exp:", work_dir)
    print("Best PPL:", best_ppl)
    print("Best ckpt:", best_ckpt)


if __name__ == "__main__":
    args = make_parser().parse_args()
    main(args)
