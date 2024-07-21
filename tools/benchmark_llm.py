import os
import re
import sys 
import json
import time
import shutil
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from mmengine.config import Config
from mmengine.runner import Runner, IterBasedTrainLoop 
from mmengine.dataset import DefaultSampler
from mmengine.dist.utils import init_dist

from sentence_vae.utils import get_model, get_tokenizer, get_dtype
from sentence_vae.models import SentenceVAE
from sentence_vae.data import TeleDSDataset, SentenceCollate


def make_parser():
    parser = argparse.ArgumentParser("SentenceVAE benchmark parser.")
    parser.add_argument("--server", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--model_dtype", type=str, default="fp16")
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--max_eval_samples", type=int, default=10)
    return parser


def main(args):
    model = get_model(args.model_dir, args.model_dtype, args.device).eval()
    tokenizer = get_tokenizer(ckpt_path=args.model_dir, max_seq_len=args.max_seq_len)

    eval_dataset = TeleDSDataset(server_ip=args.server, server_port=args.port, eval_mode=True, eval_samples=args.max_eval_samples)
    eval_sampler = DefaultSampler(eval_dataset, shuffle=False)
    eval_collate_fn = SentenceCollate(tokenizer=tokenizer, max_len=args.max_seq_len, padding=True, fix_len=False)

    eval_dataloader = DataLoader(
        dataset=eval_dataset,
        batch_size=1,
        sampler=eval_sampler,
        collate_fn=eval_collate_fn,
        num_workers=8,
        prefetch_factor=20
    )
        
    input_toks, output_toks = 0, 0
    input_times, output_times = 0, 0
    mem_usage = []
    device = torch.device(args.device)

    with torch.no_grad():
        for data in tqdm(eval_dataloader):
            input_ids, attention_mask = data 
            seq_len = torch.sum(attention_mask) // 2
            if seq_len < 1:
                continue
            input_ids = input_ids[:1, :seq_len].to(device)

            # 预填充阶段
            start_time = time.perf_counter()
            output = model(input_ids)
            input_time = time.perf_counter()
            
            input_toks += seq_len 
            input_times += input_time - start_time

            # 推理阶段
            while True:
                logits = output.logits 
                past_key_values = output.past_key_values 
                new_id = torch.argmax(logits[:1, -1:], dim=-1)
                input_ids = torch.concat((input_ids, new_id), dim=1)
                if new_id.item() == tokenizer.eos_token_id:
                    break 
                if input_ids.size(1) >= args.max_seq_len:
                    break 
                output = model(new_id, past_key_values=past_key_values)
                mem_usage.append([input_ids.size(1), torch.cuda.memory_allocated() / 1024 / 1024])

            output_time = time.perf_counter()
            output_toks += input_ids.size(1) - seq_len 
            output_times += output_time - input_time

    mem_usage = np.array(mem_usage)
    k, b = np.polyfit(mem_usage[:, 0], mem_usage[:, 1], 1)
    print(f"显存占用: {b:.2f}MB + {k:.2f}MB/token")
    print(f"Input: {input_toks / input_times:.2f} tokens/s")
    print(f"Output: {output_toks / output_times:.2f} tokens/s")


if __name__ == "__main__":
    args = make_parser().parse_args()
    main(args)
