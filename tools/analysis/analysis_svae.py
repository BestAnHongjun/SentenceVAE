import yaml
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
    )

    batch_size = cfg["batch_size"] * args.cards
    device = torch.device(args.device)
    sentences = torch.ones((batch_size, cfg["max_seq_len"]), dtype=torch.long, device=device)
    sentence_mask = torch.ones((batch_size, cfg["max_seq_len"]), dtype=torch.long, device=device)
    
    print(get_model_complexity_info(model, inputs=(sentences, sentence_mask))['out_table'])


if __name__ == "__main__":
    args = make_parser().parse_args()
    main(args)
