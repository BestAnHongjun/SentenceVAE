import torch
import argparse

from mmengine.analysis import get_model_complexity_info

from sentence_vae.utils import get_config, get_dtype, load_yaml
from sentence_vae.models import SentenceLLM


def make_parser():
    parser = argparse.ArgumentParser("SentenceLLM train parser.")
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--cards", type=int, default=4)
    return parser


def main(args):
    cfg = load_yaml(args.config)
    svae_ref_model_cfg = get_config(cfg["svae"]["ref_model_dir"])
    
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
    ).eval()

    batch_size = cfg["batch_size"] * args.cards
    device = torch.device(args.device)
    batch_sentence_mask = torch.ones((batch_size, cfg["max_sen_num"]), dtype=torch.long, device=device)
    batch_sentence_toks = torch.ones((batch_size, cfg["max_sen_num"], cfg["max_sen_len"]), dtype=torch.long, device=device)
    batch_tok_mask = torch.ones((batch_size, cfg["max_sen_num"], cfg["max_sen_len"]), dtype=torch.long, device=device)

    print(get_model_complexity_info(model, inputs=(batch_sentence_mask, batch_sentence_toks, batch_tok_mask))['out_table'])


if __name__ == "__main__":
    args = make_parser().parse_args()
    main(args)
