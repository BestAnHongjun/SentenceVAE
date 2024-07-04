import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def get_config(ckpt_path):
    print(f"Loading config from {ckpt_path}")
    cfg = AutoConfig.from_pretrained(ckpt_path)
    return cfg


def get_tokenizer(ckpt_path, max_seq_len):
    print(f"Initializaing tokenizer from {ckpt_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        ckpt_path,
        model_max_length=max_seq_len,
        padding_side="right",
        trust_remote_code=True
    )
    return tokenizer


def get_dtype(dtype):
    if isinstance(dtype, torch.dtype):
        return dtype 

    if dtype == "bf16" or dtype == "bfloat16":
        dtype = torch.bfloat16
    elif dtype == "fp16" or dtype == "float16":
        dtype = torch.float16
    elif dtype == "fp32" or dtype == "float32":
        dtype = torch.float32
    else:
        raise NotImplementedError(f"Unknown dtype {dtype}")

    return dtype


def is_model_on_gpu(model) -> bool:
    """Returns if the model is fully loaded on GPUs."""
    return all("cuda" in str(param.device) for param in model.parameters())


def get_model(ckpt_path, dtype="fp16", device="cuda"):
    print(f"Initializaing model from {ckpt_path}")
    dtype = get_dtype(dtype)
    model_kwargs = {"torch_dtype": dtype}

    device_map = "auto"
    if device == "cpu":
        device_map = "cpu"

    model = AutoModelForCausalLM.from_pretrained(ckpt_path, device_map=device_map, trust_remote_code=True, **model_kwargs)
    model.eval()

    if device == "cuda":
        if not is_model_on_gpu(model):
            print("Warning: Some parameters are not on a GPU. Calibration can be slow or hit OOM")

    return model