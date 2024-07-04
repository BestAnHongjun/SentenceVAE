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

import torch

from sentence_vae.models import SentenceVAE
from sentence_vae.utils.llm import get_config, get_tokenizer


if __name__ == "__main__":
    max_seq_len = 512
    model_dir = "model_repo/opt-2.7b"
    config = get_config(model_dir)
    model = SentenceVAE(
        hidden_size=config.hidden_size,
        vocab_size=config.vocab_size,
        device=torch.device("cuda"),
        dtype=config.torch_dtype,
        load_ref_model=False,
        num_attention_heads=config.num_attention_heads,
        num_hidden_layers=2,
        max_seq_len=max_seq_len,
        dropout=config.dropout,
        bos_id=config.bos_token_id,
        pad_id=config.pad_token_id,
        end_id=config.eos_token_id
    )
    ckpt = torch.load('/workspace/project/Sentence-VAE/exp/SentenceVAE-opt-2.7b-h2-r2/iter_180000.pth')['state_dict']
    model.load_state_dict(ckpt)
    model.eval()

    tokenizer = get_tokenizer(ckpt_path=model_dir, max_seq_len=max_seq_len)

    input_texts = [
        "What's you name?", 
        "hello!", 
        "What's your problem?", 
        "hahaha... and you?", 
        "Hello, my dear friend", 
        "Sun zhe nb!",
        "Today is Friday.",
        "Holy shit!",
        "!!!",
        "One two three four five fix seven eight nine ten~",
        "Yao yao ling xian!"
    ]

    input_ids = tokenizer.batch_encode_plus(
        input_texts, 
        padding=True,
        truncation=True,
        max_length=max_seq_len,
        return_tensors="pt"
    )
    attention_mask = input_ids['attention_mask']
    input_ids = input_ids['input_ids']

    for idx, input_text in enumerate(input_texts):
        print("--------------------")
        print(f"[{idx}]\tTest Input: ", end="")
        print(input_text)
        input_ids = tokenizer.encode(input_text, return_tensors='pt')
        print("\tVAE Output: ", end="")
        for output_id in model(input_ids, mode='predict', streaming=True):
            output_word = tokenizer.decode(output_id[0], skip_special_tokens=True)
            print(output_word, end='')
        print()
    
    exit(0)
    
    output_ids, output_mask = model(input_ids, attention_mask, mode='predict')

    output_texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    print("VAE input:")
    print(input_texts)
    print("VAE output:")
    print(output_texts)
