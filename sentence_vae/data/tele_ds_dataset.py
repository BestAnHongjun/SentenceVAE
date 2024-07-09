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
from time import sleep
from torch.utils.data import Dataset

from sentence_vae.utils import fetch_text_with_retry


class TeleDSDataset(Dataset):
    def __init__(
        self, 
        server_ip="127.0.0.1", 
        server_port=8000,
        max_samples=None
    ):
        self.server_url = f"http://{server_ip}:{server_port}"
        self.max_samples = max_samples

    def __len__(self):
        while True:
            status = fetch_text_with_retry(f"{self.server_url}/status").strip()
            if status != "ready.":
                print("TeleDS Server is not ready, retrying.")
            else: 
                break
            sleep(1)
        # print("Server is ready!")
        num = int(fetch_text_with_retry(f"{self.server_url}/count").strip())
        if self.max_samples is not None:
            return num if num < self.max_samples else self.max_samples
        return num
    
    def __getitem__(self, idx):
        text = fetch_text_with_retry(f"{self.server_url}/data/{idx}")
        return text
        