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
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, reduction='mean', eps=1e-6):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps

    def forward(self, logits, targets):
        max_logits = torch.max(logits, dim=-1, keepdim=True)[0]
        log_probs = logits - max_logits 
        log_probs = log_probs - torch.log(torch.sum(torch.exp(log_probs), dim=-1, keepdim=True))
        probs = torch.exp(log_probs)

        targets_one_hot = F.one_hot(targets, num_classes=logits.size(-1))
        targets_probs = torch.sum(probs * targets_one_hot, dim=-1)

        focal_weight = (1 - targets_probs) ** self.gamma
        log_targets_probs = torch.log(targets_probs + self.eps)
        loss = -focal_weight * log_targets_probs

        loss = loss[~torch.isnan(loss)]
        if loss.size(0) == 0:
            return torch.tensor(0, requires_grad=True).to(targets.device)

        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        
        return loss
