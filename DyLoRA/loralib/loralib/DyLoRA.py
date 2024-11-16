#!/usr/bin/env python
# coding=utf-8
# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import torch
import torch.nn as nn

class dynamic(nn.Module):
    def __init__(
        self,
        maximum_rank: int = 1,
    ):
        '''
        maximum_rank: maximum rank of the input matrix
        '''
        super(dynamic, self).__init__()
        self.maximum_rank = maximum_rank

        self.frozen = False
        self.current_rank = 0

    def get_dimension(self):
        return self.maximum_rank
    
    def get_rank(self):
        return self.current_rank
    
    def set_rank(self, rank, frozen=False):
        self.current_rank = max(0, min(rank, self.get_dimension()))
        self.frozen = frozen

    def forward(self, inputs, mode: bool = False):
        if self.training or mode:
            if self.frozen:
                # === By detaching pr, the gradients for the operations involving pr will 
                # not be computed during backpropagation. This is useful when you want to 
                # freeze part of the model and prevent it from being updated during training.
                # === In the context of the DyLoRA class, when self.frozen is True, it indicates 
                # that the rank is fixed, and the part of the input tensor corresponding to 
                # the fixed rank should not be updated.
                # === Sorts the information content of different ranks: 
                # The most informative components are retained in the lower ranks, 
                # while less informative components are truncated as the rank decreases.
                pr = inputs[:,:self.get_rank()].detach()
                r = inputs[:,self.get_rank()]
                
                if len(r.shape) == 1:
                    r = r.unsqueeze(-1)
                result = torch.cat([pr,r],dim=-1)

                # === Scales the output by a factor involving the maximum rank and the current rank to adjust the magnitude of the output tensor.
                return result * math.sqrt(self.get_dimension()/(self.get_rank()+1)) 
            else:
                return inputs[:,:self.get_rank()+1] * math.sqrt(self.get_dimension()/(self.get_rank()+1))

        else:
            # at test time, just return the reduced rank inputs
            # === self.get_rank()+1: This is why the rank in `run_glue.py` is in the range of [0, maximum_rank).
            return inputs[:,:self.get_rank()+1] * math.sqrt(self.get_dimension()/(self.get_rank()+1)) 