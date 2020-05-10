#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1h

import torch
import torch.nn as nn

class Highway(object):

	def __init__(self,conv_size, dropout):

		super(Highway, self).__init__()
		self.proj = nn.Linear(conv_size, conv_size, bias=True)
		self.gate = nn.Linear(conv_size, conv_size, bias=True)
		self.dropout = nn.Dropout(p=dropout)

	def forward(self, x_conv_out):
		x_proj = self.proj(x_conv_out)
		
		x_gate = self.gate(x_conv_out)
		x_highway = x_gate*x_proj + (1-x_gate)*x_proj
		x_out = self.dropout(x_highway)

		return x_out
### END YOUR CODE 

