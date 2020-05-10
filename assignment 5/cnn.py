#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1i

import torch.nn as nn

class cnn(object):

	def __init__(self, filters, in_channels=50, kernel_size=5):

		self.conv1d = nn.Conv1d(in_channels, filters, kernel_size, bias=True)
		self.relu = nn.ReLU()
		self.maxpool = nn.MaxPool1d(17)

	def forward(self, input):

		conv = self.conv1d(input)
		conv_out = self.maxpool(self.relu(conv))
		#print(conv_out.size())
		return conv_out

### END YOUR CODE

