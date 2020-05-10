#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Michael Hahn <mhahn2@stanford.edu>
"""

import torch.nn as nn
from cnn import cnn
from highway import Highway
import torch
# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

# from cnn import CNN
# from highway import Highway

# End "do not change" 

class ModelEmbeddings(nn.Module): 
    """
    Class that converts input words to their CNN-based embeddings.
    """
    def __init__(self, embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output 
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.
        """
        super(ModelEmbeddings, self).__init__()
        self.embed_size = embed_size
        ## A4 code
        self.max_word_length = 21
        self.char_embed = 50
        self.max_sen_len = 18
        self.batch_size = 2
        pad_token_idx = vocab.char2id['<pad>']
        self.embeddings = nn.Embedding(len(vocab.char2id), self.char_embed, padding_idx=pad_token_idx)
        self.cnn = cnn(embed_size)
        self.highway = Highway(embed_size, 0.3)
        ## End A4 code

        ### YOUR CODE HERE for part 1j


        ### END YOUR CODE

    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the 
            CNN-based embeddings for each word of the sentences in the batch
        """
        ## A4 code
        batch_size = input.size(1)
        max_len = input.size(2)
        embed = self.embeddings(input.type(torch.LongTensor))
        #print(embed.size())

        embed = embed.permute(1,0,2,3)
        embed = embed.contiguous().view(-1, self.max_word_length, self.char_embed)
        embed = embed.permute(0,2,1)
        conv = self.cnn.forward(embed)
        
        #print(conv.size())
        output = self.highway.forward(conv.squeeze(dim=2))
        output = output.view(batch_size,-1,self.embed_size)
        #print(output.size())
        return output.view(-1,batch_size,self.embed_size)
        # return output
        ## End A4 code

        ### YOUR CODE HERE for part 1j



        ### END YOUR CODE

