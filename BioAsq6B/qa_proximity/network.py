#!/usr/bin/env python3

import torch
import torch.nn as nn
from torch.autograd import Variable

class QaProxBiRNN(nn.Module):
    def __init__(self, args):
        super(QaProxBiRNN, self).__init__()
        self.args = args

        # Define layers
        #-----------------------------------------------------------------------
        # Word embeddings
        self.encoder = nn.Embedding(args.vocab_size,
                                    args.embedding_dim,
                                    padding_idx=0)
        # Context
        self.c_rnn = nn.GRU(input_size=args.embedding_dim,
                            hidden_size=args.hidden_dim)
        # Question
        self.q_rnn = nn.GRU(input_size=args.embedding_dim,
                            hidden_size=args.hidden_dim)

    def forward(self, c, c_f, q, q_f):
        """Inputs:
        c = context word indices
        c_f = context word features indices
        """
        # Embed both context and question
        c_emb = self.embedding(c)
        q_emb = self.embedding(q)

    def predict(self):
        pass
