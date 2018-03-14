#!/usr/bin/env python3
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# Network Components
# ------------------------------------------------------------------------------

class EncoderBRNN(nn.Module):
    """Bi-directional RNNs"""
    def __init__(self, input_size, hidden_size, num_layers, rnn_type,
                 dropout_rate=0, dropout_output=False, concat_layers=False,
                 padding=False, bidirection=True):
        super(EncoderBRNN, self).__init__()
        self.padding = padding
        self.dropout_output = dropout_output
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers
        self.concat_layers = concat_layers
        self.rnns = nn.ModuleList()
        for i in range(num_layers):  # stacked rnn when num_layers 2+
            input_size = input_size if i == 0 else 2 * hidden_size
            if rnn_type == 'gru':
                self.rnns.append(nn.GRU(input_size, hidden_size,
                                        num_layers=self.num_layers,
                                        bidirectional=bidirection))
            elif rnn_type == 'lstm':
                self.rnns.append(nn.LSTM(input_size, hidden_size,
                                        num_layers=self.num_layers,
                                        bidirectional=bidirection))

    def forward(self, x, x_mask):
        """Encode either padded or non-padded sequences.

        Can choose to either handle or ignore variable length sequences.
        Always handle padding in eval.

        Args:
            x: batch * len * hdim
            x_mask: batch * len (1 for padding, 0 for true)
        Output:
            x_encoded: batch * len * hdim_encoded
        """
        # print('brnn forward: ', x.size(), x_mask.size())
        if x_mask.data.sum() == 0:
            # No padding necessary.
            output = self._forward_unpadded(x, x_mask)
        elif self.padding or not self.training:
            # Pad if we care or if its during eval.
            output = self._forward_padded(x, x_mask)
        else:
            # We don't care.
            output = self._forward_unpadded(x, x_mask)

        return output.contiguous()

    def _forward_unpadded(self, x, x_mask):
        """Faster encoding that ignores any padding."""
        # Transpose batch and sequence dims
        x = x.transpose(0, 1)

        # Encode all layers
        outputs = [x]
        for i in range(self.num_layers):
            rnn_input = outputs[-1]

            # Apply dropout to hidden input
            if self.dropout_rate > 0:
                rnn_input = F.dropout(rnn_input,
                                      p=self.dropout_rate,
                                      training=self.training)
            # Forward
            rnn_output = self.rnns[i](rnn_input)[0]
            outputs.append(rnn_output)

        # Concat hidden layers
        # print('output length', len(outputs))
        if self.concat_layers:
            output = torch.cat(outputs[1:], 2)
        else:
            output = outputs[-1]

        # Transpose back
        output = output.transpose(0, 1)

        # Dropout on output layer
        if self.dropout_output and self.dropout_rate > 0:
            output = F.dropout(output,
                               p=self.dropout_rate,
                               training=self.training)
        return output

    def _forward_padded(self, x, x_mask):
        """Slower (significantly), but more precise, encoding that handles
        padding.
        """
        # Compute sorted sequence lengths
        lengths = x_mask.data.eq(0).long().sum(1).squeeze()
        _, idx_sort = torch.sort(lengths, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)

        lengths = list(lengths[idx_sort])
        idx_sort = Variable(idx_sort)
        idx_unsort = Variable(idx_unsort)

        # Sort x
        x = x.index_select(0, idx_sort)

        # Transpose batch and sequence dims
        x = x.transpose(0, 1)

        # Pack it up
        rnn_input = nn.utils.rnn.pack_padded_sequence(x, lengths)

        # Encode all layers
        outputs = [rnn_input]
        for i in range(self.num_layers):
            rnn_input = outputs[-1]

            # Apply dropout to input
            if self.dropout_rate > 0:
                dropout_input = F.dropout(rnn_input.data,
                                          p=self.dropout_rate,
                                          training=self.training)
                rnn_input = nn.utils.rnn.PackedSequence(dropout_input,
                                                        rnn_input.batch_sizes)
            outputs.append(self.rnns[i](rnn_input)[0])

        # Unpack everything
        for i, o in enumerate(outputs[1:], 1):
            outputs[i] = nn.utils.rnn.pad_packed_sequence(o)[0]

        # Concat hidden layers or take final
        if self.concat_layers:
            output = torch.cat(outputs[1:], 2)
        else:
            output = outputs[-1]

        # Transpose and unsort
        output = output.transpose(0, 1)
        output = output.index_select(0, idx_unsort)

        # Pad up to original batch sequence length
        if output.size(1) != x_mask.size(1):
            padding = torch.zeros(output.size(0),
                                  x_mask.size(1) - output.size(1),
                                  output.size(2)).type(output.data.type())
            output = torch.cat([output, Variable(padding)], 1)

        # Dropout on output layer
        if self.dropout_output and self.dropout_rate > 0:
            output = F.dropout(output,
                               p=self.dropout_rate,
                               training=self.training)
        return output


class BilinearSeqAttn(nn.Module):
    """A bilinear attention layer over a sequence X w.r.t y"""

    def __init__(self, x_size, y_size):
        super(BilinearSeqAttn, self).__init__()
        self.linear = nn.Linear(y_size, x_size)

    def forward(self, x, y, x_mask):
        """
        Args:
            x: batch * len * hdim1
            y: batch * hdim2
            x_mask: batch * len (1 for padding, 0 for true)
        Output:
            alpha = max(batch * len)
        """
        Wy = self.linear(y)
        xWy = x.bmm(Wy.unsqueeze(2)).squeeze(2)
        xWy.data.masked_fill_(x_mask.data, -float('inf'))
        p = xWy.max(1)[0]
        return p

class NTN(nn.Module):
    """Neural Tensor Network (http://stanford.io/2nTUcLt)"""
    def __init__(self, e1_size, e2_size):
        super(NTN, self).__init__()
        self.bilinear_dim = 3

        pass

    def forward(self, x, y, x_mask):
        """
        Args:
            x: batch * len * hdim1
            y: batch * hdim2
            x_mask: batch * len (1 for padding, 0 for true)
        """
        We2 = self.linear(y)
        e1We2 = x.bmm(We2.unsqueeze(2)).squeeze(2)


# ------------------------------------------------------------------------------
# Functional
# ------------------------------------------------------------------------------

def uniform_weights(x, x_mask):
    """Return uniform weights over non-masked x (a sequence of vectors).

    Args:
        x: batch * len * hdim
        x_mask: batch * len (1 for padding, 0 for true)
    Output:
        x_avg: batch * hdim
    """
    alpha = Variable(torch.ones(x.size(0), x.size(1)))
    if x.data.is_cuda:
        alpha = alpha.cuda()
    alpha = alpha * x_mask.eq(0).float()
    alpha = alpha / alpha.sum(1, keepdim=True).expand(alpha.size())
    return alpha


def weighted_avg(x, weights):
    """Return a weighted average of x (a sequence of vectors).

    Args:
        x: batch * len * hdim
        weights: batch * len, sum(dim = 1) = 1
    Output:
        x_avg: batch * hdim
    """
    return weights.unsqueeze(1).bmm(x).squeeze(1)


# ------------------------------------------------------------------------------
# Network
# ------------------------------------------------------------------------------

class QaProxBiRNN(nn.Module):
    def __init__(self, args):
        super(QaProxBiRNN, self).__init__()
        self.args = args
        self.num_rnn_layers = args.num_rnn_layers

        # Define layers
        #-----------------------------------------------------------------------
        # Word embeddings
        self.encoder = nn.Embedding(args.vocab_size,
                                    args.embedding_dim,
                                    padding_idx=0)
        # Context
        c_input_size = args.embedding_dim + args.num_features
        if args.use_idf:
            c_input_size += 1
        # print(c_input_size, args.embedding_dim, args.num_features)
        self.c_rnn = EncoderBRNN(
            rnn_type=args.rnn_type,
            input_size=c_input_size,
            hidden_size=args.hidden_size,
            concat_layers=args.concat_rnn_layers,
            num_layers=self.num_rnn_layers,
            bidirection=(not args.uni_direction)
        )
        # Question
        self.q_rnn = EncoderBRNN(
            rnn_type=args.rnn_type,
            input_size=c_input_size,
            hidden_size=args.hidden_size,
            concat_layers=args.concat_rnn_layers,
            num_layers=self.num_rnn_layers,
            bidirection=(not args.uni_direction)
        )

        c_hidden_size = args.hidden_size \
            if args.uni_direction else 2 * args.hidden_size
        q_hidden_size = args.hidden_size \
            if args.uni_direction else 2 * args.hidden_size

        if args.concat_rnn_layers:
            c_hidden_size *= self.num_rnn_layers
            q_hidden_size *= self.num_rnn_layers
        # Bilinear attention
        self.rel_attn = NTN(c_hidden_size, q_hidden_size)
        self.rel_attn = BilinearSeqAttn(c_hidden_size, q_hidden_size)
        # self.rel_attn = nn.Bilinear(c_hidden_size, q_hidden_size, 2)

    def forward(self, x1, x1_f, x1_mask, x2, x2_f, x2_mask):
        """Inputs:
        x1 = context word indices              [batch * len_c]
        x1_f = context word features indices   [batch * len_c * nfeat]
        x1_mask = context padding mask         [batch * len_c]
        x2 = question word indices             [batch * len_q]
        x2_f = question word features indices  [batch * len_q * nfeat]
        x2_mask = question padding mask        [batch * len_q]

        Note. mask is not being used in with any of RNNs
        """
        # Embed both context and question
        x1_emb = self.encoder(x1)
        x2_emb = self.encoder(x2)

        # Dropout on embeddings
        if self.args.dropout_emb > 0:
            x1_emb = nn.functional.dropout(x1_emb, p=self.args.dropout_emb,
                                           training=self.training)
            x2_emb = nn.functional.dropout(x2_emb, p=self.args.dropout_emb,
                                           training=self.training)

        # Encode (context + features) with RNN
        if self.args.no_token_feature:
            c_hiddens = self.c_rnn(x1_emb, x1_mask)
        else:
            c_hiddens = self.c_rnn(torch.cat([x1_emb, x1_f], 2), x1_mask)

        # Encode (question + features) with RNN
        if self.args.no_token_feature:
            q_hiddens = self.q_rnn(x2_emb, x2_mask)
        else:
            q_hiddens = self.q_rnn(torch.cat([x2_emb, x2_f], 2), x2_mask)

        # Merge question hiddens
        q_merge_weights = uniform_weights(q_hiddens, x2_mask)
        q_hidden = weighted_avg(q_hiddens, q_merge_weights)

        # predict relevance
        score = self.rel_attn(c_hiddens, q_hidden, x1_mask)

        return score
