#!/usr/bin/env python3
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# Network
# ------------------------------------------------------------------------------

class QaSimBiRNN(nn.Module):
    def __init__(self, conf):
        super(QaSimBiRNN, self).__init__()

        # Word embedding lookup
        self.encoder = nn.Embedding(conf['vocab-size'], conf['embedding-dim'],
                                    padding_idx=0)
        self.encoder.weight.requires_grad = False

        # BiRNN - Context
        c_input_size = conf['embedding-dim'] + conf['num-features']
        q_input_size = conf['embedding-dim']
        # BiRNN - Context
        self.c_rnn = EncoderBRNN(
            rnn_type=conf['rnn-type'],
            input_size=c_input_size,
            hidden_size=conf['hidden-size'],
            concat_layers=conf['concat-rnn-layers'],
            num_layers=conf['num-rnn-layers'],
            dropout_rate=conf['dropout-rate'],
            dropout_output=conf['dropout-output']
        )
        # BiRNN - Question
        self.q_rnn = EncoderBRNN(
            rnn_type=conf['rnn-type'],
            input_size=q_input_size,
            hidden_size=conf['hidden-size'],
            concat_layers=conf['concat-rnn-layers'],
            num_layers=conf['num-rnn-layers'],
            dropout_rate=conf['dropout-rate'],
            dropout_output=conf['dropout-output']
        )
        c_hidden_size = 2 * conf['hidden-size']
        q_hidden_size = 2 * conf['hidden-size']
        if conf['concat-rnn-layers']:
            c_hidden_size *= conf['num-rnn-layers']
            q_hidden_size *= conf['num-rnn-layers']

        # Non-linear sequence attention layer
        self.q_attn = LinearSeqAttn(q_hidden_size)
        self.c_attn = LinearSeqAttn(c_hidden_size)

        # Bilinear attention
        self.rel_attn = BilinearSeqAttn_v2v(c_hidden_size,
                                  q_hidden_size + len(conf['question-types']))

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

        # Encode (context + features) with RNN
        if x1_f is None:
            c_hiddens = self.c_rnn(x1_emb, x1_mask)
        else:
            c_hiddens = self.c_rnn(torch.cat([x1_emb, x1_f], 2), x1_mask)

        # Encode question embeddings with RNN; x2_f will be added later
        q_hiddens = self.q_rnn(x2_emb, x2_mask)

        # Attention layer for questions
        q_attn_weights = self.q_attn(q_hiddens, x2_mask)
        q_merged = weighted_avg(q_hiddens, q_attn_weights)

        # Attention layer for context
        c_attn_weights = self.c_attn(c_hiddens, x1_mask)
        c_merged = weighted_avg(c_hiddens, c_attn_weights)

        q_plus_feature = torch.cat([q_merged, x2_f], 1)
        # predict relevance
        # score = self.rel_attn(c_hiddens, q_plus_feature, x1_mask)
        score = self.rel_attn(c_merged, q_plus_feature, x1_mask)

        return score


# ------------------------------------------------------------------------------
# Network Components
# ------------------------------------------------------------------------------

class EncoderBRNN(nn.Module):
    """Bi-directional RNNs"""
    def __init__(self, rnn_type, input_size, hidden_size, num_layers,
                 concat_layers=False, dropout_rate=0, dropout_output=False):
        super(EncoderBRNN, self).__init__()
        self.padding = True
        self.dropout_rate = dropout_rate
        self.dropout_output = dropout_output
        self.num_layers = num_layers
        self.concat_layers = concat_layers
        self.rnns = nn.ModuleList()
        for i in range(num_layers):  # stacked rnn when num_layers 2+
            input_size = input_size if i == 0 else 2 * hidden_size
            if rnn_type == 'gru':
                self.rnns.append(nn.GRU(input_size, hidden_size,
                                        num_layers=self.num_layers,
                                        bidirectional=True))
            elif rnn_type == 'lstm':
                self.rnns.append(nn.LSTM(input_size, hidden_size,
                                         num_layers=self.num_layers,
                                         bidirectional=True))

    def forward(self, x, x_mask):
        """Forward wrapper: may provide options to choose padded or unpadded
        encoding. Here we just use padded sequence (slower though)

        Dimensions:
        x: batch * max_seq_len * hidden_dim
        x_mask: batch * max_seq_len (1 is for padding)
        """
        if self.padding or not self.training:
            output = self._forward_padded(x, x_mask)
        else:
            output = self._forward_unpadded(x, x_mask)

        return output.contiguous()  # make a single block in memory

    def _forward_unpadded(self, x, x_mask):
        """Faster encoding that ignores any padding."""
        # Transpose batch and sequence dims
        x = x.transpose(0, 1)

        # Encode all layers
        outputs = [x]
        for i in range(self.num_layers):
            rnn_input = outputs[-1]
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

        return output

    def _forward_padded(self, x, x_mask):
        # Sort input sequences (sequence lengths in descending order)
        lengths = x_mask.data.eq(0).long().sum(1).squeeze()
        _, idx_sort = torch.sort(lengths, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)  # to reverse the order
        lengths = list(lengths[idx_sort])
        idx_sort = Variable(idx_sort)
        idx_unsort = Variable(idx_unsort)
        x = x.index_select(0, idx_sort)
        x = x.transpose(0, 1)
        rnn_input = nn.utils.rnn.pack_padded_sequence(x, lengths)

        # Feed into RNN layers
        outputs = [rnn_input]
        for i in range(self.num_layers):
            rnn_input = outputs[-1]
            # dropout to input
            if self.dropout_rate > 0:
                dropout_input = F.dropout(rnn_input.data,
                                          p=self.dropout_rate,
                                          training=self.training)
                rnn_input = nn.utils.rnn.PackedSequence(dropout_input,
                                                        rnn_input.batch_sizes)
            outputs.append(self.rnns[i](rnn_input)[0])
        # pad_packed_sequence is the inverse of pack_padded_sequence
        for i, o in enumerate(outputs[1:], 1):
            outputs[i] = nn.utils.rnn.pad_packed_sequence(o)[0]
        # Concat hidden layers or take final
        if self.concat_layers:
            output = torch.cat(outputs[1:], 2)
        else:
            output = outputs[-1]
        # Inverse transformation
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


class LinearSeqAttn(nn.Module):
    """Attention layer for questions sequence (input_size = q_hidden_size) """
    def __init__(self, input_size):
        super(LinearSeqAttn, self).__init__()
        self.linear = nn.Linear(input_size, 1)  # https://goo.gl/9GUEcP

    def forward(self, x, x_mask):
        """
        In:
            x: batch * q_len * q_hidden_size
            x_mask:  batch * q_len
        Out:
            beta: batch * q_len
        """
        out_ = self.linear(x)
        out_.data.masked_fill_(x_mask.data.view_as(out_), -float('inf'))
        beta = F.softmax(out_, dim=1)

        return beta.squeeze(2)


class BilinearSeqAttn(nn.Module):
    """A bilinear attention layer over a sequence X w.r.t y"""
    def __init__(self, x_size, y_size):
        super(BilinearSeqAttn, self).__init__()
        self.linear = nn.Linear(y_size, x_size)

    def forward(self, x, y, x_mask):
        """
        In:
            x: batch * c_len * c_hidden_size
            y: batch * q_hidden_size
            x_mask: batch * c_len
        Out:
            out_ = batch * c_len
        """
        Wy = self.linear(y)
        xWy = x.bmm(Wy.unsqueeze(2)).squeeze(2)
        xWy.data.masked_fill_(x_mask.data, -float('inf'))
        return xWy.max(1)[0]


class BilinearSeqAttn_v2v(nn.Module):
    """Variant of BilinearSeqAttn, attention over x w.r.t y"""
    def __init__(self, x_size, y_size):
        super(BilinearSeqAttn_v2v, self).__init__()
        self.bilinear = nn.Bilinear(x_size, y_size, 1)

    def forward(self, x, y, x_mask):
        xWy = self.bilinear(x, y).squeeze(1)

        return xWy


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
