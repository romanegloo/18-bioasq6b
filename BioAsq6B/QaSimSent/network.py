#!/usr/bin/env python3
import logging, coloredlogs

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

logger = logging.getLogger(__name__)
coloredlogs.install(
    level='DEBUG',
    fmt="[%(asctime)s %(levelname)s] %(message)s"
)

# ------------------------------------------------------------------------------
# Network
# ------------------------------------------------------------------------------

class QaSimBiRNN(nn.Module):
    def __init__(self, conf):
        super(QaSimBiRNN, self).__init__()

        # Word embedding lookup; encoder can be overwritten in test mode
        self.encoder = nn.Embedding(
            conf['vocab-size'], conf['embedding-dim'], padding_idx=0
        )
        self.encoder.weight.requires_grad = False

        # BiRNN - Context
        c_input_size = conf['embedding-dim'] + conf['num-features']
        q_input_size = conf['embedding-dim'] + conf['num-features']
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
        # self.c_attn = LinearSeqAttn(c_hidden_size)
        self.c_attn = Attention(c_hidden_size)

        # Bilinear attention
        # self.rel_attn = NTN(c_hidden_size, q_hidden_size + 4)
        self.rel_attn = BilinearSeqAttn(c_hidden_size, q_hidden_size + 4)

    def forward(self, x1, x1_f, x1_mask, x2, x2_f, x2_qtype, x2_mask):
        """Inputs:
        x1 = context word indices              [batch * len_c]
        x1_f = context word features indices   [batch * len_c * nfeat]
        x1_mask = context padding mask         [batch * len_c]
        x2 = question word indices             [batch * len_q]
        x2_f = question word features indices  [batch * len_q * nfeat]
        x2_qtype = question type               [batch * 4]
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
        if x2_f is None:
            q_hiddens = self.q_rnn(x2_emb, x2_mask)
        else:
            q_hiddens = self.q_rnn(torch.cat([x2_emb, x2_f], 2), x2_mask)

        # Attention layer for questions
        q_attn_weights = self.q_attn(q_hiddens, x2_mask)
        q_merged = weighted_avg(q_hiddens, q_attn_weights)

        # Attention layer for context
        # c_attn_weights = self.c_attn(c_hiddens, x1_mask)
        # c_merged = weighted_avg(c_hiddens, c_attn_weights)
        c_merged, c_weights = \
            self.c_attn(q_merged.unsqueeze(1), c_hiddens, x1_mask)
        q_plus_qtype = torch.cat([q_merged, x2_qtype], 1)
        # predict relevance
        # score = self.rel_attn(c_hiddens, q_plus_feature, x1_mask)
        score = self.rel_attn(c_merged.squeeze(1), q_plus_qtype)

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
        out_ = out_.squeeze(2)
        out_.data.masked_fill_(x_mask.data, -float('inf'))
        beta = F.softmax(out_, dim=1)

        return beta


class BilinearSeqAttn(nn.Module):
    """Variant of BilinearSeqAttn, attention over x w.r.t y"""
    def __init__(self, x_size, y_size):
        super(BilinearSeqAttn, self).__init__()
        self.bilinear = nn.Bilinear(x_size, y_size, 1)

    def forward(self, x, y):
        return self.bilinear(x, y).squeeze(1)


class Attention(nn.Module):
    """from https://is.gd/apncgK (torchnlp and from IBM seq2seq attention)
    Args:
        dimensions (int): Dimensionality of the query and context.
        attention_type (str, optional): How to compute the attention score:

            * dot: :math:`score(H_j,q) = H_j^T q`
            * general: :math:`score(H_j, q) = H_j^T W_a q`

    Example:

         >>> attention = Attention(256)
         >>> query = torch.randn(5, 1, 256)
         >>> context = torch.randn(5, 5, 256)
         >>> output, weights = attention(query, context)
         >>> output.size()
         torch.Size([5, 1, 256])
         >>> weights.size()
         torch.Size([5, 1, 5])
    """

    def __init__(self, dimensions, attention_type='general'):
        super(Attention, self).__init__()

        if attention_type not in ['dot', 'general']:
            raise ValueError('Invalid attention type selected.')

        self.attention_type = attention_type
        if self.attention_type == 'general':
            self.linear_in = nn.Linear(dimensions, dimensions, bias=False)

        self.linear_out = nn.Linear(dimensions * 2, dimensions, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()

    def forward(self, query, context, c_mask):
        """
        Args:
            query (:class:`torch.FloatTensor` [batch size, output length, dimensions]): Sequence of
                queries to query the context.
            context (:class:`torch.FloatTensor` [batch size, query length, dimensions]): Data
                overwhich to apply the attention mechanism.

        Returns:
            :class:`tuple` with `output` and `weights`:
            * **output** (:class:`torch.LongTensor` [batch size, output length, dimensions]):
              Tensor containing the attended features.
            * **weights** (:class:`torch.FloatTensor` [batch size, output length, query length]):
              Tensor containing attention weights.
        """
        batch_size, output_len, dimensions = query.size()
        query_len = context.size(1)

        if self.attention_type == "general":
            query = query.view(batch_size * output_len, dimensions)
            query = self.linear_in(query)
            query = query.view(batch_size, output_len, dimensions)

        # (batch_size, output_len, dimensions) * (batch_size, query_len, dimensions) ->
        # (batch_size, output_len, query_len)
        context.masked_fill_(c_mask.unsqueeze(2), 0)
        attention_scores = torch.bmm(query, context.transpose(1, 2).contiguous())

        # Compute weights across every context sequence
        attention_scores = attention_scores.view(batch_size * output_len, query_len)
        attention_weights = self.softmax(attention_scores)
        attention_weights = attention_weights.view(batch_size, output_len,
                                                   query_len)

        # (batch_size, output_len, query_len) * (batch_size, query_len, dimensions) ->
        # (batch_size, output_len, dimensions)
        mix = torch.bmm(attention_weights, context)

        # concat -> (batch_size * output_len, 2*dimensions)
        combined = torch.cat((mix, query), dim=2)
        combined = combined.view(batch_size * output_len, 2 * dimensions)

        # Apply linear_out on every 2nd dimension of concat
        # output -> (batch_size, output_len, dimensions)
        output = self.linear_out(combined).view(batch_size, output_len, dimensions)
        output = self.tanh(output)

        return output, attention_weights

class NTN(nn.Module):
    """Neural Tensor Network (NTN) by Socher"""
    def __init__(self, x_size, y_size):
        super(NTN, self).__init__()
        self.k = 2  # the number of slices
        self.bilinear = nn.Bilinear(x_size, y_size, self.k)
        self.linear_V = nn.Linear(x_size + y_size, self.k)
        self.nonlinear = torch.nn.ReLU6()
        self.linear_u = nn.Linear(self.k, 1, bias=False)

    def forward(self, x, y):
        xWy = self.bilinear(x, y)
        Vx_y = self.linear_V(torch.cat((x, y), -1))
        out = self.linear_u(self.nonlinear(xWy + Vx_y)).squeeze(1)
        return out


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
