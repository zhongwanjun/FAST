from enum import IntEnum
from torch.nn.parameter import Parameter
import torch
import torch.nn as nn
from allennlp.modules.attention import DotProductAttention
from allennlp.nn import util
import sys
import rnn_util
import math
from typing import Dict, Tuple, Sequence,Optional


class CustomRNN(nn.Module):
    def __init__(self,input_size,hidden_size,batch_first=True,max_seq_length=30):
        super(CustomRNN, self).__init__()
        self.batch_first = batch_first
        self.max_seq_length = max_seq_length
        self.rnn = torch.nn.LSTM(input_size=input_size,hidden_size=hidden_size,batch_first=batch_first)
        # self.rnn = NaiveLSTM(input_sz=input_size,hidden_sz=hidden_size)
        # self.rnn = rnn_util.LayerNormLSTM(input_size=input_size,hidden_size=hidden_size,num_layers=1,
        #                                    dropout=0,bidirectional=False,layer_norm_enabled=True)
    def forward(self,inputs,seq_lengths):#,score):

        packed_inputs = torch.nn.utils.rnn.pack_padded_sequence(inputs,seq_lengths,batch_first=self.batch_first,enforce_sorted=False)
        # res , (hn,cn) = self.rnn(input=packed_inputs,delta=min_score)

        res, (hn, cn) = self.rnn(input=packed_inputs)
        padded_res,_ = nn.utils.rnn.pad_packed_sequence(res,batch_first=self.batch_first,total_length=self.max_seq_length)#batch,max_seq_length,hidden
        # padded_gate,_ = nn.utils.rnn.pad_packed_sequence(gates, batch_first=self.batch_first,total_length=self.max_seq_length)

        return hn.squeeze(0),padded_res
        # padded_res, _ = nn.utils.rnn.pad_packed_sequence(res,batch_first=self.batch_first)

   