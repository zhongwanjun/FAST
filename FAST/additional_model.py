import torch
import torch.nn as nn
from allennlp.modules.attention import DotProductAttention
from allennlp.nn import util
from typing import Dict, Tuple, Sequence,Optional
from .Custom_LSTM import CustomRNN



class GCN(
    nn.Module):  # (batch_size, nodes_num, hidden_size) (batch_size, max_nodes_num, max_nodes_num)
    def __init__(self, GCN_layer, input_size):
        super(GCN, self).__init__()
        self.GCN_layer = GCN_layer
        self.GCNweight = nn.ModuleList()
        for _ in range(GCN_layer):
            self.GCNweight.append(nn.Linear(input_size, input_size))

    def normalize_laplacian_matrix(self, adj):
        row_sum_invSqrt, temp = torch.pow(adj.sum(2) + 1e-30, -0.5), []  # 每一行求和获得每个点的度，并求-1/2次方
        for i in range(adj.size()[0]):
            temp.append(torch.diag(row_sum_invSqrt[i]))  # 有batch_size所以循环一个个的生成对角矩阵
        degree_matrix = torch.cat(temp, dim=0).view(adj.size())  # 得到的进行拼接然后reshape成它应该的样子
        return torch.matmul(torch.matmul(degree_matrix, adj), degree_matrix).cuda()  # 返回DAD      注意还没修改为返回常量！

    def forward(self, nodes_rep, adj_metric):
        normalized_laplacian_matrix = self.normalize_laplacian_matrix(adj_metric)  # 归一化拉普拉斯矩阵

        normalized_laplacian_matrix.requires_grad = False

        # claim_rep_history = []

        nodes_rep_history = [nodes_rep]

        for i in range(self.GCN_layer):  # 直接循环多轮卷积

            tmp_rep = torch.bmm(normalized_laplacian_matrix, nodes_rep_history[i])
            nodes_rep_history.append(
                torch.nn.functional.tanh(self.GCNweight[i](tmp_rep)))

        nodes_rep_history = torch.stack(nodes_rep_history, dim=0)
        return nodes_rep_history


def function_align(x, y, x_mask, y_mask, input_size):
    x_mask_tile = x_mask.unsqueeze(-1).repeat(1, 1, x.shape[-1])
    y_mask_tile = y_mask.unsqueeze(-1).repeat(1, 1, y.shape[-1])
    # x = torch.bmm(x_mask.float(), x)
    # y = torch.bmm(y_mask.float(), y)
    x = x * x_mask_tile.float()
    y = y * y_mask_tile.float()
    return torch.cat([x - y, x * y], dim=2)


def mask_mean(x, x_mask, dim):
    '''
    :param x: batch,nodes_num,hidden_size
    :param x_mask: batch,nodes_num
    :param dim:
    :return: x
    '''
    x_mask_tile = x_mask.unsqueeze(-1).repeat(1, 1, x.shape[-1])
    assert (x.shape == x_mask_tile.shape),'x shape {}, x_mask_tile shape {}'.format(x.shape,x_mask_tile.shape)

    result = torch.sum(x * x_mask_tile.float(), dim=dim) / (torch.sum(x_mask_tile.float(), dim=dim) + 1e-30)

    return result









class GCNGraphAgg(nn.Module):
    def __init__(self, input_size, node_size,wiki_size=30,max_sentence_num=30):
        super(GCNGraphAgg, self).__init__()
        self.input_size = input_size
        # self.W_nei = nn.Linear(input_size,input_size)
        # self.W_node = nn.Linear(input_size,input_size)
        self.max_sentence_num = max_sentence_num
        self.gcn_layer = 4
        # self.node_size = node_size
        self.node_size = node_size + wiki_size
        self.wiki_emb_size = 100
        self.graph_node_proj = nn.Linear(input_size, node_size)

        self.graph_node_proj_wiki = nn.Linear(self.wiki_emb_size, wiki_size)
        # self.attention = GraphRelationalAttention(self.node_size, node_ksize)
        self.align_proj = nn.Linear(self.node_size * 2, self.node_size)
        self.GCN = GCN(self.gcn_layer, self.node_size)
       
        self.rnn_coherence_proj = CustomRNN(input_size=self.node_size, hidden_size=self.node_size, batch_first=True,
                                            max_seq_length=max_sentence_num)

       

    def forward(self, hidden_states, nodes_index_mask, adj_metric, node_mask,sen2node,sentence_mask,sentence_length,nsp_score, nodes_ent_emb):
        '''
        :param hidden_states: batch,seq_len,hidden_size
        :param nodes_mask: batch,node_num, seq_len
        :param claim_node_mask: batch,claim_node_num, seq_len
        :return: logits
        '''
        # batch, node_num, hidden_size
        '''evidence nodes and edges presentation'''
        nodes_rep = torch.bmm(nodes_index_mask,
                              hidden_states)  # / (torch.sum(evi_nodes_mask,dim=-1).unsqueeze(-1).repeat(1,1,hidden_states.shape[-1])+1e-30)
        # nodes_rep = torch.cat([nodes_rep,nodes_ent_emb],dim=-1)
        nodes_rep = torch.relu(self.graph_node_proj(nodes_rep))
        wiki_nodes_rep = torch.relu(self.graph_node_proj_wiki(nodes_ent_emb))
        nodes_rep = torch.cat([nodes_rep, wiki_nodes_rep], dim=-1)
        '''GCN propagation'''
        nodes_rep_history = self.GCN(nodes_rep, adj_metric)

        joint_nodes_rep = nodes_rep_history[-1, :, :, :]  # batch,node_num,node_size

        sens_rep = torch.bmm(sen2node,joint_nodes_rep)#batch, sen_num, rep_dim

        
        final_rep,padded_output = self.rnn_coherence_proj(sens_rep, sentence_length)
        joint_sens_output = torch.cat([padded_output[:,0:-1,:],padded_output[:,1:,:]],dim=-1)
        

        tmp_sentence_mask = sentence_mask.unsqueeze(2).repeat(1,1,joint_sens_output.shape[2])
        masked_output = tmp_sentence_mask * joint_sens_output
        nsp_score = nsp_score.unsqueeze(1)
        final_rep = torch.bmm(nsp_score,masked_output).squeeze(1)

        


        return final_rep#,mean_score,scores



