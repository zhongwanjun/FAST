from utils.common import *
import nltk
import numpy as np
import re
def build_graph(all_info):
    nodes = []
    edges = []
    entity_occur = {}
    last_sen_cnt = 0
    sens = [sen['sentence'].replace('\t', ' ') for sen in all_info]
    all_kws = [sen['keywords']['entity'] for sen in all_info]
    sen2node = []
    for sen_idx, sen_kws in enumerate(all_kws):
        sen_tmp_node = []
        kws_cnt = 0
        sen_kws = list([kw for kw in set(sen_kws) if kw.strip() not in cachedStopWords])
        # sen_kws = list([kw for kw in set(sen_kws['keywords']) if kw.strip() not in cachedStopWords])
        if not keep_sen(sen_kws):
            sen2node.append([])
            continue
        for idx, kw in enumerate(sen_kws):
            kw = re.sub(r'[^a-zA-Z0-9,.\'\`!?]+', ' ', kw)
            words = [word for word in nltk.word_tokenize(kw) if
                     (word not in cachedStopWords and word.capitalize() not in cachedStopWords)]
            if keep_node(kw, words):
                sen_tmp_node.append(len(nodes))
                nodes.append({'text': kw, 'words': words, 'sentence_id': sen_idx})
                if kw not in entity_occur.keys():
                    entity_occur[kw] = 0
                entity_occur[kw] += 1
                kws_cnt += 1

        edges += [
            tuple([last_sen_cnt + i, last_sen_cnt + i + 1, 'inner'])
            for i in
            list(range(kws_cnt - 1))]

        last_sen_cnt += kws_cnt
        sen2node.append(sen_tmp_node)

    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if (j == i):
                continue
            # ans = similar_nodes(nodes[i]['words'], nodes[j]['words']
            ans = (nodes[i]['text'].strip() == nodes[j]['text'].strip())
            if (ans != 0):
                edges.append(tuple([min(i, j), max(i, j), 'inter']))

    # print(entity_occur)
    # input()
    if not nodes:
        return [], [], [],[],[]
    return nodes, list(set(edges)), entity_occur, sens, sen2node

def clean_string(string):
    return re.sub(r'[^a-zA-Z0-9,.\'!?]+', '', string)

def generate_rep_mask_based_on_graph(nodes, sens, tokenizer,max_seq_length):
    sen_start_idx = [0]
    sen_idx_pair = []
    sen_tokens = []
    all_tokens = []
    drop_nodes = []
    for sen in sens:
        sen_token = tokenizer.tokenize(sen)
        cleaned_sen_token = [clean_string(token) for token in sen_token]
        sen_token_wds = []
        # assert(len(sen_token_wds)==len(sen_token)),'{}\n{}'.format(sen_token_wds,sen_token)
        sen_tokens.append(cleaned_sen_token)
        sen_idx_pair.append(tuple([sen_start_idx[-1],sen_start_idx[-1]+len(sen_token)]))
        sen_start_idx.append(sen_start_idx[-1]+len(sen_token))

        all_tokens+=(sen_token)
    # print('all tokens are {}'.format(all_tokens))
    for nidx,node in enumerate(nodes):
        node_text = node['text']
        # node_tokens = tokenizer.tokenize(node_text)
        # cleaned_node_tokens = [clean_string(token) for token in node_tokens]
        # node_len = len(node_tokens)
        start_pos,node_len = first_index_list(sen_tokens[node['sentence_id']],clean_string(node_text))
        if start_pos != -1:
            final_start_pos = sen_start_idx[node['sentence_id']]+start_pos
            # final_start_pos = final_start_pos if final_start_pos<max_seq_length else max_seq_length
            # max_pos = min(max_seq_length,final_start_pos + node_len)
            max_pos = final_start_pos + node_len
            nodes[nidx]['spans'] = tuple([final_start_pos,max_pos])
        else:
            nodes[nidx]['spans'] = tuple([-1,-1])


        if nodes[nidx]['spans'][0] == -1:
            drop_nodes.append(nidx)
        else:
            nodes[nidx]['spans_check'] = all_tokens[final_start_pos:max_pos]
        # print('----------------')
        # print(node_text)
        # # print(node_tokens)
        # print(all_tokens[final_start_pos:max_pos])
        # print('----------------')

    # adj_matrix = np.eye(len(nodes))
    # for edge in edges:
    #     # if edge[3] == 'inner':
    #     adj_matrix[edge[0],edge[1]] = 1
    #     adj_matrix[edge[1], edge[0]] = 1
    #
    # for idx in drop_nodes:
    #     adj_matrix[idx,:] = 0
    #     adj_matrix[:,idx] = 0
    return nodes,all_tokens,drop_nodes,sen_idx_pair

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
def calculate_sentence_pair_score(model,sens,tokenizer):
    sen_prev = sens[0]
    max_length = 128
    all_input_ids,all_attention_mask,all_token_type_ids = [],[],[]
    for sen_idx, sen_now in enumerate(sens[1:]):
        inputs = tokenizer.encode_plus(sen_prev, sen_now, add_special_tokens=True, max_length=max_length,
                                       return_token_type_ids=True)
        sen_prev = sen_now
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
        attention_mask = [1] * len(input_ids)
        pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]
        pad_token_segment_id = 0
        padding_length = max_length - len(input_ids)
        input_ids = input_ids + ([pad_token] * padding_length)
        attention_mask = attention_mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
        all_input_ids.append(input_ids)
        all_attention_mask.append(attention_mask)
        all_token_type_ids.append(token_type_ids)
    all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
    all_attention_mask = torch.tensor(all_attention_mask, dtype=torch.long)
    all_token_type_ids = torch.tensor(all_token_type_ids, dtype=torch.long)
    small_dataset = TensorDataset(all_input_ids,all_attention_mask,all_token_type_ids)
    eval_sampler = SequentialSampler(small_dataset)
    eval_dataloader = DataLoader(small_dataset, sampler=eval_sampler, batch_size=4)
    #output score
    preds = None
    for batch in eval_dataloader:
        batch = tuple(t.to('cuda') for t in batch)
        with torch.no_grad():
            outputs = model(input_ids=batch[0],attention_mask=batch[1])
            logits = torch.softmax(outputs[0],dim=-1)
            if preds:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            else:
                preds = logits.detach().cpu().numpy()
    return preds
