import torch
import numpy as np
import sys

sys.path.append('/home/v-wanzho/wanjun/deepfake/code')
from utils.common import *
from tqdm import tqdm
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification
)


def calculate_sentence_pair_score(sens, model, tokenizer):
    sen_prev = sens[0]
    max_length = 128
    all_input_ids, all_attention_mask, all_token_type_ids = [], [], []
    for sen_idx, sen_now in enumerate(sens[1:]):
        inputs = tokenizer.encode_plus(sen_prev, sen_now, add_special_tokens=True, max_length=max_length,
                                       return_token_type_ids=True)
        sen_prev = sen_now
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
        attention_mask = [1] * len(input_ids)
        pad_token = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]
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
    small_dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids)
    eval_sampler = SequentialSampler(small_dataset)
    eval_dataloader = DataLoader(small_dataset, sampler=eval_sampler, batch_size=4)
    # output score
    preds = None
    for batch in eval_dataloader:
        batch = tuple(t.to('cuda') for t in batch)
        with torch.no_grad():
            outputs = model(input_ids=batch[0], attention_mask=batch[1])
            logits = torch.softmax(outputs[0], dim=-1)
            if preds is None:
                preds = logits.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
    return preds


if __name__ == '__main__':
    inf = '/mnt/wanjun/data/grover_kws_graph_info.jsonl'
    outp = '/mnt/wanjun/data/grover_human_nsp.jsonl'
    data = read_data(inf)
    tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
    model = RobertaForSequenceClassification.from_pretrained(
        '/mnt/wanjun/models/realnews_human_next_sentence_prediction_roberta_large/checkpoint-best')
       
    device = torch.device("cuda")
    n_gpu = torch.cuda.device_count()
    # if n_gpu>1:
    #   model = torch.nn.DataParallel(model)
    model.eval()
    model.to(device)
    max_seq_length = 128

    with open(outp, 'w', encoding='utf8') as outf:
        for line in tqdm(data):
            output = {}
            kws = line['information']['keywords']
            sens = [sen['sentence'].replace('\t', ' ') for sen in kws]
            try:
                score = calculate_sentence_pair_score(sens, model, tokenizer)
                if score is not None:
                    score = score.tolist()
                    assert (len(score) == len(sens) - 1), 'length of score {}, length of sentences {}'.format(
                        len(score), len(sens))
            except Exception as e:
                print(e)
                score = None

            output['next_sentence_prediction_score'] = score
            outf.write(json.dumps(output) + '\n')

