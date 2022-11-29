import pandas as pd
import numpy as np
import json
def score(probs,split='test'):
    file = open('/mnt/wanjun/data/grover_kws_graph_info.jsonl', 'r', encoding='utf8')
    lines = file.readlines()
    full_info = []
    for (i, line) in enumerate(lines):
        line = json.loads(line.strip())
        if line['split'] == split:
            full_info.append(line)

    print(len(probs))
    print(len(full_info))
    score_df = pd.DataFrame(data=probs, columns=['human', 'machine']) # THIS MUST AGREE

    score_df['labels'] = [x['label'] for x in full_info]
    score_df['orig_split'] = [x['orig_split'] for x in full_info]
    score_df['ind30k'] = [x['ind30k'] for x in full_info]
    score_df.index.name = 'raw_index'
    score_df.reset_index(inplace=True)

    acc = np.mean(score_df[['machine', 'human']].idxmax(1) == score_df['labels'])
    print("Simple accuracy is {:.3f}".format(acc))

    # So really there are 3 groups here:
    # HUMAN WRITTEN ARTICLE
    # MACHINE WRITTEN ARTICLE PAIRED WITH HUMAN WRITTEN ARTICLE
    groups = {k:v for k, v in score_df.groupby('orig_split')}
    unpaired_human = groups.pop('train_burner')

    machine_v_human = {k: v.set_index('ind30k', drop=True) for k, v in groups['gen'].groupby('labels')}
    machine_vs_human_joined = machine_v_human['machine'].join(machine_v_human['human'], rsuffix='_humanpair')
    machine_vs_human_joined['is_right'] = machine_vs_human_joined['machine'] > machine_vs_human_joined['machine_humanpair']

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Combine unpaired with machinevs human for F1 eval.
    combined_scores = pd.concat((
        unpaired_human[['machine', 'human', 'labels']],
        machine_vs_human_joined[['machine', 'human', 'labels']],
    ),0).sort_values(by='machine', ascending=False)
    is_machine = (combined_scores['labels'] == 'machine').values

    combined_scores['recall'] = np.cumsum(is_machine) / np.sum(is_machine)
    combined_scores['precision'] = np.cumsum(is_machine) / (np.arange(is_machine.shape[0]) + 1)
    combined_scores['f1'] = 2 * (combined_scores['precision'] * combined_scores['recall']) / (combined_scores['precision'] + combined_scores['recall'] + 1e-5)
    combined_acc = np.mean(combined_scores[['machine', 'human']].idxmax(1) == combined_scores['labels'])


    stats = {
        'paired_acc': np.mean(machine_vs_human_joined['is_right']),
        'unpaired_acc': combined_acc,
        'unpaired_f1': np.max(combined_scores['f1']),
    }
    return stats

if __name__=='__main__':

    
    test_result_dir = './deepfake/results/roberta_base_grover_gcn_wiki_2.npy'
    probs = np.load(test_result_dir)
    stats = score(probs)
    print(stats)
