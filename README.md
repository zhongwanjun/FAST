# Citation
Source code for the EMNLP2020 paper [Neural Deepfake Detection with Factual Structure of Text](https://aclanthology.org/2020.emnlp-main.193.pdf). If you find the code useful, please cite our paper:
```
@inproceedings{zhong-etal-2020-neural,
    title = "Neural Deepfake Detection with Factual Structure of Text",
    author = "Zhong, Wanjun  and
      Tang, Duyu  and
      Xu, Zenan  and
      Wang, Ruize  and
      Duan, Nan  and
      Zhou, Ming  and
      Wang, Jiahai  and
      Yin, Jian",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.emnlp-main.193",
    doi = "10.18653/v1/2020.emnlp-main.193",
    pages = "2461--2470",
    abstract = "Deepfake detection, the task of automatically discriminating machine-generated text, is increasingly critical with recent advances in natural language generative models. Existing approaches to deepfake detection typically represent documents with coarse-grained representations. However, they struggle to capture factual structures of documents, which is a discriminative factor between machine-generated and human-written text according to our statistical analysis. To address this, we propose a graph-based model that utilizes the factual structure of a document for deepfake detection of text. Our approach represents the factual structure of a given document as an entity graph, which is further utilized to learn sentence representations with a graph neural network. Sentence representations are then composed to a document representation for making predictions, where consistent relations between neighboring sentences are sequentially modeled. Results of experiments on two public deepfake datasets show that our approach significantly improves strong base models built with RoBERTa. Model analysis further indicates that our model can distinguish the difference in the factual structure between machine-generated text and human-written text.",
}
```
# Code Usage

## code file
### Folder
| code_file | function | usage |
| --- | --- | --- |
| data_process | | |
| extract_keywords.py | extract entities from gpt2 and grover dataset | python extract_keywords.py; need to change data folder and dataset type |
| graph_construction | | |
| contruct_graph_deepfake.py | Main function to construct graph, extract nodes, edges, tokens, and start and end idx for each sentence. Finally, each entity and sentence will be recorded by a mask index in the whole input sequence| python construct_graph_deepfake.py; need to change data path |
| build_graph.py | define functions for constructing graph | func build_graph(all_info); generate_rep_mask_based_on_graph(nodes, sens, tokenizer,max_seq_length); |
| next_sentence_prediction | | |
| process_news_data.py | construct training data for NSP model. positive ins is the NSP sentence pair. negative is the most similar sentence with B, suppose positive sentence pair is (A,B). | python process_news_data.py; need to change data path |
| run_classifier.py | training the NSP model | bash run_roberta.sh | 
| discriminator | | |
| transformers_graph_wiki/modeling_roberta.py | code for graph based model, add nsp score and wiki knowledge | 
| run_classifier_add_wiki.py | code for training and evaluate final model for the paper. (add nsp score, wiki knowledge) | bash run_roberta_add_wiki.sh |
| utils_graph_add_wiki | code for processing data for graph+nsp+wiki model | none |
| calculate_test_score_grover.py | calculate test score based on evaluation script for grover | python calculate_test_score_grover; need to change input data path and score file |


