#Code Usage
##code file
###Folder
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


