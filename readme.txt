1. data_process folder: extract keywords from document
2. graph construction: construct graph based on extracted keywords and input document
3. next sentence prediction: construct data file for training next sentence prediction model; training the model for NSP model.
4. FAST: files for training the main model. run_classifier_add_wiki.py: main file for training model. utils_graph_add_wiki.py. data process file for the model. The model is defined in the RobertaForGraphBasedSequenceClassification class in modeling_roberta.py. 
   calculate_test_score_grover.py: calculate score for grover dataset.

STEP1 keyword extraction: go to data_process folder to run extract_keywords.py to extract keywords
STEP2 nsp model training: go to next sentence prediction folder, mode detail is in the readme.txt under the next sentence prediction folder.
STEP3 graph construction: go to graph construction folder to run construct_graph_deepfake.py
STEP4: run the main model: go to FAST and run run_roberta_add_wiki.sh to train the model. calculate_test_score_grover.py aims to calculate score after make the prediction. The result file and data file are also attached.
