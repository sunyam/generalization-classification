"""
Author: Sunyam

This file contains methods to run various Deep Learning models for text classification.
"""
import torch
import os
import sys
import pickle
import numpy as np
from pathlib import Path
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.token_indexers.elmo_indexer import ELMoTokenCharactersIndexer # for ELMo
from allennlp.data.token_indexers import PretrainedBertIndexer # for BERT

import data_reader
import train
import evaluate

def tokenizer(x: str):
    """
    Tokenizer for DatasetReader.
    """
    return [w.text for w in SpacyWordSplitter(language='en_core_web_sm', pos_tags=False).split_words(x)]

def tokenizer_bert(s: str):
    """
    Tokenizer for DatasetReader; BERT uses wordpiece embeddings.
    """
    return bert_token_indexer.wordpiece_tokenizer(s)

def run_model(name, use_elmo=False, save_predictions=False, save_model=False):
    """
    Trains the given deep learning model on train set, and evaluates on test set.
    
    Parameters
    ----------
    name: str
        name of the deep learning model to be run: lstm | bilstm | stacked_bilstm | cnn | bert
    use_elmo: bool
        use ELMo embeddings if True | GloVe embeddings if False
    save_predictions: bool
        If True, stores and returns the predicted probabilities mapped to sentence ID
    save_model: bool
        If True, saves the trained model along with its vocabulary
        
    Returns
    -------
    F1-score, Precision, Recall, Accuracy, Area Under Precision-Recall Curve on the test set; dictionary mapping predictions to ID, and number of training epochs for each fold.
    """
    # token_indexer maps tokens to integers; using special built-in indexers for ELMo and BERT to ensure mapping is consistent with the original models
    if use_elmo:
        token_indexer = ELMoTokenCharactersIndexer()
    elif name == 'bert':
        global bert_token_indexer
        bert_token_indexer = PretrainedBertIndexer(pretrained_model=BERT_MODEL, do_lowercase=True)
    else:
        token_indexer = SingleIdTokenIndexer()

    if name == 'bert': # BERT uses a special wordpiece tokenizer
        reader = data_reader.GeneralizationDatasetReader(tokenizer=tokenizer_bert, token_indexers={"tokens": bert_token_indexer},
                                                         label_cols=LABEL_COLS)
    else:
        reader = data_reader.GeneralizationDatasetReader(tokenizer=tokenizer, token_indexers={"tokens": token_indexer},
                                                         label_cols=LABEL_COLS)

    map_id_pred_probability = {} # used if save_predictions is True
    f1s, precision_s, recall_s, accuracies, AUPRCs, n_epochs = [], [], [], [], [], []

    for fold_number in range(1,4): # 3-fold cross validation
        train_fname = 'train_data_fold_'+str(fold_number)+'.csv'
        val_fname = 'val_data_fold_'+str(fold_number)+'.csv'
        test_fname = 'test_data_fold_'+str(fold_number)+'.csv'

        train_dataset = reader.read(file_path=DATA_ROOT / train_fname)
        validation_dataset = reader.read(file_path=DATA_ROOT / val_fname)
        test_dataset = reader.read(file_path=DATA_ROOT / test_fname)
#         print("\n##################################\n", name, len(train_dataset), len(validation_dataset), len(test_dataset))

        # Train the model:
        if name == 'lstm':
            model, vocab, ep = train.train_lstm(train_dataset, validation_dataset, BATCH_SIZE,
                                                num_layers=1, bidirectional=False, use_elmo=use_elmo)
        elif name == 'bilstm':
            model, vocab, ep = train.train_lstm(train_dataset, validation_dataset, BATCH_SIZE,
                                                num_layers=1, bidirectional=True, use_elmo=use_elmo)
        elif name == 'stacked_bilstm':
            model, vocab, ep = train.train_lstm(train_dataset, validation_dataset, BATCH_SIZE,
                                                num_layers=2, bidirectional=True, use_elmo=use_elmo)
        elif name == 'cnn':
            model, vocab, ep = train.train_cnn(train_dataset, validation_dataset, BATCH_SIZE,
                                               num_filters=100, filter_sizes=(2,3,4,5), use_elmo=use_elmo)
        elif name == 'bert':
            model, vocab, ep = train.train_bert(train_dataset, validation_dataset, BATCH_SIZE,
                                                pretrained_model=BERT_MODEL)
        else:
            sys.exit("'name' not valid")
            
        n_epochs.append(ep) # keep track of number of actual training epochs for each fold
        
        # Predict and evaluate the model on test set:
        preds = evaluate.make_predictions(model, vocab, test_dataset, BATCH_SIZE) # Note that 'preds' is of the shape (number of samples, 2) - the columns represent the probabilities for the two classes ['generalization', 'neutral']
        f1, precision, recall, acc, auprc = evaluate.compute_metrics(preds, test_dataset)
        
        if save_predictions:
            id_pred = evaluate.map_id_prediction(preds, test_dataset)
            if set(id_pred.keys()).intersection(set(map_id_pred_probability.keys())) != set(): # sanity check
                sys.exit("Error: There is overlap in test set IDs across folds.")
            map_id_pred_probability.update(id_pred)
        
        if save_model: # save the model weights and vocabulary
            with open('./tmp/'+name+'_model'+'_fold_'+str(fold_number)+'.th', 'wb') as f:
                torch.save(model.state_dict(), f)
            vocab.save_to_files("./tmp/"+name+"_vocabulary_fold_"+str(fold_number))

        print("\nFold #{} | F1 = {}".format(fold_number, f1))
        f1s.append(f1); precision_s.append(precision); recall_s.append(recall); accuracies.append(acc); AUPRCs.append(auprc)

    mean_f1 = np.array(f1s).mean(); mean_precision = np.array(precision_s).mean(); mean_recall = np.array(recall_s).mean(); mean_accuracy = np.array(accuracies).mean(); mean_auprc = np.array(AUPRCs).mean()

    print("Total # predictions: {} | Saving Predictions = {}".format(len(map_id_pred_probability), save_predictions))
    
    return mean_f1, mean_precision, mean_recall, mean_accuracy, mean_auprc, map_id_pred_probability, n_epochs

BATCH_SIZE = 64
DATA_ROOT = Path("/home/ndg/users/sbagga1/generalization/data/")
LABEL_COLS = ["generalization", "neutral"] # Ordering is crucial
BERT_MODEL = "bert-large-uncased"
torch.manual_seed(42)