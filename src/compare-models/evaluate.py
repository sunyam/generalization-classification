"""
Author: Sunyam

This file contains functions to evaluate the model: multiple classification metrics & error analysis.
"""
import numpy as np
import torch
from typing import *
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, average_precision_score

from allennlp.data.iterators import DataIterator
from allennlp.data.iterators import BasicIterator
from allennlp.data import Instance
from allennlp.models import Model
from allennlp.nn import util as nn_util


class Predictor:
    def __init__(self, model: Model, iterator: DataIterator, cuda_device: int=-1) -> None:
        self.model = model
        self.iterator = iterator
        self.cuda_device = cuda_device
        
    def _extract_data(self, batch) -> np.ndarray:
        out_dict = self.model(**batch)
        return out_dict["class_probabilities"]
    
    def predict(self, dataset: Iterable[Instance]) -> np.ndarray:
        pred_generator = self.iterator(dataset, num_epochs=1, shuffle=False)
        self.model.eval()
        pred_generator_tqdm = tqdm(pred_generator,
                                   total=self.iterator.get_num_batches(dataset))
        preds = []
        with torch.no_grad():
            for batch in pred_generator_tqdm:
                batch = nn_util.move_to_device(batch, self.cuda_device)
                preds.append(self._extract_data(batch))
        return np.concatenate(preds, axis=0)

    
def onehot_to_label(dataset: Iterable[Instance]) -> List[str]:
    """
    Converts the input dataset (one-hot) to a list of strings: 'generalization' | 'neutral'
    Note: Assumes the ordering of label_cols is ['generalization', 'neutral']
    """
    y_true = []
    for instance in dataset:
        if list(instance.fields['label'].array) == [0, 1]:
            y_true.append("neutral")
        elif list(instance.fields['label'].array) == [1, 0]:
            y_true.append("generalization")
        else:
            print("Error:", instance)
            
    return y_true


def prob_to_label(probs):
    """
    Converts the predicted probablities to the corresponding 'hard' label.
    Note: Assumes the order of label_cols is ['generalization', 'neutral']
    """
    if probs[0] >= probs[1]:
        return "generalization"
    else:
        return "neutral"
    

def map_id_prediction(pred_probs, test_dataset):
    """
    For Error Analysis: maps the ID to the corresponding predicted Prob('generalization').
    
    Note: input is predicted probabilities.
    Returns a dictionary with key = ID | value = Prob('generalization')
    """
    out = {}
    for prediction, sample in zip(pred_probs, test_dataset):
        ID = sample.fields['ID'].metadata
        out[ID] = prediction[0] # because order is ['generalization', 'neutral']
    return out
    
    
def compute_metrics(preds, test_dataset):
    """
    Computes the classification metrics given the predictions and true labels.
    """
    y_preds = [prob_to_label(list(probabilities)) for probabilities in preds] # predicted labels
    y_true = onehot_to_label(test_dataset) # true labels
    prob_positive = [list(prob)[0] for prob in preds] # probabilities of the positive class

    # Compute classification metrics:
    f1 = f1_score(y_true, y_preds, pos_label='generalization')
    precision = precision_score(y_true, y_preds, pos_label='generalization')
    recall = recall_score(y_true, y_preds, pos_label='generalization')
    acc = accuracy_score(y_true, y_preds)
    auprc = average_precision_score(y_true, y_score=prob_positive, pos_label='generalization')
            
    return f1, precision, recall, acc, auprc
    
    
def make_predictions(model, vocab, test_dataset, batch_size, use_gpu=False):
    """
    Runs the given 'model' on the given 'test_dataset' to get predictions, and returns the predictions.
    """
    # iterate over the dataset without changing its order
    seq_iterator = BasicIterator(batch_size)
    seq_iterator.index_with(vocab)

    predictor = Predictor(model, seq_iterator, cuda_device=0 if use_gpu else -1)
    preds = predictor.predict(test_dataset)    
    return preds