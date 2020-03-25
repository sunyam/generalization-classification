"""
Author: Sunyam

This file is to run different deep learning models: LSTM, BiLSTM, Stacked BiLSTM, CNN end-to-end using ELMo or GloVe embeddings, and save results in a TSV; can also run BERT.
"""
import os
import sys
import argparse
import pickle
from run import run_model

parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Name of the model to be run', required=True)
parser.add_argument('--elmo', help='Use ELMo embeddings', action="store_true") # uses GloVe embeddings otherwise
parser.add_argument('--save_preds', help='Save predictions for Error Analysis', action="store_true")
parser.add_argument('--save_model', help='Save model weights & vocabulary', action="store_true") 
args = parser.parse_args()
    
results_path = '/home/ndg/users/sbagga1/generalization/results/'+args.model+'_ELMo'+str(args.elmo)+'.tsv'
if os.path.exists(results_path):
    sys.exit("Results file already exists: " + results_path)
print("Model = {} | ELMo = {} | SavePreds = {} | SaveModel = {} | Output Filename = {}".format(args.model, args.elmo, args.save_preds, args.save_model, results_path))

f1, precision, recall, accuracy, auprc, preds, n_epochs = run_model(name=args.model,
                                                                    use_elmo=args.elmo,
                                                                    save_predictions=args.save_preds,
                                                                    save_model=args.save_model)

# Write to TSV:
results_file = open(results_path, "w")
results_file.write("Model\tF1-score\tPrecision\tRecall\tAccuracy\tAUPRC\n")
results_file.write(args.model+'_ELMo'+str(args.elmo)+'_'+str(n_epochs)+'\t'+str(f1)+'\t'+str(precision)+'\t'+str(recall)+'\t'+str(accuracy)+'\t'+str(auprc)+'\n')        

if args.save_preds:
    preds_path = '/home/ndg/users/sbagga1/generalization/predictions/'
    with open(preds_path+args.model+'_ELMo'+str(args.elmo)+'.pickle', 'wb') as f:
        pickle.dump(preds, f)