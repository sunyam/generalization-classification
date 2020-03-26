"""
Train the CNN (best model) on the entire training set of 3,456 instances and save the weights & vocabulary.
"""
import torch
from pathlib import Path
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from allennlp.data.token_indexers.elmo_indexer import ELMoTokenCharactersIndexer

import data_reader
import train

def tokenizer(x: str):
    """
    Tokenizer for DatasetReader.
    """
    return [w.text for w in SpacyWordSplitter(language='en_core_web_sm', pos_tags=False).split_words(x)]

def train_and_save():
    token_indexer = ELMoTokenCharactersIndexer()

    reader = data_reader.GeneralizationDatasetReader(tokenizer=tokenizer, token_indexers={"tokens": token_indexer},
                                                     label_cols=LABEL_COLS)

    train_fname = 'Gen_Sentences_Annotated_All_Final_Processed.csv'
    train_dataset = reader.read(file_path=DATA_ROOT / train_fname)

    model, vocab = train.train_cnn(train_dataset, BATCH_SIZE, num_filters=100, filter_sizes=(2,3,4,5))

    # save the model weights and vocabulary
    with open('../saved_model/cnn_elmo.th', 'wb') as f:
        torch.save(model.state_dict(), f)
    vocab.save_to_files("../saved_model/vocabulary")


if __name__ == '__main__':
    BATCH_SIZE = 64
    DATA_ROOT = Path("../data/")
    LABEL_COLS = ["generalization", "neutral"] # Ordering is important

    train_and_save()
