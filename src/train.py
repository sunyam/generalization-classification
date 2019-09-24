"""
Author: Sunyam

This file contains methods to train different models: LSTM, BiLSTM, Stacked BiLSTM, CNN, ELMo, BERT.
"""
import models

import torch
import torch.nn as nn
import torch.optim as optim

from allennlp.data.vocabulary import Vocabulary
from allennlp.data.iterators import BucketIterator
from allennlp.modules.token_embedders.embedding import _read_embeddings_from_text_file # to load pre-trained embeddings
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder, TextFieldEmbedder
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder, PytorchSeq2VecWrapper, CnnEncoder
from allennlp.modules import FeedForward
from allennlp.modules.token_embedders import Embedding
from allennlp.training.trainer import Trainer
from allennlp.modules.token_embedders import ElmoTokenEmbedder # for ELMo
from allennlp.modules.token_embedders.bert_token_embedder import PretrainedBertEmbedder # for BERT
from overrides import overrides

def load_glove_embeddings(vocab):
    """
    Loads pre-trained GloVe embeddings.
    
    Returns
    -------
    TextFieldEmbedder
    """
    embedding_matrix = _read_embeddings_from_text_file(file_uri="https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.840B.300d.txt.gz", 
                                                       embedding_dim=300, 
                                                       vocab=vocab)
    print("Pre-trained GloVe loaded:", embedding_matrix.size())
    
    token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'), 
                                embedding_dim=300,
                                weight=embedding_matrix)
    word_embeddings: TextFieldEmbedder = BasicTextFieldEmbedder({"tokens": token_embedding})
    return word_embeddings


def load_elmo_embeddings(large=True):
    """
    Loads pre-trained ELMo embeddings ('large' model by default).
    
    Parameters
    ----------
    large: bool
        Set to True to load the Large ELMo model; False for small ELMo model
    
    Returns
    -------
    TextFieldEmbedder
    """
    if large: # use the Large pre-trained model
        print("Loading LARGE ELMo..")
        options_file='https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json' 
        weight_file='https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5'

    else: # use the Small pre-trained model
        print("Loading SMALL ELMo..")
        options_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json'
        weight_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5'
    
    elmo_embedder = ElmoTokenEmbedder(options_file, weight_file)
    word_embeddings = BasicTextFieldEmbedder({"tokens": elmo_embedder})
    print("Pre-trained ELMo loaded..")
    return word_embeddings


def load_bert_embeddings(pretrained_model):
    """
    Loads pre-trained BERT embedder.
    
    Parameters
    ----------
    pretrained_model: str
        BERT model to load
        
    Returns
    -------
    TextFieldEmbedder
    """
    bert_embedder = PretrainedBertEmbedder(pretrained_model=pretrained_model, top_layer_only=True)

    word_embeddings: TextFieldEmbedder = BasicTextFieldEmbedder({"tokens": bert_embedder},
                                                                allow_unmatched_keys=True) # ignoring masks 
    global BERT_DIM
    BERT_DIM = word_embeddings.get_output_dim()
    
    print("{} loaded | Dimension = {}".format(pretrained_model, BERT_DIM))
    return word_embeddings


def train_lstm(train_dataset, validation_dataset, batch_size, num_layers, use_elmo=False, epochs=15, patience=None,
               bidirectional=True, learning_rate=3e-4, hidden_size=64, num_classes=2, use_gpu=False):
    """
    Trains a LSTM and its variants (Vanilla, Bi-Directional, Stacked BiLSTM) on train_dataset; optionally, perform early stopping based on validation loss. Initialises word embeddings with pre-trained GloVe (OR) uses pre-trained ELMo model to dynamically compute embeddings.
    
    Parameters
    ----------
    train_dataset: List[Instance]
        Instances for training set
    validation_dataset: List[Instance]
        Instances for validation set
    batch_size: int
        number of Instances to process in a batch
    num_layers: int
        number of BiLSTM layers: 2 or higher for Stacked BiLSTMs
    use_elmo: bool
        use ELMo embeddings if True | GloVe embeddings if False
    epochs: int
        total number of epochs to train for (default=15)
    patience: int or None
        early stopping - number of epochs to wait for validation loss to improve; 'None' to disable early stopping
    bidirectional: bool
        True for a bidirectional LSTM
    learning_rate: float
        learning rate for Adam Optimizer
    hidden_size: int
        size of the hidden layer in the encoder
    num_classes: int
        default=2 for binary classification
    use_gpu: bool
        True to use the GPU
    
    Returns
    -------
    Trained Model, Vocabulary, Number of actual training epochs
    """
    if use_elmo:
        vocab = Vocabulary()
        word_embeddings: TextFieldEmbedder = load_elmo_embeddings()
    else:
        vocab = Vocabulary.from_instances(train_dataset + validation_dataset)
        word_embeddings: TextFieldEmbedder = load_glove_embeddings(vocab)
    
    iterator = BucketIterator(batch_size=batch_size,
                              sorting_keys=[("tokens", "num_tokens")])
    
    iterator.index_with(vocab) # numericalize the data
    
    # BiLSTM encoder
    encoder: Seq2VecEncoder = PytorchSeq2VecWrapper(nn.LSTM(word_embeddings.get_output_dim(),
                                                            hidden_size,
                                                            num_layers=num_layers,
                                                            bidirectional=bidirectional,
                                                            batch_first=True))

    # Feedforward:
    classifier_feedforward: FeedForward = nn.Linear(encoder.get_output_dim(), num_classes)

    model = models.Classifier(vocab=vocab,
                              word_embeddings=word_embeddings,
                              encoder=encoder,
                              classifier_feedforward=classifier_feedforward)
    
    if use_gpu: model.cuda()
    else: model
    
    optimizer = optim.Adam(model.parameters(), learning_rate)

    if patience == None: # No early stopping: train on both train+validation dataset
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            iterator=iterator,
            train_dataset=train_dataset + validation_dataset,
            cuda_device=0 if use_gpu else -1,
            num_epochs=epochs)

    else:
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            iterator=iterator,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            cuda_device=0 if use_gpu else -1,
            patience=patience, # stop if loss does not improve for 'patience' epochs
            num_epochs=epochs)

    metrics = trainer.train()
#     print(metrics)
    
    return model, vocab, metrics['training_epochs']


def train_cnn(train_dataset, validation_dataset, batch_size, num_filters, filter_sizes, use_elmo=False, epochs=15, 
              patience=None, learning_rate=3e-4, num_classes=2, use_gpu=False):
    """
    Trains CNN on train_dataset; optionally, perform early stopping based on validation loss. Initialises word embeddings with pre-trained GloVe (OR) uses pre-trained ELMo model to dynamically compute embeddings.
    The CNN has one convolution layer for each ngram filter size.
    
    Parameters
    ----------
    train_dataset: List[Instance]
        Instances for training set
    validation_dataset: List[Instance]
        Instances for validation set
    batch_size: int
        number of Instances to process in a batch
    num_filters: int
        output dim for each convolutional layer, which is the number of 'filters' learned by that layer
    filter_sizes: Tuple[int]
        specifies the number of convolutional layers and their sizes
    use_elmo: bool
        use ELMo embeddings if True | GloVe embeddings if False
    epochs: int
        total number of epochs to train on (default=15)
    patience: int or None
        early stopping - number of epochs to wait for validation loss to improve; 'None' to disable early stopping
    learning_rate: float
        learning rate for Adam Optimizer
    num_classes: int
        default=2 for binary classification
    use_gpu: bool
        True to use the GPU
    
    Returns
    -------
    Trained Model, Vocabulary, Number of actual training epochs
    """
    if use_elmo:
        vocab = Vocabulary()
        word_embeddings: TextFieldEmbedder = load_elmo_embeddings()
    else:
        vocab = Vocabulary.from_instances(train_dataset + validation_dataset)
        word_embeddings: TextFieldEmbedder = load_glove_embeddings(vocab)
    
    iterator = BucketIterator(batch_size=batch_size,
                              sorting_keys=[("tokens", "num_tokens")])
    
    iterator.index_with(vocab) # numericalize the data
            
    # CNN encoder
    encoder: Seq2VecEncoder = CnnEncoder(embedding_dim=word_embeddings.get_output_dim(),
                                         num_filters=num_filters, 
                                         ngram_filter_sizes=filter_sizes)

    # Feedforward:
    classifier_feedforward: FeedForward = nn.Linear(encoder.get_output_dim(), num_classes)

    model = models.Classifier(vocab=vocab,
                              word_embeddings=word_embeddings,
                              encoder=encoder,
                              classifier_feedforward=classifier_feedforward)
    
    if use_gpu: model.cuda()
    else: model
    
    optimizer = optim.Adam(model.parameters(), learning_rate)

    if patience == None: # Train on both train+validation dataset if patience is None
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            iterator=iterator,
            train_dataset=train_dataset + validation_dataset,
            cuda_device=0 if use_gpu else -1,
            num_epochs=epochs)

    else:
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            iterator=iterator,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            cuda_device=0 if use_gpu else -1,
            patience=patience, # stop if loss does not improve for 'patience' epochs
            num_epochs=epochs)

    metrics = trainer.train()
#     print(metrics)
    
    return model, vocab, metrics['training_epochs']


def train_bert(train_dataset, validation_dataset, batch_size, pretrained_model, epochs=15, patience=None, 
               learning_rate=3e-4, num_classes=2, use_gpu=False):
    """
    Trains BERT on train_dataset; with optional early stopping on validation_dataset.
    
    Parameters
    ----------
    train_dataset: List[Instance]
        Instances for training set
    validation_dataset: List[Instance]
        Instances for validation set
    batch_size: int
        number of Instances to process in a batch
    pretrained_model: str
        pretrained BERT model to use
    epochs: int
        total number of epochs to train on (default=15)
    patience: int or None
        early stopping - number of epochs to wait for validation loss to improve; 'None' to disable early stopping 
    learning_rate: float
        learning rate for Adam Optimizer
    num_classes: int
        default=2 for binary classification
    use_gpu: bool
        True to use the GPU
    
    Returns
    -------
    Trained Model, Vocabulary, Number of actual training epochs
    """
    vocab = Vocabulary()
    
    iterator = BucketIterator(batch_size=batch_size,
                              sorting_keys=[("tokens", "num_tokens")])

    iterator.index_with(vocab) # numericalize the data

    word_embeddings: TextFieldEmbedder = load_bert_embeddings(pretrained_model)
    encoder: Seq2VecEncoder = BertSentencePooler(vocab)

    # Feedforward:
    classifier_feedforward: FeedForward = nn.Linear(encoder.get_output_dim(), num_classes)

    model = models.Classifier(vocab=vocab,
                              word_embeddings=word_embeddings,
                              encoder=encoder,
                              classifier_feedforward=classifier_feedforward)

    if use_gpu: model.cuda()
    else: model
    
    optimizer = optim.Adam(model.parameters(), learning_rate)

    if patience == None: # No early stopping: train on both train+validation dataset
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            iterator=iterator,
            train_dataset=train_dataset + validation_dataset,
            cuda_device=0 if use_gpu else -1,
            num_epochs=epochs)

    else:
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            iterator=iterator,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            cuda_device=0 if use_gpu else -1,
            patience=patience, # stop if loss does not improve for 'patience' epochs
            num_epochs=epochs)

    metrics = trainer.train()
#     print(metrics)
    
    return model, vocab, metrics['training_epochs']


class BertSentencePooler(Seq2VecEncoder):
    """
    Encoder for BERT model; forward() only extracts the embeddings corresponding to the first token in the sentence.
    """
    def forward(self, embs: torch.tensor, 
                mask: torch.tensor=None) -> torch.tensor:
        # extract first token tensor
        return embs[:, 0]
    
    @overrides
    def get_output_dim(self) -> int:
        return BERT_DIM