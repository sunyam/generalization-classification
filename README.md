# generalization-classification
#### Deep learning models for text classification using AllenNLP

This project tackles a binary text classification problem where we have 3,456 instances/sentences annotated for whether they encode a <i>generalization</i> or not. This repository contains the annotated data & code for various deep learning models with the option of using pre-trained GloVe embeddings or ELMo embeddings (see ```/src/compare-models/```).

The best performing model is a CNN using ELMo embeddings which is then trained on the entire training set of 3,456 instances (see .py files in ```/src/```). It is used to predict on a test set of 16,816 sentences. The test set is obtained by extracting sentences from the beginning and end of a set of 230 txt files (see ```/src/Predict-TestSet.ipynb```).

If the notebooks don't load on Github, you can always use Jupyter's <a href="https://nbviewer.jupyter.org/">nbviewer</a>.
All implementations are in Python 3.7 and utilise <i>AllenNLP</i> (0.8.4) and <i>PyTorch</i> (1.1.0) which are powerful deep learning frameworks. Thanks to Keita for this lovely <a href="http://mlexplained.com/2019/01/30/an-in-depth-tutorial-to-allennlp-from-basics-to-elmo-and-bert/">tutorial</a> on AllenNLP.

