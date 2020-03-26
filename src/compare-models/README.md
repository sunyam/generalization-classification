3-fold cross validation is used with stratified splits.

NOTE: By default, models use pre-trained GloVe embeddings. Functionality to use ELMo embeddings instead: ```--elmo```

The different models implemented in this repository are:
- Long Short Term Memory Networks (LSTM)
- Bi-directional LSTM
- Stacked Bi-directional LSTM
- Convolutional Neural Networks (CNN)
- Bidirectional Encoder Representations from Transformers (BERT)

#### Example runs:
* To run a Bi-directional LSTM using GloVe embeddings: 
   ```python main.py --model bilstm```

* To run a CNN using ELMo embeddings:
```python main.py --model cnn --elmo```

* To run a LSTM using ELMo embeddings, save the predictions as a pickle, and save the model:
```python main.py --model lstm --elmo --save_preds --save_model```

* To run BERT:
```python main.py --model bert```

If you want to play around with the number of epochs/learning rate/optimizer etc, you would need to edit *train.py*.

#### Results
The results are averaged over 3-fold cross validation.
See */results/results.tsv* and */src/Results* notebook to check out how different models did across a range of evaluation metrics:
*f1-score, Precision, Recall, Accuracy, Area Under Precision-Recall Curve*.
CNN with ELMo achieves the best f1-score of 0.77, whereas a standard machine learning model (not shown here) with hand-engineered features could only manage a f1-score of ~.65

