"""
Author: Sunyam

This file is for reading the generalization dataset as AllenNLP 'Instances'.
"""

from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import TextField, MetadataField, ArrayField #, LabelField
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data import Instance
from allennlp.data.tokenizers import Token

from typing import *
from overrides import overrides
import numpy as np
import pandas as pd


class GeneralizationDatasetReader(DatasetReader):
    """
    Read Generalization dataset.
    """
    def __init__(self, tokenizer: Callable[[str], List[str]]=lambda x: x.split(),
                 token_indexers: Dict[str, TokenIndexer]=None,
                 label_cols: List[str]=None) -> None:
        super().__init__(lazy=False)
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers
        self.label_cols = label_cols

    @overrides
    def text_to_instance(self, tokens: List[Token],
                         ID: str=None,
                         labels: np.ndarray=None) -> Instance:
        text = TextField(tokens, self.token_indexers)
        fields = {"tokens": text}

        id_field = MetadataField(ID)
        fields["ID"] = id_field

        label_field = ArrayField(array=labels)
        fields["label"] = label_field

        return Instance(fields)

    @overrides
    def _read(self, file_path: str) -> Iterator[Instance]:
        df = pd.read_csv(file_path, lineterminator='\n')
        for _, row in df.iterrows():
            yield self.text_to_instance(
                tokens=[Token(x) for x in self.tokenizer(row['sentences'])],
                ID=row['ID'], # unique ID for every row
                labels=row[self.label_cols].values
            )
