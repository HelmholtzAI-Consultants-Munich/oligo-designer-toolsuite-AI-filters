# Model and database class for the NUpack emulation 

from typing import *

import torch
from torch .utils import data
from torch.nn.utils import rnn
import pandas as pd


class MLPDataset(data.Dataset):


    def __init__(self, path: str) -> None:
        super().__init__()
        database = pd.read_csv(path, index_col=0) # dictionary with sequences, features and labels tensors
        oligo_length = database["oligo_length"][0]
        encodings = torch.empty((len(database), 8*oligo_length))
        for i, row in database.iterrows():
            oligo = row["oligo_sequence"]
            off_target = row["off_target_sequence"]
            encodings[i,:] = torch.cat([self.encode_sequence(oligo), self.encode_sequence(off_target)])
        self.labels = torch.tensor(database["duplexing_log_score"], dtype=torch.double)
        features = torch.tensor(database[["oligo_length", "oligo_GC_content", "off_target_length", "off_target_GC_content", "number_mismatches"]].to_numpy(), dtype=torch.double)
        self.data = torch.cat([encodings, features], dim=1)


    def __len__(self):
        return self.data.shape[0]
    

    def __getitem__(self, index) -> Tuple[torch.tensor, torch.tensor]:
        return self.data[index,:], self.labels[index]
    

    def encode_sequence(self, sequence: str):
        sequence_encoding = torch.empty((4*len(sequence)))
        for i, nt in enumerate(sequence):
            #encoding order is A, T, C. G
            if nt == 'A' or nt == 'a':
                nt_encoding = torch.tensor([1,0,0,0])
            elif nt == 'C' or nt == 'c':
                nt_encoding = torch.tensor([0,1,0,0])
            elif nt == 'G' or nt == 'g':
                nt_encoding = torch.tensor([0,0,1,0])
            elif nt == 'T' or nt == 't':
                nt_encoding = torch.tensor([0,0,0,1])
            elif nt == '-':
                nt_encoding = torch.tensor([0,0,0,0])
            else:
                Warning(f"Nucleotide {nt} not recognized.")
            sequence_encoding[4*i:4*(i+1)] = nt_encoding
        return sequence_encoding.double()
    

    def normalize_labels(self):
        # the normalization parametes are inferred from the data
        self.mean = self.labels.mean()
        self.std = self.labels.std()
        self.labels = (self.labels - self.mean)/self.std # normalize


    def normalize_lables_with_params(self, mean: float, std: float) ->None:
        # the normalization parametes are passed as arguments
        self.labels = (self.labels - mean)/std # normalize



class RNNDatasetInference(data.Dataset):
    """Dataset class for the hybridization probability prediction at inference time. This class is designed to work with
    Recurrent Neural Networks (RNNs) models or Long Short term Memory (LSMT) models. The seqeunces are 
    represented using an 1-Hot encoding: 'A = [1,0,0,0], 'C' = [0,1,0,0], ...

    :param path: Path to the .csv file containing the database
    :type path: str
    """

    def __init__(self, dataset: pd.DataFrame) -> None:
        """Dataset class for the hybridization probability prediction at inference time. This class is designed to work with
        Recurrent Neural Networks (RNNs) models or Long Short term Memory (LSMT) models. The seqeunces are 
        represented using an 1-Hot encoding: 'A = [1,0,0,0], 'C' = [0,1,0,0], ...

        :param dataset: datasetructure containing the dataset
        :type path: pd.DataFrame
        """
        super().__init__()
        self.sequences = [] # list of tensors
        for i, row in dataset.iterrows():
            query = row["query_sequence"]
            target = row["off_target_sequence"]
            assert len(query) == len(target), f"In line {i} the query and taget lengths don't match.query"
            encoding = torch.empty((len(query), 8), dtype=torch.double) #update for deletions and insertions
            for j in range(len(query)):
                encoding[j,:] = torch.tensor(self.encode_nt(query[j]) + self.encode_nt(target[j]), dtype=torch.double)
            self.sequences.append(encoding)
        self.features = torch.tensor(dataset[["query_length", "query_GC_content", "off_target_length", "off_target_GC_content", "number_mismatches"]].to_numpy(), dtype=torch.double)
        

    def __len__(self):
        return len(self.sequences)
    

    def __getitem__(self, index) -> Tuple[torch.tensor, torch.tensor, torch.torch.tensor]:
        # sequences is a dictionary of tensors since the length of each sequence can change
        return self.sequences[index], self.features[index,:] 
    

    def encode_nt(self, nt:str) -> list[str]:
        """One-Hot encodings of the nucletides: 'A' = [1,0,0,0], 'C' = [0,1,0,0], 'G' = [0,0,1,0], 'T' = [0,0,0,1], 'N' = [0,0,0,0]

        :param nt: Nucleotide to encode, {'A', 'C', 'G', 'T', 'N'}.
        :type nt: str
        :return: Ecoding vetor.
        :rtype: list[int]
        """
        #encoding order is A, C, T, G
        if nt == 'A' or nt == 'a':
            nt_encoding = [1,0,0,0]
        elif nt == 'C' or nt == 'c':
            nt_encoding = [0,1,0,0]
        elif nt == 'T' or nt == 't':
            nt_encoding = [0,0,1,0]
        elif nt == 'G' or nt == 'g':
            nt_encoding = [0,0,0,1]
        elif nt == '-':
            nt_encoding = [0,0,0,0]
        else:
            Warning(f"Nucleotide {nt} not recognized.")
        return nt_encoding
    

    def normalize_labels(self):
        """Standard normalization the data lables infering the normalization parameters form the data directly.
        """
        # the normalization parametes are inferred from the data
        self.mean = self.labels.mean()
        self.std = self.labels.std()
        self.labels = (self.labels - self.mean)/self.std # normalize

    def normalize_lables_with_params(self, mean: float, std: float) ->None:
        """Standard normalization of tha data labels with given standardization parameters.

        :param mean: Mean of the labels distribution.
        :type mean: float
        :param std: Standard deviation of the labels distribution.
        :type std: float
        """
        # the normalization parametes are passed as arguments
        self.labels = (self.labels - mean)/std # normalize

class RNNDataset(RNNDatasetInference):
    """Dataset class for the hybridization probability prediction. This class is designed to work with
        Recurrent Neural Networks (RNNs) models or Long Short term Memory (LSMT) models. The seqeunces are 
        represented using an 1-Hot encoding: 'A = [1,0,0,0], 'C' = [0,1,0,0], ...

        :param path: Path to the .csv file containing the database
        :type path: str
        """
    
    def __init__(self, path: str) -> None:
        """Dataset class for the hybridization probability prediction. This class is designed to work with
        Recurrent Neural Networks (RNNs) models or Long Short term Memory (LSMT) models. The seqeunces are 
        represented using an 1-Hot encoding: 'A = [1,0,0,0], 'C' = [0,1,0,0], ...

        :param path: Path to the .csv file containing the database
        :type path: str
        """

        dataset = pd.read_csv(path, index_col=0) # dictionary with sequences, features and labels tensors
        super().__init__(dataset=dataset)
        self.labels = torch.tensor(dataset["duplexing_log_score"], dtype=torch.double)


    def __getitem__(self, index) -> Tuple[torch.tensor, torch.tensor, torch.torch.tensor]:
        # sequences is a dictionary of tensors since the length of each sequence can change
        return self.sequences[index], self.features[index,:], self.labels[index] 
    


def pack_collate(batch: list) -> Tuple[rnn.PackedSequence, torch.Tensor, torch.Tensor]:
    """Collate function for the ``Dataloader`` class that saves the sequences in ``PackedSeqeunces`` classes.
    This allows to process the data in batches without the need of padding.

    :param batch: Output of the batch sampler.
    :type batch: list
    :return: Packed inpud data
    :rtype: Tuple[rnn.PackedSequence, torch.Tensor, torch.Tensor]
    """
    # TODO: adjust with inference time
    tabular = []
    sequences = []
    for sequence, features, labels in batch:
        tabular.append((features, labels))
        sequences.append(sequence)
    features, labels = data._utils.collate.default_collate(tabular)
    # sort the sequences in decreasing length order
    lengths = torch.tensor(list(map(len, sequences)))
    lengths, perm_idx = lengths.sort(0, descending=True)
    sorted_sequences = [sequences[i] for i in perm_idx]
    padded_sequences = rnn.pad_sequence(sequences=sorted_sequences, batch_first=True)
    return rnn.pack_padded_sequence(input=padded_sequences, lengths=lengths, batch_first=True), features[perm_idx,:], labels[perm_idx] # reorder according to the original ordering


def pack_collate_inference(batch: list) -> Tuple[rnn.PackedSequence, torch.Tensor, torch.Tensor]:
    """Collate function for the ``Dataloader`` class that saves the sequences in ``PackedSeqeunces`` classes.
    This allows to process the data in batches without the need of padding. This function is used at inference time, 
    where the dataset does not contain the label information.

    :param batch: Output of the batch sampler.
    :type batch: list
    :return: Packed inpud data
    :rtype: Tuple[rnn.PackedSequence, torch.Tensor, torch.Tensor]
    """
    # TODO: adjust with inference time
    tabular = []
    sequences = []
    for sequence, features in batch:
        tabular.append((features))
        sequences.append(sequence)
    features = data._utils.collate.default_collate(tabular)
    # sort the sequences in decreasing length order
    lengths = torch.tensor(list(map(len, sequences)))
    lengths, perm_idx = lengths.sort(0, descending=True)
    sorted_sequences = [sequences[i] for i in perm_idx]
    padded_sequences = rnn.pad_sequence(sequences=sorted_sequences, batch_first=True)
    return rnn.pack_padded_sequence(input=padded_sequences, lengths=lengths, batch_first=True), features[perm_idx,:]# reorder according to the original ordering