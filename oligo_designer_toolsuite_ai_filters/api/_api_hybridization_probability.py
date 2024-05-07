import os
from typing import List, Optional

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from Bio.SeqUtils import gc_fraction
from Bio import Seq

from ._api_base import APIBase
from ..hybridization_probability._dataset import RNNDatasetInference ,pack_collate_inference
from ..hybridization_probability._models import OligoLSTM, OligoRNN



class APIHybridizationProbability(APIBase):

    def __init__(self, ai_filter_path=None) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
        if ai_filter_path is None:
            # if none the predefined models are used
            # get repository location
            repo_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            self.ai_filter_path = os.path.join(repo_path, "pretrained_models", "hybridization_probability.pt")
        else:
            self.ai_filter_path = ai_filter_path
        loaded_model = torch.load(self.ai_filter_path, map_location=self.device)
        # load the model with the right hyperparameters
        self.model = OligoLSTM(**loaded_model["hyperparameters"]["model"]) # model with best performances
        # load the pretrained weights of the model
        self.model.load_state_dict(loaded_model["weights"])
        if torch.cuda.device_count() > 1:
            print("hi")
            self.model = torch.nn.DataParallel(self.model)
        self.model.to(self.device)
        self.model.eval()
        # funtion to restore the predictions to the hybridization probablity
        self.inverse_predictions = lambda predictions: 10**(predictions*loaded_model["hyperparameters"]["dataset"]["std"] + loaded_model["hyperparameters"]["dataset"]["mean"])
    
    
    def predict(self, queries: List[Seq.Seq], gapped_queries: List[Seq.Seq], references: List[Seq.Seq], gapped_references: List[Seq.Seq], batch_size: Optional[int] = None):
        # generate the dataset
        data = self._generate_dataset(queries, gapped_queries, references, gapped_references)
        #set batch size based on the hardware available
        if batch_size is None:
            batch_size = 1024 if self.device == "cuda" else 128 #TODO: find optimal value
        dataset = RNNDatasetInference(data)
        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn = pack_collate_inference)
        predictions = torch.tensor([])
        predictions = predictions.to(self.device)
        for sequences, features in dataloader:
            sequences =sequences.to(self.device)
            features = features.to(self.device)
            with torch.no_grad():
                predictions = torch.cat((predictions, self.model(sequences, features)))
                
        predictions = predictions.detach().cpu().numpy()
        predictions = self.inverse_predictions(predictions)
        return predictions

    def _generate_dataset(
            self, queries: List[Seq.Seq], gapped_queries: List[Seq.Seq], references: List[Seq.Seq], gapped_references: List[Seq.Seq]
    ):
        """Create a database with the information of the oligos that match the blast search.

        :param queries: List with the sequences of the query oligos.
        :type queries: list
        :param gapped_queries: List with the sequences of the query oligos with gaps.
        :type gapped_queries: list
        :param references: List with the sequences of the reference oligos.
        :type references: list
        :param gapped_references: List with the sequences of the reference oligos with gaps.
        :type gapped_references: list
        :return: dataset
        :rtype: pd.DataFrame
        """

        dataset = pd.DataFrame(
            columns=[
                "query_sequence",
                "query_length",
                "query_GC_content",
                "off_target_sequence",
                "off_target_length",
                "off_target_GC_content",
                "number_mismatches",
            ]
        )
        dataset["query_sequence"] = gapped_queries
        dataset["query_length"] = [len(query) for query in queries]
        dataset["query_GC_content"] = [gc_fraction(query) for query in queries]
        dataset["off_target_sequence"] = gapped_references
        dataset["off_target_length"] = [len(reference) for reference in references]
        dataset["off_target_GC_content"] = [gc_fraction(reference) for reference in references]
        dataset["number_mismatches"] = [
            sum(query != reference for query, reference in zip(query, reference))
            for query, reference in zip(gapped_queries, gapped_references)
        ]
        return dataset
