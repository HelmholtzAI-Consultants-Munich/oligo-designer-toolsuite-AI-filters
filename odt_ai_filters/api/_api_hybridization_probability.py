import os
from typing import List

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

    def __init__(self, ai_filter_path) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
        if ai_filter_path is None:
            # if none the predefined models are used
            # get repository location
            repo_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            self.ai_filter_path = os.path.join(repo_path,"data", "pretrained_models", "hybridization_probability.pt")
        else:
            self.ai_filter_path = ai_filter_path
        loaded_model = torch.load(self.ai_filter_path, map_location=self.device)
        # load the model with the right hyperparameters
        self.model = OligoRNN(**loaded_model["hyperparameters"]["model"]) # model with best performances
        # load the pretrained weights of the model
        self.model.load_state_dict(loaded_model["weights"])
        self.model.to(self.device)
        self.model.eval()
        # funtion to restore the predictions to the hybridization probablity
        self.inverse_predictions = lambda predictions: 10**(predictions*loaded_model["hyperparameters"]["dataset"]["std"] + loaded_model["hyperparameters"]["dataset"]["mean"])
    
    
    def predict(self, queries: List[Seq.Seq], gapped_queries: List[Seq.Seq], targets: List[Seq.Seq], gapped_targets: List[Seq.Seq]):
        # generate the dataset
        dataset = self._generate_dataset(queries, gapped_queries, targets, gapped_targets)
        #set batch size based on the hardware available
        batch_size = 32 if self.device == "cuda" else 1 #TODO: find optimal value
        dataset = RNNDatasetInference(data)
        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn = pack_collate_inference)
        predictions = torch.tensor([])
        for data in dataloader:
            with torch.no_grad():
                predictions = torch.cat((predictions, self.model(*data)))
                
        predictions = predictions.detach().cpu().numpy()
        predictions = self.inverse_predictions(predictions)
        return predictions

    def _generate_dataset(
            self, queries: List[Seq.Seq], gapped_queries: List[Seq.Seq], targets: List[Seq.Seq], gapped_targets: List[Seq.Seq]
    ):
        """Create a database with the information of the oligos that match the blast search.

        :param queries: List with the sequences of the query oligos.
        :type queries: list
        :param gapped_queries: List with the sequences of the query oligos with gaps.
        :type gapped_queries: list
        :param targets: List with the sequences of the target oligos.
        :type targets: list
        :param gapped_targets: List with the sequences of the target oligos with gaps.
        :type gapped_targets: list
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
        dataset["off_target_sequence"] = gapped_targets
        dataset["off_target_length"] = [len(target) for target in targets]
        dataset["off_target_GC_content"] = [gc_fraction(target) for target in targets]
        dataset["number_mismatches"] = [
            sum(query != target for query, target in zip(query, target))
            for query, target in zip(gapped_queries, gapped_targets)
        ]
        return dataset
