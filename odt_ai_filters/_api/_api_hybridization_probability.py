import os

import pandas as pd
import numpy as np

from ._api_base import APIBase
from .._hybridization_probability._dataset import RNNDatasetInference ,pack_collate
from .._hybridization_probability._models import OligoLSTM



class APIHybridizationProbability(APIBase):

    def __init__(self) -> None:
        super().__init__()

    def predict(self):
        return super().predict()
    


import json
import torch
from torch.utils import data

from _models._hybridization_probability import RNNDatasetInference, OligoLSTM, pack_collate
import os
import pandas as pd
import numpy as np

class AIFilter():

    def __init__(self, ai_filter: str, ai_filter_path: str = None) -> None:
        self.ai_filter = ai_filter
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # get repository location
        repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)) )
        if ai_filter == "hybridization_probaility":
            if ai_filter_path is None:
                ai_filter_path = os.path.join(repo_path,"data", "models", "hybridization_probability.pt")
            loaded_model = torch.load(ai_filter_path, map_location=self.device)
            # load the model with the right hyperparameters
            self.model = OligoLSTM(**loaded_model["hyperparameters"]["model"])
            # load the pretrained weights of the model
            self.model.load_state_dict(torch.load(loaded_model["weights"]))
            self.model.to(self.device)
            self.model.eval()
            # funtion to restore the predictions to the hybridization probablity
            self.inverse_predictions = lambda predictions: np.exp(predictions*loaded_model["hyperparameters"]["dataset"]["std"] + loaded_model["hyperparameters"]["dataset"]["mean"])

        else:
            raise ValueError(f"The ai_filter {ai_filter} is not supported.")
    
    
    def predict(self, data: pd.DataFrame):
        #set batch size based on the hardware available
        batch_size = 32 if self.device == "cuda" else 1
        if self.ai_filter == "hybridization_probaility":
            dataset = RNNDatasetInference(data)
            dataloader = data.DataLoader(dataset, batch_size=batch_size, collate_fn = pack_collate)
            predictions = torch.tensor([])
            for data in dataloader:
                with torch.no_grad():
                    predictions = torch.cat((predictions, self.model(*data)))
                
        predictions = predictions.detach().cpu().numpy()
        predictions = self.inverse_predictions(predictions)
        return predictions