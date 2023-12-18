import os

import pandas as pd
import numpy as np
import torch

from ._api_base import APIBase
from ..hybridization_probability._dataset import RNNDatasetInference ,pack_collate
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
        self.inverse_predictions = lambda predictions: np.exp(predictions*loaded_model["hyperparameters"]["dataset"]["std"] + loaded_model["hyperparameters"]["dataset"]["mean"])
    
    
    def predict(self, data: pd.DataFrame):
        #set batch size based on the hardware available
        batch_size = 32 if self.device == "cuda" else 1
        dataset = RNNDatasetInference(data)
        dataloader = data.DataLoader(dataset, batch_size=batch_size, collate_fn = pack_collate)
        predictions = torch.tensor([])
        for data in dataloader:
            with torch.no_grad():
                predictions = torch.cat((predictions, self.model(*data)))
                
        predictions = predictions.detach().cpu().numpy()
        predictions = self.inverse_predictions(predictions)
        return predictions