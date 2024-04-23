import argparse
from typing import Any
import yaml
import sys
import os
import json
from tqdm import tqdm
import time
from datetime import datetime
import logging
import copy
sys.path.append(os.getcwd())

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch import optim
import optuna
from _dataset import *
from _models import *
import wandb


os.environ["WANDB_API_KEY"] = "fca16a3a714741f111075c244a99b8675fa5cfe9"
os.environ["WANDB_MODE"] = "offline"





class Objective:

    def __init__(self, config, train_dataset, validation_dataset, logging: logging.Logger, model_dir: str) -> None:
        self.config = config
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.logging = logging
        self.model_dir = model_dir

    def __call__(self, trail: optuna.Trial) -> Any:

        self.logging.info(f"Start trail number {trail.number}.")
        device = torch.device("cuda") if torch.cuda.is_available() is True else torch.device("cpu")
        self.logging.info(f'Using device: {device}')

        ################
        # define model #
        ################

        hyperparameters = {}
        hyperparameters["dataset"] = {"mean": self.train_dataset.mean.item(), "std": self.train_dataset.std.item()}
        hyperparameters["model"] = {}
        if self.config["model"] == "mlp":
            hyperparameters["model"]["input_size"] = self.config["input_size"]
            hyperparameters["model"]["hidden_size"] = trail.suggest_int("hidden_size", low=self.config["hidden_size"][0], high=self.config["hidden_size"][1])
            hyperparameters["model"]["n_layers"] = trail.suggest_int("n_layers", low=self.config["n_layers"][0], high=self.config["n_layers"][1])
            hyperparameters["model"]["act_function"] = trail.suggest_categorical("act_function", choices=self.config["act_function"])
            hyperparameters["model"]["dropout"] = trail.suggest_float("dropout", low=self.config["dropout"][0], high=self.config["dropout"][1])
            model = OligoMLP(**hyperparameters["model"])
            collate_fn = None
        elif self.config["model"] == "rnn":
            hyperparameters["model"]["input_size"] = self.config["input_size"]
            hyperparameters["model"]["features_size"] = self.config["features_size"]            
            hyperparameters["model"]["hidden_size"] = trail.suggest_int("hidden_size", low=self.config["hidden_size"][0], high=self.config["hidden_size"][1])
            hyperparameters["model"]["hidden_size_ecoder"] = trail.suggest_int("hidden_size_ecoder", low=self.config["hidden_size_ecoder"][0], high=self.config["hidden_size_ecoder"][1])
            hyperparameters["model"]["n_layers"] = trail.suggest_int("n_layers", low=self.config["n_layers"][0], high=self.config["n_layers"][1])
            hyperparameters["model"]["n_layers_mlp"] = trail.suggest_int("n_layers_mlp", low=self.config["n_layers_mlp"][0], high=self.config["n_layers_mlp"][1])
            hyperparameters["model"]["n_layers_encoder"] = trail.suggest_int("n_layers_encoder", low=self.config["n_layers_encoder"][0], high=self.config["n_layers_encoder"][1])
            hyperparameters["model"]["act_function"] = trail.suggest_categorical("act_function", choices=self.config["act_function"])
            hyperparameters["model"]["nonlinearity"] = trail.suggest_categorical("nonlinearity", choices=self.config["nonlinearity"])
            hyperparameters["model"]["pool"] = trail.suggest_categorical("pool", choices=self.config["pool"])            
            hyperparameters["model"]["dropout"] = trail.suggest_float("dropout", low=self.config["dropout"][0], high=self.config["dropout"][1])
            hyperparameters["model"]["bidirectional"] = trail.suggest_categorical("bidirectional", choices=self.config["bidirectional"])
            model = OligoRNN(**hyperparameters["model"])
            collate_fn = pack_collate
        elif self.config["model"] == "lstm":
            hyperparameters["model"]["input_size"] = self.config["input_size"]
            hyperparameters["model"]["features_size"] = self.config["features_size"]
            hyperparameters["model"]["hidden_size"] = trail.suggest_int("hidden_size", low=self.config["hidden_size"][0], high=self.config["hidden_size"][1])
            hyperparameters["model"]["hidden_size_ecoder"] = trail.suggest_int("hidden_size_ecoder", low=self.config["hidden_size_ecoder"][0], high=self.config["hidden_size_ecoder"][1])
            hyperparameters["model"]["n_layers"] = trail.suggest_int("n_layers", low=self.config["n_layers"][0], high=self.config["n_layers"][1])
            hyperparameters["model"]["n_layers_mlp"] = trail.suggest_int("n_layers_mlp", low=self.config["n_layers_mlp"][0], high=self.config["n_layers_mlp"][1])
            hyperparameters["model"]["n_layers_encoder"] = trail.suggest_int("n_layers_encoder", low=self.config["n_layers_encoder"][0], high=self.config["n_layers_encoder"][1])
            hyperparameters["model"]["act_function"] = trail.suggest_categorical("act_function", choices=self.config["act_function"])
            hyperparameters["model"]["pool"] = trail.suggest_categorical("pool", choices=self.config["pool"])            
            hyperparameters["model"]["dropout"] = trail.suggest_float("dropout", low=self.config["dropout"][0], high=self.config["dropout"][1])
            hyperparameters["model"]["bidirectional"] = trail.suggest_categorical("bidirectional", choices=self.config["bidirectional"])
            model = OligoLSTM(**hyperparameters["model"])
            collate_fn = pack_collate
        else:
            raise ValueError(f"{self.config['model']} is not supported")
        model.to(device) # load the model on the cpu
        # parallleize over multiple gpus if they are available
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        
        #####################
        # define dataloader #
        #####################

        batch_size = trail.suggest_int("batch_size", low=self.config["batch_size"][0], high=self.config["batch_size"][1])
        train_loader = DataLoader(dataset=self.train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        validation_loader = DataLoader(dataset=self.validation_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

        ####################
        # define optimizer #
        ####################

        lr = trail.suggest_float("lr", low=self.config["lr"][0], high=self.config["lr"][1], log=True)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, factor=self.config["scheduler_factor"], patience=self.config["scheduler_patience"])
        loss = nn.MSELoss()

        ###################
        # train the model #
        ###################
        os.environ["WANDB_SILENT"] = "true"
        wandb.init(project=f"{self.config['model']}_{os.path.basename(self.config['train_dataset_path']).split('.')[0]}", config={**hyperparameters["model"], **hyperparameters["dataset"], "lr": lr, "batch_size": batch_size}, name=str(trail.number))
        # wandb.define_metric("train_loss", summary="min")
        # wandb.define_metric("validation_loss", summary="min")
        max_patience = self.config["patience"] # for early sotpping
        best_validation_loss = None
        best_model = model.state_dict()
        patience = 0
        start = time.time()
        for i in range(self.config["n_epochs"]):
            train_loss = self.train_epoch(model=model, dataloader=train_loader, loss=loss, optimizer=optimizer, device=device)
            validation_loss = self.eval_epoch(model=model, dataloader=validation_loader, loss=loss, device=device)
            # self.logging.info(f"Epoch: {i}, \t Train loss: {train_loss}, \t Validation loss: {validation_loss}")
            wandb.log({"train_loss": train_loss, "validation_loss": validation_loss})
            scheduler.step(validation_loss)
            if best_validation_loss is None or validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                # patience reset 
                best_model = copy.deepcopy(model.state_dict())
                patience = 0
            else:
                # patience update
                patience += 1
            if patience >= max_patience:
                break
        logging.info(f"Computation time: {time.time() - start}.")
        wandb.summary["validation_loss"] = best_validation_loss
        wandb.finish()

        ###################
        # store the model #
        ###################

        model_file = f"{self.config['model']}_{trail.number}.pt"
        # save the model weights and hyperparameters in the same dictionary
        torch.save({"weights": best_model, "hyperparameters": hyperparameters}, os.path.join(self.model_dir, model_file)) # store the best model and the hyperparameters
        return best_validation_loss


    def train_epoch(self, model: nn.Module, dataloader: DataLoader, loss: nn.Module, optimizer: optim.Optimizer, device: torch.device) -> float:
        model.train()
        cumulative_loss = torch.zeros(1,).to(device)
        for batch in dataloader:
            batch_device = []
            for t in batch:
                batch_device.append(t.to(device))
            data = batch_device[:-1]
            label = batch_device[-1]
            pred = model(*data)
            batch_loss = loss(pred, label)
            cumulative_loss += batch_loss
            batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        loss = cumulative_loss/len(dataloader)
        return loss.item()


    def eval_epoch(self,model: nn.Module, dataloader: DataLoader, loss: nn.Module, device: torch.device) -> float:
        model.eval()
        cumulative_loss = torch.zeros(1,).to(device)
        with torch.no_grad():
            for batch in dataloader:
                batch_device = []
                for t in batch:
                    batch_device.append(t.to(device))
                data = batch_device[:-1]
                label = batch_device[-1]
                pred = model(*data)
                cumulative_loss += loss(pred, label)
        loss = cumulative_loss/len(dataloader)
        return loss.item()


def main():

    #########################
    # read input arguments #
    #########################

    parser = argparse.ArgumentParser(
        prog="Artificial Dataset",
        usage="generate_artificial_dataset [options]",
        description=main.__doc__,
    )
    parser.add_argument("-c", "--config", help="path to the configuration file", default="config/generate_artificial_dataset.yaml")
    args = parser.parse_args()
    with open(args.config, "r") as handle:
        config = yaml.safe_load(handle)

    ##############
    # set logger #
    ##############

    timestamp = datetime.now()
    file_logger = f"log_train_{config['model']}_{timestamp.year}-{timestamp.month}-{timestamp.day}-{timestamp.hour}-{timestamp.minute}.txt"
    logging.getLogger(f"train_{config['model']}")
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=logging.NOTSET,
        handlers=[logging.FileHandler(file_logger), logging.StreamHandler()],
    )
    logging.captureWarnings(True)

    ##################
    # define dataset #
    ##################

    # we define the dataset outside the trial function to avoid processig the data multiple times
    if config["model"] == "mlp":
        train_dataset = MLPDataset(path=config["train_dataset_path"])
        validation_dataset = MLPDataset(path=config["validation_dataset_path"])
    elif config["model"] == "rnn" or config["model"] == "lstm":
        train_dataset = RNNDataset(path=config["train_dataset_path"])
        validation_dataset = RNNDataset(path=config["validation_dataset_path"])
    else: 
        raise ValueError(f"{config['model']} is not supported")
    # normalize labels
    train_dataset.normalize_labels()
    validation_dataset.normalize_lables_with_params(train_dataset.mean, train_dataset.std)

    logging.info(f"Datasets generated with {len(train_dataset)} training instances and {len(validation_dataset)} validation instances.")

    #####################
    # initialize optuna #
    #####################

    optuna.logging.enable_propagation()  # Propagate logs to the root logger.
    optuna.logging.disable_default_handler()  # Stop showing logs in sys.stderr.
    study = optuna.create_study()
    logging.info("Study created.")
    # create model directory
    model_dir = os.path.join(config["models_path"], config["model"], os.path.basename(config['train_dataset_path']).split('.')[0])
    os.makedirs(model_dir, exist_ok=True)
    study.optimize(func=Objective(config=config, train_dataset=train_dataset, validation_dataset=validation_dataset, logging=logging, model_dir=model_dir), n_trials=config["n_trials"])

    # test on the vest model on the test dataset (MSE on the real coordinates)
    best_study = study.best_trial.number
    best_model_path = os.path.join(model_dir, f"{config['model']}_{best_study}.pt")
    device = torch.device("cuda") if torch.cuda.is_available() is True else torch.device("cpu")
    best_model_file = torch.load(best_model_path, map_location=device)
    if config["model"] == "mlp":
        model = OligoMLP(**best_model_file["hyperparameters"]["model"])
        test_dataset = MLPDataset(path=config["test_dataset_path"])
        collate_fn = None
    elif config["model"] == "rnn":
        model = OligoRNN(**best_model_file["hyperparameters"]["model"])
        test_dataset = RNNDataset(path=config["test_dataset_path"])
        collate_fn = pack_collate
    elif config["model"] == "lstm":
        model = OligoLSTM(**best_model_file["hyperparameters"]["model"])
        test_dataset = RNNDataset(path=config["test_dataset_path"])
        collate_fn = pack_collate
    model.load_state_dict(best_model_file["weights"])
    model.to(device=device)
    model.eval()
    test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    loss = nn.MSELoss()

    #evaluate the best model
    cumulative_loss = torch.zeros(1,).to(device)
    with torch.no_grad():
        for batch in test_loader:
            batch_device = []
            for t in batch:
                batch_device.append(t.to(device))
            data = batch_device[:-1]
            label = batch_device[-1]
            pred = model(*data)
            cumulative_loss += loss(pred, label)
    loss = cumulative_loss/len(test_loader)
    logging.info(f"MSE error on the test set of the best model extracted is : {loss}")

    # store the best model
    torch.save(best_model_file, os.path.join(model_dir, "hybridization_probability.pt"))


if __name__ == "__main__":
    main()