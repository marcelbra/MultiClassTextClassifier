import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl
from dataset import Dataset
from torch.utils.data import DataLoader
from utils import split_dataset
from pytorch_lightning import Trainer
from torch.utils.tensorboard import SummaryWriter
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint

class TextClassifier(pl.LightningModule):

    def __init__(self, hparams=None):
        super().__init__()

        # Metrics
        self.train_acc = pl.metrics.Accuracy()
        self.val_acc = pl.metrics.Accuracy(compute_on_step=False)
        self.test_acc = pl.metrics.Accuracy(compute_on_step=False)

        # Hyperparameters
        self.hparams = hparams

        # Data
        self.train_data, self.test_data, self.val_data = split_dataset()

        # Model initialization
        self.word_vec_size = 300
        self.amount_classes = 7
        self.rnn = nn.LSTM(input_size=self.word_vec_size,
                           hidden_size=hparams["lstm_hidden_dim"])

        # First FC layer
        modules = [
            nn.Linear(self.word_vec_size, hparams["FC_layer_dims"][0]),
            nn.ReLU(),
            nn.Dropout(hparams["FC_dropouts"][0])
        ]

        # Middle FC layers
        for i, (dim, d_rate) in enumerate(zip(hparams["FC_layer_dims"],
                                              hparams["FC_dropouts"])):
            if i == len(hparams["FC_layer_dims"]) - 1: continue # we reached the end
            modules.append(nn.Linear(dim, hparams["FC_layer_dims"][i+1]))
            modules.append(nn.ReLU())
            modules.append(nn.Dropout(d_rate))

        # Last FC layer
        modules.append(nn.Linear(hparams["FC_layer_dims"][-1], self.amount_classes))
        modules.append(nn.ReLU())
        modules.append(nn.Dropout(hparams["FC_dropouts"][-1]))

        self.classifier = nn.Sequential(
            *modules
        )

    def forward(self, x):
        x, _ = self.rnn(x)
        x = x[:,-1]
        pred = self.classifier(x)
        return pred

    def training_step(self, batch, batch_idx):
        y = batch["response"]
        x = batch["document"]
        y_hat = self.forward(x)
        loss = nn.CrossEntropyLoss()
        loss = loss(y_hat, y)
        self.log("training_loss", loss)
        self.log("training_acc_step", self.train_acc(y_hat, y))
        return {"loss": loss}

    def training_epoch_end(self, outs):
        self.log('train_acc_epoch', self.train_acc.compute())

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=1, shuffle=True)

    def validation_step(self, batch, batch_idx):
        y = batch["response"]
        x = batch["document"]
        y_hat = self.forward(x)
        loss = nn.CrossEntropyLoss()
        loss = loss(y_hat, y)
        self.log("val_loss", loss)
        self.val_acc(y_hat, y)
        return {"loss": loss}

    def validation_epoch_end(self, outs):
        self.log('validation_acc_epoch', self.val_acc.compute())

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=1)

    def test_step(self, batch, batch_idx):
        y = batch["response"]
        x = batch["document"]
        y_hat = self.forward(x)
        loss = nn.CrossEntropyLoss()
        loss = loss(y_hat, y)
        self.log("test_loss", loss)
        self.test_acc(y_hat, y)
        return {"loss": loss}

    def test_step_end(self, *args, **kwargs):
        self.log("test_acc", self.test_acc.compute())

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=1)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer