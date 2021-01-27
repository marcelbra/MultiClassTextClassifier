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

class TextClassifier(pl.LightningModule):

    def __init__(self, dataloader_params=None):
        super().__init__()

        self.dataloader_params = dataloader_params
        self.train, self.test, self.val = split_dataset()
        self.rnn = nn.LSTM(300, 300)
        self.classifier = nn.Sequential(
            nn.Linear(300, 100),
            nn.ReLU(),
            nn.Linear(100, 7)
        )

    def forward(self, x):
        x, _ = self.rnn(x)
        x = x[:,-1]
        pred = self.classifier(x)
        return pred

    def training_step(self, batch, batch_idx):

        y = torch.tensor([1 if x == batch["response"] else 0 for x in range(7)]).reshape(1,7)

        x = batch["documents"]
        y_hat = self.classifier(x)
        loss = nn.CrossEntropyLoss(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        y = torch.tensor([1 if x == batch["response"] else 0 for x in range(7)]).reshape(1,7)
        x = batch["documents"]
        y_hat = self.classifier(x)
        loss = nn.CrossEntropyLoss(y_hat, y)
        self.log('valid_loss', loss, on_step=True)

    def test_step(self, batch, batch_idx):
        y = torch.tensor([1 if x == batch["response"] else 0 for x in range(7)]).reshape(1,7)
        x = batch["documents"]
        y_hat = self.classifier(x)
        loss = nn.CrossEntropyLoss(y_hat, y)
        self.log('test_loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def train_dataloader(self):
        return DataLoader(self.train, **self.dataloader_params)

    def val_dataloader(self):
        return DataLoader(self.val, **self.dataloader_params)

    def test_dataloader(self):
        return DataLoader(self.test, **self.dataloader_params)


dataloader_params = {
    "batch_size": 1,
    "shuffle": True
}
torch.cuda.device("cuda:0")
model = TextClassifier(dataloader_params)

"""
trainer = Trainer()
trainer.fit(model)
"""