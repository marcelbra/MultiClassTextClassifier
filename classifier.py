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

class TextClassifier(pl.LightningModule):

    def __init__(self, dataloader_params=None):
        super().__init__()

        self.dataloader_params = dataloader_params
        self.train_data, self.test_data, self.val_data = split_dataset()
        self.writer = SummaryWriter()
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
        y = batch["response"]
        x = batch["document"]
        y_hat = self.forward(x)
        loss = nn.CrossEntropyLoss()
        self.log_loss("training", loss, batch_idx)
        return {"loss":loss(y_hat, y)}

    def validation_step(self, batch, batch_idx):
        y = batch["response"]
        x = batch["document"]
        y_hat = self.forward(x)
        loss = nn.CrossEntropyLoss()
        self.log_loss("validation", loss, batch_idx)
        return {"loss": loss(y_hat, y)}

    def test_step(self, batch, batch_idx):
        y = batch["response"]
        x = batch["document"]
        y_hat = self.forward(x)
        loss = nn.CrossEntropyLoss()
        self.log_loss("test", loss, batch_idx)
        return {"loss": loss(y_hat, y)}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=1, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=1)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=1)

    def log_loss(self, mode, loss, i):
        # for every epoch log 10 times
        if mode == "training":
            frequency = int(len(self.train_data) / 10) - 1
        elif mode == "test":
            frequency = int(len(self.test_data) / 10) - 1
        elif mode == "validation":
            frequency = int(len(self.val_data) / 10) - 1
        if i % frequency + 1 == 0:
            self.writer.add_scalar(f"{mode}_loss", loss)


dataloader_params = {
    "batch_size": 1,
    "shuffle": True
}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.device(device)
model = TextClassifier(dataloader_params)
trainer = Trainer(min_epochs=1, max_epochs=3)
trainer.fit(model)
model.writer.close()