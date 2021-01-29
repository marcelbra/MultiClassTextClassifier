from classifier import TextClassifier
import torch
from pytorch_lightning import Trainer, loggers
from pytorch_lightning.callbacks import ModelCheckpoint

def main(hparams):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.device(device)
    model = TextClassifier(hparams)
    tb_logger = loggers.TensorBoardLogger('logs/')

    checkpoint_callback = ModelCheckpoint(monitor='val_loss')
    trainer = Trainer(min_epochs=hparams["min_epochs"],
                      max_epochs=hparams["max_epochs"],
                      logger=tb_logger,
                      callbacks=[checkpoint_callback],
                      default_root_dir="/Models/checkpoints")
    trainer.fit(model)
    trainer.test()  # loads the best model automatically
    torch.save(model.state_dict(), f"Models/model_{hparams['no']}.pt")

hparams = {
    "lstm_hidden_dim": 300,
    "FC_layer_dims": [250, 200, 100, 50],
    "FC_dropouts": [0.1, 0.1, 0.1, 0.35],
    "max_epochs": 50,
    "min_epochs": 1,
    "no": 1,
}

main(hparams)