from classifier import TextClassifier
import torch
from pytorch_lightning import Trainer, loggers
from pytorch_lightning.callbacks import ModelCheckpoint

def test(hparams):
    model = TextClassifier(hparams)
    model.load_state_dict(torch.load("/home/marcelbraasch/PycharmProjects/MultiClassTextClassifier/Models/model_2.pt"))
    model.eval()
    print(*model.get_confusion(), sep="\n")

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
    model.eval()
    with open("log.txt", mode="a") as f:
        for line in model.get_confusion():
            f.write(str(line))


hparams = {
    "lstm_hidden_dim": 300,
    "FC_layer_dims": [32],
    "FC_dropouts": [0.0],
    "max_epochs": 8,
    "min_epochs": 1,
    "no": 5,
    "bidirectional": False,
    "lstm_num_layers": 1,
    "optimizer": "Adam",
    "lr": 1e-3
}

main(hparams)

"""
hparams = {
    "lstm_hidden_dim": 300,
    "FC_layer_dims": [32],
    "FC_dropouts": [0.0],
    "max_epochs": 8,
    "min_epochs": 1,
    "no": 5,
    "bidirectional": False,
    "lstm_num_layers": 1,
    "optimizer": "Adam",
    "lr": 4e-4
}

DATALOADER:0 TEST RESULTS
{'test_acc': 0.6803559064865112, 'test_loss': 0.7835955023765564}
--------------------------------------------------------------------------------
[392, 12, 1, 42, 36, 39, 10]
[30, 700, 12, 5, 5, 5, 24]
[9, 14, 121, 0, 1, 1, 455]
[135, 5, 3, 43, 19, 21, 8]
[140, 2, 5, 13, 55, 46, 4]
[119, 6, 3, 18, 40, 46, 5]
[8, 15, 83, 0, 1, 1, 1625]


"""

import optuna

def objective(trial):

    # Layer amount and dims
    num_layers = trial.suggest_int("num_layers", 1, 3)
    FC_layer_dims = [trial.suggest_int(f"{i}th_layer", 7, 300) for i in range(num_layers)]
    FC_dropouts = [trial.suggest_float(f"{i}th_droput", 0.0, 0.5) for i in range(num_layers)]
    bidirectional = trial.suggest_categorical("bidirectional", [False, True])
    lstm_num_layers = trial.suggest_int("lstm_num_layers", 1, 2)
    lr = trial.suggest_loguniform("lr", 1e-4, 1e-3)
    hparams = {
        "lstm_hidden_dim": 300,
        "FC_layer_dims": FC_layer_dims,
        "FC_dropouts": FC_dropouts,
        "max_epochs": 8,
        "min_epochs": 1,
        "no": trial.number,
        "bidirectional": bidirectional,
        "lstm_num_layers": lstm_num_layers,
        "optimizer": "Adam",
        "lr": lr
    }

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)