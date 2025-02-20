import torch
from byol import BYOL
from wav_mixture import WavMixtureModule
from lightning import Trainer
from lightning.pytorch.callbacks import TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger
from pathlib import Path
import json


def main():

    model = BYOL(lr=0.0004)
    datamodule = WavMixtureModule("/home/users/fgm/workspace/data/musdb18hq")
    logger = CSVLogger("./logs", "BYOL")
    cbks = [TQDMProgressBar(100)]

    trainer = Trainer(
        accelerator="cuda",
        # devices=torch.cuda.device_count(),  # Use all available GPUs
        # strategy="ddp",  # Enable Distributed Data Parallel (DDP)
        # num_nodes=2,  # Adjust based on available nodes
        # sync_batchnorm=True,  # Synchronize batch norms across GPUs
        precision="16-mixed",
        max_epochs=130,
        logger=logger,
        callbacks=cbks,
        log_every_n_steps=400,
        val_check_interval=0.5,
    )

    trainer.fit(model, datamodule=datamodule)

if __name__ == "__main__":
    main()
