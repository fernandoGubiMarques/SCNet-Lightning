import torch
from scnet_module import SCNetLightning
from byol import BYOL
from wav_module import WavModule
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger
from pathlib import Path
from omegaconf import OmegaConf
from argparse import ArgumentParser
import json


def main():

    parser = ArgumentParser()
    parser.add_argument("--config", "-c", type=str, default="config/default.yaml")
    parser.add_argument("--pretrain", "-p", type=Path)
    parser.add_argument("--freeze", "-f", action='store_true')
    args = parser.parse_args()

    config = OmegaConf.load(args.config)

    torch.set_float32_matmul_precision("high")
    model = SCNetLightning(
        config.model,
        config.optim,
        config.data,
        config.augment,
        config.inference,
        config.train
    )

    pretrained = BYOL.load_from_checkpoint(args.pretrain)
    model.backbone.load_state_dict(pretrained.online_encoder.state_dict())

    if args.freeze:
        for param in model.backbone.parameters():
            param.requires_grad = False

    datamodule = WavModule(config.data, config.loader)

    logger = CSVLogger("./logs", "SCNet_FT")
    log_dir = Path(logger.log_dir)

    logger.log_hyperparams(
        {
            **config,
            "pretrain": str(args.pretrain),
            "freeze": str(args.freeze)
        }
    )

    cbks = [
        ModelCheckpoint(
            logger.log_dir,
            "checkpoints/{step}-" + stem,
            "val_nsdr_" + stem,
            mode="max",
            save_top_k=3,
        )
        for stem in config.data.sources
    ]

    cbks.append(
        ModelCheckpoint(
            dirpath=logger.log_dir,
            filename="checkpoints/{step}-nsdr",
            monitor="val_nsdr",
            mode="max",
            save_top_k=3,
        )
    )

    cbks.append(TQDMProgressBar(100))

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

    test_metrics = [
        {
            "model": str(ckpt_file),
            **(trainer.test(datamodule=datamodule, ckpt_path=ckpt_file)[0]),
        }
        for ckpt_file in log_dir.glob("checkpoints/*.ckpt")
    ]

    with open(log_dir / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f)


if __name__ == "__main__":
    main()
