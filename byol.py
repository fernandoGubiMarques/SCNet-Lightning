import torch
import torch.nn.functional as F
from torch import nn
import lightning as L
from half_scnet import SCNet_Backbone
from byol_heads import ProjectionHead, PredictionHead
import einops
from wav_mixture import *


class BYOL(L.LightningModule):
    def __init__(self, ema_decay=0.99, lr=1e-3, weight_decay=1e-6):
        """
        Args:
            ema_decay: Exponential moving average decay for target network
            lr: Learning rate
            weight_decay: Weight decay for optimizer
        """
        super().__init__()
        # self.save_hyperparameters()

        # Online network
        self.online_encoder = SCNet_Backbone(["blank"])
        self.online_projector = ProjectionHead()
        self.predictor = PredictionHead()

        self.augment = nn.Sequential(
            RandomPitchShift(),
            RandomBandPass(),
            RandomChorus(),
            RandomPhaser(),
            RandomAddNoise(),
        )

        # Target network (no predictor)
        self.target_encoder = SCNet_Backbone(["blank"])
        self.target_projector = ProjectionHead()

        for param in self.target_encoder.parameters():
            param.requires_grad = False

        for param in self.target_projector.parameters():
            param.requires_grad = False

        self.target_encoder.eval()
        self.target_projector.eval()

        # Initialize target network with online network parameters
        self._initialize_target_network()
        self.ema_decay = ema_decay

        # Optimizer parameters
        self.lr = lr
        self.weight_decay = weight_decay

    def _initialize_target_network(self):
        """Initialize target network with the same parameters as the online network."""
        for param_o, param_t in zip(
            self.online_encoder.parameters(), self.target_encoder.parameters()
        ):
            param_t.data.copy_(param_o.data)
            param_t.requires_grad = False  # Target network is not trained directly
        for param_o, param_t in zip(
            self.online_projector.parameters(), self.target_projector.parameters()
        ):
            param_t.data.copy_(param_o.data)
            param_t.requires_grad = False  # Target network is not trained directly

    @torch.no_grad()
    def _update_target_network(self):
        """Updates the target network parameters using exponential moving average (EMA)."""
        for param_o, param_t in zip(
            self.online_encoder.parameters(), self.target_encoder.parameters()
        ):
            param_t.data = (
                self.ema_decay * param_t.data + (1 - self.ema_decay) * param_o.data
            )
        for param_o, param_t in zip(
            self.online_projector.parameters(), self.target_projector.parameters()
        ):
            param_t.data = (
                self.ema_decay * param_t.data + (1 - self.ema_decay) * param_o.data
            )

    # def forward(self, x):
    #     """Forward pass through the online network."""
    #     representation = self.online_encoder(x)
    #     projection = self.online_projector(representation)
    #     prediction = self.predictor(projection)
    #     return prediction

    def training_step(self, batch, batch_idx):
        """Computes the BYOL loss and updates the target network."""
        batch = batch[0]
        x1, x2 = self.augment(batch), self.augment(
            batch
        )  # Expecting two augmentations of the same image

        # Online network
        z1 = self.online_encoder(x1)
        z2 = self.online_encoder(x2)

        z1 = [einops.rearrange(z1[0], "B ... L -> B L (...)")] + [
            einops.rearrange(i, "B ... L -> B L (...)") for i in z1[1]
        ]

        z2 = [einops.rearrange(z2[0], "B ... L -> B L (...)")] + [
            einops.rearrange(i, "B ... L -> B L (...)") for i in z2[1]
        ]

        z1 = self.online_projector(*z1)
        z2 = self.online_projector(*z2)

        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        # Target network (detached)
        with torch.no_grad():
            t1 = self.target_encoder(x1)
            t2 = self.target_encoder(x2)

            t1 = [einops.rearrange(t1[0], "B ... L -> B L (...)")] + [
                einops.rearrange(i, "B ... L -> B L (...)") for i in t1[1]
            ]

            t2 = [einops.rearrange(t2[0], "B ... L -> B L (...)")] + [
                einops.rearrange(i, "B ... L -> B L (...)") for i in t2[1]
            ]

            t1 = self.target_projector(*t1)
            t2 = self.target_projector(*t2)

        # Compute BYOL loss (negative cosine similarity)
        loss = self.loss_fn(p1, t2) + self.loss_fn(p2, t1)

        # Update target network with EMA
        self._update_target_network()

        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def loss_fn(self, p, z):
        """Computes the BYOL loss (negative cosine similarity)."""
        p = F.normalize(p, dim=-1)
        z = F.normalize(z, dim=-1)
        return 2 - 2 * (p * z).sum(dim=-1).mean()

    def configure_optimizers(self):
        """Defines the optimizer (AdamW) and learning rate scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=200
        )  # Adjust as needed
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
        # return optimizer
