"""
Main AVCrossNet model implementation.
Audio-Visual Speech Enhancement using cross-modal attention and SSM-based denoising.
"""
import random
import numpy as np
import torch
from torch import nn
from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model_components import (
    CustomConvNeXtV2,
    VideoFrameEncoder,
    Denoiser,
    CustomCrossAttention,
    UDA
)
from utils.dnn import cal_si_snr


class AVSE(nn.Module):
    """
    Audio-Visual Speech Enhancement model.
    Combines video frame encoding with audio denoising using cross-modal attention.
    """
    def __init__(self):
        super(AVSE, self).__init__()
        self.UDA = UDA
        self.CCA = CustomCrossAttention
        self.VFE = VideoFrameEncoder(CustomConvNeXtV2, 100)
        self.Denoiser = Denoiser(self.CCA, self.UDA)
        
    def load_denoiser_og(self):
        """Load pretrained denoiser weights from HuggingFace."""
        self.Denoiser.from_pretrained("PeaBrane/aTENNuate")

    def forward(self, input):
        """
        Forward pass through the AVSE model.
        
        Args:
            input: Dictionary containing:
                - noisy_audio: Noisy audio tensor
                - video_frames: Video frames tensor
        
        Returns:
            Enhanced audio tensor
        """
        a0 = input["noisy_audio"]
        video_frames = input["video_frames"]
        v0 = video_frames.permute(0, 2, 1, 3, 4)
        v1 = self.VFE(v0)
        a1 = self.Denoiser(a0.unsqueeze(1), v1)
        return a1[:, 0, ...]


class AVSEModule(LightningModule):
    """
    PyTorch Lightning module wrapper for AVSE model.
    Handles training, validation, and optimization.
    """
    def __init__(self, lr=0.001, val_dataset=None):
        super(AVSEModule, self).__init__()
        self.lr = lr
        self.val_dataset = val_dataset
        self.loss = cal_si_snr
        self.model = AVSE()

    def load_denoiser_og(self):
        """Load pretrained denoiser weights."""
        self.model.load_denoiser_og()

    def forward(self, data):
        """Forward pass through the model."""
        est_source = self.model(data)
        return est_source

    def training_step(self, batch_inp, batch_idx):
        """Training step with loss computation and logging."""
        loss = self.cal_loss(batch_inp)
        self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch_inp, batch_idx):
        """Validation step with loss computation and logging."""
        loss = self.cal_loss(batch_inp)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def enhance(self, data):
        """
        Enhance audio from input data dictionary.
        
        Args:
            data: Dictionary containing noisy_audio and video_frames
        
        Returns:
            Enhanced audio as numpy array (normalized)
        """
        inputs = dict(
            noisy_audio=torch.tensor(data["noisy_audio"][None, ...]).to(self.device),
            video_frames=torch.tensor(data["video_frames"][None, ...]).to(self.device)
        )
        estimated_audio = self(inputs).cpu().numpy()
        estimated_audio /= np.max(np.abs(estimated_audio))
        return estimated_audio

    def on_training_epoch_end(self, outputs):
        """Log audio samples to TensorBoard at end of training epoch."""
        if self.val_dataset is not None:
            with torch.no_grad():
                tensorboard = self.logger.experiment
                for index in range(5):
                    rand_int = random.randint(0, len(self.val_dataset))
                    data = self.val_dataset[rand_int]
                    estimated_audio = self.enhance(data)
                    tensorboard.add_audio(
                        "{}/{}_clean".format(self.current_epoch, index),
                        data["clean"][np.newaxis, ...],
                        sample_rate=16000
                    )
                    tensorboard.add_audio(
                        "{}/{}_noisy".format(self.current_epoch, index),
                        data["noisy_audio"][np.newaxis, ...],
                        sample_rate=16000
                    )
                    tensorboard.add_audio(
                        "{}/{}_enhanced".format(self.current_epoch, index),
                        estimated_audio.reshape(-1)[np.newaxis, ...],
                        sample_rate=16000
                    )

    def cal_loss(self, batch_inp):
        """Calculate SI-SNR loss between predicted and clean audio."""
        mask = batch_inp["clean"].T
        pred_mask = self(batch_inp).T.reshape(mask.shape)
        loss = self.loss(pred_mask.unsqueeze(2), mask.unsqueeze(2))
        loss[loss < -30] = -30
        return torch.mean(loss)

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        optimizer = torch.optim.RMSprop(self.parameters(), lr=self.lr)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optimizer, factor=0.8, patience=5),
                "monitor": "val_loss_epoch",
            },
        }
