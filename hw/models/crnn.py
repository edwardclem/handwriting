from typing import List, Tuple

import lightning as L
import torch
import torch.nn.functional as F
from torch import nn, optim
from torchaudio.models.decoder import cuda_ctc_decoder
from torchmetrics.text import CharErrorRate

from hw.models.cnn_encoder import ResNetEncoder


class CRNN(L.LightningModule):
    def __init__(
        self,
        vocab: List[str],  # list of strings, index corresponds to vocab entry
        lstm_hidden: int = 256,
        num_lstm_layers: int = 2,
        intermediate_stride: Tuple[int, int] = (1, 2),
        intermediate_timesteps: int = 96,  # length of sequence the image features are converted to
        blank_idx: int = 0,
    ):
        super(CRNN, self).__init__()
        self.save_hyperparameters()

        self.vocab = vocab
        self.blank_idx = blank_idx
        self.intermediate_timesteps = intermediate_timesteps
        self.intermediate_stride = intermediate_stride
        self.lstm_hidden = lstm_hidden
        self.num_lstm_layers = num_lstm_layers

        # b+w for IAM
        self.cnn = ResNetEncoder(
            chan_in=1, intermediate_stride=self.intermediate_stride
        )

        # flattens to (batch, hidden, 1, timesteps)
        self.flatten_layer = nn.AdaptiveAvgPool2d(
            output_size=(1, self.intermediate_timesteps)
        )

        # Assume the width and height are reduced after convolutions.
        self.rnn = nn.LSTM(
            self.cnn.output_hsize,
            self.lstm_hidden,
            num_layers=self.num_lstm_layers,
            bidirectional=True,
            batch_first=True,
        )
        self.fc = nn.Linear(self.lstm_hidden * 2, len(vocab))

        self.loss = nn.CTCLoss(blank=self.blank_idx)
        self.validation_cer = CharErrorRate()
        self.decoder = cuda_ctc_decoder(vocab)

    def forward(self, x):
        # CNN feature extraction
        x = self.cnn(x)  # Shape: (Batch, hidden, height, width)

        # flatten to (batch, hidden, timesteps)
        # amnd reshape to (batch, timesteps, hidden) for RNN input
        x = self.flatten_layer(x).squeeze(dim=2).transpose(1, 2)

        # RNN for sequence modeling
        # shape: batch, width, hidden
        x, _ = self.rnn(x)

        # Fully connected layer for predictions
        x = self.fc(x)  # (batch, seq_len, n_toks)

        # pred logprobs
        return F.log_softmax(x, dim=-1)  # batch, seq_len, n_toks

    def decode_preds(self, log_probs):
        # input: (batch, seq_len, n_toks)
        input_lengths = torch.tensor([log_probs.size(1)] * log_probs.size(0)).to(
            device=log_probs.device, dtype=torch.int32
        )

        # list of CUCTChypothesis
        decodes = self.decoder(log_probs, input_lengths)

        string_decodes = ["".join(dec[0].words) for dec in decodes]
        return string_decodes

    @torch.no_grad
    def decode(self, x):
        # e2e decoder function.
        return self.decode_preds(self(x))

    def training_step(self, batch, batch_idx):

        images, target, target_lengths, _ = batch

        # Predictions and targets
        pred_log_probs = self(images)  # (batch, seq_len, n_toks)
        # All sequences are the same length after padding
        input_lengths = torch.tensor([pred_log_probs.size(1)] * pred_log_probs.size(0))

        # need to permute to (seq_len, batch, n_toks)
        loss = self.loss(
            pred_log_probs.permute(1, 0, 2), target, input_lengths, target_lengths
        )

        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=pred_log_probs.size(1),
        )

        return loss

    def validation_step(self, batch, batch_idx):

        images, target, target_lengths, raw_seqs = batch

        # Predictions and targets
        pred_log_probs = self(images)  # (batch, seq_len, n_toks)
        # All sequences are the same length after padding
        input_lengths = torch.tensor([pred_log_probs.size(1)] * pred_log_probs.size(0))

        # need to permute to (seq_len, batch, n_toks)

        loss = self.loss(
            pred_log_probs.permute(1, 0, 2), target, input_lengths, target_lengths
        )

        self.log("val_loss", loss, batch_size=pred_log_probs.size(1), on_epoch=True)
        decode_strings = self.decode_preds(pred_log_probs)
        self.validation_cer(decode_strings, raw_seqs)
        self.log(
            "validation_cer",
            self.validation_cer,
            on_epoch=True,
            batch_size=pred_log_probs.size(1),
        )

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=0.0005, weight_decay=0.0001)

        scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0005, max_lr=0.01)

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
