import lightning as L
import torch
import torch.nn.functional as F
from torch import nn, optim
from torchaudio.models.decoder import cuda_ctc_decoder
from torchmetrics.text import CharErrorRate


class CRNN(L.LightningModule):
    def __init__(self, vocab, img_height=128):
        super(CRNN, self).__init__()

        self.vocab = vocab

        # super simple. Replace with resnet eventually.
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),  # Example Conv Layer
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=4),
        )

        # need a better system for matching these.

        # Assume the width and height are reduced after convolutions.
        self.rnn = nn.LSTM(
            32 * (img_height // 4), 256, num_layers=2, bidirectional=True
        )
        self.fc = nn.Linear(512, len(vocab["forward"]))

        self.loss = nn.CTCLoss(blank=0)

        self.validation_cer = CharErrorRate()

        vocab_list = [vocab["reverse"][i] for i in range(len(vocab["reverse"]))]
        self.decoder = cuda_ctc_decoder(vocab_list)

    def forward(self, x):
        # CNN feature extraction
        x = self.cnn(x)  # Shape: (Batch, Channels, Height, Width)
        b, c, h, w = x.size()
        x = x.permute(3, 0, 2, 1).reshape(w, b, -1)  # (Seq_Len, Batch, Features)

        # RNN for sequence modeling
        x, _ = self.rnn(x)

        # Fully connected layer for predictions
        x = self.fc(x)  # (Seq_Len, Batch, Num_Classes)

        # pred logprobs
        return F.log_softmax(x, dim=-1)

    def decode_preds(self, log_probs):
        # input: seq_len, batch, num_classes
        # convert to: batch, seq_len, n_classes
        log_probs_reordered = log_probs.permute(1, 0, 2).contiguous()
        # i don't entirely think this is correct. Need to figure out padding.
        input_lengths = torch.tensor([log_probs.size(0)] * log_probs.size(1)).to(
            device=log_probs_reordered.device, dtype=torch.int32
        )

        # list of CUCTChypothesis
        decodes = self.decoder(log_probs_reordered, input_lengths)

        string_decodes = ["".join(dec[0].words) for dec in decodes]
        return string_decodes

    @torch.no_grad
    def decode(self, x):
        # e2e decoder function.
        return self.decode_preds(self(x))

    def training_step(self, batch, batch_idx):

        images, target, target_lengths, _ = batch

        # Predictions and targets
        pred_log_probs = self(images)  # (Seq_Len, Batch, Num_Classes)
        # All sequences are the same length after padding
        input_lengths = torch.tensor([pred_log_probs.size(0)] * pred_log_probs.size(1))

        loss = self.loss(pred_log_probs, target, input_lengths, target_lengths)

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
        pred_log_probs = self(images)  # (Seq_Len, Batch, Num_Classes)
        # All sequences are the same length after padding
        input_lengths = torch.tensor([pred_log_probs.size(0)] * pred_log_probs.size(1))
        loss = self.loss(pred_log_probs, target, input_lengths, target_lengths)
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
        optimizer = optim.Adam(self.parameters(), lr=1e-2)
        return optimizer
