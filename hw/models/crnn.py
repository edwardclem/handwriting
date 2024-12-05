import lightning as L
import torch
import torch.nn.functional as F
from torch import nn, optim
from torchaudio.models.decoder import cuda_ctc_decoder
from torchmetrics.text import CharErrorRate

from hw.models.cnn_encoder import ResNetEncoder


class CRNN(L.LightningModule):
    def __init__(self, vocab, output_time_steps: int = 96):
        super(CRNN, self).__init__()

        self.vocab = vocab
        # b+w for IAM
        self.cnn = ResNetEncoder(chan_in=1, time_step=output_time_steps)

        # Assume the width and height are reduced after convolutions.
        self.rnn = nn.LSTM(
            self.cnn.output_hsize,
            256,
            num_layers=4,
            bidirectional=True,
            batch_first=True,
        )
        self.fc = nn.Linear(512, len(vocab["forward"]))

        self.loss = nn.CTCLoss(blank=0)

        self.validation_cer = CharErrorRate()

        vocab_list = [vocab["reverse"][i] for i in range(len(vocab["reverse"]))]
        self.decoder = cuda_ctc_decoder(vocab_list)

    def forward(self, x):
        # CNN feature extraction
        x = self.cnn(x)  # Shape: (Batch, width, hidden)

        # RNN for sequence modeling
        # shape: batch, width, hidden
        x, _ = self.rnn(x)

        # Fully connected layer for predictions
        x = self.fc(x)  # (batch, seq_len, n_toks)

        # pred logprobs
        return F.log_softmax(x, dim=-1)  # seq_len, n_toks

    def decode_preds(self, log_probs):
        # input: (batch, seq_len, n_toks)
        # i don't entirely think this is correct. Need to figure out padding.
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
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
