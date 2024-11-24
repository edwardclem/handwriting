import lightning as L
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchmetrics.text import CharErrorRate
from hw.models.beam import beam_decode
from tqdm import trange


class CRNN(L.LightningModule):
    def __init__(self, vocab, img_height=128):
        super(CRNN, self).__init__()

        self.vocab = vocab

        # super simple. Replace with resnet eventually.
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),  # Example Conv Layer
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Assume the width and height are reduced after convolutions.
        self.rnn = nn.LSTM(
            128 * (img_height // 4), 256, num_layers=2, bidirectional=True
        )
        self.fc = nn.Linear(512, len(vocab["forward"]))

        self.loss = nn.CTCLoss(blank=0)

        self.validation_cer = CharErrorRate()

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
        # seq_len, batch, num_classes
        lprobs = log_probs.cpu().numpy()
        decodes = [
            beam_decode(lprobs[:, i, :])
            for i in trange(lprobs.shape[1], desc="Decoding batch...")
        ]
        # convert all decodes to strings
        string_decodes = [
            "".join(self.inverse_vocab[char] for char in dec) for dec in decodes
        ]
        return string_decodes

    @torch.no_grad
    def decode(self, x):
        return self.decode_preds(self(x))

    def training_step(self, batch, batch_idx):

        images, target, target_lengths, _ = batch

        # Predictions and targets
        pred_log_probs = self(images)  # (Seq_Len, Batch, Num_Classes)
        # All sequences are the same length after padding
        input_lengths = torch.tensor([pred_log_probs.size(0)])

        loss = self.loss(pred_log_probs, target, input_lengths, target_lengths)

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):

        images, target, target_lengths, raw_seqs = batch

        # Predictions and targets
        pred_log_probs = self(images)  # (Seq_Len, Batch, Num_Classes)
        # All sequences are the same length after padding
        input_lengths = torch.tensor([pred_log_probs.size(0)] * pred_log_probs.size(1))
        loss = self.loss(pred_log_probs, target, input_lengths, target_lengths)
        self.log("val_loss", loss, batch_size=pred_log_probs.size(1))
        decode_strings = self.decode_preds(pred_log_probs)
        self.validation_cer(decode_strings, raw_seqs)
        self.log(
            "validation_cer",
            self.validation_cer,
            on_step=True,
            on_epoch=True,
            batch_size=pred_log_probs.size(1),
        )

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
